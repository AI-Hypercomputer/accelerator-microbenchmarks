"""Fused Pallas TPU kernel for blockwise and row-wise quantized GEMM."""

import math
from accelerator_microbenchmarks.core import dtypes
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Default per-kernel scoped VMEM budget (64 MiB), matching the minimum
# --xla_tpu_scoped_vmem_limit_kib=65536 across op_flags.yaml files.
_DEFAULT_SCOPED_VMEM_BYTES = 64 << 20


def _get_mxu_column_size() -> int:
  """Returns the native MXU column width for the active device."""
  return pltpu.get_tpu_info().mxu_column_size if pltpu.is_tpu_device() else 128


def _largest_aligned_divisor_le(dim: int, cap: int, align: int) -> int:
  """Returns the largest divisor of `dim` that is `<= cap` and aligned."""
  target = min(dim, cap)
  if dim % target == 0 and (target == dim or target % align == 0):
    return target
  best = dim
  for d in range(1, math.isqrt(dim) + 1):
    if dim % d == 0:
      for cand in (d, dim // d):
        if cand <= cap and (cand == dim or cand % align == 0):
          if best > cap or cand > best:
            best = cand
  return best


def default_quantized_matmul_blocks(
    m: int,
    k: int,
    n: int,
    *,
    sc_size: int,
    quantize_rhs: bool = True,
) -> tuple[int, int, int, int]:
  """Returns `(block_m, block_k, block_n, n_lane_multiplier)` for `[M, K] @ [K, N]`.

  Selects VMEM-fitting tile sizes aligned to TPU lanes:
    * `block_m = min(m, 512)`, `block_n = min(n, 4096)`.
    * For sub-channel scaling (`sc_size <= 1024` when `quantize_rhs=True`, or
      `sc_size <= 2048` when `quantize_rhs=False`), `block_k` is the largest
      multiple of `sc_size` dividing `k` up to `1024` (on-the-fly) or `2048`
      (weight-offline).
    * For full-row scaling (`sc_size == k` with `k` exceeding the single-tile K
      cap), `block_k` tiles `k` at up to `1024` (on-the-fly) or `2048`
      (weight-offline) while row/column scale factors are broadcast across K
      tiles.

  Args:
    m: Rows of LHS `A[M, K]` and output `C[M, N]`.
    k: Contracting dimension `K`.
    n: Columns of RHS `B[K, N]` and output `C[M, N]`.
    sc_size: Sub-channel block size along `K` (`k % sc_size == 0`; `sc_size ==
      k` selects full-row/column scaling).
    quantize_rhs: Whether `B` is quantized on the fly inside the kernel.

  Returns:
    A 4-tuple `(block_m, block_k, block_n, n_lane_multiplier)`.
  """
  if m <= 0 or k <= 0 or n <= 0:
    raise ValueError(f"GEMM dimensions ({m}, {k}, {n}) must be positive.")
  if sc_size <= 0 or k % sc_size != 0:
    raise ValueError(
        f"Contracting dimension k ({k}) must be divisible by positive sc_size"
        f" ({sc_size})."
    )
  _, lanes = pltpu.Tiling.COMPACT.shape
  cap_m = 512
  cap_n = 4096
  max_tile_k = 1024 if quantize_rhs else 2048

  bm = _largest_aligned_divisor_le(m, cap_m, lanes)
  bn = _largest_aligned_divisor_le(n, cap_n, lanes)

  if sc_size > max_tile_k:
    if sc_size != k:
      raise ValueError(
          f"sc_size ({sc_size}) exceeds max VMEM K tile ({max_tile_k}) and is"
          f" not equal to full k ({k})."
      )
    bk = _largest_aligned_divisor_le(k, max_tile_k, lanes)
  else:
    cap_k = max(
        sc_size,
        min(max_tile_k, sc_size * (4 if quantize_rhs else 8)),
    )
    k_groups = k // sc_size
    max_groups = max(1, cap_k // sc_size)
    best_groups = 1
    for g in range(1, math.isqrt(k_groups) + 1):
      if k_groups % g == 0:
        for cand in (g, k_groups // g):
          if cand <= max_groups and cand > best_groups:
            best_groups = cand
    bk = best_groups * sc_size

  if not quantize_rhs and sc_size == 512 and (m, n, k) == (4096, 16384, 1024):
    n_lane_multiplier = 4
  else:
    n_lane_multiplier = 2
  return bm, bk, bn, n_lane_multiplier


def _compute_scales_and_inv(
    data_f32: jax.Array,
    *,
    axis: int,
    q_max: float,
    scaling: str,
) -> tuple[jax.Array, jax.Array]:
  """Returns `(scale_f32, inv_scale_f32)` along `axis` in float32."""
  if scaling == "static":
    scale_shape = list(data_f32.shape)
    scale_shape[axis] = 1
    # Use a non-unit power-of-2 clip bound (`0.5 * q_max` -> `scale = 0.5`,
    # `inv_scale = 2.0`) so XLA cannot constant-fold scaling into a no-op.
    static_absmax = 0.5 * q_max
    scale_f32 = jnp.full(scale_shape, static_absmax / q_max, dtype=jnp.float32)
    inv_scale_f32 = jnp.reciprocal(scale_f32)
  elif scaling == "dynamic":
    abs_max = jnp.max(jnp.abs(data_f32), axis=axis, keepdims=True)
    clamped_amax = jnp.maximum(abs_max, 1e-7)
    scale_f32 = clamped_amax / q_max
    inv_scale_f32 = q_max / clamped_amax
  else:
    raise ValueError(
        f"Unsupported scaling mode '{scaling}'; expected 'dynamic' or 'static'."
    )
  return scale_f32, inv_scale_f32


def _scale_and_cast(
    data_f32: jax.Array,
    inv_scale_f32: jax.Array,
    *,
    quant_dtype: jnp.dtype,
    q_max: float,
) -> jax.Array:
  """Multiplies `data_f32` by `inv_scale_f32`, clips, and casts to `quant_dtype`."""
  scaled = data_f32 * inv_scale_f32
  if jnp.issubdtype(quant_dtype, jnp.integer):
    return jnp.clip(jnp.round(scaled), -q_max, q_max).astype(quant_dtype)
  return jnp.clip(scaled, -q_max, q_max).astype(quant_dtype)


def _quantize_subchannel_tile(
    data_f32: jax.Array,
    *,
    axis: int,
    quant_dtype: jnp.dtype,
    scaling: str,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes one sub-channel tile inside the Pallas kernel."""
  q_max = float(dtypes.get_dtype_max(quant_dtype))
  scale_f32, inv_scale_f32 = _compute_scales_and_inv(
      data_f32, axis=axis, q_max=q_max, scaling=scaling
  )
  data_q = _scale_and_cast(
      data_f32, inv_scale_f32, quant_dtype=quant_dtype, q_max=q_max
  )
  return data_q, scale_f32


def quantize_subchannel(
    x: jax.Array,
    *,
    contract_axis: int,
    sc_size: int,
    quant_dtype: jax.typing.DTypeLike,
    scaling: str = "dynamic",
) -> tuple[jax.Array, jax.Array]:
  """Quantizes 2D operand `x` in sub-channels of length `sc_size`.

  Used both to pre-quantize offline weights (`contract_axis=0` on `B[K, N]`,
  producing `w_scales` of shape `[K // sc_size, 1, N]` for `quantized_matmul`)
  and as the pure-JAX reference for unit tests.

  Args:
    x: 2D array (`[M, K]` with `contract_axis=1` or `[K, N]` with
      `contract_axis=0`).
    contract_axis: Contracting dimension (`1` for LHS `A`, `0` for RHS `B`).
    sc_size: Sub-channel block size dividing `x.shape[contract_axis]`.
    quant_dtype: Target quantized dtype (e.g. `float8_e4m3fn`, `int8`, `int4`).
    scaling: `'dynamic'` or `'static'`.

  Returns:
    `(x_q, scales)` where `x_q` has shape `x.shape` and dtype `quant_dtype`, and
    `scales` has shape `[M, K // sc_size, 1]` (when `contract_axis=1`) or
    `[K // sc_size, 1, N]` (when `contract_axis=0`) in `float32`.
  """
  if x.ndim != 2:
    raise ValueError(f"Expected 2D array, got shape {x.shape}.")
  if contract_axis not in (0, 1):
    raise ValueError(f"contract_axis must be 0 or 1, got {contract_axis}.")
  dim0, dim1 = x.shape
  k_len = dim1 if contract_axis == 1 else dim0
  if sc_size <= 0 or k_len % sc_size != 0:
    raise ValueError(
        f"Contracting dimension ({k_len}) must be divisible by positive"
        f" sc_size ({sc_size})."
    )
  q_dtype = jnp.dtype(quant_dtype)
  num_blocks = k_len // sc_size
  if contract_axis == 1:
    x_blocks = x.reshape(dim0, num_blocks, sc_size).astype(jnp.float32)
    q_blocks, sf = _quantize_subchannel_tile(
        x_blocks, axis=2, quant_dtype=q_dtype, scaling=scaling
    )
    return q_blocks.reshape(dim0, dim1), sf
  else:
    x_blocks = x.reshape(num_blocks, sc_size, dim1).astype(jnp.float32)
    q_blocks, sf = _quantize_subchannel_tile(
        x_blocks, axis=1, quant_dtype=q_dtype, scaling=scaling
    )
    return q_blocks.reshape(dim0, dim1), sf


def quantized_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    w_scales: jax.Array | None = None,
    *,
    sc_size: int = 512,
    quant_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    rhs_quant_dtype: jax.typing.DTypeLike | None = None,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    acc_dtype: jax.typing.DTypeLike = jnp.float32,
    quantize_rhs: bool = True,
    dequantize: bool = True,
    scaling: str = "dynamic",
    rhs_transposed: bool = False,
    block_m: int | None = None,
    block_k: int | None = None,
    block_n: int | None = None,
    n_lane_multiplier: int | None = None,
    scoped_vmem_bytes: int = _DEFAULT_SCOPED_VMEM_BYTES,
    interpret: bool = False,
) -> jax.Array:
  """Fused Pallas kernel for blockwise (`B128`/`B512`) and row-wise (`K`) quantized GEMM.

  Supports two execution modes depending on `sc_size` relative to `block_k`:
    1. Sub-channel / single-K-tile mode (`sc_size <= block_k`, `block_k %
       sc_size == 0`): Quantizes `lhs` (and `rhs` when `quantize_rhs=True`) in
       VMEM sub-channels of length `sc_size` along `K`, runs `dot_general` on
       the MXU, optionally rescales each sub-channel partial product by
       `lhs_scale * rhs_scale` (`dequantize=True`), and accumulates across `K`
       tiles in VMEM scratch.
    2. Multi-K-tile row-wise mode (`sc_size == K > block_k`): Computes row/col
       scales over full `K`, then streams `[block_m, block_k]` and
       `[block_k, block_n]` tiles through VMEM—fusing `scale + cast` into the
       VMEM matmul tile loop without triggering XLA's downcast-convert fusion
       slowdown—and applies `(lhs_scale * rhs_scale)` on the final `K` step.

  Args:
    lhs: Left-hand side operand of shape `[M, K]` (unquantized).
    rhs: Right-hand side operand of shape `[K, N]` (or `[N, K]` if
      `rhs_transposed=True`), unquantized when `quantize_rhs=True` or already in
      `rhs_quant_dtype` when `quantize_rhs=False`.
    w_scales: Pre-computed RHS dequantization scales of shape `[K // sc_size, 1,
      N]` (or `[1, N]`) when `quantize_rhs=False` and `dequantize=True`; `None`
      otherwise.
    sc_size: Sub-channel size along `K` (e.g. `128`, `512`, or `K` for
      row-wise). Must divide `K`.
    quant_dtype: Target quantized LHS dtype fed to the MXU (`float8_e4m3fn`,
      `float8_e5m2`, `int8`).
    rhs_quant_dtype: Optional quantized RHS dtype (defaults to `quant_dtype`;
      supports `int4` with FP8 `quant_dtype` for `W4A8`).
    out_dtype: Output array dtype written to HBM.
    acc_dtype: Accumulator scratchpad and scale multiply dtype (`float32` or
      `bfloat16`).
    quantize_rhs: If `True`, quantizes `rhs` on the fly inside the kernel; if
      `False`, `rhs` is already quantized offline in HBM.
    dequantize: If `True`, multiplies by LHS and RHS dequantization scales.
    scaling: `'dynamic'` (amax computed per sub-channel/row) or `'static'`
      (fixed clip bound `0.5 * Q_MAX`, scale `0.5`).
    rhs_transposed: Whether `rhs` has layout `[N, K]` instead of `[K, N]`.
    block_m: Optional tile size along `M` (dividing `M`).
    block_k: Optional tile size along `K` (dividing `K`).
    block_n: Optional tile size along `N` (dividing `N`).
    n_lane_multiplier: Optional multiplier of the native MXU column width for
      inner N strip-mining.
    scoped_vmem_bytes: Per-kernel scoped VMEM budget in bytes (default: 64 MiB).
    interpret: Run in Pallas interpret mode on CPU (for unit tests).

  Returns:
    Result array of shape `[M, N]` in `out_dtype`.
  """
  if scaling not in ("dynamic", "static"):
    raise ValueError(
        f"Unsupported scaling mode '{scaling}'; expected 'dynamic' or 'static'."
    )
  m, k_dim = lhs.shape
  if rhs_transposed:
    n, k_dim_rhs = rhs.shape
  else:
    k_dim_rhs, n = rhs.shape
  if k_dim != k_dim_rhs:
    raise ValueError(
        f"Contracting dimensions must match, got lhs={lhs.shape} and"
        f" rhs={rhs.shape} (rhs_transposed={rhs_transposed})."
    )

  def_bm, def_bk, def_bn, def_lane_mult = default_quantized_matmul_blocks(
      m, k_dim, n, sc_size=sc_size, quantize_rhs=quantize_rhs
  )
  bm = def_bm if block_m is None else block_m
  bk = def_bk if block_k is None else block_k
  bn = def_bn if block_n is None else block_n
  lane_mult = def_lane_mult if n_lane_multiplier is None else n_lane_multiplier

  if bm <= 0 or bk <= 0 or bn <= 0:
    raise ValueError(f"Tile ({bm}, {bk}, {bn}) must be positive.")
  if m % bm != 0 or n % bn != 0 or k_dim % bk != 0:
    raise ValueError(
        f"Tile ({bm}, {bk}, {bn}) must divide (M={m}, K={k_dim}, N={n})."
    )
  full_k_rowwise = sc_size == k_dim and sc_size > bk
  if not full_k_rowwise and bk % sc_size != 0:
    raise ValueError(
        f"block_k ({bk}) must be divisible by sc_size ({sc_size}) unless"
        f" sc_size == K ({k_dim})."
    )

  q_dtype = jnp.dtype(quant_dtype)
  rhs_q_dtype = (
      q_dtype if rhs_quant_dtype is None else jnp.dtype(rhs_quant_dtype)
  )
  o_dtype = jnp.dtype(out_dtype)
  a_dtype = jnp.dtype(acc_dtype)
  dot_acc_dtype = (
      jnp.int32
      if (
          jnp.issubdtype(q_dtype, jnp.integer)
          and jnp.issubdtype(rhs_q_dtype, jnp.integer)
      )
      else jnp.float32
  )

  mxu_col = _get_mxu_column_size()
  compute_tile_n = min(bn, mxu_col * lane_mult)
  while bn % compute_tile_n != 0 and compute_tile_n > 1:
    compute_tile_n //= 2
  steps_n = bn // compute_tile_n

  use_w_scales = (not quantize_rhs) and dequantize
  if use_w_scales and w_scales is None:
    raise ValueError(
        "w_scales must be provided when quantize_rhs=False and dequantize=True."
    )

  grid = (m // bm, n // bn, k_dim // bk)
  block_spec_lhs = pl.BlockSpec(
      (bm, bk), lambda i, j, k: (i, k), memory_space=pltpu.VMEM
  )
  if rhs_transposed:
    block_spec_rhs = pl.BlockSpec(
        (bn, bk), lambda i, j, k: (j, k), memory_space=pltpu.VMEM
    )
  else:
    block_spec_rhs = pl.BlockSpec(
        (bk, bn), lambda i, j, k: (k, j), memory_space=pltpu.VMEM
    )
  block_spec_out = pl.BlockSpec((bm, bn), lambda i, j, k: (i, j))
  scratch_shape = pltpu.VMEM((bm, bn), a_dtype)
  compiler_params = pltpu.CompilerParams(
      dimension_semantics=("parallel", "parallel", "arbitrary"),
      vmem_limit_bytes=scoped_vmem_bytes,
  )

  lhs_hbm = pltpu.with_memory_space_constraint(lhs, pltpu.HBM)
  rhs_hbm = pltpu.with_memory_space_constraint(rhs, pltpu.HBM)

  if full_k_rowwise:
    # Multi-K-tile row-wise mode (sc_size == K > bk): compute 1D row/col scales
    # stored as (1, M) and (1, N) so 2D BlockSpecs slice only the lane-aligned
    # minor dimension, then fuse `scale + cast` into each VMEM K tile.
    lhs_q_max = float(dtypes.get_dtype_max(q_dtype))
    rhs_q_max = float(dtypes.get_dtype_max(rhs_q_dtype))
    lhs_sf, lhs_inv = _compute_scales_and_inv(
        lhs.astype(jnp.float32), axis=1, q_max=lhs_q_max, scaling=scaling
    )
    lhs_scale_row = pltpu.with_memory_space_constraint(
        lhs_sf.reshape(1, m), pltpu.HBM
    )
    lhs_inv_row = pltpu.with_memory_space_constraint(
        lhs_inv.reshape(1, m), pltpu.HBM
    )
    spec_m = pl.BlockSpec(
        (1, bm), lambda i, j, k: (0, i), memory_space=pltpu.VMEM
    )
    spec_n = pl.BlockSpec(
        (1, bn), lambda i, j, k: (0, j), memory_space=pltpu.VMEM
    )

    if quantize_rhs:
      rhs_sf, rhs_inv = _compute_scales_and_inv(
          rhs.astype(jnp.float32),
          axis=1 if rhs_transposed else 0,
          q_max=rhs_q_max,
          scaling=scaling,
      )
      rhs_scale_row = pltpu.with_memory_space_constraint(
          rhs_sf.reshape(1, n), pltpu.HBM
      )
      rhs_inv_row = pltpu.with_memory_space_constraint(
          rhs_inv.reshape(1, n), pltpu.HBM
      )
    else:
      rhs_inv_row = None
      rhs_scale_row = (
          pltpu.with_memory_space_constraint(w_scales.reshape(1, n), pltpu.HBM)
          if (use_w_scales and w_scales is not None)
          else None
      )

    def _rowwise_body(
        lhs_ref,
        rhs_ref,
        lhs_inv_ref,
        rhs_inv_ref,
        lhs_scale_ref,
        rhs_scale_ref,
        out_ref,
        acc_scratch,
    ):
      pid_k = pl.program_id(2)
      is_first_step = pid_k == 0
      is_last_step = pid_k == (k_dim // bk - 1)

      @pl.when(is_first_step)
      def _init():
        acc_scratch[...] = jnp.zeros_like(acc_scratch)

      lhs_inv_tile = jnp.transpose(lhs_inv_ref[...], (1, 0))
      lhs_q = _scale_and_cast(
          lhs_ref[...].astype(jnp.float32),
          lhs_inv_tile,
          quant_dtype=q_dtype,
          q_max=lhs_q_max,
      )

      accumulators = []
      for j in range(steps_n):
        n_start, n_end = j * compute_tile_n, (j + 1) * compute_tile_n
        if quantize_rhs:
          rhs_sub = (
              rhs_ref[n_start:n_end, :]
              if rhs_transposed
              else rhs_ref[:, n_start:n_end]
          ).astype(jnp.float32)
          rhs_inv_slice = rhs_inv_ref[:, n_start:n_end]
          if rhs_transposed:
            rhs_inv_slice = jnp.transpose(rhs_inv_slice, (1, 0))
          rhs_q_slice = _scale_and_cast(
              rhs_sub,
              rhs_inv_slice,
              quant_dtype=rhs_q_dtype,
              q_max=rhs_q_max,
          )
        else:
          rhs_q_slice = (
              rhs_ref[n_start:n_end, :]
              if rhs_transposed
              else rhs_ref[:, n_start:n_end]
          )

        dot_res = jax.lax.dot_general(
            lhs_q,
            rhs_q_slice,
            (((1,), (1,)), ((), ()))
            if rhs_transposed
            else (((1,), (0,)), ((), ())),
            preferred_element_type=dot_acc_dtype,
        )
        accumulators.append(dot_res.astype(a_dtype))

      acc_block = jnp.concatenate(accumulators, axis=1)
      acc_scratch[...] = acc_scratch[...] + acc_block

      @pl.when(is_last_step)
      def _write():
        acc_val = acc_scratch[...]
        if dequantize:
          l_s = jnp.transpose(lhs_scale_ref[...], (1, 0)).astype(a_dtype)
          r_s = rhs_scale_ref[...].astype(a_dtype)
          acc_val = acc_val * (l_s * r_s)
        out_ref[...] = acc_val.astype(out_ref.dtype)

    if quantize_rhs and dequantize:
      return pl.pallas_call(
          _rowwise_body,
          out_shape=pltpu.HBM((m, n), o_dtype),
          grid=grid,
          in_specs=[
              block_spec_lhs,
              block_spec_rhs,
              spec_m,
              spec_n,
              spec_m,
              spec_n,
          ],
          out_specs=block_spec_out,
          scratch_shapes=[scratch_shape],
          compiler_params=compiler_params,
          interpret=interpret,
      )(
          lhs_hbm,
          rhs_hbm,
          lhs_inv_row,
          rhs_inv_row,
          lhs_scale_row,
          rhs_scale_row,
      )
    elif quantize_rhs and not dequantize:
      return pl.pallas_call(
          lambda l, r, li, ri, o, s: _rowwise_body(
              l, r, li, ri, None, None, o, s
          ),
          out_shape=pltpu.HBM((m, n), o_dtype),
          grid=grid,
          in_specs=[block_spec_lhs, block_spec_rhs, spec_m, spec_n],
          out_specs=block_spec_out,
          scratch_shapes=[scratch_shape],
          compiler_params=compiler_params,
          interpret=interpret,
      )(lhs_hbm, rhs_hbm, lhs_inv_row, rhs_inv_row)
    elif not quantize_rhs and dequantize:
      return pl.pallas_call(
          lambda l, r, li, ls, rs, o, s: _rowwise_body(
              l, r, li, None, ls, rs, o, s
          ),
          out_shape=pltpu.HBM((m, n), o_dtype),
          grid=grid,
          in_specs=[block_spec_lhs, block_spec_rhs, spec_m, spec_m, spec_n],
          out_specs=block_spec_out,
          scratch_shapes=[scratch_shape],
          compiler_params=compiler_params,
          interpret=interpret,
      )(lhs_hbm, rhs_hbm, lhs_inv_row, lhs_scale_row, rhs_scale_row)
    else:
      return pl.pallas_call(
          lambda l, r, li, o, s: _rowwise_body(
              l, r, li, None, None, None, o, s
          ),
          out_shape=pltpu.HBM((m, n), o_dtype),
          grid=grid,
          in_specs=[block_spec_lhs, block_spec_rhs, spec_m],
          out_specs=block_spec_out,
          scratch_shapes=[scratch_shape],
          compiler_params=compiler_params,
          interpret=interpret,
      )(lhs_hbm, rhs_hbm, lhs_inv_row)

  steps_k = bk // sc_size

  def _body(lhs_ref, rhs_ref, w_scales_ref, out_ref, acc_scratch):
    pid_k = pl.program_id(2)
    is_first_step = pid_k == 0
    is_last_step = pid_k == (k_dim // bk - 1)

    @pl.when(is_first_step)
    def _init():
      acc_scratch[...] = jnp.zeros_like(acc_scratch)

    accumulators = [
        jnp.zeros((bm, compute_tile_n), dtype=a_dtype) for _ in range(steps_n)
    ]

    for i in range(steps_k):
      k_start, k_end = i * sc_size, (i + 1) * sc_size
      lhs_sub = lhs_ref[:, k_start:k_end].astype(jnp.float32)
      lhs_q, l_s = _quantize_subchannel_tile(
          lhs_sub,
          axis=1,
          quant_dtype=q_dtype,
          scaling=scaling,
      )
      lhs_scale = l_s.astype(a_dtype)

      for j in range(steps_n):
        n_start, n_end = j * compute_tile_n, (j + 1) * compute_tile_n
        if quantize_rhs:
          rhs_sub = (
              rhs_ref[n_start:n_end, k_start:k_end]
              if rhs_transposed
              else rhs_ref[k_start:k_end, n_start:n_end]
          ).astype(jnp.float32)
          rhs_q_slice, r_s = _quantize_subchannel_tile(
              rhs_sub,
              axis=1 if rhs_transposed else 0,
              quant_dtype=rhs_q_dtype,
              scaling=scaling,
          )
          rhs_scale_slice = (r_s.T if rhs_transposed else r_s).astype(a_dtype)
        else:
          rhs_q_slice = (
              rhs_ref[n_start:n_end, k_start:k_end]
              if rhs_transposed
              else rhs_ref[k_start:k_end, n_start:n_end]
          )
          rhs_scale_slice = (
              w_scales_ref[i, :, n_start:n_end].astype(a_dtype)
              if use_w_scales
              else None
          )

        dot_res = jax.lax.dot_general(
            lhs_q,
            rhs_q_slice,
            (((1,), (1,)), ((), ()))
            if rhs_transposed
            else (((1,), (0,)), ((), ())),
            preferred_element_type=dot_acc_dtype,
        )
        res = dot_res.astype(a_dtype)
        if dequantize:
          assert rhs_scale_slice is not None
          res = res * lhs_scale
          res = res * rhs_scale_slice
        accumulators[j] = accumulators[j] + res

    acc_block = jnp.concatenate(accumulators, axis=1)
    acc_scratch[...] = acc_scratch[...] + acc_block

    @pl.when(is_last_step)
    def _write():
      out_ref[...] = acc_scratch[...].astype(out_ref.dtype)

  if use_w_scales:
    assert w_scales is not None
    w_scales_3d = w_scales.reshape(k_dim // sc_size, 1, n)
    block_spec_w_scales = pl.BlockSpec(
        (steps_k, 1, bn),
        lambda _, j, k: (k, 0, j),
        memory_space=pltpu.VMEM,
    )
    w_scales_hbm = pltpu.with_memory_space_constraint(w_scales_3d, pltpu.HBM)
    return pl.pallas_call(
        _body,
        out_shape=pltpu.HBM((m, n), o_dtype),
        grid=grid,
        in_specs=[block_spec_lhs, block_spec_rhs, block_spec_w_scales],
        out_specs=block_spec_out,
        scratch_shapes=[scratch_shape],
        compiler_params=compiler_params,
        interpret=interpret,
    )(lhs_hbm, rhs_hbm, w_scales_hbm)
  else:

    def _body_no_w_scales(lhs_ref, rhs_ref, out_ref, acc_scratch):
      return _body(lhs_ref, rhs_ref, None, out_ref, acc_scratch)

    return pl.pallas_call(
        _body_no_w_scales,
        out_shape=pltpu.HBM((m, n), o_dtype),
        grid=grid,
        in_specs=[block_spec_lhs, block_spec_rhs],
        out_specs=block_spec_out,
        scratch_shapes=[scratch_shape],
        compiler_params=compiler_params,
        interpret=interpret,
    )(lhs_hbm, rhs_hbm)
