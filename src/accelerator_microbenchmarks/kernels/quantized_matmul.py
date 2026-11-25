"""Fused Pallas TPU kernel for sub-channel blockwise quantized GEMM."""

from typing import NamedTuple

from accelerator_microbenchmarks.core import dtypes
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Usable fraction (60%) of per-core VMEM for explicit DMA tile buffers plus
# `acc_scratch` (e.g., <= 76.8 MiB on 128 MiB VMEM, admitting the 64 MiB
# `(2048, 2048, 2048)` tile; <= 38.4 MiB on 64 MiB VMEM), leaving ~40% headroom
# for in-kernel float32 quantization/dot strips (~10-25 MiB) and XLA
# overlay/SparseCore reserved VMEM.
_USABLE_VMEM_NUMERATOR = 3
_USABLE_VMEM_DENOMINATOR = 5

# Maximum number of float32 elements per inner N-strip (`16 * mxu_col^2`, e.g.
# `4 MiB` on 256-wide MXUs) to bound live in-kernel `float32` quantization and
# dot temporaries within the VMEM scratch headroom across TPU generations.
_MAX_ACC_STRIP_MXU_BLOCKS = 16

# Matches `_quantize_per_channel` in `benchmarks/matmul.py`.
_DYNAMIC_AMAX_EPS = 1e-7
_STATIC_SCALE_FACTOR = 0.5

# Supported unquantized float and quantized dtypes.
_SUPPORTED_FLOAT_DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
_SUPPORTED_LHS_QUANT_DTYPES = (
    jnp.dtype(jnp.float8_e4m3fn),
    jnp.dtype(jnp.float8_e5m2),
    jnp.dtype(jnp.int8),
)


def default_vmem_budget_bytes() -> int:
  """Returns per-TensorCore VMEM capacity in bytes from TPU hardware specs."""
  return pltpu.get_tpu_info().vmem_capacity_bytes


def _mxu_column_size() -> int:
  """Returns the hardware MXU systolic column width (`128` or `256`)."""
  return pltpu.get_tpu_info().mxu_column_size


def _padded_tile_bytes(rows: int, cols: int, bits: int) -> int:
  """Returns the hardware-padded VMEM footprint in bytes of a 2D tile."""
  sublanes, lanes = pltpu.Tiling.COMPACT.shape
  row_align = sublanes if rows == 1 else sublanes * max(1, 32 // bits)
  padded_elems = pl.align_to(rows, row_align) * pl.align_to(cols, lanes)
  return pl.cdiv(padded_elems * bits, 8)


def _estimate_vmem_bytes(
    m: int,
    k: int,
    n: int,
    block_m: int,
    block_k: int,
    block_n: int,
    *,
    in_bits: int,
    rhs_bits: int,
    out_bits: int,
    acc_bits: int,
) -> int:
  """Estimates explicit DMA buffer and accumulator VMEM bytes."""
  lhs_bufs = 1 if (block_m == m and block_k == k) else 2
  rhs_bufs = 1 if (block_k == k and block_n == n) else 2
  out_bufs = 1 if (block_m == m and block_n == n) else 2
  lhs_bytes = lhs_bufs * _padded_tile_bytes(block_m, block_k, in_bits)
  rhs_bytes = rhs_bufs * _padded_tile_bytes(block_k, block_n, rhs_bits)
  out_bytes = out_bufs * _padded_tile_bytes(block_m, block_n, out_bits)
  acc_bytes = _padded_tile_bytes(block_m, block_n, acc_bits)
  return lhs_bytes + rhs_bytes + out_bytes + acc_bytes


def _aligned_divisors(dim: int, min_align: int) -> tuple[int, ...]:
  """Returns descending divisors of `dim` aligned to `min_align`, plus `dim`."""
  align = max(1, min_align)
  if dim % align != 0:
    return (dim,)
  num_groups = dim // align
  divisors = {dim}
  d = 1
  while d * d <= num_groups:
    if num_groups % d == 0:
      divisors.add(d * align)
      divisors.add((num_groups // d) * align)
    d += 1
  return tuple(sorted(divisors, reverse=True))


def validate_quantized_matmul_blocks(
    m: int,
    k: int,
    n: int,
    *,
    sc_size: int,
    block_m: int,
    block_k: int,
    block_n: int,
    in_dtype: jax.typing.DTypeLike,
    out_dtype: jax.typing.DTypeLike,
    acc_dtype: jax.typing.DTypeLike = jnp.float32,
    vmem_budget_bytes: int | None = None,
) -> None:
  """Validates explicit `(block_m, block_k, block_n)` tile sizes.

  Args:
    m: Rows of `lhs` and output.
    k: Contracting dimension shared by `lhs` and `rhs`.
    n: Columns of `rhs` and output.
    sc_size: Sub-channel quantization block size along `K` (`0 < sc_size < k`,
      multiple of TPU lane width `128`).
    block_m: Tile size along `M`.
    block_k: Tile size along `K`.
    block_n: Tile size along `N`.
    in_dtype: Input operand element dtype.
    out_dtype: Output element dtype.
    acc_dtype: Accumulator scratch dtype.
    vmem_budget_bytes: Per-core VMEM capacity in bytes; defaults to hardware
      `vmem_capacity_bytes` when `None`.

  Raises:
    ValueError: If shapes, `sc_size`, divisibility, TPU alignment, or VMEM
      capacity constraints are violated.
  """
  if m <= 0 or k <= 0 or n <= 0:
    raise ValueError(f"Matrix dimensions ({m}, {k}, {n}) must be positive.")
  sublanes, lanes = pltpu.Tiling.COMPACT.shape
  if sc_size <= 0 or sc_size >= k or k % sc_size != 0:
    raise ValueError(
        f"sc_size must satisfy 0 < sc_size < k and divide k={k}, got"
        f" sc_size={sc_size}."
    )
  if sc_size % lanes != 0:
    raise ValueError(f"sc_size={sc_size} must be a multiple of lanes={lanes}.")
  if block_m <= 0 or block_k <= 0 or block_n <= 0:
    raise ValueError(
        "Tile sizes (block_m, block_k, block_n) must be positive, got"
        f" ({block_m}, {block_k}, {block_n})."
    )
  if m % block_m != 0 or k % block_k != 0 or n % block_n != 0:
    raise ValueError(
        f"Tile sizes ({block_m}, {block_k}, {block_n}) must divide matrix"
        f" dimensions ({m}, {k}, {n})."
    )
  if block_k % sc_size != 0:
    raise ValueError(
        f"block_k={block_k} must be divisible by sc_size={sc_size}."
    )

  if block_m < m and block_m % sublanes != 0:
    raise ValueError(
        f"block_m={block_m} (< m={m}) must be a multiple of"
        f" sublanes={sublanes}."
    )
  if block_n < n and block_n % lanes != 0:
    raise ValueError(
        f"block_n={block_n} (< n={n}) must be a multiple of lanes={lanes}."
    )

  budget = (
      default_vmem_budget_bytes()
      if vmem_budget_bytes is None
      else vmem_budget_bytes
  )
  if budget <= 0:
    raise ValueError(
        f"vmem_budget_bytes must be positive, got {vmem_budget_bytes}."
    )
  usable_vmem = (budget * _USABLE_VMEM_NUMERATOR) // _USABLE_VMEM_DENOMINATOR
  in_bits = dtypes.get_dtype_bits(jnp.dtype(in_dtype))
  out_bits = dtypes.get_dtype_bits(jnp.dtype(out_dtype))
  acc_bits = dtypes.get_dtype_bits(jnp.dtype(acc_dtype))
  required_bytes = _estimate_vmem_bytes(
      m,
      k,
      n,
      block_m,
      block_k,
      block_n,
      in_bits=in_bits,
      rhs_bits=in_bits,
      out_bits=out_bits,
      acc_bits=acc_bits,
  )
  if required_bytes > usable_vmem:
    raise ValueError(
        f"Tile sizes ({block_m}, {block_k}, {block_n}) require"
        f" {required_bytes / (1 << 20):.2f} MiB of tile VMEM, exceeding usable"
        f" budget {usable_vmem / (1 << 20):.2f} MiB (60% of"
        f" {budget / (1 << 20):.2f} MiB VMEM capacity)."
    )


def default_quantized_matmul_blocks(
    m: int,
    k: int,
    n: int,
    *,
    sc_size: int,
    in_dtype: jax.typing.DTypeLike,
    out_dtype: jax.typing.DTypeLike,
    acc_dtype: jax.typing.DTypeLike = jnp.float32,
    vmem_budget_bytes: int | None = None,
) -> tuple[int, int, int]:
  """Selects `(block_m, block_k, block_n)` from TPU hardware VMEM specs.

  Queries per-core `vmem_capacity_bytes` (e.g. `64 MiB` on TPU v7x, `128 MiB`
  on TPU v6e) and `mxu_column_size`, accounting for single-buffered stationary
  full-matrix tiles, then ranks aligned dividing tiles that fit in usable VMEM
  by:
    1. Sufficient MXU tile widths (`block_m >= 2 * mxu_col`, `block_k >= 4 *
       mxu_col`, `block_n >= 4 * mxu_col` when matrix dimensions permit) and
       `block_m <= 8 * mxu_col` to bound live in-kernel `float32` quantization
       and dot temporaries within the VMEM scratch headroom across TPU
       generations.
    2. Sufficient grid pipeline steps (`>= 4..8` on large tensors) so DMA
       cold-fill and drain latencies overlap with compute.
    3. Lowest combined HBM traffic and VPU re-quantization work across the grid.
    4. Largest `block_k`, largest output tile `block_m * block_n`, and balanced
       output aspect ratio.

  Args:
    m: Rows of `lhs` and output.
    k: Contracting dimension shared by `lhs` and `rhs`.
    n: Columns of `rhs` and output.
    sc_size: Sub-channel quantization block size along `K` (`0 < sc_size < k`,
      multiple of TPU lane width `128`).
    in_dtype: Input operand element dtype.
    out_dtype: Output element dtype.
    acc_dtype: Accumulator scratch dtype.
    vmem_budget_bytes: Optional VMEM capacity override in bytes; defaults to
      `default_vmem_budget_bytes()` when `None`.

  Returns:
    A validated `(block_m, block_k, block_n)` tuple.

  Raises:
    ValueError: If inputs are invalid or no aligned dividing tile fits in VMEM.
  """
  if m <= 0 or k <= 0 or n <= 0:
    raise ValueError(f"Matrix dimensions ({m}, {k}, {n}) must be positive.")
  sublanes, lanes = pltpu.Tiling.COMPACT.shape
  if sc_size <= 0 or sc_size >= k or k % sc_size != 0:
    raise ValueError(
        f"sc_size must satisfy 0 < sc_size < k and divide k={k}, got"
        f" sc_size={sc_size}."
    )
  if sc_size % lanes != 0:
    raise ValueError(f"sc_size={sc_size} must be a multiple of lanes={lanes}.")
  budget = (
      default_vmem_budget_bytes()
      if vmem_budget_bytes is None
      else vmem_budget_bytes
  )
  if budget <= 0:
    raise ValueError(
        f"vmem_budget_bytes must be positive, got {vmem_budget_bytes}."
    )

  mxu_col = _mxu_column_size()
  in_bits = dtypes.get_dtype_bits(jnp.dtype(in_dtype))
  out_bits = dtypes.get_dtype_bits(jnp.dtype(out_dtype))
  acc_bits = dtypes.get_dtype_bits(jnp.dtype(acc_dtype))
  usable_vmem = (budget * _USABLE_VMEM_NUMERATOR) // _USABLE_VMEM_DENOMINATOR
  max_m_tile = 8 * mxu_col

  alg_io_bytes = (m * k * in_bits + k * n * in_bits + m * n * out_bits) // 8
  full_vmem = _estimate_vmem_bytes(
      m,
      k,
      n,
      m,
      k,
      n,
      in_bits=in_bits,
      rhs_bits=in_bits,
      out_bits=out_bits,
      acc_bits=acc_bits,
  )
  if (
      m <= max_m_tile
      and alg_io_bytes <= usable_vmem // 12
      and full_vmem <= usable_vmem
  ):
    return (m, k, n)

  min_wide_m = min(m, 2 * mxu_col)
  min_wide_k = min(k, max(4 * mxu_col, sc_size))
  min_wide_n = min(n, 4 * mxu_col)
  if alg_io_bytes >= usable_vmem:
    min_steps = 8
  elif alg_io_bytes >= usable_vmem // 6:
    min_steps = 4
  else:
    min_steps = 1

  best_tile: tuple[int, int, int] | None = None
  best_key: tuple[int, int, int, int, int, int, int, int] | None = None
  n_align = mxu_col if n % mxu_col == 0 else lanes
  for bm in _aligned_divisors(m, sublanes):
    for bk in _aligned_divisors(k, sc_size):
      for bn in _aligned_divisors(n, n_align):
        if bm == m and bk == k and bn == n and min_steps > 1:
          continue
        if bk < k:
          in_buffer_bytes = 2 * _padded_tile_bytes(
              bm, bk, in_bits
          ) + 2 * _padded_tile_bytes(bk, bn, in_bits)
          if in_buffer_bytes > usable_vmem // 2:
            continue
        vmem_bytes = _estimate_vmem_bytes(
            m,
            k,
            n,
            bm,
            bk,
            bn,
            in_bits=in_bits,
            rhs_bits=in_bits,
            out_bits=out_bits,
            acc_bits=acc_bits,
        )
        if vmem_bytes > usable_vmem:
          continue

        gm, gn, gk = m // bm, n // bn, k // bk
        total_steps = gm * gn * gk
        has_wide = (
            1
            if (bm >= min_wide_m and bk >= min_wide_k and bn >= min_wide_n)
            else 0
        )
        within_max_m = 1 if bm <= max_m_tile else 0
        has_enough_steps = 1 if total_steps >= min_steps else 0

        lhs_hbm_reads = 1 if gk == 1 else gn
        rhs_hbm_reads = 1 if (gk == 1 and gn == 1) else gm
        hbm_bits = (
            lhs_hbm_reads * m * k * in_bits
            + rhs_hbm_reads * k * n * in_bits
            + m * n * out_bits
        )
        vpu_bits = gn * m * k * 32 + gm * k * n * 32
        total_cost = hbm_bits + vpu_bits
        key = (
            has_wide,
            within_max_m,
            has_enough_steps,
            -total_cost,
            bk,
            bm * bn,
            -max(bm, bn),
            bn,
        )
        if best_key is None or key > best_key:
          best_key = key
          best_tile = (bm, bk, bn)

  if best_tile is not None:
    return best_tile
  if full_vmem <= usable_vmem:
    return (m, k, n)

  raise ValueError(
      f"No VMEM-fitting tile divides ({m}, {k}, {n}) with sc_size={sc_size}"
      f" within vmem_budget_bytes={budget}."
  )


def _select_compute_tile_n(block_m: int, block_n: int) -> int:
  """Selects an inner N strip width dividing `block_n` within strip capacity."""
  mxu_col = _mxu_column_size()
  _, lanes = pltpu.Tiling.COMPACT.shape
  max_strip_elems = _MAX_ACC_STRIP_MXU_BLOCKS * mxu_col * mxu_col
  max_cols = max(mxu_col, max_strip_elems // max(1, block_m))
  limit = min(block_n, 16 * mxu_col, max_cols)
  for align in (mxu_col, lanes):
    if block_n % align != 0:
      continue
    for cand in _aligned_divisors(block_n, align):
      if cand <= limit:
        return cand
  return block_n


def _scale_and_cast(
    tile: jax.Array,
    inv_scale_f32: jax.Array | float,
    *,
    q_dtype: jnp.dtype,
    q_max: float,
) -> jax.Array:
  """Scales a VMEM tile by `inv_scale_f32` and casts to `q_dtype`."""
  q = tile.astype(jnp.float32) * inv_scale_f32
  if jnp.issubdtype(q_dtype, jnp.integer):
    q = jnp.clip(jnp.round(q), -q_max, q_max)
  else:
    q = jnp.clip(q, -q_max, q_max)
  return q.astype(q_dtype)


def _quantize_tile(
    tile: jax.Array,
    *,
    q_dtype: jnp.dtype,
    q_max: float,
    reduce_axis: int,
    dynamic: bool,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes a 2D sub-channel tile and returns `(q_tile, scale_f32)`."""
  if dynamic:
    # Reduce `abs` and `max` in native input precision (e.g. bfloat16, bit-exact
    # with float32) at 2x VPU/XLU throughput before casting amax to float32.
    amax = jnp.max(jnp.abs(tile), axis=reduce_axis, keepdims=True).astype(
        jnp.float32
    )
    clamped_amax = jnp.maximum(amax, _DYNAMIC_AMAX_EPS)
    scale_f32 = clamped_amax / q_max
    inv_scale_f32 = q_max / clamped_amax
  else:
    scale_f32 = jnp.full(
        (tile.shape[0], 1) if reduce_axis == 1 else (1, tile.shape[1]),
        _STATIC_SCALE_FACTOR,
        dtype=jnp.float32,
    )
    inv_scale_f32 = 1.0 / _STATIC_SCALE_FACTOR
  q_tile = _scale_and_cast(tile, inv_scale_f32, q_dtype=q_dtype, q_max=q_max)
  return q_tile, scale_f32


class _ValidatedOperands(NamedTuple):
  m: int
  k: int
  n: int
  in_dtype: jnp.dtype
  lhs_quant_dtype: jnp.dtype
  rhs_quant_dtype: jnp.dtype
  out_dtype: jnp.dtype


def validate_quant_dtype_pair(
    quant_dtype: jax.typing.DTypeLike,
    rhs_quant_dtype: jax.typing.DTypeLike,
) -> None:
  """Validates that `(quant_dtype, rhs_quant_dtype)` is a supported pair.

  Args:
    quant_dtype: Quantized LHS operand dtype.
    rhs_quant_dtype: Quantized RHS operand dtype.

  Raises:
    ValueError: If `quant_dtype` is unsupported or `(quant_dtype,
      rhs_quant_dtype)` is not an allowed pairing.
  """
  lhs_q_dt = jnp.dtype(quant_dtype)
  rhs_q_dt = jnp.dtype(rhs_quant_dtype)
  if lhs_q_dt not in _SUPPORTED_LHS_QUANT_DTYPES:
    raise ValueError(
        "quant_dtype must be float8_e4m3fn, float8_e5m2, or int8, got"
        f" {lhs_q_dt}."
    )
  if rhs_q_dt != lhs_q_dt and not (
      rhs_q_dt == jnp.int4 and lhs_q_dt in (jnp.float8_e4m3fn, jnp.float8_e5m2)
  ):
    raise ValueError(
        "rhs_quant_dtype must match quant_dtype or be 'int4' paired with an"
        " FP8 quant_dtype ('float8_e4m3fn' or 'float8_e5m2'), got"
        f" quant_dtype='{lhs_q_dt}', rhs_quant_dtype='{rhs_q_dt}'."
    )


def _validate_operands_and_dtypes(
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    quant_dtype: jax.typing.DTypeLike,
    rhs_quant_dtype: jax.typing.DTypeLike | None,
    out_dtype: jax.typing.DTypeLike | None,
) -> _ValidatedOperands:
  """Validates operand shapes and resolves input/quant/output dtypes."""
  if lhs.ndim != 2 or rhs.ndim != 2:
    raise ValueError(
        f"lhs and rhs must be 2D, got shapes {lhs.shape} and {rhs.shape}."
    )
  m, k = lhs.shape
  k_rhs, n = rhs.shape
  if k != k_rhs:
    raise ValueError(
        f"Contracting dimension mismatch: lhs {lhs.shape} vs rhs {rhs.shape}."
    )

  in_dt = jnp.dtype(lhs.dtype)
  if in_dt not in _SUPPORTED_FLOAT_DTYPES:
    raise ValueError(f"lhs.dtype must be bfloat16 or float32, got {in_dt}.")
  if jnp.dtype(rhs.dtype) != in_dt:
    raise ValueError(f"rhs.dtype ({rhs.dtype}) must match lhs.dtype ({in_dt}).")

  lhs_q_dt = jnp.dtype(quant_dtype)
  rhs_q_dt = lhs_q_dt if rhs_quant_dtype is None else jnp.dtype(rhs_quant_dtype)
  validate_quant_dtype_pair(lhs_q_dt, rhs_q_dt)

  out_dt = in_dt if out_dtype is None else jnp.dtype(out_dtype)
  if out_dt not in _SUPPORTED_FLOAT_DTYPES:
    raise ValueError(f"out_dtype must be bfloat16 or float32, got {out_dt}.")

  return _ValidatedOperands(
      m=m,
      k=k,
      n=n,
      in_dtype=in_dt,
      lhs_quant_dtype=lhs_q_dt,
      rhs_quant_dtype=rhs_q_dt,
      out_dtype=out_dt,
  )


def quantized_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    sc_size: int,
    block_m: int,
    block_k: int,
    block_n: int,
    quant_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    rhs_quant_dtype: jax.typing.DTypeLike | None = None,
    out_dtype: jax.typing.DTypeLike | None = None,
    dynamic: bool = True,
    dequantize: bool = True,
    vmem_budget_bytes: int | None = None,
    interpret: bool = False,
) -> jax.Array:
  """Fused Pallas kernel for on-the-fly sub-channel quantized GEMM.

  Streams `[block_m, block_k]` LHS tiles and `[block_k, block_n]` RHS tiles
  from HBM into VMEM, quantizes one `[block_m, sc_size]` LHS sub-channel block
  at a time on the VPU, reuses it across N-strips on the MXU without
  materializing quantized tensors or 3D reshapes in HBM, rescales with 1D
  left-associated broadcasts `(dot * sf_a) * sf_b`, and accumulates into VMEM
  `float32` scratch.

  Args:
    lhs: 2D unquantized activation matrix `[M, K]` (`bfloat16` or `float32`).
    rhs: 2D unquantized weight matrix `[K, N]` matching `lhs.dtype`.
    sc_size: Sub-channel block size along `K` (`0 < sc_size < K`, multiple of
      `128` dividing `block_k`).
    block_m: Explicit tile size along `M` (must divide `M`).
    block_k: Explicit tile size along `K` (must divide `K` and be divisible by
      `sc_size`).
    block_n: Explicit tile size along `N` (must divide `N`).
    quant_dtype: Quantized LHS (and default RHS) dtype (`float8_e4m3fn`,
      `float8_e5m2`, or `int8`).
    rhs_quant_dtype: Optional quantized RHS dtype; defaults to `quant_dtype`
      when `None`; may also be `int4` when `quant_dtype` is FP8 (`W4A8`).
    out_dtype: Output tensor dtype (`bfloat16` or `float32`); defaults to
      `lhs.dtype` when `None`.
    dynamic: If `True`, computes per-subchannel `amax` on the fly; if `False`,
      applies static clip bound `0.5 * Q_MAX` (`scale = 0.5`).
    dequantize: If `True`, multiplies each sub-channel MXU dot product by its
      LHS and RHS dequantization scales before accumulating.
    vmem_budget_bytes: Optional per-core VMEM limit in bytes; defaults to
      `default_vmem_budget_bytes()` when `None`.
    interpret: Runs `pallas_call` in CPU interpret mode for unit tests.

  Returns:
    Output matrix `[M, N]` of dtype `out_dtype`.

  Raises:
    ValueError: If operand shapes, dtypes, `sc_size`, or explicit block sizes
      are invalid or exceed usable hardware VMEM.
  """
  # 1. Validate operand shapes and resolve input/quant/output dtypes.
  operands = _validate_operands_and_dtypes(
      lhs,
      rhs,
      quant_dtype=quant_dtype,
      rhs_quant_dtype=rhs_quant_dtype,
      out_dtype=out_dtype,
  )

  # 2. Validate explicit tile sizes against shape alignment and hardware VMEM.
  budget = (
      default_vmem_budget_bytes()
      if vmem_budget_bytes is None
      else vmem_budget_bytes
  )
  validate_quantized_matmul_blocks(
      operands.m,
      operands.k,
      operands.n,
      sc_size=sc_size,
      block_m=block_m,
      block_k=block_k,
      block_n=block_n,
      in_dtype=operands.in_dtype,
      out_dtype=operands.out_dtype,
      acc_dtype=jnp.float32,
      vmem_budget_bytes=budget,
  )

  # 3. Derive quantization bounds, MXU accumulator dtype, and grid/strip counts.
  lhs_q_dt = operands.lhs_quant_dtype
  rhs_q_dt = operands.rhs_quant_dtype
  out_dt = operands.out_dtype
  lhs_q_max = dtypes.get_dtype_max(lhs_q_dt)
  rhs_q_max = dtypes.get_dtype_max(rhs_q_dt)
  both_int = jnp.issubdtype(lhs_q_dt, jnp.integer) and jnp.issubdtype(
      rhs_q_dt, jnp.integer
  )
  dot_acc_dt = jnp.int32 if both_int else jnp.float32

  num_m = operands.m // block_m
  num_k = operands.k // block_k
  num_n = operands.n // block_n
  steps_k = block_k // sc_size
  compute_tile_n = _select_compute_tile_n(block_m, block_n)
  steps_n = block_n // compute_tile_n

  # 4. Define the fused sub-channel Pallas TensorCore kernel.
  def kernel(lhs_ref, rhs_ref, out_ref, acc_scratch):
    i_k = pl.program_id(2)

    @pl.when(i_k == 0)
    def _init_acc():
      zero_strip = jnp.zeros((block_m, compute_tile_n), dtype=jnp.float32)
      for j in range(steps_n):
        acc_scratch[:, pl.ds(j * compute_tile_n, compute_tile_n)] = zero_strip

    # Quantize one `[block_m, sc_size]` LHS sub-channel at a time and reuse it
    # across N-strips so neither full-block quantized LHS lists nor 2D scale
    # broadcast matrices spill to VMEM.
    # Static Python unrolling produces constant VMEM slice offsets and lets the
    # VLIW scheduler overlap MXU and VPU execution across strips.
    for s in range(steps_k):
      k_slice = pl.ds(s * sc_size, sc_size)
      q_a, sf_a = _quantize_tile(
          lhs_ref[:, k_slice],
          q_dtype=lhs_q_dt,
          q_max=lhs_q_max,
          reduce_axis=1,
          dynamic=dynamic,
      )
      for j in range(steps_n):
        n_slice = pl.ds(j * compute_tile_n, compute_tile_n)
        q_b, sf_b = _quantize_tile(
            rhs_ref[k_slice, n_slice],
            q_dtype=rhs_q_dt,
            q_max=rhs_q_max,
            reduce_axis=0,
            dynamic=dynamic,
        )
        dot = jax.lax.dot_general(
            q_a,
            q_b,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=dot_acc_dt,
        ).astype(jnp.float32)
        if dequantize:
          dot = (dot * sf_a) * sf_b
        acc_scratch[:, n_slice] = acc_scratch[:, n_slice] + dot

    @pl.when(i_k == num_k - 1)
    def _write_out():
      for j in range(steps_n):
        n_slice = pl.ds(j * compute_tile_n, compute_tile_n)
        out_ref[:, n_slice] = acc_scratch[:, n_slice].astype(out_dt)

  # 5. Launch pallas_call with HBM-pinned operands for MARKER XProf timing.
  return pl.pallas_call(
      kernel,
      # Pin output and inputs to HBM so all pipelined DMAs execute inside the
      # MARKER-scoped Pallas call for accurate XProf step timing.
      out_shape=pltpu.HBM((operands.m, operands.n), out_dt),
      grid=(num_m, num_n, num_k),
      in_specs=[
          pl.BlockSpec((block_m, block_k), lambda i, j, p: (i, p)),
          pl.BlockSpec((block_k, block_n), lambda i, j, p: (p, j)),
      ],
      out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, p: (i, j)),
      scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          vmem_limit_bytes=budget,
      ),
      name="quantized_matmul",
      interpret=interpret,
  )(
      pltpu.with_memory_space_constraint(lhs, pltpu.HBM),
      pltpu.with_memory_space_constraint(rhs, pltpu.HBM),
  )
