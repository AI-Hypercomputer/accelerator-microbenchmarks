"""Fused Pallas TPU kernel for the transposed quantization microbenchmark."""

import math

from accelerator_microbenchmarks.core import dtypes
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Default per-kernel scoped VMEM budget (64 MiB). Matches the minimum
# --xla_tpu_scoped_vmem_limit_kib=65536 across op_flags.yaml files and is passed
# to pltpu.CompilerParams(vmem_limit_bytes=...) so tile selection and compiler
# VMEM allocation are identical across TPU generations.
_DEFAULT_SCOPED_VMEM_BYTES = 64 << 20

# Usable fraction (80%) of scoped VMEM for persistent and double-buffered tile
# allocations, reserving 20% headroom for compiler scratch and register spills.
_USABLE_VMEM_NUMERATOR = 4
_USABLE_VMEM_DENOMINATOR = 5
# Physical DMA-pipelining thresholds determined from TPU tile sweeps:
# Inputs up to 4 MiB run as a single tile so the quantize phase reuses X in VMEM
# without a second HBM read; larger inputs split into multiple tiles so DMA
# transfers overlap with VPU compute.
_SINGLE_TILE_MAX_IN_BYTES = 4 << 20
# Cap each input tile at 16 MiB (e.g. 2048x4096 bf16 or 2048x2048 f32) so
# double-buffered input/output tiles and persistent scratch fit inside usable
# VMEM.
_MAX_INPUT_TILE_BYTES = 16 << 20


def _aligned_divisors(dim: int, lanes: int) -> tuple[int, ...]:
  """Returns divisors of `dim` aligned to `lanes` (descending), or `(dim,)`."""
  if dim % lanes != 0:
    return (dim,)
  num_lane_groups = dim // lanes
  divisors: list[int] = []
  for k in range(1, math.isqrt(num_lane_groups) + 1):
    if num_lane_groups % k == 0:
      divisors.append(k * lanes)
      if k * k != num_lane_groups:
        divisors.append((num_lane_groups // k) * lanes)
  divisors.sort(reverse=True)
  return tuple(divisors)


def default_transpose_blocks(
    m: int,
    n: int,
    *,
    in_dtype: jax.typing.DTypeLike,
    out_dtype: jax.typing.DTypeLike,
    scale_dtype: jax.typing.DTypeLike,
    scoped_vmem_bytes: int = _DEFAULT_SCOPED_VMEM_BYTES,
) -> tuple[int, int]:
  """Derives a VMEM-fitting `(block_m, block_n)` tile for an `[M, N]` input.

  Both `block_m` (minor dimension of transposed `OUT` `[N, M]`) and `block_n`
  (minor dimension of `X` `[M, N]`) must be multiples of `num_lanes` unless
  equal to the full dimension. Valid multi-tile pairs are ranked by:
    1. Largest `min(block_m, block_n)` (balanced aspect ratio so both HBM reads
       and transposed HBM writes maintain wide contiguous row bursts).
    2. Largest tile area `block_m * block_n` (fewest grid steps).
    3. Largest `block_n` (contiguous minor dimension of `X`, which is read in
       both phases).

  Args:
    m: Number of rows of `X`.
    n: Number of columns of `X`.
    in_dtype: Input element dtype.
    out_dtype: Quantized output element dtype.
    scale_dtype: Per-column scale factor dtype.
    scoped_vmem_bytes: Scoped VMEM budget in bytes (default: 64 MiB).

  Returns:
    A `(block_m, block_n)` tuple dividing `(m, n)` that fits in scoped VMEM.

  Raises:
    ValueError: If dimensions or `scoped_vmem_bytes` are non-positive, or if no
      aligned dividing tile fits within `scoped_vmem_bytes`.
  """
  if m <= 0 or n <= 0:
    raise ValueError(f"Input shape ({m}, {n}) must be positive.")
  if scoped_vmem_bytes <= 0:
    raise ValueError(
        f"scoped_vmem_bytes must be positive, got {scoped_vmem_bytes}."
    )

  sublanes, lanes = pltpu.Tiling.COMPACT.shape

  in_bits = dtypes.get_dtype_bits(jnp.dtype(in_dtype))
  out_bits = dtypes.get_dtype_bits(jnp.dtype(out_dtype))
  sf_bits = dtypes.get_dtype_bits(jnp.dtype(scale_dtype))

  usable_vmem = (
      scoped_vmem_bytes * _USABLE_VMEM_NUMERATOR
  ) // _USABLE_VMEM_DENOMINATOR
  single_tile_max_in_bytes = min(
      _SINGLE_TILE_MAX_IN_BYTES, scoped_vmem_bytes // 16
  )
  max_in_tile_bytes = min(_MAX_INPUT_TILE_BYTES, scoped_vmem_bytes // 4)

  def padded_bytes(rows: int, cols: int, bits: int) -> int:
    # 2D tiles pack `32 // bits` sublane groups along rows (`16` rows for
    # 16-bit, `32` for 8-bit, `64` for 4-bit), whereas 1-row vectors use compact
    # sublanes.
    row_align = sublanes if rows == 1 else sublanes * max(1, 32 // bits)
    padded_elems = pl.align_to(rows, row_align) * pl.align_to(cols, lanes)
    return pl.cdiv(padded_elems * bits, 8)

  # Persistent VMEM buffers held across all grid steps: float32 amax scratch
  # (sublanes, n) and single-buffered scale output (1, n).
  persistent_bytes = padded_bytes(sublanes, n, 32) + padded_bytes(1, n, sf_bits)

  def fits_vmem(bm: int, bn: int) -> bool:
    in_bytes = padded_bytes(bm, bn, in_bits)
    if in_bytes > max_in_tile_bytes:
      return False
    out_bytes = padded_bytes(bn, bm, out_bits)
    # Single-tile grid=(2, 1, 1) requests pl.Buffered(1) on x_ref (1 buffer)
    # while out_ref uses phase-indexed windowing (2 buffers). In-kernel f32
    # expressions (abs, scale multiply, transpose) are streamed in VREGs via
    # subtile unrolling rather than materialized as full tiles in VMEM.
    in_buffers = 1 if (bm == m and bn == n) else 2
    total_bytes = in_buffers * in_bytes + 2 * out_bytes + persistent_bytes
    return total_bytes <= usable_vmem

  if padded_bytes(m, n, in_bits) <= single_tile_max_in_bytes and fits_vmem(
      m, n
  ):
    return m, n

  best_tile: tuple[int, int] | None = None
  best_key: tuple[int, int, int] = (-1, -1, -1)
  for bm in _aligned_divisors(m, lanes):
    for bn in _aligned_divisors(n, lanes):
      if bm == m and bn == n:
        continue
      if not fits_vmem(bm, bn):
        continue
      key = (min(bm, bn), bm * bn, bn)
      if key > best_key:
        best_key = key
        best_tile = (bm, bn)

  if best_tile is not None:
    return best_tile
  if fits_vmem(m, n):
    return m, n

  raise ValueError(
      f"No VMEM-fitting tile divides ({m}, {n}) within"
      f" scoped_vmem_bytes={scoped_vmem_bytes} for"
      f" in_dtype={jnp.dtype(in_dtype)}, out_dtype={jnp.dtype(out_dtype)}."
  )


def quantize_transpose(
    x: jax.Array,
    *,
    out_dtype: jax.typing.DTypeLike,
    scale_dtype: jax.typing.DTypeLike,
    q_max: float,
    block_m: int | None = None,
    block_n: int | None = None,
    scoped_vmem_bytes: int = _DEFAULT_SCOPED_VMEM_BYTES,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes X^T in a single Pallas kernel without materializing X^T.

  Computes the same result as the plain JAX reference

    xt = x.T
    sf = (q_max / max(|xt|, axis=-1, keepdims=True)).astype(scale_dtype)
    out = (xt * sf).astype(out_dtype)

  with one kernel that walks over the tiles of X twice:

  1. Phase 0 (amax): reads each [block_m, block_n] tile of X and keeps a
     running max of |X| per column in VMEM scratch.
  2. Phase 1 (quantize): re-reads each tile, scales it by its columns' scale
     factors, transposes it in VMEM, casts it, and writes the
     [block_n, block_m] tile of OUT. It also writes the scale factors.

  Because both phases run in the same tile pipeline, the input DMA never
  drains between them, and with a single tile the quantize phase reuses the
  tile loaded by the amax phase. X^T is never written to HBM.

  Args:
    x: Input of shape [M, N].
    out_dtype: Quantized output dtype.
    scale_dtype: Scaling factor dtype.
    q_max: Largest finite value of `out_dtype`.
    block_m: Optional row tile size dividing `M` (must be a multiple of 128
      lanes or equal `M`). Must be set together with `block_n`; if omitted,
      derived by `default_transpose_blocks`.
    block_n: Optional column tile size dividing `N` (must be a multiple of 128
      lanes or equal `N`). Must be set together with `block_m`; if omitted,
      derived by `default_transpose_blocks`.
    scoped_vmem_bytes: Per-kernel scoped VMEM budget in bytes (default: 64 MiB),
      passed to `default_transpose_blocks` and
      `pltpu.CompilerParams(vmem_limit_bytes=...)`.
    interpret: Run the kernel in Pallas interpret mode (for CPU tests).

  Returns:
    A tuple (out, sf) with out of shape [N, M] in `out_dtype` and sf of shape
    [N, 1] in `scale_dtype`.

  Raises:
    ValueError: If `block_m` / `block_n` are invalid or no VMEM-fitting tile
      divides the input shape.
  """
  m, n = x.shape
  sublanes, lanes = pltpu.Tiling.COMPACT.shape
  if (block_m is None) != (block_n is None):
    raise ValueError(
        "block_m and block_n must both be specified or both be None."
    )
  if block_m is not None and block_n is not None:
    if block_m <= 0 or block_n <= 0 or m % block_m != 0 or n % block_n != 0:
      raise ValueError(
          f"Tile ({block_m}, {block_n}) must be positive and divide ({m}, {n})."
      )
    if (block_m != m and block_m % lanes != 0) or (
        block_n != n and block_n % lanes != 0
    ):
      raise ValueError(
          f"Tile ({block_m}, {block_n}) must be aligned to {lanes} lanes or"
          f" equal the full dimension ({m}, {n})."
      )
    bm, bn = block_m, block_n
  else:
    bm, bn = default_transpose_blocks(
        m,
        n,
        in_dtype=x.dtype,
        out_dtype=out_dtype,
        scale_dtype=scale_dtype,
        scoped_vmem_bytes=scoped_vmem_bytes,
    )

  def kernel(x_ref, out_ref, sf_ref, amax_ref):
    phase, j, i = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    cols = slice(None) if bn == n else pl.ds(pl.multiple_of(j * bn, bn), bn)

    @pl.when(phase == 0)
    def _():
      # max is exact in f32, so the result matches a bf16 reduction.
      tile_amax = jnp.max(
          jnp.abs(x_ref[...].astype(jnp.float32)), axis=0, keepdims=True
      )
      tile_amax = jnp.broadcast_to(tile_amax, (sublanes, bn))

      @pl.when(i == 0)
      def _():
        amax_ref[:, cols] = tile_amax

      @pl.when(i > 0)
      def _():
        amax_ref[:, cols] = jnp.maximum(amax_ref[:, cols], tile_amax)

    @pl.when(phase == 1)
    def _():
      # Same expression and dtypes as the JAX reference, so the scales match
      # bit for bit: amax in the input dtype, divide, then cast.
      amax = amax_ref[0:1, cols].astype(x_ref.dtype)
      sf = (q_max / amax).astype(scale_dtype)

      @pl.when(i == 0)
      def _():
        sf_ref[:, cols] = sf

      scaled = x_ref[...] * jnp.broadcast_to(sf, (bm, bn))
      out_ref[...] = jnp.transpose(scaled, (1, 0)).astype(out_ref.dtype)

  in_pipeline_mode = pl.Buffered(1) if (bm == m and bn == n) else None
  out, sf_row = pl.pallas_call(
      kernel,
      # Pin both outputs to HBM so all HBM writes complete inside the
      # MARKER-scoped kernel call rather than in a post-kernel VMEM->HBM copy.
      out_shape=(
          pltpu.HBM((n, m), out_dtype),
          pltpu.HBM((1, n), scale_dtype),
      ),
      grid=(2, n // bn, m // bm),
      in_specs=[
          pl.BlockSpec(
              (bm, bn),
              lambda p, j, i: (i, j),
              pipeline_mode=in_pipeline_mode,
          )
      ],
      out_specs=(
          # Phase 0 parks on OUT tile (0, 0) without writing it; phase 1 starts
          # on the same tile, so nothing is written back before it is filled.
          pl.BlockSpec((bn, bm), lambda p, j, i: (j * p, i * p)),
          pl.BlockSpec(
              (1, n),
              lambda p, j, i: (0, 0),
              pipeline_mode=pl.Buffered(1),
          ),
      ),
      scratch_shapes=[pltpu.VMEM((sublanes, n), jnp.float32)],
      compiler_params=pltpu.CompilerParams(
          # Phase 1 needs every phase 0 step, so the grid runs in order.
          dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
          vmem_limit_bytes=scoped_vmem_bytes,
      ),
      interpret=interpret,
  )(
      # Keep X in HBM so the kernel's own pipelined DMAs read it. Otherwise the
      # compiler may copy a small X into VMEM before the kernel starts. That
      # copy is not part of the kernel's MARKER-tagged trace event, so the
      # benchmark's xprof timing would leave out the HBM read of X.
      pltpu.with_memory_space_constraint(x, pltpu.HBM)
  )
  return out, sf_row.reshape(n, 1)
