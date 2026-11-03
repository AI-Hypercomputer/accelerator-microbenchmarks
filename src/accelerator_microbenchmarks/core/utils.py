"""Utility functions for accelerator microbenchmarks."""

import math
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy import typing as npt


def random_bits_array(
    shape: Sequence[int], dtype: npt.DTypeLike, seed: int | None = None
) -> np.ndarray:
  """Returns a host array of `shape` and `dtype` filled with random bits.

  Generates random uint64 words and reinterprets their bytes as `dtype`. This
  avoids the float64 temporary and the slow elementwise cast (e.g. for
  ml_dtypes such as bfloat16) of `np.random.normal(...).astype(dtype)`, keeping
  peak host memory at roughly the payload size.

  The values are arbitrary bit patterns (floating-point dtypes may contain
  NaN/Inf, and `bool` may hold bytes other than 0/1), so only use this where
  the contents are opaque, e.g. host<->device transfer payloads.

  Args:
    shape: Output shape.
    dtype: Output dtype (numpy or ml_dtypes, e.g. jnp.bfloat16).
    seed: Seed for `np.random.default_rng`. None for nondeterministic output.

  Returns:
    A C-contiguous array with the given shape and dtype.
  """
  dtype = np.dtype(dtype)
  nbytes = math.prod(shape) * dtype.itemsize
  rng = np.random.default_rng(seed)
  words = rng.integers(
      0,
      np.iinfo(np.uint64).max,
      size=-(-nbytes // 8),
      dtype=np.uint64,
      endpoint=True,
  )
  return words.view(np.uint8)[:nbytes].view(dtype).reshape(shape)


def random_normal_array(
    key: jax.Array, shape: Sequence[int], dtype: jax.typing.DTypeLike
) -> jax.Array:
  """Returns `jax.random.normal` samples of `shape` cast to `dtype`.

  Unlike `jax.random.normal`, also supports integer, boolean and sub-byte
  dtypes by sampling in float32 and casting (integers truncate toward zero and
  booleans are almost all True), so only use this where the contents are
  opaque, e.g. bandwidth benchmark payloads. Traceable, so it can run inside
  `jax.jit` with `out_shardings` to generate inputs directly on device.

  Args:
    key: PRNG key.
    shape: Output shape.
    dtype: Output dtype.

  Returns:
    An array with the given shape and dtype.
  """
  # jax.random.normal only supports >= 8-bit inexact dtypes.
  if jnp.issubdtype(dtype, jnp.inexact) and jnp.finfo(dtype).bits >= 8:
    return jax.random.normal(key, shape, dtype=dtype)
  return jax.random.normal(key, shape, dtype=jnp.float32).astype(dtype)
