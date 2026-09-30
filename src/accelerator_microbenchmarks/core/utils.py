"""Utility functions for accelerator microbenchmarks."""

import math
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from numpy import typing as npt


def parse_dtype(dtype_str: str) -> jnp.dtype:
  """Parses a string into a jax.numpy dtype.

  Args:
    dtype_str: The string representation of the dtype (e.g., 'bfloat16').

  Returns:
    The corresponding jax.numpy dtype.

  Raises:
    ValueError: If the dtype string does not correspond to a valid jnp dtype.
  """
  if not hasattr(jnp, dtype_str):
    raise ValueError(
        f"Invalid dtype string: '{dtype_str}'. Could not find"
        f" jax.numpy.{dtype_str}."
    )
  return getattr(jnp, dtype_str)


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
