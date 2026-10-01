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


def get_dtype_info(dtype_or_str: str | jnp.dtype) -> jnp.finfo | jnp.iinfo:
  """Returns the jax.numpy finfo or iinfo descriptor for a numeric dtype.

  Args:
    dtype_or_str: Either a string representation (e.g., 'float8_e4m3fn',
      'float4_e2m1fn', 'int4', 'bfloat16') or a `jax.numpy` dtype.

  Returns:
    The corresponding `jnp.finfo` (for floating-point dtypes) or `jnp.iinfo`
    (for integer dtypes) descriptor.

  Raises:
    ValueError: If the dtype is not a floating-point or integer numeric type.
  """
  dtype = (
      parse_dtype(dtype_or_str)
      if isinstance(dtype_or_str, str)
      else jnp.dtype(dtype_or_str)
  )
  if jnp.issubdtype(dtype, jnp.floating):
    return jnp.finfo(dtype)
  if jnp.issubdtype(dtype, jnp.integer):
    return jnp.iinfo(dtype)
  raise ValueError(f"Unsupported non-numeric dtype: '{dtype}'.")


def get_dtype_bytes(dtype_or_str: str | jnp.dtype) -> float:
  """Returns the physical storage size in bytes per element (`bits / 8.0`).

  Unlike `jnp.dtype(dtype).itemsize` (which reports `1` byte for 4-bit container
  dtypes such as `int4` and `float4_e2m1fn`), `get_dtype_info().bits / 8.0`
  returns the exact physical width (`0.5` bytes for 4-bit types).

  Args:
    dtype_or_str: Either a string representation or a `jax.numpy` dtype.

  Returns:
    The physical storage size in bytes per element as a float.
  """
  return float(get_dtype_info(dtype_or_str).bits) / 8.0


def get_dtype_max(dtype_or_str: str | jnp.dtype) -> float:
  """Returns the maximum representable finite value for a numeric dtype."""
  return float(get_dtype_info(dtype_or_str).max)
