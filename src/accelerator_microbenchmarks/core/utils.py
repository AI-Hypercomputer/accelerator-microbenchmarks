"""Utility functions for accelerator microbenchmarks."""

import math
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy import typing as npt


def parse_dtype(dtype_str: str) -> np.dtype:
  """Parses a dtype name into the dtype JAX materializes arrays with.

  This is the single dtype parser for all benchmarks, so it only returns dtypes
  that JAX uses as-is: names that are not `jax.numpy` dtypes, and dtypes that
  JAX would canonicalize to a different dtype (e.g. 'float64' while
  `jax_enable_x64` is off, which JAX silently truncates to float32), are
  rejected instead of being benchmarked under the wrong byte count.

  Args:
    dtype_str: The string representation of the dtype (e.g., 'bfloat16').

  Returns:
    The corresponding dtype (e.g. `np.dtype(jnp.bfloat16)`).

  Raises:
    ValueError: If `dtype_str` is not a string (e.g. None), is not a
      `jax.numpy` dtype (including non-dtype attributes such as 'ones'), or JAX
      would canonicalize it to a different dtype (e.g. a 64-bit dtype while
      `jax_enable_x64` is off).
  """
  if not isinstance(dtype_str, str):
    raise ValueError(
        f"Invalid dtype: {dtype_str!r}. Expected a dtype name string."
    )
  attr = getattr(jnp, dtype_str, None)
  if attr is None:
    raise ValueError(
        f"Invalid dtype string: '{dtype_str}'. Could not find"
        f" jax.numpy.{dtype_str}."
    )
  try:
    dtype = jnp.dtype(attr)
  except TypeError as exc:
    raise ValueError(
        f"Invalid dtype string: '{dtype_str}'. jax.numpy.{dtype_str} is not a"
        " dtype."
    ) from exc
  canonical = jax.dtypes.canonicalize_dtype(dtype)
  if canonical != dtype:
    if jax.config.jax_enable_x64:
      raise ValueError(
          f"Unexpected: JAX canonicalizes dtype '{dtype_str}' to"
          f" '{canonical}' even though jax_enable_x64 is on."
      )
    raise ValueError(
        f"JAX canonicalizes dtype '{dtype_str}' to '{canonical}' because"
        " jax_enable_x64 is off; set JAX_ENABLE_X64=1 to benchmark real"
        " 64-bit tensors."
    )
  return dtype


def validate_dtype(
    dtype_str: str,
    supported_dtypes: Sequence[jax.typing.DTypeLike] | None = None,
) -> None:
  """Validates that a dtype string can be benchmarked as requested.

  Args:
    dtype_str: The string representation of the dtype (e.g., 'bfloat16').
    supported_dtypes: Optional jnp scalar types (e.g. `jnp.bfloat16`) or
      abstract dtype classes (e.g. `jnp.floating`) the dtype must be a
      sub-dtype of.

  Raises:
    ValueError: If `parse_dtype` rejects `dtype_str`, or the dtype is not in
      `supported_dtypes`.
    TypeError: If a `supported_dtypes` entry is not a class (e.g. an
      `np.dtype` instance).
  """
  dtype = parse_dtype(dtype_str)
  if supported_dtypes is not None:
    supported_names = []
    for supported in supported_dtypes:
      # jnp scalar types (e.g. `jnp.bfloat16`) are classes at runtime, even
      # though JAX's type stubs declare them as a Protocol.
      if not isinstance(supported, type):
        raise TypeError(
            "supported_dtypes entries must be jnp scalar types or abstract"
            f" dtype classes, got {supported!r}."
        )
      supported_names.append(supported.__name__)
    if not any(
        jnp.issubdtype(dtype, supported) for supported in supported_dtypes
    ):
      raise ValueError(
          f"Unsupported dtype: '{dtype_str}'. Supported dtypes:"
          f" {supported_names}."
      )


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


def get_dtype_bits(dtype_or_str: str | jnp.dtype) -> int:
  """Returns the physical element width in bits of a numeric dtype.

  Unlike `jnp.dtype(dtype).itemsize * 8` (which reports `8` bits for 4-bit
  container dtypes such as `int4` and `float4_e2m1fn`), this returns the exact
  physical width (`4` bits for 4-bit types).

  Args:
    dtype_or_str: Either a string representation or a `jax.numpy` dtype.

  Returns:
    The number of bits per element.

  Raises:
    ValueError: If the dtype is not a floating-point or integer numeric type.
  """
  return int(get_dtype_info(dtype_or_str).bits)


def is_sub_byte_dtype(dtype_or_str: str | jnp.dtype) -> bool:
  """Returns whether a numeric dtype is narrower than one byte (e.g. `int4`).

  NumPy reports `itemsize == 1` for these dtypes, but TPUs store them packed
  (e.g. two `int4` elements per byte), so byte counts based on `itemsize` are
  too large.

  Args:
    dtype_or_str: Either a string representation or a `jax.numpy` dtype.

  Returns:
    True for integer or floating-point dtypes narrower than 8 bits; False
    otherwise (including `bool` and complex dtypes).
  """
  dtype = (
      parse_dtype(dtype_or_str)
      if isinstance(dtype_or_str, str)
      else jnp.dtype(dtype_or_str)
  )
  if not (
      jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.floating)
  ):
    return False
  return get_dtype_bits(dtype) < 8


def get_dtype_bytes(dtype_or_str: str | jnp.dtype) -> float:
  """Returns the physical storage size in bytes per element (`bits / 8.0`).

  Unlike `jnp.dtype(dtype).itemsize` (which reports `1` byte for 4-bit container
  dtypes such as `int4` and `float4_e2m1fn`), `get_dtype_bits() / 8.0` returns
  the exact physical width (`0.5` bytes for 4-bit types).

  Args:
    dtype_or_str: Either a string representation or a `jax.numpy` dtype.

  Returns:
    The physical storage size in bytes per element as a float.
  """
  return get_dtype_bits(dtype_or_str) / 8.0


def get_dtype_max(dtype_or_str: str | jnp.dtype) -> float:
  """Returns the maximum representable finite value for a numeric dtype."""
  return float(get_dtype_info(dtype_or_str).max)
