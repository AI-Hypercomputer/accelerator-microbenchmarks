"""Curated dtype enum and dtype helpers for benchmarks."""

from typing import Any, Self

from accelerator_microbenchmarks.core import constants
import jax
import jax.numpy as jnp
import numpy as np


class DType(constants.ParamEnum):
  """Dtype names accepted by `base.dtype_field()` benchmark parameters."""

  BOOL = "bool"
  INT4 = "int4"
  INT8 = "int8"
  INT16 = "int16"
  INT32 = "int32"
  INT64 = "int64"
  UINT8 = "uint8"
  BFLOAT16 = "bfloat16"
  FLOAT16 = "float16"
  FLOAT32 = "float32"
  FLOAT64 = "float64"
  FLOAT8_E4M3FN = "float8_e4m3fn"
  FLOAT8_E5M2 = "float8_e5m2"
  FLOAT4_E2M1FN = "float4_e2m1fn"
  COMPLEX64 = "complex64"

  @property
  def jnp_dtype(self) -> np.dtype:
    """Returns the NumPy/JAX dtype corresponding to this enum member."""
    return np.dtype(getattr(jnp, self.value))

  @classmethod
  def from_value(cls, value: Any) -> Self | None:
    """Returns the member whose value is `value`, or None if there is none.

    Unlike `DType(value)`, this does not raise: unknown names (e.g. 'bf16') and
    non-str values (e.g. None or `jnp.float32`) return None.

    Args:
      value: A `DType` member or its value (e.g. 'bfloat16').
    """
    if not isinstance(value, str):
      return None
    try:
      return cls(value)
    except ValueError:
      return None


def dtypes_matching(*constraints: jax.typing.DTypeLike) -> tuple[DType, ...]:
  """Returns members that are sub-dtypes of any constraint (all if none).

  Args:
    *constraints: Optional dtype-likes to match against via `jnp.issubdtype`,
      e.g. jnp scalar types (`jnp.bfloat16`), abstract dtype classes
      (`jnp.floating`), dtype names (`"float32"`) or `np.dtype`s. Python
      builtins match only the dtype NumPy maps them to (e.g. `float` matches
      float64, not every floating dtype).

  Returns:
    Matching `DType` members in `DType` declaration order, or all `DType`
    members when `constraints` is empty.

  Raises:
    TypeError: If `jnp.issubdtype` cannot interpret a constraint (e.g.
      `"floating"`).
  """
  if not constraints:
    return tuple(DType)
  return tuple(
      member
      for member in DType
      if any(
          jnp.issubdtype(member.jnp_dtype, constraint)
          for constraint in constraints
      )
  )


def parse_dtype(dtype_str: str) -> np.dtype:
  """Parses a `DType` name into the dtype JAX materializes arrays with.

  This is the single dtype parser for all benchmarks, so it only returns dtypes
  that JAX uses as-is: values outside `DType`, and dtypes that JAX would
  canonicalize to a different dtype (e.g. 'float64' while `jax_enable_x64` is
  off, which JAX silently truncates to float32), are rejected instead of being
  benchmarked under the wrong byte count.

  Args:
    dtype_str: The string representation of the dtype (e.g., 'bfloat16').

  Returns:
    The corresponding dtype (e.g. `np.dtype(jnp.bfloat16)`).

  Raises:
    ValueError: If `dtype_str` is not a `DType` value (including non-str inputs
      such as None and JAX dtypes outside `DType` such as 'uint16'), or JAX
      would canonicalize it to a different dtype (e.g. a 64-bit dtype while
      `jax_enable_x64` is off).
  """
  member = DType.from_value(dtype_str)
  if member is None:
    supported = ", ".join(DType)
    raise ValueError(
        f"Invalid dtype {dtype_str!r}. Supported dtypes: {supported}"
    )
  dtype = member.jnp_dtype
  canonical = jax.dtypes.canonicalize_dtype(dtype)
  if canonical != dtype:
    if jax.config.jax_enable_x64:
      raise ValueError(
          f"Unexpected: JAX canonicalizes dtype '{member.value}' to"
          f" '{canonical}' even though jax_enable_x64 is on."
      )
    raise ValueError(
        f"JAX canonicalizes dtype '{member.value}' to '{canonical}' because"
        " jax_enable_x64 is off; set JAX_ENABLE_X64=1 to benchmark real"
        " 64-bit tensors."
    )
  return dtype


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
