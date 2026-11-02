"""Curated dtype enum for benchmark configuration parameters."""

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
