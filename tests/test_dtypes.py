"""Unit tests for core/dtypes.py."""

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.core import dtypes
import jax.numpy as jnp
import numpy as np


class DTypesTest(parameterized.TestCase):
  """Tests for the DType enum and dtypes_matching helper."""

  def test_every_member_is_a_jax_dtype(self):
    """Verifies every DType member maps to a JAX dtype with matching name."""
    # Single change-detector for DType membership and declaration order; other
    # test files derive their expected dtype lists from dtypes.DType.
    self.assertEqual(
        [member.value for member in dtypes.DType],
        [
            "bool",
            "int4",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "float8_e4m3fn",
            "float8_e5m2",
            "float4_e2m1fn",
            "complex64",
        ],
    )
    for member in dtypes.DType:
      with self.subTest(dtype=member.value):
        self.assertEqual(member.jnp_dtype.name, member.value)

  @parameterized.named_parameters(
      ("value", "bfloat16", dtypes.DType.BFLOAT16),
      ("member", dtypes.DType.FLOAT32, dtypes.DType.FLOAT32),
      ("unknown_name", "bf16", None),
      ("jax_dtype_outside_enum", "uint16", None),
      ("none", None, None),
      ("int", 1, None),
      ("jnp_scalar_type", jnp.float32, None),
      ("np_dtype", np.dtype("float32"), None),
  )
  def test_from_value(self, value, expected):
    """Verifies from_value returns the matching member or None."""
    self.assertIs(dtypes.DType.from_value(value), expected)

  def test_dtypes_matching(self):
    """Verifies dtypes_matching filters members in DType declaration order."""
    self.assertEqual(dtypes.dtypes_matching(), tuple(dtypes.DType))
    self.assertEqual(
        dtypes.dtypes_matching(jnp.floating),
        (
            dtypes.DType.BFLOAT16,
            dtypes.DType.FLOAT16,
            dtypes.DType.FLOAT32,
            dtypes.DType.FLOAT64,
            dtypes.DType.FLOAT8_E4M3FN,
            dtypes.DType.FLOAT8_E5M2,
            dtypes.DType.FLOAT4_E2M1FN,
        ),
    )
    self.assertEqual(
        dtypes.dtypes_matching(jnp.integer),
        (
            dtypes.DType.INT4,
            dtypes.DType.INT8,
            dtypes.DType.INT16,
            dtypes.DType.INT32,
            dtypes.DType.INT64,
            dtypes.DType.UINT8,
        ),
    )
    self.assertEqual(
        dtypes.dtypes_matching(jnp.bfloat16, jnp.float32),
        (dtypes.DType.BFLOAT16, dtypes.DType.FLOAT32),
    )
    # Order follows DType declaration order without duplicates.
    self.assertEqual(
        dtypes.dtypes_matching(jnp.float32, jnp.bfloat16, jnp.bfloat16),
        (dtypes.DType.BFLOAT16, dtypes.DType.FLOAT32),
    )
    self.assertEqual(dtypes.dtypes_matching(jnp.uint16), ())

  @parameterized.named_parameters(
      ("jnp_scalar_type", jnp.float32, (dtypes.DType.FLOAT32,)),
      ("np_scalar_type", np.float32, (dtypes.DType.FLOAT32,)),
      ("name", "float32", (dtypes.DType.FLOAT32,)),
      ("np_dtype", np.dtype("float32"), (dtypes.DType.FLOAT32,)),
  )
  def test_dtypes_matching_accepts_dtype_likes(self, constraint, expected):
    """Verifies the dtype-like forms jnp.issubdtype resolves are accepted."""
    self.assertEqual(dtypes.dtypes_matching(constraint), expected)

  def test_dtypes_matching_rejects_non_dtype_like(self):
    """Verifies a constraint that is not a dtype-like raises TypeError."""
    with self.assertRaises(TypeError):
      dtypes.dtypes_matching("floating")


if __name__ == "__main__":
  absltest.main()
