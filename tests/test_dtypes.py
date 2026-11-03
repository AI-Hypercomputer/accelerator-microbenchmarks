"""Unit tests for core/dtypes.py."""

import re

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.core import dtypes
import jax
import jax.numpy as jnp
import numpy as np

_ALL_DTYPE_NAMES = ", ".join(dtypes.DType)


class DTypesTest(parameterized.TestCase):
  """Tests for the DType enum, dtypes_matching and the dtype helpers."""

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

  @parameterized.named_parameters(
      ("bfloat16", "bfloat16", jnp.bfloat16),
      ("bfloat16_member", dtypes.DType.BFLOAT16, jnp.bfloat16),
      ("float32", "float32", jnp.float32),
      ("float16", "float16", jnp.float16),
      ("int32", "int32", jnp.int32),
      ("int16", "int16", jnp.int16),
      ("int8", "int8", jnp.int8),
      ("uint8", "uint8", jnp.uint8),
      ("int4", "int4", jnp.int4),
      ("float8_e4m3fn", "float8_e4m3fn", jnp.float8_e4m3fn),
      ("float8_e5m2", "float8_e5m2", jnp.float8_e5m2),
      ("float4_e2m1fn", "float4_e2m1fn", jnp.float4_e2m1fn),
      ("bool", "bool", jnp.bool_),
      ("complex64", "complex64", jnp.complex64),
  )
  def test_parse_dtype_valid(self, dtype_str, expected_dtype):
    """Verifies that valid DType values parse to their expected np.dtype."""
    dtype = dtypes.parse_dtype(dtype_str)
    self.assertIsInstance(dtype, np.dtype)
    self.assertEqual(dtype, np.dtype(expected_dtype))

  @parameterized.named_parameters(
      ("invalid", "invalid_type"),
      ("alias", "bf16"),
      ("uppercase", "BFLOAT16"),
      ("empty", ""),
      ("numeric", "123"),
      ("jnp_non_dtype_ones", "ones"),
      ("jnp_non_dtype_linalg", "linalg"),
      ("jax_dtype_outside_enum_uint16", "uint16"),
      ("jax_dtype_outside_enum_uint64", "uint64"),
      ("jax_dtype_outside_enum_complex128", "complex128"),
      ("none", None),
      ("int", 1),
      ("np_dtype", np.dtype("float32")),
      ("jnp_scalar_type", jnp.float32),
  )
  def test_parse_dtype_invalid(self, dtype_value):
    """Verifies non-DType values (including non-str and non-enum JAX dtypes) raise ValueError."""
    expected_msg = (
        rf"^Invalid dtype {re.escape(repr(dtype_value))}\. Supported dtypes:"
        rf" {_ALL_DTYPE_NAMES}$"
    )
    # Enable x64 as well so 64-bit JAX dtypes outside DType (uint64, complex128)
    # are verified to fail at the DType lookup rather than the x64 guard.
    with jax.enable_x64(True), self.assertRaisesRegex(ValueError, expected_msg):
      dtypes.parse_dtype(dtype_value)

  @parameterized.named_parameters(
      ("float64", "float64", "float32"),
      ("int64", "int64", "int32"),
  )
  def test_parse_dtype_64_bit_requires_x64(self, dtype_str, canonical_str):
    """Verifies 64-bit DType members are rejected unless jax_enable_x64 is set."""
    with jax.enable_x64(False), self.assertRaisesRegex(
        ValueError,
        f"JAX canonicalizes dtype '{dtype_str}' to '{canonical_str}' because"
        " jax_enable_x64 is off",
    ):
      dtypes.parse_dtype(dtype_str)
    with jax.enable_x64(True):
      self.assertEqual(dtypes.parse_dtype(dtype_str), np.dtype(dtype_str))

  @parameterized.named_parameters(
      ("float32_str", "float32", 4.0, float(jnp.finfo(jnp.float32).max)),
      ("bfloat16_str", "bfloat16", 2.0, float(jnp.finfo(jnp.bfloat16).max)),
      ("float8_e4m3fn_str", "float8_e4m3fn", 1.0, 448.0),
      ("float8_e5m2_str", "float8_e5m2", 1.0, 57344.0),
      ("int8_str", "int8", 1.0, 127.0),
      ("float4_e2m1fn_str", "float4_e2m1fn", 0.5, 6.0),
      ("int4_str", "int4", 0.5, 7.0),
  )
  def test_get_dtype_bytes_and_max(
      self, dtype_or_str, expected_bytes, expected_max
  ):
    """Verifies bit-width byte calculation and max representable value via finfo/iinfo."""
    self.assertAlmostEqual(dtypes.get_dtype_bytes(dtype_or_str), expected_bytes)
    self.assertAlmostEqual(
        dtypes.get_dtype_max(dtype_or_str), expected_max, places=1
    )

  @parameterized.named_parameters(
      ("float32_str", "float32", 32),
      ("bfloat16_dtype", jnp.bfloat16, 16),
      ("float8_e4m3fn_str", "float8_e4m3fn", 8),
      ("int8_dtype", jnp.int8, 8),
      # Sub-byte dtypes have itemsize 1 but are 4 bits wide.
      ("int4_str", "int4", 4),
      ("float4_e2m1fn_dtype", jnp.float4_e2m1fn, 4),
  )
  def test_get_dtype_bits(self, dtype_or_str, expected_bits):
    """Verifies the physical bit width, including sub-byte dtypes."""
    bits = dtypes.get_dtype_bits(dtype_or_str)
    self.assertIsInstance(bits, int)
    self.assertEqual(bits, expected_bits)

  @parameterized.named_parameters(
      ("bool", jnp.bool_),
      ("complex64", jnp.complex64),
  )
  def test_get_dtype_bits_rejects_non_numeric_dtypes(self, dtype):
    """Verifies non-numeric dtypes raise ValueError."""
    with self.assertRaisesRegex(ValueError, "Unsupported non-numeric dtype"):
      dtypes.get_dtype_bits(dtype)

  @parameterized.named_parameters(
      ("int4_str", "int4", True),
      ("uint4_dtype", jnp.uint4, True),
      ("float4_e2m1fn_str", "float4_e2m1fn", True),
      ("int8_str", "int8", False),
      ("float8_e4m3fn_dtype", jnp.float8_e4m3fn, False),
      ("bfloat16_str", "bfloat16", False),
      ("bool_str", "bool", False),
      ("complex64_dtype", jnp.complex64, False),
  )
  def test_is_sub_byte_dtype(self, dtype_or_str, expected):
    """Verifies only < 8-bit integer and floating-point dtypes are sub-byte."""
    self.assertEqual(dtypes.is_sub_byte_dtype(dtype_or_str), expected)


if __name__ == "__main__":
  absltest.main()
