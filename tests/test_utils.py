"""Unit tests for utils.py."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp
import numpy as np


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("bfloat16", "bfloat16", jnp.bfloat16),
      ("float32", "float32", jnp.float32),
      ("float16", "float16", jnp.float16),
      ("int32", "int32", jnp.int32),
      ("int8", "int8", jnp.int8),
      ("int4", "int4", jnp.int4),
      ("float8_e4m3fn", "float8_e4m3fn", jnp.float8_e4m3fn),
      ("float8_e5m2", "float8_e5m2", jnp.float8_e5m2),
      ("float4_e2m1fn", "float4_e2m1fn", jnp.float4_e2m1fn),
      ("bool", "bool", jnp.bool_),
      ("complex64", "complex64", jnp.complex64),
  )
  def test_parse_dtype_valid(self, dtype_str, expected_dtype):
    """Verifies that valid dtype strings parse to their expected np.dtype."""
    dtype = utils.parse_dtype(dtype_str)
    self.assertIsInstance(dtype, np.dtype)
    self.assertEqual(dtype, np.dtype(expected_dtype))

  @parameterized.named_parameters(
      ("invalid", "invalid_type"),
      ("empty", ""),
      ("numeric", "123"),
  )
  def test_parse_dtype_invalid(self, dtype_str):
    with self.assertRaisesRegex(
        ValueError, f"Invalid dtype string: '{dtype_str}'. Could not find"
    ):
      utils.parse_dtype(dtype_str)

  @parameterized.named_parameters(
      ("none", None),
      ("int", 1),
      ("np_dtype", np.dtype("float32")),
  )
  def test_parse_dtype_rejects_non_string(self, dtype_value):
    """Verifies non-string values raise ValueError, not TypeError."""
    with self.assertRaisesRegex(ValueError, "Expected a dtype name string"):
      utils.parse_dtype(dtype_value)

  @parameterized.parameters("ones", "array", "linalg", "pi")
  def test_parse_dtype_rejects_non_dtype_jnp_attribute(self, dtype_str):
    """Verifies non-dtype jnp attributes raise ValueError, not TypeError."""
    with self.assertRaisesRegex(
        ValueError,
        f"Invalid dtype string: '{dtype_str}'. jax.numpy.{dtype_str} is not a"
        " dtype.",
    ):
      utils.parse_dtype(dtype_str)

  @parameterized.named_parameters(
      ("float64", "float64", "float32"),
      ("int64", "int64", "int32"),
      ("uint64", "uint64", "uint32"),
      ("complex128", "complex128", "complex64"),
  )
  def test_parse_dtype_64_bit_requires_x64(self, dtype_str, canonical_str):
    """Verifies 64-bit dtypes are rejected unless jax_enable_x64 is set."""
    with jax.enable_x64(False), self.assertRaisesRegex(
        ValueError,
        f"JAX canonicalizes dtype '{dtype_str}' to '{canonical_str}' because"
        " jax_enable_x64 is off",
    ):
      utils.parse_dtype(dtype_str)
    with jax.enable_x64(True):
      self.assertEqual(utils.parse_dtype(dtype_str), np.dtype(dtype_str))

  @parameterized.named_parameters(
      ("bfloat16", (64, 128), jnp.bfloat16),
      ("float32", (64, 128), jnp.float32),
      ("float16", (64, 128), jnp.float16),
      ("int32", (64, 128), jnp.int32),
      ("int8", (64, 128), jnp.int8),
      ("float8_e4m3fn", (64, 128), jnp.float8_e4m3fn),
      ("float8_e5m2", (64, 128), jnp.float8_e5m2),
      # 3 * 5 * 2 = 30 bytes, not a multiple of the 8-byte uint64 word size;
      # exercises the ceil-division and trailing-byte truncation.
      ("bfloat16_nbytes_not_multiple_of_8", (3, 5), jnp.bfloat16),
  )
  def test_random_bits_array_shape_dtype_nbytes(self, shape, dtype):
    arr = utils.random_bits_array(shape, dtype)
    self.assertIsInstance(arr, np.ndarray)
    self.assertEqual(arr.shape, shape)
    self.assertEqual(arr.dtype, np.dtype(dtype))
    self.assertEqual(arr.nbytes, math.prod(shape) * np.dtype(dtype).itemsize)
    self.assertTrue(arr.flags.c_contiguous)

  def test_random_bits_array_seed_determinism(self):
    # Compare raw bytes so NaN bit patterns do not affect equality.
    a = utils.random_bits_array((32, 128), jnp.bfloat16, seed=7)
    b = utils.random_bits_array((32, 128), jnp.bfloat16, seed=7)
    c = utils.random_bits_array((32, 128), jnp.bfloat16, seed=8)
    np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
    self.assertFalse(np.array_equal(a.view(np.uint8), c.view(np.uint8)))

  def test_random_bits_array_default_seed_is_nondeterministic(self):
    a = utils.random_bits_array((32, 128), jnp.int32)
    b = utils.random_bits_array((32, 128), jnp.int32)
    self.assertFalse(np.array_equal(a, b))

  @parameterized.parameters(
      jnp.bfloat16,
      jnp.float8_e4m3fn,
      jnp.float4_e2m1fn,
      jnp.complex64,
      jnp.int4,
      jnp.int8,
      jnp.uint8,
      jnp.int32,
      jnp.bool_,
  )
  def test_random_normal_array_shape_dtype(self, dtype):
    """Verifies float, complex, integer and bool dtypes under `jax.jit`."""
    arr = jax.jit(lambda k: utils.random_normal_array(k, (64, 128), dtype))(
        jax.random.PRNGKey(0)
    )
    self.assertEqual(arr.shape, (64, 128))
    self.assertEqual(arr.dtype, dtype)

  def test_random_normal_array_float_matches_jax_random_normal(self):
    """Verifies floating-point dtypes keep `jax.random.normal` values."""
    key = jax.random.PRNGKey(0)
    np.testing.assert_array_equal(
        utils.random_normal_array(key, (64, 128), jnp.bfloat16),
        jax.random.normal(key, (64, 128), dtype=jnp.bfloat16),
    )

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
    self.assertAlmostEqual(utils.get_dtype_bytes(dtype_or_str), expected_bytes)
    self.assertAlmostEqual(
        utils.get_dtype_max(dtype_or_str), expected_max, places=1
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
    bits = utils.get_dtype_bits(dtype_or_str)
    self.assertIsInstance(bits, int)
    self.assertEqual(bits, expected_bits)

  @parameterized.named_parameters(
      ("bool", jnp.bool_),
      ("complex64", jnp.complex64),
  )
  def test_get_dtype_bits_rejects_non_numeric_dtypes(self, dtype):
    """Verifies non-numeric dtypes raise ValueError."""
    with self.assertRaisesRegex(ValueError, "Unsupported non-numeric dtype"):
      utils.get_dtype_bits(dtype)

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
    self.assertEqual(utils.is_sub_byte_dtype(dtype_or_str), expected)

  @parameterized.parameters(
      ("float64", "JAX canonicalizes dtype 'float64'"),
      ("ones", "Invalid dtype string: 'ones'"),
  )
  def test_validate_dtype_propagates_parse_dtype_errors(
      self, dtype_str, error_regex
  ):
    """Verifies validate_dtype rejects whatever parse_dtype rejects."""
    with jax.enable_x64(False), self.assertRaisesRegex(
        ValueError, error_regex
    ):
      utils.validate_dtype(dtype_str, (jnp.floating,))

  @parameterized.parameters(
      "bfloat16", "bool", "int4", "float8_e4m3fn", "complex64"
  )
  def test_validate_dtype_accepts_non_64_bit(self, dtype_str):
    """Verifies <= 32-bit and complex64 (2 x 32-bit) dtypes are accepted."""
    utils.validate_dtype(dtype_str)

  @parameterized.named_parameters(
      ("concrete", "float32", (jnp.bfloat16, jnp.float32)),
      ("abstract", "float8_e5m2", (jnp.floating,)),
  )
  def test_validate_dtype_supported_dtypes_accepted(
      self, dtype_str, supported_dtypes
  ):
    """Verifies `supported_dtypes` accepts concrete and abstract dtypes."""
    utils.validate_dtype(dtype_str, supported_dtypes)

  @parameterized.named_parameters(
      (
          "concrete",
          "float16",
          (jnp.bfloat16, jnp.float32),
          (
              r"Unsupported dtype: 'float16'. Supported dtypes: \['bfloat16',"
              r" 'float32'\]"
          ),
      ),
      ("abstract", "complex64", (jnp.floating,), "Unsupported dtype"),
  )
  def test_validate_dtype_supported_dtypes_rejected(
      self, dtype_str, supported_dtypes, error_regex
  ):
    """Verifies dtypes outside `supported_dtypes` are rejected."""
    with self.assertRaisesRegex(ValueError, error_regex):
      utils.validate_dtype(dtype_str, supported_dtypes)


if __name__ == "__main__":
  absltest.main()
