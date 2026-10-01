"""Unit tests for utils.py."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.core import utils
import jax.numpy as jnp
import numpy as np


class UtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("bfloat16", "bfloat16", jnp.bfloat16),
      ("float32", "float32", jnp.float32),
      ("float16", "float16", jnp.float16),
      ("float64", "float64", jnp.float64),
      ("int32", "int32", jnp.int32),
      ("int8", "int8", jnp.int8),
      ("int4", "int4", jnp.int4),
      ("float8_e4m3fn", "float8_e4m3fn", jnp.float8_e4m3fn),
      ("float8_e5m2", "float8_e5m2", jnp.float8_e5m2),
      ("float4_e2m1fn", "float4_e2m1fn", jnp.float4_e2m1fn),
  )
  def test_parse_dtype_valid(self, dtype_str, expected_dtype):
    """Verifies that valid dtype strings parse to their expected jnp.dtype."""
    self.assertEqual(utils.parse_dtype(dtype_str), expected_dtype)

  @parameterized.named_parameters(
      ("invalid", "invalid_type"),
      ("empty", ""),
      ("numeric", "123"),
  )
  def test_parse_dtype_invalid(self, dtype_str):
    with self.assertRaisesRegex(
        ValueError, f"Invalid dtype string: '{dtype_str}'"
    ):
      utils.parse_dtype(dtype_str)

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


if __name__ == "__main__":
  absltest.main()
