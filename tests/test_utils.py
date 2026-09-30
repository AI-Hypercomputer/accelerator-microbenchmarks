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
      ("float8_e4m3fn", "float8_e4m3fn", jnp.float8_e4m3fn),
      ("float8_e5m2", "float8_e5m2", jnp.float8_e5m2),
  )
  def test_parse_dtype_valid(self, dtype_str, expected_dtype):
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


if __name__ == "__main__":
  absltest.main()
