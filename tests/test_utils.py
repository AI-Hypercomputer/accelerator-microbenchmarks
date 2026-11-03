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


if __name__ == "__main__":
  absltest.main()
