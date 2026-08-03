"""Unit tests for utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.core import utils
import jax.numpy as jnp


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


if __name__ == "__main__":
  absltest.main()
