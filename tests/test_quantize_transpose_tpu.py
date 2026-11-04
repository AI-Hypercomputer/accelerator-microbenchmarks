"""TPU smoke tests for the Pallas `quantize_transpose` kernel."""

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.kernels import quantize_transpose
import jax
import jax.numpy as jnp
import numpy as np


def _reference_quantize_transpose(x, out_dtype, scale_dtype, q_max):
  xt = x.T
  sf = (q_max / jnp.max(jnp.abs(xt), axis=-1, keepdims=True)).astype(
      scale_dtype
  )
  return (xt * sf).astype(out_dtype), sf


class QuantizeTransposeTpuTest(parameterized.TestCase):
  """Tests Mosaic lowering and execution of `quantize_transpose` on TPU."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This test requires TPU devices.")

  @parameterized.named_parameters(
      ("fp8_bf16_sf", jnp.bfloat16, jnp.float8_e4m3fn, jnp.bfloat16, 448.0),
      ("fp4_bf16_sf", jnp.bfloat16, jnp.float4_e2m1fn, jnp.bfloat16, 6.0),
      ("int4_bf16_sf", jnp.bfloat16, jnp.int4, jnp.bfloat16, 7.0),
      ("int4_fp32_sf", jnp.bfloat16, jnp.int4, jnp.float32, 7.0),
      (
          "f32_in_fp8_bf16_sf",
          jnp.float32,
          jnp.float8_e4m3fn,
          jnp.bfloat16,
          448.0,
      ),
  )
  def test_multi_tile_matches_reference_on_tpu(
      self, in_dtype, out_dtype, sf_dtype, q_max
  ):
    """Compiles and runs a multi-tile grid on TPU (`interpret=False`)."""
    x = jax.random.normal(jax.random.PRNGKey(0), (512, 768), in_dtype)
    vmem_kib = 1792 if in_dtype == jnp.float32 else 896
    blocks = quantize_transpose.default_transpose_blocks(
        512,
        768,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scale_dtype=sf_dtype,
        scoped_vmem_bytes=vmem_kib * 1024,
    )
    out, sf = quantize_transpose.quantize_transpose(
        x,
        out_dtype=out_dtype,
        scale_dtype=sf_dtype,
        q_max=q_max,
        block_m=blocks[0],
        block_n=blocks[1],
    )
    ref_out, ref_sf = _reference_quantize_transpose(
        x, out_dtype, sf_dtype, q_max
    )
    self.assertEqual(blocks, (256, 384))
    self.assertEqual(out.shape, (768, 512))
    self.assertEqual(out.dtype, jnp.dtype(out_dtype))
    self.assertEqual(sf.shape, (768, 1))
    self.assertEqual(sf.dtype, jnp.dtype(sf_dtype))
    np.testing.assert_array_equal(
        np.asarray(out, np.float32), np.asarray(ref_out, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(sf, np.float32), np.asarray(ref_sf, np.float32)
    )

  def test_default_budget_single_tile_on_tpu(self):
    """Exercises the single-tile `pl.Buffered(1)` path under the 64 MiB default."""
    x = jax.random.normal(jax.random.PRNGKey(1), (512, 1024), jnp.bfloat16)
    blocks = quantize_transpose.default_transpose_blocks(
        512,
        1024,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.float8_e4m3fn,
        scale_dtype=jnp.bfloat16,
    )
    out, sf = quantize_transpose.quantize_transpose(
        x,
        out_dtype=jnp.float8_e4m3fn,
        scale_dtype=jnp.bfloat16,
        q_max=448.0,
    )
    ref_out, ref_sf = _reference_quantize_transpose(
        x, jnp.float8_e4m3fn, jnp.bfloat16, 448.0
    )
    self.assertEqual(blocks, (512, 1024))
    np.testing.assert_array_equal(
        np.asarray(out, np.float32), np.asarray(ref_out, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(sf, np.float32), np.asarray(ref_sf, np.float32)
    )

  def test_default_budget_multi_tile_on_tpu(self):
    """Exercises default multi-tile selection (> 4 MiB) under the 64 MiB default."""
    x = jax.random.normal(jax.random.PRNGKey(2), (4096, 2048), jnp.bfloat16)
    blocks = quantize_transpose.default_transpose_blocks(
        4096,
        2048,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.float8_e4m3fn,
        scale_dtype=jnp.bfloat16,
    )
    out, sf = quantize_transpose.quantize_transpose(
        x,
        out_dtype=jnp.float8_e4m3fn,
        scale_dtype=jnp.bfloat16,
        q_max=448.0,
    )
    ref_out, ref_sf = _reference_quantize_transpose(
        x, jnp.float8_e4m3fn, jnp.bfloat16, 448.0
    )
    self.assertEqual(blocks, (2048, 2048))
    np.testing.assert_array_equal(
        np.asarray(out, np.float32), np.asarray(ref_out, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(sf, np.float32), np.asarray(ref_sf, np.float32)
    )


if __name__ == "__main__":
  absltest.main()
