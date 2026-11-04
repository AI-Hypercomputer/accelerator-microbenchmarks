"""Unit tests for quantize_transpose.py (Pallas interpret mode on CPU)."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.kernels import quantize_transpose
import jax
import jax.numpy as jnp
import numpy as np

# Default dtypes for tile derivation tests: bf16 input, fp8 output, fp32 scales.
_DTYPES = {
    "in_dtype": jnp.bfloat16,
    "out_dtype": jnp.float8_e4m3fn,
    "scale_dtype": jnp.float32,
}


def setUpModule():  # pylint: disable=invalid-name
  jax.config.update("jax_platform_name", "cpu")


def _reference_quantize_transpose(x, out_dtype, scale_dtype, q_max):
  xt = x.T
  sf = (q_max / jnp.max(jnp.abs(xt), axis=-1, keepdims=True)).astype(
      scale_dtype
  )
  return (xt * sf).astype(out_dtype), sf


class DefaultTransposeBlocksTest(parameterized.TestCase):
  """Tests dynamic tile derivation in `default_transpose_blocks`."""

  @parameterized.parameters(
      # Inputs of at most 4 MiB use a single tile.
      (512, 1024, 512, 1024),
      (512, 4096, 512, 4096),
      (256, 7168, 256, 7168),
      (100, 200, 100, 200),
      # Larger inputs pick the largest lane-aligned dividing tile (up to 16 MiB
      # input tile), breaking equal-area ties toward balanced tiles and wider
      # block_n.
      (16384, 4096, 2048, 4096),
      (8192, 16384, 2048, 4096),
      (8192, 7168, 2048, 3584),
      (4096, 1024, 2048, 1024),
      (2304, 1920, 1152, 1920),
  )
  def test_default_blocks(self, m, n, expected_bm, expected_bn):
    """Selects single-tile or balanced lane-aligned multi-tile blocks."""
    self.assertEqual(
        quantize_transpose.default_transpose_blocks(m, n, **_DTYPES),
        (expected_bm, expected_bn),
    )

  @parameterized.parameters(
      # For float32 inputs (4B), input tiles are capped at 16 MiB (4M elements).
      (4096, 4096, jnp.float32, jnp.float8_e4m3fn, 2048, 2048),
      (16384, 4096, jnp.float32, jnp.float8_e4m3fn, 2048, 2048),
      (8192, 7168, jnp.float32, jnp.float8_e4m3fn, 2048, 1792),
      # Wider out_dtype (bfloat16, 2B) reduces usable double-buffered tile size.
      (4096, 4096, jnp.bfloat16, jnp.bfloat16, 2048, 2048),
      # Semi-aligned and unaligned shapes fall back to full dimension per axis.
      (3000, 4096, jnp.bfloat16, jnp.float8_e4m3fn, 3000, 2048),
      (4096, 3000, jnp.bfloat16, jnp.float8_e4m3fn, 2048, 3000),
      (3000, 1000, jnp.bfloat16, jnp.float8_e4m3fn, 3000, 1000),
  )
  def test_default_blocks_dtype_budget(
      self, m, n, in_dtype, out_dtype, expected_bm, expected_bn
  ):
    """Accounts for input/output dtype bitwidths and unaligned fallbacks."""
    self.assertEqual(
        quantize_transpose.default_transpose_blocks(
            m,
            n,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            scale_dtype=jnp.float32,
        ),
        (expected_bm, expected_bn),
    )

  @parameterized.parameters(
      (16384, 4096, 32 << 20, 8, 2048, 2048),
      (16384, 8192, 128 << 20, 16, 2048, 4096),
  )
  def test_default_blocks_scales_with_vmem_and_sublanes(
      self, m, n, scoped_vmem_bytes, num_sublanes, expected_bm, expected_bn
  ):
    """Respects scoped VMEM budget and fixed physical 16 MiB input tile cap."""
    with mock.patch.object(
        quantize_transpose.pltpu.Tiling,
        "shape",
        new_callable=mock.PropertyMock,
        return_value=(num_sublanes, 128),
    ):
      self.assertEqual(
          quantize_transpose.default_transpose_blocks(
              m,
              n,
              **_DTYPES,
              scoped_vmem_bytes=scoped_vmem_bytes,
          ),
          (expected_bm, expected_bn),
      )

  def test_default_blocks_no_fitting_tile_raises(self):
    """Verifies unaligned shape exceeding VMEM budget raises ValueError."""
    with self.assertRaisesRegex(ValueError, "No VMEM-fitting tile divides"):
      quantize_transpose.default_transpose_blocks(6000, 6000, **_DTYPES)

  def test_default_blocks_vmem_exact_boundaries(self):
    """Verifies exact byte boundaries for padded and persistent buffers."""
    # For m=8, n=256 (bf16), 2D row alignment (16 rows for 16-bit) pads X to
    # (16, 256) = 8192B. Single-tile cutoff (vmem // 16 == 8192B) is 131072B.
    self.assertEqual(
        quantize_transpose.default_transpose_blocks(
            8, 256, **_DTYPES, scoped_vmem_bytes=131072
        ),
        (8, 256),
    )
    self.assertEqual(
        quantize_transpose.default_transpose_blocks(
            8, 256, **_DTYPES, scoped_vmem_bytes=131056
        ),
        (8, 128),
    )

    # For m=128, n=128 (bf16 -> f32, sf=bf16):
    #   1 * in_bytes (32768B) + 2 * out_bytes (131072B)
    #   + persistent_bytes (4096B amax + 2048B 1-row sf) = 169984B.
    # Exact usable_vmem == 169984B occurs at scoped_vmem_bytes == 212480B.
    self.assertEqual(
        quantize_transpose.default_transpose_blocks(
            128,
            128,
            in_dtype=jnp.bfloat16,
            out_dtype=jnp.float32,
            scale_dtype=jnp.bfloat16,
            scoped_vmem_bytes=212480,
        ),
        (128, 128),
    )
    with self.assertRaisesRegex(ValueError, "No VMEM-fitting tile divides"):
      quantize_transpose.default_transpose_blocks(
          128,
          128,
          in_dtype=jnp.bfloat16,
          out_dtype=jnp.float32,
          scale_dtype=jnp.bfloat16,
          scoped_vmem_bytes=212479,
      )

  @parameterized.named_parameters(
      ("zero_m", 0, 1024, 64 << 20),
      ("negative_n", 1024, -128, 64 << 20),
      ("zero_vmem", 1024, 1024, 0),
  )
  def test_default_blocks_non_positive_inputs_raise(
      self, m, n, scoped_vmem_bytes
  ):
    """Verifies non-positive shape or VMEM budget raises ValueError."""
    with self.assertRaisesRegex(ValueError, "must be positive"):
      quantize_transpose.default_transpose_blocks(
          m,
          n,
          **_DTYPES,
          scoped_vmem_bytes=scoped_vmem_bytes,
      )


class QuantizeTransposeTest(parameterized.TestCase):
  """Tests Pallas `quantize_transpose` kernel execution."""

  @parameterized.named_parameters(
      ("fp8_bf16_sf", jnp.float8_e4m3fn, jnp.bfloat16, 448.0),
      ("fp4_bf16_sf", jnp.float4_e2m1fn, jnp.bfloat16, 6.0),
      ("int4_fp32_sf", jnp.int4, jnp.float32, 7.0),
  )
  def test_matches_reference_with_multi_tile_grid(
      self, out_dtype, sf_dtype, q_max
  ):
    """Verifies bit-exact output on a grid with several tiles in each dim."""
    x = jax.random.normal(jax.random.PRNGKey(0), (512, 768), jnp.bfloat16)
    # A small 896 KiB scoped VMEM budget makes the derived tile (256, 384),
    # which splits the input into a 2x2 grid.
    blocks = quantize_transpose.default_transpose_blocks(
        512,
        768,
        in_dtype=x.dtype,
        out_dtype=out_dtype,
        scale_dtype=sf_dtype,
        scoped_vmem_bytes=896 * 1024,
    )
    out, sf = quantize_transpose.quantize_transpose(
        x,
        out_dtype=out_dtype,
        scale_dtype=sf_dtype,
        q_max=q_max,
        block_m=blocks[0],
        block_n=blocks[1],
        interpret=True,
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

  def test_matches_reference_with_single_tile(self):
    """Verifies output when the whole input fits in a single tile."""
    x = jax.random.normal(jax.random.PRNGKey(1), (256, 384), jnp.bfloat16)
    out, sf = quantize_transpose.quantize_transpose(
        x,
        out_dtype=jnp.float8_e4m3fn,
        scale_dtype=jnp.bfloat16,
        q_max=448.0,
        interpret=True,
    )
    ref_out, ref_sf = _reference_quantize_transpose(
        x, jnp.float8_e4m3fn, jnp.bfloat16, 448.0
    )
    np.testing.assert_array_equal(
        np.asarray(out, np.float32), np.asarray(ref_out, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(sf, np.float32), np.asarray(ref_sf, np.float32)
    )

  @parameterized.named_parameters(
      ("only_block_m", 128, None, "must both be specified"),
      ("only_block_n", None, 128, "must both be specified"),
      ("non_positive_block_m", 0, 128, "must be positive and divide"),
      ("non_positive_block_n", 128, -128, "must be positive and divide"),
      ("non_dividing_block_m", 100, 128, "must be positive and divide"),
      ("non_dividing_block_n", 128, 100, "must be positive and divide"),
      ("unaligned_block_m", 64, 128, "must be aligned to 128 lanes"),
      ("unaligned_block_n", 128, 64, "must be aligned to 128 lanes"),
  )
  def test_invalid_block_m_or_block_n_raises(
      self, block_m, block_n, expected_error
  ):
    """Verifies invalid or unaligned block_m / block_n raise ValueError."""
    x = jnp.ones((256, 256), dtype=jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, expected_error):
      quantize_transpose.quantize_transpose(
          x,
          out_dtype=jnp.float8_e4m3fn,
          scale_dtype=jnp.bfloat16,
          q_max=448.0,
          block_m=block_m,
          block_n=block_n,
          interpret=True,
      )


if __name__ == "__main__":
  absltest.main()
