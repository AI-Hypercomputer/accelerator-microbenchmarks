"""Unit tests for kernels/quantized_matmul.py (Pallas interpret mode on CPU)."""

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.kernels import quantized_matmul
import jax
import jax.numpy as jnp
import numpy as np


def setUpModule():  # pylint: disable=invalid-name
  jax.config.update("jax_platform_name", "cpu")


def _reference_quantized_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    sc_size: int,
    quant_dtype: jax.typing.DTypeLike = jnp.float8_e4m3fn,
    rhs_quant_dtype: jax.typing.DTypeLike | None = None,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    acc_dtype: jax.typing.DTypeLike = jnp.float32,
    dequantize: bool = True,
    scaling: str = "dynamic",
) -> jax.Array:
  """Pure-JAX sub-channel / row-wise quantized matmul reference."""
  m, k = lhs.shape
  _, n = rhs.shape
  q_dtype = jnp.dtype(quant_dtype)
  rhs_q_dtype = (
      q_dtype if rhs_quant_dtype is None else jnp.dtype(rhs_quant_dtype)
  )
  a_dtype = jnp.dtype(acc_dtype)
  dot_acc_dtype = (
      jnp.int32
      if (
          jnp.issubdtype(q_dtype, jnp.integer)
          and jnp.issubdtype(rhs_q_dtype, jnp.integer)
      )
      else jnp.float32
  )
  qa, sf_a = quantized_matmul.quantize_subchannel(
      lhs,
      contract_axis=1,
      sc_size=sc_size,
      quant_dtype=q_dtype,
      scaling=scaling,
  )
  qb, sf_b = quantized_matmul.quantize_subchannel(
      rhs,
      contract_axis=0,
      sc_size=sc_size,
      quant_dtype=rhs_q_dtype,
      scaling=scaling,
  )
  acc = jnp.zeros((m, n), dtype=a_dtype)
  for b_idx in range(k // sc_size):
    k0, k1 = b_idx * sc_size, (b_idx + 1) * sc_size
    part = jax.lax.dot_general(
        qa[:, k0:k1],
        qb[k0:k1, :],
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=dot_acc_dtype,
    ).astype(a_dtype)
    if dequantize:
      s_a = sf_a[:, b_idx, :].astype(a_dtype)
      s_b = sf_b[b_idx, :, :].astype(a_dtype)
      part = part * s_a * s_b
    acc = acc + part
  return acc.astype(jnp.dtype(out_dtype))


class DefaultQuantizedMatmulBlocksTest(parameterized.TestCase):
  """Tests tile derivation in `default_quantized_matmul_blocks`."""

  @parameterized.parameters(
      # Sub-channel B512 and B128 (on-the-fly vs. weight-offline).
      (16384, 16384, 4096, 512, True, (512, 1024, 4096, 2)),
      (16384, 16384, 4096, 512, False, (512, 2048, 4096, 2)),
      (4096, 16384, 16384, 128, True, (512, 512, 4096, 2)),
      (4096, 16384, 16384, 128, False, (512, 1024, 4096, 2)),
      (4096, 1024, 16384, 512, False, (512, 1024, 4096, 4)),
      # Full-row scaling (sc_size == k), including k > max_tile_k.
      (1024, 1024, 4096, 1024, True, (512, 1024, 4096, 2)),
      (4096, 16384, 4096, 16384, True, (512, 1024, 4096, 2)),
      (4096, 16384, 4096, 16384, False, (512, 2048, 4096, 2)),
  )
  def test_default_blocks(self, m, k, n, sc_size, quantize_rhs, expected):
    self.assertEqual(
        quantized_matmul.default_quantized_matmul_blocks(
            m, k, n, sc_size=sc_size, quantize_rhs=quantize_rhs
        ),
        expected,
    )

  @parameterized.named_parameters(
      ("non_positive_m", 0, 1024, 1024, 128, "must be positive"),
      ("non_positive_sc", 1024, 1024, 1024, 0, "divisible by positive sc_size"),
      ("non_dividing_sc", 1024, 1024, 1024, 300, "divisible by positive"),
      (
          "sc_exceeds_max_tile_not_full_k",
          1024,
          16384,
          1024,
          4096,
          "exceeds max VMEM K tile",
      ),
  )
  def test_default_blocks_invalid_raises(self, m, k, n, sc_size, err_regex):
    with self.assertRaisesRegex(ValueError, err_regex):
      quantized_matmul.default_quantized_matmul_blocks(m, k, n, sc_size=sc_size)


class QuantizeSubchannelTest(parameterized.TestCase):
  """Tests `quantize_subchannel` helper."""

  def test_shapes_and_values(self):
    x = jnp.array([[7.0, 3.6, 3.5, -3.5]], dtype=jnp.bfloat16)
    q_lhs, sf_lhs = quantized_matmul.quantize_subchannel(
        x, contract_axis=1, sc_size=2, quant_dtype=jnp.int4, scaling="dynamic"
    )
    self.assertEqual(q_lhs.shape, (1, 4))
    self.assertEqual(q_lhs.dtype, jnp.int4)
    self.assertEqual(sf_lhs.shape, (1, 2, 1))
    np.testing.assert_array_equal(
        np.asarray(q_lhs.astype(jnp.int8)),
        np.array([[7, 4, 7, -7]], dtype=np.int8),
    )

    q_rhs, sf_rhs = quantized_matmul.quantize_subchannel(
        x.T, contract_axis=0, sc_size=2, quant_dtype=jnp.int8, scaling="static"
    )
    self.assertEqual(q_rhs.shape, (4, 1))
    self.assertEqual(q_rhs.dtype, jnp.int8)
    self.assertEqual(sf_rhs.shape, (2, 1, 1))
    np.testing.assert_allclose(np.asarray(sf_rhs), 0.5)

  def test_invalid_inputs_raise(self):
    with self.assertRaisesRegex(ValueError, "Expected 2D array"):
      quantized_matmul.quantize_subchannel(
          jnp.ones((2, 2, 2)),
          contract_axis=1,
          sc_size=2,
          quant_dtype=jnp.int8,
      )
    with self.assertRaisesRegex(ValueError, "contract_axis must be 0 or 1"):
      quantized_matmul.quantize_subchannel(
          jnp.ones((4, 4)), contract_axis=2, sc_size=2, quant_dtype=jnp.int8
      )
    with self.assertRaisesRegex(ValueError, "divisible by positive sc_size"):
      quantized_matmul.quantize_subchannel(
          jnp.ones((4, 4)), contract_axis=1, sc_size=3, quant_dtype=jnp.int8
      )
    with self.assertRaisesRegex(ValueError, "Unsupported scaling mode"):
      quantized_matmul.quantize_subchannel(
          jnp.ones((4, 4)),
          contract_axis=1,
          sc_size=2,
          quant_dtype=jnp.int8,
          scaling="invalid",
      )


class QuantizedMatmulKernelTest(parameterized.TestCase):
  """Tests Pallas `quantized_matmul` kernel execution in interpret mode."""

  @parameterized.product(
      quantize_rhs=(True, False),
      dequantize=(False, True),
      scaling=("dynamic", "static"),
      rhs_quant_dtype=(jnp.float8_e4m3fn, jnp.int4),
  )
  def test_subchannel_matches_reference(
      self, quantize_rhs, dequantize, scaling, rhs_quant_dtype
  ):
    """Verifies sub-channel Pallas kernel matches reference across 2 K tiles."""
    m, k, n = 128, 256, 128
    sc_size = 64
    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    lhs = jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key_b, (k, n), dtype=jnp.bfloat16)

    if quantize_rhs:
      rhs_in, w_scales = rhs, None
    else:
      rhs_in, w_scales = quantized_matmul.quantize_subchannel(
          rhs,
          contract_axis=0,
          sc_size=sc_size,
          quant_dtype=rhs_quant_dtype,
          scaling=scaling,
      )
      if not dequantize:
        w_scales = None

    out = quantized_matmul.quantized_matmul(
        lhs,
        rhs_in,
        w_scales,
        sc_size=sc_size,
        quant_dtype=jnp.float8_e4m3fn,
        rhs_quant_dtype=rhs_quant_dtype,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        quantize_rhs=quantize_rhs,
        dequantize=dequantize,
        scaling=scaling,
        block_m=128,
        block_k=128,
        block_n=128,
        interpret=True,
    )
    expected = _reference_quantized_matmul(
        lhs,
        rhs,
        sc_size=sc_size,
        quant_dtype=jnp.float8_e4m3fn,
        rhs_quant_dtype=rhs_quant_dtype,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        dequantize=dequantize,
        scaling=scaling,
    )
    self.assertEqual(out.shape, (m, n))
    self.assertEqual(out.dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

  @parameterized.product(
      quantize_rhs=(True, False),
      dequantize=(False, True),
      scaling=("dynamic", "static"),
      rhs_quant_dtype=(jnp.float8_e4m3fn, jnp.int4),
  )
  def test_full_k_rowwise_multi_tile_matches_reference(
      self, quantize_rhs, dequantize, scaling, rhs_quant_dtype
  ):
    """Verifies multi-K-tile row-wise mode (`sc_size == K > block_k`)."""
    m, k, n = 128, 256, 128
    sc_size = k
    key_a, key_b = jax.random.split(jax.random.PRNGKey(42))
    lhs = jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key_b, (k, n), dtype=jnp.bfloat16)

    if quantize_rhs:
      rhs_in, w_scales = rhs, None
    else:
      rhs_in, w_scales = quantized_matmul.quantize_subchannel(
          rhs,
          contract_axis=0,
          sc_size=sc_size,
          quant_dtype=rhs_quant_dtype,
          scaling=scaling,
      )
      if not dequantize:
        w_scales = None

    out = quantized_matmul.quantized_matmul(
        lhs,
        rhs_in,
        w_scales,
        sc_size=sc_size,
        quant_dtype=jnp.float8_e4m3fn,
        rhs_quant_dtype=rhs_quant_dtype,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        quantize_rhs=quantize_rhs,
        dequantize=dequantize,
        scaling=scaling,
        block_m=128,
        block_k=128,
        block_n=128,
        interpret=True,
    )
    expected = _reference_quantized_matmul(
        lhs,
        rhs,
        sc_size=sc_size,
        quant_dtype=jnp.float8_e4m3fn,
        rhs_quant_dtype=rhs_quant_dtype,
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        dequantize=dequantize,
        scaling=scaling,
    )
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

  def test_rhs_transposed_and_int8_accumulation(self):
    """Verifies `rhs_transposed=True` and `int8 x int8` int32 accumulation."""
    m, k, n = 128, 128, 128
    key_a, key_b = jax.random.split(jax.random.PRNGKey(7))
    lhs = jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key_b, (k, n), dtype=jnp.bfloat16)
    out = quantized_matmul.quantized_matmul(
        lhs,
        rhs.T,
        sc_size=64,
        quant_dtype=jnp.int8,
        out_dtype=jnp.bfloat16,
        quantize_rhs=True,
        dequantize=True,
        rhs_transposed=True,
        block_m=128,
        block_k=128,
        block_n=128,
        interpret=True,
    )
    expected = _reference_quantized_matmul(
        lhs,
        rhs,
        sc_size=64,
        quant_dtype=jnp.int8,
        out_dtype=jnp.bfloat16,
        dequantize=True,
    )
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

  def test_validation_errors(self):
    lhs = jnp.ones((128, 128), dtype=jnp.bfloat16)
    rhs = jnp.ones((128, 128), dtype=jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, "Unsupported scaling mode"):
      quantized_matmul.quantized_matmul(lhs, rhs, sc_size=64, scaling="bad")
    with self.assertRaisesRegex(
        ValueError, "Contracting dimensions must match"
    ):
      quantized_matmul.quantized_matmul(
          lhs, jnp.ones((64, 128), dtype=jnp.bfloat16), sc_size=64
      )
    with self.assertRaisesRegex(ValueError, "must be positive"):
      quantized_matmul.quantized_matmul(lhs, rhs, sc_size=64, block_m=0)
    with self.assertRaisesRegex(ValueError, "must divide"):
      quantized_matmul.quantized_matmul(lhs, rhs, sc_size=64, block_m=100)
    with self.assertRaisesRegex(ValueError, "must be divisible by sc_size"):
      quantized_matmul.quantized_matmul(
          jnp.ones((128, 256), dtype=jnp.bfloat16),
          jnp.ones((256, 128), dtype=jnp.bfloat16),
          sc_size=128,
          block_k=64,
      )
    with self.assertRaisesRegex(ValueError, "w_scales must be provided"):
      quantized_matmul.quantized_matmul(
          lhs,
          rhs.astype(jnp.float8_e4m3fn),
          None,
          sc_size=64,
          quantize_rhs=False,
          dequantize=True,
      )


if __name__ == "__main__":
  absltest.main()
