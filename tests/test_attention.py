"""Unit tests for attention.py."""

from absl.testing import absltest
from accelerator_microbenchmarks.benchmarks import attention
from accelerator_microbenchmarks.core import registry
import jax
import jax.numpy as jnp
import numpy as np


# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class AttentionBenchmarkTest(absltest.TestCase):
  """Unit tests for attention.py."""

  def setUp(self):
    super().setUp()
    # Create a dummy mesh for testing
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    self.bm = attention.AttentionBenchmark(mesh=self.mock_mesh)

  def test_benchmark_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("attention_flashed")
    self.assertEqual(bm_class, attention.AttentionBenchmark)

  def test_generate_inputs(self):
    """Verify the shape and type of the generated inputs (Q, K, V)."""
    params = {
        "batch": 2,
        "seq_len": 128,
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 64,
    }
    self.bm.setup(**params)
    q, k, v = self.bm.generate_inputs(**params)

    self.assertEqual(q.shape, (2, 4, 128, 64))
    self.assertEqual(k.shape, (2, 4, 128, 64))
    self.assertEqual(v.shape, (2, 4, 128, 64))

    self.assertEqual(q.dtype, jnp.bfloat16)
    self.assertEqual(k.dtype, jnp.bfloat16)
    self.assertEqual(v.dtype, jnp.bfloat16)

  def test_run_op(self):
    """Verify that running the op returns the expected shape."""
    params = {
        "batch": 1,
        "seq_len": 64,
        "num_q_heads": 2,
        "num_kv_heads": 2,
        "head_dim": 32,
        "causal": True,
    }
    self.bm.setup(**params)
    q, k, v = self.bm.generate_inputs(**params)
    out = self.bm.run_op(q, k, v)

    self.assertEqual(out.shape, (1, 2, 64, 32))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_get_total_bytes(self):
    """Verify the byte calculation."""
    params = {
        "batch": 2,
        "seq_len": 128,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 64,
    }
    # q_len = kv_len = 128
    # itemsize = 2 (bfloat16)
    # Q: 2 * 4 * 128 * 64 * 2 = 131072
    # K: 2 * 2 * 128 * 64 * 2 = 65536
    # V: 2 * 2 * 128 * 64 * 2 = 65536
    # Out: 2 * 4 * 128 * 64 * 2 = 131072
    # Total = 393216
    expected_bytes = 393216.0
    self.assertAlmostEqual(self.bm.get_total_bytes(**params), expected_bytes)

  def test_get_arithmetic_intensity(self):
    """Verify the intensity calculation."""
    params = {
        "batch": 1,
        "seq_len": 128,
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 64,
    }
    # q_len = kv_len = 128
    # flops: (4 * 128 * 128 - 2 * 128 * 128) * 4 * 64 = (65536 - 32768) * 256 = 32768 * 256 = 8388608
    # bytes: 1 * ((4*128*64*2) + (4*128*64*2) + (4*128*64*2) + (4*128*64*2)) = 262144
    # intensity = 8388608 / 262144 = 32.0
    expected_intensity = 32.0
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(**params), expected_intensity
    )

  def test_calculate_metrics(self):
    """Verify that metrics are correctly calculated."""
    params = {
        "batch": 1,
        "seq_len": 128,
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 64,
    }
    # total_flops = 8388608 (from above)
    # avg_ms = 10.0ms -> avg_latency_s = 0.01s
    # tflops_per_sec = (8388608 / 0.01) / 1e12 = 8.388608e8 / 1e12 = 0.0008388608
    times_ms = [10.0, 10.0, 10.0]
    metrics = self.bm.calculate_metrics(times_ms, **params)

    self.assertAlmostEqual(metrics["avg_ms"], 10.0)
    self.assertAlmostEqual(metrics["total_flops"], 8388608)
    self.assertAlmostEqual(metrics["tflops_per_sec"], 0.0008388608)
    self.assertAlmostEqual(metrics["intensity"], 32.0)

  def test_run_op_bwd(self):
    """Verify that running the op in bwd mode returns grads with expected shape."""
    params = {
        "batch": 1,
        "seq_len": 64,
        "num_q_heads": 2,
        "num_kv_heads": 2,
        "head_dim": 32,
        "causal": True,
        "mode": "bwd",
    }
    self.bm.setup(**params)
    q, k, v = self.bm.generate_inputs(**params)
    dq, dk, dv = self.bm.run_op(q, k, v)

    self.assertEqual(dq.shape, (1, 2, 64, 32))
    self.assertEqual(dk.shape, (1, 2, 64, 32))
    self.assertEqual(dv.shape, (1, 2, 64, 32))
    self.assertEqual(dq.dtype, jnp.bfloat16)
    self.assertEqual(dk.dtype, jnp.bfloat16)
    self.assertEqual(dv.dtype, jnp.bfloat16)

  def test_get_total_bytes_bwd(self):
    """Verify the byte calculation for bwd mode."""
    params = {
        "batch": 2,
        "seq_len": 128,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 64,
        "mode": "bwd",
    }
    expected_bytes = 786432.0
    self.assertAlmostEqual(self.bm.get_total_bytes(**params), expected_bytes)

  def test_get_arithmetic_intensity_bwd(self):
    """Verify the intensity calculation for bwd mode."""
    params = {
        "batch": 1,
        "seq_len": 128,
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 64,
        "mode": "bwd",
    }
    expected_intensity = 32.0
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(**params), expected_intensity
    )

  def test_get_arithmetic_intensity_non_causal(self):
    """Verify the intensity calculation for non-causal attention."""
    params = {
        "batch": 1,
        "seq_len": 128,
        "num_q_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 64,
        "causal": False,
    }
    expected_intensity = 64.0
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(**params), expected_intensity
    )



if __name__ == "__main__":
  absltest.main()
