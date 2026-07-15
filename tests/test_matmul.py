"""Unit tests for matlmul.py."""

from absl.testing import absltest
from accelerator_microbenchmarks.benchmarks import matmul
from accelerator_microbenchmarks.core import registry
import jax
import jax.numpy as jnp
import numpy as np

# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class GeneralizedGemmBenchmarkTest(absltest.TestCase):
  """Unit tests for Generalized GEMM benchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    self.bm = matmul.GeneralizedGemmBenchmark(mesh=self.mock_mesh)

  def test_benchmark_registered(self):
    """Test that the benchmark is properly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("gemm_generalized")
    self.assertEqual(bm_class, matmul.GeneralizedGemmBenchmark)

  def test_generate_inputs(self):
    """Test generating inputs without scaling factors."""
    params = {
        "m": 64,
        "k": 64,
        "n": 64,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
    }
    self.bm.setup(**params)
    inputs = self.bm.generate_inputs(**params)
    self.assertLen(inputs, 2)
    a, b = inputs
    self.assertEqual(a.shape, (64, 64))
    self.assertEqual(b.shape, (64, 64))
    self.assertEqual(a.dtype, jnp.bfloat16)
    self.assertEqual(b.dtype, jnp.bfloat16)

  def test_generate_inputs_with_scaling_factors(self):
    """Test generating inputs with scaling factors."""
    params = {
        "m": 64,
        "k": 64,
        "n": 64,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
        "use_scaling_factors": True,
    }
    self.bm.setup(**params)
    inputs = self.bm.generate_inputs(**params)
    self.assertLen(inputs, 4)
    a, b, sf0, sf1 = inputs
    self.assertEqual(a.shape, (64, 64))
    self.assertEqual(b.shape, (64, 64))
    self.assertEqual(sf0.shape, (64, 1))
    self.assertEqual(sf1.shape, (1, 64))
    self.assertEqual(a.dtype, jnp.bfloat16)
    self.assertEqual(b.dtype, jnp.bfloat16)
    self.assertEqual(sf0.dtype, jnp.float32)
    self.assertEqual(sf1.dtype, jnp.float32)

  def test_run_op(self):
    """Test run op without scaling factors."""
    params = {
        "m": 64,
        "k": 64,
        "n": 64,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
    }
    self.bm.setup(**params)
    inputs = self.bm.generate_inputs(**params)
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (64, 64))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_run_op_with_scaling_factors(self):
    """Test run op with scaling factors."""
    params = {
        "m": 64,
        "k": 64,
        "n": 64,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
        "use_scaling_factors": True,
    }
    self.bm.setup(**params)
    inputs = self.bm.generate_inputs(**params)
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (64, 64))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_get_total_bytes(self):
    """Test calculating total bytes."""
    params = {
        "m": 64,
        "k": 64,
        "n": 64,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
    }
    # Read A (64 * 64 * 2), Read B (64 * 64 * 2), Write Out (64 * 64 * 2)
    # 8192 * 3 = 24576
    expected_bytes = 24576.0
    self.assertAlmostEqual(self.bm.get_total_bytes(**params), expected_bytes)

  def test_get_arithmetic_intensity(self):
    """Test calculating arithmetic intensity."""
    params = {
        "m": 64,
        "k": 64,
        "n": 64,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
    }
    # flops = 2 * 64 * 64 * 64 = 524288
    # bytes = 24576
    # intensity = 524288 / 24576 = 21.333333333333332
    expected_intensity = 524288 / 24576
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(**params), expected_intensity
    )

  def test_calculate_metrics(self):
    """Test calculating performance metrics."""
    params = {
        "m": 64,
        "k": 64,
        "n": 64,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
    }
    self.bm.setup(**params)
    metrics = self.bm.calculate_metrics([1.0, 1.5, 2.0], **params)
    self.assertIn("avg_ms", metrics)
    self.assertIn("tflops_per_sec", metrics)
    self.assertIn("total_flops", metrics)
    self.assertIn("intensity", metrics)
    self.assertEqual(metrics["total_flops"], 524288)
    self.assertAlmostEqual(metrics["intensity"], 524288 / 24576)


if __name__ == "__main__":
  absltest.main()
