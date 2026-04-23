"""Unit tests for the BaseBenchmark execution loop on CPU."""

from unittest.mock import patch
from absl.testing import absltest
import jax
import jax.numpy as jnp

# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")

from google3.experimental.users.rahulasharma.jax_new.jax_benchmarks.src.jax_benchmarks.core.base import BaseBenchmark


class DummyBenchmark(BaseBenchmark):

  def run_op(self, x):
    return x * 2.0

  def generate_inputs(self, **params):
    return (jnp.ones((10, 10)),)

  def get_arithmetic_intensity(self, **params):
    return 1.0

  def get_total_bytes(self, **params):
    return 400.0


class TestBaseBenchmark(absltest.TestCase):

  def test_calculate_metrics_iqr(self):
    bm = DummyBenchmark()
    # Deliberately introduce outliers
    times_ms = [10.0, 11.0, 10.5, 9.5, 100.0, 0.1, 10.2]
    metrics = bm.calculate_metrics(times_ms)

    # 100.0 and 0.1 should be filtered out by IQR
    # Left with [10.0, 11.0, 10.5, 9.5, 10.2] -> mean should be ~10.24
    self.assertGreater(metrics["avg_ms"], 9.0)
    self.assertLess(metrics["avg_ms"], 12.0)

  def test_run_orchestration(self):
    bm = DummyBenchmark()

    params = {
        "warmup_tries": 2,
        "num_runs": 5,
        "hardware_stats": {
            "tflops": {"float32": 100.0},
            "hbm_bw": [(1024, 100.0), (1048576, 200.0)],
            "ici": {"peak_bw_gbps": 50.0, "bidirectional": True},
        },
        "dtype": "float32",
        "xprof_timing": False,
    }

    result = bm.run(**params)

    self.assertEqual(result.metadata.benchmark_name, "DummyBenchmark")
    self.assertEqual(result.metrics["actual_runs"], 5)

    # Validate Roofline values computed correctly
    self.assertIn("roofline_tflops_limit", result.metrics)
    self.assertIn("peak_bw_at_size_gb_s", result.metrics)
    self.assertEqual(
        result.metrics["peak_bw_at_size_gb_s"], 100.0
    )  # size is 400 bytes, < 1024


if __name__ == "__main__":
  absltest.main()
