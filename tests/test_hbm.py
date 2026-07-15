"""Unit tests for hbm.py."""

from absl.testing import absltest
from accelerator_microbenchmarks.benchmarks import hbm
from accelerator_microbenchmarks.core import registry
import jax
import jax.numpy as jnp
import numpy as np


# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class HBMBandwidthBenchmarkTest(absltest.TestCase):
  """Unit tests for hbm.py."""

  def setUp(self):
    super().setUp()
    # Create a dummy mesh for testing on CPU
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    self.bm = hbm.HBMBandwidthBenchmark(mesh=self.mock_mesh)
    self.params = {
        "size": 1024,
        "dtype": "bfloat16",
    }

  def test_benchmark_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("hbm_bandwidth")
    self.assertEqual(bm_class, hbm.HBMBandwidthBenchmark)

  def test_generate_inputs(self):
    """Verify the shape and type of the generated inputs."""
    self.bm.setup(**self.params)
    inputs = self.bm.generate_inputs(**self.params)
    self.assertEqual(len(inputs), 1)
    x = inputs[0]

    self.assertEqual(x.shape, (1024,))
    self.assertEqual(x.dtype, jnp.bfloat16)

  def test_run_op(self):
    """Verify that running the op returns the expected shape."""
    self.bm.setup(**self.params)
    inputs = self.bm.generate_inputs(**self.params)
    out = self.bm.run_op(*inputs)

    self.assertEqual(out.shape, (1024,))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_get_total_bytes(self):
    """Verify the byte calculation."""
    # size * itemsize (2 bytes for bfloat16) * 2 = 1024 * 2 * 2 = 4096
    expected_bytes = 4096.0
    self.assertAlmostEqual(
        self.bm.get_total_bytes(**self.params), expected_bytes
    )

  def test_get_arithmetic_intensity(self):
    """Verify the intensity calculation."""
    expected_intensity = 0.25
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(**self.params), expected_intensity
    )

  def test_calculate_metrics(self):
    """Verify that metrics are correctly calculated."""
    # total_bytes = 4096
    # wall_clock_avg_ms = 10.0ms -> avg_latency_s = 0.01s
    # bandwidth_gb_s = (4096 / 0.01) / 1e9 = 409600 / 1e9 = 0.0004096
    times_ms = [10.0, 10.0, 10.0]
    metrics = self.bm.calculate_metrics(times_ms, **self.params)

    self.assertAlmostEqual(metrics["avg_ms"], 10.0)
    self.assertAlmostEqual(metrics["bandwidth_gb_s"], 0.0004096)
    self.assertAlmostEqual(metrics["total_bytes_mb"], 0.004096)
    self.assertAlmostEqual(metrics["intensity"], 0.25)


if __name__ == "__main__":
  absltest.main()
