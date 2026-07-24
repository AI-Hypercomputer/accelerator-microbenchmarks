"""Unit tests for host_device.py."""

from absl.testing import absltest
from accelerator_microbenchmarks.benchmarks import host_device
from accelerator_microbenchmarks.core import registry
import jax
import numpy as np


# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class HostToDeviceBenchmarkTest(absltest.TestCase):
  """Unit tests for HostToDeviceBenchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    self.bm = host_device.HostToDeviceBenchmark(mesh=self.mock_mesh)
    self.params = {
        "data_size_mib": 4,
    }

  def test_benchmark_registered(self):
    """Verify registration."""
    bm_class = registry.benchmark_registry.get_benchmark("host_to_device")
    self.assertEqual(bm_class, host_device.HostToDeviceBenchmark)

  def test_generate_inputs(self):
    """Verify input shape."""
    self.bm.setup(**self.params)
    inputs = self.bm.generate_inputs(**self.params)
    self.assertLen(inputs, 1)
    host_data = inputs[0]
    # 4MB of float32 = 1M elements.
    # 1M elements / 128 = 8192 rows.
    # Shape should be (8192, 128)
    self.assertEqual(host_data.shape, (8192, 128))
    self.assertEqual(host_data.dtype, np.float32)

  def test_run_op(self):
    """Verify run_op returns a JAX array on device."""
    self.bm.setup(**self.params)
    inputs = self.bm.generate_inputs(**self.params)
    out = self.bm.run_op(*inputs)
    self.assertIsInstance(out, jax.Array)
    self.assertEqual(out.shape, (8192, 128))

  def test_get_total_bytes(self):
    """Verify byte calculation."""
    expected_bytes = 4.0 * 1024 * 1024
    self.assertAlmostEqual(
        self.bm.get_total_bytes(**self.params), expected_bytes
    )

  def test_calculate_metrics(self):
    """Verify metrics calculation."""
    times_ms = [10.0, 10.0, 10.0]
    metrics = self.bm.calculate_metrics(times_ms, **self.params)
    # avg_ms = 10.0 -> avg_latency_s = 0.01s
    # data_size_mib = 4 -> 4 / 1024 = 0.00390625 GiB
    # bandwidth_gb_s = 0.00390625 / 0.01 = 0.390625 GiB/s
    self.assertAlmostEqual(metrics["avg_ms"], 10.0)
    self.assertAlmostEqual(metrics["bandwidth_gb_s"], 0.390625)
    self.assertAlmostEqual(metrics["total_bytes_mib"], 4.0)


class DeviceToHostBenchmarkTest(absltest.TestCase):
  """Unit tests for DeviceToHostBenchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    self.bm = host_device.DeviceToHostBenchmark(mesh=self.mock_mesh)
    # Use small runs for testing to limit memory
    self.params = {
        "data_size_mib": 4,
        "num_runs": 2,
        "warmup_tries": 1,
    }

  def test_benchmark_registered(self):
    """Verify registration."""
    bm_class = registry.benchmark_registry.get_benchmark("device_to_host")
    self.assertEqual(bm_class, host_device.DeviceToHostBenchmark)

  def test_generate_inputs(self):
    """Verify input is a JAX Array on device."""
    self.bm.setup(**self.params)
    inputs = self.bm.generate_inputs(**self.params)
    self.assertLen(inputs, 1)
    device_array = inputs[0]
    self.assertIsInstance(device_array, jax.Array)
    self.assertEqual(device_array.shape, (8192, 128))

  def test_run_op(self):
    """Verify run_op returns a numpy array."""
    self.bm.setup(**self.params)
    inputs = self.bm.generate_inputs(**self.params)

    out = self.bm.run_op(*inputs)
    self.assertIsInstance(out, np.ndarray)
    self.assertEqual(out.shape, (8192, 128))

  def test_get_total_bytes(self):
    """Verify byte calculation."""
    expected_bytes = 4.0 * 1024 * 1024
    self.assertAlmostEqual(
        self.bm.get_total_bytes(**self.params), expected_bytes
    )

  def test_calculate_metrics(self):
    """Verify metrics calculation."""
    times_ms = [10.0, 10.0]
    metrics = self.bm.calculate_metrics(times_ms, **self.params)
    self.assertAlmostEqual(metrics["avg_ms"], 10.0)
    self.assertAlmostEqual(metrics["bandwidth_gb_s"], 0.390625)
    self.assertAlmostEqual(metrics["total_bytes_mib"], 4.0)


if __name__ == "__main__":
  absltest.main()
