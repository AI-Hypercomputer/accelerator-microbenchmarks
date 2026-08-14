"""Unit tests for device_to_device.py."""

import os

from absl.testing import absltest
from accelerator_microbenchmarks.benchmarks import device_to_device
from accelerator_microbenchmarks.core import registry
import jax

# Force 4 CPU devices for testing multi-device D2D transfers
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_force_host_platform_device_count=4"
)

# Set CPU backend for fast testing without hardware requirements
jax.config.update("jax_platform_name", "cpu")


class DeviceToDeviceParamsTest(absltest.TestCase):
  """Unit tests for DeviceToDeviceParams parameter expansion and properties."""

  def test_expand_test_cases_pair_generation(self):
    """Verifies N * (N - 1) sweep pairs where src != dst."""
    cfg = device_to_device.DeviceToDeviceParams(
        data_size_mib=1, direction="uni"
    )
    cases = cfg.expand_test_cases()
    num_devices = len(jax.devices())
    expected_pairs = num_devices * (num_devices - 1)
    if num_devices <= 1:
      self.assertEmpty(cases)
    else:
      self.assertLen(cases, expected_pairs)
      for c in cases:
        self.assertIsInstance(c, device_to_device.DeviceToDeviceTestCaseParams)
        self.assertNotEqual(c.src_device_index, c.dst_device_index)

  def test_expand_test_cases_attribute_propagation(self):
    """Verifies config flags propagate correctly into generated test cases."""
    cfg = device_to_device.DeviceToDeviceParams(
        data_size_mib=16,
        direction="bi",
        seed=42,
        warmup_tries=3,
        num_runs=7,
    )
    cases = cfg.expand_test_cases()
    num_devices = len(jax.devices())
    if num_devices > 1:
      first_case = cases[0]
      self.assertEqual(first_case.data_size_mib, 16)
      self.assertEqual(first_case.direction, "bi")
      self.assertEqual(first_case.seed, 42)
      self.assertEqual(first_case.warmup_tries, 3)
      self.assertEqual(first_case.num_runs, 7)

  def test_data_size_bytes_property(self):
    """Verifies MiB to bytes conversion."""
    cfg = device_to_device.DeviceToDeviceParams(data_size_mib=4)
    self.assertEqual(cfg.data_size_bytes, 4 * 1024 * 1024)


class DeviceToDeviceBenchmarkTest(absltest.TestCase):
  """Unit tests for DeviceToDeviceBenchmark."""

  def setUp(self):
    super().setUp()
    self.bm_class = registry.benchmark_registry.get_benchmark(
        "device_to_device"
    )
    self.bm = self.bm_class(
        device_to_device.DeviceToDeviceTestCaseParams(
            data_size_mib=1,
            direction="uni",
        )
    )

  def test_benchmark_registered(self):
    """Verify registration."""
    self.assertEqual(self.bm_class, device_to_device.DeviceToDeviceBenchmark)

  def test_get_total_bytes(self):
    """Verify byte calculation."""
    self.bm.config.data_size_mib = 1
    self.bm.config.direction = "uni"
    expected_bytes_uni = 1.0 * 1024 * 1024
    self.assertEqual(
        self.bm.get_total_bytes(),
        expected_bytes_uni,
    )
    self.bm.config.direction = "bi"
    expected_bytes_bi = 2.0 * 1024 * 1024
    self.assertEqual(
        self.bm.get_total_bytes(),
        expected_bytes_bi,
    )

  def test_calculate_metrics(self):
    """Verify bandwidth metrics calculation."""
    times_ms = [10.0, 10.0]
    self.bm.config.data_size_mib = 1
    self.bm.config.direction = "uni"
    self.bm.config.src_device_index = 0
    self.bm.config.dst_device_index = 1
    metrics = self.bm.calculate_metrics(times_ms)
    self.assertAlmostEqual(metrics["avg_ms"], 10.0)
    # total_bytes = 1048576 bytes
    # avg_latency_s = 0.01 s
    # bandwidth_gb_s = 1048576 / (0.01 * 1e9) = 0.1048576 GB/s
    self.assertAlmostEqual(metrics["bandwidth_gb_s"], 0.1048576)
    self.assertEqual(metrics["src_device_index"], 0)
    self.assertEqual(metrics["dst_device_index"], 1)
    self.assertEqual(metrics["direction"], "uni")

  def test_get_arithmetic_intensity(self):
    """Verify arithmetic intensity for pure memory transfer is 0.0."""
    self.assertEqual(self.bm.get_arithmetic_intensity(), 0.0)

  def test_calculate_metrics_zero_latency(self):
    """Verify bandwidth calculation handles zero latency cleanly (inf)."""
    metrics = self.bm.calculate_metrics([0.0, 0.0])
    self.assertEqual(metrics["bandwidth_gb_s"], float("inf"))

  def test_get_run_identifier(self):
    """Verify run identifier string generation."""
    self.bm.config.src_device_index = 0
    self.bm.config.dst_device_index = 1
    self.bm.config.direction = "uni"
    self.bm.config.data_size_mib = 1
    self.assertEqual(self.bm.get_run_identifier(), "d2d_0_to_1_uni_1mib")
    self.bm.config.direction = "bi"
    self.assertEqual(self.bm.get_run_identifier(), "d2d_0_to_1_bi_1mib")

  def test_requires_multihost_sync(self):
    """Verify D2D requires multihost sync for asymmetric target device."""
    self.assertTrue(self.bm.requires_multihost_sync)

  def test_get_device_to_measure_uninitialized(self):
    """Verify get_device_to_measure before setup raises ValueError."""
    uninitialized_bm = self.bm_class(
        device_to_device.DeviceToDeviceTestCaseParams()
    )
    with self.assertRaisesRegex(ValueError, "Mesh not initialized."):
      uninitialized_bm.get_device_to_measure()

  def test_get_device_to_measure_initialized(self):
    """Verify get_device_to_measure resolves destination device after setup."""
    self.bm.config.src_device_index = 0
    self.bm.config.dst_device_index = 1
    self.bm.setup()
    dev = self.bm.get_device_to_measure()
    self.assertEqual(dev, self.bm.mesh.devices.flat[1])

  def test_get_device_to_measure_out_of_bounds(self):
    """Verify get_device_to_measure with invalid dst_device_index raises ValueError."""
    self.bm.setup()
    self.bm.config.dst_device_index = 9999
    with self.assertRaisesRegex(ValueError, "exceeds mesh size"):
      self.bm.get_device_to_measure()

  def test_run_op_uninitialized(self):
    """Verify run_op before setup raises ValueError."""
    uninitialized_bm = self.bm_class(
        device_to_device.DeviceToDeviceTestCaseParams()
    )
    with self.assertRaisesRegex(ValueError, "JIT function not initialized."):
      uninitialized_bm.run_op(None)

  def test_generate_inputs_uninitialized(self):
    """Verify generate_inputs before setup raises ValueError."""
    uninitialized_bm = self.bm_class(
        device_to_device.DeviceToDeviceTestCaseParams()
    )
    with self.assertRaisesRegex(ValueError, "Mesh not initialized."):
      uninitialized_bm.generate_inputs()

  def test_generate_inputs_and_run_op(self):
    """Verify setup, input generation, and kernel execution."""
    self.bm.config.src_device_index = 0
    self.bm.config.dst_device_index = 1
    self.bm.config.data_size_mib = 1
    self.bm.setup()
    inputs = self.bm.generate_inputs()
    self.assertLen(inputs, 1)
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, inputs[0].shape)
    self.assertEqual(out.dtype, inputs[0].dtype)


if __name__ == "__main__":
  absltest.main()
