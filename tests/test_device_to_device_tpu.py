"""Tests for Device-to-Device (D2D) transfer bandwidth utilizing TPU devices."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import device_to_device
import jax


class DeviceToDeviceTPUTest(parameterized.TestCase):
  """Unit tests for DeviceToDeviceBenchmark on physical TPU hardware."""

  def setUp(self):
    super().setUp()

    self.params = {
        "data_size_mib": 16,
        "warmup_tries": 2,
        "num_runs": 5,
    }

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()

  def _check_tpu_requirements(self):
    devices = jax.devices()
    if len(devices) < 2 or "tpu" not in devices[0].platform.lower():
      self.skipTest("This test requires at least 2 TPU devices.")

  @parameterized.parameters("uni", "bi")
  def test_e2e_device_to_device_transfer_on_tpu(self, direction):
    """Verify unidirectional and bidirectional D2D transfers execute and report valid bandwidth."""
    self._check_tpu_requirements()

    params = dict(
        self.params,
        direction=direction,
        src_device_index=0,
        dst_device_index=1,
    )
    config = device_to_device.DeviceToDeviceTestCaseParams(**params)
    bm = device_to_device.DeviceToDeviceBenchmark(config=config)
    bm.setup()
    result = bm.run()

    self.assertIn("bandwidth_gb_s", result.metrics)
    self.assertGreater(result.metrics["bandwidth_gb_s"], 0.0)
    self.assertIn("avg_ms", result.metrics)
    self.assertGreater(result.metrics["avg_ms"], 0.0)
    self.assertEqual(result.metrics["direction"], direction)

  def test_e2e_device_to_device_with_xprof_timing(self):
    """Verify D2D transfer executes with XProf timing enabled on TPU."""
    self._check_tpu_requirements()

    params = dict(
        self.params,
        direction="uni",
        src_device_index=0,
        dst_device_index=1,
        xprof_timing=True,
    )
    config = device_to_device.DeviceToDeviceTestCaseParams(**params)
    bm = device_to_device.DeviceToDeviceBenchmark(config=config)
    bm.setup()
    result = bm.run()

    self.assertIn("bandwidth_gb_s", result.metrics)
    self.assertGreater(result.metrics["bandwidth_gb_s"], 0.0)
    self.assertIn("xprof_url", result.metrics)


if __name__ == "__main__":
  absltest.main()
