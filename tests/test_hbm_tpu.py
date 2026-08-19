"""Tests for HBM bandwidth utilizing TPU Ghostfish."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import hbm
from accelerator_microbenchmarks.core import profiler
from accelerator_microbenchmarks.core import system
import jax

_UTILIZATION_THRESHOLD = 0.8


class HBMBandwidthTPUTest(parameterized.TestCase):
  """Unit tests for HBMBandwidthBenchmark on physical TPU hardware."""

  def setUp(self):
    super().setUp()

    # Use Xprof to pull exact on-device execution time, meaning we don't need
    # massive arrays to dilute python overhead anymore.
    self.params = {
        "size": 1024 * 1024 * 128,
        "warmup_tries": 3,
        "num_runs": 10,
        "xprof_timing": True,
    }

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()

  def _get_peak_bandwidth_per_core(self):
    # We only need ghostfish (v7x)
    gfc = system.get_system("gfc")
    # Peak BW is the highest value in the curve.
    # The curve specifies the peak bandwidth for an entire chip (2 TensorCores).
    # Since this microbenchmark forces execution on a single local device,
    # the theoretical peak for our test is half of the chip's peak.
    chip_peak_bw = gfc.hbm.curve_gbps[-1][1]
    return chip_peak_bw / 2.0

  @parameterized.parameters(
      ("copy",),
      ("scale",),
      ("add",),
      ("triad",),
  )
  def test_hbm_utilization_above_80_percent(self, op_type):
    """Verify that HBM utilization is >80% for the given operation."""
    if not jax.devices() or jax.devices()[0].platform != "tpu":
      self.skipTest("This test requires a TPU backend.")

    params = dict(self.params, op_type=op_type)
    config = hbm.HBMBandwidthParams(**params)
    self.bm = hbm.HBMBandwidthBenchmark(config=config)
    self.bm.setup()
    result = self.bm.run()

    # Use xprof_avg_ms for precise device-level timing if available
    xprof_avg_ms = result.metrics.get("xprof_avg_ms", None)
    if xprof_avg_ms is not None:
      total_bytes = self.bm.get_total_bytes()
      bw_gb_s = (total_bytes / (xprof_avg_ms / 1000.0)) / 1e9
    else:
      bw_gb_s = result.metrics.get("bandwidth_gb_s", 0)

    peak_bw = self._get_peak_bandwidth_per_core()
    utilization = bw_gb_s / peak_bw

    print(f"HBM utilization for '{op_type}' was {utilization*100:.2f}%")
    print(f"Measured {bw_gb_s:.2f} GB/s, Peak {peak_bw} GB/s")
    self.assertGreaterEqual(
        utilization,
        _UTILIZATION_THRESHOLD,
        f"HBM utilization for '{op_type}' was {utilization*100:.2f}%, "
        f"which is below the {_UTILIZATION_THRESHOLD*100:.2f}% threshold.",
    )

  def test_hbm_target_device_execution(self):
    """Verify HBM bandwidth execution on a targeted device_id."""
    if not jax.devices() or jax.devices()[0].platform != "tpu":
      self.skipTest("This test requires a TPU backend.")

    local_devices = jax.local_devices()
    target_dev_id = min(7, len(local_devices) - 1)
    target_device = local_devices[target_dev_id]

    undeclared_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if undeclared_outputs_dir:
      temp_dir = os.path.join(undeclared_outputs_dir, "tpu_tensorboard_profile")
      os.makedirs(temp_dir, exist_ok=True)
      print(
          "Preserving real JAX profiler trace to Undeclared Test Outputs"
          f" directory: {temp_dir}"
      )
    else:
      temp_dir = self.create_tempdir().full_path
      print(f"Saving JAX trace locally to: {temp_dir}")

    params = dict(
        self.params,
        op_type="copy",
        device_id=target_dev_id,
        xprof_dir=temp_dir,
    )
    config = hbm.HBMBandwidthParams(**params)
    self.bm = hbm.HBMBandwidthBenchmark(config=config)
    self.bm.setup()

    # 1. Verify configured target device on benchmark instance
    self.assertEqual(self.bm.get_device_to_measure(), target_device)

    # 2. Verify input array buffer placement and sharding on the target device
    inputs = self.bm.generate_inputs()
    for inp in inputs:
      self.assertIn(target_device, inp.devices())

    result = self.bm.run()
    self.assertIn("bandwidth_gb_s", result.metrics)
    self.assertEqual(result.metadata.params.get("device_id"), target_dev_id)

    # 3. Verify target device execution through XProf trace analysis channel
    trace = profiler._load_xprof_trace(  # pylint: disable=protected-access
        self.bm._xprof_dir_actual  # pylint: disable=protected-access
    )
    self.assertIsNotNone(trace, "XProf trace was not generated or loaded.")
    events = trace.get("traceEvents", [])

    pid_to_name = {
        e["pid"]: e.get("args", {}).get("name", "")
        for e in events
        if e.get("name") == "process_name" and "pid" in e
    }

    candidate_events = profiler._extract_candidate_events(events)  # pylint: disable=protected-access
    self.assertNotEmpty(
        candidate_events,
        "No candidate timing marker events found in XProf trace.",
    )

    active_pids = {e["pid"] for e in candidate_events if "pid" in e}
    active_device_names = {
        pid_to_name[pid] for pid in active_pids if pid in pid_to_name
    }

    expected_device_name = f"/device:TPU:{target_device.local_hardware_id}"
    self.assertIn(
        expected_device_name,
        active_device_names,
        f"Expected XProf trace to record execution on {expected_device_name}, "
        f"but found active devices: {active_device_names}",
    )


if __name__ == "__main__":
  absltest.main()
