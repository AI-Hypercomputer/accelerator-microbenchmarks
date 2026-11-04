"""Tests for HBM bandwidth utilizing TPU v7x."""

import os
import statistics
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import hbm
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
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
        "num_elements": 1024 * 1024 * 128,
        "warmup_tries": 3,
        "num_runs": 10,
    }

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()

  def _get_peak_bandwidth_per_core(self):
    # We only need v7x (Ironwood)
    v7x_system = system.get_hardware_spec(system.TpuVersion.TPU7X)
    return v7x_system.peak_hbm_bandwidth_per_device

  @parameterized.parameters(
      ("copy",),
      ("scale",),
      ("add",),
      ("triad",),
      ("read_only",),
      ("write_only",),
  )
  def test_hbm_utilization_above_80_percent(self, op_type):
    """Verify that HBM utilization is >80% for the given operation."""
    if not jax.devices() or jax.devices()[0].platform != "tpu":
      self.skipTest("This test requires a TPU backend.")

    params = dict(self.params, op_type=op_type)
    config = hbm.HBMBandwidthParams(**params)
    self.bm = hbm.HBMBandwidthBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        xprof_config=base.XprofConfig(xprof_timing=True),
    )
    self.bm.setup()
    result = self.bm.run()

    # Use xprof_bandwidth_per_device_gb_s for device-level timing if available
    bw_gb_s = result.metrics.get(
        "xprof_bandwidth_per_device_gb_s",
        result.metrics.get("wall_clock_bandwidth_per_device_gb_s", 0),
    )

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

  @parameterized.parameters(
      ("copy", "device", False),
      ("scale", "device", False),
      ("add", "chip", False),
      ("triad", "chip", False),
      ("read_only", "host", False),
      ("write_only", "host", False),
      # Non-zero local_index cases (preserving b/565335271 write_only coverage
      # on a non-zero TPU device as well as read/copy/chip coverage).
      ("copy", "device", True),
      ("read_only", "device", True),
      ("write_only", "device", True),
      ("copy", "chip", True),
      ("write_only", "chip", True),
  )
  def test_hbm_scope_execution(
      self, op_type: str, scope: str, use_nonzero_local_index: bool
  ):
    """Verify HBM bandwidth execution across scopes and non-zero local_index."""
    if not jax.devices() or jax.devices()[0].platform != "tpu":
      self.skipTest("This test requires a TPU backend.")

    local_devices = list(jax.local_devices())
    chips_by_coords: dict[tuple[int, ...], list[jax.Device]] = {}
    for dev in local_devices:
      chips_by_coords.setdefault(tuple(dev.coords), []).append(dev)
    chips = list(chips_by_coords.values())

    if not use_nonzero_local_index:
      local_index = 0
    elif scope == "device":
      local_index = min(7, len(local_devices) - 1)
      if local_index == 0:
        self.skipTest("Non-zero device local_index requires >= 2 local TPUs.")
    elif scope == "chip":
      local_index = min(3, len(chips) - 1)
      if local_index == 0:
        self.skipTest("Non-zero chip local_index requires >= 2 local chips.")
    else:
      local_index = 0

    if scope == "device":
      target_devices = [local_devices[local_index]]
    elif scope == "chip":
      target_devices = chips[local_index]
    else:
      target_devices = local_devices

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
        op_type=op_type,
        scope=scope,
        local_index=local_index,
    )
    config = hbm.HBMBandwidthParams(**params)
    self.bm = hbm.HBMBandwidthBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        xprof_config=base.XprofConfig(xprof_timing=True, xprof_dir=temp_dir),
    )
    self.bm.setup()

    expected_devices = self.bm.get_devices_to_measure()
    # 1. Verify configured target devices on benchmark instance
    self.assertEqual(expected_devices, target_devices)
    self.assertEqual(self.bm.get_device_to_measure(), target_devices[0])
    if scope == "chip":
      self.assertLen(
          expected_devices, system.TPU7X_HARDWARE_SPEC.devices_per_chip
      )
      self.assertLen({tuple(d.coords) for d in expected_devices}, 1)

    # 2. Verify input and output array buffer placement on the target devices
    inputs = self.bm.generate_inputs()
    for inp in inputs:
      self.assertEqual(inp.devices(), set(expected_devices))
    out = self.bm.run_op(*inputs)
    out.block_until_ready()
    self.assertEqual(
        out.devices(),
        set(expected_devices),
        f"Output buffer for {op_type} was placed on {out.devices()} instead"
        f" of {expected_devices}",
    )
    if use_nonzero_local_index:
      self.assertNotIn(local_devices[0], out.devices())

    result = self.bm.run()
    wc_bw = result.metrics[f"wall_clock_bandwidth_per_{scope}_gb_s"]
    xp_bw = result.metrics[f"xprof_bandwidth_per_{scope}_gb_s"]
    self.assertGreater(wc_bw, 0.0)
    self.assertGreater(xp_bw, 0.0)
    if scope == "device":
      self.assertIsNone(result.metrics["wall_clock_bandwidth_per_chip_gb_s"])
      self.assertIsNone(result.metrics["wall_clock_bandwidth_per_host_gb_s"])
      self.assertIsNone(result.metrics["xprof_bandwidth_per_chip_gb_s"])
      self.assertIsNone(result.metrics["xprof_bandwidth_per_host_gb_s"])
    elif scope == "chip":
      self.assertAlmostEqual(
          result.metrics["wall_clock_bandwidth_per_device_gb_s"], wc_bw / 2.0
      )
      self.assertIsNone(result.metrics["wall_clock_bandwidth_per_host_gb_s"])
      self.assertAlmostEqual(
          result.metrics["xprof_bandwidth_per_device_gb_s"], xp_bw / 2.0
      )
      self.assertIsNone(result.metrics["xprof_bandwidth_per_host_gb_s"])
    else:
      n = len(expected_devices)
      self.assertAlmostEqual(
          result.metrics["wall_clock_bandwidth_per_device_gb_s"], wc_bw / n
      )
      self.assertAlmostEqual(
          result.metrics["wall_clock_bandwidth_per_chip_gb_s"],
          (wc_bw / n) * 2.0,
      )
      self.assertAlmostEqual(
          result.metrics["xprof_bandwidth_per_device_gb_s"], xp_bw / n
      )
      self.assertAlmostEqual(
          result.metrics["xprof_bandwidth_per_chip_gb_s"],
          (xp_bw / n) * 2.0,
      )
    self.assertEqual(result.metadata.params.get("scope"), scope)
    self.assertEqual(result.metadata.params.get("local_index"), local_index)
    self.assertIn(
        f"_scope_{scope}_idx_{local_index}_",
        self.bm._xprof_dir_actual,  # pylint: disable=protected-access
    )

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

    expected_device_names = {
        f"/device:TPU:{d.local_hardware_id}" for d in expected_devices
    }
    self.assertTrue(
        expected_device_names.issubset(active_device_names),
        f"Expected XProf trace to record execution on {expected_device_names},"
        f" but found active devices: {active_device_names}",
    )
    if use_nonzero_local_index:
      device_0_name = f"/device:TPU:{local_devices[0].local_hardware_id}"
      self.assertNotIn(
          device_0_name,
          active_device_names,
          f"Expected non-zero local_index={local_index} ({scope}) not to"
          f" execute MARKER on {device_0_name}, got {active_device_names}",
      )

    # 4. Verify multi-device scopes aggregate XProf latency from slowest device
    trace = profiler.load_xprof_trace(
        self.bm._xprof_dir_actual,  # pylint: disable=protected-access
        self.bm.match_xprof_op_fallback,
    )
    self.assertIsNotNone(trace)
    per_device_slices = [
        trace.device_step_durations_ms(local_device_id=d.local_hardware_id)
        for d in expected_devices
    ]
    for d, dev_slice in zip(expected_devices, per_device_slices):
      self.assertLen(
          dev_slice,
          self.params["num_runs"],
          f"Expected {self.params['num_runs']} XProf steps on {d}, got"
          f" {len(dev_slice)}",
      )
    slowest_slice = max(per_device_slices, key=statistics.median)
    expected_xprof_p50 = self.bm.calculate_latency_stats(
        slowest_slice, prefix=constants.TimingDomain.XPROF
    )["xprof_p50_ms"]
    self.assertAlmostEqual(
        result.metrics["xprof_p50_ms"], expected_xprof_p50, places=6
    )
    expected_agg = (
        constants.XprofDeviceTimingAggregation.LOCAL_SLOWEST_DEVICE
        if len(expected_devices) > 1
        else constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST
    )
    self.assertEqual(
        result.metadata.xprof_config.device_timing_aggregation, expected_agg
    )


if __name__ == "__main__":
  absltest.main()
