"""Unit tests for BaseRunExecutor and SoakingRunExecutor."""

import contextlib
import dataclasses
import io
import os
import shutil
import tempfile
import time
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import config as config_module
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import executor
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class DummyBenchmark(base.BaseBenchmark):
  """A dummy benchmark for testing execution strategies."""

  def __init__(
      self,
      config=None,
      hardware_spec=None,
      mesh=None,
      xprof_config=None,
  ):
    if config is None:
      config = base.BaseBenchmarkParams()
    if hardware_spec is None:
      hardware_spec = system.get_hardware_spec(system.TpuVersion.TPU7X)
    super().__init__(
        config=config,
        hardware_spec=hardware_spec,
        mesh=mesh,
        xprof_config=xprof_config,
    )

  def run_op(self, x):
    return x * 2.0

  def generate_inputs(self, **_params):
    return (jnp.ones((10, 10)),)

  def get_total_bytes(self, **_params):
    return 400.0

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ):
    del latency_ms, prefix
    return {}


@dataclasses.dataclass
class SoakingDummyParams(
    config_module.SoakingExecutionParamsMixin, base.BaseBenchmarkParams
):
  """Dummy parameters with SoakingExecutionParamsMixin enabled."""


class SoakingDummyBenchmark(DummyBenchmark):
  """Dummy benchmark using SoakingDummyParams."""

  Config = SoakingDummyParams  # pylint: disable=invalid-name


class SoakingBenchmarkTest(absltest.TestCase):
  """Tests for SoakingExecutionParamsMixin and SoakingRunExecutor."""

  def setUp(self):
    super().setUp()
    self._platform_patcher = mock.patch(
        "accelerator_microbenchmarks.core.platform.get_platform_info",
        return_value=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
    )
    self._platform_patcher.start()

  def tearDown(self):
    self._platform_patcher.stop()
    super().tearDown()

  def test_iqr_bypass_for_soaking_benchmark(self):
    """Verifies SoakingExecutionParamsMixin retains tail samples without IQR."""
    cfg = SoakingDummyParams(
        warmup_tries=1, min_duration_s=1.0, samples_per_run=2
    )
    bm = SoakingDummyBenchmark(config=cfg)
    times_ms = [1.0, 1.0, 1.0, 1.0, 100.0]
    metrics = bm.calculate_metrics(times_ms)
    # Without IQR removal, average of [1, 1, 1, 1, 100] is 20.8
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 20.8, places=4)
    self.assertAlmostEqual(
        metrics[constants.WALL_CLOCK_PRE_SOAKING_P50_MS], 1.0, places=4
    )
    self.assertAlmostEqual(
        metrics[constants.WALL_CLOCK_POST_SOAKING_P50_MS], 100.0, places=4
    )
    self.assertAlmostEqual(
        metrics[constants.WALL_CLOCK_SLOWDOWN_RATIO], 100.0, places=4
    )
    self.assertNotIn(constants.XPROF_PRE_SOAKING_P50_MS, metrics)
    self.assertNotIn(constants.XPROF_POST_SOAKING_P50_MS, metrics)
    self.assertNotIn(constants.XPROF_SLOWDOWN_RATIO, metrics)

  def test_base_executor_warmup_and_iqr_boundary(self):
    """Verifies BaseRunExecutor warmup blocking and exact 1.5*IQR boundary."""
    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=2, num_runs=1)
    )
    with mock.patch.object(
        jax, "block_until_ready", wraps=jax.block_until_ready
    ) as mock_block:
      bm.executor.run_warmup(bm.generate_inputs())
      self.assertEqual(mock_block.call_count, 2)

    # q1=10, q3=20, iqr=10 -> upper_bound = 20 + 15 = 35.0 (inclusive)
    stats = bm.calculate_latency_stats(
        [10.0, 10.0, 20.0, 20.0, 35.0], constants.TimingDomain.WALL_CLOCK
    )
    self.assertAlmostEqual(stats["wall_clock_avg_ms"], 19.0, places=4)

  def test_soaking_calculate_metrics_empty_times_returns_none(self):
    """Verifies empty times_ms sets wall-clock soaking and thermal keys to None."""
    cfg = SoakingDummyParams(
        warmup_tries=1, min_duration_s=1.0, samples_per_run=2
    )
    bm = SoakingDummyBenchmark(config=cfg)
    metrics = bm.calculate_metrics([])
    for key in (
        constants.WALL_CLOCK_PRE_SOAKING_P50_MS,
        constants.WALL_CLOCK_POST_SOAKING_P50_MS,
        constants.WALL_CLOCK_SLOWDOWN_RATIO,
        *constants.THERMAL_METRIC_KEYS,
    ):
      self.assertIsNone(metrics[key])

  def test_base_benchmark_calculate_metrics_integrates_thermal_keys(self):
    """Verifies end-to-end integration of thermal keys in calculate_metrics."""
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "tpu7x_fw_counters.xplane.pb"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      worst_dir = os.path.join(tmpdir, "worst_window")
      os.makedirs(worst_dir)
      shutil.copy(
          data_path, os.path.join(worst_dir, "tpu7x_fw_counters.xplane.pb")
      )
      baseline_dir = os.path.join(tmpdir, "baseline_window")
      os.makedirs(baseline_dir)
      shutil.copy(
          data_path, os.path.join(baseline_dir, "tpu7x_fw_counters.xplane.pb")
      )
      bench = SoakingDummyBenchmark(
          config=SoakingDummyParams(min_duration_s=1.0)
      )
      bench._xprof_dir_actual = tmpdir  # pylint: disable=protected-access
      metrics = bench.calculate_metrics([10.0] * 50 + [20.0] * 50)

    for key in constants.THERMAL_METRIC_KEYS:
      self.assertIn(key, metrics)
    self.assertEqual(metrics["hbm_peak_temp_c"], 40.0)
    self.assertAlmostEqual(
        metrics["vdd_core_power_mean_w"], 352.5091138564901, places=4
    )
    self.assertIsNone(metrics["peak_temp_c"])
    self.assertEqual(metrics["hbm_throttle_time_pct"], 0.0)

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
      return_value=None,
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
      return_value=[1.0],
  )
  @mock.patch("jax.profiler.trace")
  def test_samples_per_run_and_run_chunk(
      self, mock_trace, mock_parse_xprof, mock_upload_xprof
  ):
    """Verifies _samples_per_run, _run_chunk(samples=...), and banner."""
    del mock_trace, mock_parse_xprof, mock_upload_xprof

    class ExperimentalSoakingDummyBenchmark(SoakingDummyBenchmark):
      name = "exp_soak_dummy"
      is_experimental = True

    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = SoakingDummyParams(
          warmup_tries=2, min_duration_s=0.01, samples_per_run=2
      )
      bm = ExperimentalSoakingDummyBenchmark(
          config=cfg,
          xprof_config=base.XprofConfig(xprof_timing=True, xprof_dir=tmpdir),
      )
      self.assertEqual(bm.executor._samples_per_run(), 2)  # pylint: disable=protected-access
      out_inputs = bm.executor._run_chunk(*bm.generate_inputs(), samples=2)  # pylint: disable=protected-access
      self.assertLen(out_inputs, 1)
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        res = bm.run()
      self.assertIn(
          "Running experimental benchmark 'exp_soak_dummy'", buf.getvalue()
      )
      # Phase 1 (1) + Phase 2 (>= 1) + Phase 3 (1) >= 3 windows.
      self.assertGreaterEqual(res.metrics["actual_runs"], 6)
      self.assertGreaterEqual(len(res.raw_times_ms), 3)
      # Worst window has samples_per_run=2 traced iterations but the mocked
      # trace yields 1 duration, so the truncation is recorded and warned.
      self.assertEqual(res.metrics["xprof_captured_runs"], 1)
      self.assertEqual(res.metrics["xprof_expected_runs"], 2)
      self.assertIn("Warning: XProf captured 1/2 traced runs", buf.getvalue())

  def test_soaking_warmup_executes_exact_warmup_tries(self):
    """Verifies SoakingRunExecutor.run_warmup executes exact warmup_tries."""
    cfg = SoakingDummyParams(
        warmup_tries=10,
        min_duration_s=0.01,
        samples_per_run=25,
    )
    bm = SoakingDummyBenchmark(config=cfg)
    with mock.patch.object(bm, "run_op", wraps=bm.run_op) as mock_run_op:
      bm.executor.run_warmup(bm.generate_inputs())
      self.assertEqual(mock_run_op.call_count, 10)

  def test_soaking_executor_validates_xprof_requirements(self):
    """Verifies SoakingRunExecutor fails fast when XProf or soak requirements are unmet."""
    # 1. xprof_timing=False
    bm_no_xprof = SoakingDummyBenchmark(
        config=SoakingDummyParams(warmup_tries=0, min_duration_s=0.01),
        xprof_config=base.XprofConfig(
            xprof_timing=False, xprof_dir="/tmp/soak"
        ),
    )
    with self.assertRaisesRegex(
        ValueError, "SoakingRunExecutor requires xprof_timing=True"
    ):
      bm_no_xprof.run()

    # 2. local_xprof_dir=None
    bm_no_dir = SoakingDummyBenchmark(
        config=SoakingDummyParams(warmup_tries=0, min_duration_s=0.01),
        xprof_config=base.XprofConfig(xprof_timing=True, xprof_dir=""),
    )
    with self.assertRaisesRegex(
        ValueError, "SoakingRunExecutor requires xprof_timing=True"
    ):
      bm_no_dir.executor.execute(bm_no_dir.generate_inputs(), None)

    # 3. min_duration_s <= 0.0
    bm_zero_dur = SoakingDummyBenchmark(
        config=SoakingDummyParams(warmup_tries=0, min_duration_s=0.01),
        xprof_config=base.XprofConfig(xprof_timing=True, xprof_dir="/tmp/soak"),
    )
    bm_zero_dur.config.min_duration_s = 0.0
    with self.assertRaisesRegex(
        ValueError, "SoakingRunExecutor requires min_duration_s > 0.0."
    ):
      bm_zero_dur.run()

  def test_soaking_loop_records_baseline_and_worst_windows(self):
    """Verifies SoakingRunExecutor.execute records baseline_window and worst_window."""
    cfg = SoakingDummyParams(
        warmup_tries=0, samples_per_run=1, min_duration_s=0.01
    )

    @contextlib.contextmanager
    def fake_trace(trace_dir, **_kwargs):
      os.makedirs(trace_dir, exist_ok=True)
      yield

    with tempfile.TemporaryDirectory() as tmpdir:
      bm = SoakingDummyBenchmark(
          config=cfg,
          xprof_config=base.XprofConfig(xprof_timing=True, xprof_dir=tmpdir),
      )
      with (
          mock.patch(
              "jax.profiler.trace",
              side_effect=fake_trace,
          ),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              side_effect=[[1.0], [2.5]],
          ),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
              return_value=None,
          ),
      ):
        res = bm.run()
      actual_trace_root = bm._xprof_dir_actual  # pylint: disable=protected-access
      self.assertCountEqual(
          os.listdir(actual_trace_root), ["baseline_window", "worst_window"]
      )
    self.assertEqual(bm.executor.first_window_xprof_duration, [1.0])
    self.assertEqual(bm.executor.last_window_xprof_duration, [2.5])
    self.assertEqual(bm.executor.xprof_durations, [2.5])
    self.assertAlmostEqual(
        res.metrics["xprof_pre_soaking_p50_ms"], 1.0, places=4
    )
    self.assertAlmostEqual(
        res.metrics["xprof_post_soaking_p50_ms"], 2.5, places=4
    )
    self.assertAlmostEqual(res.metrics["xprof_slowdown_ratio"], 2.5, places=4)
    # Soaking counts only the worst window: samples_per_run=1, 1 duration.
    self.assertEqual(res.metrics["xprof_captured_runs"], 1)
    self.assertEqual(res.metrics["xprof_expected_runs"], cfg.samples_per_run)

    # Also verify non-positive synced_p50 passes has_xprof_timings=False.
    with (
        mock.patch(
            "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
            side_effect=[[-1.0], [-1.0]],
        ),
        mock.patch(
            "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
            return_value=None,
        ),
    ):
      fallback_metrics = bm._apply_xprof_timing_and_sync({})  # pylint: disable=protected-access
    self.assertIsNone(fallback_metrics[constants.XPROF_PRE_SOAKING_P50_MS])
    self.assertIsNone(fallback_metrics[constants.XPROF_POST_SOAKING_P50_MS])
    self.assertIsNone(fallback_metrics[constants.XPROF_SLOWDOWN_RATIO])

  def test_soaking_execute_phases_and_chunk_accounting(self):
    """Verifies Phase 1, Phase 2 soak chunks, and Phase 3 execute via _run_window."""
    cfg = SoakingDummyParams(
        warmup_tries=0, samples_per_run=2, min_duration_s=0.01
    )
    events: list[str] = []
    real_block = jax.block_until_ready

    def recording_block(x):
      events.append("block")
      return real_block(x)

    @contextlib.contextmanager
    def fake_trace(trace_dir, **_kwargs):
      os.makedirs(trace_dir, exist_ok=True)
      events.append(f"trace_enter:{os.path.basename(trace_dir)}")
      yield
      events.append(f"trace_exit:{os.path.basename(trace_dir)}")

    with tempfile.TemporaryDirectory() as tmpdir:
      bm = SoakingDummyBenchmark(
          config=cfg,
          xprof_config=base.XprofConfig(xprof_timing=True, xprof_dir=tmpdir),
      )
      real_run_op = bm.run_op

      def recording_run_op(x):
        events.append("run_op")
        return real_run_op(x)

      with (
          mock.patch.object(bm, "run_op", side_effect=recording_run_op),
          mock.patch.object(
              jax, "block_until_ready", side_effect=recording_block
          ),
          mock.patch("jax.profiler.trace", side_effect=fake_trace),
      ):
        exec_result = bm.executor.execute(bm.generate_inputs(), tmpdir)
    raw_times = exec_result.raw_times_ms
    actual_runs = exec_result.actual_runs

    baseline_enter = events.index("trace_enter:baseline_window")
    baseline_exit = events.index("trace_exit:baseline_window")
    worst_enter = events.index("trace_enter:worst_window")
    worst_exit = events.index("trace_exit:worst_window")
    # Both traced windows execute `samples_per_run` iterations and block once.
    self.assertEqual(
        events[baseline_enter + 1 : baseline_exit],
        ["run_op"] * cfg.samples_per_run + ["block"],
    )
    self.assertEqual(
        events[worst_enter + 1 : worst_exit],
        ["run_op"] * cfg.samples_per_run + ["block"],
    )
    # Every executed iteration is counted once in actual_runs and each chunk
    # contributes one wall-clock sample.
    self.assertEqual(actual_runs, events.count("run_op"))
    self.assertEqual(actual_runs, len(raw_times) * cfg.samples_per_run)
    # Phase 1 + >= 1 soak chunk + Phase 3.
    self.assertGreaterEqual(len(raw_times), 3)
    self.assertLen(bm.executor.soak_wall_ms, len(raw_times) - 2)
    self.assertEqual(raw_times[1:-1], bm.executor.soak_wall_ms)
    # Only the traced `worst_window` feeds xprof_* stats.
    self.assertEqual(exec_result.xprof_expected_runs, cfg.samples_per_run)

  def test_soak_wall_ms_reflects_device_completion_time(self):
    """Verifies soak chunks are timed to device completion, not async dispatch."""
    cfg = SoakingDummyParams(
        warmup_tries=0, samples_per_run=1, min_duration_s=0.05
    )
    real_block = jax.block_until_ready
    delay_s = 0.02

    def slow_block(x):
      time.sleep(delay_s)
      return real_block(x)

    with tempfile.TemporaryDirectory() as tmpdir:
      bm = SoakingDummyBenchmark(
          config=cfg,
          xprof_config=base.XprofConfig(xprof_timing=True, xprof_dir=tmpdir),
      )
      with (
          mock.patch.object(jax, "block_until_ready", side_effect=slow_block),
          mock.patch(
              "jax.profiler.trace", return_value=contextlib.nullcontext()
          ),
      ):
        raw_times = bm.executor.execute(
            bm.generate_inputs(), tmpdir
        ).raw_times_ms

    self.assertNotEmpty(bm.executor.soak_wall_ms)
    for soak_ms in bm.executor.soak_wall_ms:
      self.assertGreaterEqual(soak_ms, delay_s * 1000.0)
    self.assertGreaterEqual(raw_times[0], delay_s * 1000.0)
    self.assertGreaterEqual(raw_times[-1], delay_s * 1000.0)

  def test_wall_clock_soaking_stats_derived_from_untraced_soak_chunks(self):
    """Verifies wall-clock pre_soaking/post_soaking p50 use first vs last soak chunk and ignore Phase 1/3 windows.

    On fast clock-skip throttling hosts the traced `worst_window` can read
    un-throttled after a profiler-init gap, so the wall-clock slowdown must be
    sourced from the untraced Phase 2 soak chunks (`soak_wall_ms[0]` vs
    `soak_wall_ms[-1]`).
    """
    cfg = SoakingDummyParams(
        warmup_tries=1, min_duration_s=1.0, samples_per_run=2
    )
    bm = SoakingDummyBenchmark(config=cfg)
    soak = [10.0] * 7 + [11.0, 11.5, 12.0]
    bm.executor.soak_wall_ms = list(soak)
    # Traced windows (first/last) are deliberately faster than the soak.
    times_ms = [9.0] + soak + [9.0]
    metrics = bm.calculate_metrics(times_ms)
    self.assertAlmostEqual(
        metrics[constants.WALL_CLOCK_PRE_SOAKING_P50_MS], 10.0, places=4
    )
    self.assertAlmostEqual(
        metrics[constants.WALL_CLOCK_POST_SOAKING_P50_MS], 12.0, places=4
    )
    self.assertAlmostEqual(
        metrics[constants.WALL_CLOCK_SLOWDOWN_RATIO], 1.2, places=4
    )
    # Aggregate wall-clock stats still cover every window.
    self.assertAlmostEqual(
        metrics["wall_clock_p50_ms"], float(np.percentile(times_ms, 50)), 4
    )

  def test_resolve_max_device_id_selects_slowest_throttled_device(
      self,
  ):
    """Verifies _resolve_max_device_id selects the slowest local device."""
    cfg = SoakingDummyParams(
        warmup_tries=0, min_duration_s=0.01, samples_per_run=2
    )
    bm = SoakingDummyBenchmark(
        config=cfg,
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            device_mode=constants.XprofDeviceMode.MAX_DEVICE,
        ),
    )
    fake_devs = [mock.MagicMock(local_hardware_id=i) for i in range(8)]

    def fake_parse_xprof(trace_dir, fallback_fn=None, local_device_id=0):
      del fallback_fn
      if "baseline_window" in trace_dir:
        return [5.008, 5.010]
      if local_device_id in (4, 5):
        return [6.021, 6.019]
      return [5.009, 5.009]

    with tempfile.TemporaryDirectory() as tmpdir:
      w0 = os.path.join(tmpdir, "baseline_window")
      w1 = os.path.join(tmpdir, "worst_window")
      os.makedirs(w0, exist_ok=True)
      os.makedirs(w1, exist_ok=True)
      with (
          mock.patch.object(jax, "local_devices", return_value=fake_devs),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              side_effect=fake_parse_xprof,
          ),
      ):
        target_dev_id = bm.executor._resolve_max_device_id(w1)  # pylint: disable=protected-access
        self.assertIn(target_dev_id, (4, 5))
        d0 = bm.executor._extract_window_durations(w0, target_device_id=target_dev_id)  # pylint: disable=protected-access
        d1 = bm.executor._extract_window_durations(w1, target_device_id=target_dev_id)  # pylint: disable=protected-access
    self.assertEqual(d0 + d1, [5.008, 5.010, 6.021, 6.019])

    class HostCpuSoakingDummyBenchmark(SoakingDummyBenchmark):
      xprof_target_host_cpu = True

    host_cpu_bm = HostCpuSoakingDummyBenchmark(config=cfg)
    self.assertIsNone(host_cpu_bm.executor._resolve_max_device_id("/tmp/fake"))  # pylint: disable=protected-access

  def test_xprof_device_mode_first_vs_max_device(self):
    """Verifies XprofConfig.device_mode ('first_device' vs 'max_device')."""
    fake_devs = [mock.MagicMock(local_hardware_id=i) for i in range(8)]

    def fake_parse_xprof(trace_dir, fallback_fn=None, local_device_id=0):
      del trace_dir, fallback_fn
      if local_device_id in (4, 5):
        return [6.021, 6.019]
      return [5.009, 5.009]

    with (
        mock.patch.object(jax, "local_devices", return_value=fake_devs),
        mock.patch(
            "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
            side_effect=fake_parse_xprof,
        ),
    ):
      bm_first = DummyBenchmark(
          xprof_config=base.XprofConfig(
              xprof_timing=True,
              device_mode=constants.XprofDeviceMode.FIRST_DEVICE,
          )
      )
      self.assertEqual(
          bm_first.executor._parse_xprof_durations("/tmp/fake"), [5.009, 5.009]  # pylint: disable=protected-access
      )

      bm_max = DummyBenchmark(
          xprof_config=base.XprofConfig(
              xprof_timing=True,
              device_mode=constants.XprofDeviceMode.MAX_DEVICE,
          )
      )
      self.assertEqual(
          bm_max.executor._parse_xprof_durations("/tmp/fake"), [6.021, 6.019]  # pylint: disable=protected-access
      )

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
      return_value=None,
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
      return_value=[1.0],
  )
  @mock.patch("jax.profiler.trace")
  def test_profiler_options_base_and_soaking(
      self, mock_trace, mock_parse, mock_upload
  ):
    """Verifies trace-reduction options for all benchmarks plus soaking FW keys."""
    del mock_parse, mock_upload
    mock_trace.return_value = contextlib.nullcontext()

    # 0. Standard benchmark with xprof_timing=False or local_xprof_dir=None
    # does not invoke trace (even when only one condition is false).
    no_trace_bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=1),
        xprof_config=base.XprofConfig(xprof_timing=False),
    )
    no_trace_result = no_trace_bm.run()
    mock_trace.assert_not_called()
    self.assertNotIn("xprof_captured_runs", no_trace_result.metrics)
    self.assertNotIn("xprof_expected_runs", no_trace_result.metrics)
    no_trace_exec = no_trace_bm.executor.execute(
        no_trace_bm.generate_inputs(), "/tmp/std_xprof"
    )
    mock_trace.assert_not_called()
    self.assertIsNone(no_trace_exec.xprof_expected_runs)

    # 1. Standard non-mixin benchmark (DummyBenchmark)
    std_bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=1),
        xprof_config=base.XprofConfig(
            xprof_timing=True, xprof_dir="/tmp/std_xprof"
        ),
    )
    std_exec = std_bm.executor.execute(std_bm.generate_inputs(), None)
    mock_trace.assert_not_called()
    self.assertIsNone(std_exec.xprof_expected_runs)
    std_result = std_bm.run()
    mock_trace.assert_called_once()
    std_kwargs = mock_trace.call_args.kwargs
    self.assertFalse(std_kwargs["create_perfetto_link"])
    std_opts = std_kwargs["profiler_options"]
    self.assertEqual(
        std_opts.advanced_configuration,
        executor._BASE_PROFILER_ADVANCED_CONFIG,  # pylint: disable=protected-access
    )
    # The host tracer stays enabled: H2D/D2H time against /host:CPU events.
    self.assertNotEqual(std_opts.host_tracer_level, 0)
    # The whole standard loop is traced, so expected == actual_runs.
    self.assertEqual(
        std_result.metrics["xprof_expected_runs"],
        std_result.metrics["actual_runs"],
    )
    self.assertEqual(std_result.metrics["xprof_captured_runs"], 1)

    # 2. SoakingExecutionParamsMixin benchmark
    mock_trace.reset_mock()
    soak_bm = SoakingDummyBenchmark(
        config=SoakingDummyParams(
            warmup_tries=0, min_duration_s=0.01, samples_per_run=1
        ),
        xprof_config=base.XprofConfig(
            xprof_timing=True, xprof_dir="/tmp/soak_xprof"
        ),
    )
    soak_bm.run()
    mock_trace.assert_called()
    self.assertFalse(mock_trace.call_args.kwargs["create_perfetto_link"])
    opts = mock_trace.call_args.kwargs.get("profiler_options")
    self.assertIsNotNone(opts)
    self.assertEqual(opts.host_tracer_level, 0)
    # Soaking options are a superset of the base trace-reduction options.
    self.assertLessEqual(
        executor._BASE_PROFILER_ADVANCED_CONFIG.items(),  # pylint: disable=protected-access
        opts.advanced_configuration.items(),
    )
    self.assertTrue(
        opts.advanced_configuration["tpu_e2e_enable_fw_thermal_event"]
    )
    self.assertTrue(
        opts.advanced_configuration["tpu_e2e_enable_fw_power_level_event"]
    )
    self.assertTrue(
        opts.advanced_configuration["tpu_e2e_enable_fw_throttle_event"]
    )
    self.assertEqual(opts.advanced_configuration["tpu_power_trace_level"], 1)
    self.assertFalse(opts.advanced_configuration["tpu_perf_counters"])

  @mock.patch("jax.profiler.trace")
  @mock.patch(
      "jax.profiler.ProfileOptions",
      side_effect=RuntimeError("ProfileOptions unavailable"),
  )
  def test_profiler_options_failure_raises(self, mock_options, mock_trace):
    """Verifies both executors raise instead of tracing with default options."""
    del mock_options
    with tempfile.TemporaryDirectory() as tmpdir:
      xprof_config = base.XprofConfig(xprof_timing=True, xprof_dir=tmpdir)
      std_bm = DummyBenchmark(
          config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=1),
          xprof_config=xprof_config,
      )
      soak_bm = SoakingDummyBenchmark(
          config=SoakingDummyParams(
              warmup_tries=0, min_duration_s=0.01, samples_per_run=1
          ),
          xprof_config=xprof_config,
      )
      for bm in (std_bm, soak_bm):
        with self.subTest(executor=type(bm.executor).__name__):
          with self.assertRaisesRegex(
              RuntimeError, "ProfileOptions unavailable"
          ):
            bm.executor.execute(bm.generate_inputs(), tmpdir)
    # A default-options trace would reintroduce run truncation.
    mock_trace.assert_not_called()

  def test_global_slowest_per_step_all_ok_multihost(self):
    """Verifies two-stage global_slowest_per_step reduction across local devices and hosts."""
    fake_devs = [
        mock.MagicMock(local_hardware_id=0),
        mock.MagicMock(local_hardware_id=1),
    ]

    def fake_parse_xprof(trace_dir, fallback_fn=None, local_device_id=0):
      del trace_dir, fallback_fn
      if local_device_id == 0:
        return [10.0, 22.0, 30.0]
      return [14.0, 18.0, 35.0]

    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=3),
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            device_mode=constants.XprofDeviceMode.GLOBAL_SLOWEST_PER_STEP,
        ),
    )

    with (
        mock.patch.object(
            jax, "local_devices", return_value=fake_devs
        ),
        mock.patch(
            "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
            side_effect=fake_parse_xprof,
        ),
        mock.patch.object(
            multihost_utils,
            "process_allgather",
            side_effect=[
                np.array([[0, 3], [0, 3]], dtype=np.int32),
                np.array(
                    [[14.0, 22.0, 35.0], [16.0, 20.0, 31.0]], dtype=np.float32
                ),
            ],
        ) as mock_gather,
    ):
      durations = bm.executor.get_xprof_durations("/tmp/fake")

    self.assertEqual(durations, [16.0, 22.0, 35.0])
    self.assertEqual(mock_gather.call_count, 2)
    np.testing.assert_array_equal(
        mock_gather.call_args_list[0].args[0],
        np.array([0, 3], dtype=np.int32),
    )
    np.testing.assert_allclose(
        mock_gather.call_args_list[1].args[0],
        np.array([14.0, 22.0, 35.0], dtype=np.float32),
    )

  def test_global_slowest_per_step_single_host_reduces_to_in_host_step_max(
      self,
  ):
    """Verifies single-host global_slowest_per_step reduces to in-host per-step max."""
    fake_devs = [
        mock.MagicMock(local_hardware_id=0),
        mock.MagicMock(local_hardware_id=1),
    ]

    def fake_parse_xprof(trace_dir, fallback_fn=None, local_device_id=0):
      del trace_dir, fallback_fn
      if local_device_id == 0:
        return [10.0, 22.0, 30.0]
      return [14.0, 18.0, 35.0]

    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=3),
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            device_mode=constants.XprofDeviceMode.GLOBAL_SLOWEST_PER_STEP,
        ),
    )

    with (
        mock.patch.object(
            jax, "local_devices", return_value=fake_devs
        ),
        mock.patch(
            "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
            side_effect=fake_parse_xprof,
        ),
        mock.patch.object(
            multihost_utils,
            "process_allgather",
            wraps=multihost_utils.process_allgather,
        ) as spy_gather,
    ):
      durations = bm.executor.get_xprof_durations("/tmp/fake")

    self.assertEqual(durations, [14.0, 22.0, 35.0])
    self.assertEqual(spy_gather.call_count, 2)

  def test_global_slowest_per_step_parse_failed_returns_empty_on_all_hosts(
      self,
  ):
    """Verifies PARSE_FAILED on any host causes all hosts to return []."""
    fake_devs = [
        mock.MagicMock(local_hardware_id=0),
        mock.MagicMock(local_hardware_id=1),
    ]
    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=3),
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            device_mode=constants.XprofDeviceMode.GLOBAL_SLOWEST_PER_STEP,
        ),
    )

    with self.subTest("local_empty_slice"):

      def fake_empty(trace_dir, fallback_fn=None, local_device_id=0):
        del trace_dir, fallback_fn
        return [10.0, 20.0, 30.0] if local_device_id == 0 else []

      with (
          mock.patch.object(
              jax, "local_devices", return_value=fake_devs
          ),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              side_effect=fake_empty,
          ),
          mock.patch.object(
              multihost_utils,
              "process_allgather",
              wraps=multihost_utils.process_allgather,
          ) as spy_gather,
      ):
        self.assertEqual(bm.executor.get_xprof_durations("/tmp/fake"), [])
      spy_gather.assert_called_once()
      np.testing.assert_array_equal(
          spy_gather.call_args.args[0], np.array([1, 0], dtype=np.int32)
      )

    with self.subTest("local_exception"):
      with (
          mock.patch.object(
              jax, "local_devices", return_value=fake_devs
          ),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              side_effect=RuntimeError("corrupt trace"),
          ),
          mock.patch.object(
              multihost_utils,
              "process_allgather",
              wraps=multihost_utils.process_allgather,
          ) as spy_gather,
      ):
        self.assertEqual(bm.executor.get_xprof_durations("/tmp/fake"), [])
      spy_gather.assert_called_once()
      np.testing.assert_array_equal(
          spy_gather.call_args.args[0], np.array([1, 0], dtype=np.int32)
      )

    with self.subTest("remote_host_parse_failed"):
      with (
          mock.patch.object(
              jax, "local_devices", return_value=fake_devs
          ),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              return_value=[14.0, 22.0, 35.0],
          ),
          mock.patch.object(
              multihost_utils,
              "process_allgather",
              side_effect=[np.array([[0, 3], [1, 0]], dtype=np.int32)],
          ) as mock_gather,
      ):
        self.assertEqual(bm.executor.get_xprof_durations("/tmp/fake"), [])
      mock_gather.assert_called_once()

  def test_global_slowest_per_step_unequal_local_counts_raises_on_all_hosts(
      self,
  ):
    """Verifies UNEQUAL_LOCAL_COUNTS on any host raises ValueError on all hosts."""
    fake_devs = [
        mock.MagicMock(local_hardware_id=0),
        mock.MagicMock(local_hardware_id=1),
    ]
    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=3),
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            device_mode=constants.XprofDeviceMode.GLOBAL_SLOWEST_PER_STEP,
        ),
    )

    with self.subTest("local_unequal_device_counts"):

      def fake_unequal(trace_dir, fallback_fn=None, local_device_id=0):
        del trace_dir, fallback_fn
        return [10.0, 20.0, 30.0] if local_device_id == 0 else [12.0, 18.0]

      with (
          mock.patch.object(
              jax, "local_devices", return_value=fake_devs
          ),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              side_effect=fake_unequal,
          ),
          mock.patch.object(
              multihost_utils,
              "process_allgather",
              wraps=multihost_utils.process_allgather,
          ) as spy_gather,
      ):
        with self.assertRaisesRegex(
            ValueError, "Unequal XProf step counts.*statuses="
        ):
          bm.executor.get_xprof_durations("/tmp/fake")
      spy_gather.assert_called_once()
      np.testing.assert_array_equal(
          spy_gather.call_args.args[0], np.array([2, 0], dtype=np.int32)
      )

    with self.subTest("remote_unequal_device_counts"):
      with (
          mock.patch.object(
              jax, "local_devices", return_value=fake_devs
          ),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              return_value=[14.0, 22.0, 35.0],
          ),
          mock.patch.object(
              multihost_utils,
              "process_allgather",
              side_effect=[np.array([[0, 3], [2, 0]], dtype=np.int32)],
          ) as mock_gather,
      ):
        with self.assertRaisesRegex(
            ValueError, "Unequal XProf step counts.*statuses="
        ):
          bm.executor.get_xprof_durations("/tmp/fake")
      mock_gather.assert_called_once()

  def test_global_slowest_per_step_differing_step_counts_across_hosts_raises_on_all_hosts(
      self,
  ):
    """Verifies differing n_steps across OK hosts raises ValueError on all hosts."""
    fake_devs = [
        mock.MagicMock(local_hardware_id=0),
        mock.MagicMock(local_hardware_id=1),
    ]
    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=3),
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            device_mode=constants.XprofDeviceMode.GLOBAL_SLOWEST_PER_STEP,
        ),
    )

    with (
        mock.patch.object(
            jax, "local_devices", return_value=fake_devs
        ),
        mock.patch(
            "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
            return_value=[14.0, 22.0, 35.0],
        ),
        mock.patch.object(
            multihost_utils,
            "process_allgather",
            side_effect=[np.array([[0, 3], [0, 2]], dtype=np.int32)],
        ) as mock_gather,
    ):
      with self.assertRaisesRegex(
          ValueError, "Unequal XProf step counts.*step_counts=\\[3, 2\\]"
      ):
        bm.executor.get_xprof_durations("/tmp/fake")
    mock_gather.assert_called_once()

  def test_soaking_executor_global_slowest_per_step_both_windows_and_baseline_error_raises_immediately(
      self,
  ):
    """Verifies SoakingRunExecutor runs two-stage gathers on both windows and raises immediately on baseline ValueError."""
    fake_devs = [
        mock.MagicMock(local_hardware_id=0),
        mock.MagicMock(local_hardware_id=1),
    ]
    cfg = SoakingDummyParams(
        warmup_tries=0, min_duration_s=0.01, samples_per_run=2
    )
    bm = SoakingDummyBenchmark(
        config=cfg,
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            device_mode=constants.XprofDeviceMode.GLOBAL_SLOWEST_PER_STEP,
        ),
    )

    with self.subTest("both_windows_ok"):

      def fake_parse_both(trace_dir, fallback_fn=None, local_device_id=0):
        del fallback_fn
        if "baseline_window" in trace_dir:
          return [5.0, 6.0] if local_device_id == 0 else [5.5, 5.8]
        return [7.0, 8.0] if local_device_id == 0 else [7.5, 7.2]

      with (
          mock.patch.object(
              jax, "local_devices", return_value=fake_devs
          ),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              side_effect=fake_parse_both,
          ),
          mock.patch.object(
              multihost_utils,
              "process_allgather",
              side_effect=[
                  np.array([[0, 2], [0, 2]], dtype=np.int32),
                  np.array([[5.5, 6.0], [5.9, 5.7]], dtype=np.float32),
                  np.array([[0, 2], [0, 2]], dtype=np.int32),
                  np.array([[7.5, 8.0], [7.2, 8.5]], dtype=np.float32),
              ],
          ) as mock_gather,
      ):
        durations = bm.executor.get_xprof_durations("/tmp/fake")

      np.testing.assert_allclose(durations, [7.5, 8.5], rtol=1e-5)
      np.testing.assert_allclose(
          bm.executor.first_window_xprof_duration, [5.9, 6.0], rtol=1e-5
      )
      np.testing.assert_allclose(
          bm.executor.last_window_xprof_duration, [7.5, 8.5], rtol=1e-5
      )
      self.assertEqual(mock_gather.call_count, 4)

    with self.subTest("baseline_error_raises_immediately"):

      def fake_parse_baseline_bad(
          trace_dir, fallback_fn=None, local_device_id=0
      ):
        del fallback_fn
        if "baseline_window" in trace_dir:
          return [5.0, 6.0] if local_device_id == 0 else [5.5]
        return [7.0, 8.0]

      with (
          mock.patch.object(
              jax, "local_devices", return_value=fake_devs
          ),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              side_effect=fake_parse_baseline_bad,
          ),
          mock.patch.object(
              multihost_utils,
              "process_allgather",
              wraps=multihost_utils.process_allgather,
          ) as spy_gather,
      ):
        with self.assertRaisesRegex(ValueError, "Unequal XProf step counts"):
          bm.executor.get_xprof_durations("/tmp/fake")

      self.assertEqual(spy_gather.call_count, 1)

  def test_extract_local_device_slices_and_custom_agg_fn(self):
    """Verifies _extract_local_device_slices returns raw 2D slices and _parse_multihost_durations supports custom agg_fn."""
    fake_devs = [
        mock.MagicMock(local_hardware_id=0),
        mock.MagicMock(local_hardware_id=1),
    ]

    def fake_parse_xprof(trace_dir, fallback_fn=None, local_device_id=0):
      del trace_dir, fallback_fn
      if local_device_id == 0:
        return [10.0, 22.0, 30.0]
      return [14.0, 18.0, 35.0]

    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=3),
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            device_mode=constants.XprofDeviceMode.GLOBAL_SLOWEST_PER_STEP,
        ),
    )

    with (
        mock.patch.object(
            jax, "local_devices", return_value=fake_devs
        ),
        mock.patch(
            "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
            side_effect=fake_parse_xprof,
        ),
    ):
      status, slices = bm.executor._extract_local_device_slices("/tmp/fake")  # pylint: disable=protected-access
      self.assertEqual(status, executor._XprofParseStatus.OK)  # pylint: disable=protected-access
      self.assertEqual(slices, [[10.0, 22.0, 30.0], [14.0, 18.0, 35.0]])

      empty_status, empty_slices = bm.executor._extract_local_device_slices("")  # pylint: disable=protected-access
      self.assertEqual(empty_status, executor._XprofParseStatus.PARSE_FAILED)  # pylint: disable=protected-access
      self.assertEqual(empty_slices, [])

      with mock.patch.object(
          multihost_utils,
          "process_allgather",
          side_effect=[
              np.array([[0, 3], [0, 3]], dtype=np.int32),
              np.array(
                  [[10.0, 18.0, 30.0], [12.0, 16.0, 31.0]], dtype=np.float32
              ),
          ],
      ):
        min_durations = bm.executor._parse_multihost_durations(  # pylint: disable=protected-access
            "/tmp/fake", agg_fn=np.min
        )
      self.assertEqual(min_durations, [10.0, 16.0, 30.0])

  def test_parse_multihost_owner_durations(self):
    """Verifies _parse_multihost_owner_durations for owner, non-owner, empty, and error paths."""

    class MultihostSyncDummyBenchmark(DummyBenchmark):
      """Dummy benchmark with requires_multihost_sync=True."""

      @property
      def requires_multihost_sync(self) -> bool:
        return True

      def get_device_to_measure(self):
        mock_device = mock.Mock()
        mock_device.id = 0
        mock_device.process_index = 0
        mock_device.local_hardware_id = 0
        return mock_device

    bm = MultihostSyncDummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=3),
        xprof_config=base.XprofConfig(xprof_timing=True),
    )

    with self.subTest("owner_host_parses_and_broadcasts"):
      with (
          mock.patch("jax.process_index", return_value=0),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              return_value=[4.0, 4.1, 4.5],
          ) as mock_parse,
          mock.patch.object(
              multihost_utils,
              "broadcast_one_to_all",
              side_effect=lambda x, is_source=False: x,
          ) as mock_bcast,
      ):
        durations = bm.executor.get_xprof_durations("/tmp/fake")
      mock_parse.assert_called_once()
      self.assertEqual(mock_bcast.call_count, 2)
      np.testing.assert_allclose(durations, [4.0, 4.1, 4.5], rtol=1e-5)

    with self.subTest("non_owner_host_skips_parse_and_receives_broadcast"):
      with (
          mock.patch("jax.process_index", return_value=1),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
          ) as mock_parse,
          mock.patch.object(
              multihost_utils,
              "broadcast_one_to_all",
              side_effect=[
                  np.int32(3),
                  np.array([4.0, 4.1, 4.5], dtype=np.float32),
              ],
          ) as mock_bcast,
      ):
        durations = bm.executor.get_xprof_durations("/tmp/fake")
      mock_parse.assert_not_called()
      self.assertEqual(mock_bcast.call_count, 2)
      np.testing.assert_allclose(durations, [4.0, 4.1, 4.5], rtol=1e-5)

    with self.subTest("empty_trace_returns_empty"):
      with (
          mock.patch("jax.process_index", return_value=0),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              return_value=[],
          ),
          mock.patch.object(
              multihost_utils,
              "broadcast_one_to_all",
              return_value=np.int32(0),
          ) as mock_bcast,
      ):
        durations = bm.executor.get_xprof_durations("/tmp/fake")
      self.assertEqual(durations, [])
      mock_bcast.assert_called_once()

    with self.subTest("owner_parse_value_error_broadcasts_zero_steps"):
      with (
          mock.patch("jax.process_index", return_value=0),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              side_effect=ValueError("corrupt trace json"),
          ),
          mock.patch.object(
              multihost_utils,
              "broadcast_one_to_all",
              side_effect=lambda x, is_source=False: x,
          ) as mock_bcast,
      ):
        durations = bm.executor.get_xprof_durations("/tmp/fake")
      self.assertEqual(durations, [])
      mock_bcast.assert_called_once_with(np.int32(0), is_source=True)

    with self.subTest("broadcast_exception_falls_back_to_local"):
      with (
          mock.patch("jax.process_index", return_value=0),
          mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              return_value=[4.0, 4.1, 4.5],
          ),
          mock.patch.object(
              multihost_utils,
              "broadcast_one_to_all",
              side_effect=RuntimeError("broadcast failed"),
          ),
      ):
        durations = bm.executor.get_xprof_durations("/tmp/fake")
      self.assertEqual(durations, [4.0, 4.1, 4.5])


class PostRunProcessingTest(absltest.TestCase):
  """Tests for the untimed per-run hook BaseBenchmark.post_run_processing."""

  def test_called_once_per_timed_run_with_outputs_and_index(self):
    """Verifies each timed run, not warmup, calls the hook with its outputs."""
    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=2, num_runs=3)
    )
    real_run_op = bm.run_op
    run_outputs = []

    def recording_run_op(x):
      outputs = real_run_op(x)
      run_outputs.append(outputs)
      return outputs

    with (
        mock.patch.object(bm, "run_op", side_effect=recording_run_op),
        mock.patch.object(bm, "post_run_processing") as mock_hook,
    ):
      inputs = bm.executor.run_warmup(bm.generate_inputs())
      mock_hook.assert_not_called()
      run_outputs.clear()
      actual_runs = bm.executor.execute(
          inputs, local_xprof_dir=None
      ).actual_runs

    self.assertEqual(actual_runs, 3)
    self.assertLen(run_outputs, actual_runs)
    self.assertLen(mock_hook.call_args_list, actual_runs)
    for run_index, (call, outputs) in enumerate(
        zip(mock_hook.call_args_list, run_outputs)
    ):
      self.assertIs(call.args[0], outputs)
      self.assertEqual(call.kwargs, {"run_index": run_index})

  def test_excluded_from_wall_clock_timing(self):
    """Verifies time spent in the hook is not counted in raw_times."""
    hook_delay_s = 0.1
    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=1, num_runs=3)
    )
    # Warm up first so that the timed runs do not include op compilation.
    inputs = bm.executor.run_warmup(bm.generate_inputs())

    def slow_hook(*args, **kwargs):
      del args, kwargs
      time.sleep(hook_delay_s)

    with mock.patch.object(
        bm, "post_run_processing", side_effect=slow_hook
    ) as mock_hook:
      raw_times = bm.executor.execute(inputs, local_xprof_dir=None).raw_times_ms

    self.assertEqual(mock_hook.call_count, 3)
    self.assertLen(raw_times, 3)
    for run_ms in raw_times:
      self.assertLess(run_ms, hook_delay_s * 1000.0)


if __name__ == "__main__":
  absltest.main()
