"""Unit tests for BaseRunExecutor and SoakingRunExecutor."""

import contextlib
import dataclasses
import io
import os
import shutil
import tempfile
import time
import unittest

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import config as config_module
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
import jax
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
    self._platform_patcher = unittest.mock.patch(
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
        metrics[constants.WALL_CLOCK_INITIAL_P50_MS], 1.0, places=4
    )
    self.assertAlmostEqual(
        metrics[constants.WALL_CLOCK_SUSTAINED_P50_MS], 100.0, places=4
    )
    self.assertAlmostEqual(
        metrics[constants.WALL_CLOCK_SLOWDOWN_RATIO], 100.0, places=4
    )
    self.assertNotIn(constants.XPROF_INITIAL_P50_MS, metrics)
    self.assertNotIn(constants.XPROF_SUSTAINED_P50_MS, metrics)
    self.assertNotIn(constants.XPROF_SLOWDOWN_RATIO, metrics)

  def test_base_executor_warmup_and_iqr_boundary(self):
    """Verifies BaseRunExecutor warmup blocking and exact 1.5*IQR boundary."""
    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=2, num_runs=1)
    )
    with unittest.mock.patch.object(
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
        constants.WALL_CLOCK_INITIAL_P50_MS,
        constants.WALL_CLOCK_SUSTAINED_P50_MS,
        constants.WALL_CLOCK_SLOWDOWN_RATIO,
        *constants.THERMAL_METRIC_KEYS,
    ):
      self.assertIsNone(metrics[key])

  def test_base_benchmark_calculate_metrics_integrates_thermal_keys(self):
    """Verifies end-to-end integration of thermal keys in calculate_metrics."""
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "bbc8_thermal.xplane.pb"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      shutil.copy(data_path, os.path.join(tmpdir, "bbc8_thermal.xplane.pb"))
      bench = SoakingDummyBenchmark(
          config=SoakingDummyParams(min_duration_s=1.0)
      )
      bench._xprof_dir_actual = tmpdir  # pylint: disable=protected-access
      metrics = bench.calculate_metrics([10.0] * 50 + [20.0] * 50)

    for key in constants.THERMAL_METRIC_KEYS:
      self.assertIn(key, metrics)
    self.assertEqual(metrics["thermal_throttle_level"], 12.0)
    self.assertEqual(metrics["hbm_peak_temp_c"], 68.5)
    self.assertEqual(metrics["peak_temp_c"], 74.2)
    self.assertEqual(metrics["hbm_throttle_pct"], 64.1)

  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
      return_value=None,
  )
  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
      return_value=[1.0],
  )
  @unittest.mock.patch("jax.profiler.trace")
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

  def test_soaking_warmup_executes_exact_warmup_tries(self):
    """Verifies SoakingRunExecutor.run_warmup executes exact warmup_tries."""
    cfg = SoakingDummyParams(
        warmup_tries=10,
        min_duration_s=0.01,
        samples_per_run=25,
    )
    bm = SoakingDummyBenchmark(config=cfg)
    with unittest.mock.patch.object(
        bm, "run_op", wraps=bm.run_op
    ) as mock_run_op:
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
          unittest.mock.patch(
              "jax.profiler.trace",
              side_effect=fake_trace,
          ),
          unittest.mock.patch(
              "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
              side_effect=[[1.0], [2.5]],
          ),
          unittest.mock.patch(
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
    self.assertEqual(bm.executor.xprof_durations, [1.0, 2.5])
    self.assertAlmostEqual(res.metrics["xprof_initial_p50_ms"], 1.0, places=4)
    self.assertAlmostEqual(res.metrics["xprof_sustained_p50_ms"], 2.5, places=4)
    self.assertAlmostEqual(res.metrics["xprof_slowdown_ratio"], 2.5, places=4)

    # Also verify non-positive synced_p50 passes has_xprof_timings=False.
    with (
        unittest.mock.patch(
            "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
            side_effect=[[-1.0], [-1.0]],
        ),
        unittest.mock.patch(
            "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
            return_value=None,
        ),
    ):
      fallback_metrics = bm._apply_xprof_timing_and_sync({})  # pylint: disable=protected-access
    self.assertIsNone(fallback_metrics[constants.XPROF_INITIAL_P50_MS])
    self.assertIsNone(fallback_metrics[constants.XPROF_SUSTAINED_P50_MS])
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
          unittest.mock.patch.object(
              bm, "run_op", side_effect=recording_run_op
          ),
          unittest.mock.patch.object(
              jax, "block_until_ready", side_effect=recording_block
          ),
          unittest.mock.patch("jax.profiler.trace", side_effect=fake_trace),
      ):
        raw_times, actual_runs = bm.executor.execute(
            bm.generate_inputs(), tmpdir
        )

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
          unittest.mock.patch.object(
              jax, "block_until_ready", side_effect=slow_block
          ),
          unittest.mock.patch(
              "jax.profiler.trace", return_value=contextlib.nullcontext()
          ),
      ):
        raw_times, _ = bm.executor.execute(bm.generate_inputs(), tmpdir)

    self.assertNotEmpty(bm.executor.soak_wall_ms)
    for soak_ms in bm.executor.soak_wall_ms:
      self.assertGreaterEqual(soak_ms, delay_s * 1000.0)
    self.assertGreaterEqual(raw_times[0], delay_s * 1000.0)
    self.assertGreaterEqual(raw_times[-1], delay_s * 1000.0)

  def test_wall_clock_soaking_stats_derived_from_untraced_soak_chunks(self):
    """Verifies wall-clock initial/sustained p50 use first vs last soak chunk and ignore Phase 1/3 windows.

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
        metrics[constants.WALL_CLOCK_INITIAL_P50_MS], 10.0, places=4
    )
    self.assertAlmostEqual(
        metrics[constants.WALL_CLOCK_SUSTAINED_P50_MS], 12.0, places=4
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
    fake_devs = [unittest.mock.MagicMock(local_hardware_id=i) for i in range(8)]

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
          unittest.mock.patch.object(
              jax, "local_devices", return_value=fake_devs
          ),
          unittest.mock.patch(
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
    fake_devs = [unittest.mock.MagicMock(local_hardware_id=i) for i in range(8)]

    def fake_parse_xprof(trace_dir, fallback_fn=None, local_device_id=0):
      del trace_dir, fallback_fn
      if local_device_id in (4, 5):
        return [6.021, 6.019]
      return [5.009, 5.009]

    with (
        unittest.mock.patch.object(
            jax, "local_devices", return_value=fake_devs
        ),
        unittest.mock.patch(
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

  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
      return_value=None,
  )
  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
      return_value=[1.0],
  )
  @unittest.mock.patch("jax.profiler.trace")
  def test_profiler_options_isolated_to_execution_control_mixin(
      self, mock_trace, mock_parse, mock_upload
  ):
    """Verifies standard benchmarks omit profiler_options while soaking enables them."""
    del mock_parse, mock_upload
    mock_trace.return_value = contextlib.nullcontext()

    # 0. Standard benchmark with xprof_timing=False or local_xprof_dir=None
    # does not invoke trace (even when only one condition is false).
    no_trace_bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=1),
        xprof_config=base.XprofConfig(xprof_timing=False),
    )
    no_trace_bm.run()
    mock_trace.assert_not_called()
    no_trace_bm.executor.execute(
        no_trace_bm.generate_inputs(), "/tmp/std_xprof"
    )
    mock_trace.assert_not_called()

    # 1. Standard non-mixin benchmark (DummyBenchmark)
    std_bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=1),
        xprof_config=base.XprofConfig(
            xprof_timing=True, xprof_dir="/tmp/std_xprof"
        ),
    )
    std_bm.executor.execute(std_bm.generate_inputs(), None)
    mock_trace.assert_not_called()
    std_bm.run()
    mock_trace.assert_called_once_with(
        unittest.mock.ANY, create_perfetto_link=False
    )
    self.assertNotIn("profiler_options", mock_trace.call_args.kwargs)

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
    opts = mock_trace.call_args.kwargs.get("profiler_options")
    self.assertIsNotNone(opts)
    self.assertEqual(opts.host_tracer_level, 0)
    self.assertTrue(
        opts.advanced_configuration["tpu_e2e_enable_fw_thermal_event"]
    )
    self.assertEqual(opts.advanced_configuration["tpu_power_trace_level"], 1)
    self.assertFalse(opts.advanced_configuration["tpu_perf_counters"])


if __name__ == "__main__":
  absltest.main()
