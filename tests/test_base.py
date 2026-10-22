"""Unit tests for the BaseBenchmark execution loop on CPU."""

import contextlib
import csv
import dataclasses
import enum
import io
import json
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
import jax
import jax.numpy as jnp
import numpy as np


# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class _Color(constants.ParamEnum):
  """Sample ParamEnum used to exercise the shared base class."""

  RED = "red"
  GREEN = "green"


class DummyBenchmark(base.BaseBenchmark):
  """A dummy benchmark for testing the BaseBenchmark class."""

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


class BaseBenchmarkParamsTest(absltest.TestCase):
  """Tests for BaseBenchmarkParams configuration and expansion contract."""

  def test_expand_test_cases_default_identity(self):
    """Verifies default 1-to-1 identity mapping: returns [self]."""
    params = base.BaseBenchmarkParams(warmup_tries=5, num_runs=20)
    cases = params.expand_test_cases()
    self.assertLen(cases, 1)
    self.assertIs(cases[0], params)
    self.assertEqual(cases[0].warmup_tries, 5)
    self.assertEqual(cases[0].num_runs, 20)

  def test_expand_test_cases_subclass_inheritance(self):
    """Verifies subclasses without override inherit the [self] identity contract."""

    @dataclasses.dataclass
    class CustomConfig(base.BaseBenchmarkParams):
      custom_flag: str = "test"

    cfg = CustomConfig(warmup_tries=3, custom_flag="hello")
    cases = cfg.expand_test_cases()
    self.assertLen(cases, 1)
    self.assertIs(cases[0], cfg)
    self.assertEqual(cases[0].custom_flag, "hello")

  def test_warmup_tries_negative_raises_error(self):
    """Verifies that warmup_tries < 0 raises ValueError."""
    with self.assertRaises(ValueError):
      base.BaseBenchmarkParams(warmup_tries=-5)

  def test_num_runs_negative_raises_error(self):
    """Verifies that num_runs < 0 raises ValueError."""
    with self.assertRaises(ValueError):
      base.BaseBenchmarkParams(num_runs=-2)

  def test_min_duration_s_negative_raises_error(self):
    """Verifies that min_duration_s < 0 raises ValueError."""
    with self.assertRaises(ValueError):
      base.BaseBenchmarkParams(min_duration_s=-2.0)

  def test_custom_min_max_and_enum_validation(self):
    """Verifies min, max, and enum validation on custom subclass."""

    class SampleEnum(enum.Enum):
      ALPHA = "alpha"
      BETA = "beta"

    @dataclasses.dataclass
    class CustomParams(base.BaseBenchmarkParams):
      score: int = dataclasses.field(
          default=5,
          metadata={"min": 1, "max": 10},
      )
      mode: SampleEnum = dataclasses.field(default=SampleEnum.ALPHA)

    # Valid string enum conversion
    cfg = CustomParams(score=8, mode="beta")
    self.assertEqual(cfg.score, 8)
    self.assertEqual(cfg.mode, SampleEnum.BETA)

    # Below min
    with self.assertRaisesRegex(ValueError, "score must be >= 1"):
      CustomParams(score=0)

    # Above max
    with self.assertRaisesRegex(ValueError, "score must be <= 10"):
      CustomParams(score=11)

    # Invalid enum
    with self.assertRaisesRegex(ValueError, "Invalid mode 'gamma'"):
      CustomParams(mode="gamma")

    # Test _validate_bounds() independently
    cfg_direct = CustomParams(score=5)
    cfg_direct.score = 0
    with self.assertRaisesRegex(ValueError, "score must be >= 1"):
      cfg_direct._validate_bounds()  # pylint: disable=protected-access

    # Test _coerce_enums() independently
    cfg_direct.score = 5
    cfg_direct.mode = "gamma"
    with self.assertRaisesRegex(ValueError, "Invalid mode 'gamma'"):
      cfg_direct._coerce_enums()  # pylint: disable=protected-access


class ParamEnumTest(absltest.TestCase):
  """Tests for the shared ParamEnum base class."""

  def test_supported_options_str(self):
    """Verifies values are joined in declaration order."""
    self.assertEqual(_Color.supported_options_str(), "red, green")

  def test_member_is_str(self):
    """Verifies members interpolate as their value, without a `__str__`."""
    self.assertEqual(f"{_Color.RED}", "red")
    self.assertEqual(_Color.RED, "red")

  def test_coerces_from_raw_string(self):
    """Verifies a ParamEnum field is coerced by BaseBenchmarkParams."""

    @dataclasses.dataclass
    class ColorParams(base.BaseBenchmarkParams):
      color: _Color = dataclasses.field(default=_Color.RED)

    cfg = ColorParams(color="green")
    self.assertIs(cfg.color, _Color.GREEN)


class BaseBenchmarkTest(absltest.TestCase):
  """Tests for the BaseBenchmark class."""

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

  def test_base_benchmark_default_roofline_mode(self):
    """Verifies BaseBenchmark defaults to RooflineMode.NONE."""
    self.assertEqual(
        base.BaseBenchmark.roofline_mode, constants.RooflineMode.NONE
    )
    bm = DummyBenchmark()
    self.assertEqual(bm.roofline_mode, constants.RooflineMode.NONE)

  def test_calculate_metrics_iqr(self):
    bm = DummyBenchmark()
    # Deliberately introduce outliers
    times_ms = [10.0, 11.0, 10.5, 9.5, 100.0, 0.1, 10.2]
    metrics = bm.calculate_metrics(times_ms)

    # 100.0 and 0.1 should be filtered out by IQR
    # Left with [10.0, 11.0, 10.5, 9.5, 10.2] -> mean should be ~10.24
    self.assertGreater(metrics["wall_clock_avg_ms"], 9.0)
    self.assertLess(metrics["wall_clock_avg_ms"], 12.0)

  def test_init_without_mesh(self):
    bm = DummyBenchmark()
    self.assertIsNone(bm.mesh)
    self.assertEqual(bm.config.warmup_tries, 10)
    self.assertEqual(bm.config.num_runs, 10)
    self.assertIsNone(bm._jit_fn)  # pylint: disable=protected-access

  def test_init_with_mesh(self):
    mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    bm = DummyBenchmark(mesh=mock_mesh)
    self.assertEqual(bm.mesh, mock_mesh)

  def test_create_default_mesh(self):
    bm = DummyBenchmark()
    mesh = bm._create_default_mesh()  # pylint: disable=protected-access
    self.assertIsInstance(mesh, jax.sharding.Mesh)
    self.assertEqual(mesh.axis_names, ("device",))

  def test_calculate_metrics_empty(self):
    bm = DummyBenchmark()
    metrics = bm.calculate_metrics([])
    self.assertEqual(metrics["wall_clock_avg_ms"], 0.0)

  def test_run_orchestration(self):
    """Tests the full run orchestration of the BaseBenchmark."""

    params = {
        "warmup_tries": 2,
        "num_runs": 5,
        "dtype": "float32",
    }

    config = base.SingleDtypeBenchmarkParams(**params)
    hw_spec = system.get_hardware_spec(system.TpuVersion.TPU7X)
    bm = DummyBenchmark(config=config, hardware_spec=hw_spec)
    result = bm.run()

    self.assertEqual(result.metadata.benchmark_name, "DummyBenchmark")
    self.assertEqual(result.metrics["actual_runs"], 5)
    self.assertIsNotNone(result.metadata.platform_info)
    self.assertEqual(result.metadata.hardware_spec, hw_spec)
    self.assertIsNone(result.metadata.xprof_config)
    # XProf run counts are omitted entirely when xprof_timing is disabled.
    self.assertNotIn("xprof_captured_runs", result.metrics)
    self.assertNotIn("xprof_expected_runs", result.metrics)

    # Validate Roofline values are NOT computed by default for NONE mode
    self.assertNotIn("roofline_tflops_limit_per_device", result.metrics)
    self.assertNotIn(
        "wall_clock_compute_roofline_efficiency_pct", result.metrics
    )
    self.assertNotIn("peak_hbm_bw_per_device_gb_s", result.metrics)
    self.assertNotIn(
        "wall_clock_memory_roofline_efficiency_pct", result.metrics
    )

  def test_default_get_arithmetic_intensity_raises_not_implemented_error(self):
    """Tests that BaseBenchmark.get_arithmetic_intensity raises NotImplementedError by default."""
    bm = DummyBenchmark()
    with self.assertRaisesRegex(
        NotImplementedError,
        "DummyBenchmark must implement get_arithmetic_intensity",
    ):
      bm.get_arithmetic_intensity()

  def test_run_orchestration_with_compute_mode(self):
    """Tests run orchestration when roofline_mode is COMPUTE."""

    class ComputeDummyBenchmark(DummyBenchmark):
      roofline_mode = constants.RooflineMode.COMPUTE

      def get_arithmetic_intensity(self) -> float:
        return 1.0

      def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
        metrics = super().calculate_metrics(times_ms)
        metrics["wall_clock_tflops_per_device"] = 50.0
        return metrics

    params = {
        "warmup_tries": 2,
        "num_runs": 5,
        "dtype": "float32",
    }
    config = base.SingleDtypeBenchmarkParams(**params)
    hw_spec = system.get_hardware_spec(system.TpuVersion.TPU7X)
    bm = ComputeDummyBenchmark(config=config, hardware_spec=hw_spec)
    result = bm.run()

    self.assertIn("roofline_tflops_limit_per_device", result.metrics)
    self.assertIn("wall_clock_compute_roofline_efficiency_pct", result.metrics)
    self.assertIn("peak_hbm_bw_per_device_gb_s", result.metrics)
    self.assertNotIn(
        "wall_clock_memory_roofline_efficiency_pct", result.metrics
    )

  @mock.patch("jax.profiler.trace")
  def test_xprof_naming_with_identifier(self, mock_trace):
    class BenchmarkWithId(DummyBenchmark):

      def get_run_identifier(self) -> str:
        return "test_id_42"

    mock_trace.return_value = contextlib.nullcontext()

    params = {
        "warmup_tries": 1,
        "num_runs": 1,
    }

    config = base.BaseBenchmarkParams(**params)
    xprof_config = base.XprofConfig(
        xprof_timing=True, xprof_dir="/tmp/test_xprof"
    )
    bm = BenchmarkWithId(config=config, xprof_config=xprof_config)
    bm.run()

    mock_trace.assert_called_once()
    called_path = mock_trace.call_args[0][0]
    self.assertIn("BenchmarkWithId_test_id_42_", called_path)
    self.assertTrue(called_path.startswith("/tmp/test_xprof/"))

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  def test_run_with_xprof_timing(
      self, mock_trace, mock_upload, mock_parse_durations
  ):
    """Tests that benchmark uses XProf durations for derived metrics only."""
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = "http://mock_xprof_url"
    # Mock XProf durations to be exactly 5.0 ms
    mock_parse_durations.return_value = [5.0, 5.0, 5.0, 5.0, 5.0]

    params = {
        "warmup_tries": 1,
        "num_runs": 5,
    }

    config = base.BaseBenchmarkParams(**params)
    xprof_config = base.XprofConfig(
        xprof_timing=True, xprof_dir="/tmp/test_xprof"
    )
    bm = DummyBenchmark(config=config, xprof_config=xprof_config)
    result = bm.run()

    # Assert that the base metrics retain the host timings (not 5.0 ms)
    self.assertNotEqual(result.metrics.get("wall_clock_avg_ms"), 5.0)
    self.assertNotEqual(result.metrics.get("wall_clock_p50_ms"), 5.0)

    # Assert XProf metrics correctly reflect the mocked XProf durations
    self.assertEqual(result.metrics["xprof_avg_ms"], 5.0)
    self.assertEqual(result.metrics["xprof_url"], "http://mock_xprof_url")
    self.assertEqual(result.metadata.xprof_config, xprof_config)

    mock_upload.assert_called_once()
    mock_parse_durations.assert_called_once()

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  def test_apply_xprof_timing_and_sync_host_cpu_override(
      self, mock_trace, mock_upload, mock_parse_durations
  ):
    """Tests that xprof_target_host_cpu=True passes local_device_id=None."""

    class HostCpuBenchmark(DummyBenchmark):

      @property
      def xprof_target_host_cpu(self) -> bool:
        return True

    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = "http://mock_xprof_url"
    mock_parse_durations.return_value = [2.0]

    params = {
        "warmup_tries": 1,
        "num_runs": 1,
    }
    config = base.BaseBenchmarkParams(**params)
    xprof_config = base.XprofConfig(
        xprof_timing=True, xprof_dir="/tmp/test_xprof"
    )
    bm = HostCpuBenchmark(config=config, xprof_config=xprof_config)
    bm.run()

    mock_parse_durations.assert_called_once()
    self.assertIsNone(
        mock_parse_durations.call_args.kwargs.get("local_device_id")
    )

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  def test_apply_xprof_timing_and_sync_empty_durations_fallback(
      self, mock_trace, mock_upload, mock_parse_durations
  ):
    """Tests that empty XProf durations set xprof metrics to None while keeping host timings."""
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = None
    mock_parse_durations.return_value = []  # Empty durations

    params = {
        "warmup_tries": 1,
        "num_runs": 1,
    }
    config = base.BaseBenchmarkParams(**params)
    xprof_config = base.XprofConfig(
        xprof_timing=True, xprof_dir="/tmp/test_xprof"
    )
    bm = DummyBenchmark(config=config, xprof_config=xprof_config)
    result = bm.run()

    mock_parse_durations.assert_called_once()
    self.assertIsNone(result.metrics["xprof_avg_ms"])
    self.assertIsNone(result.metrics["xprof_p50_ms"])
    self.assertGreater(result.metrics["wall_clock_avg_ms"], 0.0)
    self.assertEqual(result.metrics["xprof_captured_runs"], 0)
    self.assertEqual(result.metrics["xprof_expected_runs"], 1)

  @mock.patch("jax.experimental.multihost_utils.broadcast_one_to_all")
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  @mock.patch("jax.process_index")
  def test_apply_xprof_timing_and_sync_with_multihost_sync(
      self,
      mock_process_index,
      mock_trace,
      mock_upload,
      mock_parse_durations,
      mock_broadcast,
  ):
    """Tests requires_multihost_sync=True (e.g., asymmetric D2D): non-owner hosts use broadcast metrics."""

    class MultihostBenchmark(DummyBenchmark):

      @property
      def requires_multihost_sync(self) -> bool:
        return True

      def get_device_to_measure(self):
        mock_device = mock.Mock()
        mock_device.id = 0
        mock_device.process_index = 0  # Host 0 is the owner
        mock_device.local_hardware_id = 0
        return mock_device

    mock_process_index.return_value = 1  # Current host is Host 1 (non-owner)
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = "http://mock_xprof_url"
    mock_broadcast.side_effect = [
        np.int32(3),
        np.array([4.0, 4.1, 4.5], dtype=np.float32),
    ]

    params = {
        "warmup_tries": 1,
        "num_runs": 3,
    }
    config = base.BaseBenchmarkParams(**params)
    xprof_config = base.XprofConfig(
        xprof_timing=True, xprof_dir="/tmp/test_xprof"
    )
    bm = MultihostBenchmark(config=config, xprof_config=xprof_config)
    result = bm.run()

    # When requires_multihost_sync=True (e.g. Device-to-Device asymmetric
    # benchmarks), only the owner host parses the trace; non-owner hosts
    # skip local trace parsing and use broadcast durations.
    mock_parse_durations.assert_not_called()
    self.assertEqual(mock_broadcast.call_count, 2)
    self.assertAlmostEqual(result.metrics["xprof_avg_ms"], 4.2, places=4)
    self.assertAlmostEqual(result.metrics["xprof_p50_ms"], 4.1, places=4)
    # Captured run count is derived from the broadcast owner durations
    # (the non-owner's local parse is skipped).
    self.assertEqual(result.metrics["xprof_captured_runs"], 3)

    with self.subTest("value_error_propagates_in_first_device_mode"):
      bm_first = DummyBenchmark(config=config, xprof_config=xprof_config)
      with mock.patch.object(
          bm_first.executor,
          "get_xprof_durations",
          side_effect=ValueError("unequal steps"),
      ):
        with self.assertRaisesRegex(ValueError, "unequal steps"):
          bm_first.run()

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  @mock.patch("jax.process_index")
  def test_apply_xprof_timing_and_sync_without_multihost_sync(
      self,
      mock_process_index,
      mock_trace,
      mock_upload,
      mock_parse_durations,
  ):
    """Tests requires_multihost_sync=False (e.g., Collectives): each host parses its trace locally."""

    class MultihostBenchmark(DummyBenchmark):

      @property
      def requires_multihost_sync(self) -> bool:
        # Each host parses its local trace without cross-host sync
        return False

    mock_process_index.return_value = 1  # Current host is Host 1
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = "http://mock_xprof_url"
    mock_parse_durations.return_value = [3.5, 3.5, 3.5]

    params = {
        "warmup_tries": 1,
        "num_runs": 3,
    }
    config = base.BaseBenchmarkParams(**params)
    xprof_config = base.XprofConfig(
        xprof_timing=True, xprof_dir="/tmp/test_xprof"
    )
    bm = MultihostBenchmark(config=config, xprof_config=xprof_config)
    result = bm.run()

    # When requires_multihost_sync=False (e.g., Collective symmetric
    # benchmarks), even though hosts exchange data during execution, each host
    # parses its own local XProf trace for its local device without invoking
    # cross-host metric broadcast.
    mock_parse_durations.assert_called_once()
    self.assertEqual(result.metrics["xprof_avg_ms"], 3.5)
    self.assertEqual(result.metrics["xprof_captured_runs"], 3)
    self.assertEqual(result.metrics["xprof_expected_runs"], 3)

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  def test_xprof_truncation_recorded_and_warned(
      self, mock_trace, mock_upload, mock_parse_durations
  ):
    """Tests captured < expected runs are recorded and a warning is printed."""
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = None
    mock_parse_durations.return_value = [2.0, 2.0, 2.0]

    config = base.BaseBenchmarkParams(warmup_tries=1, num_runs=10)
    xprof_config = base.XprofConfig(
        xprof_timing=True, xprof_dir="/tmp/test_xprof"
    )
    bm = DummyBenchmark(config=config, xprof_config=xprof_config)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      result = bm.run()

    self.assertEqual(result.metrics["actual_runs"], 10)
    self.assertEqual(result.metrics["xprof_captured_runs"], 3)
    self.assertEqual(result.metrics["xprof_expected_runs"], 10)
    self.assertIn("Warning: XProf captured 3/10 traced runs", buf.getvalue())

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  def test_xprof_zero_captured_runs_not_reported_as_truncation(
      self, mock_trace, mock_upload
  ):
    """Tests parse failures / no events warn as 'no durations', not truncation."""
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = None
    xprof_config = base.XprofConfig(
        xprof_timing=True, xprof_dir="/tmp/test_xprof"
    )

    for case_name, parse_kwargs in (
        ("parse_exception", {"side_effect": RuntimeError("bad trace")}),
        ("no_matching_events", {"return_value": []}),
    ):
      with self.subTest(case_name):
        bm = DummyBenchmark(
            config=base.BaseBenchmarkParams(warmup_tries=1, num_runs=4),
            xprof_config=xprof_config,
        )
        buf = io.StringIO()
        with (
            mock.patch(
                "accelerator_microbenchmarks.core.profiler.parse_xprof_durations",
                **parse_kwargs,
            ),
            contextlib.redirect_stdout(buf),
        ):
          result = bm.run()

        self.assertEqual(result.metrics["xprof_captured_runs"], 0)
        self.assertEqual(result.metrics["xprof_expected_runs"], 4)
        self.assertIsNone(result.metrics["xprof_p50_ms"])
        self.assertIn(
            "Warning: No XProf durations parsed for 4 traced runs",
            buf.getvalue(),
        )
        self.assertNotIn("truncated trace", buf.getvalue())

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  def test_xprof_captured_more_than_expected_raises(
      self, mock_trace, mock_upload, mock_parse_durations
  ):
    """Tests more parsed durations than traced runs raises instead of passing."""
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = None
    mock_parse_durations.return_value = [2.0] * 5

    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=1, num_runs=2),
        xprof_config=base.XprofConfig(
            xprof_timing=True, xprof_dir="/tmp/test_xprof"
        ),
    )
    with self.assertRaisesRegex(
        RuntimeError, r"XProf parsed 5 durations for 2 traced runs"
    ):
      bm.run()

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  def test_xprof_run_counts_saved_to_summary_csv_and_detailed_json(
      self, mock_trace, mock_upload, mock_parse_durations
  ):
    """Tests xprof run counts are exported to summary.csv and detailed.json."""
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = None
    mock_parse_durations.return_value = [2.0, 2.0]

    config = base.BaseBenchmarkParams(warmup_tries=1, num_runs=4)
    xprof_config = base.XprofConfig(
        xprof_timing=True, xprof_dir="/tmp/test_xprof"
    )
    result = DummyBenchmark(config=config, xprof_config=xprof_config).run()

    output_dir = self.create_tempdir().full_path
    report.save_output([result], output_dir=output_dir)

    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, newline="") as f:
      summary_rows = list(csv.DictReader(f))
    self.assertLen(summary_rows, 1)
    self.assertEqual(summary_rows[0]["xprof_captured_runs"], "2")
    self.assertEqual(summary_rows[0]["xprof_expected_runs"], "4")
    with open(os.path.join(output_dir, "detailed.json")) as f:
      detailed = json.load(f)
    self.assertEqual(detailed[0]["metrics"]["xprof_captured_runs"], 2)
    self.assertEqual(detailed[0]["metrics"]["xprof_expected_runs"], 4)

  @mock.patch("jax.experimental.multihost_utils.broadcast_one_to_all")
  @mock.patch("jax.experimental.multihost_utils.process_allgather")
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  @mock.patch("jax.process_index")
  def test_run_with_global_slowest_per_step_computes_identical_metrics_and_bandwidth_on_non_owner_host(
      self,
      mock_process_index,
      mock_trace,
      mock_upload,
      mock_parse_durations,
      mock_allgather,
      mock_broadcast,
  ):
    """Verifies global_slowest_per_step runs on all hosts, skips broadcast, and derives bandwidth from p50."""

    class MultihostCollectiveDummyBenchmark(DummyBenchmark):
      """Dummy benchmark requiring multihost sync with bandwidth metrics."""

      @property
      def requires_multihost_sync(self) -> bool:
        return True

      def get_device_to_measure(self):
        mock_device = mock.Mock()
        mock_device.id = 0
        mock_device.process_index = 0
        mock_device.local_hardware_id = 0
        return mock_device

      def calculate_throughput_metrics(
          self, latency_ms: float, prefix: constants.TimingDomain
      ):
        return {f"{prefix}_bandwidth_per_chip_gb_s": 1000.0 / latency_ms}

    fake_devs = [
        mock.MagicMock(local_hardware_id=0),
        mock.MagicMock(local_hardware_id=1),
    ]
    mock_process_index.return_value = 1  # Non-owner host
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = None
    mock_parse_durations.side_effect = (
        lambda _dir, _fn=None, local_device_id=0: [8.0, 20.0, 28.0]
        if local_device_id == 0
        else [10.0, 18.0, 30.0]
    )
    mock_allgather.side_effect = [
        np.array([[0, 3], [0, 3]], dtype=np.int32),
        np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 20.0]], dtype=np.float32),
    ]

    bm = MultihostCollectiveDummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=3),
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            xprof_dir="/tmp/test_xprof",
            device_mode=constants.XprofDeviceMode.GLOBAL_SLOWEST_PER_STEP,
        ),
    )
    with mock.patch.object(
        jax, "local_devices", return_value=fake_devs
    ):
      result = bm.run()

    mock_broadcast.assert_not_called()
    self.assertEqual(mock_allgather.call_count, 2)
    self.assertAlmostEqual(result.metrics["xprof_p50_ms"], 25.0, places=4)
    self.assertAlmostEqual(result.metrics["xprof_avg_ms"], 70.0 / 3.0, places=4)
    self.assertAlmostEqual(
        result.metrics["xprof_bandwidth_per_chip_gb_s"], 40.0, places=4
    )

  @mock.patch("jax.experimental.multihost_utils.process_allgather")
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  def test_run_with_global_slowest_per_step_parse_failed_emits_none_metrics_across_hosts(
      self,
      mock_trace,
      mock_upload,
      mock_parse_durations,
      mock_allgather,
  ):
    """Verifies PARSE_FAILED status on any host sets all XProf latency and bandwidth metrics to None."""

    class BandwidthDummyBenchmark(DummyBenchmark):

      def calculate_throughput_metrics(
          self, latency_ms: float, prefix: constants.TimingDomain
      ):
        return {f"{prefix}_bandwidth_per_chip_gb_s": 1000.0 / latency_ms}

    fake_devs = [mock.MagicMock(local_hardware_id=0)]
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = None
    mock_parse_durations.return_value = [15.0, 25.0, 30.0]
    mock_allgather.side_effect = [np.array([[0, 3], [1, 0]], dtype=np.int32)]

    bm = BandwidthDummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=3),
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            xprof_dir="/tmp/test_xprof",
            device_mode=constants.XprofDeviceMode.GLOBAL_SLOWEST_PER_STEP,
        ),
    )
    with mock.patch.object(
        jax, "local_devices", return_value=fake_devs
    ):
      result = bm.run()

    self.assertIsNone(result.metrics["xprof_p50_ms"])
    self.assertIsNone(result.metrics["xprof_avg_ms"])
    self.assertIsNone(result.metrics["xprof_p90_ms"])
    self.assertIsNone(result.metrics["xprof_std_ms"])
    self.assertIsNone(result.metrics["xprof_bandwidth_per_chip_gb_s"])

  @mock.patch("jax.experimental.multihost_utils.process_allgather")
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @mock.patch("jax.profiler.trace")
  def test_run_with_global_slowest_per_step_propagates_unequal_step_count_value_error(
      self,
      mock_trace,
      mock_upload,
      mock_parse_durations,
      mock_allgather,
  ):
    """Verifies _apply_xprof_timing_and_sync propagates ValueError under GLOBAL_SLOWEST_PER_STEP."""
    fake_devs = [mock.MagicMock(local_hardware_id=0)]
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = None
    mock_parse_durations.return_value = [15.0, 25.0, 30.0]
    mock_allgather.side_effect = [np.array([[0, 3], [0, 2]], dtype=np.int32)]

    bm = DummyBenchmark(
        config=base.BaseBenchmarkParams(warmup_tries=0, num_runs=3),
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            xprof_dir="/tmp/test_xprof",
            device_mode=constants.XprofDeviceMode.GLOBAL_SLOWEST_PER_STEP,
        ),
    )
    with mock.patch.object(
        jax, "local_devices", return_value=fake_devs
    ):
      with self.assertRaisesRegex(ValueError, "Unequal XProf step counts"):
        bm.run()

  def test_xprof_config_immutability(self):
    """Verifies that XprofConfig is frozen and immutable."""
    cfg = base.XprofConfig(xprof_timing=True, xprof_dir="/tmp/test")
    with self.assertRaises(dataclasses.FrozenInstanceError):
      cfg.xprof_timing = False

  def test_xprof_config_disabled_metadata_none(self):
    """Verifies that xprof_timing=False results in xprof_config is None."""
    xprof_config = base.XprofConfig(xprof_timing=False, xprof_dir="/tmp/test")
    bm = DummyBenchmark(xprof_config=xprof_config)
    result = bm.run()
    self.assertIsNone(result.metadata.xprof_config)

  def test_base_benchmark_default_xprof_config(self):
    """Verifies that BaseBenchmark defaults xprof_config to XprofConfig instance."""
    bm = DummyBenchmark()
    self.assertIsInstance(bm.xprof_config, base.XprofConfig)
    self.assertFalse(bm.xprof_config.xprof_timing)
    self.assertEqual(bm.xprof_config.xprof_dir, "/tmp/tensorboard")

  def test_run_metadata_env_flags(self):
    """Tests BaseBenchmark.run() captures runtime flags or defaults to empty."""
    # When flags are populated in os.environ
    env_vars = {
        "XLA_FLAGS": "--xla_test_flag=1",
        "LIBTPU_INIT_ARGS": "--tpu_test_arg=2",
    }
    with mock.patch.dict(os.environ, env_vars, clear=False):
      bm = DummyBenchmark()
      result = bm.run()
      self.assertEqual(result.metadata.xla_flags, "--xla_test_flag=1")
      self.assertEqual(result.metadata.libtpu_init_args, "--tpu_test_arg=2")

    # When flags are unset in os.environ
    with mock.patch.dict(os.environ):
      os.environ.pop("XLA_FLAGS", None)
      os.environ.pop("LIBTPU_INIT_ARGS", None)
      bm = DummyBenchmark()
      result = bm.run()
      self.assertEqual(result.metadata.xla_flags, "")
      self.assertEqual(result.metadata.libtpu_init_args, "")

  def test_derive_chip_metrics_with_chip_bandwidth_enabled(self):
    """Tests derive_chip_metrics derives chip bandwidth when derive_chip_bandwidth=True."""
    hw_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        devices_per_chip=2,
    )
    bm = DummyBenchmark(hardware_spec=hw_spec)
    self.assertTrue(bm.derive_chip_bandwidth)

    metrics = {"wall_clock_bandwidth_per_device_gb_s": 100.0}
    derived = bm.derive_chip_metrics(metrics)
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", derived)
    self.assertIn("wall_clock_bandwidth_per_chip_gb_s", derived)
    self.assertEqual(derived["wall_clock_bandwidth_per_chip_gb_s"], 200.0)

    # If wall_clock_bandwidth_per_device_gb_s is None, wall_clock_bandwidth_per_chip_gb_s is also None
    metrics_none = {"wall_clock_bandwidth_per_device_gb_s": None}
    derived_none = bm.derive_chip_metrics(metrics_none)
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", derived_none)
    self.assertIn("wall_clock_bandwidth_per_chip_gb_s", derived_none)
    self.assertIsNone(derived_none["wall_clock_bandwidth_per_chip_gb_s"])

    # If wall_clock_bandwidth_per_chip_gb_s is already present, it is not overwritten
    metrics_existing = {
        "wall_clock_bandwidth_per_device_gb_s": 100.0,
        "wall_clock_bandwidth_per_chip_gb_s": 300.0,
    }
    derived_existing = bm.derive_chip_metrics(metrics_existing)
    self.assertEqual(
        derived_existing["wall_clock_bandwidth_per_chip_gb_s"], 300.0
    )

  def test_derive_chip_metrics_with_chip_bandwidth_disabled(self):
    """Tests derive_chip_metrics does NOT derive chip bandwidth when derive_chip_bandwidth=False."""

    class NoChipBwBenchmark(DummyBenchmark):
      derive_chip_bandwidth = False

    hw_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        devices_per_chip=2,
    )
    bm = NoChipBwBenchmark(hardware_spec=hw_spec)
    self.assertFalse(bm.derive_chip_bandwidth)

    metrics = {"wall_clock_bandwidth_per_device_gb_s": 100.0}
    derived = bm.derive_chip_metrics(metrics)
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", derived)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", derived)

  def test_derive_chip_metrics_tflops(self):
    """Tests derive_chip_metrics derives tflops_per_chip from tflops_per_device."""
    hw_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        devices_per_chip=2,
    )
    bm = DummyBenchmark(hardware_spec=hw_spec)

    metrics = {"wall_clock_tflops_per_device": 150.0}
    derived = bm.derive_chip_metrics(metrics)
    self.assertIn("wall_clock_tflops_per_device", derived)
    self.assertIn("wall_clock_tflops_per_chip", derived)
    self.assertEqual(derived["wall_clock_tflops_per_chip"], 300.0)

    # If wall_clock_tflops_per_device is None, wall_clock_tflops_per_chip is also None
    metrics_none = {"wall_clock_tflops_per_device": None}
    derived_none = bm.derive_chip_metrics(metrics_none)
    self.assertIn("wall_clock_tflops_per_device", derived_none)
    self.assertIn("wall_clock_tflops_per_chip", derived_none)
    self.assertIsNone(derived_none["wall_clock_tflops_per_chip"])

    # If wall_clock_tflops_per_chip is already present, it is not overwritten
    metrics_existing = {
        "wall_clock_tflops_per_device": 150.0,
        "wall_clock_tflops_per_chip": 400.0,
    }
    derived_existing = bm.derive_chip_metrics(metrics_existing)
    self.assertEqual(derived_existing["wall_clock_tflops_per_chip"], 400.0)

  def test_apply_xprof_timing_and_sync_fallback_schema_cleanliness(self):
    """Tests that xprof timing fallback to None does not pollute unrelated benchmark metrics."""

    class BenchmarkWithNoChipBw(DummyBenchmark):
      derive_chip_bandwidth = False

      @property
      def requires_multihost_sync(self) -> bool:
        return False

      def calculate_throughput_metrics(self, latency_ms: float, prefix: str):
        del latency_ms
        return {f"{prefix}_bandwidth_per_device_gb_s": 50.0}

    bm = BenchmarkWithNoChipBw()
    metrics = {"wall_clock_bandwidth_per_device_gb_s": 50.0}
    # pylint: disable=protected-access
    # Simulate empty duration fallback (synced_avg = 0.0)
    result_metrics = bm._apply_xprof_timing_and_sync(metrics)

    # Wall clock throughput remains intact!
    self.assertEqual(
        result_metrics["wall_clock_bandwidth_per_device_gb_s"], 50.0
    )
    # XProf throughput is set to None on fallback
    self.assertIn("xprof_bandwidth_per_device_gb_s", result_metrics)
    self.assertIsNone(result_metrics["xprof_bandwidth_per_device_gb_s"])
    self.assertNotIn("xprof_bandwidth_per_chip_gb_s", result_metrics)
    self.assertNotIn("xprof_tflops_per_device", result_metrics)


if __name__ == "__main__":
  absltest.main()
