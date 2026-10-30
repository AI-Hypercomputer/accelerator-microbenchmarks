"""Test utilities for validating benchmark report schemas."""

from collections.abc import Iterable, Sequence
import dataclasses
from typing import Optional
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import profiler
from accelerator_microbenchmarks.core import system

DEFAULT_TEST_PLATFORM_INFO: platform.PlatformInfo = platform.PlatformInfo(
    tpu_type=system.TpuVersion.TPU7X,
    topology="2x2x1",
    total_devices=4,
    local_devices=4,
    process_count=1,
    process_index=0,
    python_version="3.11.0",
    jax_version="0.4.30",
    jaxlib_version="0.4.30",
    libtpu_version="0.1.dev20240101",
)
DEFAULT_TEST_HARDWARE_SPEC: system.HardwareSpec = system.TPU7X_HARDWARE_SPEC


@dataclasses.dataclass(frozen=True)
class FakeXprofTrace(profiler.XprofTrace):
  """Empty `XprofTrace` that remembers the directory it was "loaded" from."""

  trace_dir: str = ""


def fake_load_xprof_trace(
    trace_dir: str, is_xprof_op_fn=None
) -> FakeXprofTrace:
  """Stub for `profiler.load_xprof_trace` resolving TPU:0-7 and /host:CPU PIDs.

  Unmocked `device_step_durations_ms` calls on it return `[]` (no events).

  Args:
    trace_dir: Directory the trace would be loaded from.
    is_xprof_op_fn: Unused custom XProf op matcher.

  Returns:
    An empty trace tagged with `trace_dir`.
  """
  del is_xprof_op_fn
  return FakeXprofTrace(
      timing_events_by_pid={},
      step_spans_by_pid={},
      process_pids={
          **{f"/device:TPU:{i}": i + 1 for i in range(8)},
          "/host:CPU": 100,
      },
      fallback_pid=None,
      trace_dir=trace_dir,
  )


# Messages printed when an XProf parse exception is caught and swallowed.
_SWALLOWED_XPROF_PARSE_ERRORS = (
    "Error parsing local XProf trace",
    "Warning: Local XProf parsing failed",
    "Warning: Failed to parse XProf durations",
)


def _swallowed_parse_errors(mock_print: mock.MagicMock) -> list[str]:
  return [
      str(call.args[0])
      for call in mock_print.call_args_list
      if call.args
      and str(call.args[0]).startswith(_SWALLOWED_XPROF_PARSE_ERRORS)
  ]


def patch_xprof_trace_loading(
    test: absltest.TestCase,
) -> tuple[mock.MagicMock, mock.MagicMock]:
  """Stubs trace loading with `fake_load_xprof_trace` for the whole test.

  Also fails the test if an XProf parse exception is swallowed, unless the
  test acknowledges it with `assert_swallowed_xprof_parse_error`.

  Args:
    test: The running test case.

  Returns:
    `(mock_load_trace, mock_print)`.
  """
  mock_load_trace = test.enter_context(
      mock.patch.object(
          profiler, "load_xprof_trace", side_effect=fake_load_xprof_trace
      )
  )
  mock_print = test.enter_context(mock.patch("builtins.print", wraps=print))
  test.addCleanup(
      lambda: test.assertEmpty(
          _swallowed_parse_errors(mock_print),
          "XProf parse exception was swallowed unexpectedly.",
      )
  )
  return mock_load_trace, mock_print


def assert_swallowed_xprof_parse_error(
    test: absltest.TestCase, mock_print: mock.MagicMock
) -> None:
  """Asserts an XProf parse exception was swallowed, then clears the record."""
  test.assertNotEmpty(_swallowed_parse_errors(mock_print))
  mock_print.reset_mock()


# Default keys produced by base classes or execution runners that are not
# intended to be displayed in concise ASCII summary tables.
DEFAULT_IGNORED_KEYS: frozenset[str] = frozenset({
    # Framework & execution parameters
    "device_id",
    "warmup_tries",
    "num_runs",
    "min_duration_s",
    "samples_per_run",
    "seed",
    # Intermediate or redundant metrics
    "wall_clock_avg_ms",
    "wall_clock_p90_ms",
    "wall_clock_std_ms",
    "wall_clock_tflops_per_device",
    "xprof_tflops_per_device",
    "wall_clock_bandwidth_per_device_gb_s",
    "xprof_bandwidth_per_device_gb_s",
    "raw_times_ms",
    *constants.SOAKING_METRIC_KEYS,
    *constants.THERMAL_METRIC_KEYS,
    "intensity",
    "peak_hbm_bw_per_device_gb_s",
    "roofline_tflops_limit_per_device",
    "wall_clock_compute_roofline_efficiency_pct",
    "xprof_compute_roofline_efficiency_pct",
    "wall_clock_memory_roofline_efficiency_pct",
    "xprof_memory_roofline_efficiency_pct",
    # Base metadata and flat columns
    "benchmark",
    "test_name",
    "start",
    "start_time",
    "end_time",
    "platform",
})

# Keys populated by external profilers or metadata enrichment rather than
# calculate_metrics
DEFAULT_EXTRA_AVAILABLE_KEYS: frozenset[str] = frozenset({
    "xprof_p50_ms",
    "xprof_avg_ms",
    "xprof_p90_ms",
    "xprof_std_ms",
    constants.XPROF_PRE_SOAKING_P50_MS,
    constants.XPROF_POST_SOAKING_P50_MS,
    constants.XPROF_SLOWDOWN_RATIO,
    "xprof_pre_soaking_tflops_per_device",
    "xprof_post_soaking_tflops_per_device",
    "xprof_tflops_per_device",
    "xprof_tflops_per_chip",
    "xprof_bandwidth_per_device_gb_s",
    "xprof_bandwidth_per_host_gb_s",
    "xprof_bandwidth_per_chip_gb_s",
    "xprof_bandwidth_per_host_gbps",
    "wall_clock_tflops_per_chip",
    "wall_clock_bandwidth_per_chip_gb_s",
    # Added by apply_roofline_analysis (RooflineMode.NETWORK_ICI) after
    # calculate_metrics.
    "peak_ici_bandwidth_per_chip_gb_s",
    "wall_clock_ici_roofline_efficiency_pct",
    "xprof_ici_roofline_efficiency_pct",
})


def assert_schema_matches_output(
    test_case: absltest.TestCase,
    benchmark: base.BaseBenchmark,
    ignored_keys: Optional[Iterable[str]] = None,
    unignored_keys: Optional[Iterable[str]] = None,
    extra_available_keys: Optional[Iterable[str]] = None,
    dummy_times_ms: Optional[Sequence[float]] = None,
) -> None:
  """Validates bidirectional contract between REPORT_SCHEMA and benchmark outputs.

  1. Forward check: All columns in REPORT_SCHEMA exist in output keys (or
  extra_available_keys).
  2. Reverse check: All output keys exist in REPORT_SCHEMA or ignored_keys.
  """
  schema = getattr(benchmark, "REPORT_SCHEMA", None)
  test_case.assertIsNotNone(
      schema, f"{benchmark.__class__.__name__} is missing REPORT_SCHEMA"
  )
  test_case.assertNotEmpty(
      schema, f"{benchmark.__class__.__name__} has empty REPORT_SCHEMA"
  )

  schema_cols = {col for col, _ in benchmark.REPORT_SCHEMA}

  # Extract params from config
  params = dataclasses.asdict(benchmark.config)

  # Calculate metrics using synthetic times (CPU safe)
  raw_times = dummy_times_ms or [1.0, 1.0, 1.0]
  metrics = benchmark.calculate_metrics(raw_times)
  if hasattr(benchmark, "derive_chip_metrics"):
    metrics = benchmark.derive_chip_metrics(metrics)

  output_keys = set(params.keys()) | set(metrics.keys())
  effective_extra = DEFAULT_EXTRA_AVAILABLE_KEYS | set(
      extra_available_keys or []
  )
  effective_available = output_keys | effective_extra

  # Calculate effective ignored keys
  effective_ignored = (DEFAULT_IGNORED_KEYS - set(unignored_keys or [])) | set(
      ignored_keys or []
  )

  # 1. Forward validation: no missing columns in schema
  missing_from_output = set(schema_cols) - effective_available
  test_case.assertEmpty(
      missing_from_output,
      f"{benchmark.__class__.__name__}.REPORT_SCHEMA defines columns"
      f" {missing_from_output} which are not produced by config"
      f" ({set(params.keys())}) or calculate_metrics ({set(metrics.keys())}).",
  )

  # 2. Reverse validation: no unaccounted outputs
  unreported_keys = output_keys - set(schema_cols) - effective_ignored
  test_case.assertEmpty(
      unreported_keys,
      f"{benchmark.__class__.__name__} produced keys {unreported_keys} that are"
      " neither included in REPORT_SCHEMA nor marked in ignored_keys.",
  )
