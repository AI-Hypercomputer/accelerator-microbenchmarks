"""Test utilities for validating benchmark report schemas."""

from collections.abc import Iterable, Sequence
import dataclasses
from typing import Optional

from absl.testing import absltest
from accelerator_microbenchmarks.core import base

# Default keys produced by base classes or execution runners that are not
# intended to be displayed in concise ASCII summary tables.
DEFAULT_IGNORED_KEYS: frozenset[str] = frozenset({
    # Framework & execution parameters
    "device_id",
    "warmup_tries",
    "num_runs",
    "min_duration_s",
    "xprof_timing",
    "xprof_dir",
    "xla_dump_dir",
    "system",
    "use_trace_roofline",
    "hardware_stats",
    "seed",
    # Intermediate or redundant metrics
    "avg_ms",
    "p90_ms",
    "std_ms",
    "raw_times_ms",
    "intensity",
    "throughput",
    # Base metadata and flat columns
    "benchmark",
    "test_name",
    "start",
    "start_time",
    "end_time",
    "KET_ms",
    "platform",
})

# Keys populated by external profilers or metadata enrichment rather than
# calculate_metrics
DEFAULT_EXTRA_AVAILABLE_KEYS: frozenset[str] = frozenset({
    "xprof_p50_ms",
    "xprof_avg_ms",
    "xprof_p90_ms",
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
