"""Tests for benchmark report and ASCII table formatting."""

import io
import json
import os
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
import numpy as np
import pandas as pd


def _make_metadata(
    name: str = "DummyBenchmark",
    params: dict[str, Any] | None = None,
    platform_info: platform.PlatformInfo | None = None,
    hardware_spec: system.HardwareSpec | None = None,
    test_name: str | None = None,
    start_time: str = "2026-08-18T10:00:00",
    end_time: str = "2026-08-18T10:01:00",
    xla_flags: str = "",
    libtpu_init_args: str = "",
) -> base.BenchmarkMetadata:
  """Creates a dummy BenchmarkMetadata for testing."""
  return base.BenchmarkMetadata(
      benchmark_name=name,
      test_name=test_name or f"{name}_test",
      start_time=start_time,
      end_time=end_time,
      params=params or {},
      platform_info=platform_info
      if platform_info is not None
      else test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
      hardware_spec=hardware_spec
      if hardware_spec is not None
      else test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
      xla_flags=xla_flags,
      libtpu_init_args=libtpu_init_args,
  )


def _make_result(
    name: str = "DummyBenchmark",
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    metadata: base.BenchmarkMetadata | None = None,
    raw_times_ms: list[float] | None = None,
) -> base.BenchmarkResult:
  """Creates a dummy BenchmarkResult for testing."""
  if metadata is None:
    metadata = _make_metadata(name=name, params=params)
  return base.BenchmarkResult(
      metadata=metadata,
      metrics=metrics or {},
      raw_times_ms=raw_times_ms if raw_times_ms is not None else [1.0],
  )


@registry.benchmark_registry.register("dummy_report_benchmark")
class DummyBenchmark(base.BaseBenchmark):
  """A dummy benchmark for report testing."""

  REPORT_SCHEMA = (
      ("param_col", report.format_str),
      ("metric_val", report.format_2f),
  )


@registry.benchmark_registry.register("dummy_report_benchmark_1")
class DummyBenchmark1(base.BaseBenchmark):
  """First dummy benchmark for report testing."""

  REPORT_SCHEMA = (
      ("param1", report.format_str),
      ("metric1", report.format_2f),
  )


@registry.benchmark_registry.register("dummy_report_benchmark_2")
class DummyBenchmark2(base.BaseBenchmark):
  """Second dummy benchmark for report testing."""

  REPORT_SCHEMA = (
      ("param2", report.format_str),
      ("metric2", report.format_4f),
  )


@registry.benchmark_registry.register("dummy_empty_schema_benchmark")
class DummyEmptySchemaBenchmark(base.BaseBenchmark):
  """Dummy benchmark with empty schema."""

  REPORT_SCHEMA = ()
  REPORT_FORMATTERS = ()


class DummyNoSchemaBenchmark(base.BaseBenchmark):
  """Benchmark without REPORT_SCHEMA attribute."""

  pass


@registry.benchmark_registry.register("dummy_dual_formatter_benchmark")
class DummyDualFormatterBenchmark(base.BaseBenchmark):
  """Dummy benchmark with multiple formatters."""

  REPORT_SCHEMA = (
      ("dtype", report.format_str),
      ("src_device_index", report.format_str),
      ("dst_device_index", report.format_str),
      ("bandwidth_gb_s", report.format_2f),
  )
  REPORT_FORMATTERS = (
      report.format_standard_table,
      report.format_device_matrix,
  )


class ReportTest(absltest.TestCase):
  """Tests for benchmark table formatting and report generation."""

  def test_render_banner_box(self):
    """Tests _render_banner_box helper formatting, min_width, and padding."""
    # pylint: disable=protected-access
    # Test default min_width (88)
    box = report._render_banner_box("Short Title", "Line 1\nLine 2")
    lines = box.splitlines()
    self.assertLen(lines, 6)
    self.assertEqual(lines[0], "=" * report.DEFAULT_BANNER_MIN_WIDTH)
    self.assertEqual(lines[1], "Short Title")
    self.assertEqual(lines[2], "=" * report.DEFAULT_BANNER_MIN_WIDTH)
    self.assertEqual(lines[3], "Line 1")
    self.assertEqual(lines[4], "Line 2")
    self.assertEqual(lines[5], "=" * report.DEFAULT_BANNER_MIN_WIDTH)

    # Test title wider than min_width with padding
    long_title = "T" * 90
    box_long = report._render_banner_box(long_title, "content")
    expected_width = len(long_title) + report.DEFAULT_BANNER_PADDING
    self.assertEqual(box_long.splitlines()[0], "=" * expected_width)

  def test_format_benchmark_table(self):
    """Tests declarative format_benchmark_table formatting."""
    self.assertEqual(
        report.format_benchmark_table(pd.DataFrame(), schema=[], title="empty"),
        "",
    )
    schema = [
        ("param_col", report.format_str),
        ("metric_val", report.format_2f),
        ("missing_col", report.format_str),
    ]
    df = pd.DataFrame([{"param_col": "hello", "metric_val": 12.3456}])
    table = report.format_benchmark_table(df, schema=schema, title="custom")
    self.assertIn("Benchmark Results (custom)", table)
    self.assertIn("param_col", table)
    self.assertIn("metric_val", table)
    self.assertIn("missing_col", table)
    self.assertIn("hello", table)
    self.assertIn("12.35", table)
    self.assertIn("-", table)  # missing column formatted as '-'
    self.assertTrue(table.startswith("="))
    self.assertTrue(table.endswith("="))

  def test_results_to_dataframe(self):
    """Tests flattening BenchmarkResult objects into an unformatted DataFrame."""
    self.assertTrue(report.results_to_dataframe([]).empty)
    res = _make_result(
        "DummyBenchmark",
        params={"m": 1024, "n": 2048},
        metrics={"avg_ms": 1.23, "tflops_per_sec": 500.0},
        metadata=_make_metadata(
            "DummyBenchmark",
            params={"m": 1024, "n": 2048},
            xla_flags="--xla_test_flag=1",
            libtpu_init_args="--tpu_arg=2",
        ),
    )
    df = report.results_to_dataframe([res])
    self.assertLen(df, 1)
    self.assertEqual(df["benchmark"].iloc[0], "DummyBenchmark")
    self.assertEqual(df["test_name"].iloc[0], "DummyBenchmark_test")
    self.assertEqual(df["m"].iloc[0], 1024)
    self.assertEqual(df["n"].iloc[0], 2048)
    self.assertEqual(df["tpu_type"].iloc[0], system.TpuVersion.TPU7X.value)
    self.assertEqual(df["xla_flags"].iloc[0], "--xla_test_flag=1")
    self.assertEqual(df["libtpu_init_args"].iloc[0], "--tpu_arg=2")
    self.assertEqual(df["KET_ms"].iloc[0], 1.23)
    self.assertEqual(df["throughput"].iloc[0], 500.0)
    self.assertEqual(df["start"].iloc[0], "2026-08-18T10:00:00")

    # Fallback to bandwidth_gb_s when tflops_per_sec is missing
    res_bw = _make_result(
        "DummyBenchmark",
        params={"m": 1024},
        metrics={"avg_ms": 2.5, "bandwidth_gb_s": 900.0},
    )
    df_bw = report.results_to_dataframe([res_bw])
    self.assertEqual(df_bw["KET_ms"].iloc[0], 2.5)
    self.assertEqual(df_bw["throughput"].iloc[0], 900.0)
    self.assertEqual(df_bw["start"].iloc[0], "2026-08-18T10:00:00")

    # Fallback to throughput metric key when tflops/bandwidth are missing
    res_tp = _make_result(
        "DummyBenchmark",
        metrics={"avg_ms": 0.5, "throughput": 1234.5},
    )
    df_tp = report.results_to_dataframe([res_tp])
    self.assertEqual(df_tp["throughput"].iloc[0], 1234.5)

    # Robust handling of empty/missing metadata
    res_empty = _make_result(name="", params={}, metrics={"avg_ms": 1.0})
    df_none = report.results_to_dataframe([res_empty])
    self.assertLen(df_none, 1)
    self.assertEqual(df_none["benchmark"].iloc[0], "")
    self.assertEqual(df_none["xla_flags"].iloc[0], "")
    self.assertEqual(df_none["libtpu_init_args"].iloc[0], "")
    self.assertEqual(df_none["KET_ms"].iloc[0], 1.0)
    self.assertEqual(df_none["throughput"].iloc[0], 0.0)

  def test_value_formatters(self):
    """Tests float, 2f, 4f, and string formatters with valid and invalid inputs."""
    # format_float
    self.assertEqual(report.format_float(3.14159, precision=3), "3.142")
    self.assertEqual(report.format_float(12.3456, precision=1), "12.3")
    self.assertEqual(report.format_float(3.14159), "3.14")
    self.assertEqual(report.format_float(42, precision=2), "42.00")
    self.assertEqual(
        report.format_float(np.float32(1.23456), precision=3), "1.235"
    )
    self.assertEqual(report.format_float(np.int64(10), precision=1), "10.0")
    self.assertEqual(report.format_float(None), "-")
    self.assertEqual(report.format_float(True), "-")
    self.assertEqual(report.format_float(False), "-")
    self.assertEqual(report.format_float(np.bool_(True)), "-")
    self.assertEqual(report.format_float(np.bool_(False)), "-")
    self.assertEqual(report.format_float(float("nan")), "-")
    self.assertEqual(report.format_float(np.nan), "-")
    self.assertEqual(report.format_float(float("inf")), "-")
    self.assertEqual(report.format_float(float("-inf")), "-")
    self.assertEqual(report.format_float("invalid_str"), "-")

    # format_2f
    self.assertEqual(report.format_2f(7538.214), "7538.21")
    self.assertEqual(report.format_2f(100), "100.00")
    self.assertEqual(report.format_2f(np.float64(100.0)), "100.00")
    self.assertEqual(report.format_2f(np.float32(100.5)), "100.50")
    self.assertEqual(report.format_2f(np.int32(5)), "5.00")
    self.assertEqual(report.format_2f(None), "-")
    self.assertEqual(report.format_2f(True), "-")
    self.assertEqual(report.format_2f(False), "-")
    self.assertEqual(report.format_2f(np.bool_(True)), "-")
    self.assertEqual(report.format_2f(float("nan")), "-")
    self.assertEqual(report.format_2f(float("inf")), "-")
    self.assertEqual(report.format_2f(float("-inf")), "-")
    self.assertEqual(report.format_2f("invalid"), "-")

    # format_4f
    self.assertEqual(report.format_4f(0.07111), "0.0711")
    self.assertEqual(report.format_4f(np.float32(1.23456)), "1.2346")
    self.assertEqual(report.format_4f(5), "5.0000")
    self.assertEqual(report.format_4f(np.int64(7)), "7.0000")
    self.assertEqual(report.format_4f(None), "-")
    self.assertEqual(report.format_4f(True), "-")
    self.assertEqual(report.format_4f(False), "-")
    self.assertEqual(report.format_4f(np.bool_(False)), "-")
    self.assertEqual(report.format_4f(float("nan")), "-")
    self.assertEqual(report.format_4f(float("inf")), "-")
    self.assertEqual(report.format_4f(float("-inf")), "-")
    self.assertEqual(report.format_4f("invalid"), "-")

    # format_str
    self.assertEqual(report.format_str("copy"), "copy")
    self.assertEqual(report.format_str(123), "123")
    self.assertEqual(report.format_str(None), "-")
    self.assertEqual(report.format_str(""), "-")
    self.assertEqual(report.format_str(float("nan")), "-")
    self.assertEqual(report.format_str(np.nan), "-")
    self.assertEqual(report.format_str(pd.NA), "-")

  def test_format_standard_table(self):
    """Tests format_standard_table rendering schema from benchmark class."""
    dummy_res = _make_result(
        "DummyBenchmark",
        params={"param_col": "val1"},
        metrics={"metric_val": 123.456},
    )
    df = report.results_to_dataframe([dummy_res])
    table = report.format_standard_table(df, DummyBenchmark)
    self.assertIn("Benchmark Results (DummyBenchmark)", table)
    self.assertIn("param_col", table)
    self.assertIn("metric_val", table)
    self.assertIn("val1", table)
    self.assertIn("123.46", table)

  def test_format_standard_table_empty_or_missing_schema(self):
    """Tests format_standard_table returns empty string on empty/missing schema (ProductiveCoverage)."""
    df = pd.DataFrame([{"col": 1}])
    self.assertEqual(
        report.format_standard_table(df, DummyEmptySchemaBenchmark), ""
    )
    self.assertEqual(
        report.format_standard_table(df, DummyNoSchemaBenchmark), ""
    )

  def test_format_standard_table_empty_or_none_df(self):
    """Tests format_standard_table returns empty string on None or empty DataFrame."""
    self.assertEqual(report.format_standard_table(None, DummyBenchmark), "")
    self.assertEqual(
        report.format_standard_table(pd.DataFrame(), DummyBenchmark), ""
    )

  def test_format_device_matrix(self):
    """Tests format_device_matrix with single-sweep and multi-sweep configurations."""
    # Empty / invalid
    self.assertEqual(report.format_device_matrix(None), "")
    self.assertEqual(report.format_device_matrix(pd.DataFrame()), "")
    self.assertEqual(
        report.format_device_matrix(
            pd.DataFrame([{"src_device_index": 0, "bandwidth_gb_s": 100.0}])
        ),
        "",
    )

    # Single-sweep (with missing pair formatted as '-')
    single_sweep_data = [
        {"src_device_index": 0, "dst_device_index": 1, "bandwidth_gb_s": 50.0},
        {"src_device_index": 1, "dst_device_index": 0, "bandwidth_gb_s": 55.0},
        {"src_device_index": 2, "dst_device_index": 0, "bandwidth_gb_s": 60.0},
    ]
    single_matrix = report.format_device_matrix(
        pd.DataFrame(single_sweep_data),
        bench_cls=DummyDualFormatterBenchmark,
    )
    self.assertIn("Device-to-Device Bandwidth Matrix (GB/s)", single_matrix)
    self.assertIn("D0", single_matrix)
    self.assertIn("D1", single_matrix)
    self.assertIn("D2", single_matrix)
    self.assertEqual(report.DEFAULT_DIAGONAL_MARKER, "X")
    self.assertIn(report.DEFAULT_DIAGONAL_MARKER, single_matrix)
    self.assertEqual(single_matrix.count(report.DEFAULT_DIAGONAL_MARKER), 3)
    row_tokens = {
        parts[0]: parts[1:]
        for line in single_matrix.splitlines()
        if (parts := line.split()) and parts[0] in ("D0", "D1", "D2")
    }
    self.assertEqual(row_tokens["D0"][0], report.DEFAULT_DIAGONAL_MARKER)
    self.assertEqual(row_tokens["D1"][1], report.DEFAULT_DIAGONAL_MARKER)
    self.assertEqual(row_tokens["D2"][2], report.DEFAULT_DIAGONAL_MARKER)
    self.assertIn("50.00", single_matrix)
    self.assertIn("55.00", single_matrix)
    self.assertIn("-", single_matrix)
    self.assertTrue(single_matrix.startswith("="))
    self.assertTrue(single_matrix.endswith("="))

    # Custom diagonal marker
    custom_matrix = report.format_device_matrix(
        pd.DataFrame(single_sweep_data),
        diagonal_marker="N/A",
    )
    self.assertIn("N/A", custom_matrix)
    self.assertEqual(custom_matrix.count("N/A"), 3)
    custom_tokens = {
        parts[0]: parts[1:]
        for line in custom_matrix.splitlines()
        if (parts := line.split()) and parts[0] in ("D0", "D1", "D2")
    }
    self.assertEqual(custom_tokens["D0"][0], "N/A")
    self.assertEqual(custom_tokens["D1"][1], "N/A")
    self.assertEqual(custom_tokens["D2"][2], "N/A")

    # Multi-sweep
    multi_sweep_data = [
        {
            "dtype": "bfloat16",
            "direction": "uni",
            "data_size_mib": 1024,
            "src_device_index": 0,
            "dst_device_index": 1,
            "bandwidth_gb_s": 80.0,
        },
        {
            "dtype": "bfloat16",
            "direction": "uni",
            "data_size_mib": 1024,
            "src_device_index": 1,
            "dst_device_index": 0,
            "bandwidth_gb_s": 85.0,
        },
        {
            "dtype": "float32",
            "direction": "bi",
            "data_size_mib": 2048,
            "src_device_index": 0,
            "dst_device_index": 1,
            "bandwidth_gb_s": 160.0,
        },
        {
            "dtype": "float32",
            "direction": "bi",
            "data_size_mib": 2048,
            "src_device_index": 1,
            "dst_device_index": 0,
            "bandwidth_gb_s": 165.0,
        },
    ]
    multi_matrix = report.format_device_matrix(pd.DataFrame(multi_sweep_data))
    self.assertIn(
        "Device-to-Device Bandwidth Matrix (GB/s) [dtype=bfloat16,"
        " direction=uni, data_size_mib=1024]",
        multi_matrix,
    )
    self.assertIn(
        "Device-to-Device Bandwidth Matrix (GB/s) [dtype=float32, direction=bi,"
        " data_size_mib=2048]",
        multi_matrix,
    )
    self.assertIn("80.00", multi_matrix)
    self.assertIn("160.00", multi_matrix)
    self.assertLen(multi_matrix.split("\n\n"), 2)

  def test_format_device_matrix_preserves_nan_sweep_keys(self):
    """Tests format_device_matrix preserves sweep keys with None/NaN."""
    nan_sweep_data = [
        {
            "dtype": None,
            "direction": "uni",
            "data_size_mib": 1024,
            "src_device_index": 0,
            "dst_device_index": 1,
            "bandwidth_gb_s": 75.0,
        },
        {
            "dtype": None,
            "direction": "uni",
            "data_size_mib": 1024,
            "src_device_index": 1,
            "dst_device_index": 0,
            "bandwidth_gb_s": 80.0,
        },
    ]
    matrix = report.format_device_matrix(pd.DataFrame(nan_sweep_data))
    self.assertTrue(matrix)
    self.assertIn(
        "Device-to-Device Bandwidth Matrix (GB/s) [dtype=nan, direction=uni,"
        " data_size_mib=1024]",
        matrix,
    )
    self.assertIn("D0", matrix)
    self.assertIn("D1", matrix)
    self.assertIn(report.DEFAULT_DIAGONAL_MARKER, matrix)
    self.assertIn("75.00", matrix)
    self.assertIn("80.00", matrix)

  def test_generate_benchmark_report(self):
    """Tests report generation executing registered REPORT_FORMATTERS."""
    self.assertEqual(report.generate_benchmark_report(None), "")
    self.assertEqual(report.generate_benchmark_report(pd.DataFrame()), "")
    self.assertEqual(
        report.generate_benchmark_report(pd.DataFrame([{"foo": 1}])), ""
    )

    # Unknown benchmark without schema returns empty string
    unknown_res = _make_result("UnknownBench", {}, {})
    self.assertEqual(
        report.generate_benchmark_report(
            report.results_to_dataframe([unknown_res])
        ),
        "",
    )

    # Registered benchmark executing formatters
    dummy_res = _make_result(
        "DummyBenchmark",
        params={"param_col": "val1"},
        metrics={"metric_val": 123.45},
    )
    df = report.results_to_dataframe([dummy_res, unknown_res])
    rep = report.generate_benchmark_report(df)
    self.assertIn("Benchmark Results (DummyBenchmark)", rep)
    self.assertIn("val1", rep)
    self.assertIn("123.45", rep)
    self.assertEqual(rep.count("Benchmark Results (DummyBenchmark)"), 1)

  def test_generate_benchmark_report_multi_benchmark(self):
    """Tests multi-benchmark report generation with two distinct registered benchmarks."""
    res1 = _make_result(
        "DummyBenchmark1",
        params={"param1": "foo"},
        metrics={"metric1": 42.0},
    )
    res2 = _make_result(
        "DummyBenchmark2",
        params={"param2": "bar"},
        metrics={"metric2": 0.1234},
    )
    df = report.results_to_dataframe([res1, res2])
    rep = report.generate_benchmark_report(df)
    self.assertIn("Benchmark Results (DummyBenchmark1)", rep)
    self.assertIn("Benchmark Results (DummyBenchmark2)", rep)
    self.assertIn("foo", rep)
    self.assertIn("bar", rep)
    self.assertIn("42.00", rep)
    self.assertIn("0.1234", rep)
    table1 = report.format_benchmark_table(
        report.results_to_dataframe([res1]),
        schema=DummyBenchmark1.REPORT_SCHEMA,
        title="DummyBenchmark1",
    )
    table2 = report.format_benchmark_table(
        report.results_to_dataframe([res2]),
        schema=DummyBenchmark2.REPORT_SCHEMA,
        title="DummyBenchmark2",
    )
    self.assertEqual(rep, f"{table1}\n\n{table2}")

  def test_generate_benchmark_report_dual_formatter(self):
    """Tests that a benchmark with multiple REPORT_FORMATTERS renders all sections."""
    res1 = _make_result(
        "DummyDualFormatterBenchmark",
        params={
            "dtype": "bfloat16",
            "src_device_index": 0,
            "dst_device_index": 1,
        },
        metrics={"bandwidth_gb_s": 90.0},
    )
    res2 = _make_result(
        "DummyDualFormatterBenchmark",
        params={
            "dtype": "bfloat16",
            "src_device_index": 1,
            "dst_device_index": 0,
        },
        metrics={"bandwidth_gb_s": 95.0},
    )
    df = report.results_to_dataframe([res1, res2])
    rep = report.generate_benchmark_report(df)

    # Should contain standard table
    self.assertIn("Benchmark Results (DummyDualFormatterBenchmark)", rep)
    self.assertIn("90.00", rep)
    self.assertIn("95.00", rep)

    # Should also contain device matrix
    self.assertIn("Device-to-Device Bandwidth Matrix (GB/s)", rep)
    self.assertIn("D0", rep)
    self.assertIn("D1", rep)
    self.assertIn(report.DEFAULT_DIAGONAL_MARKER, rep)

    # Separated by \n\n
    sections = rep.split("\n\n")
    self.assertGreaterEqual(len(sections), 2)

  def test_generate_benchmark_report_formatter_injection(self):
    """Tests that generate_benchmark_report injects bench_cls into formatters."""
    received_calls = []

    def custom_formatter(df, bench_cls=None):
      received_calls.append((df, bench_cls))
      return f"SECTION_{bench_cls.__name__ if bench_cls else 'None'}"

    @registry.benchmark_registry.register("dummy_injection_benchmark")
    class DummyInjectionBenchmark(base.BaseBenchmark):
      REPORT_FORMATTERS = (custom_formatter,)

    res = _make_result(
        "DummyInjectionBenchmark",
        params={"param_col": "foo"},
        metrics={"metric_val": 1.0},
    )
    df = report.results_to_dataframe([res])
    rep = report.generate_benchmark_report(df)
    self.assertEqual(rep, "SECTION_DummyInjectionBenchmark")
    self.assertLen(received_calls, 1)
    self.assertIs(received_calls[0][1], DummyInjectionBenchmark)
    self.assertIn("param_col", received_calls[0][0].columns)

  def test_save_output(self):
    """Tests saving summary.csv and detailed.json to output directory."""
    dummy_res = _make_result(
        "DummyBenchmark",
        params={"param_col": "val1"},
        metrics={"metric_val": 123.45, "avg_ms": 5.0, "tflops_per_sec": 200.0},
        metadata=_make_metadata(
            "DummyBenchmark",
            params={"param_col": "val1"},
            xla_flags="--xla_test_flag=1",
            libtpu_init_args="--tpu_arg=2",
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      report.save_output([dummy_res], tmpdir)

      csv_path = os.path.join(tmpdir, "summary.csv")
      json_path = os.path.join(tmpdir, "detailed.json")
      self.assertTrue(os.path.exists(csv_path))
      self.assertTrue(os.path.exists(json_path))

      df = pd.read_csv(csv_path)
      self.assertLen(df, 1)
      self.assertEqual(df["benchmark"].iloc[0], "DummyBenchmark")
      self.assertEqual(df["avg_ms"].iloc[0], 5.0)
      self.assertEqual(df["tflops_per_sec"].iloc[0], 200.0)
      self.assertEqual(df["KET_ms"].iloc[0], 5.0)
      self.assertEqual(df["throughput"].iloc[0], 200.0)
      self.assertEqual(df["start"].iloc[0], "2026-08-18T10:00:00")
      self.assertEqual(df["xla_flags"].iloc[0], "--xla_test_flag=1")
      self.assertEqual(df["libtpu_init_args"].iloc[0], "--tpu_arg=2")

      with open(json_path, "r") as f:
        data = json.load(f)
      self.assertLen(data, 1)
      self.assertEqual(data[0]["metadata"]["benchmark_name"], "DummyBenchmark")
      self.assertEqual(data[0]["metadata"]["xla_flags"], "--xla_test_flag=1")
      self.assertEqual(data[0]["metadata"]["libtpu_init_args"], "--tpu_arg=2")

    # Non-standard objects in BenchmarkResult serialize with default=str
    class CustomObj:
      """Dummy custom object for serialization test."""

      def __str__(self):
        return "custom_str_repr"

    res_with_custom_obj = _make_result(
        "DummyBenchmark",
        metrics={"val": np.int64(42), "obj": CustomObj()},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      report.save_output([res_with_custom_obj], tmpdir)
      with open(os.path.join(tmpdir, "detailed.json"), "r") as f:
        data = json.load(f)
      self.assertLen(data, 1)
      self.assertEqual(data[0]["metrics"]["obj"], "custom_str_repr")

  def test_save_output_throughput_not_overwritten(self):
    """Tests that throughput is not overwritten when saving output."""
    metadata = _make_metadata(
        name="DummyBenchmark",
        params={"param1": "val1"},
    )
    metrics = {
        "avg_ms": 10.0,
        "bandwidth_gb_s": 100.0,
        "throughput": 0.0,
    }
    result = _make_result(metadata=metadata, metrics=metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
      report.save_output([result], tmpdir)

      csv_path = os.path.join(tmpdir, "summary.csv")
      json_path = os.path.join(tmpdir, "detailed.json")
      self.assertTrue(os.path.exists(csv_path))
      self.assertTrue(os.path.exists(json_path))

      df = pd.read_csv(csv_path)
      self.assertNotIn("Unnamed: 0", df.columns)
      self.assertEqual(df["throughput"].iloc[0], 100.0)
      self.assertEqual(df["KET_ms"].iloc[0], 10.0)
      self.assertEqual(df["benchmark"].iloc[0], "DummyBenchmark")

  def test_save_output_tflops(self):
    """Tests that tflops_per_sec is mapped to throughput when saving output."""
    metadata = _make_metadata(
        name="DummyGemm",
        params={"param1": "val1"},
    )
    metrics = {
        "avg_ms": 10.0,
        "tflops_per_sec": 250.0,
    }
    result = _make_result(metadata=metadata, metrics=metrics)

    with tempfile.TemporaryDirectory() as tmpdir:
      report.save_output([result], tmpdir)

      csv_path = os.path.join(tmpdir, "summary.csv")
      self.assertTrue(os.path.exists(csv_path))

      df = pd.read_csv(csv_path)
      self.assertEqual(df["throughput"].iloc[0], 250.0)

  def test_report_results(self):
    """Tests centralized reporting pipeline report_results."""
    dummy_res = _make_result(
        "DummyBenchmark",
        params={"param_col": "val1"},
        metrics={"metric_val": 123.45},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        report.report_results(
            [dummy_res],
            output_dir=tmpdir,
            print_table=True,
        )
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "summary.csv")))
        self.assertTrue(os.path.exists(os.path.join(tmpdir, "detailed.json")))
        self.assertIn(
            "Benchmark Results (DummyBenchmark)", mock_stdout.getvalue()
        )

      # print_table=False
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        report.report_results(
            [dummy_res],
            output_dir=None,
            print_table=False,
        )
        self.assertEqual(mock_stdout.getvalue(), "")

  def test_generate_benchmark_report_default_fallback(self):
    """Tests that a benchmark without REPORT_FORMATTERS falls back to format_standard_table."""

    @registry.benchmark_registry.register("dummy_fallback_benchmark")
    class DummyFallbackBenchmark(base.BaseBenchmark):
      REPORT_SCHEMA = (("val", report.format_str),)

    res = _make_result(
        DummyFallbackBenchmark.__name__, params={"val": "fallback_ok"}
    )
    df = report.results_to_dataframe([res])
    rep = report.generate_benchmark_report(df)
    self.assertIn("Benchmark Results (DummyFallbackBenchmark)", rep)
    self.assertIn("fallback_ok", rep)


if __name__ == "__main__":
  absltest.main()
