"""Tests for benchmark report and ASCII table formatting."""

import io
import json
import os
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
import numpy as np
import pandas as pd


def _make_metadata(
    name: str = "DummyBenchmark",
    params: dict[str, Any] | None = None,
    device_info: dict[str, Any] | None = None,
    test_name: str | None = None,
    start_time: str = "2026-08-18T10:00:00",
    end_time: str = "2026-08-18T10:01:00",
) -> base.BenchmarkMetadata:
  """Creates a dummy BenchmarkMetadata for testing."""
  return base.BenchmarkMetadata(
      benchmark_name=name,
      test_name=test_name or f"{name}_test",
      start_time=start_time,
      end_time=end_time,
      params=params or {},
      device_info=device_info
      if device_info is not None
      else {"platform": "tpu"},
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


class ReportTest(absltest.TestCase):
  """Tests for benchmark table formatting and report generation."""

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
    )
    df = report.results_to_dataframe([res])
    self.assertLen(df, 1)
    self.assertEqual(df["benchmark"].iloc[0], "DummyBenchmark")
    self.assertEqual(df["test_name"].iloc[0], "DummyBenchmark_test")
    self.assertEqual(df["m"].iloc[0], 1024)
    self.assertEqual(df["n"].iloc[0], 2048)
    self.assertEqual(df["avg_ms"].iloc[0], 1.23)
    self.assertEqual(df["tflops_per_sec"].iloc[0], 500.0)
    self.assertEqual(df["platform"].iloc[0], "tpu")
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

  def test_format_benchmark_table_integration(self):
    """Tests summary table generation with valid and empty schemas/results."""
    self.assertEqual(
        report.format_benchmark_table(pd.DataFrame(), schema=[]), ""
    )
    dummy_res = _make_result(
        "DummyBenchmark",
        params={"param_col": "val1"},
        metrics={"metric_val": 123.45},
    )
    df = report.results_to_dataframe([dummy_res])
    self.assertEqual(report.format_benchmark_table(df, schema=()), "")
    table = report.format_benchmark_table(
        df,
        schema=DummyBenchmark.REPORT_SCHEMA,
        title="DummyBenchmark",
    )
    self.assertIn("Benchmark Results (DummyBenchmark)", table)
    self.assertIn("val1", table)
    self.assertIn("123.45", table)

  def test_generate_benchmark_report(self):
    """Tests multi-result report generation and autodiscovery."""
    self.assertEqual(report.generate_benchmark_report(pd.DataFrame()), "")
    dummy_res = _make_result(
        "DummyBenchmark",
        params={"param_col": "val1"},
        metrics={"metric_val": 123.45},
    )
    unknown_res = _make_result("UnknownBench", {}, {})
    unknown_df = report.results_to_dataframe([unknown_res])

    # Unknown benchmark without schema returns empty string.
    self.assertEqual(report.generate_benchmark_report(unknown_df), "")

    # Auto-discovery without schema
    df = report.results_to_dataframe([dummy_res, unknown_res])
    rep_auto = report.generate_benchmark_report(df)
    self.assertIn("Benchmark Results (DummyBenchmark)", rep_auto)
    self.assertIn("val1", rep_auto)
    self.assertIn("123.45", rep_auto)
    self.assertEqual(rep_auto.count("Benchmark Results (DummyBenchmark)"), 1)

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
    # Autodiscovery
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

  def test_empty_schema_returns_empty_string(self):
    """Tests empty schema REPORT_SCHEMA = () with non-empty results returning empty string."""
    res = _make_result(
        "DummyEmptySchemaBenchmark",
        params={"param": "value"},
        metrics={"avg_ms": 1.23},
    )
    df = report.results_to_dataframe([res])
    self.assertEqual(
        report.format_benchmark_table(
            df,
            schema=DummyEmptySchemaBenchmark.REPORT_SCHEMA,
            title="DummyEmptySchemaBenchmark",
        ),
        "",
    )
    # Autodiscovery
    rep = report.generate_benchmark_report(df)
    self.assertEqual(rep, "")

  def test_save_output(self):
    """Tests saving summary.csv and detailed.json to output directory."""
    dummy_res = _make_result(
        "DummyBenchmark",
        params={"param_col": "val1"},
        metrics={"metric_val": 123.45, "avg_ms": 5.0, "tflops_per_sec": 200.0},
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

      with open(json_path, "r") as f:
        data = json.load(f)
      self.assertLen(data, 1)
      self.assertEqual(data[0]["metadata"]["benchmark_name"], "DummyBenchmark")

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
        device_info={"device": "tpu"},
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
        device_info={"device": "tpu"},
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


if __name__ == "__main__":
  absltest.main()
