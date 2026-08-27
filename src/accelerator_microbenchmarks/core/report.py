"""Human-readable benchmark summary report tables formatting module."""

import dataclasses
import json
import math
import os
from typing import Any, Callable, Optional, Sequence

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import registry
import numpy as np
import pandas as pd

# ==============================================================================
# Value Formatters
# ==============================================================================


def format_float(val: Any, precision: int = 2) -> str:
  """Formats a numeric value to a fixed decimal precision.

  Handles ints, floats, NumPy scalars, and returns '-' for None,
  booleans, non-numeric types, NaN, or Inf.

  Args:
    val: Value to format.
    precision: Number of decimal places to format.

  Returns:
    Formatted numeric string or '-' on None, boolean, NaN, Inf, or invalid
    input.
  """
  if isinstance(val, (bool, np.bool_)) or val is None:
    return "-"
  try:
    num = float(val)
  except (ValueError, TypeError):
    return "-"
  if math.isnan(num) or math.isinf(num):
    return "-"
  return f"{num:.{precision}f}"


def format_2f(val: Any) -> str:
  """Convenience shortcut for 2 decimal places (throughput, bandwidth, FLOPS)."""
  return format_float(val, precision=2)


def format_4f(val: Any) -> str:
  """Convenience shortcut for 4 decimal places (latency in ms)."""
  return format_float(val, precision=4)


def format_str(val: Any) -> str:
  """Formats strings, returning '-' for None, NaN, pd.isna, or empty strings."""
  if val is None:
    return "-"
  if isinstance(val, str):
    return val if val else "-"
  try:
    if pd.isna(val):
      return "-"
  except (ValueError, TypeError):
    pass
  return str(val)


# ==============================================================================
# Benchmark Table Formatter
# ==============================================================================


def format_benchmark_table(
    df: pd.DataFrame,
    schema: Sequence[tuple[str, Callable[[Any], str]]],
    title: str = "",
) -> str:
  """Formats a DataFrame into an ASCII summary table using declarative column schema."""
  if df is None or df.empty or not schema:
    return ""

  cols = [col_name for col_name, _ in schema]
  formatters = [formatter for _, formatter in schema]

  sub_df = df.reindex(columns=cols)
  table_body = sub_df.to_string(index=False, formatters=formatters, na_rep="-")

  table_lines = table_body.splitlines()
  width = max(len(line) for line in table_lines) if table_lines else 88
  width = max(width, len(title) + 24, 88)
  top_bar = "=" * width
  banner_title = (
      f"Benchmark Results ({title})" if title else "Benchmark Results"
  )

  return f"{top_bar}\n{banner_title}\n{top_bar}\n{table_body}\n{top_bar}"


# ==============================================================================
# Public Reporting API
# ==============================================================================


def results_to_dataframe(
    results: Sequence[base.BenchmarkResult],
) -> pd.DataFrame:
  """Flattens a sequence of BenchmarkResult objects into a DataFrame."""
  if not results:
    return pd.DataFrame()

  flat_results = []
  for res in results:
    metadata = res.metadata
    params = metadata.params or {}
    device_info = metadata.device_info or {}
    metrics = res.metrics or {}
    benchmark_name = metadata.benchmark_name or ""
    test_name = metadata.test_name or ""
    start_time = metadata.start_time or ""

    entry = {
        **params,
        **metrics,
        **device_info,
        "benchmark": benchmark_name,
        "test_name": test_name,
        "KET_ms": metrics.get("avg_ms", 0.0),
        "throughput": metrics.get(
            "tflops_per_sec",
            metrics.get("bandwidth_gb_s", metrics.get("throughput", 0.0)),
        ),
        "start": start_time,
    }
    flat_results.append(entry)

  return pd.DataFrame(flat_results)


def generate_benchmark_report(
    df: pd.DataFrame,
) -> str:
  """Generates a human-readable benchmark report string for all results in the DataFrame."""
  if df is None or df.empty or "benchmark" not in df.columns:
    return ""

  all_benchmarks = registry.benchmark_registry.get_all()
  class_map = {cls.__name__: cls for cls in all_benchmarks.values()}

  tables = []
  for name, group_df in df.groupby("benchmark", sort=False):
    bench_name = str(name)
    if bench_name not in class_map:
      continue

    bench_cls = class_map[bench_name]
    bench_schema = getattr(bench_cls, "REPORT_SCHEMA", ())

    if bench_schema:
      table_str = format_benchmark_table(
          df=group_df,
          schema=bench_schema,
          title=bench_name,
      )
      if table_str:
        tables.append(table_str)

  return "\n\n".join(tables)


def save_output(
    results: Sequence[base.BenchmarkResult],
    output_dir: str,
    df: Optional[pd.DataFrame] = None,
) -> None:
  """Saves benchmark results to summary.csv and detailed.json in output_dir."""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

  if df is None:
    df = results_to_dataframe(results)

  # 1. Summary CSV (Digestible)
  csv_path = os.path.join(output_dir, "summary.csv")
  df.to_csv(csv_path, index=False)
  print(f"Summary saved to: {csv_path}")

  # 2. Detailed JSON (Complete)
  json_path = os.path.join(output_dir, "detailed.json")
  with open(json_path, "w") as f:
    json.dump(
        [dataclasses.asdict(r) for r in results],
        f,
        indent=2,
        default=str,
    )


def report_results(
    results: Sequence[base.BenchmarkResult],
    output_dir: Optional[str] = None,
    print_table: bool = True,
) -> None:
  """Reports benchmark results to stdout and saves to output directory."""
  if not results:
    return

  df = results_to_dataframe(results)

  if print_table:
    report_str = generate_benchmark_report(df=df)
    if report_str:
      print(f"\n{report_str}")

  if output_dir:
    save_output(results, output_dir=output_dir, df=df)
