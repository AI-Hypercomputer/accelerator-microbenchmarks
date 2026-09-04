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

DEFAULT_BANNER_MIN_WIDTH: int = 88
DEFAULT_BANNER_PADDING: int = 4
DEFAULT_DIAGONAL_MARKER: str = "X"

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


def _render_banner_box(
    title: str,
    body: str,
    min_width: int = DEFAULT_BANNER_MIN_WIDTH,
    padding: int = DEFAULT_BANNER_PADDING,
) -> str:
  """Renders a text box wrapped in '=' banners.

  Args:
    title: The title text to display in the header banner.
    body: The multi-line body string.
    min_width: The minimum character width of the banner box.
    padding: Additional horizontal padding added to the title length when
      calculating width.

  Returns:
    The formatted ASCII banner string.
  """
  body_lines = body.splitlines() if body else []
  max_body_line = max((len(line) for line in body_lines), default=0)
  width = max(max_body_line, len(title) + padding, min_width)
  bar = "=" * width
  return f"{bar}\n{title}\n{bar}\n{body}\n{bar}"


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

  banner_title = (
      f"Benchmark Results ({title})" if title else "Benchmark Results"
  )
  return _render_banner_box(title=banner_title, body=table_body)


def format_standard_table(
    df: pd.DataFrame,
    bench_cls: type[base.BaseBenchmark],
) -> str:
  """Renders standard flat ASCII summary table using bench_cls.REPORT_SCHEMA."""
  if df is None or df.empty or bench_cls is None:
    return ""

  schema = getattr(bench_cls, "REPORT_SCHEMA", ())
  if not schema:
    return ""

  return format_benchmark_table(df=df, schema=schema, title=bench_cls.__name__)


def format_device_matrix(
    df: pd.DataFrame,
    bench_cls: Optional[type[base.BaseBenchmark]] = None,
    diagonal_marker: str = DEFAULT_DIAGONAL_MARKER,
) -> str:
  """Renders aligned N x N pairwise bandwidth matrix ASCII grids for device_to_device."""
  del bench_cls
  if df is None or df.empty:
    return ""
  if (
      "src_device_index" not in df.columns
      or "dst_device_index" not in df.columns
  ):
    return ""

  src_col = "src_device_index"
  dst_col = "dst_device_index"
  metric_col = "bandwidth_gb_s"
  title_prefix = "Device-to-Device Bandwidth Matrix (GB/s)"

  sweep_candidates = ["dtype", "direction", "data_size_mib"]
  effective_sweeps = [c for c in sweep_candidates if c in df.columns]

  matrices = []
  groups = (
      df.groupby(effective_sweeps, sort=False, dropna=False)
      if effective_sweeps
      else [(None, df)]
  )

  for sweep_key, sub_df in groups:
    src_devs = sub_df[src_col].dropna().unique()
    dst_devs = sub_df[dst_col].dropna().unique()
    unique_devs = set(src_devs) | set(dst_devs)
    if not unique_devs:
      continue

    sorted_devs = sorted(int(d) for d in unique_devs)
    clean_sub_df = sub_df.dropna(subset=[src_col, dst_col]).copy()
    clean_sub_df[src_col] = clean_sub_df[src_col].astype(int)
    clean_sub_df[dst_col] = clean_sub_df[dst_col].astype(int)
    if metric_col not in clean_sub_df.columns:
      clean_sub_df[metric_col] = np.nan
    clean_sub_df = clean_sub_df.drop_duplicates(
        subset=[src_col, dst_col], keep="last"
    )

    # 1. Pivot into N x N matrix and reindex to full symmetric grid
    matrix = clean_sub_df.pivot(
        index=src_col, columns=dst_col, values=metric_col
    )
    matrix = matrix.reindex(index=sorted_devs, columns=sorted_devs)

    # 2. Format values using format_2f (maps numeric to .2f and NaN to '-')
    formatted_matrix = matrix.map(format_2f)

    # 3. Format row and column headers to D0, D1, etc.
    formatted_matrix.index = pd.Index([f"D{d}" for d in sorted_devs])
    formatted_matrix.columns = pd.Index([f"D{d}" for d in sorted_devs])

    # 4. Replace diagonal with diagonal marker
    for d in sorted_devs:
      formatted_matrix.loc[f"D{d}", f"D{d}"] = diagonal_marker

    body = formatted_matrix.to_string()

    config_parts = []
    if sweep_key is not None and effective_sweeps:
      if isinstance(sweep_key, tuple):
        for k, v in zip(effective_sweeps, sweep_key):
          config_parts.append(f"{k}={v}")
      else:
        config_parts.append(f"{effective_sweeps[0]}={sweep_key}")
    config_str = ", ".join(config_parts)
    title = f"{title_prefix} [{config_str}]" if config_str else title_prefix

    matrices.append(_render_banner_box(title=title, body=body))

  return "\n\n".join(matrices)


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
    platform_info = dataclasses.asdict(metadata.platform_info)
    metrics = res.metrics or {}
    benchmark_name = metadata.benchmark_name or ""
    test_name = metadata.test_name or ""
    start_time = metadata.start_time or ""

    entry = {
        **params,
        **metrics,
        **platform_info,
        "xla_flags": metadata.xla_flags,
        "libtpu_init_args": metadata.libtpu_init_args,
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
    bench_cls = class_map.get(bench_name)
    if bench_cls is None:
      continue

    formatters = getattr(bench_cls, "REPORT_FORMATTERS", None)
    if formatters is None:
      formatters = (
          (format_standard_table,)
          if getattr(bench_cls, "REPORT_SCHEMA", ())
          else ()
      )
    for formatter in formatters:
      section = formatter(group_df, bench_cls)
      if section:
        tables.append(section)

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
