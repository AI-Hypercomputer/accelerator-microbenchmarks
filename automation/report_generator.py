# Copyright 2026 The TPUMS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Consolidated Markdown Report and HTML Preview Generator for TPUMS Automation.

Reads the aggregated benchmark category results (CSV data from Metrics
Aggregator or GCS)
and generates:
  1. A comprehensive Markdown summary report.
  2. A modern, styled HTML preview report.
"""

import argparse
from collections.abc import Sequence
import csv
import datetime
import html
import io
import json
import logging
import os
import subprocess
import sys
from typing import Any, Optional

from automation import metrics_aggregator
from automation import utils
from automation import workload_submitter

DEFAULT_OUTPUT_REPORT = "/tmp/tpums_daily_report.md"
_REPO_URL = "https://github.com/AI-Hypercomputer/accelerator-microbenchmarks"
CATEGORY_LEADING_COLUMNS = metrics_aggregator.CATEGORY_LEADING_COLUMNS

CATEGORY_TITLES = {
    "collectives": (
        "📊 Collectives Communication",
        (
            "Inter-node / Inter-chip all-gather, all-reduce, all-to-all,"
            " reduce-scatter"
        ),
    ),
    "hbm": (
        "💾 High Bandwidth Memory (HBM)",
        "On-device HBM memory copy bandwidth",
    ),
    "gemm": (
        "⚡ Generalized Matrix Multiplication (GEMM)",
        "Compute intensity and per-device TFLOP/s",
    ),
    "host_device": (
        "🖥️ Host-to-Device (H2D / D2H)",
        "PCIe / host memory transfer bandwidth",
    ),
    "device_to_device": (
        "🔗 Device-to-Device (D2D)",
        "Peer-to-peer inter-chip transfer bandwidth and latency",
    ),
}


def count_statuses(
    results: Sequence[workload_submitter.BenchmarkResult],
) -> tuple[int, int]:
  """Returns (succeeded, failed) JobSet counts."""
  statuses = [r.status for r in results]
  return statuses.count("SUCCESS"), sum(
      s in utils.FAILED_STATUSES for s in statuses
  )


def _markdown_table(
    headers: Sequence[str], rows: Sequence[dict[str, Any]]
) -> list[str]:
  """Renders `rows` as Markdown pipe-table lines."""
  return [
      "| " + " | ".join(headers) + " |",
      "| " + " | ".join(["---"] * len(headers)) + " |",
      *(
          "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
          for row in rows
      ),
  ]


def _html_table(
    headers: Sequence[str],
    rows: Sequence[dict[str, Any]],
    pad: str,
    style: str = "",
) -> list[str]:
  """Renders `rows` as a `table-container` <div> wrapping an HTML <table>.

  Args:
    headers: Column names, in display order.
    rows: Rows to render; missing cells are blank.
    pad: Indentation of the wrapper <div>; the table is nested 2 spaces in.
    style: Optional inline style for the wrapper <div>.

  Returns:
    HTML lines.
  """
  style_attr = f' style="{style}"' if style else ""
  inner = pad + "  "
  lines = [
      f'{pad}<div class="table-container"{style_attr}>',
      f"{inner}<table>",
      f"{inner}  <thead>",
      f"{inner}    <tr>",
  ]
  lines += [f"{inner}      <th>{html.escape(h)}</th>" for h in headers]
  lines += [f"{inner}    </tr>", f"{inner}  </thead>", f"{inner}  <tbody>"]
  for row in rows:
    lines.append(f"{inner}    <tr>")
    lines += [
        f"{inner}      <td>{html.escape(str(row.get(h, '')))}</td>"
        for h in headers
    ]
    lines.append(f"{inner}    </tr>")
  lines += [f"{inner}  </tbody>", f"{inner}</table>", f"{pad}</div>"]
  return lines


def _has_xprof_d2d_bandwidth(rows: Sequence[dict[str, Any]]) -> bool:
  """Returns True if xprof_bandwidth_per_device_gb_s is present and non-empty."""
  return any(
      r.get("xprof_bandwidth_per_device_gb_s") not in (None, "") for r in rows
  )


def format_d2d_pairwise_matrix(
    rows: list[dict[str, Any]],
) -> Optional[tuple[list[str], list[list[str]]]]:
  """Builds an N x N pairwise bandwidth matrix from D2D benchmark rows.

  Uses `xprof_bandwidth_per_device_gb_s` primarily, falling back to
  `wall_clock_bandwidth_per_device_gb_s` if the XProf column is not present in
  the rows.

  Args:
    rows: Device-to-device metric rows with src/dst device indices.

  Returns:
    (sorted_device_labels, matrix_rows) where each cell is a formatted
    bandwidth string (e.g. '85.50') or '-' for self/missing entries, or None
    if rows contain no valid device indices.
  """
  devs = set()
  bw_map: dict[tuple[int, int], str] = {}
  use_xprof = _has_xprof_d2d_bandwidth(rows)
  for r in rows:
    try:
      src = int(float(r.get("src_device_index", -1)))
      dst = int(float(r.get("dst_device_index", -1)))
    except (ValueError, TypeError):
      continue
    if src < 0 or dst < 0:
      continue
    devs.add(src)
    devs.add(dst)
    bw = r.get(
        "xprof_bandwidth_per_device_gb_s"
        if use_xprof
        else "wall_clock_bandwidth_per_device_gb_s"
    )
    bw = bw if bw is not None else ""
    try:
      bw_val = f"{float(bw):.2f}"
    except (ValueError, TypeError):
      bw_val = str(bw) if bw else "-"
    bw_map[(src, dst)] = bw_val

  if not devs:
    return None

  sorted_devs = sorted(devs)
  dev_labels = [f"D{d}" for d in sorted_devs]
  matrix = []
  for src in sorted_devs:
    row = []
    for dst in sorted_devs:
      if src == dst:
        row.append("-")
      else:
        row.append(bw_map.get((src, dst), "-"))
    matrix.append(row)
  return dev_labels, matrix


def fetch_aggregated_csv_results(
    gcs_bucket: str, date_str: Optional[str] = None
) -> dict[str, list[dict[str, Any]]]:
  """Fetches the aggregated category CSV files from GCS."""
  if not gcs_bucket:
    return {}

  target_prefix = utils.format_gcs_uri(
      gcs_bucket, date_str, "aggregated_results"
  )
  cmd = ["gcloud", "storage", "ls", f"{target_prefix}/*.csv"]
  res = subprocess.run(
      cmd,
      capture_output=True,
      text=True,
      check=False,
      stdin=subprocess.DEVNULL,
      timeout=utils.CLI_TIMEOUT_SECONDS,
  )
  if res.returncode != 0:
    # Covers both "matched no objects" and auth errors.
    logging.warning(
        "No aggregated CSVs listed under %s (gcloud exit %d): %s",
        target_prefix,
        res.returncode,
        (res.stderr or "").strip() or "<no stderr>",
    )
    return {}
  if not res.stdout.strip():
    return {}

  category_files = [
      line for line in res.stdout.split() if line.endswith(".csv")
  ]

  categorized: dict[str, list[dict[str, Any]]] = {}
  for c_file in category_files:
    cat_name = c_file.rsplit("/", 1)[-1].removesuffix(".csv")
    cat_cmd = ["gcloud", "storage", "cat", c_file]
    cat_res = subprocess.run(
        cat_cmd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
    if cat_res.returncode != 0:
      logging.error(
          "Failed to read %s (gcloud exit %d): %s",
          c_file,
          cat_res.returncode,
          (cat_res.stderr or "").strip() or "<no stderr>",
      )
      continue
    if cat_res.stdout.strip():
      reader = csv.DictReader(io.StringIO(cat_res.stdout))
      categorized[cat_name] = [dict(row) for row in reader]

  return categorized


def _gcloud_storage_stdout(args: Sequence[str]) -> Optional[str]:
  """Returns the stdout of a best-effort `gcloud storage` call, or None.

  Environment metadata is optional, so a non-zero exit (logged with stderr),
  empty output or a timeout skips the value instead of failing the report.

  Args:
    args: Arguments after `gcloud storage` (e.g. ['ls', 'gs://...']).

  Returns:
    The stripped stdout, or None if the call failed, timed out or printed
    nothing.
  """
  cmd = ["gcloud", "storage", *args]
  try:
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
  except subprocess.TimeoutExpired as e:
    logging.warning("`%s` timed out after %ss.", " ".join(cmd), e.timeout)
    return None
  if proc.returncode != 0:
    logging.warning(
        "`%s` failed (gcloud exit %d): %s",
        " ".join(cmd),
        proc.returncode,
        (proc.stderr or "").strip() or "<no stderr>",
    )
    return None
  return proc.stdout.strip() or None


def fetch_environment_metadata(
    gcs_bucket: str,
    date_str: Optional[str] = None,
    results: Optional[Sequence[workload_submitter.BenchmarkResult]] = None,
) -> dict[str, Optional[str]]:
  """Extracts runtime environment metadata (git commit, JAX version, libtpu version)."""
  metadata: dict[str, Optional[str]] = {
      utils.ENV_GIT_COMMIT: None,
      utils.ENV_JAX_VERSION: None,
      utils.ENV_LIBTPU_VERSION: None,
  }

  target_prefix = utils.format_gcs_uri(gcs_bucket, date_str)

  # 1. Look for detailed.json across results or GCS
  detailed_files = []
  if results:
    for r in results:
      gcs_path = r.gcs_artifact_path
      if gcs_path:
        detailed_files.append(utils.format_gcs_uri(gcs_path, "detailed.json"))
  if not detailed_files and target_prefix:
    listing = _gcloud_storage_stdout(["ls", f"{target_prefix}/*/detailed.json"])
    if listing:
      detailed_files = [
          line for line in listing.split() if line.endswith(".json")
      ]

  # TODO(b/544832391): Support strict verification or multi-environment
  # reporting when workloads under the same prefix have drifted versions.
  for d_file in detailed_files:
    content = _gcloud_storage_stdout(["cat", d_file])
    if not content:
      continue
    try:
      data = json.loads(content)
      item = (
          data[0]
          if isinstance(data, list) and data
          else (data if isinstance(data, dict) else {})
      )
      pinfo = item.get("metadata", {}).get("platform_info", {})
      for key in (utils.ENV_JAX_VERSION, utils.ENV_LIBTPU_VERSION):
        raw_val = pinfo.get(key)
        if not raw_val:
          continue
        new_val = str(raw_val)
        existing_val = metadata[key]
        if not existing_val:
          metadata[key] = new_val
        elif existing_val != new_val:
          logging.warning(
              "Mismatched %s across workloads in %s: %s vs %s",
              key,
              target_prefix,
              existing_val,
              new_val,
          )
    except (ValueError, AttributeError) as e:
      logging.warning("Skipping unparseable %s: %s", d_file, e)
      continue

  # 2. Look for git_commit.txt on GCS
  if not metadata[utils.ENV_GIT_COMMIT] and target_prefix:
    # A single URL: multi-URL `ls` exits non-zero if any URL matches nothing,
    # even when others match.
    git_listing = _gcloud_storage_stdout(
        ["ls", f"{target_prefix}/*/git_commit.txt"]
    )
    git_lines = [
        line for line in (git_listing or "").split() if line.endswith(".txt")
    ]
    for git_file in git_lines:
      commit = _gcloud_storage_stdout(["cat", git_file])
      if commit:
        commit_val = commit.splitlines()[0].strip()
        if not commit_val:
          continue
        existing_commit = metadata[utils.ENV_GIT_COMMIT]
        if not existing_commit:
          metadata[utils.ENV_GIT_COMMIT] = commit_val
        elif existing_commit != commit_val:
          logging.warning(
              "Mismatched git_commit across workloads in %s: %s vs %s",
              target_prefix,
              existing_commit,
              commit_val,
          )

  return metadata


def get_display_headers(
    cat_key: str,
    rows: Sequence[dict[str, Any]],
    leading_columns_only: bool = False,
) -> list[str]:
  """Returns ordered display column headers for a benchmark category table.

  Args:
    cat_key: Benchmark category key (e.g. `"collectives"`, `"hbm"`,
      `"device_to_device"`, `"gemm"`, `"host_device"`).
    rows: Sequence of parsed CSV row dictionaries for the category.
    leading_columns_only: If True, returns only the curated primary parameter
      and hardware-timed `xprof_*` metric columns defined in
      `CATEGORY_LEADING_COLUMNS[cat_key]`, omitting trailing execution metadata
      and wall-clock columns (such as `warmup_tries`, `num_runs`, and
      `wall_clock_*`). If False, appends all remaining discovered CSV columns
      after the leading columns.

  Returns:
    Ordered list of column header names present in `rows`.
  """
  if not rows:
    return []
  target_cols = CATEGORY_LEADING_COLUMNS.get(cat_key, ())
  discovered_cols = []
  seen = set()
  for r in rows:
    for k in r.keys():
      if k not in seen and k != "_source_path":
        seen.add(k)
        discovered_cols.append(k)
  leading = [h for h in target_cols if h in seen]
  if leading_columns_only:
    return leading
  trailing = [h for h in discovered_cols if h not in leading]
  return leading + trailing


def format_consolidated_report(
    results: Sequence[workload_submitter.BenchmarkResult],
    categorized_metrics: Optional[dict[str, list[dict[str, Any]]]] = None,
    cluster: str = utils.DEFAULT_CLUSTER,
    project: str = utils.DEFAULT_PROJECT,
    region: str = utils.DEFAULT_REGION,
    namespace: str = utils.DEFAULT_NAMESPACE,
    git_commit: Optional[str] = None,
    jax_version: Optional[str] = None,
    libtpu_version: Optional[str] = None,
    include_run_header: bool = True,
    leading_columns_only: bool = False,
    d2d_matrix_only: bool = False,
) -> str:
  """Formats a consolidated Markdown report summarizing aggregated category results.

  Args:
    results: List of JobSet execution result objects or dicts.
    categorized_metrics: Mapping from category key to list of CSV row dicts.
    cluster: GKE cluster name shown in the run header.
    project: GCP project ID shown in the run header.
    region: GCP region shown in the run header.
    namespace: Kubernetes namespace shown in the run header.
    git_commit: Optional Git commit hash of the benchmark repository.
    jax_version: Optional JAX version string.
    libtpu_version: Optional libtpu version string.
    include_run_header: If True, includes the top-level daily automated run
      title, cluster/namespace banner, and JobSet execution summary. If False,
      omits the run-specific header (retaining the `**Environment:**` line and
      category tables).
    leading_columns_only: Passed to `get_display_headers` to restrict each
      category table to the curated primary columns in
      `CATEGORY_LEADING_COLUMNS`.
    d2d_matrix_only: If True, omits the collapsible raw D2D transfer table when
      the Pairwise Bandwidth Matrix is rendered.

  Returns:
    Formatted Markdown report string.
  """
  now_str = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y-%m-%d %H:%M:%S UTC"
  )
  total_jobsets = len(results)
  is_dry_run = bool(results) and all(r.status == "DRY_RUN" for r in results)
  successful_jobsets, failed_jobsets = count_statuses(results)

  if is_dry_run:
    summary_line = (
        f"**Execution Summary:** JobSets: {total_jobsets} Planned (Dry-Run"
        " Preview)"
    )
  else:
    summary_line = (
        "**Execution Summary:** JobSets:"
        f" {successful_jobsets}/{total_jobsets} Succeeded ({failed_jobsets}"
        " Failed)"
    )

  env_items = []
  if git_commit:
    short_commit = git_commit[:7]
    commit_url = f"{_REPO_URL}/commit/{git_commit}"
    env_items.append(f"**GitHub Commit:** [`{short_commit}`]({commit_url})")
  if jax_version:
    env_items.append(f"**JAX:** `{jax_version}`")
  if libtpu_version:
    env_items.append(f"**libtpu:** `{libtpu_version}`")

  lines: list[str] = []
  if include_run_header:
    lines.extend([
        f"# 🚀 TPUMS Daily Automated Benchmark Report — {now_str}",
        "",
        (
            f"**Cluster:** `{cluster}` (`{project}` / `{region}`) |"
            f" **Namespace:** `{namespace}`"
        ),
        summary_line,
    ])
  if env_items:
    lines.append("**Environment:** " + " | ".join(env_items))
  if lines:
    lines.append("")
  lines.extend([
      "## Aggregated Benchmark Results by Category",
      "",
  ])

  if categorized_metrics and any(categorized_metrics.values()):
    for cat_key, rows in categorized_metrics.items():
      if not rows:
        continue
      headers = get_display_headers(
          cat_key, rows, leading_columns_only=leading_columns_only
      )
      if not headers:
        continue
      title, desc = CATEGORY_TITLES.get(
          cat_key, (cat_key.capitalize(), "Benchmark metrics")
      )
      if cat_key == "device_to_device" and not _has_xprof_d2d_bandwidth(rows):
        title = f"{title} (Wall Clock Time)"
      lines.append(f"### {title}")
      lines.append(f"*{desc}*")
      lines.append("")

      if cat_key == "device_to_device":
        matrix_data = format_d2d_pairwise_matrix(rows)
        if matrix_data:
          labels, matrix_rows = matrix_data
          lines.append("#### Pairwise Bandwidth Matrix (GB/s)")
          lines.append("```")
          col_w = 8
          header_line = f"{'':6s}" + "".join(
              f"{lbl:>{col_w}s}" for lbl in labels
          )
          lines.append(header_line)
          for lbl, r_vals in zip(labels, matrix_rows):
            r_line = f"{lbl:6s}" + "".join(f"{val:>{col_w}s}" for val in r_vals)
            lines.append(r_line)
          lines.append("```")
          lines.append("")
          if d2d_matrix_only:
            continue

        lines.append("<details>")
        lines.append(
            f"<summary>View Raw Device-to-Device Transfer Table ({len(rows)}"
            " rows)</summary>"
        )
        lines.append("")
        lines.extend(_markdown_table(headers, rows))
        lines.extend(["", "</details>", ""])
      else:
        lines.extend(_markdown_table(headers, rows))
        lines.append("")
  else:
    lines.append("*No aggregated metric tables available.*")
    lines.append("")

  return "\n".join(lines)


def generate_html_report(
    results: Sequence[workload_submitter.BenchmarkResult],
    categorized_metrics: Optional[dict[str, list[dict[str, Any]]]] = None,
    cluster: str = utils.DEFAULT_CLUSTER,
    project: str = utils.DEFAULT_PROJECT,
    region: str = utils.DEFAULT_REGION,
    namespace: str = utils.DEFAULT_NAMESPACE,
    git_commit: Optional[str] = None,
    jax_version: Optional[str] = None,
    libtpu_version: Optional[str] = None,
) -> str:
  """Generates a modern, responsive standalone HTML report preview with rich styling."""
  now_str = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y-%m-%d %H:%M:%S UTC"
  )

  # Compute summary counts
  total_jobsets = len(results)
  successful_jobsets, failed_jobsets = count_statuses(results)
  total_duration = max((r.duration_seconds for r in results), default=0.0)

  html_lines = [
      "<!DOCTYPE html>",
      '<html lang="en">',
      "<head>",
      '  <meta charset="UTF-8">',
      (
          '  <meta name="viewport" content="width=device-width,'
          ' initial-scale=1.0">'
      ),
      f"  <title>TPUMS Daily Benchmark Report — {html.escape(now_str)}</title>",
      "  <style>",
      "    :root {",
      (
          "      --bg: #0f172a; --card-bg: #1e293b; --border: #334155; --text:"
          " #f8fafc; --muted: #94a3b8;"
      ),
      (
          "      --primary: #38bdf8; --success: #34d399; --danger: #f87171;"
          " --warning: #fbbf24;"
      ),
      "    }",
      "    * { box-sizing: border-box; margin: 0; padding: 0; }",
      (
          "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe"
          " UI', Roboto, Helvetica, Arial, sans-serif; background-color:"
          " var(--bg); color: var(--text); padding: 32px 16px; line-height:"
          " 1.5; }"
      ),
      "    .container { max-width: 1200px; margin: 0 auto; }",
      (
          "    .header { margin-bottom: 28px; border-bottom: 1px solid"
          " var(--border); padding-bottom: 20px; }"
      ),
      (
          "    .header h1 { font-size: 26px; font-weight: 700; color:"
          " var(--primary); margin-bottom: 8px; display: flex; align-items:"
          " center; gap: 8px; }"
      ),
      (
          "    .meta-pills { display: flex; gap: 12px; flex-wrap: wrap;"
          " margin-top: 10px; }"
      ),
      (
          "    .pill { background: #334155; padding: 4px 12px; border-radius:"
          " 9999px; font-size: 13px; color: #cbd5e1; }"
      ),
      (
          "    .stat-grid { display: grid; grid-template-columns:"
          " repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom:"
          " 32px; }"
      ),
      (
          "    .stat-card { background: var(--card-bg); border: 1px solid"
          " var(--border); border-radius: 12px; padding: 20px; }"
      ),
      (
          "    .stat-card .label { font-size: 13px; color: var(--muted);"
          " text-transform: uppercase; font-weight: 600; letter-spacing:"
          " 0.5px; }"
      ),
      (
          "    .stat-card .value { font-size: 28px; font-weight: 700;"
          " margin-top: 4px; }"
      ),
      "    .stat-card.success .value { color: var(--success); }",
      "    .stat-card.danger .value { color: var(--danger); }",
      "    .section { margin-bottom: 40px; }",
      (
          "    .section-title { font-size: 20px; font-weight: 600;"
          " margin-bottom: 14px; color: #e2e8f0; display: flex; align-items:"
          " center; gap: 8px; }"
      ),
      (
          "    .table-container { background: var(--card-bg); border: 1px solid"
          " var(--border); border-radius: 12px; overflow-x: auto;"
          " margin-bottom: 24px; }"
      ),
      (
          "    table { width: 100%; border-collapse: collapse; text-align:"
          " left; font-size: 14px; }"
      ),
      (
          "    th { background-color: #1e293b; color: #94a3b8; font-weight:"
          " 600; padding: 12px 16px; border-bottom: 1px solid var(--border);"
          " text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px;"
          " white-space: nowrap; }"
      ),
      (
          "    td { padding: 12px 16px; border-bottom: 1px solid #1e293b;"
          " color: #e2e8f0; white-space: nowrap; }"
      ),
      "    tr:last-child td { border-bottom: none; }",
      "    tr:hover td { background-color: rgba(51, 65, 85, 0.4); }",
      (
          "    code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco,"
          " Consolas, monospace; background: #334155; padding: 2px 6px;"
          " border-radius: 4px; font-size: 13px; }"
      ),
      (
          "    .desc-text { font-size: 13px; color: var(--muted);"
          " margin-bottom: 10px; margin-top: -6px; }"
      ),
      (
          "    .matrix-container { background: var(--card-bg); border: 1px"
          " solid var(--border); border-radius: 12px; padding: 16px 20px;"
          " overflow-x: auto; margin-top: -12px; margin-bottom: 28px; }"
      ),
      (
          "    .matrix-title { font-size: 14px; font-weight: 600; color:"
          " #94a3b8; margin-bottom: 12px; display: flex; align-items: center;"
          " gap: 6px; }"
      ),
      (
          "    .matrix-table { width: auto; min-width: 320px; border-collapse:"
          " collapse; font-family: monospace; font-size: 13px; }"
      ),
      (
          "    .matrix-table th { background: #1e293b; color: #38bdf8;"
          " text-align: center; padding: 8px 14px; border: 1px solid"
          " var(--border); font-weight: 600; }"
      ),
      (
          "    .matrix-table td { text-align: center; padding: 8px 14px;"
          " border: 1px solid #334155; }"
      ),
      (
          "    .matrix-diag { background: #0f172a; color: #64748b; font-weight:"
          " bold; }"
      ),
      "    .matrix-cell { color: #34d399; font-weight: 600; }",
      "    .d2d-details { margin-bottom: 24px; }",
      (
          "    .d2d-summary { cursor: pointer; font-size: 13px; font-weight:"
          " 500; color: #38bdf8; padding: 8px 14px; background: #1e293b;"
          " border: 1px solid var(--border); border-radius: 8px; display:"
          " inline-block; user-select: none; margin-bottom: 8px; }"
      ),
      "    .d2d-summary:hover { background: #334155; }",
      "  </style>",
      "</head>",
      "<body>",
      '  <div class="container">',
      '    <div class="header">',
      "      <h1>🚀 TPUMS Automated Benchmark Report</h1>",
      '      <div class="meta-pills">',
      f'        <span class="pill">🕒 {html.escape(now_str)}</span>',
      (
          '        <span class="pill">🖥️ Cluster:'
          f" <b>{html.escape(cluster)}</b></span>"
      ),
      (
          '        <span class="pill">📁 Project:'
          f" <b>{html.escape(project)}</b></span>"
      ),
      (
          '        <span class="pill">📍 Region:'
          f" <b>{html.escape(region)}</b></span>"
      ),
      (
          '        <span class="pill">🏷️ Namespace:'
          f" <b>{html.escape(namespace)}</b></span>"
      ),
  ]
  if git_commit:
    short_commit = git_commit[:7]
    commit_url = f"{_REPO_URL}/commit/{git_commit}"
    html_lines.append(
        '        <span class="pill">🐙 GitHub: <a'
        f' href="{html.escape(commit_url)}" target="_blank" style="color:'
        " #38bdf8; text-decoration:"
        f' none;"><code>{html.escape(short_commit)}</code></a></span>'
    )
  if jax_version:
    html_lines.append(
        '        <span class="pill">⚡ JAX:'
        f" <b>{html.escape(jax_version)}</b></span>"
    )
  if libtpu_version:
    html_lines.append(
        '        <span class="pill">🔧 libtpu:'
        f" <b>{html.escape(libtpu_version)}</b></span>"
    )
  html_lines.extend([
      "      </div>",
      "    </div>",
      "",
      '    <div class="stat-grid">',
      (
          '      <div class="stat-card"><div class="label">Total'
          f' JobSets</div><div class="value">{total_jobsets}</div></div>'
      ),
      (
          '      <div class="stat-card success"><div'
          ' class="label">Successful JobSets</div><div'
          f' class="value">{successful_jobsets}</div></div>'
      ),
      (
          '      <div class="stat-card'
          f" {'danger' if failed_jobsets > 0 else ''}\"><div"
          ' class="label">Failed JobSets</div><div'
          f' class="value">{failed_jobsets}</div></div>'
      ),
      (
          '      <div class="stat-card"><div class="label">Total Suite'
          f' Duration</div><div class="value">{total_duration:.1f}s</div></div>'
      ),
      "    </div>",
  ])

  if categorized_metrics and any(categorized_metrics.values()):
    html_lines.append('    <div class="section">')
    html_lines.append(
        '      <h2 class="section-title">📈 Aggregated Benchmark Metrics</h2>'
    )

    for cat_key, rows in categorized_metrics.items():
      if not rows:
        continue
      title, desc = CATEGORY_TITLES.get(
          cat_key, (cat_key.capitalize(), "Benchmark metrics")
      )
      if cat_key == "device_to_device" and not _has_xprof_d2d_bandwidth(rows):
        title = f"{title} (Wall Clock Time)"
      headers = metrics_aggregator.ordered_columns(cat_key, rows)

      html_lines.append(
          '      <h3 style="margin-top: 24px; margin-bottom: 6px; font-size:'
          f' 16px;">{html.escape(title)}</h3>'
      )
      html_lines.append(f'      <p class="desc-text">{html.escape(desc)}</p>')

      if cat_key == "device_to_device":
        matrix_data = format_d2d_pairwise_matrix(rows)
        if matrix_data:
          labels, matrix_rows = matrix_data
          html_lines.append(
              '      <div class="matrix-container" style="margin-top: 4px;'
              ' margin-bottom: 12px;">'
          )
          html_lines.append(
              '        <div class="matrix-title">🔗 Pairwise Bandwidth Matrix'
              " (GB/s)</div>"
          )
          html_lines.append('        <table class="matrix-table">')
          html_lines.append("          <thead>")
          html_lines.append("            <tr>")
          html_lines.append("              <th>Src \\ Dst</th>")
          for lbl in labels:
            html_lines.append(f"              <th>{html.escape(lbl)}</th>")
          html_lines.append("            </tr>")
          html_lines.append("          </thead>")
          html_lines.append("          <tbody>")
          for src_lbl, r_vals in zip(labels, matrix_rows):
            html_lines.append("            <tr>")
            html_lines.append(
                "              <th style='background:#1e293b;"
                f" color:#38bdf8;'>{html.escape(src_lbl)}</th>"
            )
            for dst_lbl, val in zip(labels, r_vals):
              if src_lbl == dst_lbl:
                html_lines.append(
                    "              <td class='matrix-diag'>-</td>"
                )
              else:
                html_lines.append(
                    "              <td"
                    f" class='matrix-cell'>{html.escape(val)}</td>"
                )
            html_lines.append("            </tr>")
          html_lines.append("          </tbody>")
          html_lines.append("        </table>")
          html_lines.append("      </div>")

        html_lines.append('      <details class="d2d-details">')
        html_lines.append(
            '        <summary class="d2d-summary">View Raw Device-to-Device'
            f" Transfer Table ({len(rows)} rows)</summary>"
        )
        html_lines.extend(
            _html_table(headers, rows, " " * 8, style="margin-top: 8px;")
        )
        html_lines.append("      </details>")
      else:
        html_lines.extend(_html_table(headers, rows, " " * 6))

    html_lines.append("    </div>")

  html_lines.extend([
      "  </div>",
      "</body>",
      "</html>",
  ])
  return "\n".join(html_lines)


def generate_and_publish_report(
    results: Sequence[workload_submitter.BenchmarkResult],
    categorized_metrics: Optional[dict[str, list[dict[str, Any]]]] = None,
    cluster: str = utils.DEFAULT_CLUSTER,
    project: str = utils.DEFAULT_PROJECT,
    region: str = utils.DEFAULT_REGION,
    namespace: str = utils.DEFAULT_NAMESPACE,
    output_file: str = DEFAULT_OUTPUT_REPORT,
    gcs_bucket: str = utils.DEFAULT_GCS_BUCKET,
    date_str: Optional[str] = None,
    dry_run: bool = False,
    git_commit: Optional[str] = None,
    jax_version: Optional[str] = None,
    libtpu_version: Optional[str] = None,
) -> tuple[str, str]:
  """Generates Markdown and HTML reports, writes local files, and publishes to GCS."""
  if not gcs_bucket and not dry_run:
    raise ValueError("gcs_bucket is required when dry_run is False.")
  if not date_str:
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

  # Auto-fetch aggregated CSV results from GCS if not provided in-memory
  if not categorized_metrics and not dry_run:
    categorized_metrics = fetch_aggregated_csv_results(
        gcs_bucket, date_str=date_str
    )

  # Auto-discover workload run results from GCS if not provided in-memory.
  # Discovered dicts carry every BenchmarkResult field except these two.
  if not results and not dry_run:
    results = [
        workload_submitter.BenchmarkResult(
            config_rel_path="", duration_seconds=0.0, **d
        )
        for d in metrics_aggregator.discover_and_fetch_gcs_results(
            gcs_bucket, date_str=date_str
        )
    ]

  # Auto-discover runtime environment metadata if not explicitly provided
  if not dry_run and not (git_commit and jax_version and libtpu_version):
    discovered_env = fetch_environment_metadata(
        gcs_bucket=gcs_bucket, date_str=date_str, results=results
    )
    git_commit = git_commit or discovered_env.get(utils.ENV_GIT_COMMIT)
    jax_version = jax_version or discovered_env.get(utils.ENV_JAX_VERSION)
    libtpu_version = libtpu_version or discovered_env.get(
        utils.ENV_LIBTPU_VERSION
    )

  report_md = format_consolidated_report(
      results=results,
      categorized_metrics=categorized_metrics,
      cluster=cluster,
      project=project,
      region=region,
      namespace=namespace,
      git_commit=git_commit,
      jax_version=jax_version,
      libtpu_version=libtpu_version,
  )

  report_html = generate_html_report(
      results=results,
      categorized_metrics=categorized_metrics,
      cluster=cluster,
      project=project,
      region=region,
      namespace=namespace,
      git_commit=git_commit,
      jax_version=jax_version,
      libtpu_version=libtpu_version,
  )

  if dry_run:
    print("\n[Dry-Run: Report Generator - Markdown Report Generated (dry-run)]")
    print("[Dry-Run: Report Generator - HTML Preview Generated (dry-run)]")
    return report_md, report_html

  # Write Markdown file
  with open(output_file, "w", encoding="utf-8") as f:
    f.write(report_md)
  logging.info("Saved consolidated Markdown report to %s", output_file)

  # Write HTML preview file
  html_output_path = os.path.splitext(output_file)[0] + ".html"
  with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(report_html)
  logging.info("Saved HTML preview report to %s", html_output_path)

  # Publish both to GCS
  gcs_md_uri = utils.format_gcs_uri(
      gcs_bucket, date_str, "consolidated_report.md"
  )
  gcs_html_uri = utils.format_gcs_uri(
      gcs_bucket, date_str, "consolidated_report.html"
  )

  upload_failures: list[str] = []
  for local_path, target_uri, label in (
      (output_file, gcs_md_uri, "Markdown"),
      (html_output_path, gcs_html_uri, "HTML preview"),
  ):
    try:
      proc = subprocess.run(
          ["gcloud", "storage", "cp", local_path, target_uri],
          capture_output=True,
          text=True,
          check=False,
          stdin=subprocess.DEVNULL,
          timeout=utils.CLI_TIMEOUT_SECONDS,
      )
    except subprocess.TimeoutExpired as e:
      upload_failures.append(label)
      logging.error(
          "Timed out uploading %s report to %s after %ss.",
          label,
          target_uri,
          e.timeout,
      )
      continue
    if proc.returncode == 0:
      logging.info("Uploaded %s report to %s", label, target_uri)
    else:
      upload_failures.append(label)
      logging.error(
          "Failed to upload %s report to %s (gcloud exit %d): %s",
          label,
          target_uri,
          proc.returncode,
          (proc.stderr or "").strip() or "<no stderr>",
      )

  if upload_failures:
    raise RuntimeError(
        "Failed to upload consolidated report(s) to GCS: "
        + ", ".join(upload_failures)
    )

  return report_md, report_html


def main(argv: Sequence[str]) -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Generate and publish TPUMS benchmark summary reports"
          " (Markdown & HTML)."
      )
  )
  parser.add_argument("--cluster", default=utils.DEFAULT_CLUSTER)
  parser.add_argument("--project", default=utils.DEFAULT_PROJECT)
  parser.add_argument("--region", default=utils.DEFAULT_REGION)
  parser.add_argument("--namespace", default=utils.DEFAULT_NAMESPACE)
  parser.add_argument("--gcs-bucket", default=utils.DEFAULT_GCS_BUCKET)
  parser.add_argument(
      "--date",
      default=None,
      help="Target date YYYY-MM-DD (default: current UTC date).",
  )
  parser.add_argument("--output-report", default=DEFAULT_OUTPUT_REPORT)
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help=(
          "Generate reports in dry-run mode without writing local files or"
          " uploading to GCS."
      ),
  )
  parser.add_argument("--git-commit", default=None, help="GitHub commit hash.")
  parser.add_argument(
      "--jax-version", default=None, help="JAX runtime version."
  )
  parser.add_argument("--libtpu-version", default=None, help="libtpu version.")

  args = parser.parse_args(argv[1:])
  if not args.gcs_bucket and not args.dry_run:
    raise ValueError(
        "Must specify --gcs-bucket or set GCS_BUCKET environment variable."
    )
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  generate_and_publish_report(
      results=[],
      categorized_metrics=None,
      cluster=args.cluster,
      project=args.project,
      region=args.region,
      namespace=args.namespace,
      output_file=args.output_report,
      gcs_bucket=args.gcs_bucket,
      date_str=args.date,
      dry_run=args.dry_run,
      git_commit=args.git_commit,
      jax_version=args.jax_version,
      libtpu_version=args.libtpu_version,
  )


if __name__ == "__main__":
  main(sys.argv)
