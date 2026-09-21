#!/usr/bin/env python3
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

"""Ingest, Normalize, and Aggregate TPUMS Benchmark Metrics Across Categories.

Categorizes results strictly via the directory and file paths in the GCS bucket,
standardized across benchmark categories.
"""

from collections.abc import Sequence
import csv
import datetime
import io
import logging
import os
import re
import subprocess
import sys
from typing import Any, Optional

DEFAULT_GCS_BUCKET = os.environ.get("GCS_BUCKET", "")

DEFAULT_TOPOLOGY = "2x2x1"

# Matches a TPU topology such as "2x2x1" or "4x4x8" delimited by a non-digit
# boundary, so "2x2x16" is never truncated to "2x2x1".
_TOPOLOGY_RE = re.compile(r"(?<!\d)(\d+x\d+x\d+)(?!\d)")


def format_gcs_uri(bucket: str, *subpaths: str) -> str:
  clean = bucket.rstrip("/")
  base = clean if clean.startswith("gs://") else f"gs://{clean}"
  if not subpaths:
    return base
  return f"{base}/{'/'.join(subpaths)}"


def infer_topology_from_gcs_path(*candidates: str) -> str:
  """Extracts the TPU topology from a workload name or GCS path.

  Args:
    *candidates: Strings to scan, in priority order (e.g. workload name then
      full GCS path).

  Returns:
    The first topology found, or DEFAULT_TOPOLOGY if none match.
  """
  for candidate in candidates:
    match = _TOPOLOGY_RE.search(str(candidate))
    if match:
      return match.group(1)
  logging.warning(
      "Could not infer topology from %s; defaulting to %s.",
      " or ".join(repr(c) for c in candidates),
      DEFAULT_TOPOLOGY,
  )
  return DEFAULT_TOPOLOGY


def verify_gcs_bucket_accessible(gcs_bucket: str) -> None:
  """Fails fast if the GCS bucket is missing or not readable by this identity.

  Args:
    gcs_bucket: Bucket name or gs:// prefix to probe.

  Raises:
    RuntimeError: If the bucket cannot be listed.
  """
  bucket_uri = format_gcs_uri(gcs_bucket)
  proc = subprocess.run(
      ["gcloud", "storage", "ls", bucket_uri],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    raise RuntimeError(
        f"GCS bucket {bucket_uri} is not accessible (exit"
        f" {proc.returncode}): {proc.stderr.strip()}"
    )


# Leading cross-workload discriminator and metric columns placed before raw CSV columns.
CATEGORY_LEADING_COLUMNS: dict[str, tuple[str, ...]] = {
    "collectives": (
        "topology",
        "mesh_shape",
        "sharding_strategy",
        "benchmark",
        "dtype",
        "matrix_dim",
        "replica_group_rank",
        "shard_size_mib",
        "xprof_p50_ms",
        "xprof_bandwidth_per_chip_gb_s",
    ),
    "hbm": (
        "benchmark",
        "dtype",
        "matrix_dim",
        "xprof_p50_ms",
        "xprof_bandwidth_per_device_gb_s",
    ),
    "host_device": (
        "benchmark",
        "dtype",
        "matrix_dim",
        "xprof_p50_ms",
        "xprof_bandwidth_per_device_gb_s",
    ),
    "device_to_device": (
        "benchmark",
        "dtype",
        "matrix_dim",
        "xprof_p50_ms",
        "xprof_bandwidth_per_device_gb_s",
    ),
    "gemm": (
        "benchmark",
        "in_dtype",
        "out_dtype",
        "m",
        "k",
        "n",
        "xprof_p50_ms",
        "xprof_tflops_per_device",
    ),
}


# Maps a report category to the path fragments that identify it. Fragments use
# underscore separators only; `infer_category_from_path` normalizes hyphens
# before matching.
_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "collectives",
        (
            "collectives",
            "all_gather",
            "all_reduce",
            "all_to_all",
            "reduce_scatter",
        ),
    ),
    ("hbm", ("hbm",)),
    (
        "host_device",
        (
            "host_to_device",
            "device_to_host",
            "host_device",
        ),
    ),
    ("device_to_device", ("device_to_device",)),
    ("gemm", ("gemm",)),
)

SUPPORTED_CATEGORIES: tuple[str, ...] = tuple(
    cat for cat, _ in _CATEGORY_PATTERNS
)


def infer_category_from_path(gcs_or_file_path: str) -> Optional[str]:
  """Categorizes results purely through directory and file path in GCS."""
  path_lower = (
      str(gcs_or_file_path).lower().replace("\\", "/").replace("-", "_")
  )
  for cat, patterns in _CATEGORY_PATTERNS:
    if any(p in path_lower for p in patterns):
      return cat
  return None


def normalize_row(
    row: dict[str, Any], category: str, topology: str = DEFAULT_TOPOLOGY
) -> dict[str, Any]:
  """Injects cross-workload aggregation columns not present in per-benchmark summary.csv."""
  norm = dict(row)

  if category == "collectives":
    norm.setdefault("topology", topology)

  elif category == "device_to_device":
    for dev_col in ("src_device_index", "dst_device_index"):
      if dev_col in norm and norm[dev_col] is not None and norm[dev_col] != "":
        try:
          norm[dev_col] = int(float(norm[dev_col]))
        except (ValueError, TypeError):
          pass

  return norm


def _parse_gcs_metric_file(m_file: str) -> list[dict[str, Any]]:
  """Helper to download and parse a single CSV/TSV file from GCS."""
  proc = subprocess.run(
      ["gcloud", "storage", "cat", m_file],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0 or not proc.stdout.strip():
    return []
  delimiter = "\t" if m_file.endswith(".tsv") else ","
  reader = csv.DictReader(io.StringIO(proc.stdout), delimiter=delimiter)
  rows = []
  for r in reader:
    d = dict(r)
    d["_source_path"] = m_file
    rows.append(d)
  return rows


def fetch_gcs_raw_metrics(gcs_output_dir: str) -> list[dict[str, Any]]:
  """Downloads and parses metric CSV/TSVs directly for a workload output directory."""
  clean_dir = gcs_output_dir.rstrip("/")
  res = subprocess.run(
      ["gcloud", "storage", "ls", f"{clean_dir}/*"],
      capture_output=True,
      text=True,
      check=False,
  )
  if res.returncode != 0 or not res.stdout.strip():
    return []

  metric_files = [
      line.strip()
      for line in res.stdout.strip().splitlines()
      if (line.strip().endswith(".csv") or line.strip().endswith(".tsv"))
      and not line.strip().endswith("/")
  ]
  all_rows = []
  for mf in metric_files:
    all_rows.extend(_parse_gcs_metric_file(mf))
  return all_rows


def discover_and_fetch_gcs_results(
    gcs_base: str, date_str: str
) -> list[dict[str, Any]]:
  """Discovers all workload metric files on GCS in a fast single pass."""
  clean_base = gcs_base.rstrip("/")
  if clean_base and not clean_base.startswith("gs://"):
    clean_base = f"gs://{clean_base}"
  target_prefix = f"{clean_base}/{date_str}" if date_str else clean_base

  metric_files = []
  for pattern in [
      f"{target_prefix}/*/summary.csv",
      f"{target_prefix}/*/*.csv",
      f"{target_prefix}/*/*.tsv",
  ]:
    res = subprocess.run(
        ["gcloud", "storage", "ls", pattern],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0 and res.stdout.strip():
      for line in res.stdout.strip().splitlines():
        line_clean = line.strip()
        if (
            line_clean
            and not line_clean.endswith("/")
            and "/tensorboard/" not in line_clean
            and "/aggregated_results/" not in line_clean
            and line_clean not in metric_files
        ):
          metric_files.append(line_clean)

  if not metric_files:
    logging.warning("No metric files found on GCS under %s", target_prefix)
    return []

  discovered = []
  for mf in metric_files:
    parts = mf.split("/")
    wl_name = parts[-2] if len(parts) >= 2 else parts[-1]
    if wl_name in ("aggregated_results", "daily_benchmarks"):
      continue

    topo = infer_topology_from_gcs_path(wl_name, mf)

    rows = _parse_gcs_metric_file(mf)
    if rows:
      discovered.append({
          "workload_name": wl_name,
          "config_name": wl_name,
          "gcs_artifact_path": mf.rsplit("/", 1)[0],
          "topology": topo,
          "status": "SUCCESS",
          "metrics": rows,
      })

  logging.info(
      "Discovered %d workload metric file(s) on GCS containing %d total"
      " row(s).",
      len(discovered),
      sum(len(r["metrics"]) for r in discovered),
  )
  return discovered


def aggregate_and_upload_category_results(
    raw_results: list[dict[str, Any]],
    gcs_bucket: str,
    date_str: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, list[dict[str, Any]]]:
  """Categorizes results strictly by GCS/file path and uploads aggregated CSVs to GCS."""
  if not gcs_bucket and not dry_run:
    raise ValueError(
        "gcs_bucket is required when dry_run is False; callers must resolve it"
        " from --gcs-bucket or the GCS_BUCKET environment variable."
    )
  if not date_str:
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

  if not raw_results and not dry_run and gcs_bucket:
    raw_results = discover_and_fetch_gcs_results(gcs_bucket, date_str=date_str)

  categorized: dict[str, list[dict[str, Any]]] = {
      cat: [] for cat in SUPPORTED_CATEGORIES
  }

  for r in raw_results:
    metrics = r.get("metrics", [])
    topology = r.get("topology", DEFAULT_TOPOLOGY)
    gcs_path = (
        r.get("gcs_artifact_path", "")
        or r.get("workload_name", "")
        or r.get("config_name", "")
    )

    for row in metrics:
      source_path = row.get("_source_path") or gcs_path
      cat = infer_category_from_path(source_path)
      if cat is None or cat not in SUPPORTED_CATEGORIES:
        logging.warning(
            "Unknown or unmapped category for path '%s'; skipping row.",
            source_path,
        )
        continue
      norm_row = normalize_row(row, category=cat, topology=topology)
      norm_row.pop("_source_path", None)

      leading_cols = CATEGORY_LEADING_COLUMNS.get(cat, ())
      ordered_keys = [k for k in leading_cols if k in norm_row] + [
          k for k in norm_row if k not in leading_cols
      ]
      filtered_row = {k: norm_row[k] for k in ordered_keys}
      categorized[cat].append(filtered_row)

  if dry_run:
    print(
        "\n[Dry-Run: Metrics Aggregator - Path-Based Aggregation Output Plan"
        " (CSV-Derived Schema)]"
    )
    for cat in SUPPORTED_CATEGORIES:
      sample_gcs = format_gcs_uri(
          gcs_bucket,
          date_str,
          "aggregated_results",
          f"{cat}.csv",
      )
      rows = categorized.get(cat, [])
      cols = (
          list(rows[0].keys())
          if rows
          else list(CATEGORY_LEADING_COLUMNS.get(cat, ()))
      )
      cols_preview = f" (Columns: {', '.join(cols[:6])}...)" if cols else ""
      print(f"* Category '{cat:<15}' -> {sample_gcs}{cols_preview}")
    print()
    return categorized

  if gcs_bucket:
    upload_failures: list[str] = []
    for cat, rows in categorized.items():
      if not rows:
        continue
      leading_schema = CATEGORY_LEADING_COLUMNS.get(cat, ())
      seen_cols: set[str] = set()
      discovered_cols: list[str] = []
      for r_dict in rows:
        for k in r_dict.keys():
          if k not in seen_cols:
            seen_cols.add(k)
            discovered_cols.append(k)
      leading_cols = [c for c in leading_schema if c in seen_cols]
      trailing_cols = [c for c in discovered_cols if c not in leading_cols]
      available_cols = leading_cols + trailing_cols or list(rows[0].keys())

      csv_buffer = io.StringIO()
      writer = csv.DictWriter(
          csv_buffer,
          fieldnames=available_cols,
          delimiter=",",
          extrasaction="ignore",
      )
      writer.writeheader()
      for r_dict in rows:
        writer.writerow(r_dict)

      local_tmp_path = f"/tmp/tpums_aggregated_{cat}.csv"
      with open(local_tmp_path, "w", encoding="utf-8") as f:
        f.write(csv_buffer.getvalue())

      gcs_uri = format_gcs_uri(
          gcs_bucket, date_str, "aggregated_results", f"{cat}.csv"
      )
      proc = subprocess.run(
          ["gcloud", "storage", "cp", local_tmp_path, gcs_uri],
          capture_output=True,
          text=True,
          check=False,
      )
      if proc.returncode == 0:
        logging.info(
            "Saved aggregated %s CSV (%d rows) to %s", cat, len(rows), gcs_uri
        )
      else:
        upload_failures.append(cat)
        logging.error(
            "Failed to upload aggregated %s CSV (%d rows) to %s (gcloud exit"
            " %d): %s",
            cat,
            len(rows),
            gcs_uri,
            proc.returncode,
            proc.stderr.strip() or "<no stderr>",
        )

    if upload_failures:
      raise RuntimeError(
          "Failed to upload aggregated CSVs for categories: "
          + ", ".join(sorted(upload_failures))
      )

  return categorized


def main(argv: Sequence[str]) -> None:
  import argparse  # pylint: disable=g-import-not-at-top

  parser = argparse.ArgumentParser(
      description="Aggregate TPUMS benchmark metrics by GCS path."
  )
  parser.add_argument(
      "--gcs-bucket",
      default=DEFAULT_GCS_BUCKET,
      help="GCS bucket/prefix for metrics.",
  )
  parser.add_argument(
      "--date",
      default=None,
      help="Target date YYYY-MM-DD (default: current UTC date).",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print aggregation target schemas without network writes.",
  )

  args = parser.parse_args(argv[1:])
  if not args.gcs_bucket and not args.dry_run:
    raise ValueError(
        "Must specify --gcs-bucket or set GCS_BUCKET environment variable."
    )
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  if not args.dry_run:
    verify_gcs_bucket_accessible(args.gcs_bucket)
  aggregate_and_upload_category_results(
      raw_results=[],
      gcs_bucket=args.gcs_bucket,
      date_str=args.date,
      dry_run=args.dry_run,
  )


if __name__ == "__main__":
  main(sys.argv)
