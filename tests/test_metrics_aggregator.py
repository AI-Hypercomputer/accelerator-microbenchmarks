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

"""Unit tests for TPUMS metrics_aggregator module."""

import subprocess
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from automation import metrics_aggregator


class MetricsAggregatorTest(parameterized.TestCase):
  """Tests for metrics_aggregator categorization and upload."""

  def test_aggregate_category_results(self):
    """Rows are categorized by path with leading columns first."""
    raw_data = [
        {
            "topology": "2x2x1",
            "gcs_artifact_path": "gs://test-bucket/2026-08-27/tpums-2x2x1-gemm",
            "config_name": "gemm",
            "metrics": [{
                "_source_path": (
                    "gs://test-bucket/2026-08-27/tpums-2x2x1-gemm/summary.csv"
                ),
                "m": 2048,
                "n": 2048,
                "k": 2048,
                "in_dtype": "bf16",
                "out_dtype": "bf16",
                "num_runs": "10",
                "tflops_per_device": "240.5",
                "p50_ms": "1.25",
                "warmup_tries": "10",
                "hardware_stats": "dummy_stats",
            }],
        },
        {
            "topology": "2x2x2",
            "gcs_artifact_path": (
                "gs://test-bucket/2026-08-27/tpums-2x2x2-all-reduce"
            ),
            "config_name": "all_reduce",
            "metrics": [{
                "_source_path": (
                    "gs://test-bucket/2026-08-27/tpums-2x2x2-all-reduce/summary.csv"
                ),
                "mesh_shape": "16x4x2",
                "sharding_strategy": "16x4x1",
                "matrix_dim": "1000",
                "dtype": "bf16",
                "num_runs": "10",
                "replica_group_rank": "16",
                "shard_size_mib": "512.0",
                "bandwidth_per_chip_gb_s": "380.0",
                "p50_ms": "1.31",
                "warmup_tries": "5",
                "hardware_stats": "dummy_stats",
            }],
        },
    ]
    categorized = metrics_aggregator.aggregate_and_upload_category_results(
        raw_results=raw_data,
        gcs_bucket="test-bucket",
        date_str="2026-08-27",
        dry_run=True,
    )
    self.assertIn("gemm", categorized)
    self.assertIn("collectives", categorized)
    self.assertLen(categorized["gemm"], 1)
    self.assertLen(categorized["collectives"], 1)

    gemm_row = categorized["gemm"][0]
    self.assertNotIn("_source_path", gemm_row)
    self.assertIn("warmup_tries", gemm_row)
    self.assertIn("hardware_stats", gemm_row)
    self.assertEqual(
        list(gemm_row.keys()),
        [
            "in_dtype",
            "out_dtype",
            "m",
            "k",
            "n",
            "num_runs",
            "tflops_per_device",
            "p50_ms",
            "warmup_tries",
            "hardware_stats",
        ],
    )

    coll_row = categorized["collectives"][0]
    self.assertNotIn("_source_path", coll_row)
    self.assertIn("warmup_tries", coll_row)
    self.assertIn("hardware_stats", coll_row)
    self.assertEqual(coll_row.get("topology"), "2x2x2")
    self.assertEqual(coll_row.get("mesh_shape"), "16x4x2")
    self.assertEqual(coll_row.get("sharding_strategy"), "16x4x1")
    self.assertEqual(coll_row.get("matrix_dim"), "1000")
    self.assertEqual(coll_row.get("replica_group_rank"), "16")
    self.assertEqual(coll_row.get("shard_size_mib"), "512.0")
    self.assertEqual(coll_row.get("p50_ms"), "1.31")
    self.assertEqual(
        list(coll_row.keys()),
        [
            "topology",
            "mesh_shape",
            "sharding_strategy",
            "dtype",
            "matrix_dim",
            "replica_group_rank",
            "shard_size_mib",
            "num_runs",
            "bandwidth_per_chip_gb_s",
            "p50_ms",
            "warmup_tries",
            "hardware_stats",
        ],
    )

  @parameterized.named_parameters(
      ("h2d_config", "configs/tpu7x/2x2x1/host_to_device.yaml", "host_device"),
      ("d2h_config", "configs/tpu7x/2x2x1/device_to_host.yaml", "host_device"),
      (
          "h2d_gcs",
          "gs://bucket/date/tpums-2x2x1-host-to-device/summary.csv",
          "host_device",
      ),
      (
          "d2h_gcs",
          "gs://bucket/date/tpums-2x2x1-device-to-host/summary.csv",
          "host_device",
      ),
      (
          "d2d_gcs_hyphens",
          "gs://bucket/daily/tpums-2x2x1-device-to-device/summary.csv",
          "device_to_device",
      ),
      (
          "d2d_local",
          "/tmp/results/device_to_device/summary.csv",
          "device_to_device",
      ),
      ("v6e_collectives", "configs/v6e/2x4/all_gather.yaml", "collectives"),
      ("v6e_gemm", "configs/v6e/2x2/gemm.yaml", "gemm"),
      ("v6e_d2d", "configs/v6e/2x2/device_to_device.yaml", "device_to_device"),
      (
          "unknown_workload",
          "gs://bucket/date/tpums-unknown-workload/summary.csv",
          None,
      ),
      ("unrecognized_file", "unrecognized_path.csv", None),
  )
  def test_infer_category_from_path(self, path, expected):
    """Categories come from path fragments; hyphens match underscores."""
    self.assertEqual(
        metrics_aggregator.infer_category_from_path(path), expected
    )

  def test_aggregate_skips_none_category(self):
    """Rows from unrecognized paths are dropped."""
    raw_data = [{
        "topology": "2x2x1",
        "gcs_artifact_path": "gs://bucket/date/unknown-workload",
        "config_name": "unknown_workload",
        "metrics": [{
            "_source_path": "gs://bucket/date/unknown-workload/output.csv",
            "val": 123,
        }],
    }]
    categorized = metrics_aggregator.aggregate_and_upload_category_results(
        raw_results=raw_data,
        gcs_bucket="test-bucket",
        date_str="2026-08-27",
        dry_run=True,
    )
    for unused_cat, rows in categorized.items():
      self.assertEmpty(rows)

  @parameterized.named_parameters(
      (
          "host_device",
          "host_device",
          {"benchmark": "HostToDeviceBenchmark", "data_size_mib": "4096"},
      ),
      (
          "hbm",
          "hbm",
          {"benchmark": "HBMBandwidthBenchmark", "p50_ms": "0.85"},
      ),
      (
          "gemm",
          "gemm",
          {"benchmark": "GemmBenchmark", "tflops_per_device": "1089.4"},
      ),
  )
  def test_normalize_row_passthrough(self, category, row):
    """Other categories are copied unchanged, minus `_source_path`."""
    norm = metrics_aggregator.normalize_row(
        {**row, "_source_path": "gs://b/d/wl/summary.csv"}, category=category
    )
    self.assertEqual(norm, row)

  def test_normalize_row_collectives(self):
    """Result-level topology fills a missing column but never overrides it."""
    row = {"benchmark": "AllGatherBenchmark", "p50_ms": "12.0"}
    self.assertEqual(
        metrics_aggregator.normalize_row(row, "collectives", topology="4x4x4"),
        {**row, "topology": "4x4x4"},
    )
    self.assertEqual(
        metrics_aggregator.normalize_row(
            {**row, "topology": "2x2x2"}, "collectives", topology="4x4x4"
        )["topology"],
        "2x2x2",
    )
    # No topology column is injected when topology is unknown.
    self.assertNotIn(
        "topology", metrics_aggregator.normalize_row(row, "collectives")
    )

  def test_normalize_row_device_to_device(self):
    """D2D device indices are cast to int."""
    raw = {
        "benchmark": "device_to_device",
        "dtype": "float32",
        "direction": "uni",
        "src_device_index": "0",
        "dst_device_index": "1.0",
        "data_size_mib": "256",
        "bandwidth_per_device_gb_s": "85.5",
        "avg_ms": "3.14",
        "p50_ms": "3.10",
        "p90_ms": "3.20",
        "xprof_p50_ms": "3.08",
        "num_runs": "5",
    }
    norm = metrics_aggregator.normalize_row(raw, category="device_to_device")
    # Only the device indices are cast to int; other columns pass through.
    self.assertEqual(
        norm, {**raw, "src_device_index": 0, "dst_device_index": 1}
    )

  def test_aggregate_requires_gcs_bucket(self):
    with self.assertRaises(ValueError):
      metrics_aggregator.aggregate_and_upload_category_results(
          raw_results=[],
          gcs_bucket="",
          date_str="2026-09-14",
          dry_run=False,
      )

  def test_ordered_columns(self):
    """Leading columns come first, then the rest in first-seen order."""
    rows = [
        {"extra": 1, "dtype": "bf16"},
        {"benchmark": "hbm", "later": 2},
    ]
    self.assertEqual(
        metrics_aggregator.ordered_columns("hbm", rows),
        ["benchmark", "dtype", "extra", "later"],
    )

  def test_aggregate_uploads_csv_via_stdin(self):
    """Aggregated CSVs stream to gcloud via stdin, not /tmp."""
    raw_data = [{
        "topology": "2x2x1",
        "metrics": [{
            "_source_path": "gs://bucket/date/tpums-2x2x1-gemm/summary.csv",
            "m": 2048,
            "benchmark": "gemm",
        }],
    }]
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
    with mock.patch.object(subprocess, "run", return_value=ok) as mock_run:
      metrics_aggregator.aggregate_and_upload_category_results(
          raw_results=raw_data,
          gcs_bucket="test-bucket",
          date_str="2026-09-14",
          dry_run=False,
      )
    mock_run.assert_called_once()
    self.assertEqual(
        mock_run.call_args.args[0],
        [
            "gcloud",
            "storage",
            "cp",
            "-",
            "gs://test-bucket/2026-09-14/aggregated_results/gemm.csv",
        ],
    )
    self.assertEqual(
        mock_run.call_args.kwargs["input"], "benchmark,m\r\ngemm,2048\r\n"
    )

  def test_aggregate_raises_and_logs_on_upload_failure(self):
    """Upload failures are logged with stderr and raised."""
    raw_data = [{
        "workload_name": "tpums-2x2x1-gemm",
        "topology": "2x2x1",
        "metrics": [{
            "_source_path": "gs://bucket/date/tpums-2x2x1-gemm/summary.csv",
            "m": 2048,
            "p50_ms": "1.25",
        }],
    }]
    failed = subprocess.CompletedProcess(
        args=["gcloud"], returncode=1, stdout="", stderr="AccessDeniedException"
    )
    with mock.patch.object(subprocess, "run", return_value=failed):
      with self.assertLogs(level="ERROR") as logs:
        with self.assertRaises(RuntimeError):
          metrics_aggregator.aggregate_and_upload_category_results(
              raw_results=raw_data,
              gcs_bucket="test-bucket",
              date_str="2026-09-14",
              dry_run=False,
          )
    self.assertIn("AccessDeniedException", "\n".join(logs.output))

  def test_aggregate_upload_timeout_continues_then_raises(self):
    """A hung category upload doesn't block the others; it raises at the end."""
    raw_data = [{
        "workload_name": "wl",
        "topology": "2x2x1",
        "metrics": [
            {"_source_path": "gs://b/d/tpums-2x2x1-gemm/summary.csv", "m": 1},
            {"_source_path": "gs://b/d/tpums-2x2x1-hbm/summary.csv", "x": 2},
        ],
    }]
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
    with mock.patch.object(
        subprocess,
        "run",
        side_effect=[subprocess.TimeoutExpired("gcloud", 300), ok],
    ) as mock_run:
      with self.assertLogs(level="ERROR"):
        with self.assertRaisesRegex(RuntimeError, "Failed to upload"):
          metrics_aggregator.aggregate_and_upload_category_results(
              raw_results=raw_data,
              gcs_bucket="test-bucket",
              date_str="2026-09-14",
              dry_run=False,
          )
    self.assertEqual(mock_run.call_count, 2)

  def test_fetch_gcs_raw_metrics_logs_unreadable_summary(self):
    """A missing or unreadable summary.csv is logged, not silently empty."""
    failed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="NotFoundException"
    )
    with mock.patch.object(subprocess, "run", return_value=failed):
      with self.assertLogs(level="WARNING") as logs:
        rows = metrics_aggregator.fetch_gcs_raw_metrics("gs://b/d/wl")
    self.assertEmpty(rows)
    self.assertIn("NotFoundException", "\n".join(logs.output))

  def test_discover_skips_metric_files_without_topology(self):
    """GCS dirs without a topology are skipped, not labeled 2x2x1."""
    good = "gs://b/2026-09-14/tpums-4x4-all-gather/summary.csv"
    bad = "gs://b/2026-09-14/scratch-dir/summary.csv"

    def fake_run(cmd, **unused_kwargs):
      if cmd[2] == "ls":
        out = f"{good}\n{bad}\n" if cmd[3].endswith("summary.csv") else ""
      else:  # gcloud storage cat <file>
        out = "benchmark,xprof_p50_ms\nall_gather,1.0\n"
      return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    with mock.patch.object(subprocess, "run", side_effect=fake_run):
      with self.assertLogs(level="WARNING") as logs:
        discovered = metrics_aggregator.discover_and_fetch_gcs_results(
            "gs://b", "2026-09-14"
        )
    self.assertLen(discovered, 1)
    self.assertEqual(discovered[0]["topology"], "4x4")
    self.assertEqual(discovered[0]["workload_name"], "tpums-4x4-all-gather")
    self.assertIn("scratch-dir", "\n".join(logs.output))

  def test_discover_logs_gcloud_ls_failure(self):
    """Non-zero gcloud storage ls exit logs exit code and stderr."""
    failed = subprocess.CompletedProcess(
        args=["gcloud", "storage", "ls"],
        returncode=1,
        stdout="",
        stderr="AccessDeniedException: 403",
    )
    with mock.patch.object(subprocess, "run", return_value=failed):
      with self.assertLogs(level="WARNING") as logs:
        discovered = metrics_aggregator.discover_and_fetch_gcs_results(
            "gs://b", "2026-09-14"
        )
    self.assertEmpty(discovered)
    output = "\n".join(logs.output)
    self.assertIn("exit 1", output)
    self.assertIn("AccessDeniedException", output)


if __name__ == "__main__":
  absltest.main()
