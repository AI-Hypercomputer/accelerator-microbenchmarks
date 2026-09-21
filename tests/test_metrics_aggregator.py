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

"""Unit tests for TPUMS metrics_aggregator module."""

import json
import os
import subprocess
import tempfile
import time
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from automation import metrics_aggregator
import yaml


class MetricsAggregatorTest(parameterized.TestCase):

  def test_aggregate_category_results(self):
    raw_data = [
        {
            "topology": "2x2x1",
            "gcs_artifact_path": (
                "gs://test-bucket/2026-08-27/tpums-2x2x1-gemm"
            ),
            "config_name": "gemm",
            "metrics": [{
                "_source_path": (
                    "gs://test-bucket/2026-08-27/tpums-2x2x1-gemm/gemm.tsv"
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
                    "gs://test-bucket/2026-08-27/tpums-2x2x2-all-reduce/collectives/all_reduce.tsv"
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

  def test_infer_category_host_device(self):
    self.assertEqual(
        metrics_aggregator.infer_category_from_path(
            "configs/tpu7x/2x2x1/host_to_device.yaml"
        ),
        "host_device",
    )
    self.assertEqual(
        metrics_aggregator.infer_category_from_path(
            "configs/tpu7x/2x2x1/device_to_host.yaml"
        ),
        "host_device",
    )
    self.assertEqual(
        metrics_aggregator.infer_category_from_path(
            "gs://bucket/date/tpums-2x2x1-host-to-device/summary.csv"
        ),
        "host_device",
    )
    self.assertEqual(
        metrics_aggregator.infer_category_from_path(
            "gs://bucket/date/tpums-2x2x1-device-to-host/summary.csv"
        ),
        "host_device",
    )

  def test_infer_category_fallback_none(self):
    self.assertIsNone(
        metrics_aggregator.infer_category_from_path(
            "gs://bucket/date/tpums-unknown-workload/summary.csv"
        )
    )
    self.assertIsNone(
        metrics_aggregator.infer_category_from_path("unrecognized_path.csv")
    )

  def test_aggregate_skips_none_category(self):
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

  def test_normalize_row_host_device(self):
    h2d_row = {
        "benchmark": "HostToDeviceBenchmark",
        "data_size_mib": "4096",
        "dtype": "float32",
        "bandwidth_per_device_gb_s": "28.5",
        "avg_ms": "145.0",
        "p50_ms": "144.5",
        "p90_ms": "146.0",
        "num_runs": "10",
    }
    norm_h2d = metrics_aggregator.normalize_row(h2d_row, category="host_device")
    self.assertNotIn("transfer_type", norm_h2d)
    self.assertEqual(norm_h2d["dtype"], "float32")
    self.assertEqual(norm_h2d["bandwidth_per_device_gb_s"], "28.5")
    self.assertEqual(norm_h2d["data_size_mib"], "4096")
    self.assertEqual(norm_h2d["p50_ms"], "144.5")

    d2h_row = {
        "benchmark": "DeviceToHostBenchmark",
        "data_size_mib": "8192",
        "dtype": "bfloat16",
        "bandwidth_per_device_gb_s": "32.1",
        "avg_ms": "260.0",
        "p50_ms": "259.0",
        "p90_ms": "262.0",
        "num_runs": "10",
    }
    norm_d2h = metrics_aggregator.normalize_row(d2h_row, category="host_device")
    self.assertNotIn("transfer_type", norm_d2h)
    self.assertEqual(norm_d2h["dtype"], "bfloat16")
    self.assertEqual(norm_d2h["bandwidth_per_device_gb_s"], "32.1")
    self.assertEqual(norm_d2h["data_size_mib"], "8192")
    self.assertEqual(norm_d2h["p50_ms"], "259.0")

  def test_normalize_row_collectives(self):
    coll_row = {
        "benchmark": "all_reduce",
        "mesh_shape": "16x4x2",
        "sharding_strategy": "16x4x1",
        "matrix_dim": "1000",
        "shard_size_mib": "512.0",
        "bandwidth_per_chip_gb_s": "380.0",
        "p50_ms": "1.31",
    }
    norm = metrics_aggregator.normalize_row(
        coll_row, category="collectives", topology="4x4x4"
    )
    self.assertEqual(norm["topology"], "4x4x4")
    self.assertEqual(norm["mesh_shape"], "16x4x2")
    self.assertEqual(norm["sharding_strategy"], "16x4x1")
    self.assertEqual(norm["matrix_dim"], "1000")
    self.assertEqual(norm["shard_size_mib"], "512.0")
    self.assertEqual(norm["p50_ms"], "1.31")
    self.assertEqual(norm["bandwidth_per_chip_gb_s"], "380.0")
    self.assertNotIn("op_type", norm)

    a2a_row = {
        "benchmark": "all_to_all",
        "mesh_shape": "2x2x2",
        "sharding_strategy": "2x2x1",
        "matrix_dim": "8192",
        "shard_size_mib": "64.0",
        "bandwidth_per_chip_gb_s": "350.0",
        "p50_ms": "0.5",
    }
    norm_a2a = metrics_aggregator.normalize_row(
        a2a_row, category="collectives", topology="2x2x2"
    )
    self.assertEqual(norm_a2a["topology"], "2x2x2")
    self.assertEqual(norm_a2a["shard_size_mib"], "64.0")
    self.assertNotIn("op_type", norm_a2a)

  def test_infer_category_from_path_device_to_device(self):
    paths = [
        "gs://test-bucket/daily/tpums-2x2x1-device-to-device/summary.csv",
        "/tmp/results/device_to_device/metrics.tsv",
        "gs://bucket/date/tpums-2x2x1-device_to_device/summary.csv",
    ]
    for p in paths:
      cat = metrics_aggregator.infer_category_from_path(p)
      self.assertEqual(cat, "device_to_device", f"Failed for {p}")

  def test_normalize_row_device_to_device(self):
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
    self.assertEqual(norm["dtype"], "float32")
    self.assertEqual(norm["direction"], "uni")
    self.assertEqual(norm["src_device_index"], 0)
    self.assertEqual(norm["dst_device_index"], 1)
    self.assertEqual(norm["data_size_mib"], "256")
    self.assertEqual(norm["bandwidth_per_device_gb_s"], "85.5")
    self.assertEqual(norm["p50_ms"], "3.10")
    self.assertEqual(norm["xprof_p50_ms"], "3.08")
    self.assertEqual(norm["num_runs"], "5")

  def test_normalize_row_scoped_schema(self):
    norm_col = metrics_aggregator.normalize_row(
        {
            "benchmark": "AllGatherBenchmark",
            "bandwidth_per_chip_gb_s": "550.5",
            "p50_ms": "12.0",
        },
        "collectives",
        topology="4x4x4",
    )
    self.assertEqual(norm_col["topology"], "4x4x4")
    self.assertNotIn("op_type", norm_col)
    self.assertEqual(norm_col["bandwidth_per_chip_gb_s"], "550.5")
    self.assertEqual(norm_col["p50_ms"], "12.0")

    norm_hbm = metrics_aggregator.normalize_row(
        {
            "benchmark": "HbmBandwidthBenchmark",
            "bandwidth_per_device_gb_s": "3232.0",
            "bandwidth_per_chip_gb_s": "6464.0",
            "p50_ms": "0.85",
        },
        "hbm",
    )
    self.assertEqual(norm_hbm["bandwidth_per_device_gb_s"], "3232.0")
    self.assertEqual(norm_hbm["bandwidth_per_chip_gb_s"], "6464.0")
    self.assertEqual(norm_hbm["p50_ms"], "0.85")

    norm_gemm = metrics_aggregator.normalize_row(
        {
            "benchmark": "GemmBenchmark",
            "tflops_per_device": "1089.4",
            "tflops_per_chip": "2178.8",
            "p50_ms": "1.53",
        },
        "gemm",
    )
    self.assertEqual(norm_gemm["tflops_per_device"], "1089.4")
    self.assertEqual(norm_gemm["tflops_per_chip"], "2178.8")
    self.assertEqual(norm_gemm["p50_ms"], "1.53")

    norm_h2d = metrics_aggregator.normalize_row(
        {
            "benchmark": "HostToDeviceBenchmark",
            "bandwidth_per_device_gb_s": "30.86",
            "p50_ms": "150.0",
        },
        "host_device",
    )
    self.assertNotIn("transfer_type", norm_h2d)
    self.assertEqual(norm_h2d["bandwidth_per_device_gb_s"], "30.86")
    self.assertEqual(norm_h2d["p50_ms"], "150.0")

    norm_d2d = metrics_aggregator.normalize_row(
        {
            "benchmark": "DeviceToDeviceBenchmark",
            "src_device_index": "0",
            "dst_device_index": "1",
            "bandwidth_per_device_gb_s": "679.5",
            "p50_ms": "0.97",
        },
        "device_to_device",
    )
    self.assertEqual(norm_d2d["src_device_index"], 0)
    self.assertEqual(norm_d2d["dst_device_index"], 1)
    self.assertEqual(norm_d2d["bandwidth_per_device_gb_s"], "679.5")
    self.assertEqual(norm_d2d["p50_ms"], "0.97")

  def test_csv_columns_preserved_with_leading_discriminators(self):
    self.assertIn("collectives", metrics_aggregator.SUPPORTED_CATEGORIES)
    self.assertEqual(
        metrics_aggregator.CATEGORY_LEADING_COLUMNS["collectives"],
        (
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
    )

    # Verify aggregation preserves all summary.csv columns verbatim (except _source_path)
    raw_data = [{
        "topology": "2x2x1",
        "gcs_artifact_path": "gs://bucket/date/tpums-2x2x1-gemm",
        "config_name": "gemm",
        "metrics": [{
            "_source_path": "gs://bucket/date/tpums-2x2x1-gemm/summary.csv",
            "m": 2048,
            "n": 2048,
            "k": 2048,
            "in_dtype": "bf16",
            "out_dtype": "bf16",
            "p50_ms": "1.25",
            "python_version": "3.12.14",
            "xprof_dir": "/tmp/tensorboard",
            "warmup_tries": "10",
        }],
    }]
    categorized = metrics_aggregator.aggregate_and_upload_category_results(
        raw_results=raw_data,
        gcs_bucket="test-bucket",
        date_str="2026-08-27",
        dry_run=True,
    )
    row = categorized["gemm"][0]
    self.assertEqual(
        list(row.keys())[:5],
        ["in_dtype", "out_dtype", "m", "k", "n"],
    )
    self.assertEqual(row["m"], 2048)
    self.assertEqual(row["p50_ms"], "1.25")
    self.assertEqual(row["python_version"], "3.12.14")
    self.assertEqual(row["xprof_dir"], "/tmp/tensorboard")
    self.assertEqual(row["warmup_tries"], "10")
    self.assertNotIn("_source_path", row)

  def test_infer_topology_from_gcs_path(self):
    self.assertEqual(
        metrics_aggregator.infer_topology_from_gcs_path(
            "tpums-4x4x8-all-reduce", "gs://b/d/tpums-4x4x8-all-reduce/s.csv"
        ),
        "4x4x8",
    )
    # The workload name wins over the full path.
    self.assertEqual(
        metrics_aggregator.infer_topology_from_gcs_path(
            "tpums-2x2x2-all-gather", "gs://b/2x2x1/x/summary.csv"
        ),
        "2x2x2",
    )

  def test_infer_topology_from_gcs_path_not_truncated_by_prefix(self):
    # The old substring scan matched "2x2x1" inside "2x2x16".
    self.assertEqual(
        metrics_aggregator.infer_topology_from_gcs_path("tpums-2x2x16-a2a"),
        "2x2x16",
    )

  def test_infer_topology_from_gcs_path_defaults_with_warning(self):
    with self.assertLogs(level="WARNING"):
      topo = metrics_aggregator.infer_topology_from_gcs_path("no-topo-here")
    self.assertEqual(topo, metrics_aggregator.DEFAULT_TOPOLOGY)

  def test_aggregate_requires_gcs_bucket(self):
    with self.assertRaises(ValueError):
      metrics_aggregator.aggregate_and_upload_category_results(
          raw_results=[],
          gcs_bucket="",
          date_str="2026-09-14",
          dry_run=False,
      )

  def test_aggregate_raises_and_logs_on_upload_failure(self):
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


if __name__ == "__main__":
  absltest.main()
