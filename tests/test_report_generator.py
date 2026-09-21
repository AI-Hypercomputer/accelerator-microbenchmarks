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

"""Unit tests for TPUMS report_generator module."""

import json
import os
import subprocess
import tempfile
import time
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from automation import metrics_aggregator
from automation import report_generator
from automation import workload_submitter
import yaml


class ReportGeneratorTest(parameterized.TestCase):

  def test_format_consolidated_report(self):
    result = workload_submitter.BenchmarkResult(
        workload_name="tpums-gemm-01",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        status="SUCCESS",
        duration_seconds=120.5,
        gcs_artifact_path="gs://test-bucket/daily/gemm",
        metrics=[{
            "m": 2048,
            "k": 2048,
            "n": 2048,
            "tflops_per_device": "240.5",
            "step_time_ms_avg": "1.25",
            "warmup_tries": "10",
        }],
    )
    categorized = {
        "gemm": [{
            "m": 2048,
            "k": 2048,
            "n": 2048,
            "tflops_per_device": "240.5",
            "step_time_ms_avg": "1.25",
            "warmup_tries": "10",
        }],
        "collectives": [{
            "topology": "4x4x4",
            "mesh_shape": "16x4x2",
            "sharding_strategy": "16x4x1",
            "matrix_dim": "1000",
            "dtype": "bf16",
            "num_runs": "10",
            "replica_group_rank": "16",
            "shard_size_mib": "512.0",
            "bandwidth_per_chip_gb_s": "380.0",
            "p50_ms": "1.31",
            "custom_dynamic_col": "active",
        }],
    }
    report_md = report_generator.format_consolidated_report(
        [result],
        categorized_metrics=categorized,
        cluster="example-tpu-cluster",
        project="example-gcp-project",
        region="us-central1",
        namespace="default",
        git_commit="9bcfb129e8b6f85210ffb9b6fddc466235db799a",
        jax_version="0.11.1",
        libtpu_version="0.0.46.1",
    )
    self.assertIn("TPUMS Daily Automated Benchmark Report", report_md)
    self.assertIn("Execution Summary", report_md)
    self.assertIn("JobSets: 1/1 Succeeded", report_md)
    self.assertIn("example-tpu-cluster", report_md)
    self.assertIn("240.5", report_md)
    self.assertIn("Generalized Matrix Multiplication", report_md)
    self.assertIn("mesh_shape", report_md)
    self.assertIn("sharding_strategy", report_md)
    self.assertIn("16x4x2", report_md)
    self.assertIn("16x4x1", report_md)
    self.assertIn("matrix_dim", report_md)
    self.assertIn("replica_group_rank", report_md)
    self.assertIn("shard_size_mib", report_md)
    self.assertIn(
        "| topology | mesh_shape | sharding_strategy | dtype | matrix_dim |"
        " replica_group_rank | shard_size_mib | num_runs |",
        report_md,
    )
    self.assertIn("custom_dynamic_col", report_md)
    self.assertIn("**Environment:**", report_md)
    self.assertIn("GitHub Commit:", report_md)
    self.assertIn("9bcfb12", report_md)
    self.assertIn("JAX:", report_md)
    self.assertIn("0.11.1", report_md)
    self.assertIn("libtpu:", report_md)
    self.assertIn("0.0.46.1", report_md)
    self.assertIn("warmup_tries", report_md)

    report_html = report_generator.generate_html_report(
        [result],
        categorized_metrics=categorized,
        cluster="example-tpu-cluster",
        project="example-gcp-project",
        region="us-central1",
        namespace="default",
        git_commit="9bcfb129e8b6f85210ffb9b6fddc466235db799a",
        jax_version="0.11.1",
        libtpu_version="0.0.46.1",
    )
    self.assertIn("<!DOCTYPE html>", report_html)
    self.assertIn("TPUMS Automated Benchmark Report", report_html)
    self.assertIn("Successful JobSets", report_html)
    self.assertIn("Failed JobSets", report_html)
    self.assertIn("example-tpu-cluster", report_html)
    self.assertIn("240.5", report_html)
    self.assertIn("mesh_shape", report_html)
    self.assertIn("sharding_strategy", report_html)
    self.assertIn("16x4x2", report_html)
    self.assertIn("16x4x1", report_html)
    self.assertIn("matrix_dim", report_html)
    self.assertIn("shard_size_mib", report_html)
    self.assertIn("custom_dynamic_col", report_html)
    self.assertIn("🐙 GitHub:", report_html)
    self.assertIn("9bcfb12", report_html)
    self.assertIn("⚡ JAX:", report_html)
    self.assertIn("0.11.1", report_html)
    self.assertIn("🔧 libtpu:", report_html)
    self.assertIn("0.0.46.1", report_html)
    self.assertIn("warmup_tries", report_html)

  def test_fetch_environment_metadata_mocked(self):
    mock_detailed_json = json.dumps([{
        "metadata": {
            "platform_info": {
                "jax_version": "0.11.1",
                "libtpu_version": "0.0.46.1",
                "git_commit": "abcdef1234567890",
            }
        }
    }])
    with mock.patch("subprocess.run") as mock_run:
      mock_run.side_effect = [
          mock.Mock(returncode=0, stdout="gs://bucket/date/wl/detailed.json\n"),
          mock.Mock(returncode=0, stdout=mock_detailed_json),
      ]
      env = report_generator.fetch_environment_metadata(
          gcs_bucket="gs://bucket", date_str="2026-08-27"
      )
      self.assertEqual(env["jax_version"], "0.11.1")
      self.assertEqual(env["libtpu_version"], "0.0.46.1")
      self.assertEqual(env["git_commit"], "abcdef1234567890")

  def test_format_consolidated_report_host_device(self):
    result = workload_submitter.BenchmarkResult(
        workload_name="tpums-h2d-01",
        config_name="host_to_device",
        config_rel_path="configs/tpu7x/2x2x1/host_to_device.yaml",
        topology="2x2x1",
        status="SUCCESS",
        duration_seconds=30.0,
        gcs_artifact_path="gs://test-bucket/daily/h2d",
        metrics=[{
            "data_size_mib": "4096",
            "num_runs": "10",
            "bandwidth_per_device_gb_s": "28.5",
            "time_ms_avg": "145.0",
        }],
    )
    categorized = {
        "host_device": [
            {
                "data_size_mib": "4096",
                "dtype": "float32",
                "num_runs": "10",
                "bandwidth_per_device_gb_s": "28.5",
                "time_ms_avg": "145.0",
                "time_ms_p50": "144.5",
                "time_ms_p90": "146.0",
            },
            {
                "data_size_mib": "4096",
                "dtype": "float32",
                "num_runs": "10",
                "bandwidth_per_device_gb_s": "32.1",
                "time_ms_avg": "130.0",
                "time_ms_p50": "129.5",
                "time_ms_p90": "131.0",
            },
        ]
    }
    report_md = report_generator.format_consolidated_report(
        [result],
        categorized_metrics=categorized,
    )
    self.assertIn("Host-to-Device (H2D / D2H)", report_md)
    self.assertIn(
        "| dtype | data_size_mib | num_runs | bandwidth_per_device_gb_s |",
        report_md,
    )
    self.assertIn("28.5", report_md)
    self.assertIn("32.1", report_md)

    report_html = report_generator.generate_html_report(
        [result],
        categorized_metrics=categorized,
    )
    self.assertIn("Host-to-Device (H2D / D2H)", report_html)
    self.assertIn("<th>dtype</th>", report_html)
    self.assertIn("<td>float32</td>", report_html)
    self.assertIn("bandwidth_per_device_gb_s", report_html)
    self.assertIn("28.5", report_html)
    self.assertIn("32.1", report_html)

  def test_format_d2d_pairwise_matrix(self):
    rows = [
        {
            "src_device_index": "0",
            "dst_device_index": "1",
            "bandwidth_per_device_gb_s": "85.5",
        },
        {
            "src_device_index": "1",
            "dst_device_index": "0",
            "bandwidth_per_device_gb_s": "86.2",
        },
    ]
    matrix_data = report_generator.format_d2d_pairwise_matrix(rows)
    self.assertIsNotNone(matrix_data)
    labels, matrix_rows = matrix_data
    self.assertEqual(labels, ["D0", "D1"])
    self.assertEqual(matrix_rows[0], ["-", "85.50"])
    self.assertEqual(matrix_rows[1], ["86.20", "-"])

  def test_format_consolidated_report_and_html_device_to_device(self):
    result = workload_submitter.BenchmarkResult(
        workload_name="tpums-2x2x1-device-to-device",
        config_name="device_to_device",
        config_rel_path="configs/tpu7x/2x2x1/device_to_device.yaml",
        topology="2x2x1",
        status="SUCCESS",
        duration_seconds=42.0,
        gcs_artifact_path="gs://bucket/date/tpums-2x2x1-device-to-device",
    )
    categorized = {
        "device_to_device": [
            {
                "dtype": "float32",
                "direction": "uni",
                "src_device_index": 0,
                "dst_device_index": 1,
                "data_size_mib": "256",
                "num_runs": "5",
                "bandwidth_per_device_gb_s": "85.50",
                "time_ms_avg": "3.14",
                "time_ms_p50": "3.10",
                "time_ms_p90": "3.20",
                "xprof_p50_ms": "3.08",
            },
            {
                "dtype": "float32",
                "direction": "uni",
                "src_device_index": 1,
                "dst_device_index": 0,
                "data_size_mib": "256",
                "num_runs": "5",
                "bandwidth_per_device_gb_s": "86.20",
                "time_ms_avg": "3.12",
                "time_ms_p50": "3.08",
                "time_ms_p90": "3.18",
                "xprof_p50_ms": "3.05",
            },
        ]
    }
    report_md = report_generator.format_consolidated_report(
        [result], categorized_metrics=categorized
    )
    self.assertIn("Device-to-Device (D2D) (Wall Clock Time)", report_md)
    self.assertIn("Pairwise Bandwidth Matrix (GB/s)", report_md)
    self.assertIn("<details>", report_md)
    self.assertIn(
        "<summary>View Raw Device-to-Device Transfer Table (2 rows)</summary>",
        report_md,
    )
    self.assertIn(
        "| dtype | xprof_p50_ms | direction | src_device_index |"
        " dst_device_index | data_size_mib | num_runs |"
        " bandwidth_per_device_gb_s |",
        report_md,
    )
    self.assertIn("D0", report_md)
    self.assertIn("D1", report_md)
    self.assertIn("85.50", report_md)
    self.assertIn("86.20", report_md)

    report_html = report_generator.generate_html_report(
        [result], categorized_metrics=categorized
    )
    self.assertIn("Device-to-Device (D2D) (Wall Clock Time)", report_html)
    self.assertIn("matrix-container", report_html)
    self.assertIn("Pairwise Bandwidth Matrix (GB/s)", report_html)
    self.assertIn("matrix-table", report_html)
    self.assertIn('<details class="d2d-details">', report_html)
    self.assertIn(
        "View Raw Device-to-Device Transfer Table (2 rows)</summary>",
        report_html,
    )
    self.assertIn("85.50", report_html)
    self.assertIn("86.20", report_html)

  def test_format_consolidated_report_and_html_device_to_device_domain_prefixed(
      self,
  ):
    result = workload_submitter.BenchmarkResult(
        workload_name="tpums-2x2x1-device-to-device",
        config_name="device_to_device",
        config_rel_path="configs/tpu7x/2x2x1/device_to_device.yaml",
        topology="2x2x1",
        status="SUCCESS",
        duration_seconds=42.0,
        gcs_artifact_path="gs://bucket/date/tpums-2x2x1-device-to-device",
    )
    categorized = {
        "device_to_device": [
            {
                "dtype": "float32",
                "direction": "uni",
                "src_device_index": 0,
                "dst_device_index": 1,
                "data_size_mib": "256",
                "num_runs": "5",
                "wall_clock_p50_ms": "0.9944",
                "wall_clock_bandwidth_per_device_gb_s": "269.95",
                "xprof_p50_ms": "0.3959",
                "xprof_bandwidth_per_device_gb_s": "678.02",
            },
            {
                "dtype": "float32",
                "direction": "uni",
                "src_device_index": 1,
                "dst_device_index": 0,
                "data_size_mib": "256",
                "num_runs": "5",
                "wall_clock_p50_ms": "0.9980",
                "wall_clock_bandwidth_per_device_gb_s": "268.50",
                "xprof_p50_ms": "0.3965",
                "xprof_bandwidth_per_device_gb_s": "677.10",
            },
        ]
    }
    report_md = report_generator.format_consolidated_report(
        [result], categorized_metrics=categorized
    )
    self.assertIn("Device-to-Device (D2D)", report_md)
    self.assertNotIn("Wall Clock Time", report_md)
    self.assertIn("Pairwise Bandwidth Matrix (GB/s)", report_md)
    self.assertIn("678.02", report_md)
    self.assertIn("677.10", report_md)

    report_html = report_generator.generate_html_report(
        [result], categorized_metrics=categorized
    )
    self.assertIn("Device-to-Device (D2D)", report_html)
    self.assertNotIn("Wall Clock Time", report_html)
    self.assertIn("Pairwise Bandwidth Matrix (GB/s)", report_html)
    self.assertIn("678.02", report_html)
    self.assertIn("677.10", report_html)

  def test_format_consolidated_report_and_html_device_to_device_wall_clock_fallback(
      self,
  ):
    result = workload_submitter.BenchmarkResult(
        workload_name="tpums-2x2x1-device-to-device",
        config_name="device_to_device",
        config_rel_path="configs/tpu7x/2x2x1/device_to_device.yaml",
        topology="2x2x1",
        status="SUCCESS",
        duration_seconds=42.0,
        gcs_artifact_path="gs://bucket/date/tpums-2x2x1-device-to-device",
    )
    categorized = {
        "device_to_device": [
            {
                "dtype": "float32",
                "direction": "uni",
                "src_device_index": 0,
                "dst_device_index": 1,
                "data_size_mib": "256",
                "num_runs": "5",
                "wall_clock_p50_ms": "0.9944",
                "wall_clock_bandwidth_per_device_gb_s": "269.95",
            },
            {
                "dtype": "float32",
                "direction": "uni",
                "src_device_index": 1,
                "dst_device_index": 0,
                "data_size_mib": "256",
                "num_runs": "5",
                "wall_clock_p50_ms": "0.9980",
                "wall_clock_bandwidth_per_device_gb_s": "268.50",
            },
        ]
    }
    report_md = report_generator.format_consolidated_report(
        [result], categorized_metrics=categorized
    )
    self.assertIn("Device-to-Device (D2D) (Wall Clock Time)", report_md)
    self.assertIn("Pairwise Bandwidth Matrix (GB/s)", report_md)
    self.assertIn("269.95", report_md)
    self.assertIn("268.50", report_md)

    report_html = report_generator.generate_html_report(
        [result], categorized_metrics=categorized
    )
    self.assertIn("Device-to-Device (D2D) (Wall Clock Time)", report_html)
    self.assertIn("Pairwise Bandwidth Matrix (GB/s)", report_html)
    self.assertIn("269.95", report_html)
    self.assertIn("268.50", report_html)

  def test_generate_html_report_dry_run_zero_failures(self):
    dry_run_result = workload_submitter.BenchmarkResult(
        workload_name="tpums-2x2x1-gemm",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        status="DRY_RUN",
        duration_seconds=0.0,
        gcs_artifact_path="",
    )
    html_report = report_generator.generate_html_report([dry_run_result])
    # Total should be 1, but Failed JobSets must be 0, not 1
    self.assertIn(
        '<div class="stat-card"><div class="label">Total JobSets</div><div'
        ' class="value">1</div></div>',
        html_report,
    )
    self.assertIn(
        '<div class="stat-card "><div class="label">Failed JobSets</div><div'
        ' class="value">0</div></div>',
        html_report,
    )
    self.assertNotIn("stat-card danger", html_report)

  def test_generate_html_report_total_suite_duration_uses_max_not_sum(self):
    results = [
        workload_submitter.BenchmarkResult(
            workload_name="tpums-4x4x4-all-gather",
            config_name="all_gather",
            config_rel_path="configs/tpu7x/4x4x4/all_gather.yaml",
            topology="4x4x4",
            status="SUCCESS",
            duration_seconds=120.0,
            gcs_artifact_path="gs://bucket/date/tpums-4x4x4-all-gather",
        ),
        workload_submitter.BenchmarkResult(
            workload_name="tpums-2x2x1-device-to-device",
            config_name="device_to_device",
            config_rel_path="configs/tpu7x/2x2x1/device_to_device.yaml",
            topology="2x2x1",
            status="SUCCESS",
            duration_seconds=300.0,
            gcs_artifact_path="gs://bucket/date/tpums-2x2x1-device-to-device",
        ),
    ]
    html_report = report_generator.generate_html_report(results)
    self.assertIn(
        '<div class="stat-card"><div class="label">Total Suite'
        ' Duration</div><div class="value">300.0s</div></div>',
        html_report,
    )
    self.assertNotIn("420.0s", html_report)


if __name__ == "__main__":
  absltest.main()
