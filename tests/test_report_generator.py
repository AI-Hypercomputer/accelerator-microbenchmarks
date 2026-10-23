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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from automation import metrics_aggregator
from automation import report_generator
from automation import utils
from automation import workload_submitter


def _result(**overrides):
  """Builds a successful tpu7x 2x2x1 BenchmarkResult with field overrides."""
  fields = dict(
      workload_name="tpums-2x2x1-gemm",
      config_name="gemm",
      config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
      topology="2x2x1",
      status="SUCCESS",
      duration_seconds=0.0,
      gcs_artifact_path="gs://bucket/date/tpums-2x2x1-gemm",
  )
  fields.update(overrides)
  return workload_submitter.BenchmarkResult(**fields)


class ReportGeneratorTest(parameterized.TestCase):
  """Tests for report_generator Markdown/HTML rendering."""

  def test_format_consolidated_report(self):
    """Markdown and HTML reports include tables and metadata."""
    result = _result(duration_seconds=120.5)
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
    self.assertNotIn("badge", report_html)

  def test_fetch_environment_metadata_mocked(self):
    """Versions come from detailed.json; git_commit from git_commit.txt."""
    mock_detailed_json_partial = json.dumps(
        [
            {
                "metadata": {
                    "platform_info": {
                        "jax_version": "0.11.1",
                    }
                }
            }
        ]
    )
    mock_detailed_json_full = json.dumps([{
        "metadata": {
            "platform_info": {
                "jax_version": "0.11.1",
                "libtpu_version": "0.0.46.1",
            }
        }
    }])
    with (
        mock.patch("subprocess.run") as mock_run,
        self.assertNoLogs(level="WARNING"),
    ):
      mock_run.side_effect = [
          mock.Mock(
              returncode=0,
              stdout=(
                  "gs://bucket/date/wl1/detailed.json\n"
                  "gs://bucket/date/wl2/detailed.json\n"
              ),
          ),
          mock.Mock(returncode=0, stdout=mock_detailed_json_partial),
          mock.Mock(returncode=0, stdout=mock_detailed_json_full),
          mock.Mock(
              returncode=0,
              stdout=(
                  "gs://bucket/date/wl1/git_commit.txt\n"
                  "gs://bucket/date/wl2/git_commit.txt\n"
              ),
          ),
          mock.Mock(returncode=0, stdout="abcdef1234567890\n"),
          mock.Mock(returncode=0, stdout="abcdef1234567890\n"),
      ]
      env = report_generator.fetch_environment_metadata(
          gcs_bucket="gs://bucket", date_str="2026-08-27"
      )
    self.assertEqual(env[utils.ENV_JAX_VERSION], "0.11.1")
    self.assertEqual(env[utils.ENV_LIBTPU_VERSION], "0.0.46.1")
    self.assertEqual(env[utils.ENV_GIT_COMMIT], "abcdef1234567890")
    # A single URL: a multi-URL `ls` fails if any URL matches nothing.
    self.assertEqual(
        mock_run.call_args_list[3].args[0],
        ["gcloud", "storage", "ls", "gs://bucket/2026-08-27/*/git_commit.txt"],
    )

    with mock.patch("subprocess.run") as mock_run_empty:
      empty_env = report_generator.fetch_environment_metadata(gcs_bucket="")
    mock_run_empty.assert_not_called()
    self.assertEqual(empty_env, dict.fromkeys(empty_env))

  def test_fetch_environment_metadata_without_git_commit_leaves_it_empty(self):
    """No git_commit.txt on GCS leaves git_commit empty (no remote lookup)."""
    no_match = "One or more URLs matched no objects."
    with mock.patch("subprocess.run") as mock_run:
      mock_run.side_effect = [
          mock.Mock(returncode=1, stdout="", stderr=no_match),
          mock.Mock(returncode=1, stdout="", stderr=no_match),
      ]  # A 3rd call (e.g. `git ls-remote`) would raise StopIteration.
      with self.assertLogs(level="WARNING") as logs:
        env = report_generator.fetch_environment_metadata(
            gcs_bucket="gs://bucket", date_str="2026-08-27"
        )
    self.assertIsNone(env[utils.ENV_GIT_COMMIT])
    self.assertEqual(mock_run.call_count, 2)
    self.assertIn("matched no objects", "\n".join(logs.output))

  def test_generate_and_publish_report_normalizes_discovered_results(self):
    """Results discovered on GCS are converted to BenchmarkResult."""
    discovered = [{
        "workload_name": "tpums-2x2x1-gemm",
        "config_name": "tpums-2x2x1-gemm",
        "gcs_artifact_path": "gs://bucket/2026-08-27/tpums-2x2x1-gemm",
        "topology": "2x2x1",
        "status": "SUCCESS",
        "metrics": [{"m": "1"}],
    }]
    out = self.create_tempfile("report.md").full_path
    with mock.patch.object(
        report_generator, "fetch_aggregated_csv_results", return_value={}
    ), mock.patch.object(
        report_generator.metrics_aggregator,
        "discover_and_fetch_gcs_results",
        return_value=discovered,
    ), mock.patch.object(
        report_generator,
        "fetch_environment_metadata",
        return_value=dict.fromkeys((
            utils.ENV_GIT_COMMIT,
            utils.ENV_JAX_VERSION,
            utils.ENV_LIBTPU_VERSION,
        )),
    ) as mock_env, mock.patch(
        "subprocess.run", return_value=mock.Mock(returncode=0, stderr="")
    ):
      report_md, _ = report_generator.generate_and_publish_report(
          results=[],
          gcs_bucket="gs://bucket",
          date_str="2026-08-27",
          output_file=out,
      )
    self.assertIn("JobSets: 1/1 Succeeded", report_md)
    (passed,) = mock_env.call_args.kwargs["results"]
    self.assertIsInstance(passed, workload_submitter.BenchmarkResult)
    self.assertEqual(passed.topology, "2x2x1")
    self.assertEqual(passed.metrics, [{"m": "1"}])

  def test_generate_and_publish_report_raises_on_upload_failure(self):
    """Failed report uploads raise after the local files are written."""
    out = self.create_tempfile("report.md").full_path
    with mock.patch(
        "subprocess.run",
        return_value=mock.Mock(returncode=1, stderr="denied"),
    ) as mock_run:
      with self.assertRaisesRegex(RuntimeError, "Markdown, HTML preview"):
        report_generator.generate_and_publish_report(
            results=[_result()],
            categorized_metrics={"gemm": [{"m": "1"}]},
            gcs_bucket="gs://bucket",
            date_str="2026-08-27",
            output_file=out,
            git_commit="abc",
            jax_version="0.11.1",
            libtpu_version="0.0.46.1",
        )
    self.assertEqual(mock_run.call_count, 2)  # Only the two `cp` calls.
    self.assertTrue(os.path.exists(out))
    self.assertTrue(os.path.exists(os.path.splitext(out)[0] + ".html"))

  def test_fetch_environment_metadata_timeout_skips_value(self):
    """A hung metadata lookup is skipped instead of failing the report."""
    with mock.patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired("gcloud", 300),
    ):
      with self.assertLogs(level="WARNING"):
        env = report_generator.fetch_environment_metadata(
            gcs_bucket="gs://bucket", date_str="2026-08-27"
        )
    self.assertEqual(env, dict.fromkeys(env))

  def test_fetch_environment_metadata_skips_unparseable_detailed_json(self):
    """Malformed detailed.json is logged and skipped."""
    with mock.patch("subprocess.run") as mock_run:
      mock_run.side_effect = [
          mock.Mock(returncode=0, stdout="gs://bucket/d/wl/detailed.json\n"),
          mock.Mock(returncode=0, stdout="not json"),
          mock.Mock(
              returncode=1,
              stdout="",
              stderr="One or more URLs matched no objects.",
          ),  # ls */git_commit.txt
      ]
      with self.assertLogs(level="WARNING") as logs:
        env = report_generator.fetch_environment_metadata(
            gcs_bucket="gs://bucket", date_str="2026-08-27"
        )
    self.assertIsNone(env[utils.ENV_JAX_VERSION])
    self.assertIn("Skipping unparseable", "\n".join(logs.output))

  def test_fetch_aggregated_csv_results_logs_unreadable_category(self):
    """An unreadable category CSV is logged; readable ones still render."""
    with mock.patch("subprocess.run") as mock_run:
      mock_run.side_effect = [
          mock.Mock(
              returncode=0,
              stdout=(
                  "gs://b/d/aggregated_results/gemm.csv\n"
                  "gs://b/d/aggregated_results/hbm.csv\n"
              ),
              stderr="",
          ),
          mock.Mock(returncode=0, stdout="m,k\n1,2\n", stderr=""),
          mock.Mock(returncode=1, stdout="", stderr="denied"),
      ]
      with self.assertLogs(level="ERROR") as logs:
        categorized = report_generator.fetch_aggregated_csv_results(
            "gs://b", "d"
        )
    self.assertEqual(categorized, {"gemm": [{"m": "1", "k": "2"}]})
    self.assertIn("hbm.csv", "\n".join(logs.output))

  def test_generate_and_publish_report_dry_run_skips_network(self):
    with mock.patch("subprocess.run") as mock_run:
      report_md, _ = report_generator.generate_and_publish_report(
          results=[_result(status="DRY_RUN")],
          gcs_bucket="gs://bucket",
          dry_run=True,
      )
    mock_run.assert_not_called()
    self.assertIn("JobSets: 1 Planned", report_md)

  def test_generate_and_publish_report_requires_gcs_bucket(self):
    with self.assertRaises(ValueError):
      report_generator.generate_and_publish_report(
          results=[_result()],
          gcs_bucket="",
          dry_run=False,
      )

  def test_fetch_environment_metadata_warns_on_version_or_commit_mismatch(self):
    """Verifies warnings are logged when metadata differs across workloads."""
    mock_detailed_json_1 = json.dumps([{
        "metadata": {
            "platform_info": {
                "jax_version": "0.11.1",
                "libtpu_version": "0.0.46.1",
            }
        }
    }])
    mock_detailed_json_2 = json.dumps([{
        "metadata": {
            "platform_info": {
                "jax_version": "0.11.2",
                "libtpu_version": "0.0.46.2",
            }
        }
    }])
    with (
        mock.patch("subprocess.run") as mock_run,
        self.assertLogs(level="WARNING") as log_ctx,
    ):
      mock_run.side_effect = [
          mock.Mock(
              returncode=0,
              stdout=(
                  "gs://bucket/date/wl1/detailed.json\n"
                  "gs://bucket/date/wl2/detailed.json\n"
              ),
          ),
          mock.Mock(returncode=0, stdout=mock_detailed_json_1),
          mock.Mock(returncode=0, stdout=mock_detailed_json_2),
          mock.Mock(
              returncode=0,
              stdout=(
                  "gs://bucket/date/wl1/git_commit.txt\n"
                  "gs://bucket/date/wl2/git_commit.txt\n"
              ),
          ),
          mock.Mock(returncode=0, stdout="1111111\n"),
          mock.Mock(returncode=0, stdout="2222222\n"),
      ]
      env = report_generator.fetch_environment_metadata(
          gcs_bucket="gs://bucket", date_str="2026-08-27"
      )
    self.assertEqual(env["jax_version"], "0.11.1")
    self.assertEqual(env["libtpu_version"], "0.0.46.1")
    self.assertEqual(env["git_commit"], "1111111")
    joined_logs = "\n".join(log_ctx.output)
    self.assertIn(
        "Mismatched jax_version across workloads in gs://bucket/2026-08-27:"
        " 0.11.1 vs 0.11.2",
        joined_logs,
    )
    self.assertIn(
        "Mismatched libtpu_version across workloads in gs://bucket/2026-08-27:"
        " 0.0.46.1 vs 0.0.46.2",
        joined_logs,
    )
    self.assertIn(
        "Mismatched git_commit across workloads in gs://bucket/2026-08-27:"
        " 1111111 vs 2222222",
        joined_logs,
    )

  def test_fetch_aggregated_csv_results_without_date_str(self):
    """Verifies fetch_aggregated_csv_results accepts a dated GCS run directory directly."""
    with mock.patch("subprocess.run") as mock_run:
      mock_run.side_effect = [
          mock.Mock(
              returncode=0,
              stdout="gs://my-bucket/2026-10-11/aggregated_results/gemm.csv\n",
          ),
          mock.Mock(
              returncode=0,
              stdout="benchmark,m\nGeneralizedGemmBenchmark,8192\n",
          ),
      ]
      categorized = report_generator.fetch_aggregated_csv_results(
          "gs://my-bucket/2026-10-11"
      )
      self.assertEqual(
          categorized,
          {"gemm": [{"benchmark": "GeneralizedGemmBenchmark", "m": "8192"}]},
      )
      mock_run.assert_any_call(
          [
              "gcloud",
              "storage",
              "ls",
              "gs://my-bucket/2026-10-11/aggregated_results/*.csv",
          ],
          capture_output=True,
          text=True,
          check=False,
          stdin=subprocess.DEVNULL,
          timeout=utils.CLI_TIMEOUT_SECONDS,
      )

  def test_format_consolidated_report_host_device(self):
    """Host-device tables render leading columns first."""
    result = _result(duration_seconds=30.0)
    categorized = {
        "host_device": [
            {
                "data_size_mib": "4096.0",
                "dtype": "float32",
                "num_runs": "10",
                "bandwidth_per_device_gb_s": "28.5",
                "time_ms_avg": "145.0",
                "time_ms_p50": "144.5",
                "time_ms_p90": "146.0",
            },
            {
                "data_size_mib": "4096.0",
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
    self.assertIn("4096.0", report_md)
    self.assertIn("28.5", report_md)
    self.assertIn("32.1", report_md)

    report_html = report_generator.generate_html_report(
        [result],
        categorized_metrics=categorized,
    )
    self.assertIn("Host-to-Device (H2D / D2H)", report_html)
    self.assertIn("<th>dtype</th>", report_html)
    self.assertIn("<th>data_size_mib</th>", report_html)
    self.assertIn("<td>float32</td>", report_html)
    self.assertIn("<td>4096.0</td>", report_html)
    self.assertIn("bandwidth_per_device_gb_s", report_html)
    self.assertIn("28.5", report_html)
    self.assertIn("32.1", report_html)

  def test_format_d2d_pairwise_matrix(self):
    """D2D rows become an N x N bandwidth matrix."""
    rows = [
        {
            "src_device_index": "0",
            "dst_device_index": "1",
            "wall_clock_bandwidth_per_device_gb_s": "85.5",
        },
        {
            "src_device_index": "1",
            "dst_device_index": "0",
            "wall_clock_bandwidth_per_device_gb_s": "86.2",
        },
    ]
    matrix_data = report_generator.format_d2d_pairwise_matrix(rows)
    self.assertIsNotNone(matrix_data)
    labels, matrix_rows = matrix_data
    self.assertEqual(labels, ["D0", "D1"])
    self.assertEqual(matrix_rows[0], ["-", "85.50"])
    self.assertEqual(matrix_rows[1], ["86.20", "-"])

  @parameterized.named_parameters(
      dict(
          testcase_name="xprof_bandwidth",
          extra_cols=(
              {
                  "wall_clock_bandwidth_per_device_gb_s": "269.95",
                  "xprof_bandwidth_per_device_gb_s": "678.02",
              },
              {
                  "wall_clock_bandwidth_per_device_gb_s": "268.50",
                  "xprof_bandwidth_per_device_gb_s": "677.10",
              },
          ),
          wall_clock_title=False,
          values=("678.02", "677.10"),
      ),
      dict(
          testcase_name="wall_clock_fallback",
          extra_cols=(
              {"wall_clock_bandwidth_per_device_gb_s": "269.95"},
              {"wall_clock_bandwidth_per_device_gb_s": "268.50"},
          ),
          wall_clock_title=True,
          values=("269.95", "268.50"),
      ),
  )
  def test_format_reports_device_to_device(
      self, extra_cols, wall_clock_title, values
  ):
    """D2D renders a pairwise matrix plus a collapsible raw table."""
    rows = [
        {
            "dtype": "float32",
            "direction": "uni",
            "src_device_index": src,
            "dst_device_index": 1 - src,
            "data_size_mib": "256",
            "num_runs": "5",
            **cols,
        }
        for src, cols in enumerate(extra_cols)
    ]
    categorized = {"device_to_device": rows}
    report_md = report_generator.format_consolidated_report(
        [_result()], categorized_metrics=categorized
    )
    report_html = report_generator.generate_html_report(
        [_result()], categorized_metrics=categorized
    )
    for report in (report_md, report_html):
      self.assertIn("Device-to-Device (D2D)", report)
      self.assertEqual("(Wall Clock Time)" in report, wall_clock_title)
      self.assertIn("Pairwise Bandwidth Matrix (GB/s)", report)
      self.assertIn("View Raw Device-to-Device Transfer Table (2 rows)", report)
      for value in values:
        self.assertIn(value, report)
    # D2D leading columns come first, in schema order.
    self.assertIn(
        "| dtype | direction | src_device_index | dst_device_index |"
        " data_size_mib |",
        report_md,
    )
    self.assertIn('<details class="d2d-details">', report_html)
    self.assertIn("matrix-table", report_html)

  def test_reports_treat_all_empty_categories_as_no_metrics(self):
    """Empty per-category lists (dry-run / all failed) render the fallback."""
    empty = {cat: [] for cat in metrics_aggregator.SUPPORTED_CATEGORIES}
    report_md = report_generator.format_consolidated_report(
        [_result()], categorized_metrics=empty
    )
    self.assertIn("*No aggregated metric tables available.*", report_md)
    report_html = report_generator.generate_html_report(
        [_result()], categorized_metrics=empty
    )
    self.assertNotIn("Aggregated Benchmark Metrics", report_html)

  def test_generate_html_report_dry_run_zero_failures(self):
    """Dry-run results are not counted as failures."""
    dry_run_result = _result(status="DRY_RUN")
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
        _result(duration_seconds=120.0),
        _result(duration_seconds=300.0),
    ]
    html_report = report_generator.generate_html_report(results)
    self.assertIn(
        '<div class="stat-card"><div class="label">Total Suite'
        ' Duration</div><div class="value">300.0s</div></div>',
        html_report,
    )
    self.assertNotIn("420.0s", html_report)

  def test_get_display_headers_leading_columns_only(self):
    """Verifies leading_columns_only returns only leading columns without fallback."""
    rows = [{
        "benchmark": "GeneralizedGemmBenchmark",
        "m": "8192",
        "warmup_tries": "10",
        "num_runs": "20",
    }]
    self.assertEqual(
        report_generator.get_display_headers(
            "gemm", rows, leading_columns_only=True
        ),
        ["benchmark", "m"],
    )
    # Unknown category or rows with no leading columns return [] instead of
    # falling back to raw trailing keys.
    self.assertEqual(
        report_generator.get_display_headers(
            "unknown_cat", rows, leading_columns_only=True
        ),
        [],
    )
    self.assertEqual(
        report_generator.get_display_headers(
            "gemm", [{"warmup_tries": "10"}], leading_columns_only=True
        ),
        [],
    )
    self.assertEqual(
        report_generator.get_display_headers(
            "gemm", [{"_source_path": "gs://b/d/s.csv"}]
        ),
        [],
    )

  def test_format_consolidated_report_omit_run_header_and_leading_columns_only(
      self,
  ):
    """Verifies include_run_header=False, leading_columns_only=True, and d2d_matrix_only=True."""
    categorized = {
        "device_to_device": [
            {
                "benchmark": "DeviceToDeviceBenchmark",
                "dtype": "float32",
                "direction": "uni",
                "src_device_index": 0,
                "dst_device_index": 1,
                "xprof_p50_ms": "0.3955",
                "xprof_bandwidth_per_device_gb_s": "678.64",
                "num_runs": "5",
            },
            {
                "benchmark": "DeviceToDeviceBenchmark",
                "dtype": "float32",
                "direction": "uni",
                "src_device_index": 1,
                "dst_device_index": 0,
                "xprof_p50_ms": "0.3948",
                "xprof_bandwidth_per_device_gb_s": "679.92",
                "num_runs": "5",
            },
        ],
        "gemm": [{
            "benchmark": "GeneralizedGemmBenchmark",
            "in_dtype": "bfloat16",
            "out_dtype": "bfloat16",
            "m": "8192",
            "k": "4096",
            "n": "4096",
            "xprof_p50_ms": "0.25",
            "xprof_tflops_per_device": "1090.25",
            "warmup_tries": "10",
            "wall_clock_p50_ms": "0.50",
        }],
        "unknown_extra": [{"raw_col": "leaked_val"}],
    }
    report_md = report_generator.format_consolidated_report(
        results=[],
        categorized_metrics=categorized,
        git_commit="27915153b5c44b81394e7ac720e2860e3e0c61ee",
        jax_version="0.11.1",
        libtpu_version="0.0.46.2",
        include_run_header=False,
        leading_columns_only=True,
        d2d_matrix_only=True,
    )
    self.assertTrue(
        report_md.startswith("**Environment:** **GitHub Commit:** [`2791515`]")
    )
    self.assertNotIn("# 🚀 TPUMS Daily Automated Benchmark Report", report_md)
    self.assertNotIn("**Cluster:**", report_md)
    self.assertNotIn("**Execution Summary:**", report_md)
    self.assertNotIn("warmup_tries", report_md)
    self.assertNotIn("wall_clock_p50_ms", report_md)
    self.assertNotIn("unknown_extra", report_md)
    self.assertNotIn("leaked_val", report_md)
    self.assertIn("#### Pairwise Bandwidth Matrix (GB/s)", report_md)
    self.assertIn("D0           -  678.64", report_md)
    self.assertIn(
        "D1      679.92       -\n```\n\n### ⚡ Generalized Matrix"
        " Multiplication (GEMM)",
        report_md,
    )
    self.assertNotIn("<details>", report_md)
    self.assertNotIn("View Raw Device-to-Device Transfer Table", report_md)
    self.assertNotIn("DeviceToDeviceBenchmark", report_md)
    self.assertIn("1090.25", report_md)

    # When leading_columns_only=True and d2d_matrix_only=False (default), the
    # collapsible raw D2D table is still rendered using only leading columns.
    report_leading_with_raw_d2d = report_generator.format_consolidated_report(
        results=[],
        categorized_metrics=categorized,
        include_run_header=False,
        leading_columns_only=True,
        d2d_matrix_only=False,
    )
    self.assertIn(
        "#### Pairwise Bandwidth Matrix (GB/s)", report_leading_with_raw_d2d
    )
    self.assertIn("<details>", report_leading_with_raw_d2d)
    self.assertIn(
        "<summary>View Raw Device-to-Device Transfer Table (2 rows)</summary>",
        report_leading_with_raw_d2d,
    )
    self.assertIn("DeviceToDeviceBenchmark", report_leading_with_raw_d2d)
    self.assertIn("src_device_index", report_leading_with_raw_d2d)
    self.assertIn("direction", report_leading_with_raw_d2d)
    self.assertNotIn("num_runs", report_leading_with_raw_d2d)

    # Default leading_columns_only=False preserves the collapsible raw table
    # with all columns after the Pairwise Bandwidth Matrix on the same input.
    report_default_cols = report_generator.format_consolidated_report(
        results=[],
        categorized_metrics=categorized,
        include_run_header=False,
        leading_columns_only=False,
    )
    self.assertIn("#### Pairwise Bandwidth Matrix (GB/s)", report_default_cols)
    self.assertIn("<details>", report_default_cols)
    self.assertIn(
        "<summary>View Raw Device-to-Device Transfer Table (2 rows)</summary>",
        report_default_cols,
    )
    self.assertIn("DeviceToDeviceBenchmark", report_default_cols)
    self.assertIn("src_device_index", report_default_cols)
    self.assertIn("num_runs", report_default_cols)

    report_no_env = report_generator.format_consolidated_report(
        results=[],
        categorized_metrics=categorized,
        include_run_header=False,
        leading_columns_only=True,
        d2d_matrix_only=True,
    )
    self.assertTrue(
        report_no_env.startswith("## Aggregated Benchmark Results by Category")
    )

    # Fallback: if D2D rows lack src/dst indices (matrix_data is None), the raw
    # table is still emitted even when d2d_matrix_only=True.
    report_d2d_fallback = report_generator.format_consolidated_report(
        results=[],
        categorized_metrics={
            "device_to_device": [{
                "benchmark": "DeviceToDeviceBenchmark",
                "dtype": "float32",
                "xprof_p50_ms": "0.3955",
                "xprof_bandwidth_per_device_gb_s": "678.64",
            }],
        },
        include_run_header=False,
        leading_columns_only=True,
        d2d_matrix_only=True,
    )
    self.assertNotIn(
        "#### Pairwise Bandwidth Matrix (GB/s)", report_d2d_fallback
    )
    self.assertIn("<details>", report_d2d_fallback)
    self.assertIn("DeviceToDeviceBenchmark", report_d2d_fallback)


if __name__ == "__main__":
  absltest.main()
