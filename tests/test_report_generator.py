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
    mock_detailed_json = json.dumps([{
        "metadata": {
            "platform_info": {
                "jax_version": "0.11.1",
                "libtpu_version": "0.0.46.1",
            }
        }
    }])
    with mock.patch("subprocess.run") as mock_run:
      mock_run.side_effect = [
          mock.Mock(returncode=0, stdout="gs://bucket/date/wl/detailed.json\n"),
          mock.Mock(returncode=0, stdout=mock_detailed_json),
          mock.Mock(
              returncode=0, stdout="gs://bucket/date/wl/git_commit.txt\n"
          ),
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
        mock_run.call_args_list[2].args[0],
        ["gcloud", "storage", "ls", "gs://bucket/2026-08-27/*/git_commit.txt"],
    )

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


if __name__ == "__main__":
  absltest.main()
