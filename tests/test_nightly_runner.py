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

"""Unit tests for TPUMS nightly_runner orchestrator."""

import pathlib
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from automation import metrics_aggregator
from automation import nightly_runner
from automation import report_generator
from automation import utils
from automation import workload_submitter

_CONFIG_DIR = str(pathlib.Path(__file__).parent.parent / "configs" / "tpu7x")


class NightlyRunnerTest(parameterized.TestCase):
  """Tests for the nightly_runner pipeline orchestrator."""

  def setUp(self):
    super().setUp()
    self.mock_creds = self.enter_context(
        mock.patch.object(workload_submitter, "fetch_cluster_credentials")
    )
    self.mock_agg = self.enter_context(
        mock.patch.object(
            metrics_aggregator,
            "aggregate_and_upload_category_results",
            return_value={"gemm": []},
        )
    )
    self.mock_gen_report = self.enter_context(
        mock.patch.object(report_generator, "generate_and_publish_report")
    )

  @parameterized.named_parameters(
      ("tpu7x_suite", "automation/canonical_suite_v7x.yaml", "tpu7x"),
      ("v6e_suite", "automation/canonical_suite_v6e.yaml", "v6e"),
  )
  def test_run_pipeline_dry_run_canonical_suite(self, suite_file, acc):
    with self.assertLogs(level="INFO") as logs:
      results = nightly_runner.run_pipeline(dry_run=True, suite_file=suite_file)
    self.assertIn(f"Accelerator      : {acc}", "\n".join(logs.output))
    self.assertLen(results, 11)
    for r in results:
      self.assertIn(f"{acc}/", r.config_rel_path)
    self.mock_creds.assert_called_once()
    self.mock_agg.assert_called_once()
    self.mock_gen_report.assert_called_once()

  @parameterized.named_parameters(
      ("tpu7x", {"accelerator": "tpu7x", "config_dir": _CONFIG_DIR}, "tpu7x"),
      ("v6e", {"accelerator": "v6e"}, "v6e"),
  )
  def test_run_pipeline_with_categories(self, kwargs, acc):
    results = nightly_runner.run_pipeline(
        categories=["gemm"], dry_run=True, **kwargs
    )
    self.assertNotEmpty(results)
    for r in results:
      self.assertIn("gemm", r.config_name)
      self.assertIn(f"{acc}/", r.config_rel_path)

  @mock.patch.object(
      metrics_aggregator,
      "verify_gcs_bucket_accessible",
      side_effect=RuntimeError("Bucket not accessible"),
  )
  def test_run_pipeline_verifies_gcs_bucket_when_not_dry_run(
      self, mock_verify_bucket
  ):
    """An inaccessible bucket fails before credentials are fetched."""
    with self.assertRaises(RuntimeError):
      nightly_runner.run_pipeline(
          accelerator="tpu7x",
          gcs_bucket="gs://non-existent-bucket",
          dry_run=False,
      )
    mock_verify_bucket.assert_called_once_with("gs://non-existent-bucket")
    self.mock_creds.assert_not_called()

  @parameterized.named_parameters(
      ("no_mode", {}, "exactly one of --suite-file"),
      ("topologies_without_accelerator", {"topologies": ["4x4"]}, "exactly"),
      (
          "suite_with_config_dir",
          {
              "suite_file": "automation/canonical_suite_v6e.yaml",
              "config_dir": "configs/v6e",
          },
          "can only be combined with --accelerator",
      ),
  )
  @mock.patch.object(metrics_aggregator, "verify_gcs_bucket_accessible")
  def test_run_pipeline_invalid_selection_fails_first(
      self, kwargs, regex, mock_verify_bucket
  ):
    """Selection errors surface before any GCS or cluster access."""
    with self.assertRaisesRegex(ValueError, regex):
      nightly_runner.run_pipeline(gcs_bucket="gs://b", dry_run=False, **kwargs)
    mock_verify_bucket.assert_not_called()
    self.mock_creds.assert_not_called()

  def test_run_pipeline_v6e_topologies(self):
    """--accelerator v6e --topologies 4x4 scans configs/v6e."""
    with self.assertLogs(level="INFO") as logs:
      results = nightly_runner.run_pipeline(
          accelerator="v6e", topologies=["4x4"], dry_run=True
      )
    self.assertNotEmpty(results)
    for r in results:
      self.assertIn("v6e/4x4/", r.config_rel_path)
    self.assertIn(
        utils.ACCELERATOR_CONFIGS["v6e"].config_dir,
        "\n".join(logs.output),
    )

  def test_run_pipeline_no_matching_configs_raises(self):
    """An empty selection fails instead of exiting 0 with nothing run."""
    with self.assertRaisesRegex(ValueError, "No YAML"):
      nightly_runner.run_pipeline(
          accelerator="tpu7x",
          config_dir=_CONFIG_DIR,
          categories=["no_such_benchmark"],
          dry_run=True,
      )
    self.mock_creds.assert_not_called()

  def test_run_pipeline_accelerator_topology_mismatch_raises(self):
    """--accelerator v6e with a 3D topology fails before any submission."""
    with self.assertRaisesRegex(ValueError, "do not match accelerator"):
      nightly_runner.run_pipeline(
          accelerator="v6e", topologies=["2x2x1"], dry_run=True
      )
    self.mock_creds.assert_not_called()

  @parameterized.named_parameters(
      ("no_metrics", [], "FAILED", "0/1 succeeded (1 failed)"),
      ("with_metrics", [{"m": "1"}], "SUCCESS", "1/1 succeeded (0 failed)"),
  )
  def test_run_pipeline_success_without_metrics_is_failed(
      self, metrics, status, summary
  ):
    """A JobSet that exits 0 without summary.csv rows is marked FAILED."""
    result = workload_submitter.BenchmarkResult(
        workload_name="tpums-2x2x1-gemm-abc12",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        status="SUCCESS",
        duration_seconds=1.0,
        gcs_artifact_path="gs://b/2026-10-15/tpums-2x2x1-gemm",
    )
    with (
        mock.patch.object(metrics_aggregator, "verify_gcs_bucket_accessible"),
        mock.patch.object(
            workload_submitter, "submit_workloads", return_value=[result]
        ),
        mock.patch.object(
            metrics_aggregator, "fetch_gcs_raw_metrics", return_value=metrics
        ),
        self.assertLogs(level="INFO") as logs,
    ):
      results = nightly_runner.run_pipeline(
          configs=["configs/tpu7x/2x2x1/gemm.yaml"],
          gcs_bucket="gs://b",
          dry_run=False,
      )
    self.assertEqual(results[0].status, status)
    if not metrics:
      self.assertIn("summary.csv", results[0].error_message)
    self.assertIn(summary, "\n".join(logs.output))
    reported = self.mock_gen_report.call_args.kwargs["results"]
    self.assertEqual(reported[0].status, status)

  @mock.patch.object(nightly_runner, "run_pipeline")
  def test_main_forwards_accelerator_and_selection_flags(self, mock_run):
    """CLI parses --accelerator / --topologies / --config-dir defaults."""
    nightly_runner.main([
        "nightly_runner",
        "--dry-run",
        "--accelerator",
        "v6e",
        "--topologies",
        "4x4",
    ])
    kwargs = mock_run.call_args.kwargs
    self.assertEqual(kwargs["accelerator"], "v6e")
    self.assertEqual(kwargs["topologies"], ["4x4"])
    self.assertIsNone(kwargs["config_dir"])
    self.assertIsNone(kwargs["categories"])
    self.assertIsNone(kwargs["suite_file"])
    self.assertTrue(kwargs["cleanup"])

    nightly_runner.main(
        ["nightly_runner", "--dry-run", "--suite-file", "custom.yaml"]
    )
    kwargs = mock_run.call_args.kwargs
    self.assertIsNone(kwargs["accelerator"])
    self.assertEqual(kwargs["suite_file"], "custom.yaml")

    with self.assertRaises(SystemExit):
      nightly_runner.main(
          ["nightly_runner", "--dry-run", "--accelerator", "v5p"]
      )

  @mock.patch.object(nightly_runner, "run_pipeline")
  def test_main_accepts_full_production_flags(self, mock_run):
    """Accepts the full flag set of a production nightly invocation."""
    nightly_runner.main([
        "nightly_runner",
        "--cluster=c",
        "--project=p",
        "--region=southamerica-west1",
        "--zone=southamerica-west1-a",
        "--namespace=default",
        "--queue=multislice-queue",
        "--priority=medium",
        "--reservation-name=r",
        "--docker-image=img",
        "--gcs-bucket=gs://b/v6e",
        "--poll-interval=60",
        "--cleanup",
        "--suite-file=automation/canonical_suite_v6e.yaml",
        "--output-report=/tmp/report.md",
    ])
    kwargs = mock_run.call_args.kwargs
    self.assertIsNone(kwargs["accelerator"])
    self.assertIsNone(kwargs["config_dir"])
    self.assertEqual(
        kwargs["suite_file"], "automation/canonical_suite_v6e.yaml"
    )
    self.assertEqual(kwargs["reservation_name"], "r")
    self.assertEqual(kwargs["poll_interval_seconds"], 60)
    self.assertFalse(kwargs["dry_run"])

    with self.assertRaisesRegex(ValueError, "--cluster"):
      nightly_runner.main(
          ["nightly_runner", "--gcs-bucket=gs://b", "--accelerator=v6e"]
      )
    # --accelerator and --suite-file are different selection modes.
    with self.assertRaises(SystemExit):
      nightly_runner.main([
          "nightly_runner",
          "--dry-run",
          "--accelerator=v6e",
          "--suite-file=automation/canonical_suite_v6e.yaml",
      ])


if __name__ == "__main__":
  absltest.main()
