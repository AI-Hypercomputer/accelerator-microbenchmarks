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

"""Unit tests for TPUMS nightly_runner orchestrator."""

import json
import os
import pathlib
import subprocess
import tempfile
import time
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from automation import metrics_aggregator
from automation import nightly_runner
from automation import report_generator
from automation import workload_submitter
import yaml

_CONFIG_DIR = str(pathlib.Path(__file__).parent.parent / "configs" / "tpu7x")


class NightlyRunnerTest(parameterized.TestCase):

  @mock.patch.object(workload_submitter, "fetch_cluster_credentials")
  @mock.patch.object(report_generator, "fetch_environment_metadata")
  @mock.patch.object(
      metrics_aggregator, "aggregate_and_upload_category_results"
  )
  @mock.patch.object(report_generator, "generate_and_publish_report")
  def test_run_pipeline_dry_run_e2e(
      self,
      mock_gen_report,
      mock_agg,
      mock_fetch_env,
      mock_creds,
  ):
    mock_fetch_env.return_value = {
        "git_commit": "abcdef1234567890",
        "jax_version": "0.11.1",
        "libtpu_version": "0.0.46.1",
    }
    mock_agg.return_value = {"gemm": []}
    results = nightly_runner.run_pipeline(
        config_dir=_CONFIG_DIR,
        dry_run=True,
    )
    self.assertLen(results, 11)
    mock_creds.assert_called_once()
    mock_agg.assert_called_once()
    mock_gen_report.assert_called_once()

  @mock.patch.object(workload_submitter, "fetch_cluster_credentials")
  @mock.patch.object(report_generator, "fetch_environment_metadata")
  @mock.patch.object(
      metrics_aggregator, "aggregate_and_upload_category_results"
  )
  @mock.patch.object(report_generator, "generate_and_publish_report")
  def test_run_pipeline_with_filter(
      self,
      mock_gen_report,
      mock_agg,
      mock_fetch_env,
      mock_creds,
  ):
    mock_fetch_env.return_value = {
        "git_commit": "abcdef1234567890",
        "jax_version": "0.11.1",
        "libtpu_version": "0.0.46.1",
    }
    mock_agg.return_value = {"gemm": []}
    results = nightly_runner.run_pipeline(
        config_dir=_CONFIG_DIR,
        filter_patterns=["gemm_generalized"],
        dry_run=True,
    )
    self.assertTrue(len(results) >= 1)
    for r in results:
      self.assertIn("gemm", r.config_name)

  @mock.patch.object(
      metrics_aggregator,
      "verify_gcs_bucket_accessible",
      side_effect=RuntimeError("Bucket not accessible"),
  )
  @mock.patch.object(workload_submitter, "fetch_cluster_credentials")
  def test_run_pipeline_verifies_gcs_bucket_when_not_dry_run(
      self,
      mock_creds,
      mock_verify_bucket,
  ):
    with self.assertRaises(RuntimeError):
      nightly_runner.run_pipeline(
          config_dir=_CONFIG_DIR,
          gcs_bucket="gs://non-existent-bucket",
          dry_run=False,
      )
    mock_verify_bucket.assert_called_once_with("gs://non-existent-bucket")
    mock_creds.assert_not_called()


if __name__ == "__main__":
  absltest.main()
