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

"""Automated benchmark runner for TPUMS on GKE TPU clusters via kubectl & Kueue.

Orchestrates the benchmark pipeline by coordinating:
  - Workload Submitter: Resolves configs from canonical suite YAMLs or
    configs/<accelerator>/ (tpu7x or v6e), generates parameterized JobSet
    manifests, submits via kubectl, and polls status.
  - Metrics Aggregator: Ingests each workload's GCS summary.csv, normalizes
    schemas, and aggregates category metrics (collectives, hbm, host_device,
    device_to_device, gemm).
  - Report Generator: Generates consolidated Markdown summary tables
    and responsive HTML preview reports.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import datetime
import logging
import sys
from typing import Optional

from automation import metrics_aggregator
from automation import report_generator
from automation import utils
from automation import workload_submitter

DEFAULT_CLEANUP = True


def _default_output_report(
    now_utc: Optional[datetime.datetime] = None,
) -> str:
  """Derives the dated default Markdown report path from DEFAULT_OUTPUT_REPORT."""
  if now_utc is None:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
  base, ext = report_generator.DEFAULT_OUTPUT_REPORT.rsplit(".", 1)
  return f"{base}_{now_utc.strftime('%Y%m%d')}.{ext}"


def run_pipeline(
    config_dir: Optional[str] = None,
    configs: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    topologies: Optional[list[str]] = None,
    suite_file: Optional[str] = None,
    cluster: str = utils.DEFAULT_CLUSTER,
    project: str = utils.DEFAULT_PROJECT,
    region: str = utils.DEFAULT_REGION,
    zone: str = utils.DEFAULT_ZONE,
    namespace: str = utils.DEFAULT_NAMESPACE,
    queue: str = utils.DEFAULT_QUEUE,
    priority: str = utils.DEFAULT_PRIORITY,
    docker_image: str = utils.DEFAULT_DOCKER_IMAGE,
    gcs_bucket: str = utils.DEFAULT_GCS_BUCKET,
    timeout_minutes: Optional[int] = utils.DEFAULT_TIMEOUT_MINUTES,
    declared_duration_minutes: int = utils.DEFAULT_DECLARED_DURATION_MINUTES,
    poll_interval_seconds: int = utils.DEFAULT_POLL_INTERVAL_SECONDS,
    cleanup: bool = DEFAULT_CLEANUP,
    upload_trace: bool = False,
    output_report: Optional[str] = None,
    dry_run: bool = False,
    git_commit: Optional[str] = None,
    git_branch: str = utils.DEFAULT_GIT_BRANCH,
    jax_version: Optional[str] = None,
    libtpu_version: Optional[str] = None,
    reservation_name: Optional[str] = utils.DEFAULT_RESERVATION_NAME,
    accelerator: Optional[str] = None,
) -> list[workload_submitter.BenchmarkResult]:
  """Executes the benchmark automation pipeline (Workload Submission, Metrics Aggregation, Report Generation)."""
  # Resolve the selection first so flag misuse fails before any GCS or
  # cluster access. Logs the scanned config directory or suite file.
  config_files = workload_submitter.resolve_config_files(
      config_dir=config_dir,
      configs=configs,
      categories=categories,
      topologies=topologies,
      suite_file=suite_file,
      accelerator=accelerator,
  )
  if not config_files:
    raise ValueError(
        "No YAML configuration files found matching selection criteria."
    )
  acc = workload_submitter.infer_configs_accelerator(config_files, accelerator)

  if not dry_run and gcs_bucket:
    metrics_aggregator.verify_gcs_bucket_accessible(gcs_bucket)

  now_utc = datetime.datetime.now(datetime.timezone.utc)
  date_str = now_utc.strftime("%Y-%m-%d")
  if not output_report:
    output_report = _default_output_report(now_utc)

  logging.info("=" * 80)
  logging.info("TPUMS External GKE Automated Benchmark Pipeline")
  logging.info(
      "Target Cluster   : %s (%s / %s / %s)", cluster, project, region, zone
  )
  logging.info("Accelerator      : %s", acc)
  logging.info(
      "Namespace / Queue: %s / %s (Priority: %s)", namespace, queue, priority
  )
  if reservation_name:
    logging.info("Reservation Name : %s", reservation_name)
  logging.info("Docker Image     : %s", docker_image)
  logging.info("GCS Bucket       : %s", gcs_bucket)
  logging.info("Report Output    : %s", output_report)
  logging.info("=" * 80)
  logging.info(
      "Resolved %d benchmark config(s) for execution...", len(config_files)
  )

  # Fetch cluster credentials first
  workload_submitter.fetch_cluster_credentials(
      cluster=cluster,
      project=project,
      region=region,
      zone=zone,
      dry_run=dry_run,
  )

  # ----------------------------------------------------------------------------
  # Workload Submitter: Submit workloads, poll status, cleanup
  # ----------------------------------------------------------------------------
  results = workload_submitter.submit_workloads(
      config_files=config_files,
      namespace=namespace,
      queue=queue,
      priority=priority,
      docker_image=docker_image,
      gcs_bucket=gcs_bucket,
      timeout_minutes=timeout_minutes,
      declared_duration_minutes=declared_duration_minutes,
      poll_interval_seconds=poll_interval_seconds,
      cleanup=cleanup,
      upload_trace=upload_trace,
      git_branch=git_branch,
      dry_run=dry_run,
      reservation_name=reservation_name,
      accelerator=acc,
  )

  # Ingest summary.csv metrics; a JobSet that exited 0 without producing any
  # rows did not benchmark anything, so it is reported as failed.
  if not dry_run:
    for r in results:
      if r.status != "SUCCESS":
        continue
      r.metrics = metrics_aggregator.fetch_gcs_raw_metrics(r.gcs_artifact_path)
      if not r.metrics:
        r.status = "FAILED"
        r.error_message = (
            f"JobSet succeeded but {r.gcs_artifact_path}/summary.csv has no"
            " rows."
        )
        logging.error("%s: %s", r.workload_name, r.error_message)

  total_jobsets = len(results)
  if dry_run:
    logging.info("Dry-run preview completed for %d JobSet(s).", total_jobsets)
  else:
    successful_jobsets, failed_jobsets = report_generator.count_statuses(
        results
    )
    logging.info(
        "JobSet execution finished: %d/%d succeeded (%d failed).",
        successful_jobsets,
        total_jobsets,
        failed_jobsets,
    )

  # ----------------------------------------------------------------------------
  # Metrics Aggregator: Ingest GCS metrics and aggregate by category
  # ----------------------------------------------------------------------------
  raw_result_dicts = [dataclasses.asdict(r) for r in results]
  categorized_metrics = (
      metrics_aggregator.aggregate_and_upload_category_results(
          raw_results=raw_result_dicts,
          gcs_bucket=gcs_bucket,
          date_str=date_str,
          dry_run=dry_run,
      )
  )

  # ----------------------------------------------------------------------------
  # Report Generator: Publish Markdown and HTML summary reports
  # ----------------------------------------------------------------------------
  report_generator.generate_and_publish_report(
      results=results,
      categorized_metrics=categorized_metrics,
      cluster=cluster,
      project=project,
      region=region,
      namespace=namespace,
      output_file=output_report,
      gcs_bucket=gcs_bucket,
      date_str=date_str,
      dry_run=dry_run,
      git_commit=git_commit,
      jax_version=jax_version,
      libtpu_version=libtpu_version,
  )

  logging.info("=" * 80)
  logging.info(
      "Daily Pipeline Execution Finished. Report saved to %s", output_report
  )
  logging.info("=" * 80)

  return results


def main(argv: Sequence[str]) -> None:
  default_report = _default_output_report()

  parser = argparse.ArgumentParser(
      description="TPUMS Automated Benchmark Pipeline Orchestrator."
  )
  for flag, default, help_text in (
      ("--cluster", utils.DEFAULT_CLUSTER, "GKE cluster name"),
      ("--project", utils.DEFAULT_PROJECT, "GCP project ID"),
      ("--region", utils.DEFAULT_REGION, "GCP region"),
      ("--zone", utils.DEFAULT_ZONE, "GCP zone"),
  ):
    parser.add_argument(
        flag, default=default, help=f"{help_text} (default: {default!r})."
    )
  utils.add_workload_args(parser, default_cleanup=DEFAULT_CLEANUP)
  utils.add_selection_args(parser)
  parser.add_argument(
      "--output-report",
      default=default_report,
      help=f"Local report filepath (default: {default_report}).",
  )
  parser.add_argument("--git-commit", default=None, help="GitHub commit hash.")
  parser.add_argument(
      "--jax-version", default=None, help="JAX runtime version."
  )
  parser.add_argument("--libtpu-version", default=None, help="libtpu version.")

  args = parser.parse_args(argv[1:])
  utils.validate_workload_args(args, require_cluster=True)
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )

  logging.info(
      "Starting TPUMS Pipeline Orchestrator (Dry-Run: %s)...", args.dry_run
  )
  run_pipeline(
      config_dir=args.config_dir,
      configs=args.configs,
      categories=args.categories,
      topologies=args.topologies,
      suite_file=args.suite_file,
      cluster=args.cluster,
      project=args.project,
      region=args.region,
      zone=args.zone,
      namespace=args.namespace,
      queue=args.queue,
      priority=args.priority,
      docker_image=args.docker_image,
      gcs_bucket=args.gcs_bucket,
      timeout_minutes=args.timeout_minutes,
      declared_duration_minutes=args.declared_duration_minutes,
      poll_interval_seconds=args.poll_interval,
      cleanup=args.cleanup,
      upload_trace=args.upload_trace,
      output_report=args.output_report,
      dry_run=args.dry_run,
      git_commit=args.git_commit,
      git_branch=args.git_branch,
      jax_version=args.jax_version,
      libtpu_version=args.libtpu_version,
      reservation_name=args.reservation_name,
      accelerator=args.accelerator,
  )


if __name__ == "__main__":
  main(sys.argv)
