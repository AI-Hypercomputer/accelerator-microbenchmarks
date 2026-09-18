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

"""Automated benchmark runner for TPUMS on GKE TPU clusters via kubectl & Kueue.

Orchestrates the benchmark pipeline by coordinating:
  - Workload Submitter: Discovers configs in configs/tpu7x/, generates
    parameterized JobSet manifests, submits via kubectl, and polls status.
  - Metrics Aggregator: Ingests GCS host CSV/TSVs, normalizes schemas,
    and aggregates category metrics (collectives, hbm, host_device,
    device_to_device, gemm).
  - Report Generator: Generates consolidated Markdown summary tables
    and responsive HTML preview reports.
"""

from collections.abc import Sequence
import dataclasses
import datetime
import logging
import os
import pathlib
import sys
from typing import Optional

# Ensure repository root is in sys.path for direct CLI execution
_REPO_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
_THIRD_PARTY_PY = str(pathlib.Path(__file__).parent.parent.parent.resolve())
for p in [_REPO_ROOT, _THIRD_PARTY_PY]:
  if p not in sys.path:
    sys.path.insert(0, p)

from automation import metrics_aggregator
from automation import report_generator
from automation import workload_submitter

DEFAULT_CLUSTER = os.environ.get("CLUSTER", "")
DEFAULT_PROJECT = os.environ.get("PROJECT", "")
DEFAULT_REGION = os.environ.get("REGION", "us-central1")
DEFAULT_ZONE = os.environ.get("ZONE", "us-central1-c")
DEFAULT_NAMESPACE = os.environ.get("NAMESPACE", "default")
DEFAULT_QUEUE = os.environ.get("QUEUE", "multislice-queue")
DEFAULT_PRIORITY = os.environ.get("PRIORITY", "medium")
DEFAULT_DOCKER_IMAGE = os.environ.get(
    "DOCKER_IMAGE", "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest"
)
DEFAULT_GCS_BUCKET = workload_submitter.DEFAULT_GCS_OUTPUT_DIR
DEFAULT_RESERVATION_NAME = os.environ.get("RESERVATION_NAME", "")
DEFAULT_TIMEOUT_MINUTES: Optional[int] = None
DEFAULT_DECLARED_DURATION_MINUTES = 30
DEFAULT_POLL_INTERVAL_SECONDS = 60
DEFAULT_CLEANUP = True
DEFAULT_CONFIG_DIR = workload_submitter.DEFAULT_CONFIG_DIR


def run_pipeline(
    config_dir: str = DEFAULT_CONFIG_DIR,
    configs: Optional[list[str]] = None,
    filter_patterns: Optional[list[str]] = None,
    topologies: Optional[list[str]] = None,
    suite_file: Optional[str] = None,
    cluster: str = DEFAULT_CLUSTER,
    project: str = DEFAULT_PROJECT,
    region: str = DEFAULT_REGION,
    zone: str = DEFAULT_ZONE,
    namespace: str = DEFAULT_NAMESPACE,
    queue: str = DEFAULT_QUEUE,
    priority: str = DEFAULT_PRIORITY,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    gcs_bucket: str = DEFAULT_GCS_BUCKET,
    timeout_minutes: Optional[int] = DEFAULT_TIMEOUT_MINUTES,
    declared_duration_minutes: int = DEFAULT_DECLARED_DURATION_MINUTES,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    cleanup: bool = DEFAULT_CLEANUP,
    upload_trace: bool = False,
    output_report: Optional[str] = None,
    dry_run: bool = False,
    git_commit: Optional[str] = None,
    git_branch: str = workload_submitter.DEFAULT_GIT_BRANCH,
    jax_version: Optional[str] = None,
    libtpu_version: Optional[str] = None,
    reservation_name: Optional[str] = DEFAULT_RESERVATION_NAME,
) -> list[workload_submitter.BenchmarkResult]:
  """Executes the benchmark automation pipeline (Workload Submission, Metrics Aggregation, Report Generation)."""
  if not dry_run and gcs_bucket:
    metrics_aggregator.verify_gcs_bucket_accessible(gcs_bucket)

  now_utc = datetime.datetime.now(datetime.timezone.utc)
  date_str = now_utc.strftime("%Y-%m-%d")
  if not output_report:
    output_report = f"/tmp/tpums_daily_report_{now_utc.strftime('%Y%m%d')}.md"

  # ----------------------------------------------------------------------------
  # Discover and Resolve Benchmark Configs
  # ----------------------------------------------------------------------------
  actual_config_dir = config_dir or DEFAULT_CONFIG_DIR

  logging.info("=" * 80)
  logging.info("TPUMS External GKE Automated Benchmark Pipeline")
  logging.info(
      "Target Cluster   : %s (%s / %s / %s)", cluster, project, region, zone
  )
  logging.info(
      "Namespace / Queue: %s / %s (Priority: %s)", namespace, queue, priority
  )
  if reservation_name:
    logging.info("Reservation Name : %s", reservation_name)
  logging.info("Docker Image     : %s", docker_image)
  if filter_patterns or topologies:
    logging.info("Config Directory : %s", actual_config_dir)
  elif configs:
    logging.info("Explicit Configs : %d file(s)", len(configs))
  else:
    logging.info(
        "Suite File       : %s",
        workload_submitter.resolve_canonical_suite_path(suite_file),
    )
  logging.info("GCS Bucket       : %s", gcs_bucket)
  logging.info("Report Output    : %s", output_report)
  logging.info("=" * 80)

  config_files = workload_submitter.resolve_config_files(
      config_dir=actual_config_dir,
      configs=configs,
      filter_patterns=filter_patterns,
      topologies=topologies,
      suite_file=suite_file,
  )

  if not config_files:
    logging.error(
        "No YAML configuration files found matching selection criteria."
    )
    return []

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
  )

  total_jobsets = len(results)
  if dry_run:
    logging.info("Dry-run preview completed for %d JobSet(s).", total_jobsets)
  else:
    successful_jobsets = sum(1 for r in results if r.status == "SUCCESS")
    failed_jobsets = sum(
        1 for r in results if r.status in ("FAILED", "TIMEOUT", "EVICTED")
    )
    logging.info(
        "JobSet execution finished: %d/%d succeeded (%d failed).",
        successful_jobsets,
        total_jobsets,
        failed_jobsets,
    )

  # Ingest metric CSVs/TSVs for successful runs
  if not dry_run:
    for r in results:
      if r.status == "SUCCESS":
        r.metrics = metrics_aggregator.fetch_gcs_raw_metrics(
            r.gcs_artifact_path
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
      gcs_bucket=gcs_bucket if not dry_run else None,
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
  import argparse  # pylint: disable=g-import-not-at-top

  default_report = f"/tmp/tpums_daily_report_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')}.md"

  parser = argparse.ArgumentParser(
      description="TPUMS Automated Benchmark Pipeline Orchestrator."
  )
  parser.add_argument(
      "--cluster",
      default=DEFAULT_CLUSTER,
      help=f"GKE cluster name (default: {DEFAULT_CLUSTER}).",
  )
  parser.add_argument(
      "--project",
      default=DEFAULT_PROJECT,
      help=f"GCP project ID (default: {DEFAULT_PROJECT}).",
  )
  parser.add_argument(
      "--region",
      default=DEFAULT_REGION,
      help=f"GCP region (default: {DEFAULT_REGION}).",
  )
  parser.add_argument(
      "--zone",
      default=DEFAULT_ZONE,
      help=f"GCP zone (default: {DEFAULT_ZONE}).",
  )
  parser.add_argument(
      "--namespace",
      default=DEFAULT_NAMESPACE,
      help=f"Target namespace (default: {DEFAULT_NAMESPACE}).",
  )
  parser.add_argument(
      "--queue",
      default=DEFAULT_QUEUE,
      help=f"Target queue (default: {DEFAULT_QUEUE}).",
  )
  parser.add_argument(
      "--priority",
      default=DEFAULT_PRIORITY,
      help=f"Priority class (default: {DEFAULT_PRIORITY}).",
  )
  parser.add_argument(
      "--docker-image",
      default=DEFAULT_DOCKER_IMAGE,
      help="Base TPU container image.",
  )
  parser.add_argument(
      "--gcs-bucket", default=DEFAULT_GCS_BUCKET, help="GCS output bucket."
  )
  parser.add_argument(
      "--timeout-minutes",
      type=int,
      default=DEFAULT_TIMEOUT_MINUTES,
      help="Timeout in minutes per workload (default: None / infinite).",
  )
  parser.add_argument(
      "--declared-duration-minutes",
      type=int,
      default=DEFAULT_DECLARED_DURATION_MINUTES,
      help="Declared duration.",
  )
  parser.add_argument(
      "--poll-interval",
      type=int,
      default=DEFAULT_POLL_INTERVAL_SECONDS,
      help="Status dashboard polling interval in seconds (default: 60).",
  )
  parser.add_argument(
      "--cleanup",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Automatically delete JobSets on the cluster after completion"
          " (default: True)."
      ),
  )
  parser.add_argument(
      "--config-dir",
      dest="config_dir",
      default=DEFAULT_CONFIG_DIR,
      help=(
          "Configs directory scanned when --filter or --topologies is used"
          f" (default: {DEFAULT_CONFIG_DIR})."
      ),
  )

  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      "--configs",
      nargs="+",
      default=None,
      help="Explicit list of YAML config files to run.",
  )
  group.add_argument(
      "--filter",
      nargs="+",
      default=None,
      help=(
          "One or more filter patterns for config names/paths (e.g. '--filter"
          " all_gather gemm')."
      ),
  )
  group.add_argument(
      "--topologies",
      nargs="+",
      default=None,
      help=(
          "Filter configs by target topology (e.g. '--topologies 2x2x1 2x2x2')."
      ),
  )
  group.add_argument(
      "--suite-file",
      default=None,
      help="Path to a suite YAML file listing benchmark configs to run.",
  )

  parser.add_argument(
      "--upload-trace",
      action="store_true",
      default=False,
      help="Upload XLA/TensorBoard trace directories to GCS (default: False).",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help=(
          "Execute dry-run across the full pipeline (output manifests &"
          " reports)."
      ),
  )
  parser.add_argument(
      "--output-report",
      default=default_report,
      help=f"Local report filepath (default: {default_report}).",
  )
  parser.add_argument("--git-commit", default=None, help="GitHub commit hash.")
  parser.add_argument(
      "--git-branch",
      default=workload_submitter.DEFAULT_GIT_BRANCH,
      help=(
          "Git branch to clone in the JobSet container (default: "
          f"{workload_submitter.DEFAULT_GIT_BRANCH})."
      ),
  )
  parser.add_argument(
      "--jax-version", default=None, help="JAX runtime version."
  )
  parser.add_argument("--libtpu-version", default=None, help="libtpu version.")
  parser.add_argument(
      "--reservation-name",
      default=DEFAULT_RESERVATION_NAME,
      help=(
          "Optional GCE reservation name to add to JobSet nodeSelector "
          f"(default: {DEFAULT_RESERVATION_NAME or 'None'})."
      ),
  )

  args = parser.parse_args(argv[1:])
  if not args.gcs_bucket and not args.dry_run:
    raise ValueError(
        "Must specify --gcs-bucket or set GCS_BUCKET environment variable."
    )
  if not args.docker_image and not args.dry_run:
    raise ValueError(
        "Must specify --docker-image or set DOCKER_IMAGE environment variable."
    )
  if not args.dry_run:
    if not args.cluster or not args.project:
      raise ValueError(
          "Must specify --cluster and --project (or set CLUSTER and PROJECT"
          " environment variables)."
      )
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )

  logging.info(
      "Starting TPUMS Pipeline Orchestrator (Dry-Run: %s)...", args.dry_run
  )
  run_pipeline(
      config_dir=args.config_dir,
      configs=args.configs,
      filter_patterns=args.filter,
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
  )


if __name__ == "__main__":
  main(sys.argv)
