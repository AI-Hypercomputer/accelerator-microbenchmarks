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

"""Parallel TPU Microbenchmark Workload Submission, Lifecycle Management, and Cleanup via kubectl."""

from collections.abc import Sequence
import csv
from dataclasses import dataclass, field
import datetime
import io
import json
import logging
import math
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
from typing import Any, Optional

import yaml

DEFAULT_NAMESPACE = "default"
DEFAULT_QUEUE = "multislice-queue"
DEFAULT_PRIORITY = "medium"
DEFAULT_DOCKER_IMAGE = os.environ.get(
    "DOCKER_IMAGE", "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest"
)
DEFAULT_GIT_BRANCH = os.environ.get("GIT_BRANCH", "main")
DEFAULT_GCS_OUTPUT_DIR = os.environ.get("GCS_BUCKET", "")
DEFAULT_RESERVATION_NAME = os.environ.get("RESERVATION_NAME", "")
DEFAULT_TIMEOUT_MINUTES: Optional[int] = None
DEFAULT_DECLARED_DURATION_MINUTES = 30
DEFAULT_MAX_RETRIES = 2
DEFAULT_POLL_INTERVAL_SECONDS = 60
DEFAULT_TICK_INTERVAL_SECONDS = 15
DEFAULT_CLEANUP = False
DEFAULT_TEMPLATE_PATH = os.path.join(
    pathlib.Path(__file__).parent, "jobset_template.yaml"
)
DEFAULT_CANONICAL_SUITE_PATH = os.path.join(
    pathlib.Path(__file__).parent, "canonical_suite.yaml"
)
DEFAULT_CONFIG_DIR = str(
    pathlib.Path(__file__).parent.parent / "configs" / "tpu7x"
)


def resolve_canonical_suite_path(suite_file: Optional[str] = None) -> str:
  """Resolves the canonical suite YAML file path."""
  if suite_file:
    p = pathlib.Path(suite_file)
    if p.is_absolute() and p.exists():
      return str(p.resolve())
    candidates = [
        (pathlib.Path.cwd() / p).resolve(),
        (pathlib.Path(__file__).parent.parent / p).resolve(),
        (pathlib.Path(__file__).parent / p).resolve(),
        p.resolve(),
    ]
    for c in candidates:
      if c.exists():
        return str(c)
    raise FileNotFoundError(f"Suite YAML file not found: {suite_file}")

  # Default canonical suite path search candidates
  default_candidates = [
      # 1. Local automation directory (same directory as this module)
      pathlib.Path(__file__).parent / "canonical_suite.yaml",
      # 2. Package relative path
      pathlib.Path(__file__).parent.parent
      / "automation"
      / "canonical_suite.yaml",
      # 3. Bazel runfiles path
      pathlib.Path(
          "third_party/py/accelerator_microbenchmarks/automation/canonical_suite.yaml"
      ),
  ]
  for c in default_candidates:
    if c.exists():
      return str(c.resolve())
  return str(pathlib.Path(__file__).parent / "canonical_suite.yaml")


def load_canonical_configs(suite_file: Optional[str] = None) -> list[str]:
  """Loads canonical benchmark config paths from a suite YAML file."""
  resolved_path = resolve_canonical_suite_path(suite_file)
  if not os.path.exists(resolved_path):
    raise FileNotFoundError(
        f"Canonical suite YAML file not found: {resolved_path}"
    )

  with open(resolved_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  if isinstance(data, list):
    configs = [str(item) for item in data if item]
  elif isinstance(data, dict):
    configs_val = (
        data.get("configs") or data.get("suite") or data.get("workloads")
    )
    if not isinstance(configs_val, list):
      raise ValueError(
          f"Suite YAML file {resolved_path} must contain a 'configs' list."
      )
    configs = [str(item) for item in configs_val if item]
  else:
    raise ValueError(
        f"Invalid format in suite YAML file {resolved_path}: expected list or"
        " dict."
    )

  if not configs:
    raise ValueError(
        f"No benchmark configurations found in suite YAML file {resolved_path}."
    )
  return configs


def resolve_template_path(template_path: Optional[str] = None) -> str:
  """Resolves the JobSet template YAML file path."""
  candidates = []
  if template_path:
    candidates.append(pathlib.Path(template_path))
  # 1. Local automation directory (same directory as this module)
  candidates.append(pathlib.Path(__file__).parent / "jobset_template.yaml")
  # 2. Package relative path
  candidates.append(
      pathlib.Path(__file__).parent.parent
      / "automation"
      / "jobset_template.yaml"
  )
  # 3. Bazel runfiles path
  candidates.append(
      pathlib.Path(
          "third_party/py/accelerator_microbenchmarks/automation/jobset_template.yaml"
      )
  )
  for c in candidates:
    if c.exists():
      return str(c.resolve())
  return str(pathlib.Path(__file__).parent / "jobset_template.yaml")


def fetch_cluster_credentials(
    cluster: str,
    project: str,
    region: str = "us-central1",
    zone: Optional[str] = None,
    dry_run: bool = False,
) -> None:
  """Fetches cluster credentials via gcloud."""
  if dry_run:
    logging.info("[Dry-Run] Fetching credentials for cluster %s...", cluster)
    return

  cmd = [
      "gcloud",
      "container",
      "clusters",
      "get-credentials",
      cluster,
      f"--project={project}",
  ]
  logging.info("Fetching cluster credentials for %s...", cluster)

  region_cmd = cmd + [f"--region={region}"]
  try:
    subprocess.run(
        region_cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    logging.info("Successfully fetched credentials via region %s", region)
    return
  except subprocess.CalledProcessError as e:
    logging.debug(
        "Failed to fetch credentials via region %s: %s", region, e.output
    )
    if zone:
      zone_cmd = cmd + [f"--zone={zone}"]
      try:
        subprocess.run(
            zone_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logging.info("Successfully fetched credentials via zone %s", zone)
        return
      except subprocess.CalledProcessError as e2:
        logging.error(
            "Failed to fetch credentials via zone %s: %s", zone, e2.output
        )
        raise e2
    else:
      logging.error(
          "Failed to fetch credentials via region %s and no zone specified.",
          region,
      )
      raise e


def compute_topology_info(
    topology: str,
    accelerator: str = "tpu7x",
    chips_per_node: int = 4,
    cores_per_chip: int = 2,
) -> dict[str, Any]:
  """Computes topology node counts, chips, and placement policy dynamically."""
  parts = topology.lower().split("x")
  if len(parts) < 2 or not all(p.isdigit() for p in parts):
    raise ValueError(
        f"Invalid topology format: '{topology}'. Expected format like '2x2x1'."
    )
  dims = [int(p) for p in parts]
  total_chips = math.prod(dims)
  num_nodes = max(1, total_chips // chips_per_node)
  total_cores = total_chips * cores_per_chip
  placement_policy = (
      f"{accelerator}-{total_cores}-{topology}-placement-policy"
      if num_nodes > 1
      else None
  )
  return {
      "accelerator": accelerator,
      "num_nodes": num_nodes,
      "chips_per_node": chips_per_node,
      "total_chips": total_chips,
      "placement_policy": placement_policy,
  }


_KNOWN_TOPOLOGIES = ("2x2x1", "2x2x2", "2x4x4", "4x4x4", "4x4x8")
TOPOLOGY_MAP = {t: compute_topology_info(t) for t in _KNOWN_TOPOLOGIES}
DEFAULT_TOPOLOGY = "2x2x1"

# Matches a TPU topology delimited by non-digit boundaries, so a path segment
# like "2x2x16" is never truncated to "2x2x1".
_TOPOLOGY_RE = re.compile(r"(?<!\d)(\d+x\d+x\d+)(?!\d)")


@dataclass
class BenchmarkResult:
  """Parsed results for an individual microbenchmark run."""

  workload_name: str
  config_name: str
  config_rel_path: str
  topology: str
  status: str  # SUCCESS, FAILED, TIMEOUT, EVICTED, DRY_RUN
  duration_seconds: float
  gcs_artifact_path: str
  metrics: list[dict[str, Any]] = field(default_factory=list)
  error_message: Optional[str] = None


@dataclass
class ActiveWorkload:
  """State tracking object for an actively running or queued JobSet."""

  workload_name: str
  config_name: str
  config_rel_path: str
  topology: str
  manifest_yaml: str
  gcs_output_dir: str
  start_time: float
  status: (
      str
  ) = (  # SUBMITTED, QUEUED, RUNNING, SUCCESS, FAILED, EVICTED, TIMEOUT
      "SUBMITTED"
  )
  ready_pods: int = 0
  successful_pods: int = 0
  failed_pods: int = 0
  total_pods: int = 0
  duration_seconds: float = 0.0
  error_message: Optional[str] = None
  terminal: bool = False
  cleaned_up: bool = False

  def __post_init__(self) -> None:
    if self.total_pods <= 0:
      try:
        self.total_pods = compute_topology_info(self.topology)["num_nodes"]
      except ValueError:
        self.total_pods = TOPOLOGY_MAP.get(
            self.topology, TOPOLOGY_MAP.get("2x2x1", {"num_nodes": 1})
        )["num_nodes"]


def resolve_config_files(
    config_dir: str = DEFAULT_CONFIG_DIR,
    configs: Optional[list[str]] = None,
    filter_patterns: Optional[list[str]] = None,
    topologies: Optional[list[str]] = None,
    suite_file: Optional[str] = None,
) -> list[pathlib.Path]:
  """Resolves target config files honoring mutual exclusivity and multi-pattern filters."""
  specified = [
      k
      for k, v in [
          ("configs", configs),
          ("filter", filter_patterns),
          ("topologies", topologies),
          ("suite_file", suite_file),
      ]
      if v
  ]
  if len(specified) > 1:
    raise ValueError(
        f"Arguments {specified} are mutually exclusive. Please specify only"
        " one."
    )

  repo_root = pathlib.Path(os.path.abspath(__file__)).parent.parent

  if configs:
    resolved = []
    for c in configs:
      p = pathlib.Path(c)
      if not p.is_absolute():
        resolved_p = pathlib.Path(os.path.abspath(pathlib.Path.cwd() / p))
        if not resolved_p.exists():
          resolved_p = pathlib.Path(os.path.abspath(repo_root / p))
        p = resolved_p
      if not p.exists():
        raise FileNotFoundError(f"Config file not found: {c}")
      resolved.append(p)
    return resolved

  if filter_patterns or topologies:
    config_path = pathlib.Path(config_dir)
    if not config_path.exists():
      logging.warning("Config directory %s does not exist.", config_dir)
      return []
    all_yaml_files = sorted(config_path.glob("**/*.yaml"))

    if filter_patterns:
      lower_patterns = [
          pat.lower().strip() for pat in filter_patterns if pat.strip()
      ]
      return [
          p
          for p in all_yaml_files
          if any(pat in str(p).lower() for pat in lower_patterns)
      ]

    topo_set = {t.lower().strip() for t in topologies if t.strip()}
    return [
        p for p in all_yaml_files if any(t in str(p).lower() for t in topo_set)
    ]

  # Canonical default suite: loaded directly from canonical_suite.yaml (or specified suite_file)
  canonical_configs = load_canonical_configs(suite_file)
  resolved_default = []
  matched_paths = set()
  for c in canonical_configs:
    p = pathlib.Path(c)
    if not p.is_absolute():
      cand = pathlib.Path(os.path.abspath(repo_root / p))
      if not cand.exists():
        cand = pathlib.Path(os.path.abspath(pathlib.Path.cwd() / p))
      p = cand
    else:
      p = pathlib.Path(os.path.abspath(p))
    if p.exists() and p not in matched_paths:
      resolved_default.append(p)
      matched_paths.add(p)

  suite_source = resolve_canonical_suite_path(suite_file)
  logging.info(
      "Resolved canonical default suite (%d configs) from %s.",
      len(resolved_default),
      suite_source,
  )
  return resolved_default


def infer_topology_from_config(
    config_path: str, config_data: dict[str, Any]
) -> str:
  """Infers the target TPU topology from the config file path."""
  del config_data  # Topology is encoded in the config path, not its contents.
  path_str = str(config_path).lower()
  match = _TOPOLOGY_RE.search(path_str)
  if match:
    topology = match.group(1)
    if topology not in TOPOLOGY_MAP:
      logging.warning(
          "Topology %r from %r is not in TOPOLOGY_MAP %s; proceeding with"
          " dynamically computed placement info.",
          topology,
          config_path,
          sorted(TOPOLOGY_MAP),
      )
    return topology

  logging.error(
      "Could not infer topology from config path %r; defaulting to %s.",
      config_path,
      DEFAULT_TOPOLOGY,
  )
  return DEFAULT_TOPOLOGY


def sanitize_workload_name(
    base_name: str, topology: Optional[str] = None, max_len: int = 45
) -> str:
  """Generates a safe DNS-compliant Kubernetes JobSet name incorporating topology without timestamps."""
  clean_base = re.sub(r"[^a-z0-9-]", "-", base_name.lower()).strip("-")
  if topology:
    clean_topo = re.sub(r"[^a-z0-9-]", "-", topology.lower()).strip("-")
    if clean_topo not in clean_base:
      full_name = f"tpums-{clean_topo}-{clean_base}"
    else:
      full_name = f"tpums-{clean_base}"
  else:
    full_name = f"tpums-{clean_base}"

  clean_full = re.sub(r"-+", "-", full_name).strip("-")
  return clean_full[:max_len].rstrip("-")


def generate_jobset_manifest(
    workload_name: str,
    namespace: str,
    queue_name: str,
    priority_class: str,
    topology: str,
    docker_image: str,
    config_rel_path: str,
    gcs_output_dir: str,
    declared_duration_minutes: int = DEFAULT_DECLARED_DURATION_MINUTES,
    template_path: Optional[str] = None,
    upload_trace: bool = False,
    git_branch: str = DEFAULT_GIT_BRANCH,
    reservation_name: Optional[str] = DEFAULT_RESERVATION_NAME,
) -> str:
  """Generates a native GKE JobSet manifest using automation/jobset_template.yaml."""
  try:
    topo_info = compute_topology_info(topology)
  except ValueError:
    topo_info = TOPOLOGY_MAP.get(topology, TOPOLOGY_MAP["2x2x1"])
  num_nodes = topo_info["num_nodes"]
  chips_per_node = topo_info["chips_per_node"]
  placement_policy = topo_info.get("placement_policy")

  tmpl_file = resolve_template_path(template_path)
  if os.path.exists(tmpl_file):
    with open(tmpl_file, "r", encoding="utf-8") as f:
      template_raw = f.read()
  else:
    raise FileNotFoundError(f"JobSet template file not found at: {tmpl_file}")

  # Strip leading commentary before apiVersion: jobset.x-k8s.io
  api_idx = template_raw.find("apiVersion: jobset.x-k8s.io")
  if api_idx != -1:
    manifest = template_raw[api_idx:]
  else:
    manifest = template_raw

  # Substitute template variables
  exclusive_topology = (
      "kubernetes.io/hostname"
      if num_nodes == 1
      else "cloud.google.com/gke-nodepool"
  )
  replacements = {
      "${WORKLOAD_NAME}": workload_name,
      "${NAMESPACE}": namespace,
      "${QUEUE_NAME}": queue_name,
      "${PRIORITY_CLASS}": priority_class,
      "${TOPOLOGY}": topology,
      "${EXCLUSIVE_TOPOLOGY}": exclusive_topology,
      "${PARALLELISM}": str(num_nodes),
      "${COMPLETIONS}": str(num_nodes),
      "${CONFIG_PATH}": config_rel_path,
      "${GIT_BRANCH}": git_branch,
      "${GCS_OUTPUT_DIR}": gcs_output_dir,
      "${DECLARED_DURATION}": str(declared_duration_minutes),
      "${DOCKER_IMAGE}": docker_image or "jax-tpu:latest",
      "${CHIPS_PER_NODE}": str(chips_per_node),
      "${UPLOAD_TRACE}": "true" if upload_trace else "false",
  }

  for k, v in replacements.items():
    manifest = manifest.replace(k, v)

  # Handle multi-node placement policy substitution
  if placement_policy:
    manifest = re.sub(
        r"#\s*cloud\.google\.com/placement-policy-name:.*",
        f"cloud.google.com/placement-policy-name: {placement_policy}",
        manifest,
    )
  else:
    manifest = re.sub(
        r"\n\s*#[^\n]*(?:placement[ -]policy)[^\n]*",
        "",
        manifest,
    )

  # Handle optional GCE reservation-name substitution in nodeSelector
  if reservation_name:
    manifest = re.sub(
        r"#\s*cloud\.google\.com/reservation-name:\s*\$\{RESERVATION_NAME\}",
        f"cloud.google.com/reservation-name: {reservation_name}",
        manifest,
    )
  else:
    manifest = re.sub(
        r"\n\s*#[^\n]*(?:Optional GCE reservation"
        r" name|reservation-name:\s*\$\{RESERVATION_NAME\})[^\n]*",
        "",
        manifest,
    )

  return manifest


def apply_jobset(manifest_yaml: str) -> None:
  """Applies the generated JobSet manifest to the Kubernetes cluster via kubectl."""
  try:
    proc = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest_yaml,
        text=True,
        capture_output=True,
        check=True,
    )
    logging.info("JobSet applied via kubectl: %s", proc.stdout.strip())
  except FileNotFoundError:
    logging.warning("kubectl binary not found on PATH; skipping apply_jobset.")


def delete_jobset(workload_name: str, namespace: str) -> None:
  """Deletes the JobSet via kubectl to release NAP nodes and cluster quota cleanly."""
  cmd = [
      "kubectl",
      "delete",
      "jobset",
      workload_name,
      "-n",
      namespace,
      "--wait=false",
  ]
  logging.info("Cleaning up JobSet [%s] on cluster...", workload_name)
  try:
    subprocess.run(cmd, check=False, capture_output=True)
  except FileNotFoundError:
    logging.warning("kubectl binary not found on PATH; skipping delete_jobset.")


def fetch_namespace_jobsets(namespace: str) -> dict[str, dict[str, Any]]:
  """Fetches all JobSets in the target namespace in a single bulk query."""
  cmd = ["kubectl", "get", "jobsets", "-n", namespace, "-o", "json"]
  try:
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
  except FileNotFoundError:
    return {}
  if res.returncode != 0 or not res.stdout.strip():
    return {}
  try:
    data = json.loads(res.stdout)
    return {
        item["metadata"]["name"]: item
        for item in data.get("items", [])
        if "metadata" in item and "name" in item["metadata"]
    }
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("Failed to parse JobSets JSON: %s", e)
    return {}


def fetch_namespace_pods(namespace: str) -> dict[str, list[dict[str, Any]]]:
  """Fetches all Pods in the target namespace grouped by workload name."""
  cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
  try:
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
  except FileNotFoundError:
    return {}
  if res.returncode != 0 or not res.stdout.strip():
    return {}
  try:
    data = json.loads(res.stdout)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pod in data.get("items", []):
      labels = pod.get("metadata", {}).get("labels", {})
      wl = labels.get("jobset.sigs.k8s.io/jobset-name")
      if wl:
        grouped.setdefault(wl, []).append(pod)
    return grouped
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("Failed to parse Pods JSON: %s", e)
    return {}


def update_workload_states(
    active_workloads: dict[str, ActiveWorkload],
    jobsets_data: dict[str, dict[str, Any]],
    pods_data: dict[str, list[dict[str, Any]]],
    timeout_minutes: Optional[int] = DEFAULT_TIMEOUT_MINUTES,
) -> bool:
  """Updates active workload objects from bulk cluster queries.

  Returns True if all workloads have reached a terminal state.
  """
  timeout_seconds = (
      timeout_minutes * 60
      if (timeout_minutes is not None and timeout_minutes > 0)
      else None
  )
  all_terminal = True

  for name, wl in active_workloads.items():
    if wl.terminal:
      continue

    now = time.time()
    wl.duration_seconds = now - wl.start_time

    # Check timeout
    if timeout_seconds is not None and wl.duration_seconds > timeout_seconds:
      wl.status = "TIMEOUT"
      wl.error_message = f"Exceeded timeout of {timeout_minutes} minutes"
      wl.terminal = True
      continue

    # Inspect Pods first to track pod-level counts
    pod_list = pods_data.get(name, [])
    ready_count = 0
    success_count = 0
    failed_count = 0
    any_running = False
    if pod_list:
      for pod in pod_list:
        status_dict = pod.get("status", {})
        phase = status_dict.get("phase", "")
        if phase == "Succeeded":
          success_count += 1
        elif phase == "Failed":
          failed_count += 1
        elif phase == "Running":
          any_running = True

        conditions = status_dict.get("conditions", [])
        is_ready = False
        for cond in conditions:
          if cond.get("type") == "Ready" and cond.get("status") == "True":
            is_ready = True
            break
        if is_ready or phase == "Succeeded":
          ready_count += 1

      wl.ready_pods = ready_count
      wl.successful_pods = success_count
      wl.failed_pods = failed_count
      wl.total_pods = max(wl.total_pods, len(pod_list))

    # Inspect JobSet status conditions
    jobset_obj = jobsets_data.get(name)
    is_suspended = False
    if jobset_obj:
      status_dict = jobset_obj.get("status", {})
      conditions = status_dict.get("conditions", [])
      for cond in conditions:
        c_type = cond.get("type")
        c_status = cond.get("status")
        c_reason = cond.get("reason", "")
        c_msg = cond.get("message", "")

        if c_type == "Completed" and c_status == "True":
          wl.status = "SUCCESS"
          wl.terminal = True
          if wl.successful_pods == 0:
            wl.successful_pods = wl.total_pods
          wl.ready_pods = wl.successful_pods
          break
        elif c_type == "Failed" and c_status == "True":
          wl.status = "FAILED"
          wl.error_message = f"JobSet Failed: {c_reason}: {c_msg}"
          wl.terminal = True
          if wl.failed_pods == 0:
            wl.failed_pods = max(1, wl.total_pods - wl.successful_pods)
          break
        elif c_type == "Suspended" and c_status == "True":
          is_suspended = True

      # Check top-level spec.suspend if present
      spec_dict = jobset_obj.get("spec", {})
      if spec_dict.get("suspend", False):
        is_suspended = True

    if wl.terminal:
      continue

    # Determine status from pods if not already terminal from JobSet conditions
    if pod_list:
      if any_running:
        wl.status = "RUNNING"
      elif success_count == len(pod_list) and len(pod_list) > 0:
        wl.status = "SUCCESS"
        wl.terminal = True
        wl.ready_pods = success_count
      elif failed_count > 0 and (failed_count + success_count == len(pod_list)):
        wl.status = "FAILED"
        wl.terminal = True
      else:
        wl.status = "PENDING"
    else:
      if not wl.terminal:
        wl.ready_pods = 0
      if is_suspended:
        wl.status = "QUEUED"
      else:
        wl.status = "PENDING"

    if not wl.terminal:
      all_terminal = False

  return all_terminal


def render_status_dashboard(
    active_workloads: dict[str, ActiveWorkload],
    start_time: float,
) -> str:
  """Renders a clean tabular status dashboard for console logging."""
  elapsed = int(time.time() - start_time)
  elapsed_str = f"{elapsed // 60}m {elapsed % 60:02d}s"
  now_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y-%m-%d %H:%M:%S UTC"
  )

  counts = {
      "SUCCESS": 0,
      "RUNNING": 0,
      "PENDING": 0,
      "QUEUED": 0,
      "SUBMITTED": 0,
      "FAILED": 0,
      "TIMEOUT": 0,
  }
  for wl in active_workloads.values():
    counts[wl.status] = counts.get(wl.status, 0) + 1

  completed_cnt = counts["SUCCESS"]
  running_cnt = counts["RUNNING"]
  queued_cnt = counts["QUEUED"] + counts["PENDING"] + counts["SUBMITTED"]
  failed_cnt = counts["FAILED"] + counts["TIMEOUT"]

  summary = (
      f"[JobSets Succeeded: {completed_cnt} | Running: {running_cnt} | "
      f"Queued/Pending: {queued_cnt} | Failed: {failed_cnt}] (Total:"
      f" {len(active_workloads)})"
  )

  lines = [
      "=" * 92,
      f"📊 [TPUMS Benchmark Progress] — {now_utc} (Elapsed: {elapsed_str})",
      f"Progress: {summary}",
      "-" * 92,
      (
          f"{'Config':<32} | {'Topo':<5} | {'Status':<12} | {'Pods':<6} |"
          f" {'Elapsed':<8} | {'Workload Name'}"
      ),
      f"{'-'*32}-+-{'-'*5}-+-{'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*20}",
  ]

  for wl in active_workloads.values():
    dur_sec = int(wl.duration_seconds)
    dur_str = f"{dur_sec // 60}m {dur_sec % 60:02d}s"
    displayed_pods = (
        wl.successful_pods if wl.status == "SUCCESS" else wl.ready_pods
    )
    pods_str = f"{displayed_pods}/{wl.total_pods}"
    status_icon = (
        "🟢 "
        if wl.status == "SUCCESS"
        else (
            "🔴 "
            if wl.status in ("FAILED", "TIMEOUT")
            else (
                "⚡ "
                if wl.status == "RUNNING"
                else ("⏳ " if wl.status == "PENDING" else "💤 ")
            )
        )
    )
    status_display = f"{status_icon}{wl.status}"
    cfg_display = wl.config_name[:32]
    # Each status_icon begins with a 2-column-wide emoji (1 Unicode code point).
    # Padding to 11 code points yields 12 visual columns, matching the header.
    lines.append(
        f"{cfg_display:<32} | {wl.topology:<5} | {status_display:<11} |"
        f" {pods_str:<6} | {dur_str:<8} | {wl.workload_name}"
    )

  lines.append("=" * 92)
  return "\n".join(lines)


@dataclass
class WorkloadManager:
  """Manages the submission, monitoring lifecycle, and cleanup of benchmark JobSets."""

  namespace: str = DEFAULT_NAMESPACE
  queue: str = DEFAULT_QUEUE
  priority: str = DEFAULT_PRIORITY
  docker_image: str = DEFAULT_DOCKER_IMAGE
  gcs_bucket: str = DEFAULT_GCS_OUTPUT_DIR
  git_branch: str = DEFAULT_GIT_BRANCH
  template_path: Optional[str] = None
  timeout_minutes: Optional[int] = DEFAULT_TIMEOUT_MINUTES
  declared_duration_minutes: int = DEFAULT_DECLARED_DURATION_MINUTES
  poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS
  tick_interval_seconds: int = DEFAULT_TICK_INTERVAL_SECONDS
  cleanup: bool = DEFAULT_CLEANUP
  upload_trace: bool = False
  dry_run: bool = False
  reservation_name: Optional[str] = DEFAULT_RESERVATION_NAME

  active_workloads: dict[str, ActiveWorkload] = field(default_factory=dict)
  start_time: float = field(default_factory=time.time)

  def prepare_workloads(
      self, config_files: list[pathlib.Path]
  ) -> list[BenchmarkResult]:
    """Prepares and registers all benchmark workloads from config files."""
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    gcs_base = (self.gcs_bucket or "").rstrip("/")
    dry_run_results: list[BenchmarkResult] = []

    for config_file in config_files:
      config_name = config_file.stem
      try:
        rel_path = pathlib.Path(os.path.abspath(config_file)).relative_to(
            pathlib.Path(os.path.abspath(__file__)).parent.parent
        )
      except ValueError:
        try:
          rel_path = config_file.resolve().relative_to(
              pathlib.Path(__file__).parent.parent.resolve()
          )
        except ValueError:
          rel_path = pathlib.Path(os.path.abspath(config_file))

      with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

      topology = infer_topology_from_config(str(config_file), config_data)
      workload_name = sanitize_workload_name(config_name, topology=topology)

      if gcs_base.startswith("gs://"):
        gcs_output_dir = f"{gcs_base}/{date_str}/{workload_name}"
      else:
        gcs_output_dir = f"gs://{gcs_base}/{date_str}/{workload_name}"

      manifest = generate_jobset_manifest(
          workload_name=workload_name,
          namespace=self.namespace,
          queue_name=self.queue,
          priority_class=self.priority,
          topology=topology,
          docker_image=self.docker_image,
          config_rel_path=str(rel_path),
          gcs_output_dir=gcs_output_dir,
          declared_duration_minutes=self.declared_duration_minutes,
          template_path=self.template_path,
          upload_trace=self.upload_trace,
          git_branch=self.git_branch,
          reservation_name=self.reservation_name,
      )

      if self.dry_run:
        print(
            "\n[Dry-Run: Workload Submitter - JobSet Manifest for"
            f" '{config_name}' ({topology})]"
        )
        print(manifest)
        dry_run_results.append(
            BenchmarkResult(
                workload_name=workload_name,
                config_name=config_name,
                config_rel_path=str(rel_path),
                topology=topology,
                status="DRY_RUN",
                duration_seconds=0.0,
                gcs_artifact_path=gcs_output_dir,
            )
        )
        continue

      self.active_workloads[workload_name] = ActiveWorkload(
          workload_name=workload_name,
          config_name=config_name,
          config_rel_path=str(rel_path),
          topology=topology,
          manifest_yaml=manifest,
          gcs_output_dir=gcs_output_dir,
          start_time=time.time(),
      )

    return dry_run_results

  def submit_all(self) -> None:
    """Dispatches all prepared JobSets to the cluster in parallel via kubectl."""
    logging.info(
        "Dispatching %d JobSet(s) in parallel via kubectl...",
        len(self.active_workloads),
    )
    for name, wl in self.active_workloads.items():
      try:
        logging.info(
            "Applying JobSet [%s] (%s, %s)...",
            name,
            wl.config_name,
            wl.topology,
        )
        apply_jobset(wl.manifest_yaml)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Failed to apply JobSet %s: %s", name, e)
        wl.status = "FAILED"
        wl.error_message = str(e)
        wl.terminal = True

  def update_states(
      self,
      jobsets_data: dict[str, dict[str, Any]],
      pods_data: dict[str, list[dict[str, Any]]],
  ) -> bool:
    """Updates workload statuses and durations; returns True if all are terminal."""
    return update_workload_states(
        self.active_workloads,
        jobsets_data,
        pods_data,
        timeout_minutes=self.timeout_minutes,
    )

  def render_dashboard(self) -> str:
    """Generates the ASCII status dashboard string."""
    return render_status_dashboard(self.active_workloads, self.start_time)

  def cleanup_workloads(self, only_terminal: bool = True) -> None:
    """Deletes JobSets from the namespace for finished (or all) workloads."""
    for name, wl in self.active_workloads.items():
      if (not only_terminal or wl.terminal) and not wl.cleaned_up:
        delete_jobset(name, self.namespace)
        wl.cleaned_up = True

  def poll(self) -> list[BenchmarkResult]:
    """Monitors active workloads in parallel with periodic status dashboards and immediate cleanup."""
    if not self.start_time:
      self.start_time = time.time()
    last_report_time = 0.0
    deadline = (
        self.start_time + (self.timeout_minutes * 60)
        if (self.timeout_minutes is not None and self.timeout_minutes > 0)
        else float("inf")
    )
    timeout_label = (
        f"{self.timeout_minutes}m"
        if (self.timeout_minutes is not None and self.timeout_minutes > 0)
        else "infinite"
    )
    lines_to_clear = 0

    def _clear_dashboard() -> None:
      nonlocal lines_to_clear
      if sys.stdout.isatty() and lines_to_clear > 0:
        sys.stdout.write(f"\033[{lines_to_clear}A\033[J")
        sys.stdout.flush()
        lines_to_clear = 0

    if self.cleanup:

      def _sig_handler(signum, frame):
        del frame
        _clear_dashboard()
        logging.warning(
            "Received interrupt signal (%d). Deleting active JobSets...", signum
        )
        self.cleanup_workloads(only_terminal=False)
        sys.exit(1)

      try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
      except (ValueError, AttributeError):
        pass  # Ignore if running outside main thread

    logging.info(
        "Tracking %d active workload(s) in parallel (Timeout: %s, Update"
        " Interval: %ds, Auto-Cleanup: %s)...",
        len(self.active_workloads),
        timeout_label,
        self.poll_interval_seconds,
        self.cleanup,
    )

    while time.time() < deadline:
      jobsets_data = fetch_namespace_jobsets(self.namespace)
      pods_data = fetch_namespace_pods(self.namespace)

      all_done = self.update_states(jobsets_data, pods_data)

      cleaned_any = False
      if self.cleanup and any(
          wl.terminal and not wl.cleaned_up
          for wl in self.active_workloads.values()
      ):
        _clear_dashboard()
        self.cleanup_workloads(only_terminal=True)
        cleaned_any = True

      now = time.time()
      if (
          now - last_report_time >= self.poll_interval_seconds
          or all_done
          or cleaned_any
      ):
        dashboard = self.render_dashboard()
        if sys.stdout.isatty():
          _clear_dashboard()
          sys.stdout.write(dashboard + "\n")
          sys.stdout.flush()
          if not all_done:
            lines_to_clear = len(dashboard.splitlines())
        else:
          print("\n" + dashboard + "\n", flush=True)
        last_report_time = now

      if all_done:
        logging.info("All parallel workloads reached terminal state.")
        break

      time.sleep(self.tick_interval_seconds)

    results = []
    for wl in self.active_workloads.values():
      results.append(
          BenchmarkResult(
              workload_name=wl.workload_name,
              config_name=wl.config_name,
              config_rel_path=wl.config_rel_path,
              topology=wl.topology,
              status=wl.status if wl.terminal else "TIMEOUT",
              duration_seconds=wl.duration_seconds,
              gcs_artifact_path=wl.gcs_output_dir,
              error_message=wl.error_message,
          )
      )

    if self.cleanup:
      self.cleanup_workloads(only_terminal=False)

    return results

  def run(self, config_files: list[pathlib.Path]) -> list[BenchmarkResult]:
    """High-level pipeline driver: prepare -> dispatch -> poll."""
    dry_run_results = self.prepare_workloads(config_files)
    if self.dry_run:
      return dry_run_results
    self.submit_all()
    return self.poll()


def poll_active_workloads(
    active_workloads: dict[str, ActiveWorkload],
    namespace: str = DEFAULT_NAMESPACE,
    timeout_minutes: Optional[int] = DEFAULT_TIMEOUT_MINUTES,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    tick_interval_seconds: int = DEFAULT_TICK_INTERVAL_SECONDS,
    cleanup: bool = DEFAULT_CLEANUP,
) -> list[BenchmarkResult]:
  """Monitors active workloads in parallel with periodic status dashboards and immediate cleanup."""
  manager = WorkloadManager(
      namespace=namespace,
      timeout_minutes=timeout_minutes,
      poll_interval_seconds=poll_interval_seconds,
      tick_interval_seconds=tick_interval_seconds,
      cleanup=cleanup,
      active_workloads=active_workloads,
  )
  return manager.poll()


def submit_workloads(
    config_files: list[pathlib.Path],
    namespace: str = DEFAULT_NAMESPACE,
    queue: str = DEFAULT_QUEUE,
    priority: str = DEFAULT_PRIORITY,
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    gcs_bucket: str = DEFAULT_GCS_OUTPUT_DIR,
    template_path: Optional[str] = None,
    timeout_minutes: Optional[int] = DEFAULT_TIMEOUT_MINUTES,
    declared_duration_minutes: int = DEFAULT_DECLARED_DURATION_MINUTES,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    cleanup: bool = DEFAULT_CLEANUP,
    upload_trace: bool = False,
    git_branch: str = DEFAULT_GIT_BRANCH,
    max_retries: int = DEFAULT_MAX_RETRIES,
    dry_run: bool = False,
    reservation_name: Optional[str] = DEFAULT_RESERVATION_NAME,
) -> list[BenchmarkResult]:
  """Prepares and submits all targeted benchmark configs in parallel using jobset_template.yaml."""
  del max_retries  # Unused in parallel batch submission
  manager = WorkloadManager(
      namespace=namespace,
      queue=queue,
      priority=priority,
      docker_image=docker_image,
      gcs_bucket=gcs_bucket,
      git_branch=git_branch,
      template_path=template_path,
      timeout_minutes=timeout_minutes,
      declared_duration_minutes=declared_duration_minutes,
      poll_interval_seconds=poll_interval_seconds,
      cleanup=cleanup,
      upload_trace=upload_trace,
      dry_run=dry_run,
      reservation_name=reservation_name,
  )
  return manager.run(config_files)


def main(argv: Sequence[str]) -> None:
  import argparse  # pylint: disable=g-import-not-at-top

  parser = argparse.ArgumentParser(
      description="Parallel TPUMS Workload Submitter & Cleaner via kubectl."
  )
  parser.add_argument(
      "--config-dir",
      dest="config_dir",
      default=DEFAULT_CONFIG_DIR,
      help=(
          "Directory of configs scanned when --filter or --topologies is used"
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
      help="One or more filter patterns for config names/paths.",
  )
  group.add_argument(
      "--topologies",
      nargs="+",
      default=None,
      help="Filter configs by target topology (e.g. '2x2x1', '2x2x2').",
  )
  group.add_argument(
      "--suite-file",
      default=None,
      help=(
          "Path to a suite YAML file listing benchmark configs to run "
          f"(default: {DEFAULT_CANONICAL_SUITE_PATH})."
      ),
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
      "--docker-image", default=DEFAULT_DOCKER_IMAGE, help="Container image."
  )
  parser.add_argument(
      "--gcs-bucket",
      default=DEFAULT_GCS_OUTPUT_DIR,
      help=f"GCS output directory (default: {DEFAULT_GCS_OUTPUT_DIR}).",
  )
  parser.add_argument(
      "--git-branch",
      default=DEFAULT_GIT_BRANCH,
      help=(
          "Git branch to clone in the JobSet container (default:"
          f" {DEFAULT_GIT_BRANCH})."
      ),
  )
  parser.add_argument(
      "--template-path",
      default=DEFAULT_TEMPLATE_PATH,
      help=f"Path to JobSet template (default: {DEFAULT_TEMPLATE_PATH}).",
  )
  parser.add_argument(
      "--timeout-minutes",
      type=int,
      default=DEFAULT_TIMEOUT_MINUTES,
      help="Workload timeout in minutes (default: None / infinite).",
  )
  parser.add_argument(
      "--poll-interval",
      type=int,
      default=DEFAULT_POLL_INTERVAL_SECONDS,
      help="Periodic status dashboard interval in seconds (default: 60).",
  )
  parser.add_argument(
      "--cleanup",
      action=argparse.BooleanOptionalAction,
      default=DEFAULT_CLEANUP,
      help=(
          "Automatically delete JobSets on the cluster after completion"
          " (default: False / --no-cleanup)."
      ),
  )
  parser.add_argument(
      "--upload-trace",
      action="store_true",
      default=False,
      help="Upload XLA/TensorBoard trace directories to GCS (default: False).",
  )
  parser.add_argument(
      "--reservation-name",
      default=DEFAULT_RESERVATION_NAME,
      help=(
          "Optional GCE reservation name to add to JobSet nodeSelector "
          f"(default: {DEFAULT_RESERVATION_NAME or 'None'})."
      ),
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print JobSet YAML manifests without submitting to cluster.",
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
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )

  config_files = resolve_config_files(
      config_dir=args.config_dir,
      configs=args.configs,
      filter_patterns=args.filter,
      topologies=args.topologies,
      suite_file=args.suite_file,
  )

  logging.info(
      "Resolved %d benchmark configuration(s) for submission.",
      len(config_files),
  )
  submit_workloads(
      config_files=config_files,
      namespace=args.namespace,
      queue=args.queue,
      priority=args.priority,
      docker_image=args.docker_image,
      gcs_bucket=args.gcs_bucket,
      git_branch=args.git_branch,
      template_path=args.template_path,
      timeout_minutes=args.timeout_minutes,
      poll_interval_seconds=args.poll_interval,
      cleanup=args.cleanup,
      upload_trace=args.upload_trace,
      dry_run=args.dry_run,
      reservation_name=args.reservation_name,
  )


if __name__ == "__main__":
  main(sys.argv)
