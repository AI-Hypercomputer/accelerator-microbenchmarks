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

import argparse
import collections
from collections.abc import Sequence
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
from typing import Any, Optional

from automation import utils
import yaml

DEFAULT_TICK_INTERVAL_SECONDS = 15
DEFAULT_CLEANUP = False
# Consecutive failed cluster status queries tolerated by `poll()` before it
# gives up (~75s at the default tick interval).
MAX_CONSECUTIVE_POLL_FAILURES = 5


def _find_existing(
    path: str | os.PathLike[str], *bases: pathlib.Path
) -> Optional[pathlib.Path]:
  """Resolves `path` (absolute, or relative to the first base containing it).

  Args:
    path: Absolute or relative file/directory path.
    *bases: Directories tried in order when `path` is relative (default: the
      package root, then the current directory, so package-relative paths such
      as 'configs/v6e/2x2/gemm.yaml' resolve inside the package even when run
      from another checkout of it).

  Returns:
    The absolute path if it exists, else None.
  """
  p = pathlib.Path(path)
  bases = bases or (utils.PACKAGE_ROOT, pathlib.Path.cwd())
  candidates = [p] if p.is_absolute() else [base / p for base in bases]
  return next(
      (pathlib.Path(os.path.abspath(c)) for c in candidates if c.exists()), None
  )


def load_canonical_configs(suite_path: str) -> list[str]:
  """Loads benchmark config paths from a resolved suite YAML file.

  Args:
    suite_path: Path to the suite YAML file.

  Returns:
    The suite's config paths, relative to the package root.

  Raises:
    FileNotFoundError: If `suite_path` does not exist.
    ValueError: If the suite is not a mapping with a non-empty 'configs' list.
  """
  with open(suite_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  if not isinstance(data, dict):
    raise ValueError(
        f"Invalid format in suite YAML file {suite_path}: expected a"
        " mapping with a 'configs' list."
    )
  configs_val = data.get("configs")
  if not isinstance(configs_val, list):
    raise ValueError(
        f"Suite YAML file {suite_path} must contain a 'configs' list."
    )
  configs = [str(item) for item in configs_val if item]

  if not configs:
    raise ValueError(
        f"No benchmark configurations found in suite YAML file {suite_path}."
    )
  return configs


def fetch_cluster_credentials(
    cluster: str,
    project: str,
    region: str = utils.DEFAULT_REGION,
    zone: Optional[str] = None,
    dry_run: bool = False,
) -> None:
  """Fetches cluster credentials via gcloud, trying `region` then `zone`."""
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

  locations = [("region", region)] + ([("zone", zone)] if zone else [])
  for i, (kind, location) in enumerate(locations):
    try:
      subprocess.run(
          cmd + [f"--{kind}={location}"],
          check=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          stdin=subprocess.DEVNULL,
          timeout=utils.CLI_TIMEOUT_SECONDS,
      )
      logging.info("Successfully fetched credentials via %s %s", kind, location)
      return
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
      is_last = i + 1 == len(locations)
      (logging.error if is_last else logging.debug)(
          "Failed to fetch credentials via %s %s: %s",
          kind,
          location,
          e.output or e,  # TimeoutExpired.output is None if nothing printed.
      )
      if is_last:
        raise


@dataclasses.dataclass
class BenchmarkResult:
  """Parsed results for an individual microbenchmark run."""

  workload_name: str
  config_name: str
  config_rel_path: str
  topology: str
  status: str  # SUCCESS, FAILED, TIMEOUT, DRY_RUN
  duration_seconds: float
  gcs_artifact_path: str
  metrics: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  error_message: Optional[str] = None


@dataclasses.dataclass
class ActiveWorkload:
  """State tracking object for an actively running or queued JobSet."""

  workload_name: str
  config_name: str
  config_rel_path: str
  topology: str
  manifest_yaml: str
  gcs_output_dir: str
  start_time: float
  total_pods: int  # Pods (= TPU hosts) in the JobSet.
  status: (
      str
  ) = (  # SUBMITTED, QUEUED, PENDING, RUNNING, SUCCESS, FAILED, TIMEOUT
      "SUBMITTED"
  )
  ready_pods: int = 0
  successful_pods: int = 0
  failed_pods: int = 0
  duration_seconds: float = 0.0
  error_message: Optional[str] = None
  terminal: bool = False
  cleaned_up: bool = False


def _package_relative_path(config_file: pathlib.Path) -> str:
  """Returns `config_file` relative to the package root (absolute if outside)."""
  abs_path = pathlib.Path(os.path.abspath(config_file))
  if abs_path.is_relative_to(utils.PACKAGE_ROOT):
    return str(abs_path.relative_to(utils.PACKAGE_ROOT))
  return str(abs_path)


def resolve_config_files(
    config_dir: Optional[str] = None,
    configs: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    topologies: Optional[list[str]] = None,
    suite_file: Optional[str] = None,
    accelerator: Optional[str] = None,
) -> list[pathlib.Path]:
  """Resolves the config files of exactly one selection mode.

  Modes (flags of different modes cannot be mixed):

    1. `suite_file`: the configs listed in the suite YAML.
    2. `configs`: explicit config files.
    3. `accelerator`: every topology config under `config_dir` (default:
       configs/<accelerator>), optionally narrowed by `topologies` and/or
       `categories` (AND semantics).

  Relative paths resolve against the package root first, then the current
  directory, because JobSets can only run configs inside the package.

  Args:
    config_dir: Directory scanned in accelerator mode.
    configs: Explicit config file paths.
    categories: Benchmark name substrings matched against config file names
      (accelerator mode).
    topologies: Topologies to keep, e.g. ['2x2x1'] (accelerator mode).
    suite_file: Suite YAML listing the configs to run.
    accelerator: Accelerator whose configs are scanned.

  Returns:
    Absolute, de-duplicated paths of the selected config files, all targeting
    the same accelerator.

  Raises:
    ValueError: If no or more than one selection mode is given, a
      mode-specific flag is used in another mode, a config is a directory,
      a config's topology/accelerator cannot be inferred, or the selected
      configs do not all target one accelerator.
    FileNotFoundError: If a config file, the config directory or the suite
      file cannot be found.
  """
  modes = [
      flag
      for flag, value in (
          ("--suite-file", suite_file),
          ("--configs", configs),
          ("--accelerator", accelerator),
      )
      if value
  ]
  if len(modes) != 1:
    raise ValueError(
        "Specify exactly one of --suite-file, --configs or --accelerator"
        f" (got {modes or 'none'})."
    )
  scan_only = [
      flag
      for flag, value in (
          ("--config-dir", config_dir),
          ("--topologies", topologies),
          ("--categories", categories),
      )
      if value
  ]
  if scan_only and not accelerator:
    raise ValueError(
        f"{scan_only} can only be combined with --accelerator, not with"
        f" {modes[0]}."
    )

  if suite_file:
    suite_source = _find_existing(suite_file)
    if suite_source is None:
      raise FileNotFoundError(f"Suite YAML file not found: {suite_file}")
    resolved = []
    for c in load_canonical_configs(str(suite_source)):
      # Entries become the JobSet CONFIG_PATH inside the GitHub checkout, so
      # they are package-relative by contract.
      p = _find_existing(c, utils.PACKAGE_ROOT)
      if p is None:
        raise FileNotFoundError(
            f"Config {c!r} listed in suite {suite_source} not found."
        )
      resolved.append(p)
    logging.info("Resolved suite %s (%d entries).", suite_source, len(resolved))
  elif configs:
    resolved = []
    for c in configs:
      p = _find_existing(c)
      if p is None:
        raise FileNotFoundError(f"Config file not found: {c}")
      if p.is_dir():
        raise ValueError(
            f"--configs expects YAML files but {c!r} is a directory; use"
            " --accelerator <acc> --config-dir <dir> to scan a directory."
        )
      resolved.append(p)
  else:
    acc = utils.normalize_accelerator(accelerator)
    topo_set = set(utils.normalize_topologies(acc, topologies))
    patterns = [t.lower() for t in utils.split_csv_tokens(categories)]
    if categories and not patterns:
      raise ValueError(f"--categories {categories} has no non-empty names.")
    config_dir = config_dir or utils.get_accelerator_config(acc).config_dir
    config_path = _find_existing(config_dir)
    if config_path is None or not config_path.is_dir():
      raise FileNotFoundError(f"Config directory not found: {config_dir}")
    logging.info("Scanning config directory %s.", config_path)
    resolved = []
    for p in sorted(config_path.glob("**/*.yaml")):
      if patterns and not any(pat in p.stem.lower() for pat in patterns):
        continue
      try:
        topology = utils.extract_topology(_package_relative_path(p))
      except ValueError:
        # e.g. configs/tpu7x/attention_jax_flash.yaml: not a nightly target.
        logging.warning("Skipping %s: no topology directory in its path.", p)
        continue
      if topo_set and topology not in topo_set:
        continue
      resolved.append(p)

  resolved = list(dict.fromkeys(resolved))  # De-duplicate, keeping order.
  infer_configs_accelerator(resolved, accelerator=accelerator)
  return resolved


def infer_configs_accelerator(
    config_files: Sequence[pathlib.Path], accelerator: Optional[str] = None
) -> Optional[str]:
  """Returns the single accelerator targeted by `config_files`.

  Args:
    config_files: Config paths; each must encode its accelerator and topology
      (e.g. configs/v6e/2x2/gemm.yaml).
    accelerator: Optional expected accelerator every config must match.

  Returns:
    The common accelerator, or `accelerator` (None if unset) when
    `config_files` is empty.

  Raises:
    ValueError: If a config's topology or accelerator cannot be inferred,
      contradicts `accelerator`, or the configs span multiple accelerators.
  """
  by_acc: dict[str, list[str]] = {}
  for p in config_files:
    rel = _package_relative_path(p)
    acc = utils.infer_accelerator(
        accelerator, config_path=rel, topology=utils.extract_topology(rel)
    )
    by_acc.setdefault(acc, []).append(rel)
  if len(by_acc) > 1:
    raise ValueError(
        "Selected configs span multiple accelerators"
        f" {sorted(by_acc)}: {by_acc}. Run each accelerator separately."
    )
  return next(iter(by_acc), accelerator)


def infer_topology_from_config(config_path: str) -> str:
  """Infers the target TPU topology from the config file path.

  Args:
    config_path: Config path whose directory segments encode the topology (e.g.
      'configs/v6e/4x4/all_gather.yaml'). The last match wins.

  Returns:
    Topology string such as '2x2x1' (tpu7x) or '4x4' (v6e).

  Raises:
    ValueError: If no topology can be found in `config_path`. Provisioning a
      guessed topology would waste capacity and produce misleading metrics.
  """
  topology = utils.extract_topology(config_path)
  if topology not in utils.KNOWN_TOPOLOGIES:
    logging.warning(
        "Topology %r from %r is not a known topology %s; proceeding with"
        " dynamically computed placement info.",
        topology,
        config_path,
        sorted(utils.KNOWN_TOPOLOGIES),
    )
  return topology


def sanitize_workload_name(
    base_name: str, topology: str, max_len: int = 45
) -> str:
  """Generates a safe DNS-compliant Kubernetes JobSet name incorporating topology without timestamps."""
  clean_base = re.sub(r"[^a-z0-9-]", "-", base_name.lower()).strip("-")
  clean_topo = re.sub(r"[^a-z0-9-]", "-", topology.lower()).strip("-")
  if clean_topo and clean_topo not in clean_base:
    clean_base = f"{clean_topo}-{clean_base}"
  full_name = re.sub(r"-+", "-", f"tpums-{clean_base}").strip("-")
  return full_name[:max_len].rstrip("-")


def generate_jobset_manifest(
    workload_name: str,
    namespace: str,
    queue_name: str,
    priority_class: str,
    topology: str,
    docker_image: str,
    config_rel_path: str,
    gcs_output_dir: str,
    declared_duration_minutes: int = utils.DEFAULT_DECLARED_DURATION_MINUTES,
    template_path: str = utils.DEFAULT_TEMPLATE_PATH,
    upload_trace: bool = False,
    git_branch: str = utils.DEFAULT_GIT_BRANCH,
    reservation_name: Optional[str] = utils.DEFAULT_RESERVATION_NAME,
    accelerator: Optional[str] = None,
    topo_info: Optional[dict[str, Any]] = None,
) -> str:
  """Generates a native GKE JobSet manifest using automation/jobset_template.yaml."""
  if topo_info is None:
    acc = utils.infer_accelerator(
        accelerator, config_path=config_rel_path, topology=topology
    )
    topo_info = utils.compute_topology_info(topology, accelerator=acc)
  num_nodes = topo_info["num_nodes"]
  placement_policy = topo_info["placement_policy"]

  with open(template_path, "r", encoding="utf-8") as f:
    template_raw = f.read()

  # Strip leading commentary before apiVersion: jobset.x-k8s.io
  api_idx = template_raw.find("apiVersion: jobset.x-k8s.io")
  manifest = template_raw[api_idx:] if api_idx != -1 else template_raw

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
      "${ACCELERATOR}": topo_info["gke_accelerator"],
      "${TOPOLOGY}": topology,
      "${EXCLUSIVE_TOPOLOGY}": exclusive_topology,
      "${PARALLELISM}": str(num_nodes),
      "${COMPLETIONS}": str(num_nodes),
      "${CONFIG_PATH}": config_rel_path,
      "${GIT_BRANCH}": git_branch,
      "${GCS_OUTPUT_DIR}": gcs_output_dir,
      "${DECLARED_DURATION}": str(declared_duration_minutes),
      "${DOCKER_IMAGE}": docker_image,
      "${CHIPS_PER_NODE}": str(topo_info["chips_per_node"]),
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


def apply_jobset(manifest_yaml: str) -> str:
  """Creates a JobSet from the manifest via `kubectl create`.

  Args:
    manifest_yaml: JobSet manifest using `metadata.generateName`.

  Returns:
    The server-generated JobSet name.

  Raises:
    FileNotFoundError: If kubectl is not on PATH.
    subprocess.CalledProcessError: If kubectl exits non-zero.
    subprocess.TimeoutExpired: If kubectl does not finish in time.
    RuntimeError: If the created JobSet name cannot be parsed from the output.
  """
  proc = subprocess.run(
      ["kubectl", "create", "-f", "-"],
      input=manifest_yaml,
      text=True,
      capture_output=True,
      check=True,
      timeout=utils.CLI_TIMEOUT_SECONDS,
  )
  out = proc.stdout.strip()
  logging.info("JobSet created via kubectl: %s", out)
  match = re.search(r"jobset(?:\.[a-z0-9.-]+)?/([^\s]+)", out, re.IGNORECASE)
  if not match:
    raise RuntimeError(
        f"Could not parse the created JobSet name from kubectl output: {out!r}"
    )
  return match.group(1)


def delete_jobset(workload_name: str, namespace: str) -> bool:
  """Deletes the JobSet via kubectl to release NAP nodes and cluster quota.

  Args:
    workload_name: JobSet name.
    namespace: Namespace of the JobSet.

  Returns:
    True if the JobSet is gone (deleted now or already absent), else False.

  Raises:
    FileNotFoundError: If kubectl is not on PATH.
  """
  cmd = [
      "kubectl",
      "delete",
      "jobset",
      workload_name,
      "-n",
      namespace,
      "--ignore-not-found",
      "--wait=false",
  ]
  logging.info("Cleaning up JobSet [%s] on cluster...", workload_name)
  try:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
  except subprocess.TimeoutExpired as e:
    logging.error(
        "Timed out deleting JobSet %s after %ss.", workload_name, e.timeout
    )
    return False
  if proc.returncode != 0:
    logging.error(
        "Failed to delete JobSet %s (exit %d): %s",
        workload_name,
        proc.returncode,
        (proc.stderr or "").strip() or "<no stderr>",
    )
    return False
  return True


def _kubectl_get_items(resource: str, namespace: str) -> list[dict[str, Any]]:
  """Returns the items of `kubectl get <resource> -n <namespace> -o json`.

  Args:
    resource: Kubernetes resource type (e.g. 'jobsets', 'pods').
    namespace: Namespace to query.

  Returns:
    The `items` list of the returned Kubernetes List object.

  Raises:
    FileNotFoundError: If kubectl is not on PATH.
    RuntimeError: If kubectl times out, exits non-zero or prints invalid JSON.
  """
  cmd = ["kubectl", "get", resource, "-n", namespace, "-o", "json"]
  try:
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )
  except subprocess.TimeoutExpired as e:
    raise RuntimeError(f"`{' '.join(cmd)}` timed out after {e.timeout}s") from e
  if res.returncode != 0:
    raise RuntimeError(
        f"`{' '.join(cmd)}` failed (exit {res.returncode}):"
        f" {(res.stderr or '').strip() or '<no stderr>'}"
    )
  try:
    return json.loads(res.stdout).get("items", [])
  except json.JSONDecodeError as e:
    raise RuntimeError(f"`{' '.join(cmd)}` returned invalid JSON: {e}") from e


def fetch_namespace_jobsets(namespace: str) -> dict[str, dict[str, Any]]:
  """Fetches all JobSets in the target namespace in a single bulk query.

  Args:
    namespace: Namespace to query.

  Returns:
    JobSet objects keyed by JobSet name.

  Raises:
    FileNotFoundError: If kubectl is not on PATH.
    RuntimeError: If kubectl exits non-zero or prints invalid JSON.
  """
  return {
      item["metadata"]["name"]: item
      for item in _kubectl_get_items("jobsets", namespace)
      if "name" in item.get("metadata", {})
  }


def fetch_namespace_pods(namespace: str) -> dict[str, list[dict[str, Any]]]:
  """Fetches all Pods in the target namespace grouped by workload name.

  Args:
    namespace: Namespace to query.

  Returns:
    Pod objects grouped by their `jobset.sigs.k8s.io/jobset-name` label.

  Raises:
    FileNotFoundError: If kubectl is not on PATH.
    RuntimeError: If kubectl exits non-zero or prints invalid JSON.
  """
  grouped: dict[str, list[dict[str, Any]]] = {}
  for pod in _kubectl_get_items("pods", namespace):
    labels = pod.get("metadata", {}).get("labels", {})
    wl = labels.get("jobset.sigs.k8s.io/jobset-name")
    if wl:
      grouped.setdefault(wl, []).append(pod)
  return grouped


def _timeout_seconds(timeout_minutes: Optional[int]) -> Optional[int]:
  """Returns the timeout in seconds, or None for no (or non-positive) timeout."""
  return (
      timeout_minutes * 60 if timeout_minutes and timeout_minutes > 0 else None
  )


def update_workload_states(
    active_workloads: dict[str, ActiveWorkload],
    jobsets_data: dict[str, dict[str, Any]],
    pods_data: dict[str, list[dict[str, Any]]],
    timeout_minutes: Optional[int] = utils.DEFAULT_TIMEOUT_MINUTES,
) -> bool:
  """Updates active workload objects from bulk cluster queries.

  Args:
    active_workloads: Workloads keyed by JobSet name; updated in place.
    jobsets_data: `kubectl get jobsets` items keyed by JobSet name.
    pods_data: `kubectl get pods` items grouped by JobSet name.
    timeout_minutes: Per-workload timeout; None or <= 0 means no timeout.

  Returns:
    True if all workloads have reached a terminal state.
  """
  timeout_seconds = _timeout_seconds(timeout_minutes)
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

        is_ready = any(
            cond.get("type") == "Ready" and cond.get("status") == "True"
            for cond in status_dict.get("conditions", [])
        )
        if is_ready or phase == "Succeeded":
          ready_count += 1

      wl.ready_pods = ready_count
      wl.successful_pods = success_count
      wl.failed_pods = failed_count
      wl.total_pods = max(wl.total_pods, len(pod_list))

    # Inspect JobSet status conditions
    jobset_obj = jobsets_data.get(name)
    is_suspended = False
    if jobset_obj is not None:
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
    else:
      # apply_jobset() saw it created and this listing succeeded (fetch errors
      # raise), so the JobSet was deleted before reporting completion.
      if pod_list and success_count >= wl.total_pods:
        wl.status = "SUCCESS"
        wl.ready_pods = success_count
      else:
        wl.status = "FAILED"
        wl.error_message = (
            f"JobSet {name} disappeared from the cluster before completing."
        )
      wl.terminal = True

    if wl.terminal:
      continue

    # Determine status from pods if not already terminal from JobSet conditions
    if pod_list:
      if any_running:
        wl.status = "RUNNING"
      elif success_count >= wl.total_pods:
        wl.status = "SUCCESS"
        wl.terminal = True
        wl.ready_pods = success_count
      elif failed_count > 0 and (failed_count + success_count >= wl.total_pods):
        wl.status = "FAILED"
        wl.terminal = True
      else:
        wl.status = "PENDING"
    else:
      wl.ready_pods = 0
      wl.status = "QUEUED" if is_suspended else "PENDING"

    if not wl.terminal:
      all_terminal = False

  return all_terminal


_STATUS_ICONS = {
    "SUCCESS": "🟢 ",
    "FAILED": "🔴 ",
    "TIMEOUT": "🔴 ",
    "RUNNING": "⚡ ",
    "PENDING": "⏳ ",
}


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

  counts = collections.Counter(wl.status for wl in active_workloads.values())

  completed_cnt = counts["SUCCESS"]
  running_cnt = counts["RUNNING"]
  queued_cnt = counts["QUEUED"] + counts["PENDING"] + counts["SUBMITTED"]
  failed_cnt = sum(counts[status] for status in utils.FAILED_STATUSES)

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
    status_display = f"{_STATUS_ICONS.get(wl.status, '💤 ')}{wl.status}"
    cfg_display = wl.config_name[:32]
    # Each status icon begins with a 2-column-wide emoji (1 Unicode code point).
    # Padding to 11 code points yields 12 visual columns, matching the header.
    lines.append(
        f"{cfg_display:<32} | {wl.topology:<5} | {status_display:<11} |"
        f" {pods_str:<6} | {dur_str:<8} | {wl.workload_name}"
    )

  lines.append("=" * 92)
  return "\n".join(lines)


@dataclasses.dataclass
class WorkloadManager:
  """Manages the submission, monitoring lifecycle, and cleanup of benchmark JobSets."""

  namespace: str = utils.DEFAULT_NAMESPACE
  queue: str = utils.DEFAULT_QUEUE
  priority: str = utils.DEFAULT_PRIORITY
  docker_image: str = utils.DEFAULT_DOCKER_IMAGE
  gcs_bucket: str = utils.DEFAULT_GCS_BUCKET
  git_branch: str = utils.DEFAULT_GIT_BRANCH
  template_path: str = utils.DEFAULT_TEMPLATE_PATH
  timeout_minutes: Optional[int] = utils.DEFAULT_TIMEOUT_MINUTES
  declared_duration_minutes: int = utils.DEFAULT_DECLARED_DURATION_MINUTES
  poll_interval_seconds: int = utils.DEFAULT_POLL_INTERVAL_SECONDS
  tick_interval_seconds: int = DEFAULT_TICK_INTERVAL_SECONDS
  cleanup: bool = DEFAULT_CLEANUP
  upload_trace: bool = False
  dry_run: bool = False
  reservation_name: Optional[str] = utils.DEFAULT_RESERVATION_NAME
  accelerator: Optional[str] = None

  active_workloads: dict[str, ActiveWorkload] = dataclasses.field(
      default_factory=dict
  )
  start_time: float = dataclasses.field(default_factory=time.time)

  def prepare_workloads(
      self, config_files: list[pathlib.Path]
  ) -> list[BenchmarkResult]:
    """Prepares and registers all benchmark workloads from config files."""
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    dry_run_results: list[BenchmarkResult] = []
    # Workload name -> config, including dry-run configs never registered.
    prepared: dict[str, str] = {
        name: wl.config_rel_path for name, wl in self.active_workloads.items()
    }

    for config_file in config_files:
      config_name = config_file.stem
      rel_path = _package_relative_path(config_file)
      if os.path.isabs(rel_path):
        raise ValueError(
            f"Config {config_file} is outside {utils.PACKAGE_ROOT}. JobSets run"
            " configs from the GitHub checkout of --git-branch, so only configs"
            " inside the package (and exported to GitHub) can be submitted."
        )
      topology = infer_topology_from_config(rel_path)
      workload_name = sanitize_workload_name(config_name, topology=topology)
      if workload_name in prepared:
        raise ValueError(
            f"Duplicate sanitized workload name {workload_name!r} for config"
            f" {config_file} (collides with {prepared[workload_name]!r})."
        )
      prepared[workload_name] = rel_path
      gcs_output_dir = utils.format_gcs_uri(
          self.gcs_bucket, date_str, workload_name
      )

      # Validates self.accelerator against the config path and topology.
      acc = utils.infer_accelerator(
          self.accelerator, config_path=rel_path, topology=topology
      )
      topo_info = utils.compute_topology_info(topology, accelerator=acc)
      manifest = generate_jobset_manifest(
          workload_name=workload_name,
          namespace=self.namespace,
          queue_name=self.queue,
          priority_class=self.priority,
          topology=topology,
          docker_image=self.docker_image,
          config_rel_path=rel_path,
          gcs_output_dir=gcs_output_dir,
          declared_duration_minutes=self.declared_duration_minutes,
          template_path=self.template_path,
          upload_trace=self.upload_trace,
          git_branch=self.git_branch,
          reservation_name=self.reservation_name,
          accelerator=acc,
          topo_info=topo_info,
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
                config_rel_path=rel_path,
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
          config_rel_path=rel_path,
          topology=topology,
          manifest_yaml=manifest,
          gcs_output_dir=gcs_output_dir,
          start_time=time.time(),
          total_pods=topo_info["num_nodes"],
      )

    return dry_run_results

  def submit_all(self) -> None:
    """Dispatches all prepared JobSets to the cluster in parallel via kubectl."""
    logging.info(
        "Dispatching %d JobSet(s) in parallel via kubectl...",
        len(self.active_workloads),
    )
    updated_workloads: dict[str, ActiveWorkload] = {}
    for name, wl in list(self.active_workloads.items()):
      key = name
      try:
        logging.info(
            "Applying JobSet [%s] (%s, %s)...",
            name,
            wl.config_name,
            wl.topology,
        )
        wl.workload_name = apply_jobset(wl.manifest_yaml)
        key = wl.workload_name
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Failed to apply JobSet %s: %s", name, e)
        wl.status = "FAILED"
        wl.error_message = str(e)
        wl.terminal = True
      updated_workloads[key] = wl
    self.active_workloads = updated_workloads

  def cleanup_workloads(self, only_terminal: bool = True) -> None:
    """Deletes JobSets from the namespace for finished (or all) workloads."""
    for name, wl in self.active_workloads.items():
      if (not only_terminal or wl.terminal) and not wl.cleaned_up:
        # Failed deletes stay un-cleaned and are retried on the next tick.
        wl.cleaned_up = delete_jobset(name, self.namespace)

  def poll(self) -> list[BenchmarkResult]:
    """Monitors active workloads in parallel with periodic status dashboards and immediate cleanup."""
    last_report_time = 0.0
    timeout_seconds = _timeout_seconds(self.timeout_minutes)
    deadline = self.start_time + (timeout_seconds or float("inf"))
    timeout_label = (
        f"{self.timeout_minutes}m" if timeout_seconds else "infinite"
    )
    lines_to_clear = 0
    consecutive_failures = 0

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
      try:
        jobsets_data = fetch_namespace_jobsets(self.namespace)
        pods_data = fetch_namespace_pods(self.namespace)
      except RuntimeError as e:
        # Tolerate transient API-server errors; a missing kubectl
        # (FileNotFoundError) propagates immediately.
        consecutive_failures += 1
        _clear_dashboard()
        logging.error(
            "Cluster status query failed (%d/%d): %s",
            consecutive_failures,
            MAX_CONSECUTIVE_POLL_FAILURES,
            e,
        )
        if consecutive_failures >= MAX_CONSECUTIVE_POLL_FAILURES:
          raise
        time.sleep(self.tick_interval_seconds)
        continue
      consecutive_failures = 0

      all_done = update_workload_states(
          self.active_workloads,
          jobsets_data,
          pods_data,
          timeout_minutes=self.timeout_minutes,
      )

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
        dashboard = render_status_dashboard(
            self.active_workloads, self.start_time
        )
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
      leaked = [
          n for n, wl in self.active_workloads.items() if not wl.cleaned_up
      ]
      if leaked:
        # Don't raise: the results and report still matter.
        logging.error(
            "JobSets not deleted; run `kubectl delete jobset -n %s %s`",
            self.namespace,
            " ".join(leaked),
        )

    return results

  def run(self, config_files: list[pathlib.Path]) -> list[BenchmarkResult]:
    """High-level pipeline driver: prepare -> dispatch -> poll."""
    dry_run_results = self.prepare_workloads(config_files)
    if self.dry_run:
      return dry_run_results
    self.submit_all()
    return self.poll()


def submit_workloads(
    config_files: list[pathlib.Path],
    namespace: str = utils.DEFAULT_NAMESPACE,
    queue: str = utils.DEFAULT_QUEUE,
    priority: str = utils.DEFAULT_PRIORITY,
    docker_image: str = utils.DEFAULT_DOCKER_IMAGE,
    gcs_bucket: str = utils.DEFAULT_GCS_BUCKET,
    template_path: str = utils.DEFAULT_TEMPLATE_PATH,
    timeout_minutes: Optional[int] = utils.DEFAULT_TIMEOUT_MINUTES,
    declared_duration_minutes: int = utils.DEFAULT_DECLARED_DURATION_MINUTES,
    poll_interval_seconds: int = utils.DEFAULT_POLL_INTERVAL_SECONDS,
    cleanup: bool = DEFAULT_CLEANUP,
    upload_trace: bool = False,
    git_branch: str = utils.DEFAULT_GIT_BRANCH,
    dry_run: bool = False,
    reservation_name: Optional[str] = utils.DEFAULT_RESERVATION_NAME,
    accelerator: Optional[str] = None,
) -> list[BenchmarkResult]:
  """Prepares and submits all targeted benchmark configs in parallel using jobset_template.yaml."""
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
      accelerator=accelerator,
  )
  return manager.run(config_files)


def main(argv: Sequence[str]) -> None:
  parser = argparse.ArgumentParser(
      description="Parallel TPUMS Workload Submitter & Cleaner via kubectl."
  )
  utils.add_selection_args(parser)
  utils.add_workload_args(parser, default_cleanup=DEFAULT_CLEANUP)
  parser.add_argument(
      "--template-path",
      default=utils.DEFAULT_TEMPLATE_PATH,
      help=f"Path to JobSet template (default: {utils.DEFAULT_TEMPLATE_PATH}).",
  )
  args = parser.parse_args(argv[1:])
  utils.validate_workload_args(args)
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )

  config_files = resolve_config_files(
      config_dir=args.config_dir,
      configs=args.configs,
      categories=args.categories,
      topologies=args.topologies,
      suite_file=args.suite_file,
      accelerator=args.accelerator,
  )

  if not config_files:
    raise ValueError(
        "No YAML configuration files found matching selection criteria."
    )
  accelerator = infer_configs_accelerator(config_files, args.accelerator)
  logging.info(
      "Resolved %d %s benchmark configuration(s) for submission.",
      len(config_files),
      accelerator,
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
      declared_duration_minutes=args.declared_duration_minutes,
      poll_interval_seconds=args.poll_interval,
      cleanup=args.cleanup,
      upload_trace=args.upload_trace,
      dry_run=args.dry_run,
      reservation_name=args.reservation_name,
      accelerator=accelerator,
  )


if __name__ == "__main__":
  main(sys.argv)
