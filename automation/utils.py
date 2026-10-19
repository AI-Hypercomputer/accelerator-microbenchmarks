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

"""Shared constants, accelerator/topology helpers, and CLI flags for TPUMS automation."""

import argparse
from collections.abc import Iterable
import dataclasses
import math
import os
import pathlib
import re
from typing import Any, Optional

ENV_GIT_COMMIT = "git_commit"
ENV_JAX_VERSION = "jax_version"
ENV_LIBTPU_VERSION = "libtpu_version"

DEFAULT_CLUSTER = os.environ.get("CLUSTER", "")
DEFAULT_PROJECT = os.environ.get("PROJECT", "")
DEFAULT_REGION = os.environ.get("REGION", "us-central1")
DEFAULT_ZONE = os.environ.get("ZONE", f"{DEFAULT_REGION}-c")
DEFAULT_NAMESPACE = os.environ.get("NAMESPACE", "default")
DEFAULT_QUEUE = os.environ.get("QUEUE", "multislice-queue")
DEFAULT_PRIORITY = os.environ.get("PRIORITY", "medium")
DEFAULT_DOCKER_IMAGE = os.environ.get(
    "DOCKER_IMAGE", "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest"
)
DEFAULT_GIT_BRANCH = os.environ.get("GIT_BRANCH", "main")
DEFAULT_GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
DEFAULT_RESERVATION_NAME = os.environ.get("RESERVATION_NAME", "")
DEFAULT_TIMEOUT_MINUTES: Optional[int] = None
DEFAULT_DECLARED_DURATION_MINUTES = 30
DEFAULT_POLL_INTERVAL_SECONDS = 60
# Upper bound for one kubectl/gcloud invocation, which can otherwise block
# forever on a stuck connection or an interactive auth prompt.
CLI_TIMEOUT_SECONDS = 300
# Terminal workload statuses counted as failures in logs and reports.
FAILED_STATUSES = ("FAILED", "TIMEOUT")

AUTOMATION_DIR = pathlib.Path(os.path.abspath(__file__)).parent
# Package root (third_party/py/accelerator_microbenchmarks); config paths in
# suites and JobSets are relative to it.
PACKAGE_ROOT = AUTOMATION_DIR.parent
_CONFIGS_ROOT = PACKAGE_ROOT / "configs"
DEFAULT_TEMPLATE_PATH = str(AUTOMATION_DIR / "jobset_template.yaml")


@dataclasses.dataclass(frozen=True)
class AcceleratorDefaultConfig:
  """Default hardware, topology, and suite configuration for a TPU generation."""

  accelerator: str
  gke_accelerator_label: str
  cores_per_chip: int
  default_chips_per_node: int
  topology_dims: int
  known_topologies: tuple[str, ...]
  suite_filename: str

  @property
  def config_dir(self) -> str:
    """Default --config-dir scanned in --accelerator mode."""
    return str(_CONFIGS_ROOT / self.accelerator)

  @property
  def suite_path(self) -> str:
    """Path to the canonical nightly suite YAML for this accelerator."""
    return str(AUTOMATION_DIR / self.suite_filename)


ACCELERATOR_CONFIGS: dict[str, AcceleratorDefaultConfig] = {
    "tpu7x": AcceleratorDefaultConfig(
        accelerator="tpu7x",
        gke_accelerator_label="tpu7x",
        cores_per_chip=2,
        default_chips_per_node=4,
        topology_dims=3,
        known_topologies=(
            "1x1x1",
            "2x2x1",
            "2x2x2",
            "2x2x4",
            "2x4x4",
            "4x4x4",
            "4x4x8",
        ),
        suite_filename="canonical_suite_v7x.yaml",
    ),
    "v6e": AcceleratorDefaultConfig(
        accelerator="v6e",
        gke_accelerator_label="tpu-v6e-slice",
        cores_per_chip=1,
        default_chips_per_node=4,
        topology_dims=2,
        known_topologies=(
            "1x1",
            "2x2",
            "2x4",
            "4x4",
            "4x8",
            "8x8",
            "8x16",
            "16x16",
        ),
        suite_filename="canonical_suite_v6e.yaml",
    ),
}

SUPPORTED_ACCELERATORS = tuple(ACCELERATOR_CONFIGS)
# Topologies with checked-in configs; others are allowed but logged.
KNOWN_TOPOLOGIES = frozenset().union(
    *(cfg.known_topologies for cfg in ACCELERATOR_CONFIGS.values())
)

# Matches 3D (tpu7x, e.g. '2x2x1') or 2D (v6e, e.g. '4x4') topologies bounded
# by non-alphanumeric delimiters so 3D strings never truncate to 2D prefixes
# and 4D/alphanumeric tokens do not partially match. Zero-valued dimensions
# (e.g. '0x4') never match.
_TOPOLOGY_RE = re.compile(
    r"(?<![a-zA-Z\d])([1-9]\d*x[1-9]\d*(?:x[1-9]\d*)?)(?![a-zA-Z\d])"
)


def split_csv_tokens(items: Optional[Iterable[str]]) -> list[str]:
  """Flattens CLI values that may be comma-separated into stripped tokens.

  Args:
    items: Values such as ['2x2,4x4', ' 8x8 '] (or None).

  Returns:
    Non-empty stripped tokens, e.g. ['2x2', '4x4', '8x8'].
  """
  return [
      tok.strip() for tok in ",".join(items or ()).split(",") if tok.strip()
  ]


def _parse_topology_dims(topology: Optional[str]) -> Optional[list[int]]:
  """Returns the positive integer dimensions of an 'AxB[xC]' topology or None."""
  parts = (topology or "").lower().strip().split("x")
  if len(parts) < 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
    return None
  return [int(p) for p in parts]


def format_gcs_uri(bucket: Optional[str], *subpaths: Optional[str]) -> str:
  """Formats a GCS bucket and optional subpaths into a canonical gs:// URI.

  Args:
    bucket: Bucket name or gs:// prefix; blank means "no bucket".
    *subpaths: Path segments to append; empty or None segments are skipped.

  Returns:
    The joined gs:// URI, or "" if `bucket` is blank.
  """
  clean = (bucket or "").rstrip("/")
  if clean in ("", "gs:"):
    return ""
  base = clean if clean.startswith("gs://") else f"gs://{clean}"
  cleaned_subpaths = [str(s).strip("/") for s in subpaths if s]
  return "/".join([base, *filter(None, cleaned_subpaths)])


def extract_topology(*candidates: Any) -> str:
  """Extracts the rightmost TPU topology from the first matching candidate.

  Scans `candidates` in priority order and returns the rightmost topology match
  within the first candidate that contains at least one match so that parent
  directories or bucket prefixes containing topology-like tokens (e.g.
  `gs://my-2x2-bucket/.../tpums-4x4-all-reduce`) never shadow the workload's
  actual topology segment.

  Args:
    *candidates: Strings or path-like objects to scan in priority order.

  Returns:
    The rightmost topology found in the first matching candidate (e.g.
    '2x2x1' for tpu7x, '4x4' for v6e).

  Raises:
    ValueError: If no candidate contains a valid 2D or 3D TPU topology.
  """
  for candidate in candidates:
    if candidate is None:
      continue
    matches = _TOPOLOGY_RE.findall(str(candidate).lower())
    if matches:
      return matches[-1]
  target = " or ".join(repr(c) for c in candidates) or "<no candidates>"
  raise ValueError(f"Could not infer TPU topology from {target}")


def normalize_accelerator(accelerator: str) -> str:
  """Validates and canonicalizes an explicit accelerator name.

  Args:
    accelerator: Accelerator string such as 'tpu7x', ' V6E '.

  Returns:
    Canonical accelerator key in `ACCELERATOR_CONFIGS`.

  Raises:
    ValueError: If `accelerator` is not a supported accelerator.
  """
  key = (accelerator or "").lower().strip()
  if key not in ACCELERATOR_CONFIGS:
    raise ValueError(
        f"Unsupported accelerator {accelerator!r}. Supported:"
        f" {list(SUPPORTED_ACCELERATORS)}."
    )
  return key


def infer_accelerator(
    accelerator: Optional[str] = None,
    config_path: Optional[str] = None,
    topology: Optional[str] = None,
) -> str:
  """Resolves the accelerator from an explicit value, config path and topology.

  Each provided input is resolved independently: explicit `accelerator`,
  `config_path` directory segment (e.g. 'configs/v6e/...', scanning rightmost
  segment first), and `topology` dimensionality (2D -> 'v6e', 3D -> 'tpu7x').
  All resolved values must agree.

  Args:
    accelerator: Explicit accelerator string ('tpu7x' or 'v6e').
    config_path: Path to benchmark YAML config.
    topology: Topology string such as '2x2' or '2x2x1'.

  Returns:
    Canonical accelerator name ('tpu7x' or 'v6e').

  Raises:
    ValueError: If `accelerator` is unsupported; if the inputs imply different
      accelerators; or if none of them identifies an accelerator.
  """
  explicit_acc = normalize_accelerator(accelerator) if accelerator else None

  path_acc = None
  if config_path:
    parts = pathlib.PurePath(config_path.lower().replace("\\", "/")).parts
    path_acc = next(
        (p for p in reversed(parts) if p in ACCELERATOR_CONFIGS), None
    )

  topo_acc = None
  dims = _parse_topology_dims(topology)
  if dims:
    topo_acc = next(
        (
            a
            for a, cfg in ACCELERATOR_CONFIGS.items()
            if len(dims) == cfg.topology_dims
        ),
        None,
    )

  resolved = {a for a in (explicit_acc, path_acc, topo_acc) if a}
  if len(resolved) > 1:
    raise ValueError(
        f"Conflicting TPU accelerators: accelerator={accelerator!r}"
        f" ({explicit_acc}), config_path={config_path!r} ({path_acc}),"
        f" topology={topology!r} ({topo_acc})."
    )
  if resolved:
    return resolved.pop()

  raise ValueError(
      "Cannot infer TPU accelerator from config path"
      f" {config_path!r} or topology {topology!r}; pass --accelerator"
      f" explicitly (one of {list(SUPPORTED_ACCELERATORS)})."
  )


def compute_topology_info(
    topology: str, accelerator: Optional[str] = None
) -> dict[str, Any]:
  """Computes topology node counts, chips, and placement policy dynamically."""
  dims = _parse_topology_dims(topology)
  if not dims:
    raise ValueError(
        f"Invalid topology {topology!r}: expected positive integer dimensions"
        " like '2x2x1' (tpu7x) or '4x4' (v6e)."
    )
  clean_topology = topology.lower().strip()

  acc = infer_accelerator(accelerator, topology=clean_topology)
  acc_cfg = ACCELERATOR_CONFIGS[acc]
  if len(dims) != acc_cfg.topology_dims:
    raise ValueError(
        f"Topology '{topology}' has {len(dims)} dimensions but {acc} expects"
        f" {acc_cfg.topology_dims}D topologies (e.g."
        f" '{acc_cfg.known_topologies[0]}')."
    )
  total_chips = math.prod(dims)
  # A host never exposes more chips than the slice has (e.g. 1x1).
  chips_per_node = min(acc_cfg.default_chips_per_node, total_chips)
  if total_chips % chips_per_node != 0:
    raise ValueError(
        f"Total chips ({total_chips}) for topology '{topology}' is not"
        f" divisible by chips_per_node ({chips_per_node})."
    )
  num_nodes = total_chips // chips_per_node
  total_cores = total_chips * acc_cfg.cores_per_chip
  return {
      "accelerator": acc,
      "gke_accelerator": acc_cfg.gke_accelerator_label,
      "num_nodes": num_nodes,
      "chips_per_node": chips_per_node,
      "total_chips": total_chips,
      "total_cores": total_cores,
      "placement_policy": (
          f"{acc}-{total_cores}-{clean_topology}-placement-policy"
          if num_nodes > 1
          else None
      ),
  }


def normalize_topologies(
    accelerator: str, topologies: Optional[Iterable[str]]
) -> list[str]:
  """Validates a --topologies filter against the --accelerator.

  Args:
    accelerator: Accelerator selected with --accelerator.
    topologies: Topology values, possibly comma-separated (e.g. ['2x2,4x4']).

  Returns:
    The lower-cased topology tokens, e.g. ['2x2', '4x4'].

  Raises:
    ValueError: If `accelerator` is unsupported, or if a topology is malformed
      or has the wrong dimensionality for `accelerator`.
  """
  acc_cfg = get_accelerator_config(accelerator)
  tokens = [t.lower() for t in split_csv_tokens(topologies)]
  mismatched = [
      t
      for t in tokens
      if len(_parse_topology_dims(t) or ()) != acc_cfg.topology_dims
  ]
  if mismatched:
    raise ValueError(
        f"Topologies {mismatched} do not match accelerator"
        f" {acc_cfg.accelerator!r} ({acc_cfg.topology_dims}D topologies"
        f" expected, e.g. '{acc_cfg.known_topologies[0]}')."
    )
  return tokens


def get_accelerator_config(accelerator: str) -> AcceleratorDefaultConfig:
  """Returns the registered config for `accelerator`.

  Args:
    accelerator: Accelerator name (case/whitespace-insensitive).

  Returns:
    The accelerator's entry in `ACCELERATOR_CONFIGS`.

  Raises:
    ValueError: If `accelerator` is empty or not in `ACCELERATOR_CONFIGS`.
  """
  return ACCELERATOR_CONFIGS[normalize_accelerator(accelerator)]


def add_selection_args(parser: argparse.ArgumentParser) -> None:
  """Registers the workload selection flags.

  Exactly one selection mode is required (enforced by argparse here and by
  `workload_submitter.resolve_config_files` for programmatic callers):

    1. --suite-file: run the configs listed in a suite YAML.
    2. --configs: run explicit config files.
    3. --accelerator: scan --config-dir, optionally narrowed by --topologies
       and/or --categories (AND semantics). These three flags are only valid
       in this mode.

  Args:
    parser: Parser to register the flags on.
  """
  group = parser.add_mutually_exclusive_group(required=True)
  canonical_suites = ", ".join(
      f"automation/{cfg.suite_filename} for {acc}"
      for acc, cfg in ACCELERATOR_CONFIGS.items()
  )
  group.add_argument(
      "--suite-file",
      default=None,
      help=(
          "Mode 1: path to a suite YAML file listing the benchmark configs to"
          " run. All configs must target one accelerator. Canonical nightly"
          f" suites: {canonical_suites}."
      ),
  )
  group.add_argument(
      "--configs",
      nargs="+",
      default=None,
      help=(
          "Mode 2: explicit YAML config files to run (package-relative or"
          " absolute; directories are rejected). All configs must target one"
          " accelerator."
      ),
  )
  group.add_argument(
      "--accelerator",
      choices=SUPPORTED_ACCELERATORS,
      default=None,
      help=(
          "Mode 3: run every topology config under --config-dir for this TPU"
          " accelerator, optionally narrowed by --topologies and/or"
          " --categories."
      ),
  )
  config_dir_defaults = ", ".join(
      f"configs/{acc} for {acc}" for acc in ACCELERATOR_CONFIGS
  )
  parser.add_argument(
      "--config-dir",
      dest="config_dir",
      default=None,
      help=(
          "--accelerator mode only: directory scanned for topology configs;"
          " every selected config must match --accelerator (default:"
          f" {config_dir_defaults})."
      ),
  )
  parser.add_argument(
      "--topologies",
      nargs="+",
      default=None,
      help=(
          "--accelerator mode only: keep configs whose topology is listed"
          " (e.g. '--topologies 2x2x1 4x4x4' for tpu7x or '--topologies 2x2"
          " 4x4' for v6e)."
      ),
  )
  parser.add_argument(
      "--categories",
      nargs="+",
      default=None,
      help=(
          "--accelerator mode only: keep configs whose file name contains any"
          " of these benchmark names (e.g. '--categories all_gather gemm')."
      ),
  )


def add_workload_args(
    parser: argparse.ArgumentParser, default_cleanup: bool
) -> None:
  """Registers JobSet submission flags shared by the submitter and runner CLIs.

  Args:
    parser: Parser to register the flags on.
    default_cleanup: Default for --cleanup / --no-cleanup.
  """
  for flag, default, help_text in (
      ("--namespace", DEFAULT_NAMESPACE, "Target namespace"),
      ("--queue", DEFAULT_QUEUE, "Target Kueue LocalQueue"),
      ("--priority", DEFAULT_PRIORITY, "Priority class"),
      ("--docker-image", DEFAULT_DOCKER_IMAGE, "Base TPU container image"),
      ("--gcs-bucket", DEFAULT_GCS_BUCKET, "GCS output bucket"),
      ("--git-branch", DEFAULT_GIT_BRANCH, "Git branch cloned in the JobSet"),
      (
          "--reservation-name",
          DEFAULT_RESERVATION_NAME,
          "Optional GCE reservation name added to the JobSet nodeSelector",
      ),
  ):
    parser.add_argument(
        flag, default=default, help=f"{help_text} (default: {default!r})."
    )
  parser.add_argument(
      "--timeout-minutes",
      type=int,
      default=DEFAULT_TIMEOUT_MINUTES,
      help="Timeout in minutes per workload (default: infinite).",
  )
  parser.add_argument(
      "--declared-duration-minutes",
      type=int,
      default=DEFAULT_DECLARED_DURATION_MINUTES,
      help=(
          "Declared workload duration in minutes (default:"
          f" {DEFAULT_DECLARED_DURATION_MINUTES})."
      ),
  )
  parser.add_argument(
      "--poll-interval",
      type=int,
      default=DEFAULT_POLL_INTERVAL_SECONDS,
      help=(
          "Status dashboard polling interval in seconds (default:"
          f" {DEFAULT_POLL_INTERVAL_SECONDS})."
      ),
  )
  parser.add_argument(
      "--cleanup",
      action=argparse.BooleanOptionalAction,
      default=default_cleanup,
      help=(
          "Automatically delete JobSets on the cluster after completion"
          f" (default: {default_cleanup})."
      ),
  )
  parser.add_argument(
      "--upload-trace",
      action="store_true",
      help="Upload XLA/TensorBoard trace directories to GCS.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help=(
          "Print JobSet manifests and reports without submitting or uploading."
      ),
  )


def validate_workload_args(
    args: argparse.Namespace, require_cluster: bool = False
) -> None:
  """Raises ValueError if flags required for a real (non dry-run) run are unset.

  Args:
    args: Parsed flags from `add_workload_args` (plus --cluster/--project when
      `require_cluster`).
    require_cluster: Also require --cluster and --project.
  """
  if args.dry_run:
    return
  required = [("gcs_bucket", "GCS_BUCKET"), ("docker_image", "DOCKER_IMAGE")]
  if require_cluster:
    required += [("cluster", "CLUSTER"), ("project", "PROJECT")]
  for attr, env in required:
    if not getattr(args, attr):
      raise ValueError(
          f"Must specify --{attr.replace('_', '-')} or set {env} environment"
          " variable."
      )
