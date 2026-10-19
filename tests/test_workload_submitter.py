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

"""Unit tests for TPUMS workload_submitter module."""

import json
import os
import pathlib
import subprocess
import tempfile
import time
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from automation import utils
from automation import workload_submitter
import yaml

_CONFIG_DIR = str(pathlib.Path(__file__).parent.parent / "configs" / "tpu7x")


def _render_manifest(topology, config_rel_path, **kwargs):
  """Renders a JobSet manifest; returns (yaml_text, job_spec, pod_spec)."""
  manifest_yaml = workload_submitter.generate_jobset_manifest(
      workload_name=kwargs.pop("workload_name", "tpums-test"),
      namespace="default",
      queue_name="multislice-queue",
      priority_class="medium",
      topology=topology,
      docker_image="gcr.io/test/image:latest",
      config_rel_path=config_rel_path,
      gcs_output_dir="gs://test-bucket/daily/test",
      **kwargs,
  )
  job_spec = yaml.safe_load(manifest_yaml)["spec"]["replicatedJobs"][0][
      "template"
  ]["spec"]
  return manifest_yaml, job_spec, job_spec["template"]["spec"]


def _workload(**overrides):
  """Builds a tpu7x 2x2x1 ActiveWorkload with optional field overrides."""
  fields = dict(
      workload_name="tpums-test-workload",
      config_name="gemm",
      config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
      topology="2x2x1",
      manifest_yaml="",
      gcs_output_dir="gs://test-bucket/daily/test",
      start_time=time.time(),
      total_pods=1,
  )
  fields.update(overrides)
  return workload_submitter.ActiveWorkload(**fields)


class WorkloadSubmitterTest(parameterized.TestCase):
  """Tests for workload_submitter config resolution and lifecycle."""

  @parameterized.named_parameters(
      ("tpu7x", "configs/tpu7x/2x2x1/gemm.yaml", "2x2x1"),
      ("tpu7x_2x2x16_not_truncated", "configs/tpu7x/2x2x16/a.yaml", "2x2x16"),
      ("parent_dir_nxm_ignored", "/tmp/test_1x2/configs/v6e/4x4/g.yaml", "4x4"),
      (
          "parent_dir_3d_ignored",
          "/tmp/my-2x2-scratch/configs/tpu7x/4x4x4/all_reduce.yaml",
          "4x4x4",
      ),
      ("zero_dim_segment_ignored", "configs/v6e/2x2/run_0x4/gemm.yaml", "2x2"),
  )
  def test_infer_topology_from_config(self, config_path, expected):
    """The topology directory closest to the file wins."""
    self.assertEqual(
        workload_submitter.infer_topology_from_config(config_path), expected
    )

  def test_infer_topology_from_config_missing_raises(self):
    # No topology in the path: raise rather than provision a guessed topology.
    for bad_path in (
        "unnamed.yaml",
        "configs/v6e/unnamed.yaml",
        "configs/v6e/0x4.yaml",
    ):
      with self.subTest(bad_path=bad_path):
        with self.assertRaisesRegex(ValueError, "Could not infer TPU topology"):
          workload_submitter.infer_topology_from_config(bad_path)

  def test_sanitize_workload_name(self):
    name = workload_submitter.sanitize_workload_name("gemm", topology="2x2x1")
    self.assertEqual(name, "tpums-2x2x1-gemm")
    self.assertLessEqual(len(name), 45)
    self.assertRegex(name, r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

    # If topology already present in name, avoid duplicate
    name2 = workload_submitter.sanitize_workload_name(
        "2x2x2_all_gather", topology="2x2x2"
    )
    self.assertEqual(name2, "tpums-2x2x2-all-gather")

  def test_generate_jobset_manifest_single_node(self):
    """Single-host manifest metadata, sizing and command."""
    manifest_yaml, job_spec, pod_spec = _render_manifest(
        "2x2x1", "configs/tpu7x/2x2x1/gemm.yaml", workload_name="tpums-test-01"
    )
    doc = yaml.safe_load(manifest_yaml)
    self.assertEqual(doc["kind"], "JobSet")
    self.assertEqual(doc["metadata"]["generateName"], "tpums-test-01-")
    self.assertNotIn("name", doc["metadata"])
    self.assertEqual(
        doc["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
        "multislice-queue",
    )
    self.assertEqual(job_spec["parallelism"], 1)
    self.assertEqual(job_spec["completions"], 1)
    self.assertEqual(job_spec["completionMode"], "Indexed")
    self.assertEqual(
        pod_spec["containers"][0]["resources"]["limits"]["google.com/tpu"], 4
    )
    self.assertIn(
        'tpums benchmark run-config "configs/tpu7x/2x2x1/gemm.yaml"',
        pod_spec["containers"][0]["args"][0],
    )
    self.assertEqual(
        pod_spec["containers"][0]["image"], "gcr.io/test/image:latest"
    )
    # Only worker 0 uploads shared /tmp/results/* (and a failed summary.csv
    # upload must fail the pod); every worker uploads execution_<index>.log.
    script = pod_spec["containers"][0]["args"][0]
    self.assertIn('if [ "${JOB_COMPLETION_INDEX:-0}" = "0" ]; then', script)
    self.assertIn(
        'gcloud storage cp /tmp/results/* "gs://test-bucket/daily/test/"\n',
        script,
    )
    self.assertNotIn(
        '/tmp/results/* "gs://test-bucket/daily/test/" || true', script
    )
    self.assertIn(
        '"gs://test-bucket/daily/test/execution_${JOB_COMPLETION_INDEX:-0}.log"',
        script,
    )
    self.assertNotIn('"gs://test-bucket/daily/test/execution.log"', script)

  @parameterized.named_parameters(
      ("tpu7x_single_host", "2x2x1", "tpu7x", 1, "tpu7x", None),
      (
          "tpu7x_multi_host",
          "2x2x2",
          "tpu7x",
          2,
          "tpu7x",
          "tpu7x-16-2x2x2-placement-policy",
      ),
      ("v6e_single_host", "2x2", "v6e", 1, "tpu-v6e-slice", None),
      (
          "v6e_multi_host",
          "4x4",
          "v6e",
          4,
          "tpu-v6e-slice",
          "v6e-16-4x4-placement-policy",
      ),
  )
  def test_generate_jobset_manifest_node_selector(
      self, topology, acc, num_nodes, gke_label, placement_policy
  ):
    """Node selector and host counts per accelerator/topology."""
    _, job_spec, pod_spec = _render_manifest(
        topology, f"configs/{acc}/{topology}/all_reduce.yaml"
    )
    self.assertEqual(job_spec["parallelism"], num_nodes)
    self.assertEqual(job_spec["completions"], num_nodes)
    selector = pod_spec["nodeSelector"]
    self.assertEqual(
        selector["cloud.google.com/gke-tpu-accelerator"], gke_label
    )
    self.assertEqual(selector["cloud.google.com/gke-tpu-topology"], topology)
    self.assertEqual(
        selector.get("cloud.google.com/placement-policy-name"),
        placement_policy,
    )

  def test_generate_jobset_manifest_reservation(self):
    """Reservation selector is dropped or filled in."""
    _, _, pod_spec = _render_manifest(
        "2x2x1", "configs/tpu7x/2x2x1/gemm.yaml", reservation_name=None
    )
    self.assertNotIn(
        "cloud.google.com/reservation-name", pod_spec["nodeSelector"]
    )
    toleration_keys = [t.get("key") for t in pod_spec.get("tolerations", [])]
    self.assertIn("cloud.google.com/reservation-name", toleration_keys)

    _, _, pod_spec = _render_manifest(
        "4x4x4",
        "configs/tpu7x/4x4x4/all_gather.yaml",
        reservation_name="example-tpu-reservation",
    )
    self.assertEqual(
        pod_spec["nodeSelector"]["cloud.google.com/reservation-name"],
        "example-tpu-reservation",
    )

  @parameterized.named_parameters(
      ("default", {}, "main"),
      ("custom", {"git_branch": "feat/custom-mesh-v2"}, "feat/custom-mesh-v2"),
  )
  def test_generate_jobset_manifest_git_branch(self, kwargs, branch):
    """GIT_BRANCH env var and clone command use the branch."""
    manifest_yaml, _, pod_spec = _render_manifest(
        "2x2x1", "configs/tpu7x/2x2x1/gemm.yaml", **kwargs
    )
    env_vars = {
        item["name"]: item["value"] for item in pod_spec["containers"][0]["env"]
    }
    self.assertEqual(env_vars["GIT_BRANCH"], branch)
    self.assertNotIn("CONFIG_B64", manifest_yaml)
    self.assertIn(
        f'git clone --branch "{branch}" --single-branch', manifest_yaml
    )

  def test_generate_jobset_manifest_missing_template_raises(self):
    with self.assertRaises(FileNotFoundError):
      _render_manifest(
          "2x2x1",
          "configs/tpu7x/2x2x1/gemm.yaml",
          template_path="/non/existent/jobset_template.yaml",
      )

  @parameterized.named_parameters(
      ("none", {}),
      ("suite_and_configs", {"suite_file": "s.yaml", "configs": ["a.yaml"]}),
      ("suite_and_accelerator", {"suite_file": "s.yaml", "accelerator": "v6e"}),
      (
          "configs_and_accelerator",
          {"configs": ["a.yaml"], "accelerator": "v6e"},
      ),
      ("scan_filter_only", {"topologies": ["2x2"]}),
      ("config_dir_only", {"config_dir": _CONFIG_DIR}),
  )
  def test_resolve_config_files_requires_exactly_one_mode(self, kwargs):
    """No mode, or several modes at once, is rejected."""
    with self.assertRaisesRegex(ValueError, "exactly one of --suite-file"):
      workload_submitter.resolve_config_files(**kwargs)

  @parameterized.named_parameters(
      ("suite_config_dir", "suite_file", {"config_dir": _CONFIG_DIR}),
      ("suite_topologies", "suite_file", {"topologies": ["2x2"]}),
      ("suite_categories", "suite_file", {"categories": ["gemm"]}),
      ("configs_config_dir", "configs", {"config_dir": _CONFIG_DIR}),
      ("configs_topologies", "configs", {"topologies": ["2x2"]}),
      ("configs_categories", "configs", {"categories": ["gemm"]}),
  )
  def test_resolve_config_files_scan_flags_require_accelerator(
      self, mode, kwargs
  ):
    """--config-dir/--topologies/--categories are --accelerator-only."""
    mode_kwargs = {
        "suite_file": {"suite_file": "automation/canonical_suite_v6e.yaml"},
        "configs": {"configs": ["configs/v6e/2x2/gemm.yaml"]},
    }[mode]
    with self.assertRaisesRegex(
        ValueError, "can only be combined with --accelerator"
    ):
      workload_submitter.resolve_config_files(**mode_kwargs, **kwargs)

  # --- Mode 1: --suite-file -------------------------------------------------

  @parameterized.named_parameters(
      ("tpu7x", "tpu7x", "canonical_suite_v7x.yaml"),
      ("v6e", "v6e", "canonical_suite_v6e.yaml"),
  )
  def test_resolve_config_files_canonical_suites(self, acc, suite):
    """Each canonical suite resolves to 11 configs of one accelerator."""
    resolved = workload_submitter.resolve_config_files(
        suite_file=f"automation/{suite}"
    )
    self.assertLen(resolved, 11)
    for p in resolved:
      self.assertIn(f"/configs/{acc}/", str(p))
    self.assertEqual(
        workload_submitter.infer_configs_accelerator(resolved), acc
    )

  def test_resolve_config_files_v7x_suite_contents(self):
    """The tpu7x suite runs 4x4x4 collectives and 2x2x1 microbenchmarks."""
    resolved = workload_submitter.resolve_config_files(
        suite_file=utils.ACCELERATOR_CONFIGS["tpu7x"].suite_path
    )
    self.assertCountEqual(
        [f"{p.parent.name}/{p.name}" for p in resolved],
        [
            "2x2x1/all_gather.yaml",
            "2x2x1/all_reduce.yaml",
            "2x2x1/all_to_all.yaml",
            "2x2x1/device_to_device.yaml",
            "2x2x1/device_to_host.yaml",
            "2x2x1/gemm.yaml",
            "2x2x1/hbm.yaml",
            "2x2x1/host_to_device.yaml",
            "4x4x4/all_gather.yaml",
            "4x4x4/all_reduce.yaml",
            "4x4x4/all_to_all.yaml",
        ],
    )

  def _suite(self, entries):
    """Writes a suite YAML with `entries` and returns its path."""
    return self.create_tempfile(
        "suite.yaml", content=yaml.dump({"configs": entries})
    ).full_path

  def test_resolve_config_files_custom_suite_file_dedupes(self):
    """A custom suite selects exactly its configs, de-duplicated in order."""
    resolved = workload_submitter.resolve_config_files(
        suite_file=self._suite([
            "configs/tpu7x/4x4x4/all_gather.yaml",
            "configs/tpu7x/2x2x1/gemm.yaml",
            "configs/tpu7x/4x4x4/all_gather.yaml",
        ])
    )
    self.assertEqual(
        [f"{p.parent.name}/{p.name}" for p in resolved],
        ["4x4x4/all_gather.yaml", "2x2x1/gemm.yaml"],
    )

  def test_resolve_config_files_suite_mixed_accelerators_raises(self):
    """A suite must not span accelerators."""
    with self.assertRaisesRegex(ValueError, "span multiple accelerators"):
      workload_submitter.resolve_config_files(
          suite_file=self._suite([
              "configs/tpu7x/2x2x1/gemm.yaml",
              "configs/v6e/2x2/gemm.yaml",
          ])
      )

  def test_resolve_config_files_suite_entry_without_topology_raises(self):
    """Every suite entry must encode its topology."""
    with self.assertRaisesRegex(ValueError, "Could not infer TPU topology"):
      workload_submitter.resolve_config_files(
          suite_file=self._suite(["configs/tpu7x/attention_jax_flash.yaml"])
      )

  def test_load_canonical_configs_requires_configs_mapping(self):
    """Suites must be a mapping with a configs list; bare lists are rejected."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
      yaml.dump(["configs/test/a.yaml", "configs/test/b.yaml"], f)
      list_file = f.name
    try:
      with self.assertRaisesRegex(ValueError, "configs"):
        workload_submitter.load_canonical_configs(list_file)
    finally:
      os.remove(list_file)

    # Test dict format with 'configs'
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
      yaml.dump({"configs": ["configs/test/c.yaml"]}, f)
      dict_file = f.name
    try:
      configs = workload_submitter.load_canonical_configs(dict_file)
      self.assertEqual(configs, ["configs/test/c.yaml"])
    finally:
      os.remove(dict_file)

  def test_load_canonical_configs_missing_file(self):
    with self.assertRaises(FileNotFoundError):
      workload_submitter.load_canonical_configs("/non/existent/suite.yaml")

  def test_resolve_config_files_missing_suite_file_raises(self):
    """A missing --suite-file is an error, not a silent fallback."""
    with self.assertRaisesRegex(FileNotFoundError, "Suite YAML file not found"):
      workload_submitter.resolve_config_files(suite_file="does/not/exist.yaml")

  def test_resolve_config_files_suite_with_missing_config_raises(self):
    """Suite entries that do not exist fail loudly instead of being dropped."""
    with self.assertRaisesRegex(FileNotFoundError, "typo_gemm"):
      workload_submitter.resolve_config_files(
          suite_file=self._suite(["configs/tpu7x/2x2x1/typo_gemm.yaml"])
      )

  # --- Mode 2: --configs ----------------------------------------------------

  def test_resolve_config_files_configs_dedupes_in_order(self):
    """Explicit configs are de-duplicated (including via '..') in order."""
    resolved = workload_submitter.resolve_config_files(
        configs=[
            "configs/v6e/4x4/all_gather.yaml",
            "configs/v6e/2x2/gemm.yaml",
            "configs/v6e/2x2/../2x2/gemm.yaml",
            str(utils.PACKAGE_ROOT / "configs/v6e/4x4/all_gather.yaml"),
        ]
    )
    self.assertEqual(
        [str(p.relative_to(utils.PACKAGE_ROOT)) for p in resolved],
        ["configs/v6e/4x4/all_gather.yaml", "configs/v6e/2x2/gemm.yaml"],
    )

  def test_resolve_config_files_configs_directory_raises(self):
    """A directory passed to --configs is rejected, not run as a config."""
    with self.assertRaisesRegex(ValueError, "is a directory"):
      workload_submitter.resolve_config_files(configs=["configs/v6e/4x4"])

  def test_resolve_config_files_configs_mixed_accelerators_raises(self):
    """Explicit configs must all target one accelerator."""
    with self.assertRaisesRegex(ValueError, "span multiple accelerators"):
      workload_submitter.resolve_config_files(
          configs=[
              "configs/v6e/4x4/all_gather.yaml",
              "configs/tpu7x/2x2x1/gemm.yaml",
          ]
      )

  def test_resolve_config_files_configs_without_topology_raises(self):
    """A config whose topology cannot be inferred is rejected up front."""
    with self.assertRaisesRegex(ValueError, "Could not infer TPU topology"):
      workload_submitter.resolve_config_files(
          configs=["configs/tpu7x/attention_jax_flash.yaml"]
      )

  def test_resolve_config_files_configs_missing_raises(self):
    with self.assertRaisesRegex(FileNotFoundError, "Config file not found"):
      workload_submitter.resolve_config_files(
          configs=["configs/v6e/2x2/typo.yaml"]
      )

  def test_resolve_config_files_configs_prefers_package_root_over_cwd(self):
    """Package-relative paths resolve inside the package, not under cwd."""
    shadow = pathlib.Path(
        self.create_tempfile("configs/v6e/2x2/gemm.yaml").full_path
    )
    orig_cwd = os.getcwd()
    os.chdir(shadow.parents[3])
    try:
      resolved = workload_submitter.resolve_config_files(
          configs=["configs/v6e/2x2/gemm.yaml"]
      )
    finally:
      os.chdir(orig_cwd)
    self.assertEqual(
        resolved, [utils.PACKAGE_ROOT / "configs/v6e/2x2/gemm.yaml"]
    )

  # --- Mode 3: --accelerator ------------------------------------------------

  @parameterized.named_parameters(("tpu7x", "tpu7x"), ("v6e", "v6e"))
  def test_resolve_config_files_accelerator_scans_all_topology_configs(
      self, acc
  ):
    """--accelerator alone selects every topology config of that accelerator."""
    with self.assertLogs(level="INFO") as logs:
      resolved = workload_submitter.resolve_config_files(accelerator=acc)
    self.assertGreater(len(resolved), 11)
    for p in resolved:
      rel = p.relative_to(utils.PACKAGE_ROOT)
      self.assertEqual(rel.parts[:2], ("configs", acc))
      self.assertLen(rel.parts, 4)  # configs/<acc>/<topology>/<name>.yaml
    self.assertIn(
        utils.ACCELERATOR_CONFIGS[acc].config_dir, "\n".join(logs.output)
    )

  def test_resolve_config_files_accelerator_skips_configs_without_topology(
      self,
  ):
    """Topology-less configs under --config-dir are skipped with a warning."""
    with self.assertLogs(level="WARNING") as logs:
      resolved = workload_submitter.resolve_config_files(accelerator="tpu7x")
    self.assertNotIn("attention_jax_flash", " ".join(map(str, resolved)))
    self.assertIn("attention_jax_flash.yaml", "\n".join(logs.output))

  def test_resolve_config_files_categories_multi_pattern(self):
    resolved = workload_submitter.resolve_config_files(
        accelerator="tpu7x", categories=["all_gather", "gemm"]
    )
    self.assertNotEmpty(resolved)
    self.assertTrue(
        all(p.stem in ("all_gather", "gemm") for p in resolved), resolved
    )

  def test_resolve_config_files_v6e_categories(self):
    resolved = workload_submitter.resolve_config_files(
        accelerator="v6e", categories=["all_reduce"]
    )
    self.assertCountEqual(
        [f"{p.parent.name}/{p.name}" for p in resolved],
        [
            f"{t}/all_reduce.yaml"
            for t in ("2x2", "2x4", "4x4", "4x8", "8x8", "8x16", "16x16")
        ],
    )

  @parameterized.named_parameters(
      ("topology_token", ["2x2"]),
      ("accelerator_token", ["tpu7x"]),
      ("empty", [""]),
  )
  def test_resolve_config_files_categories_match_file_names_only(
      self, categories
  ):
    """--categories matches benchmark file names, not directories."""
    if categories == [""]:
      with self.assertRaisesRegex(ValueError, "no non-empty names"):
        workload_submitter.resolve_config_files(
            accelerator="tpu7x", categories=categories
        )
      return
    self.assertEmpty(
        workload_submitter.resolve_config_files(
            accelerator="tpu7x", categories=categories
        )
    )

  def test_resolve_config_files_topologies(self):
    resolved = workload_submitter.resolve_config_files(
        accelerator="tpu7x", topologies=["2x2x1"]
    )
    self.assertNotEmpty(resolved)
    for p in resolved:
      self.assertEqual(p.parent.name, "2x2x1")

  def test_resolve_config_files_topologies_and_categories_are_anded(self):
    resolved = workload_submitter.resolve_config_files(
        accelerator="v6e", topologies=["2x2,4x4"], categories=["all_gather"]
    )
    self.assertEqual(
        [f"{p.parent.name}/{p.name}" for p in resolved],
        ["2x2/all_gather.yaml", "4x4/all_gather.yaml"],
    )

  @parameterized.named_parameters(
      ("v6e_3d", "v6e", ["2x2x1"]),
      ("tpu7x_2d", "tpu7x", ["4x4"]),
      ("tpu7x_mixed", "tpu7x", ["2x2x1", "4x4"]),
  )
  def test_resolve_config_files_accelerator_topology_mismatch_raises(
      self, acc, topologies
  ):
    """--topologies must match --accelerator's dimensionality up front."""
    with self.assertRaisesRegex(ValueError, "do not match accelerator"):
      workload_submitter.resolve_config_files(
          accelerator=acc, topologies=topologies
      )

  def test_resolve_config_files_config_dir_conflicts_with_accelerator(self):
    """A --config-dir holding another accelerator's configs is rejected."""
    with self.assertRaisesRegex(ValueError, "Conflicting TPU accelerators"):
      workload_submitter.resolve_config_files(
          accelerator="tpu7x", config_dir="configs/v6e"
      )

  def test_resolve_config_files_config_dir_topology_subdir(self):
    """Topologies come from each config's path, so topology dirs work."""
    resolved = workload_submitter.resolve_config_files(
        accelerator="v6e", config_dir="configs/v6e/4x4", topologies=["4x4"]
    )
    self.assertLen(resolved, 3)
    self.assertEmpty(
        workload_submitter.resolve_config_files(
            accelerator="v6e", config_dir="configs/v6e/4x4", topologies=["2x2"]
        )
    )

  def test_resolve_config_files_missing_config_dir_raises(self):
    with self.assertRaisesRegex(
        FileNotFoundError, "Config directory not found"
    ):
      workload_submitter.resolve_config_files(
          accelerator="tpu7x", config_dir="/non/existent/config_dir"
      )

  def test_resolve_config_files_custom_config_dir_outside_package(self):
    """A custom --config-dir is scanned when it matches --accelerator."""
    cfg = pathlib.Path(
        self.create_tempfile("configs/tpu7x/2x2x1/custom.yaml").full_path
    )
    resolved = workload_submitter.resolve_config_files(
        accelerator="tpu7x", config_dir=str(cfg.parents[1])
    )
    self.assertEqual(resolved, [cfg])

  @mock.patch.object(workload_submitter, "submit_workloads")
  @mock.patch.object(workload_submitter, "resolve_config_files")
  def test_main_forwards_accelerator(self, mock_resolve, mock_submit):
    """CLI --accelerator reaches both config resolution and submission."""
    mock_resolve.return_value = [
        utils.PACKAGE_ROOT / "configs" / "v6e" / "2x2" / "gemm.yaml"
    ]
    workload_submitter.main(
        ["workload_submitter", "--dry-run", "--accelerator", "v6e"]
    )
    self.assertEqual(mock_resolve.call_args.kwargs["accelerator"], "v6e")
    self.assertIsNone(mock_resolve.call_args.kwargs["config_dir"])
    self.assertIsNone(mock_resolve.call_args.kwargs["categories"])
    self.assertEqual(mock_submit.call_args.kwargs["accelerator"], "v6e")

    # Modes 1/2 forward the accelerator inferred from the configs.
    workload_submitter.main([
        "workload_submitter",
        "--dry-run",
        "--configs",
        "configs/v6e/2x2/gemm.yaml",
    ])
    self.assertIsNone(mock_resolve.call_args.kwargs["accelerator"])
    self.assertEqual(mock_submit.call_args.kwargs["accelerator"], "v6e")

    for bad_args in (
        ["--accelerator", "tpu-v6e"],
        [],  # A selection mode is required.
        ["--filter", "gemm"],  # Renamed to --categories.
    ):
      with self.subTest(bad_args=bad_args):
        with self.assertRaises(SystemExit):
          workload_submitter.main(
              ["workload_submitter", "--dry-run", *bad_args]
          )

  @mock.patch.object(workload_submitter, "submit_workloads")
  def test_main_rejects_mixed_modes(self, mock_submit):
    """Scan-only flags with --configs fail before anything is submitted."""
    with self.assertRaisesRegex(
        ValueError, "can only be combined with --accelerator"
    ):
      workload_submitter.main([
          "workload_submitter",
          "--dry-run",
          "--configs",
          "configs/v6e/2x2/gemm.yaml",
          "--config-dir",
          "configs/tpu7x",
      ])
    mock_submit.assert_not_called()

  @mock.patch.object(workload_submitter, "submit_workloads")
  @mock.patch.object(
      workload_submitter, "resolve_config_files", return_value=[]
  )
  def test_main_no_matching_configs_raises(self, mock_resolve, mock_submit):
    """An empty selection fails instead of exiting 0 with nothing run."""
    del mock_resolve  # Unused.
    with self.assertRaisesRegex(ValueError, "No YAML"):
      workload_submitter.main(
          ["workload_submitter", "--dry-run", "--accelerator", "v6e"]
      )
    mock_submit.assert_not_called()

  def test_update_workload_states_and_dashboard(self):
    """Running pods then a Completed JobSet drive the status."""
    wl = _workload(start_time=time.time() - 30)
    active = {"tpums-test-workload": wl}

    # Simulate running pod
    mock_pods = {
        "tpums-test-workload": [{
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
            }
        }]
    }
    workload_submitter.update_workload_states(
        active,
        jobsets_data={"tpums-test-workload": {"status": {}}},
        pods_data=mock_pods,
        timeout_minutes=45,
    )
    self.assertEqual(wl.status, "RUNNING")
    self.assertEqual(wl.ready_pods, 1)

    dashboard = workload_submitter.render_status_dashboard(
        active, time.time() - 30
    )
    self.assertIn("tpums-test-workload", dashboard)
    self.assertIn("RUNNING", dashboard)

    # Simulate completed JobSet
    mock_jobsets = {
        "tpums-test-workload": {
            "status": {"conditions": [{"type": "Completed", "status": "True"}]}
        }
    }
    all_done = workload_submitter.update_workload_states(
        active,
        jobsets_data=mock_jobsets,
        pods_data=mock_pods,
        timeout_minutes=45,
    )
    self.assertTrue(all_done)
    self.assertEqual(wl.status, "SUCCESS")
    self.assertTrue(wl.terminal)

  def test_update_workload_states_jobset_completed_with_no_pods(self):
    """A Completed JobSet succeeds even without pod data."""
    wl = _workload(workload_name="tpums-empty-pods-success")
    mock_jobsets = {
        "tpums-empty-pods-success": {
            "status": {"conditions": [{"type": "Completed", "status": "True"}]}
        }
    }
    all_done = workload_submitter.update_workload_states(
        {"tpums-empty-pods-success": wl},
        jobsets_data=mock_jobsets,
        pods_data={},
        timeout_minutes=45,
    )
    self.assertTrue(all_done)
    self.assertEqual(wl.status, "SUCCESS")
    self.assertEqual(wl.successful_pods, 1)
    self.assertTrue(wl.terminal)

  def test_update_workload_states_jobset_failed_with_no_pods(self):
    """A Failed JobSet fails even without pod data."""
    wl = _workload(workload_name="tpums-empty-pods-failed")
    mock_jobsets = {
        "tpums-empty-pods-failed": {
            "status": {
                "conditions": [{
                    "type": "Failed",
                    "status": "True",
                    "reason": "DeadlineExceeded",
                    "message": "Job timed out",
                }]
            }
        }
    }
    all_done = workload_submitter.update_workload_states(
        {"tpums-empty-pods-failed": wl},
        jobsets_data=mock_jobsets,
        pods_data={},
        timeout_minutes=45,
    )
    self.assertTrue(all_done)
    self.assertEqual(wl.status, "FAILED")
    self.assertEqual(wl.failed_pods, 1)
    self.assertTrue(wl.terminal)

  @mock.patch.object(workload_submitter, "delete_jobset", return_value=True)
  @mock.patch.object(
      workload_submitter, "fetch_namespace_pods", return_value={}
  )
  @mock.patch.object(
      workload_submitter, "fetch_namespace_jobsets", return_value={}
  )
  def test_workload_manager_poll_cleanup_flag(
      self, mock_jobsets, mock_pods, mock_delete
  ):
    """Polling returns results and cleans up terminal JobSets."""
    del mock_jobsets, mock_pods  # Unused.
    wl = _workload(
        workload_name="tpums-cleanup-test",
        status="SUCCESS",
        terminal=True,
    )
    active = {"tpums-cleanup-test": wl}
    # Test polling returns benchmark result and records cleaned_up state
    results = workload_submitter.WorkloadManager(
        timeout_minutes=1,
        poll_interval_seconds=1,
        cleanup=True,
        active_workloads=active,
    ).poll()
    self.assertLen(results, 1)
    self.assertEqual(results[0].status, "SUCCESS")
    self.assertTrue(wl.cleaned_up)
    mock_delete.assert_called_once_with(
        "tpums-cleanup-test", utils.DEFAULT_NAMESPACE
    )

  @parameterized.product(
      fetch=("fetch_namespace_jobsets", "fetch_namespace_pods"),
      case=(
          dict(
              run_kwargs=dict(
                  return_value=subprocess.CompletedProcess(
                      args=[], returncode=1, stdout="", stderr="Unauthorized"
                  )
              ),
              error=RuntimeError,
              regex="Unauthorized",
          ),
          dict(
              run_kwargs=dict(
                  return_value=subprocess.CompletedProcess(
                      args=[], returncode=0, stdout="not json", stderr=""
                  )
              ),
              error=RuntimeError,
              regex="invalid JSON",
          ),
          dict(
              run_kwargs=dict(side_effect=FileNotFoundError("kubectl")),
              error=FileNotFoundError,
              regex="kubectl",
          ),
          dict(
              run_kwargs=dict(
                  side_effect=subprocess.TimeoutExpired("kubectl", 300)
              ),
              error=RuntimeError,
              regex="timed out after 300",
          ),
      ),
  )
  def test_fetch_namespace_queries_raise(self, fetch, case):
    """kubectl failures propagate instead of returning empty state."""
    with mock.patch("subprocess.run", **case["run_kwargs"]):
      with self.assertRaisesRegex(case["error"], case["regex"]):
        getattr(workload_submitter, fetch)("default")

  def test_fetch_namespace_jobsets_and_pods_parse_items(self):
    """JobSets are keyed by name; Pods are grouped by their JobSet label."""
    jobset = {"metadata": {"name": "tpums-a"}, "status": {}}
    pod = {
        "metadata": {
            "name": "tpums-a-pod",
            "labels": {"jobset.sigs.k8s.io/jobset-name": "tpums-a"},
        }
    }
    unlabelled_pod = {"metadata": {"name": "other-pod"}}

    def _run(cmd, **kwargs):
      del kwargs  # Unused.
      items = [jobset] if cmd[2] == "jobsets" else [pod, unlabelled_pod]
      return subprocess.CompletedProcess(
          args=cmd,
          returncode=0,
          stdout=json.dumps({"items": items}),
          stderr="",
      )

    with mock.patch("subprocess.run", side_effect=_run):
      self.assertEqual(
          workload_submitter.fetch_namespace_jobsets("default"),
          {"tpums-a": jobset},
      )
      self.assertEqual(
          workload_submitter.fetch_namespace_pods("default"),
          {"tpums-a": [pod]},
      )

  @mock.patch.object(time, "sleep")
  @mock.patch.object(
      workload_submitter, "fetch_namespace_pods", return_value={}
  )
  @mock.patch.object(
      workload_submitter,
      "fetch_namespace_jobsets",
      side_effect=RuntimeError("boom"),
  )
  def test_workload_manager_poll_raises_after_consecutive_fetch_failures(
      self, mock_jobsets, mock_pods, mock_sleep
  ):
    """poll() gives up after MAX_CONSECUTIVE_POLL_FAILURES failed queries."""
    del mock_pods, mock_sleep  # Unused.
    manager = workload_submitter.WorkloadManager(
        active_workloads={"tpums-test-workload": _workload()}
    )
    with self.assertRaisesRegex(RuntimeError, "boom"):
      manager.poll()
    self.assertEqual(
        mock_jobsets.call_count,
        workload_submitter.MAX_CONSECUTIVE_POLL_FAILURES,
    )

  @mock.patch.object(time, "sleep")
  @mock.patch.object(
      workload_submitter, "fetch_namespace_pods", return_value={}
  )
  @mock.patch.object(workload_submitter, "fetch_namespace_jobsets")
  def test_workload_manager_poll_recovers_from_transient_fetch_failure(
      self, mock_jobsets, mock_pods, mock_sleep
  ):
    """A single failed cluster query is retried on the next tick."""
    del mock_pods, mock_sleep  # Unused.
    mock_jobsets.side_effect = [
        RuntimeError("blip"),
        {
            "tpums-test-workload": {
                "status": {
                    "conditions": [{"type": "Completed", "status": "True"}]
                }
            }
        },
    ]
    manager = workload_submitter.WorkloadManager(
        active_workloads={"tpums-test-workload": _workload()}
    )
    results = manager.poll()
    self.assertLen(results, 1)
    self.assertEqual(results[0].status, "SUCCESS")
    self.assertEqual(mock_jobsets.call_count, 2)

  def test_update_workload_states_infinite_timeout(self):
    """With no timeout, a long-running live JobSet never times out."""
    wl = _workload(
        workload_name="tpums-long-running",
        total_pods=1,
        start_time=time.time() - 20 * 3600,
    )
    active = {"tpums-long-running": wl}
    workload_submitter.update_workload_states(
        active,
        jobsets_data={"tpums-long-running": {"status": {}}},
        pods_data={},
        timeout_minutes=None,
    )
    self.assertFalse(wl.terminal)
    self.assertNotEqual(wl.status, "TIMEOUT")

  @parameterized.named_parameters(
      ("no_pods", [], "FAILED"),
      ("running_pod", [{"status": {"phase": "Running"}}], "FAILED"),
      ("all_pods_succeeded", [{"status": {"phase": "Succeeded"}}], "SUCCESS"),
  )
  def test_update_workload_states_missing_jobset(self, pods, expected_status):
    """A JobSet missing from a successful listing is terminal, not PENDING."""
    wl = _workload(status="RUNNING")
    pods_data = {"tpums-test-workload": pods} if pods else {}
    all_done = workload_submitter.update_workload_states(
        {"tpums-test-workload": wl},
        jobsets_data={},
        pods_data=pods_data,
        timeout_minutes=None,
    )
    self.assertTrue(all_done)
    self.assertTrue(wl.terminal)
    self.assertEqual(wl.status, expected_status)
    if expected_status == "FAILED":
      self.assertIn("disappeared", wl.error_message)

  @parameterized.named_parameters(
      ("jobset_active", {"tpums-test-workload": {"status": {}}}, "PENDING"),
      ("jobset_disappeared", {}, "FAILED"),
  )
  def test_update_workload_states_partial_pod_listing_is_not_success(
      self, jobsets_data, expected_status
  ):
    """One succeeded pod of a 4-host workload is not a successful run."""
    wl = _workload(status="RUNNING", total_pods=4)
    workload_submitter.update_workload_states(
        {"tpums-test-workload": wl},
        jobsets_data=jobsets_data,
        pods_data={"tpums-test-workload": [{"status": {"phase": "Succeeded"}}]},
        timeout_minutes=None,
    )
    self.assertEqual(wl.status, expected_status)
    self.assertEqual(wl.terminal, expected_status == "FAILED")
    self.assertEqual(wl.total_pods, 4)

  def test_update_workload_states_partial_failure_waits_for_all_pods(self):
    """A failed pod with unlisted peers stays non-terminal until all report."""
    wl = _workload(status="RUNNING", total_pods=4)
    workload_submitter.update_workload_states(
        {"tpums-test-workload": wl},
        jobsets_data={"tpums-test-workload": {"status": {}}},
        pods_data={
            "tpums-test-workload": [
                {"status": {"phase": "Succeeded"}},
                {"status": {"phase": "Failed"}},
            ]
        },
        timeout_minutes=None,
    )
    self.assertEqual(wl.status, "PENDING")
    self.assertFalse(wl.terminal)

  @mock.patch("subprocess.run")
  def test_fetch_cluster_credentials(self, mock_subprocess_run):
    """Credentials are fetched via the region first."""
    workload_submitter.fetch_cluster_credentials(
        cluster="test-cluster",
        project="test-project",
        region="us-central1",
    )
    mock_subprocess_run.assert_called_once()
    args, kwargs = mock_subprocess_run.call_args
    self.assertIn("gcloud", args[0])
    self.assertIn("get-credentials", args[0])
    self.assertIn("test-cluster", args[0])
    self.assertIn("--project=test-project", args[0])
    self.assertIn("--region=us-central1", args[0])
    self.assertEqual(kwargs.get("stdout"), subprocess.PIPE)
    self.assertEqual(kwargs.get("stderr"), subprocess.STDOUT)
    self.assertTrue(kwargs.get("text"))
    self.assertTrue(kwargs.get("check"))
    self.assertNotIn("capture_output", kwargs)

  @mock.patch("subprocess.run")
  def test_fetch_cluster_credentials_fallback_zone(self, mock_subprocess_run):
    """A failed region lookup falls back to the zone."""
    mock_subprocess_run.side_effect = [
        subprocess.CalledProcessError(1, "gcloud", output="region failed"),
        mock.MagicMock(),
    ]
    workload_submitter.fetch_cluster_credentials(
        cluster="test-cluster",
        project="test-project",
        region="us-central1",
        zone="us-central1-c",
    )
    self.assertEqual(mock_subprocess_run.call_count, 2)
    args2, kwargs2 = mock_subprocess_run.call_args_list[1]
    self.assertIn("--zone=us-central1-c", args2[0])
    self.assertEqual(kwargs2.get("stdout"), subprocess.PIPE)
    self.assertEqual(kwargs2.get("stderr"), subprocess.STDOUT)
    self.assertTrue(kwargs2.get("text"))
    self.assertTrue(kwargs2.get("check"))
    self.assertNotIn("capture_output", kwargs2)

  @mock.patch("subprocess.run")
  def test_fetch_cluster_credentials_timeout_falls_back_to_zone(
      self, mock_subprocess_run
  ):
    """A hung region lookup times out and falls back to the zone."""
    mock_subprocess_run.side_effect = [
        subprocess.TimeoutExpired("gcloud", utils.CLI_TIMEOUT_SECONDS),
        mock.MagicMock(),
    ]
    workload_submitter.fetch_cluster_credentials(
        cluster="test-cluster",
        project="test-project",
        region="us-central1",
        zone="us-central1-c",
    )
    self.assertEqual(mock_subprocess_run.call_count, 2)
    for call in mock_subprocess_run.call_args_list:
      self.assertEqual(call.kwargs["timeout"], utils.CLI_TIMEOUT_SECONDS)

  def test_canonical_suite_includes_device_to_device(self):
    """Canonical tpu7x suite includes 2x2x1 device_to_device."""
    configs = workload_submitter.load_canonical_configs(
        utils.ACCELERATOR_CONFIGS["tpu7x"].suite_path
    )
    self.assertIn(
        "configs/tpu7x/2x2x1/device_to_device.yaml",
        configs,
    )

  def test_workload_manager_prepare_workloads_accelerator_conflict_raises(self):
    """Verifies prepare_workloads raises ValueError on accelerator/config mismatch."""
    cfg_path = pathlib.Path(_CONFIG_DIR) / "2x2x1" / "gemm.yaml"
    manager = workload_submitter.WorkloadManager(
        accelerator="v6e", dry_run=True
    )
    with self.assertRaises(ValueError):
      manager.prepare_workloads([cfg_path])

  def test_workload_manager_prepare_config_outside_package_raises(self):
    """Configs outside the package can't run from the GitHub checkout."""
    cfg = pathlib.Path(
        self.create_tempfile("configs/tpu7x/2x2x1/x.yaml").full_path
    )
    manager = workload_submitter.WorkloadManager(dry_run=True)
    with self.assertRaisesRegex(ValueError, "outside"):
      manager.prepare_workloads([cfg])

  def test_workload_manager_prepare_dry_run(self):
    cfg_path = pathlib.Path(_CONFIG_DIR) / "2x2x1" / "gemm.yaml"
    manager = workload_submitter.WorkloadManager(dry_run=True)
    results = manager.run([cfg_path])
    self.assertLen(results, 1)
    self.assertEqual(results[0].status, "DRY_RUN")
    self.assertEqual(results[0].topology, "2x2x1")
    self.assertEmpty(manager.active_workloads)

  def test_workload_manager_prepare_dry_run_rejects_duplicate_names(self):
    """--dry-run catches workload names that would collide on the cluster."""
    cfg_path = pathlib.Path(_CONFIG_DIR) / "2x2x1" / "gemm.yaml"
    for dry_run in (True, False):
      with self.subTest(dry_run=dry_run):
        manager = workload_submitter.WorkloadManager(dry_run=dry_run)
        with self.assertRaisesRegex(ValueError, "Duplicate sanitized workload"):
          manager.prepare_workloads([cfg_path, cfg_path])

  @mock.patch("subprocess.run")
  def test_apply_jobset_uses_kubectl_create_and_parses_generated_name(
      self, mock_subprocess_run
  ):
    """Verifies apply_jobset runs kubectl create and parses the generated name."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=["kubectl", "create", "-f", "-"],
        returncode=0,
        stdout="jobset.jobset.x-k8s.io/tpums-2x2x1-gemm-abc12 created\n",
        stderr="",
    )
    generated_name = workload_submitter.apply_jobset("apiVersion: v1\n")
    self.assertEqual(generated_name, "tpums-2x2x1-gemm-abc12")
    mock_subprocess_run.assert_called_once_with(
        ["kubectl", "create", "-f", "-"],
        input="apiVersion: v1\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=utils.CLI_TIMEOUT_SECONDS,
    )

  @mock.patch(
      "subprocess.run",
      return_value=subprocess.CompletedProcess(
          args=[], returncode=0, stdout="something unexpected\n", stderr=""
      ),
  )
  def test_apply_jobset_unparseable_output_raises(self, mock_subprocess_run):
    del mock_subprocess_run  # Unused.
    with self.assertRaisesRegex(RuntimeError, "Could not parse"):
      workload_submitter.apply_jobset("apiVersion: v1\n")

  @mock.patch("subprocess.run", side_effect=FileNotFoundError("kubectl"))
  def test_apply_jobset_missing_kubectl_raises(self, mock_subprocess_run):
    del mock_subprocess_run  # Unused.
    with self.assertRaises(FileNotFoundError):
      workload_submitter.apply_jobset("apiVersion: v1\n")

  @mock.patch.object(
      workload_submitter, "apply_jobset", side_effect=RuntimeError("no name")
  )
  def test_workload_manager_submit_all_marks_failed_on_apply_error(
      self, mock_apply
  ):
    """apply_jobset errors mark the workload FAILED under its original key."""
    del mock_apply  # Unused.
    cfg_path = pathlib.Path(_CONFIG_DIR) / "2x2x1" / "gemm.yaml"
    manager = workload_submitter.WorkloadManager(gcs_bucket="gs://test-bucket")
    manager.prepare_workloads([cfg_path])
    manager.submit_all()
    wl = manager.active_workloads["tpums-2x2x1-gemm"]
    self.assertEqual(wl.status, "FAILED")
    self.assertTrue(wl.terminal)
    self.assertIn("no name", wl.error_message)

  @mock.patch.object(workload_submitter, "apply_jobset")
  def test_workload_manager_prepare_and_submit(self, mock_apply):
    """Verifies submit_all re-keys active_workloads by generated JobSet name."""
    mock_apply.return_value = "tpums-2x2x1-gemm-xyz99"
    cfg_path = pathlib.Path(_CONFIG_DIR) / "2x2x1" / "gemm.yaml"
    manager = workload_submitter.WorkloadManager(
        gcs_bucket="gs://test-bucket", dry_run=False
    )
    dry_run_results = manager.prepare_workloads([cfg_path])
    self.assertEqual(dry_run_results, [])
    self.assertLen(manager.active_workloads, 1)

    manager.submit_all()
    mock_apply.assert_called_once()
    self.assertIn("tpums-2x2x1-gemm-xyz99", manager.active_workloads)
    self.assertNotIn("tpums-2x2x1-gemm", manager.active_workloads)
    wl = manager.active_workloads["tpums-2x2x1-gemm-xyz99"]
    self.assertEqual(wl.workload_name, "tpums-2x2x1-gemm-xyz99")
    self.assertTrue(wl.gcs_output_dir.endswith("/tpums-2x2x1-gemm"))
    self.assertIn(
        "tpums-2x2x1-gemm-xyz99",
        workload_submitter.render_status_dashboard(
            manager.active_workloads, manager.start_time
        ),
    )

  @parameterized.named_parameters(("deleted", True), ("delete_failed", False))
  @mock.patch.object(workload_submitter, "delete_jobset")
  def test_workload_manager_cleanup(self, deleted, mock_delete):
    """cleaned_up follows delete_jobset() so failed deletes are retried."""
    mock_delete.return_value = deleted
    wl = _workload(
        workload_name="tpums-mgr-test",
        status="SUCCESS",
        terminal=True,
    )
    manager = workload_submitter.WorkloadManager(
        namespace="test-ns",
        active_workloads={"tpums-mgr-test": wl},
    )
    manager.cleanup_workloads(only_terminal=True)
    mock_delete.assert_called_once_with("tpums-mgr-test", "test-ns")
    self.assertEqual(wl.cleaned_up, deleted)

  @mock.patch("subprocess.run")
  def test_delete_jobset_uses_ignore_not_found(self, mock_subprocess_run):
    """Already-deleted JobSets count as cleaned up."""
    mock_subprocess_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )
    self.assertTrue(workload_submitter.delete_jobset("tpums-a", "test-ns"))
    cmd = mock_subprocess_run.call_args.args[0]
    self.assertIn("--ignore-not-found", cmd)
    self.assertEqual(
        mock_subprocess_run.call_args.kwargs["timeout"],
        utils.CLI_TIMEOUT_SECONDS,
    )

  @parameterized.named_parameters(
      (
          "nonzero_exit",
          dict(
              return_value=subprocess.CompletedProcess(
                  args=[], returncode=1, stdout="", stderr="Forbidden"
              )
          ),
      ),
      (
          "timeout",
          dict(side_effect=subprocess.TimeoutExpired("kubectl", 300)),
      ),
  )
  def test_delete_jobset_reports_failure(self, run_kwargs):
    """Failed or hung deletes are logged and reported as not cleaned up."""
    with mock.patch("subprocess.run", **run_kwargs):
      with self.assertLogs(level="ERROR"):
        self.assertFalse(workload_submitter.delete_jobset("tpums-a", "test-ns"))

  @mock.patch.object(workload_submitter, "delete_jobset", return_value=False)
  @mock.patch.object(
      workload_submitter, "fetch_namespace_pods", return_value={}
  )
  @mock.patch.object(
      workload_submitter, "fetch_namespace_jobsets", return_value={}
  )
  def test_workload_manager_poll_logs_leaked_jobsets(
      self, mock_jobsets, mock_pods, mock_delete
  ):
    """JobSets that could not be deleted are reported after polling."""
    del mock_jobsets, mock_pods  # Unused.
    wl = _workload(workload_name="tpums-leak", status="SUCCESS", terminal=True)
    manager = workload_submitter.WorkloadManager(
        cleanup=True, active_workloads={"tpums-leak": wl}
    )
    with self.assertLogs(level="ERROR") as logs:
      results = manager.poll()
    self.assertEqual(results[0].status, "SUCCESS")
    self.assertFalse(wl.cleaned_up)
    self.assertGreaterEqual(mock_delete.call_count, 2)
    self.assertTrue(any("tpums-leak" in line for line in logs.output))

  @parameterized.named_parameters(
      ("tpu7x_2x2x1", "configs/tpu7x/2x2x1/gemm.yaml", 1),
      ("tpu7x_2x2x2", "configs/tpu7x/2x2x2/all_gather.yaml", 2),
      ("v6e_4x4", "configs/v6e/4x4/all_gather.yaml", 4),
  )
  def test_workload_manager_prepare_sets_total_pods_from_topology(
      self, cfg_rel, expected_pods
  ):
    """prepare_workloads populates ActiveWorkload.total_pods from topology."""
    manager = workload_submitter.WorkloadManager(gcs_bucket="gs://b")
    with mock.patch.object(
        utils, "compute_topology_info", wraps=utils.compute_topology_info
    ) as spy_topo:
      manager.prepare_workloads([utils.PACKAGE_ROOT / cfg_rel])
    self.assertEqual(spy_topo.call_count, 1)
    wl = next(iter(manager.active_workloads.values()))
    self.assertEqual(wl.total_pods, expected_pods)

  def test_active_workload_topology_pods_and_dashboard_alignment(self):
    """Dashboard renders multi-pod counts and aligns columns."""
    wl_2x2x2 = _workload(
        workload_name="tpums-2x2x2-all-gather",
        config_name="all_gather",
        config_rel_path="configs/tpu7x/2x2x2/all_gather.yaml",
        topology="2x2x2",
        start_time=time.time() - 10,
        total_pods=2,
        status="QUEUED",
    )

    wl_success = _workload(
        workload_name="tpums-2x2x1-gemm",
        start_time=time.time() - 20,
        status="SUCCESS",
        successful_pods=1,
        terminal=True,
    )
    dash = workload_submitter.render_status_dashboard(
        {"tpums-2x2x2-all-gather": wl_2x2x2, "tpums-2x2x1-gemm": wl_success},
        time.time() - 20,
    )
    self.assertIn("0/2", dash)
    self.assertIn("1/1", dash)

    # Verify visual column alignment: header pipes and row pipes must align
    # when accounting for 2-column-wide status emojis.
    lines = dash.splitlines()
    header_line = next(l for l in lines if "Config" in l and "Status" in l)
    row_lines = [l for l in lines if "tpums-2x2x2" in l or "tpums-2x2x1" in l]
    header_pipe_indices = [i for i, ch in enumerate(header_line) if ch == "|"]
    for row in row_lines:
      # Each row contains one 2-cell emoji before the 3rd pipe, so code-point
      # indices after the emoji are shifted by -1 compared to visual columns.
      row_pipes = [i for i, ch in enumerate(row) if ch == "|"]
      self.assertEqual(row_pipes[0], header_pipe_indices[0])
      self.assertEqual(row_pipes[1], header_pipe_indices[1])
      self.assertEqual(row_pipes[2] + 1, header_pipe_indices[2])


if __name__ == "__main__":
  absltest.main()
