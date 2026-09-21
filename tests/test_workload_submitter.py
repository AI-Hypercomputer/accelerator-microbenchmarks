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
from automation import workload_submitter
import yaml

_CONFIG_DIR = str(pathlib.Path(__file__).parent.parent / "configs" / "tpu7x")


class WorkloadSubmitterTest(parameterized.TestCase):

  def test_infer_topology_from_config(self):
    self.assertEqual(
        workload_submitter.infer_topology_from_config(
            "configs/tpu7x/2x2x1/gemm.yaml", {}
        ),
        "2x2x1",
    )
    self.assertEqual(
        workload_submitter.infer_topology_from_config(
            "configs/tpu7x/2x2x2/all_reduce.yaml", {}
        ),
        "2x2x2",
    )
    self.assertEqual(
        workload_submitter.infer_topology_from_config(
            "configs/tpu7x/4x4x4/all_gather.yaml", {}
        ),
        "4x4x4",
    )

    # A topology wider than any registered one must not be truncated by a
    # prefix match (the old substring scan reported "2x2x1" here).
    self.assertEqual(
        workload_submitter.infer_topology_from_config(
            "configs/tpu7x/2x2x16/all_to_all.yaml", {}
        ),
        "2x2x16",
    )

    # No topology in the path: log an error and fall back to the default.
    data = {"benchmarks": [{"name": "all_reduce", "mesh": "2x2x2"}]}
    with self.assertLogs(level="ERROR"):
      self.assertEqual(
          workload_submitter.infer_topology_from_config("unnamed.yaml", data),
          workload_submitter.DEFAULT_TOPOLOGY,
      )

  def test_sanitize_workload_name(self):
    name = workload_submitter.sanitize_workload_name(
        "gemm", topology="2x2x1"
    )
    self.assertEqual(name, "tpums-2x2x1-gemm")
    self.assertLessEqual(len(name), 45)
    self.assertRegex(name, r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

    # If topology already present in name, avoid duplicate
    name2 = workload_submitter.sanitize_workload_name(
        "2x2x2_all_gather", topology="2x2x2"
    )
    self.assertEqual(name2, "tpums-2x2x2-all-gather")

  def test_generate_jobset_manifest_single_node(self):
    manifest_yaml = workload_submitter.generate_jobset_manifest(
        workload_name="tpums-test-01",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="medium",
        topology="2x2x1",
        docker_image=(
            "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest"
        ),
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        gcs_output_dir="gs://test-bucket/daily/test",
    )

    doc = yaml.safe_load(manifest_yaml)
    self.assertEqual(doc["kind"], "JobSet")
    self.assertEqual(doc["metadata"]["name"], "tpums-test-01")
    self.assertEqual(
        doc["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
        "multislice-queue",
    )

    spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"]
    self.assertEqual(spec["parallelism"], 1)
    self.assertEqual(spec["completions"], 1)
    pod_spec = spec["template"]["spec"]
    self.assertEqual(
        pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-accelerator"],
        "tpu7x",
    )
    self.assertEqual(
        pod_spec["nodeSelector"]["cloud.google.com/gke-tpu-topology"], "2x2x1"
    )
    self.assertEqual(
        pod_spec["containers"][0]["resources"]["limits"]["google.com/tpu"], 4
    )
    container_script = pod_spec["containers"][0]["args"][0]
    self.assertIn(
        "tpums benchmark run-config"
        ' "configs/tpu7x/2x2x1/gemm.yaml"',
        container_script,
    )

  def test_generate_jobset_manifest_multi_node(self):
    manifest_yaml = workload_submitter.generate_jobset_manifest(
        workload_name="tpums-multi-node",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="medium",
        topology="2x2x2",
        docker_image="gcr.io/test/image:latest",
        config_rel_path="configs/tpu7x/2x2x2/all_reduce.yaml",
        gcs_output_dir="gs://test-bucket/daily/multi",
    )

    doc = yaml.safe_load(manifest_yaml)
    spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"]
    self.assertEqual(spec["parallelism"], 2)
    self.assertEqual(spec["completions"], 2)
    pod_spec = spec["template"]["spec"]
    self.assertEqual(
        pod_spec["nodeSelector"]["cloud.google.com/placement-policy-name"],
        "tpu7x-16-2x2x2-placement-policy",
    )

  def test_generate_jobset_manifest_without_reservation(self):
    manifest_yaml = workload_submitter.generate_jobset_manifest(
        workload_name="tpums-no-res",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="medium",
        topology="2x2x1",
        docker_image="gcr.io/test/image:latest",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        gcs_output_dir="gs://test-bucket/daily/test",
        reservation_name=None,
    )
    doc = yaml.safe_load(manifest_yaml)
    pod_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
        "spec"
    ]
    self.assertNotIn(
        "cloud.google.com/reservation-name", pod_spec["nodeSelector"]
    )
    toleration_keys = [t.get("key") for t in pod_spec.get("tolerations", [])]
    self.assertIn("cloud.google.com/reservation-name", toleration_keys)

  def test_generate_jobset_manifest_with_reservation(self):
    manifest_yaml = workload_submitter.generate_jobset_manifest(
        workload_name="tpums-with-res",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="medium",
        topology="4x4x4",
        docker_image="gcr.io/test/image:latest",
        config_rel_path="configs/tpu7x/4x4x4/all_gather.yaml",
        gcs_output_dir="gs://test-bucket/daily/test",
        reservation_name="example-tpu-reservation",
    )
    doc = yaml.safe_load(manifest_yaml)
    pod_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
        "spec"
    ]
    self.assertEqual(
        pod_spec["nodeSelector"]["cloud.google.com/reservation-name"],
        "example-tpu-reservation",
    )
    self.assertEqual(
        pod_spec["nodeSelector"]["cloud.google.com/placement-policy-name"],
        "tpu7x-128-4x4x4-placement-policy",
    )

  def test_generate_jobset_manifest_git_branch_default(self):
    manifest_yaml = workload_submitter.generate_jobset_manifest(
        workload_name="tpums-branch-default",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="medium",
        topology="2x2x1",
        docker_image="gcr.io/test/image:latest",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        gcs_output_dir="gs://test-bucket/daily/test",
    )
    doc = yaml.safe_load(manifest_yaml)
    container = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]["containers"][0]
    env_vars = {item["name"]: item["value"] for item in container["env"]}
    self.assertIn("GIT_BRANCH", env_vars)
    self.assertEqual(env_vars["GIT_BRANCH"], "main")
    self.assertNotIn("CONFIG_B64", env_vars)
    self.assertNotIn("CONFIG_B64", manifest_yaml)
    self.assertIn('git clone --branch "main" --single-branch', manifest_yaml)

  def test_generate_jobset_manifest_git_branch_custom(self):
    manifest_yaml = workload_submitter.generate_jobset_manifest(
        workload_name="tpums-branch-custom",
        namespace="default",
        queue_name="multislice-queue",
        priority_class="medium",
        topology="2x2x1",
        docker_image="gcr.io/test/image:latest",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        gcs_output_dir="gs://test-bucket/daily/test",
        git_branch="feat/custom-mesh-v2",
    )
    doc = yaml.safe_load(manifest_yaml)
    container = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]["containers"][0]
    env_vars = {item["name"]: item["value"] for item in container["env"]}
    self.assertEqual(env_vars["GIT_BRANCH"], "feat/custom-mesh-v2")
    self.assertIn(
        'git clone --branch "feat/custom-mesh-v2" --single-branch',
        manifest_yaml,
    )

  def test_resolve_config_files_mutual_exclusion(self):
    with self.assertRaises(ValueError):
      workload_submitter.resolve_config_files(
          config_dir=_CONFIG_DIR,
          configs=[os.path.join(_CONFIG_DIR, "2x2x1/gemm.yaml")],
          filter_patterns=["all_gather"],
      )

  def test_resolve_config_files_filter_multi_pattern(self):
    resolved = workload_submitter.resolve_config_files(
        config_dir=_CONFIG_DIR,
        filter_patterns=["all_gather", "gemm"],
    )
    self.assertTrue(len(resolved) > 0)
    for p in resolved:
      name = str(p).lower()
      self.assertTrue("all_gather" in name or "gemm" in name)

  def test_resolve_config_files_topologies(self):
    resolved = workload_submitter.resolve_config_files(
        config_dir=_CONFIG_DIR,
        topologies=["2x2x1"],
    )
    self.assertTrue(len(resolved) > 0)
    for p in resolved:
      self.assertIn("2x2x1", str(p))

  def test_resolve_config_files_canonical_default(self):
    # Passing a non-existent config_dir verifies canonical suite resolution
    # is decoupled from config_dir.
    resolved = workload_submitter.resolve_config_files(
        config_dir="/non/existent/config_dir",
    )
    self.assertLen(resolved, 11)
    names = [f"{p.parent.name}/{p.name}" for p in resolved]
    expected = [
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
    ]
    self.assertEqual(sorted(names), sorted(expected))
    for p in resolved:
      name = str(p)
      self.assertNotIn("2x2x2", name)
      self.assertNotIn("2x4x4", name)
      self.assertNotIn("reduce_scatter", name)

  def test_load_canonical_configs_formats(self):
    # Test list format
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
      yaml.dump(["configs/test/a.yaml", "configs/test/b.yaml"], f)
      list_file = f.name
    try:
      configs = workload_submitter.load_canonical_configs(list_file)
      self.assertEqual(configs, ["configs/test/a.yaml", "configs/test/b.yaml"])
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

  def test_resolve_config_files_custom_suite_file(self):
    custom_content = {
        "configs": [
            "configs/tpu7x/2x2x1/gemm.yaml",
            "configs/tpu7x/4x4x4/all_gather.yaml",
        ]
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
      yaml.dump(custom_content, f)
      custom_file = f.name
    try:
      resolved = workload_submitter.resolve_config_files(
          config_dir="/non/existent/config_dir",
          suite_file=custom_file,
      )
      self.assertLen(resolved, 2)
      names = sorted([f"{p.parent.name}/{p.name}" for p in resolved])
      self.assertEqual(
          names, ["2x2x1/gemm.yaml", "4x4x4/all_gather.yaml"]
      )
    finally:
      os.remove(custom_file)

  def test_load_canonical_configs_missing_file(self):
    with self.assertRaises(FileNotFoundError):
      workload_submitter.load_canonical_configs("/non/existent/suite.yaml")

  def test_update_workload_states_and_dashboard(self):
    wl = workload_submitter.ActiveWorkload(
        workload_name="tpums-test-workload",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        manifest_yaml="",
        gcs_output_dir="gs://test-bucket/daily/test",
        start_time=time.time() - 30,
    )
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
        active, jobsets_data={}, pods_data=mock_pods, timeout_minutes=45
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
    wl = workload_submitter.ActiveWorkload(
        workload_name="tpums-empty-pods-success",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        manifest_yaml="",
        gcs_output_dir="gs://test-bucket/daily/test",
        start_time=time.time(),
    )
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
    wl = workload_submitter.ActiveWorkload(
        workload_name="tpums-empty-pods-failed",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        manifest_yaml="",
        gcs_output_dir="gs://test-bucket/daily/test",
        start_time=time.time(),
    )
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

  def test_poll_active_workloads_cleanup_flag(self):
    wl = workload_submitter.ActiveWorkload(
        workload_name="tpums-cleanup-test",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        manifest_yaml="",
        gcs_output_dir="gs://test-bucket/daily/test",
        start_time=time.time(),
        status="SUCCESS",
        terminal=True,
    )
    active = {"tpums-cleanup-test": wl}
    # Test polling returns benchmark result and records cleaned_up state
    results = workload_submitter.poll_active_workloads(
        active_workloads=active,
        namespace="default",
        timeout_minutes=1,
        poll_interval_seconds=1,
        cleanup=True,
    )
    self.assertLen(results, 1)
    self.assertEqual(results[0].status, "SUCCESS")
    self.assertTrue(wl.cleaned_up)

  def test_update_workload_states_infinite_timeout(self):
    wl = workload_submitter.ActiveWorkload(
        workload_name="tpums-long-running",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        total_pods=1,
        gcs_output_dir="gs://bucket/test",
        manifest_yaml="",
        start_time=time.time() - 72000,  # 20 hours ago
    )
    active = {"tpums-long-running": wl}
    workload_submitter.update_workload_states(
        active, jobsets_data={}, pods_data={}, timeout_minutes=None
    )
    self.assertFalse(wl.terminal)
    self.assertNotEqual(wl.status, "TIMEOUT")

  @mock.patch("subprocess.run")
  def test_fetch_cluster_credentials(self, mock_subprocess_run):
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

  def test_canonical_suite_includes_device_to_device(self):
    configs = workload_submitter.load_canonical_configs()
    self.assertIn(
        "configs/tpu7x/2x2x1/device_to_device.yaml",
        configs,
    )

  def test_compute_topology_info_standard(self):
    t_2x2x1 = workload_submitter.compute_topology_info("2x2x1")
    self.assertEqual(t_2x2x1["num_nodes"], 1)
    self.assertEqual(t_2x2x1["total_chips"], 4)
    self.assertIsNone(t_2x2x1["placement_policy"])

    t_2x2x2 = workload_submitter.compute_topology_info("2x2x2")
    self.assertEqual(t_2x2x2["num_nodes"], 2)
    self.assertEqual(t_2x2x2["total_chips"], 8)
    self.assertEqual(
        t_2x2x2["placement_policy"], "tpu7x-16-2x2x2-placement-policy"
    )

    t_2x4x4 = workload_submitter.compute_topology_info("2x4x4")
    self.assertEqual(t_2x4x4["num_nodes"], 8)
    self.assertEqual(t_2x4x4["total_chips"], 32)
    self.assertEqual(
        t_2x4x4["placement_policy"], "tpu7x-64-2x4x4-placement-policy"
    )

    t_4x4x4 = workload_submitter.compute_topology_info("4x4x4")
    self.assertEqual(t_4x4x4["num_nodes"], 16)
    self.assertEqual(t_4x4x4["total_chips"], 64)
    self.assertEqual(
        t_4x4x4["placement_policy"], "tpu7x-128-4x4x4-placement-policy"
    )

    t_4x4x8 = workload_submitter.compute_topology_info("4x4x8")
    self.assertEqual(t_4x4x8["num_nodes"], 32)
    self.assertEqual(t_4x4x8["total_chips"], 128)
    self.assertEqual(
        t_4x4x8["placement_policy"], "tpu7x-256-4x4x8-placement-policy"
    )

  def test_compute_topology_info_custom_and_arbitrary(self):
    t_4x8x8 = workload_submitter.compute_topology_info("4x8x8")
    self.assertEqual(t_4x8x8["num_nodes"], 64)
    self.assertEqual(t_4x8x8["total_chips"], 256)
    self.assertEqual(
        t_4x8x8["placement_policy"], "tpu7x-512-4x8x8-placement-policy"
    )

  def test_compute_topology_info_invalid(self):
    for bad_topo in ["invalid", "abc", "1x", "2x2xa", ""]:
      with self.subTest(bad_topo=bad_topo):
        with self.assertRaises(ValueError):
          workload_submitter.compute_topology_info(bad_topo)

  def test_workload_manager_prepare_dry_run(self):
    cfg_path = pathlib.Path(_CONFIG_DIR) / "2x2x1" / "gemm.yaml"
    manager = workload_submitter.WorkloadManager(dry_run=True)
    results = manager.run([cfg_path])
    self.assertLen(results, 1)
    self.assertEqual(results[0].status, "DRY_RUN")
    self.assertEqual(results[0].topology, "2x2x1")
    self.assertEmpty(manager.active_workloads)

  @mock.patch.object(workload_submitter, "apply_jobset")
  def test_workload_manager_prepare_and_submit(self, mock_apply):
    cfg_path = pathlib.Path(_CONFIG_DIR) / "2x2x1" / "gemm.yaml"
    manager = workload_submitter.WorkloadManager(dry_run=False)
    dry_run_results = manager.prepare_workloads([cfg_path])
    self.assertEqual(dry_run_results, [])
    self.assertLen(manager.active_workloads, 1)

    manager.submit_all()
    mock_apply.assert_called_once()

  @mock.patch.object(workload_submitter, "delete_jobset")
  def test_workload_manager_cleanup(self, mock_delete):
    wl = workload_submitter.ActiveWorkload(
        workload_name="tpums-mgr-test",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        manifest_yaml="",
        gcs_output_dir="gs://test-bucket/test",
        start_time=time.time(),
        status="SUCCESS",
        terminal=True,
    )
    manager = workload_submitter.WorkloadManager(
        namespace="test-ns",
        active_workloads={"tpums-mgr-test": wl},
    )
    manager.cleanup_workloads(only_terminal=True)
    mock_delete.assert_called_once_with("tpums-mgr-test", "test-ns")
    self.assertTrue(wl.cleaned_up)

  def test_workload_manager_update_states_and_dashboard(self):
    wl = workload_submitter.ActiveWorkload(
        workload_name="tpums-mgr-dash",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        manifest_yaml="",
        gcs_output_dir="gs://test-bucket/test",
        start_time=time.time() - 10,
    )
    manager = workload_submitter.WorkloadManager(
        active_workloads={"tpums-mgr-dash": wl}
    )
    mock_pods = {
        "tpums-mgr-dash": [{
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
            }
        }]
    }
    all_done = manager.update_states(jobsets_data={}, pods_data=mock_pods)
    self.assertFalse(all_done)
    self.assertEqual(wl.status, "RUNNING")

    dash = manager.render_dashboard()
    self.assertIn("tpums-mgr-dash", dash)
    self.assertIn("RUNNING", dash)

  def test_active_workload_topology_pods_and_dashboard_alignment(self):
    wl_2x2x2 = workload_submitter.ActiveWorkload(
        workload_name="tpums-2x2x2-all-gather",
        config_name="all_gather",
        config_rel_path="configs/tpu7x/2x2x2/all_gather.yaml",
        topology="2x2x2",
        manifest_yaml="",
        gcs_output_dir="gs://test-bucket/test",
        start_time=time.time() - 10,
        status="QUEUED",
    )
    self.assertEqual(wl_2x2x2.total_pods, 2)

    wl_success = workload_submitter.ActiveWorkload(
        workload_name="tpums-2x2x1-gemm",
        config_name="gemm",
        config_rel_path="configs/tpu7x/2x2x1/gemm.yaml",
        topology="2x2x1",
        manifest_yaml="",
        gcs_output_dir="gs://test-bucket/test",
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
