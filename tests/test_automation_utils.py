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

"""Unit tests for TPUMS automation/utils.py module."""

import argparse
from absl.testing import absltest
from absl.testing import parameterized
from automation import utils


class AutomationUtilsTest(parameterized.TestCase):
  """Tests for shared automation utility functions."""

  @parameterized.named_parameters(
      ("bare_bucket", "my-bucket", (), "gs://my-bucket"),
      ("gs_bucket_trailing_slash", "gs://my-bucket/", (), "gs://my-bucket"),
      (
          "bucket_with_subpaths",
          "gs://my-bucket/",
          ("/2026-10-10/", "aggregated_results", "gemm.csv"),
          "gs://my-bucket/2026-10-10/aggregated_results/gemm.csv",
      ),
      ("blank_bucket", "", ("2026-10-10", "wl"), ""),
      ("slash_only_bucket", "gs://", ("2026-10-10",), ""),
      (
          "empty_subpath_ignored",
          "my-bucket",
          ("", None, "/", "2026-10-10"),
          "gs://my-bucket/2026-10-10",
      ),
  )
  def test_format_gcs_uri(self, bucket, subpaths, expected):
    """Verifies GCS URI normalization with and without gs:// prefix."""
    self.assertEqual(utils.format_gcs_uri(bucket, *subpaths), expected)

  def test_split_csv_tokens(self):
    self.assertEqual(
        utils.split_csv_tokens(["2x2, 4x4", " 8x8 ", ",", ""]),
        ["2x2", "4x4", "8x8"],
    )
    self.assertEqual(utils.split_csv_tokens(None), [])

  @parameterized.named_parameters(
      (
          "rightmost_in_single_string",
          ("gs://my-2x2-bucket/2026-10-10/tpums-4x4-all-reduce",),
          "4x4",
      ),
      (
          "priority_order_first_candidate_wins",
          (
              "prefix-2x2-tpums-4x4x4-gemm",
              "/tmp/2x2x1/configs/tpu7x/8x8/gemm.yaml",
          ),
          "4x4x4",
      ),
      (
          "none_and_empty_skipped",
          (
              None,
              "",
              "custom-name-without-topology",
              "/tmp/my-2x2-dir/configs/tpu7x/2x2x2/all_reduce.yaml",
          ),
          "2x2x2",
      ),
      ("case_insensitive", ("configs/tpu7x/2X2X1/gemm.yaml",), "2x2x1"),
      ("3d_not_truncated", ("configs/tpu7x/2x2x16/gemm.yaml",), "2x2x16"),
      ("zero_dim_skipped", ("configs/v6e/2x2/run_0x4/gemm.yaml",), "2x2"),
      ("v6e_16x16", ("tpums-16x16-all-gather",), "16x16"),
      (
          "workload_name_wins_over_path",
          ("tpums-2x2x2-all-gather", "gs://b/2x2x1/x/summary.csv"),
          "2x2x2",
      ),
  )
  def test_extract_topology(self, candidates, expected):
    """Verifies topology extraction priority and rightmost matching."""
    self.assertEqual(utils.extract_topology(*candidates), expected)

  @parameterized.named_parameters(
      ("no_candidates", ()),
      ("no_match", ("gs://my-bucket/no-topology-here",)),
      ("only_zero_dims", (None, "", "configs/v6e/0x4/gemm.yaml")),
      ("4d_token", ("tpums-2x2x2x2-gemm",)),
      ("alphanumeric_boundary", ("tpu7x2x2",)),
  )
  def test_extract_topology_raises_when_missing(self, candidates):
    with self.assertRaisesRegex(ValueError, "Could not infer TPU topology"):
      utils.extract_topology(*candidates)

  def test_normalize_accelerator(self):
    for raw, expected in (
        (" V6E ", "v6e"),
        ("TPU7X", "tpu7x"),
    ):
      self.assertEqual(utils.normalize_accelerator(raw), expected)
    for bad_acc in ("tpu-v6e-slice", "v7x", "trillium", "ironwood", "v5p"):
      with self.subTest(bad_acc=bad_acc):
        with self.assertRaisesRegex(ValueError, "Unsupported accelerator"):
          utils.normalize_accelerator(bad_acc)

  @parameterized.named_parameters(
      ("explicit_v6e", {"accelerator": " V6E "}, "v6e"),
      ("explicit_tpu7x", {"accelerator": "TPU7X"}, "tpu7x"),
      ("config_path_v6e", {"config_path": "configs/v6e/2x2/gemm.yaml"}, "v6e"),
      (
          "config_path_tpu7x",
          {"config_path": "configs/tpu7x/2x2x1/gemm.yaml"},
          "tpu7x",
      ),
      ("topology_2d", {"topology": "4x4"}, "v6e"),
      ("topology_3d", {"topology": "2x2x1"}, "tpu7x"),
      (
          "unknown_path_falls_back_to_topology",
          {"config_path": "custom/gemm.yaml", "topology": "2x4"},
          "v6e",
      ),
      (
          "rightmost_path_segment_wins_v6e",
          {
              "config_path": "/tmp/tpu7x-workspace/configs/v6e/2x2/gemm.yaml",
              "topology": "2x2",
          },
          "v6e",
      ),
      (
          "rightmost_path_segment_wins_tpu7x",
          {
              "config_path": "/tmp/v6e-archive/configs/tpu7x/2x2x1/gemm.yaml",
              "topology": "2x2x1",
          },
          "tpu7x",
      ),
  )
  def test_infer_accelerator(self, kwargs, expected):
    """Explicit, path and topology inputs each resolve the accelerator."""
    self.assertEqual(utils.infer_accelerator(**kwargs), expected)

  @parameterized.named_parameters(
      ("nothing", {}),
      (
          "config_path_substring_not_matched",
          {"config_path": "configs/v6e_old/gemm.yaml"},
      ),
      ("topology_4d", {"topology": "2x2x2x2"}),
      ("topology_garbage", {"topology": "big"}),
      ("topology_zero_dim", {"topology": "0x4"}),
  )
  def test_infer_accelerator_uninferable_raises(self, kwargs):
    """Never silently defaults to tpu7x when the accelerator is unknown."""
    with self.assertRaisesRegex(ValueError, "Cannot infer TPU accelerator"):
      utils.infer_accelerator(**kwargs)

  @parameterized.named_parameters(
      (
          "path_vs_topology",
          {"config_path": "configs/tpu7x/4x4/gemm.yaml", "topology": "4x4"},
      ),
      ("explicit_vs_topology", {"accelerator": "v6e", "topology": "2x2x1"}),
      (
          "explicit_vs_path",
          {"accelerator": "v6e", "config_path": "configs/tpu7x/2x2x1/x.yaml"},
      ),
  )
  def test_infer_accelerator_conflict_raises(self, kwargs):
    """Inputs implying different accelerators are rejected."""
    with self.assertRaisesRegex(ValueError, "Conflicting TPU accelerators"):
      utils.infer_accelerator(**kwargs)

  def test_accelerator_configs_registry(self):
    """Registry entries are self-consistent for tpu7x and v6e."""
    self.assertEqual(utils.SUPPORTED_ACCELERATORS, ("tpu7x", "v6e"))
    tpu7x_cfg = utils.ACCELERATOR_CONFIGS["tpu7x"]
    self.assertEqual(tpu7x_cfg.gke_accelerator_label, "tpu7x")
    self.assertEqual(tpu7x_cfg.suite_filename, "canonical_suite_v7x.yaml")
    v6e_cfg = utils.ACCELERATOR_CONFIGS["v6e"]
    self.assertEqual(v6e_cfg.gke_accelerator_label, "tpu-v6e-slice")
    self.assertEqual(v6e_cfg.suite_filename, "canonical_suite_v6e.yaml")
    self.assertTrue(v6e_cfg.config_dir.endswith("configs/v6e"))
    for acc, cfg in utils.ACCELERATOR_CONFIGS.items():
      self.assertEqual(cfg.accelerator, acc)
      for topology in cfg.known_topologies:
        self.assertIn(topology, utils.KNOWN_TOPOLOGIES)
        self.assertEqual(
            utils.compute_topology_info(topology)["accelerator"], acc
        )

  @parameterized.named_parameters(
      # (topology, accelerator, num_nodes, chips/node, chips, cores, policy)
      ("v6e_1x1", "1x1", "v6e", 1, 1, 1, 1, None),
      ("v6e_2x2_ws", " 2x2 ", "v6e", 1, 4, 4, 4, None),
      ("v6e_2x4", "2x4", "v6e", 2, 4, 8, 8, "v6e-8-2x4-placement-policy"),
      ("v6e_4x4", "4x4", None, 4, 4, 16, 16, "v6e-16-4x4-placement-policy"),
      (
          "v6e_16x16",
          "16x16",
          "v6e",
          64,
          4,
          256,
          256,
          "v6e-256-16x16-placement-policy",
      ),
      ("tpu7x_1x1x1", "1x1x1", None, 1, 1, 1, 2, None),
      ("tpu7x_2x2x1", "2x2x1", None, 1, 4, 4, 8, None),
      (
          "tpu7x_2x2x2",
          "2x2x2",
          None,
          2,
          4,
          8,
          16,
          "tpu7x-16-2x2x2-placement-policy",
      ),
      (
          "tpu7x_4x4x4",
          "4x4x4",
          "tpu7x",
          16,
          4,
          64,
          128,
          "tpu7x-128-4x4x4-placement-policy",
      ),
      (
          "tpu7x_unregistered_4x8x8",
          "4x8x8",
          None,
          64,
          4,
          256,
          512,
          "tpu7x-512-4x8x8-placement-policy",
      ),
  )
  def test_compute_topology_info(
      self,
      topology,
      accelerator,
      num_nodes,
      chips_per_node,
      total_chips,
      total_cores,
      placement_policy,
  ):
    """Node counts, chips, cores and placement policy per topology."""
    info = utils.compute_topology_info(topology, accelerator=accelerator)
    expected_acc = utils.infer_accelerator(topology=topology.strip())
    self.assertEqual(
        info,
        {
            "accelerator": expected_acc,
            "gke_accelerator": (
                utils.ACCELERATOR_CONFIGS[expected_acc].gke_accelerator_label
            ),
            "num_nodes": num_nodes,
            "chips_per_node": chips_per_node,
            "total_chips": total_chips,
            "total_cores": total_cores,
            "placement_policy": placement_policy,
        },
    )

  def test_compute_topology_info_invalid(self):
    """Malformed, indivisible or mismatched topologies raise."""
    for bad_topo in ("invalid", "abc", "1x", "2x2xa", "", "0x4", "2x0x1"):
      with self.subTest(bad_topo=bad_topo):
        with self.assertRaisesRegex(ValueError, "Invalid topology"):
          utils.compute_topology_info(bad_topo)
    with self.assertRaisesRegex(ValueError, "not divisible by chips_per_node"):
      utils.compute_topology_info("2x3", accelerator="v6e")
    with self.assertRaisesRegex(ValueError, "Conflicting TPU accelerators"):
      utils.compute_topology_info("2x2x1", accelerator="v6e")
    with self.assertRaisesRegex(ValueError, "Conflicting TPU accelerators"):
      utils.compute_topology_info("4x4", accelerator="tpu7x")
    with self.assertRaisesRegex(ValueError, "expects 3D"):
      utils.compute_topology_info("2x2x2x2", accelerator="tpu7x")

  def test_normalize_topologies(self):
    """Topologies must match the --accelerator's dimensionality."""
    norm = utils.normalize_topologies
    self.assertEqual(norm("V6E", ["4x4", " 2X2 "]), ["4x4", "2x2"])
    self.assertEqual(norm("v6e", ["2x2,4x4"]), ["2x2", "4x4"])
    self.assertEqual(norm("tpu7x", ["2x2x1, 4x4x4"]), ["2x2x1", "4x4x4"])
    self.assertEqual(norm("tpu7x", None), [])
    for acc, topologies in (
        ("tpu7x", ["4x4"]),
        ("tpu7x", ["2x2x1", "4x4"]),
        ("v6e", ["2x2x1"]),
        ("v6e", ["2x2x2x2"]),
        ("v6e", ["0x4"]),
        ("v6e", ["abc"]),
    ):
      with self.subTest(acc=acc, topologies=topologies):
        with self.assertRaisesRegex(ValueError, "do not match accelerator"):
          norm(acc, topologies)
    with self.assertRaisesRegex(ValueError, "Unsupported accelerator"):
      norm("v5p", ["2x2"])

  def test_get_accelerator_config(self):
    """Returns registered config or raises when accelerator is unset/unknown."""
    self.assertTrue(
        utils.get_accelerator_config("v6e").config_dir.endswith("configs/v6e")
    )
    self.assertEqual(
        utils.get_accelerator_config(" TPU7X ").accelerator, "tpu7x"
    )
    for bad_acc in (None, "", "v5p"):
      with self.subTest(bad_acc=bad_acc):
        with self.assertRaisesRegex(ValueError, "Unsupported accelerator"):
          utils.get_accelerator_config(bad_acc)

  def test_add_selection_args(self):
    """Exactly one selection mode is required; scan filters parse freely."""
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.HelpFormatter(prog, width=1000)
    )
    utils.add_selection_args(parser)
    args = parser.parse_args([
        "--accelerator",
        "v6e",
        "--config-dir",
        "d",
        "--topologies",
        "2x2,4x4",
        "--categories",
        "gemm",
        "hbm",
    ])
    self.assertEqual(args.accelerator, "v6e")
    self.assertEqual(args.config_dir, "d")
    self.assertEqual(args.topologies, ["2x2,4x4"])
    self.assertEqual(args.categories, ["gemm", "hbm"])
    self.assertIsNone(args.configs)
    self.assertIsNone(args.suite_file)
    self.assertFalse(hasattr(args, "filter"))

    args = parser.parse_args(["--suite-file", "s.yaml"])
    self.assertEqual(args.suite_file, "s.yaml")
    self.assertIsNone(args.accelerator)
    args = parser.parse_args(["--configs", "a.yaml", "b.yaml"])
    self.assertEqual(args.configs, ["a.yaml", "b.yaml"])

    for bad_args in (
        [],  # No selection mode.
        ["--topologies", "2x2"],  # Scan filter without a mode.
        ["--configs", "a.yaml", "--suite-file", "s.yaml"],
        ["--configs", "a.yaml", "--accelerator", "v6e"],
        ["--suite-file", "s.yaml", "--accelerator", "v6e"],
        ["--accelerator", "v5p"],
        ["--accelerator", "v6e", "--filter", "gemm"],  # Renamed flag.
    ):
      with self.subTest(bad_args=bad_args):
        with self.assertRaises(SystemExit):
          parser.parse_args(bad_args)

    help_text = parser.format_help()
    self.assertIn("automation/canonical_suite_v7x.yaml", help_text)
    self.assertIn("automation/canonical_suite_v6e.yaml", help_text)
    self.assertIn("--accelerator mode only", help_text)

  def test_add_and_validate_workload_args(self):
    """Shared workload flags parse and are validated outside dry-run."""
    parser = argparse.ArgumentParser()
    utils.add_workload_args(parser, default_cleanup=True)
    args = parser.parse_args(["--dry-run", "--gcs-bucket", ""])
    self.assertTrue(args.cleanup)
    self.assertEqual(args.poll_interval, utils.DEFAULT_POLL_INTERVAL_SECONDS)
    utils.validate_workload_args(args)  # Dry-run skips required checks.

    args = parser.parse_args(["--no-cleanup", "--gcs-bucket", ""])
    self.assertFalse(args.cleanup)
    with self.assertRaisesRegex(ValueError, "--gcs-bucket"):
      utils.validate_workload_args(args)
    args = parser.parse_args(["--gcs-bucket", "b", "--docker-image", ""])
    with self.assertRaisesRegex(ValueError, "--docker-image"):
      utils.validate_workload_args(args)


if __name__ == "__main__":
  absltest.main()
