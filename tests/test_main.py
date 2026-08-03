"""Unit tests for main.py."""

import os
import tempfile

from absl.testing import absltest
from accelerator_microbenchmarks import main
from accelerator_microbenchmarks.benchmarks import matmul
from accelerator_microbenchmarks.core import base
import pandas as pd
import yaml


class TestMainHelpers(absltest.TestCase):
  """Unit tests for testing XLA flag loading logic in main.py."""

  def setUp(self):
    super().setUp()
    self.orig_environ = dict(os.environ)
    # Clear relevant env vars to ensure clean test state
    os.environ.pop("LIBTPU_INIT_ARGS", None)
    self.temp_dir = tempfile.TemporaryDirectory()
    self.flags_file_path = os.path.join(
        self.temp_dir.name, "test_op_flags.yaml"
    )

    # Create a dummy op_flags.yaml
    self.test_flags = {
        "test_op_list": [
            "--flag_a=true",
            "--flag_b=123",
        ],
        "test_op_dict": {
            "flags": ["--flag_c=false"],
            "env": {
                "SOME_ENV_VAR": "some_value",
                "ANOTHER_VAR": "456",
            },
        },
        "psum": [
            "--mapped_flag=true",
        ],
    }
    with open(self.flags_file_path, "w") as f:
      yaml.dump(self.test_flags, f)

  def tearDown(self):
    # Restore original environment
    os.environ.clear()
    os.environ.update(self.orig_environ)
    self.temp_dir.cleanup()
    super().tearDown()

  def test_set_xla_flags_no_configs(self):
    main.set_xla_flags([], self.flags_file_path)
    self.assertNotIn("LIBTPU_INIT_ARGS", os.environ)

  def test_set_xla_flags_no_name(self):
    main.set_xla_flags([{"matrix_dim": 512}], self.flags_file_path)
    self.assertNotIn("LIBTPU_INIT_ARGS", os.environ)

  def test_set_xla_flags_list_config(self):
    main.set_xla_flags([{"name": "test_op_list"}], self.flags_file_path)
    self.assertEqual(
        os.environ.get("LIBTPU_INIT_ARGS"), "--flag_a=true --flag_b=123"
    )

  def test_set_xla_flags_dict_config(self):
    main.set_xla_flags([{"name": "test_op_dict"}], self.flags_file_path)
    self.assertEqual(os.environ.get("LIBTPU_INIT_ARGS"), "--flag_c=false")
    self.assertEqual(os.environ.get("SOME_ENV_VAR"), "some_value")
    self.assertEqual(os.environ.get("ANOTHER_VAR"), "456")

  def test_set_xla_flags_mapped_name(self):
    # all_reduce_sum should map to psum
    main.set_xla_flags([{"name": "all_reduce_sum"}], self.flags_file_path)
    self.assertEqual(os.environ.get("LIBTPU_INIT_ARGS"), "--mapped_flag=true")

  def test_set_xla_flags_unmatched_name(self):
    main.set_xla_flags([{"name": "unknown_op"}], self.flags_file_path)
    self.assertNotIn("LIBTPU_INIT_ARGS", os.environ)

  def test_set_xla_flags_missing_file(self):
    main.set_xla_flags([{"name": "test_op_list"}], "non_existent_file.yaml")
    self.assertNotIn("LIBTPU_INIT_ARGS", os.environ)

  def test_set_xla_flags_host_to_device(self):
    """Verifies that this fix does not break the original google3 path."""
    main.set_xla_flags([{"name": "host_to_device"}], None)
    init_args = os.environ.get("LIBTPU_INIT_ARGS")
    self.assertIsNotNone(init_args)
    self.assertIn("--xla_tpu_dvfs_p_state=7", init_args)

  def test_set_xla_flags_gemm_generalized(self):
    """Verifies that this fix does not break the original google3 path."""
    main.set_xla_flags([{"name": "all_reduce_sum"}], None)
    init_args = os.environ.get("LIBTPU_INIT_ARGS")
    self.assertIsNotNone(init_args)
    self.assertIn("--xla_jf_debug_level=3", init_args)

  def test_set_xla_flags_default_path_google3(self):
    """Verifies that this fix does not break the original google3 path."""
    main.set_xla_flags([{"name": "gemm_generalized"}], None)
    init_args = os.environ.get("LIBTPU_INIT_ARGS")
    self.assertIsNotNone(init_args)
    self.assertIn("--xla_tpu_vmem_scavenging_mode=NONE", init_args)

  def test_set_xla_flags_copybara_path(self):
    """Verifies that set_xla_flags works with the path used post-copybara."""
    import sys
    sys.stderr.write(f"\n--- CWD: {os.getcwd()}\n")
    sys.stderr.write(f"--- test_main.__file__: {__file__}\n")
    sys.stderr.write(f"--- main.__file__: {main.__file__}\n")
    copybara_flags_path = os.path.join(
        os.path.dirname(main.__file__), "op_flags.yaml"
    )
    sys.stderr.write(f"--- copybara_flags_path: {copybara_flags_path}\n")
    main.set_xla_flags([{"name": "all_reduce_sum"}], copybara_flags_path)
    init_args = os.environ.get("LIBTPU_INIT_ARGS")
    self.assertIsNotNone(init_args)
    self.assertIn("--xla_jf_debug_level=3", init_args)

  def test_save_output_throughput_not_overwritten(self):
    metadata = base.BenchmarkMetadata(
        benchmark_name="DummyBenchmark",
        test_name="DummyTest",
        start_time="now",
        end_time="then",
        params={"param1": "val1"},
        device_info={"device": "tpu"}
    )
    metrics = {
        "avg_ms": 10.0,
        "bandwidth_gb_s": 100.0,
        "throughput": 0.0,
    }
    result = base.BenchmarkResult(
        metadata=metadata,
        metrics=metrics,
        raw_times_ms=[10.0]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      main.save_output([result], tmpdir)

      csv_path = os.path.join(tmpdir, "summary.csv")
      self.assertTrue(os.path.exists(csv_path))

      df = pd.read_csv(csv_path)
      self.assertEqual(df["throughput"].iloc[0], 100.0)

  def test_mutually_exclusive_config_and_cli(self):
    main.FLAGS.config = None
    main.FLAGS.run = None
    with self.assertRaisesRegex(
        ValueError,
        "Cannot specify both --config \\(YAML\\) and --run \\(CLI\\)",
    ):
      main.parse_args(["main.py", "--run=gemm_generalized", "--config=my.yaml"])

  def test_only_config_allowed(self):
    main.FLAGS.config = None
    main.FLAGS.run = None
    remaining = main.parse_args(["main.py", "--config=my.yaml"])
    self.assertEqual(main.FLAGS.config, "my.yaml")
    self.assertEqual(main.FLAGS.run, None)
    self.assertEqual(remaining, ["main.py"])

  def test_only_cli_allowed(self):
    main.FLAGS.config = None
    main.FLAGS.run = None
    remaining = main.parse_args(
        ["main.py", "--run=gemm_generalized", "--m=1024", "--n=512"]
    )

    self.assertIsNone(main.FLAGS.config)
    self.assertEqual(main.FLAGS.run, "gemm_generalized")
    self.assertEqual(remaining, ["main.py", "--m=1024", "--n=512"])

    parsed_config = main._parse_benchmark_cli_args(
        "gemm_generalized", remaining[1:]
    )
    expected_config = matmul.GemmParams(m=1024, k=1024, n=512)
    self.assertEqual(parsed_config, expected_config)

  def test_cli_strict_parsing_fails_on_unknown_arg(self):
    main.FLAGS.config = None
    main.FLAGS.run = None
    remaining = main.parse_args(
        ["main.py", "--run=gemm_generalized", "--unknown_flag=123"]
    )

    with self.assertRaises(SystemExit) as cm:
      main._parse_benchmark_cli_args("gemm_generalized", remaining[1:])
    self.assertEqual(cm.exception.code, 2)

  def test_yaml_strict_parsing_fails_on_unknown_arg(self):
    gemm_cls = main.registry.benchmark_registry.get_benchmark(
        "gemm_generalized"
    )
    config_cls = gemm_cls.Config

    # Simulate a YAML config with an unknown key
    cfg = {"m": 128, "n": 128, "k": 128, "typo_batch_size": 16}

    with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
      config_cls(**cfg)


if __name__ == "__main__":
  absltest.main()
