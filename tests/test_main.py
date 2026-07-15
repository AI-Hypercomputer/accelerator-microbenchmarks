"""Unit tests for main.py."""

import os
import tempfile

from absl.testing import absltest
from accelerator_microbenchmarks import main
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


if __name__ == "__main__":
  absltest.main()
