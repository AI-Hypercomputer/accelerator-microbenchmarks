"""Unit tests for the TPUMS CLI (cli.py)."""

import io
import json
import sys
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks import cli
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import runner


class TestCli(absltest.TestCase):
  """Unit tests for cli.py."""

  def setUp(self):
    super().setUp()
    self.parser = cli.create_parser()

  def test_help_menu(self):
    """Verifies that top-level help menu renders correctly."""
    with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
      cli.run([])
      output = fake_out.getvalue()
      self.assertIn("usage: tpums", output)
      self.assertIn("platform", output)
      self.assertIn("benchmark", output)

  def test_platform_describe(self):
    """Verifies that `tpums platform describe` outputs valid platform JSON."""
    with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
      cli.run(["platform", "describe"])
      output = fake_out.getvalue()
      data = json.loads(output)
      self.assertIsInstance(data, dict)
      expected_keys = {
          "tpu_type",
          "topology",
          "total_devices",
          "local_devices",
          "process_count",
          "process_index",
          "python_version",
          "jax_version",
          "jaxlib_version",
          "libtpu_version",
      }
      self.assertEqual(set(data.keys()), expected_keys)
      self.assertIsInstance(data["tpu_type"], str)
      self.assertIsInstance(data["topology"], str)
      self.assertIsInstance(data["total_devices"], int)
      self.assertIsInstance(data["local_devices"], int)
      self.assertIsInstance(data["process_count"], int)
      self.assertIsInstance(data["process_index"], int)
      self.assertIsInstance(data["python_version"], str)
      self.assertIsInstance(data["jax_version"], str)
      self.assertIsInstance(data["jaxlib_version"], str)
      self.assertIsInstance(data["libtpu_version"], str)

  def test_platform_describe_mocked(self):
    """Verifies that `tpums platform describe` outputs mock return value."""
    mock_desc = {
        "tpu_type": "TPU v7x",
        "topology": "2x2x1",
        "total_devices": 4,
        "local_devices": 4,
        "process_count": 1,
        "process_index": 0,
        "python_version": "3.11.0",
        "jax_version": "0.4.30",
        "jaxlib_version": "0.4.30",
        "libtpu_version": "0.1.dev20240101",
    }
    with mock.patch.object(
        platform,
        "get_platform_description",
        return_value=mock_desc,
    ):
      with mock.patch.object(
          sys, "stdout", new=io.StringIO()
      ) as fake_out, mock.patch.object(
          sys, "stderr", new=io.StringIO()
      ) as fake_err:
        cli.run(["platform", "describe"])
        output = fake_out.getvalue()
        err_output = fake_err.getvalue()
        data = json.loads(output)
        self.assertEqual(data, mock_desc)
        self.assertNotIn("WARNING: Running in non-TPU", err_output)

  def test_platform_describe_cpu_warning(self):
    """Verifies that `tpums platform describe` prints warning on CPU environment."""
    mock_desc = {
        "tpu_type": "none",
        "topology": "none",
        "total_devices": 1,
        "local_devices": 1,
        "process_count": 1,
        "process_index": 0,
        "python_version": "3.11.0",
        "jax_version": "0.4.30",
        "jaxlib_version": "0.4.30",
        "libtpu_version": "unknown",
    }
    with mock.patch.object(
        platform,
        "get_platform_description",
        return_value=mock_desc,
    ):
      with mock.patch.object(
          sys, "stdout", new=io.StringIO()
      ) as fake_out, mock.patch.object(
          sys, "stderr", new=io.StringIO()
      ) as fake_err:
        cli.run(["platform", "describe"])
        output = fake_out.getvalue()
        err_output = fake_err.getvalue()
        data = json.loads(output)
        self.assertEqual(data, mock_desc)
        self.assertIn(
            "WARNING: Running in non-TPU (CPU) environment. No TPU devices"
            " detected.",
            err_output,
        )

  def test_platform_describe_runtime_error_exits(self):
    """Verifies that `tpums platform describe` outputs to stderr and exits on RuntimeError."""
    with mock.patch.object(
        platform,
        "get_platform_description",
        side_effect=RuntimeError(
            "TPU runtime environment is not properly initialized"
        ),
    ):
      with mock.patch.object(sys, "stderr", new=io.StringIO()) as fake_err:
        with self.assertRaises(SystemExit) as cm:
          cli.run(["platform", "describe"])
        self.assertEqual(cm.exception.code, 1)
        self.assertIn(
            "Error: TPU runtime environment is not properly initialized",
            fake_err.getvalue(),
        )

  def test_benchmark_list(self):
    """Verifies that `tpums benchmark list` outputs JSON list of tasks."""
    with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
      cli.run(["benchmark", "list"])
      output = fake_out.getvalue()
      data = json.loads(output)
      self.assertIsInstance(data, list)
      tasks = [item["task"] for item in data]
      self.assertIn("gemm", tasks)
      self.assertIn("hbm", tasks)
      self.assertIn("all_reduce", tasks)
      self.assertIn("device_to_device", tasks)

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_hbm(self, mock_run_benchmarks):
    """Verifies that `tpums benchmark run hbm` parses args and calls runner."""
    cli.run(["benchmark", "run", "hbm", "--size", "134217728", "--dtype", "bfloat16"])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    tasks = kwargs["tasks"]
    self.assertEqual(len(tasks), 1)
    task_name, task_config = tasks[0]
    self.assertEqual(task_name, "hbm")
    self.assertEqual(task_config.size, 134217728)
    self.assertEqual(task_config.dtype, "bfloat16")

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_gemm(self, mock_run_benchmarks):
    """Verifies that `tpums benchmark run gemm` parses args and calls runner."""
    cli.run(["benchmark", "run", "gemm", "-m", "1024", "-n", "512", "-k", "256"])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    tasks = kwargs["tasks"]
    self.assertEqual(len(tasks), 1)
    task_name, task_config = tasks[0]
    self.assertEqual(task_name, "gemm")
    self.assertEqual(task_config.m, 1024)
    self.assertEqual(task_config.n, 512)
    self.assertEqual(task_config.k, 256)

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_config(self, mock_run_benchmarks):
    """Verifies that `tpums benchmark run-config` calls runner.run_benchmarks."""
    config_path = (
        "third_party/py/accelerator_microbenchmarks/configs/7x/2x2x1/hbm_bandwidth.yaml"
    )
    cli.run(
        ["benchmark", "run-config", config_path, "--output_dir", "test_out"]
    )
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    self.assertEqual(kwargs["output_dir"], "test_out")
    self.assertEqual(kwargs["config_path"], config_path)
    self.assertNotEmpty(kwargs["tasks"])

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_config_with_unknown_flags(self, mock_run_benchmarks):
    """Verifies that unknown flags (e.g. Borg infrastructure flags) are ignored."""
    config_path = (
        "third_party/py/accelerator_microbenchmarks/configs/7x/2x2x1/hbm_bandwidth.yaml"
    )
    # Trailing unknown flags
    cli.run([
        "benchmark",
        "run-config",
        config_path,
        "--output_dir",
        "test_out",
        "--deepsea_wrap=1",
        "--unknown_flag=foo",
    ])
    mock_run_benchmarks.assert_called_once()

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_config_with_xla_flags_file_path(
      self, mock_run_benchmarks
  ):
    """Verifies that `benchmark run-config` passes xla_flags_file_path to runner."""
    config_path = (
        "third_party/py/accelerator_microbenchmarks/configs/7x/2x2x1/hbm_bandwidth.yaml"
    )
    cli.run([
        "benchmark",
        "run-config",
        config_path,
        "--xla_flags_file_path",
        "/tmp/custom_op_flags.yaml",
    ])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    self.assertEqual(
        kwargs["xla_flags_file_path"], "/tmp/custom_op_flags.yaml"
    )

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_with_common_flags(self, mock_run_benchmarks):
    """Verifies that `benchmark run` passes output_dir, profile_dir, and hw to runner."""
    cli.run([
        "benchmark",
        "run",
        "gemm",
        "--output_dir",
        "/tmp/custom_results",
        "--profile_dir",
        "/tmp/custom_profile",
        "--hw",
        "ironwood",
        "-m",
        "128",
        "-n",
        "128",
        "-k",
        "128",
    ])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    self.assertEqual(kwargs["output_dir"], "/tmp/custom_results")
    self.assertEqual(kwargs["xprof_dir"], "/tmp/custom_profile")
    self.assertEqual(kwargs["hw"], "ironwood")

  def test_google_flags_parser(self):
    """Verifies that _google_flags_parser strips Abseil flags and returns domain argv."""
    # 1. Leading and trailing Abseil flags
    raw_argv = [
        "tpums_tpu",
        "--alsologtostderr",
        "--v=1",
        "benchmark",
        "run-config",
        "some_config.yaml",
        "--output_dir=results",
    ]
    parsed_argv = cli._google_flags_parser(raw_argv)
    self.assertEqual(
        parsed_argv,
        ["benchmark", "run-config", "some_config.yaml", "--output_dir=results"],
    )

    # 2. Interspersed Borg/Abseil flags
    interspersed_argv = [
        "/usr/bin/tpums",
        "--alsologtostderr",
        "benchmark",
        "--v=2",
        "run-config",
        "some_config.yaml",
        "--hw=gfl_2x2x1",
    ]
    parsed_interspersed = cli._google_flags_parser(interspersed_argv)
    self.assertEqual(
        parsed_interspersed,
        ["benchmark", "run-config", "some_config.yaml", "--hw=gfl_2x2x1"],
    )

  @mock.patch.object(cli.app, "run")
  def test_main_invokes_app_run(self, mock_app_run):
    """Verifies that main delegates to absl.app.run with _google_flags_parser."""
    cli.main()
    mock_app_run.assert_called_once_with(
        cli.run, flags_parser=cli._google_flags_parser
    )

  def test_invalid_command_exits(self):
    """Verifies that invalid subcommands raise SystemExit."""
    with self.assertRaises(SystemExit):
      cli.run(["invalid_command"])


if __name__ == "__main__":
  absltest.main()
