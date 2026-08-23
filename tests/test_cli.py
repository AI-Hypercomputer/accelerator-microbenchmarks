"""Unit tests for the TPUMS CLI (cli.py)."""

import io
import json
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks import cli


class TestCli(absltest.TestCase):
  """Unit tests for cli.py."""

  def setUp(self):
    super().setUp()
    self.parser = cli.create_parser()

  def test_help_menu(self):
    """Verifies that top-level help menu renders correctly."""
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
      cli.main([])
      output = fake_out.getvalue()
      self.assertIn("usage: tpums", output)
      self.assertIn("platform", output)
      self.assertIn("benchmark", output)

  def test_platform_describe_not_implemented(self):
    """Verifies that `tpums platform describe` raises NotImplementedError."""
    with self.assertRaises(NotImplementedError):
      cli.main(["platform", "describe"])

  def test_benchmark_list(self):
    """Verifies that `tpums benchmark list` outputs JSON list of tasks."""
    with mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
      cli.main(["benchmark", "list"])
      output = fake_out.getvalue()
      data = json.loads(output)
      self.assertIsInstance(data, list)
      tasks = [item["task"] for item in data]
      self.assertIn("gemm", tasks)
      self.assertIn("hbm", tasks)
      self.assertIn("all_reduce", tasks)
      self.assertIn("device_to_device", tasks)

  @mock.patch("accelerator_microbenchmarks.core.runner.run_benchmarks")
  def test_benchmark_run_hbm(self, mock_run_benchmarks):
    """Verifies that `tpums benchmark run hbm` parses args and calls runner."""
    cli.main(["benchmark", "run", "hbm", "--size", "134217728", "--dtype", "bfloat16"])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    tasks = kwargs["tasks"]
    self.assertEqual(len(tasks), 1)
    task_name, task_config = tasks[0]
    self.assertEqual(task_name, "hbm")
    self.assertEqual(task_config.size, 134217728)
    self.assertEqual(task_config.dtype, "bfloat16")

  @mock.patch("accelerator_microbenchmarks.core.runner.run_benchmarks")
  def test_benchmark_run_gemm(self, mock_run_benchmarks):
    """Verifies that `tpums benchmark run gemm` parses args and calls runner."""
    cli.main(["benchmark", "run", "gemm", "-m", "1024", "-n", "512", "-k", "256"])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    tasks = kwargs["tasks"]
    self.assertEqual(len(tasks), 1)
    task_name, task_config = tasks[0]
    self.assertEqual(task_name, "gemm")
    self.assertEqual(task_config.m, 1024)
    self.assertEqual(task_config.n, 512)
    self.assertEqual(task_config.k, 256)

  @mock.patch("accelerator_microbenchmarks.core.runner.run_benchmarks")
  def test_benchmark_run_config(self, mock_run_benchmarks):
    """Verifies that `tpums benchmark run-config` calls runner.run_benchmarks."""
    config_path = (
        "third_party/py/accelerator_microbenchmarks/configs/7x/2x2x1/hbm_bandwidth.yaml"
    )
    cli.main(
        ["benchmark", "run-config", config_path, "--output_dir", "test_out"]
    )
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    self.assertEqual(kwargs["output_dir"], "test_out")
    self.assertEqual(kwargs["config_path"], config_path)
    self.assertNotEmpty(kwargs["tasks"])

  def test_invalid_command_exits(self):
    """Verifies that invalid subcommands raise SystemExit."""
    with self.assertRaises(SystemExit):
      cli.main(["invalid_command"])


if __name__ == "__main__":
  absltest.main()
