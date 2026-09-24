"""Unit tests for the TPUMS CLI (cli.py)."""

import dataclasses
import io
import json
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks import cli
from accelerator_microbenchmarks.benchmarks import attention
from accelerator_microbenchmarks.benchmarks import collectives
from accelerator_microbenchmarks.benchmarks import device_to_device
from accelerator_microbenchmarks.benchmarks import hbm
from accelerator_microbenchmarks.benchmarks import matmul
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import runner
from accelerator_microbenchmarks.tests import test_report_utils


class TestCli(parameterized.TestCase):
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

  @mock.patch(
      "accelerator_microbenchmarks.core.platform.get_platform_info",
      return_value=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
  )
  def test_platform_describe(self, mock_platform_desc):
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
      self.assertEqual(data["tpu_type"], "tpu7x")
      self.assertEqual(data["topology"], "2x2x1")
      self.assertIsInstance(data["total_devices"], int)
      self.assertIsInstance(data["local_devices"], int)
      self.assertIsInstance(data["process_count"], int)
      self.assertIsInstance(data["process_index"], int)
      self.assertIsInstance(data["python_version"], str)
      self.assertIsInstance(data["jax_version"], str)
      self.assertIsInstance(data["jaxlib_version"], str)
      self.assertIsInstance(data["libtpu_version"], str)

  def test_platform_describe_runtime_error_exits(self):
    """Verifies that `tpums platform describe` outputs to stderr and exits on RuntimeError."""
    with mock.patch.object(
        platform,
        "get_platform_info",
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
    cli.run([
        "benchmark",
        "run",
        "hbm",
        "--num_elements",
        "134217728",
        "--dtype",
        "bfloat16",
    ])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    tasks = kwargs["tasks"]
    self.assertEqual(len(tasks), 1)
    task_name, task_config = tasks[0]
    self.assertEqual(task_name, "hbm")
    self.assertEqual(task_config.num_elements, 134217728)
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

  _BENCHMARK_RUN_ENUM_CASES = (
      tuple(
          (
              f"all_reduce_{op.name.lower()}",
              "all_reduce",
              "--reduce_op",
              op.value,
              "reduce_op",
              op,
          )
          for op in collectives.ReduceOp
      )
      + tuple(
          (
              f"hbm_{op.name.lower()}",
              "hbm",
              "--op_type",
              op.value,
              "op_type",
              op,
          )
          for op in hbm.HBMKernelOp
      )
      + tuple(
          (
              f"device_to_device_{direction.name.lower()}",
              "device_to_device",
              "--direction",
              direction.value,
              "direction",
              direction,
          )
          for direction in device_to_device.TransferDirection
      )
  )

  @parameterized.named_parameters(*_BENCHMARK_RUN_ENUM_CASES)
  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_existing_enum_types(
      self,
      task_name,
      flag_name,
      flag_val,
      attr_name,
      expected_enum_val,
      mock_run_benchmarks,
  ):
    """Verifies benchmark run accepts and coerces all existing enum flags.

    Checks that runner.run_benchmarks receives a task config with the
    correct enum type and value.
    """
    cli.run(["benchmark", "run", task_name, flag_name, flag_val])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    resolved_task, task_config = kwargs["tasks"][0]
    self.assertEqual(resolved_task, task_name)
    self.assertEqual(getattr(task_config, attr_name), expected_enum_val)
    self.assertIsInstance(
        getattr(task_config, attr_name), type(expected_enum_val)
    )

  @parameterized.named_parameters(
      ("all_reduce_invalid_op", "all_reduce", "--reduce_op", "invalid_op"),
      ("all_reduce_wrong_casing", "all_reduce", "--reduce_op", "SUM"),
      ("hbm_invalid_op", "hbm", "--op_type", "invalid_kernel"),
      ("hbm_wrong_casing", "hbm", "--op_type", "SCALE"),
      (
          "device_to_device_invalid_direction",
          "device_to_device",
          "--direction",
          "invalid_dir",
      ),
      (
          "device_to_device_wrong_casing",
          "device_to_device",
          "--direction",
          "BI",
      ),
  )
  def test_benchmark_run_invalid_enum_raises(
      self, task_name, flag_name, invalid_val
  ):
    """Verifies that non-canonical or invalid enum values are rejected."""
    with self.assertRaises(SystemExit):
      cli.run(["benchmark", "run", task_name, flag_name, invalid_val])

  def test_benchmark_run_gemm_help_excludes_dtype(self):
    """Verifies that `tpums benchmark run gemm --help` does not expose --dtype."""
    with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
      with self.assertRaises(SystemExit) as cm:
        cli.run(["benchmark", "run", "gemm", "--help"])
      self.assertEqual(cm.exception.code, 0)
      help_text = fake_out.getvalue()
      self.assertIn("--in_dtype", help_text)
      self.assertIn("--out_dtype", help_text)
      self.assertNotIn("--dtype ", help_text)

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_config(self, mock_run_benchmarks):
    """Verifies that `tpums benchmark run-config` calls runner.run_benchmarks."""
    fake_config = self.create_tempfile(content="""
benchmark:
  name: gemm
  params:
    warmup_tries: 1
    num_runs: 1
""")
    config_path = fake_config.full_path
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
    fake_config = self.create_tempfile(content="""
benchmark:
  name: gemm
  params:
    warmup_tries: 1
    num_runs: 1
""")
    config_path = fake_config.full_path
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
    fake_config = self.create_tempfile(content="""
benchmark:
  name: gemm
  params:
    warmup_tries: 1
    num_runs: 1
""")
    config_path = fake_config.full_path
    cli.run([
        "benchmark",
        "run-config",
        config_path,
        "--xla_flags_file_path",
        "/tmp/custom_op_flags.yaml",
    ])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    self.assertEqual(kwargs["xla_flags_file_path"], "/tmp/custom_op_flags.yaml")

  @parameterized.named_parameters(
      (
          "negative_warmup_tries",
          """
benchmark:
  name: gemm
  params:
    warmup_tries: -1
""",
          "warmup_tries must be >= 0",
      ),
      (
          "zero_matrix_dim",
          """
benchmark:
  name: all_reduce
  params:
    matrix_dim: 0
""",
          "matrix_dim must be >= 1",
      ),
  )
  def test_benchmark_run_config_out_of_bounds_validation_raises(
      self, config_yaml, expected_error_msg
  ):
    """Verifies out-of-bounds parameters in YAML config raise ValueError."""
    fake_config = self.create_tempfile(content=config_yaml)
    with self.assertRaisesRegex(ValueError, expected_error_msg):
      cli.run(["benchmark", "run-config", fake_config.full_path])

  @parameterized.named_parameters(
      (
          "all_reduce_invalid_op",
          """
benchmark:
  name: all_reduce
  params:
    reduce_op: invalid_op
""",
          "Invalid reduce_op 'invalid_op'",
      ),
      (
          "all_reduce_wrong_casing",
          """
benchmark:
  name: all_reduce
  params:
    reduce_op: SUM
""",
          "Invalid reduce_op 'SUM'",
      ),
      (
          "hbm_invalid_op",
          """
benchmark:
  name: hbm
  params:
    op_type: invalid_kernel
""",
          "Invalid op_type 'invalid_kernel'",
      ),
      (
          "hbm_wrong_casing",
          """
benchmark:
  name: hbm
  params:
    op_type: SCALE
""",
          "Invalid op_type 'SCALE'",
      ),
      (
          "device_to_device_invalid_direction",
          """
benchmark:
  name: device_to_device
  params:
    direction: invalid_direction
""",
          "Invalid direction 'invalid_direction'",
      ),
      (
          "device_to_device_wrong_casing",
          """
benchmark:
  name: device_to_device
  params:
    direction: BI
""",
          "Invalid direction 'BI'",
      ),
      (
          "attention_invalid_mode",
          """
benchmark:
  name: attention_flashed
  params:
    mode: invalid_mode
""",
          "Invalid mode 'invalid_mode'",
      ),
      (
          "attention_wrong_casing",
          """
benchmark:
  name: attention_flashed
  params:
    mode: FWD
""",
          "Invalid mode 'FWD'",
      ),
  )
  def test_benchmark_run_config_invalid_enum_raises(
      self, config_yaml, expected_error_msg
  ):
    """Verifies that invalid enum values in YAML config raise ValueError."""
    fake_config = self.create_tempfile(content=config_yaml)
    with self.assertRaisesRegex(ValueError, expected_error_msg):
      cli.run(["benchmark", "run-config", fake_config.full_path])

  _BENCHMARK_RUN_CONFIG_ENUM_CASES = (
      tuple(
          (
              f"all_reduce_{op.name.lower()}",
              f"""
benchmark:
  name: all_reduce
  params:
    reduce_op: {op.value}
""",
              "all_reduce",
              "reduce_op",
              op,
          )
          for op in collectives.ReduceOp
      )
      + tuple(
          (
              f"hbm_{op.name.lower()}",
              f"""
benchmark:
  name: hbm
  params:
    op_type: {op.value}
""",
              "hbm",
              "op_type",
              op,
          )
          for op in hbm.HBMKernelOp
      )
      + tuple(
          (
              f"device_to_device_{direction.name.lower()}",
              f"""
benchmark:
  name: device_to_device
  params:
    direction: {direction.value}
""",
              "device_to_device",
              "direction",
              direction,
          )
          for direction in device_to_device.TransferDirection
      )
      + tuple(
          (
              f"attention_flashed_{mode.name.lower()}",
              f"""
benchmark:
  name: attention_flashed
  params:
    mode: {mode.value}
""",
              "attention_flashed",
              "mode",
              mode,
          )
          for mode in attention.AttentionMode
      )
  )

  @parameterized.named_parameters(*_BENCHMARK_RUN_CONFIG_ENUM_CASES)
  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_config_existing_enum_types(
      self,
      config_yaml,
      expected_task_name,
      attr_name,
      expected_enum_val,
      mock_run_benchmarks,
  ):
    """Verifies benchmark run accepts and coerces all existing enum flags.

    Checks that runner.run_benchmarks receives a task config with the
    correct enum type and value.
    """
    fake_config = self.create_tempfile(content=config_yaml)
    cli.run(["benchmark", "run-config", fake_config.full_path])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    task_name, task_config = kwargs["tasks"][0]
    self.assertEqual(task_name, expected_task_name)
    self.assertEqual(getattr(task_config, attr_name), expected_enum_val)
    self.assertIsInstance(
        getattr(task_config, attr_name), type(expected_enum_val)
    )

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_with_common_flags(self, mock_run_benchmarks):
    """Verifies that `benchmark run` passes output_dir and primitive xprof flags to runner."""
    cli.run([
        "benchmark",
        "run",
        "gemm",
        "--output_dir",
        "/tmp/custom_results",
        "--xprof_dir",
        "/tmp/custom_profile",
        "--xprof_timing",
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
    self.assertTrue(kwargs["xprof_timing"])
    self.assertEqual(kwargs["xprof_dir"], "/tmp/custom_profile")

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_without_xprof_timing(self, mock_run_benchmarks):
    """Verifies that `benchmark run` without --xprof_timing passes default primitive flags."""
    cli.run([
        "benchmark",
        "run",
        "gemm",
        "-m",
        "128",
        "-n",
        "128",
        "-k",
        "128",
    ])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    self.assertFalse(kwargs["xprof_timing"])
    self.assertEqual(kwargs["xprof_dir"], "/tmp/tensorboard")

  @parameterized.named_parameters(
      (
          "yaml_xprof_true",
          """
benchmark:
  name: gemm
  xprof_timing: true
  params:
    warmup_tries: 1
    num_runs: 1
""",
          [],
          True,
      ),
      (
          "yaml_xprof_false",
          """
benchmark:
  name: gemm
  xprof_timing: false
  params:
    warmup_tries: 1
    num_runs: 1
""",
          [],
          False,
      ),
      (
          "yaml_xprof_omitted",
          """
benchmark:
  name: gemm
  params:
    warmup_tries: 1
    num_runs: 1
""",
          [],
          False,
      ),
      (
          "cli_flag_ignored",
          """
benchmark:
  name: gemm
  params:
    warmup_tries: 1
    num_runs: 1
""",
          ["--xprof_timing"],
          False,
      ),
  )
  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_config_xprof_timing(
      self,
      config_yaml,
      extra_flags,
      expected_xprof_timing,
      mock_run_benchmarks,
  ):
    """Verifies xprof_timing resolution for benchmark run-config."""
    fake_config = self.create_tempfile(content=config_yaml)
    cli.run(["benchmark", "run-config", fake_config.full_path] + extra_flags)
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    self.assertIs(kwargs["xprof_timing"], expected_xprof_timing)

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_xprof_timing_enabled(self, mock_run_benchmarks):
    """Verifies `benchmark run gemm --xprof_timing` passes xprof_timing=True."""
    cli.run(["benchmark", "run", "gemm", "--xprof_timing"])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    self.assertIs(kwargs["xprof_timing"], True)

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_xprof_timing_default(self, mock_run_benchmarks):
    """Verifies `benchmark run gemm` without flag passes xprof_timing=False."""
    cli.run(["benchmark", "run", "gemm"])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    self.assertIs(kwargs["xprof_timing"], False)

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

  def test_benchmark_run_help_lists_tasks(self):
    """Verifies that `tpums benchmark run --help` lists supported benchmark tasks."""
    with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
      with self.assertRaises(SystemExit) as cm:
        cli.run(["benchmark", "run", "--help"])
      self.assertEqual(cm.exception.code, 0)
      output = fake_out.getvalue()
      self.assertIn("Supported Tasks", output)
      self.assertIn("gemm", output)
      self.assertIn("all_reduce", output)
      self.assertIn("hbm", output)

  def test_benchmark_run_help_all_registered_tasks(self):
    """Verifies that `tpums benchmark run <task> --help` outputs parameter help for all tasks."""
    tasks = registry.benchmark_registry.list_benchmark_names(
        include_experimental=False, include_aliases=False
    )
    for task_name in tasks:
      with self.subTest(task=task_name):
        bench_cls = registry.benchmark_registry.get_benchmark(task_name)
        with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
          with self.assertRaises(SystemExit) as cm:
            cli.run(["benchmark", "run", task_name, "--help"])
          self.assertEqual(cm.exception.code, 0)
          output = fake_out.getvalue()
          self.assertIn(f"usage: tpums benchmark run {task_name}", output)
          self.assertIn(f"{bench_cls.Config.__name__} parameters:", output)
          for field in dataclasses.fields(bench_cls.Config):
            self.assertIn(f"--{field.name}", output)
            self.assertIn(
                "help",
                field.metadata,
                f"Field '{field.name}' in benchmark '{task_name}' is missing"
                " 'help' metadata.",
            )
            self.assertNotEmpty(
                field.metadata["help"],
                f"Field '{field.name}' in benchmark '{task_name}' has empty"
                " 'help' metadata.",
            )
            first_words = " ".join(field.metadata["help"].split()[:2])
            self.assertIn(
                first_words,
                output,
                f"Help text for parameter '{field.name}' in benchmark"
                f" '{task_name}' missing from help output.",
            )

  @parameterized.named_parameters(
      ("true", "true", True),
      ("false", "false", False),
      ("mixed_case", "True", True),
      ("equals_form", "=true", True),
  )
  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_boolean_flag(
      self, flag_value, expected, mock_run_benchmarks
  ):
    """Verifies boolean flags accept explicit true/false values."""
    if flag_value.startswith("="):
      argv = ["benchmark", "run", "gemm", f"--transpose_a{flag_value}"]
    else:
      argv = ["benchmark", "run", "gemm", "--transpose_a", flag_value]
    cli.run(argv)
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    _, task_config = kwargs["tasks"][0]
    self.assertIs(task_config.transpose_a, expected)

  @parameterized.named_parameters(
      ("invalid_token", ["--transpose_a", "maybe"]),
      # `yes` was accepted before this CL replaced simple_parsing; the narrower
      # true/false token set is intentional.
      ("legacy_yes_token", ["--transpose_a", "yes"]),
      # Bare `--transpose_a` no longer implies True; a value is now required.
      ("bare_flag", ["--transpose_a"]),
  )
  def test_benchmark_run_boolean_flag_rejected_values(self, extra_argv):
    """Verifies unsupported boolean values exit rather than silently pass."""
    with self.assertRaises(SystemExit):
      cli.run(["benchmark", "run", "gemm"] + extra_argv)

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_boolean_flag_defaults(self, mock_run_benchmarks):
    """Verifies omitted boolean flags keep their scalar dataclass defaults."""
    cli.run(["benchmark", "run", "gemm"])
    mock_run_benchmarks.assert_called_once()
    _, kwargs = mock_run_benchmarks.call_args
    _, task_config = kwargs["tasks"][0]
    self.assertIs(task_config.transpose_a, False)
    self.assertIs(task_config.transpose_b, False)

  def test_boolean_flag_accepts_multiple_values(self):
    """Verifies booleans parse into a list for parameter sweeps.

    `parse_cli_configs` expands multi-element lists into a Cartesian product
    sweep; this test verifies the underlying argparse tokenization directly.
    """
    parser = cli.create_parser()
    args = parser.parse_args(
        ["benchmark", "run", "gemm", "--transpose_a", "true", "false"]
    )
    self.assertEqual(args.transpose_a, [True, False])

  def test_benchmark_run_boolean_flag_help(self):
    """Verifies boolean flags advertise their value form and default."""
    with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
      with self.assertRaises(SystemExit):
        cli.run(["benchmark", "run", "gemm", "--help"])
      output = fake_out.getvalue()
      self.assertIn("--transpose_a {true,false}", output)
      self.assertIn("(default: False)", output)
      self.assertNotIn("--no-transpose_a", output)

  def test_invalid_command_exits(self):
    """Verifies that invalid subcommands raise SystemExit."""
    with self.assertRaises(SystemExit):
      cli.run(["invalid_command"])

  @parameterized.named_parameters(
      (
          "single_combination",
          ["benchmark", "run", "gemm", "-m", "512"],
          [("gemm", matmul.GemmParams(m=512))],
      ),
      (
          "gemm_int_and_bool_sweep",
          [
              "benchmark",
              "run",
              "gemm",
              "-m",
              "1024",
              "2048",
              "-n",
              "512",
              "1024",
              "--transpose_a",
              "true",
              "false",
          ],
          [
              ("gemm", matmul.GemmParams(m=1024, n=512, transpose_a=True)),
              ("gemm", matmul.GemmParams(m=1024, n=512, transpose_a=False)),
              ("gemm", matmul.GemmParams(m=1024, n=1024, transpose_a=True)),
              ("gemm", matmul.GemmParams(m=1024, n=1024, transpose_a=False)),
              ("gemm", matmul.GemmParams(m=2048, n=512, transpose_a=True)),
              ("gemm", matmul.GemmParams(m=2048, n=512, transpose_a=False)),
              ("gemm", matmul.GemmParams(m=2048, n=1024, transpose_a=True)),
              ("gemm", matmul.GemmParams(m=2048, n=1024, transpose_a=False)),
          ],
      ),
      (
          "all_reduce_int_and_enum_sweep",
          [
              "benchmark",
              "run",
              "all_reduce",
              "--reduce_op",
              "sum",
              "max",
              "--matrix_dim",
              "1024",
              "2048",
          ],
          [
              (
                  "all_reduce",
                  collectives.AllReduceParams(
                      matrix_dim=1024, reduce_op=collectives.ReduceOp.SUM
                  ),
              ),
              (
                  "all_reduce",
                  collectives.AllReduceParams(
                      matrix_dim=1024, reduce_op=collectives.ReduceOp.MAX
                  ),
              ),
              (
                  "all_reduce",
                  collectives.AllReduceParams(
                      matrix_dim=2048, reduce_op=collectives.ReduceOp.SUM
                  ),
              ),
              (
                  "all_reduce",
                  collectives.AllReduceParams(
                      matrix_dim=2048, reduce_op=collectives.ReduceOp.MAX
                  ),
              ),
          ],
      ),
      (
          "device_to_device_int_and_enum_sweep",
          [
              "benchmark",
              "run",
              "device_to_device",
              "--direction",
              "uni",
              "bi",
              "--data_size_mib",
              "512",
              "1024",
          ],
          [
              (
                  "device_to_device",
                  device_to_device.DeviceToDeviceParams(
                      data_size_mib=512,
                      direction=device_to_device.TransferDirection.UNI,
                  ),
              ),
              (
                  "device_to_device",
                  device_to_device.DeviceToDeviceParams(
                      data_size_mib=512,
                      direction=device_to_device.TransferDirection.BI,
                  ),
              ),
              (
                  "device_to_device",
                  device_to_device.DeviceToDeviceParams(
                      data_size_mib=1024,
                      direction=device_to_device.TransferDirection.UNI,
                  ),
              ),
              (
                  "device_to_device",
                  device_to_device.DeviceToDeviceParams(
                      data_size_mib=1024,
                      direction=device_to_device.TransferDirection.BI,
                  ),
              ),
          ],
      ),
      (
          "hbm_int_and_str_sweep",
          [
              "benchmark",
              "run",
              "hbm",
              "--num_elements",
              "1024",
              "2048",
              "--dtype",
              "bfloat16",
              "float32",
          ],
          [
              (
                  "hbm",
                  hbm.HBMBandwidthParams(dtype="bfloat16", num_elements=1024),
              ),
              (
                  "hbm",
                  hbm.HBMBandwidthParams(dtype="bfloat16", num_elements=2048),
              ),
              (
                  "hbm",
                  hbm.HBMBandwidthParams(dtype="float32", num_elements=1024),
              ),
              (
                  "hbm",
                  hbm.HBMBandwidthParams(dtype="float32", num_elements=2048),
              ),
          ],
      ),
  )
  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_parameter_sweep(
      self, argv, expected_tasks, mock_run_benchmarks
  ):
    """Verifies `benchmark run` expands single and multi-value flags."""
    cli.run(argv)
    mock_run_benchmarks.assert_called_once()
    self.assertEqual(
        mock_run_benchmarks.call_args.kwargs["tasks"], expected_tasks
    )

  @mock.patch.object(runner, "run_benchmarks")
  def test_benchmark_run_sweep_validates_bounds_on_all_combinations(
      self, mock_run_benchmarks
  ):
    """Verifies `benchmark run` raises ValueError if any swept value violates bounds."""
    with self.assertRaisesRegex(ValueError, "m must be >= 1"):
      cli.run(["benchmark", "run", "gemm", "-m", "1024", "0"])
    mock_run_benchmarks.assert_not_called()


if __name__ == "__main__":
  absltest.main()
