"""Unit tests for the benchmark execution runner (runner.py)."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import runner
import yaml


class TestRunner(absltest.TestCase):
  """Unit tests for runner.py."""

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
        "all_reduce": [
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
    with self.assertRaises(ValueError):
      runner.set_xla_flags([], self.flags_file_path)

  def test_set_xla_flags_no_name(self):
    with self.assertRaises(KeyError):
      runner.set_xla_flags([{"matrix_dim": 512}], self.flags_file_path)

  def test_set_xla_flags_multiple_benchmarks(self):
    with self.assertRaisesRegex(ValueError, "Multiple benchmarks in config"):
      runner.set_xla_flags(
          [{"name": "benchmark_1"}, {"name": "benchmark_2"}],
          self.flags_file_path,
      )

  def test_set_xla_flags_list_config(self):
    runner.set_xla_flags([{"name": "test_op_list"}], self.flags_file_path)
    self.assertEqual(
        os.environ.get("LIBTPU_INIT_ARGS"), "--flag_a=true --flag_b=123"
    )

  def test_set_xla_flags_dict_config(self):
    runner.set_xla_flags([{"name": "test_op_dict"}], self.flags_file_path)
    self.assertEqual(os.environ.get("LIBTPU_INIT_ARGS"), "--flag_c=false")
    self.assertEqual(os.environ.get("SOME_ENV_VAR"), "some_value")
    self.assertEqual(os.environ.get("ANOTHER_VAR"), "456")

  def test_set_xla_flags_direct_name(self):
    # all_reduce should directly match all_reduce flags
    runner.set_xla_flags([{"name": "all_reduce"}], self.flags_file_path)
    self.assertEqual(os.environ.get("LIBTPU_INIT_ARGS"), "--mapped_flag=true")

  def test_set_xla_flags_unmatched_name(self):
    runner.set_xla_flags([{"name": "unknown_op"}], self.flags_file_path)
    self.assertNotIn("LIBTPU_INIT_ARGS", os.environ)

  def test_set_xla_flags_missing_file(self):
    runner.set_xla_flags([{"name": "test_op_list"}], "non_existent_file.yaml")
    self.assertNotIn("LIBTPU_INIT_ARGS", os.environ)

  def test_set_xla_flags_host_to_device(self):
    """Verifies that this fix does not break the original google3 path."""
    runner.set_xla_flags([{"name": "host_to_device"}], None)
    init_args = os.environ.get("LIBTPU_INIT_ARGS")
    self.assertIsNotNone(init_args)
    self.assertIn("--xla_tpu_dvfs_p_state=7", init_args)

  def test_set_xla_flags_all_reduce(self):
    """Verifies that all_reduce maps to all_reduce flags in op_flags.yaml."""
    runner.set_xla_flags([{"name": "all_reduce"}], None)
    init_args = os.environ.get("LIBTPU_INIT_ARGS")
    self.assertIsNotNone(init_args)
    self.assertIn("--xla_jf_debug_level=3", init_args)

  def test_set_xla_flags_default_path_google3(self):
    """Verifies that this fix does not break the original google3 path."""
    runner.set_xla_flags([{"name": "gemm_generalized"}], None)
    init_args = os.environ.get("LIBTPU_INIT_ARGS")
    self.assertIsNotNone(init_args)
    self.assertIn("--xla_tpu_vmem_scavenging_mode=NONE", init_args)

  def test_set_xla_flags_copybara_path(self):
    """Verifies that set_xla_flags works with the path used post-copybara."""
    copybara_flags_path = os.path.join(
        os.path.dirname(runner.__file__), "..", "op_flags.yaml"
    )
    runner.set_xla_flags([{"name": "all_reduce"}], copybara_flags_path)
    init_args = os.environ.get("LIBTPU_INIT_ARGS")
    self.assertIsNotNone(init_args)
    self.assertIn("--xla_jf_debug_level=3", init_args)

  def test_init_jax_distributed(self):
    with mock.patch("jax.distributed.initialize") as mock_init:
      runner.init_jax_distributed()
      mock_init.assert_called_once()

  def test_run_benchmarks(self):
    """Verifies that run_benchmarks executes typed benchmark tasks."""
    dummy_params = base.BaseBenchmarkParams(warmup_tries=1, num_runs=1)
    with tempfile.TemporaryDirectory() as tmpdir:
      with mock.patch(
          "accelerator_microbenchmarks.benchmarks.benchmark_loader.load_all_benchmarks"
      ) as mock_load_all:
        with mock.patch.object(runner, "set_xla_flags") as mock_set_xla:
          with mock.patch.object(runner, "init_jax_distributed") as mock_init_jax:
            with mock.patch.object(runner, "export_to_mlcompass"):
              with mock.patch(
                  "accelerator_microbenchmarks.core.registry.benchmark_registry.get_benchmark"
              ) as mock_get_benchmark:
                mock_bench_instance = mock.MagicMock()
                mock_metadata = base.BenchmarkMetadata(
                    benchmark_name="dummy",
                    test_name="dummy_test",
                    start_time="now",
                    end_time="then",
                    params={},
                    device_info={},
                )
                mock_bench_instance.run.return_value = base.BenchmarkResult(
                    metadata=mock_metadata,
                    metrics={"avg_ms": 1.0},
                    raw_times_ms=[1.0],
                )
                mock_bench_instance.get_run_identifier.return_value = "run_1"
                mock_bench_cls = mock.MagicMock(
                    return_value=mock_bench_instance,
                    __name__="MockBenchmark",
                )
                mock_get_benchmark.return_value = mock_bench_cls

                results = runner.run_benchmarks(
                    tasks=[("dummy", dummy_params)],
                    output_dir=tmpdir,
                )
                self.assertEqual(len(results), 1)
                self.assertTrue(
                    os.path.exists(os.path.join(tmpdir, "summary.csv"))
                )
                self.assertTrue(
                    os.path.exists(os.path.join(tmpdir, "detailed.json"))
                )
                mock_load_all.assert_called_once()
                mock_set_xla.assert_called_once()
                mock_init_jax.assert_called_once()
                mock_bench_instance.run.assert_called_once()

  def test_run_benchmarks_with_system_and_xprof(self):
    dummy_params = base.BaseBenchmarkParams(
        warmup_tries=1,
        num_runs=1,
        system="v6e_4x4",
        xprof_dir="/tmp/custom_xprof",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      with mock.patch.object(runner, "set_xla_flags"):
        with mock.patch.object(runner, "init_jax_distributed"):
          with mock.patch.object(runner, "export_to_mlcompass") as mock_export:
            with mock.patch(
                "accelerator_microbenchmarks.core.registry.benchmark_registry.get_benchmark"
            ) as mock_get_benchmark:
              mock_bench_instance = mock.MagicMock()
              mock_metadata = base.BenchmarkMetadata(
                  benchmark_name="dummy",
                  test_name="dummy_test",
                  start_time="now",
                  end_time="then",
                  params={},
                  device_info={},
              )
              mock_bench_instance.run.return_value = base.BenchmarkResult(
                  metadata=mock_metadata,
                  metrics={"avg_ms": 1.0},
                  raw_times_ms=[1.0],
              )
              mock_bench_instance.get_run_identifier.return_value = "run_1"
              mock_bench_cls = mock.MagicMock(
                  return_value=mock_bench_instance,
                  __name__="MockBenchmark",
              )
              mock_get_benchmark.return_value = mock_bench_cls

              results = runner.run_benchmarks(
                  tasks=[("dummy", dummy_params)],
                  output_dir=tmpdir,
                  xprof_dir="/tmp/fallback_xprof",
              )
              self.assertEqual(len(results), 1)
              self.assertIsNotNone(dummy_params.hardware_stats)
              self.assertEqual(dummy_params.xprof_dir, "/tmp/custom_xprof")
              mock_export.assert_called_once()

  def test_run_benchmarks_handles_exception(self):
    dummy_params = base.BaseBenchmarkParams(warmup_tries=1, num_runs=1)
    with tempfile.TemporaryDirectory() as tmpdir:
      with mock.patch.object(runner, "set_xla_flags"):
        with mock.patch.object(runner, "init_jax_distributed"):
          with mock.patch(
              "accelerator_microbenchmarks.core.registry.benchmark_registry.get_benchmark"
          ) as mock_get_benchmark:
            mock_bench_instance = mock.MagicMock()
            mock_bench_instance.run.side_effect = RuntimeError("Kernel failed")
            mock_bench_cls = mock.MagicMock(
                return_value=mock_bench_instance,
                __name__="MockBenchmark",
            )
            mock_get_benchmark.return_value = mock_bench_cls

            results = runner.run_benchmarks(
                tasks=[("dummy", dummy_params)],
                output_dir=tmpdir,
            )
            self.assertEqual(results, [])

  def test_run_benchmarks_print_table(self):
    """Verifies that run_benchmarks passes print_table to report.report_results."""
    dummy_params = base.BaseBenchmarkParams(warmup_tries=1, num_runs=1)
    with tempfile.TemporaryDirectory() as tmpdir:
      with mock.patch.object(runner, "set_xla_flags"):
        with mock.patch.object(runner, "init_jax_distributed"):
          with mock.patch.object(runner, "export_to_mlcompass"):
            with mock.patch(
                "accelerator_microbenchmarks.core.registry.benchmark_registry.get_benchmark"
            ) as mock_get_benchmark:
              with mock.patch.object(
                  runner.report, "report_results"
              ) as mock_report_results:
                mock_bench_instance = mock.MagicMock()
                mock_metadata = base.BenchmarkMetadata(
                    benchmark_name="dummy",
                    test_name="dummy_test",
                    start_time="now",
                    end_time="then",
                    params={},
                    device_info={},
                )
                mock_result = base.BenchmarkResult(
                    metadata=mock_metadata,
                    metrics={"avg_ms": 1.0},
                    raw_times_ms=[1.0],
                )
                mock_bench_instance.run.return_value = mock_result
                mock_bench_instance.get_run_identifier.return_value = "run_1"
                mock_bench_cls = mock.MagicMock(
                    return_value=mock_bench_instance,
                    __name__="MockBenchmark",
                )
                mock_get_benchmark.return_value = mock_bench_cls

                # Test print_table=True
                runner.run_benchmarks(
                    tasks=[("dummy", dummy_params)],
                    output_dir=tmpdir,
                    print_table=True,
                )
                mock_report_results.assert_called_once_with(
                    [mock_result],
                    output_dir=tmpdir,
                    print_table=True,
                )

                mock_report_results.reset_mock()

                # Test print_table=False
                runner.run_benchmarks(
                    tasks=[("dummy", dummy_params)],
                    output_dir=tmpdir,
                    print_table=False,
                )
                mock_report_results.assert_called_once_with(
                    [mock_result],
                    output_dir=tmpdir,
                    print_table=False,
                )


if __name__ == "__main__":
  absltest.main()
