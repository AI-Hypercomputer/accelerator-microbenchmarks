"""Unit tests for benchmark_loader.py."""

import dataclasses

from absl.testing import absltest
from accelerator_microbenchmarks.benchmarks import benchmark_loader
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import config
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import executor
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import system


class BenchmarkLoaderTest(absltest.TestCase):
  """Unit tests for benchmark_loader.py."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    benchmark_loader.load_all_benchmarks()

  def test_load_all_benchmarks_expected_benchmarks(self):
    """Verify that load_all_benchmarks loads expected benchmarks into registry."""
    all_benchmarks = registry.benchmark_registry.list_benchmark_names(
        include_experimental=True, include_aliases=True
    )

    self.assertCountEqual(
        all_benchmarks,
        [
            "gemm",
            "gemm_generalized",
            "gemm_throttling",
            "gemm_pulse",
            "gemm_chained",
            "gemm_quantized",
            "hbm",
            "hbm_bandwidth",
            "attention_flashed",
            "all_reduce",
            "dcn_all_reduce",
            "dcn_all_to_all",
            "all_to_all",
            "reduce_scatter",
            "all_gather",
            "transformer_layer_moe",
            "swiglu",
            "rmsnorm",
            "sigmoid",
            "layernorm",
            "softmax",
            "rope",
            "quantization",
            "quantize_transpose",
            "simple_add",
            "host_to_device",
            "device_to_host",
            "device_to_device",
        ],
    )

  def test_primary_stable_tasks(self):
    """Verify that the primary stable tasks are returned by default."""
    primary_tasks = registry.benchmark_registry.list_benchmark_names(
        include_experimental=False, include_aliases=False
    )
    self.assertCountEqual(
        primary_tasks,
        [
            "gemm",
            "hbm",
            "all_reduce",
            "dcn_all_reduce",
            "dcn_all_to_all",
            "all_gather",
            "all_to_all",
            "device_to_device",
            "host_to_device",
            "device_to_host",
        ],
    )
    for task_name in primary_tasks:
      bench_cls = registry.benchmark_registry.get_benchmark(task_name)
      self.assertIsNotNone(bench_cls)

  def test_experimental_benchmarks(self):
    """Verify that experimental benchmarks are marked as experimental."""
    self.assertTrue(
        registry.benchmark_registry.is_experimental("reduce_scatter")
    )
    self.assertTrue(
        registry.benchmark_registry.is_experimental("gemm_throttling")
    )
    self.assertTrue(registry.benchmark_registry.is_experimental("gemm_chained"))
    self.assertTrue(registry.benchmark_registry.is_experimental("gemm_pulse"))
    self.assertTrue(
        registry.benchmark_registry.is_experimental("gemm_quantized")
    )

  def test_all_benchmarks_have_valid_config(self):
    """Verify that every registered benchmark defines a Config dataclass."""
    for name, bench_cls in registry.benchmark_registry.get_all().items():
      self.assertTrue(
          hasattr(bench_cls, "Config"),
          f"Benchmark '{name}' ({bench_cls}) is missing 'Config' attribute.",
      )
      self.assertTrue(
          issubclass(bench_cls.Config, base.BaseBenchmarkParams),
          f"Benchmark '{name}' Config ({bench_cls.Config}) must subclass"
          " BaseBenchmarkParams.",
      )
      for f in dataclasses.fields(bench_cls.Config):
        self.assertTrue(
            f.metadata.get("help", "").strip(),
            f"Field '{f.name}' in benchmark '{name}' has missing/empty help.",
        )

  def test_all_benchmarks_declare_supported_device_timing_aggregations(self):
    """Verify every registered benchmark declares expected timing aggregations."""
    single_mode_benchmarks = {
        "hbm",
        "hbm_bandwidth",
        "host_to_device",
        "device_to_host",
        "device_to_device",
    }
    single_mode_only = frozenset(
        {constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST}
    )
    all_modes = frozenset(constants.XprofDeviceTimingAggregation)

    for name, bench_cls in registry.benchmark_registry.get_all().items():
      with self.subTest(benchmark=name):
        # SoakingExecutionParamsMixin requires min_duration_s > 0.0.
        bench_config = (
            bench_cls.Config(min_duration_s=1.0)
            if issubclass(bench_cls.Config, config.SoakingExecutionParamsMixin)
            else bench_cls.Config()
        )
        bm = bench_cls(
            config=bench_config,
            hardware_spec=system.TPU7X_HARDWARE_SPEC,
        )
        if issubclass(bench_cls.Config, config.PulseExecutionParamsMixin):
          expected_executor = executor.PulseRunExecutor
        elif issubclass(bench_cls.Config, config.SoakingExecutionParamsMixin):
          expected_executor = executor.SoakingRunExecutor
        else:
          expected_executor = executor.BaseRunExecutor
        self.assertIs(type(bm.executor), expected_executor)
        expected = (
            single_mode_only if name in single_mode_benchmarks else all_modes
        )
        self.assertEqual(bm.supported_device_timing_aggregations, expected)

  def test_benchmarks_without_ici(self):
    """Verify only the expected benchmarks opt out of `requires_ici`."""
    no_ici = {
        name
        for name, bench_cls in registry.benchmark_registry.get_all().items()
        if not bench_cls.requires_ici
    }
    self.assertEqual(
        no_ici,
        {
            "gemm",
            "gemm_generalized",
            "gemm_throttling",
            "gemm_pulse",
            "gemm_chained",
            "gemm_quantized",
            "hbm",
            "hbm_bandwidth",
            "host_to_device",
            "device_to_host",
        },
    )


if __name__ == "__main__":
  absltest.main()
