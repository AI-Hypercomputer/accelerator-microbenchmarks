"""Unit tests for benchmark_loader.py."""

from absl.testing import absltest
from accelerator_microbenchmarks.benchmarks import benchmark_loader
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import registry


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
            "hbm",
            "hbm_bandwidth",
            "attention_flashed",
            "all_reduce",
            "all_to_all",
            "reduce_scatter",
            "all_gather",
            "transformer_layer_moe",
            "swiglu",
            "rmsnorm",
            "rope",
            "quantization",
            "simple_add",
            "host_to_device",
            "device_to_host",
            "device_to_device",
        ],
    )

  def test_primary_stable_tasks(self):
    """Verify that the 9 primary stable tasks are returned by default."""
    primary_tasks = registry.benchmark_registry.list_benchmark_names(
        include_experimental=False, include_aliases=False
    )
    self.assertCountEqual(
        primary_tasks,
        [
            "gemm",
            "hbm",
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
            "device_to_device",
            "host_to_device",
            "device_to_host",
        ],
    )
    for task_name in primary_tasks:
      bench_cls = registry.benchmark_registry.get_benchmark(task_name)
      self.assertIsNotNone(bench_cls)

  def test_all_benchmarks_have_valid_config(self):
    """Verify that every registered benchmark defines a Config dataclass."""
    for name, bench_cls in registry.benchmark_registry.get_all().items():
      self.assertTrue(
          hasattr(bench_cls, "Config"),
          f"Benchmark '{name}' ({bench_cls}) is missing 'Config' attribute.",
      )
      self.assertTrue(
          issubclass(bench_cls.Config, base.BaseBenchmarkParams),
          f"Benchmark '{name}' Config ({bench_cls.Config}) must subclass BaseBenchmarkParams.",
      )


if __name__ == "__main__":
  absltest.main()
