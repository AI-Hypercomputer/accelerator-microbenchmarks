"""Unit tests for benchmark_loader.py."""

from absl.testing import absltest
from accelerator_microbenchmarks.benchmarks import benchmark_loader
from accelerator_microbenchmarks.core import registry


class BenchmarkLoaderTest(absltest.TestCase):
  """Unit tests for benchmark_loader.py."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    benchmark_loader.load_all_benchmarks()

  def test_load_all_benchmarks_nonempty(self):
    """Verifies that load_all_benchmarks loads a non-empty list of benchmarks."""
    benchmarks = registry.benchmark_registry.list_benchmarks()

    self.assertNotEmpty(benchmarks)

  def test_load_all_benchmarks_expected_benchmarks(self):
    """Verify that load_all_benchmarks loads expected benchmarks into registry."""
    benchmarks = registry.benchmark_registry.list_benchmarks()

    self.assertCountEqual(
        benchmarks,
        [
            "gemm_generalized",
            "hbm_bandwidth",
            "attention_flashed",
            "all_reduce_sum",
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


if __name__ == "__main__":
  absltest.main()
