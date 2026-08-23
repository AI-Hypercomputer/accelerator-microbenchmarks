"""Unit tests for registry.py."""

from absl.testing import absltest
from accelerator_microbenchmarks.core import registry


class RegistryTest(absltest.TestCase):
  """Unit tests for registry.py."""

  def setUp(self):
    super().setUp()
    # Create a fresh registry instance for each test to ensure isolation
    self.test_registry = registry.BenchmarkRegistry()

  def test_register_and_get_benchmark(self):
    """Verify that a benchmark can be registered and retrieved."""
    @self.test_registry.register("test_bm")
    class TestBenchmark:
      pass

    bm_class = self.test_registry.get_benchmark("test_bm")
    self.assertEqual(bm_class, TestBenchmark)

  def test_register_duplicate_error(self):
    """Verify that registering a duplicate benchmark raises ValueError."""
    @self.test_registry.register("test_bm")
    class TestBenchmark1:
      pass

    with self.assertRaises(ValueError):
      @self.test_registry.register("test_bm")
      class TestBenchmark2:
        pass

  def test_get_benchmark_nonexistent_error(self):
    """Verify that retrieving a non-existent benchmark raises KeyError."""
    with self.assertRaises(KeyError):
      self.test_registry.get_benchmark("nonexistent")

  def test_register_with_aliases(self):
    """Verify that register with aliases allows lookup by any registered name."""
    @self.test_registry.register("gemm", aliases=["gemm_generalized"])
    class GemmGeneralizedBenchmark:
      pass

    self.assertEqual(
        self.test_registry.get_benchmark("gemm"), GemmGeneralizedBenchmark
    )
    self.assertEqual(
        self.test_registry.get_benchmark("gemm_generalized"),
        GemmGeneralizedBenchmark,
    )

  def test_list_benchmark_names(self):
    """Verify that list_benchmark_names returns sorted registered benchmark names."""
    @self.test_registry.register("c_bm")
    class BenchmarkC:
      pass

    @self.test_registry.register("a_bm")
    class BenchmarkA:
      pass

    @self.test_registry.register("b_bm")
    class BenchmarkB:
      pass

    self.assertEqual(
        self.test_registry.list_benchmark_names(), ["a_bm", "b_bm", "c_bm"]
    )

  def test_experimental_benchmarks(self):
    """Verify that experimental benchmarks are filtered by default in listing."""
    @self.test_registry.register("stable_bm", aliases=["stable_alias"])
    class StableBenchmark:
      pass

    @self.test_registry.register(
        "exp_bm", aliases=["exp_alias"], is_experimental=True
    )
    class ExpBenchmark:
      pass

    self.assertFalse(self.test_registry.is_experimental("stable_bm"))
    self.assertTrue(self.test_registry.is_experimental("exp_bm"))
    self.assertTrue(self.test_registry.is_experimental("exp_alias"))

    # By default, only non-experimental benchmarks (without aliases) are returned
    self.assertEqual(
        self.test_registry.list_benchmark_names(
            include_experimental=False, include_aliases=False
        ),
        ["stable_bm"],
    )

    # When include_experimental=False, include_aliases=True, experimental aliases must be excluded
    self.assertEqual(
        self.test_registry.list_benchmark_names(
            include_experimental=False, include_aliases=True
        ),
        ["stable_alias", "stable_bm"],
    )

    # When include_experimental=True, include_aliases=False
    self.assertEqual(
        self.test_registry.list_benchmark_names(
            include_experimental=True, include_aliases=False
        ),
        ["exp_bm", "stable_bm"],
    )

    # When include_experimental=True, include_aliases=True
    self.assertEqual(
        self.test_registry.list_benchmark_names(
            include_experimental=True, include_aliases=True
        ),
        ["exp_alias", "exp_bm", "stable_alias", "stable_bm"],
    )

  def test_get_all(self):
    """Verify that get_all returns both primary names and aliases."""
    @self.test_registry.register("gemm", aliases=["gemm_generalized"])
    class GemmBenchmark:
      pass

    all_benchmarks = self.test_registry.get_all()
    self.assertIn("gemm", all_benchmarks)
    self.assertIn("gemm_generalized", all_benchmarks)
    self.assertEqual(all_benchmarks["gemm"], GemmBenchmark)
    self.assertEqual(all_benchmarks["gemm_generalized"], GemmBenchmark)

  def test_registry_isolation(self):
    """Verify that separate registry instances are completely isolated."""
    registry2 = registry.BenchmarkRegistry()

    @self.test_registry.register("bm1")
    class Benchmark1:
      pass

    @registry2.register("bm2")
    class Benchmark2:
      pass

    # bm1 should only be in self.test_registry
    self.assertIn("bm1", self.test_registry.list_benchmark_names())
    self.assertNotIn("bm1", registry2.list_benchmark_names())

    # bm2 should only be in registry2
    self.assertIn("bm2", registry2.list_benchmark_names())
    self.assertNotIn("bm2", self.test_registry.list_benchmark_names())


if __name__ == "__main__":
  absltest.main()
