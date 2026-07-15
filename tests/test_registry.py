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

  def test_list_benchmarks(self):
    """Verify that list_benchmarks returns sorted registered benchmark names."""
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
        self.test_registry.list_benchmarks(), ["a_bm", "b_bm", "c_bm"]
    )

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
    self.assertIn("bm1", self.test_registry.list_benchmarks())
    self.assertNotIn("bm1", registry2.list_benchmarks())

    # bm2 should only be in registry2
    self.assertIn("bm2", registry2.list_benchmarks())
    self.assertNotIn("bm2", self.test_registry.list_benchmarks())


if __name__ == "__main__":
  absltest.main()
