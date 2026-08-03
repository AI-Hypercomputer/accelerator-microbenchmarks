"""Registry system for JAX benchmarks."""

from typing import Any


class BenchmarkRegistry:
  """Registry to store and retrieve benchmark classes."""

  def __init__(self):
    self._benchmarks: dict[str, type[Any]] = {}

  def register(self, name: str):
    """Decorator to register a benchmark class."""

    def wrapper(benchmark_cls: type[Any]):
      if name in self._benchmarks:
        raise ValueError(f"Benchmark '{name}' is already registered.")
      self._benchmarks[name] = benchmark_cls
      return benchmark_cls

    return wrapper

  def get_benchmark(self, name: str) -> type[Any]:
    """Retrieve a benchmark class by name."""
    if name not in self._benchmarks:
      available = ", ".join(self.list_benchmarks())
      raise KeyError(f"Benchmark '{name}' not found. Available: {available}")
    return self._benchmarks[name]

  def list_benchmarks(self) -> list[str]:
    """List all registered benchmarks."""
    return sorted(list(self._benchmarks.keys()))

  def get_all(self) -> dict[str, type[Any]]:
    """Get a dictionary of all registered benchmarks."""
    return self._benchmarks


# Default global registry instance
benchmark_registry = BenchmarkRegistry()
