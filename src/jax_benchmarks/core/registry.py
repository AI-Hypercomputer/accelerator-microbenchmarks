"""Registry system for JAX benchmarks."""

from typing import Any, Dict, List, Type


class BenchmarkRegistry:
  """Registry to store and retrieve benchmark classes."""

  _benchmarks: Dict[str, Type[Any]] = {}

  @classmethod
  def register(cls, name: str):
    """Decorator to register a benchmark class."""

    def wrapper(benchmark_cls: Type[Any]):
      if name in cls._benchmarks:
        raise ValueError(f"Benchmark '{name}' is already registered.")
      cls._benchmarks[name] = benchmark_cls
      return benchmark_cls

    return wrapper

  @classmethod
  def get_benchmark(cls, name: str) -> Type[Any]:
    """Retrieve a benchmark class by name."""
    if name not in cls._benchmarks:
      available = ", ".join(cls.list_benchmarks())
      raise KeyError(f"Benchmark '{name}' not found. Available: {available}")
    return cls._benchmarks[name]

  @classmethod
  def list_benchmarks(cls) -> List[str]:
    """List all registered benchmarks."""
    return sorted(list(cls._benchmarks.keys()))


# Global registry instance
registry = BenchmarkRegistry()
