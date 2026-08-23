"""Registry system for JAX benchmarks."""

from typing import Any, Sequence


class BenchmarkRegistry:
  """Registry to store and retrieve benchmark classes by name or alias."""

  def __init__(self):
    # Mapping of primary benchmark name to benchmark class implementation.
    self._benchmarks: dict[str, type[Any]] = {}
    # Mapping of alternative alias names to their primary name (key: alias, value: primary name).
    self._aliases: dict[str, str] = {}
    # Set of primary benchmark names flagged as experimental (excluded from default CLI discovery).
    self._experimental: set[str] = set()

  def register(
      self,
      name: str,
      aliases: Sequence[str] = (),
      *,
      is_experimental: bool = False,
  ):
    """Decorator to register a benchmark with a name, optional aliases, and experimental status."""

    def wrapper(benchmark_cls: type[Any]):
      if name in self._benchmarks or name in self._aliases:
        raise ValueError(f"Benchmark name '{name}' is already registered.")
      self._benchmarks[name] = benchmark_cls

      if is_experimental:
        self._experimental.add(name)

      for alias in aliases:
        if alias in self._benchmarks or alias in self._aliases:
          raise ValueError(f"Alias '{alias}' is already registered.")
        self._aliases[alias] = name

      return benchmark_cls

    return wrapper

  def get_benchmark(self, name: str) -> type[Any]:
    """Retrieve a benchmark class by name or alias."""
    primary_name = self._aliases.get(name, name)
    if primary_name in self._benchmarks:
      return self._benchmarks[primary_name]

    available = ", ".join(
        self.list_benchmark_names(
            include_experimental=True, include_aliases=True
        )
    )
    raise KeyError(f"Benchmark '{name}' not found. Available: {available}")

  def list_benchmark_names(
      self, include_experimental: bool = False, include_aliases: bool = False
  ) -> list[str]:
    """List benchmark names."""
    names = [
        name
        for name in self._benchmarks
        if include_experimental or name not in self._experimental
    ]
    if not include_aliases:
      return sorted(names)

    aliases = [
        alias
        for alias, primary in self._aliases.items()
        if include_experimental or primary not in self._experimental
    ]
    return sorted(names + aliases)

  def is_experimental(self, name: str) -> bool:
    """Check if a benchmark (by name or alias) is experimental."""
    primary_name = self._aliases.get(name, name)
    return primary_name in self._experimental

  def get_all(self) -> dict[str, type[Any]]:
    """Get a dictionary of all registered benchmarks (both names and aliases)."""
    result = dict(self._benchmarks)
    for alias, primary in self._aliases.items():
      if primary in self._benchmarks:
        result[alias] = self._benchmarks[primary]
    return result


# Default global registry instance
benchmark_registry = BenchmarkRegistry()
