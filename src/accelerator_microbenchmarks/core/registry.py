"""Registry system for JAX benchmarks."""

import dataclasses
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
    # Mapping of (primary_name, generation) to generation-specific benchmark
    # subclass overrides.
    self._overrides: dict[tuple[str, str], type[Any]] = {}
    # Mapping of (primary_name, generation) to generation-specific parameter
    # defaults.
    self._param_defaults: dict[tuple[str, str], dict[str, Any]] = {}

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
      benchmark_cls.is_experimental = is_experimental

      if is_experimental:
        self._experimental.add(name)

      for alias in aliases:
        if alias in self._benchmarks or alias in self._aliases:
          raise ValueError(f"Alias '{alias}' is already registered.")
        self._aliases[alias] = name

      return benchmark_cls

    return wrapper

  def register_override(self, name: str, *, generation: str):
    """Registers `cls` as the implementation of `name` on `generation`."""

    def wrapper(cls: type[Any]) -> type[Any]:
      primary = self._aliases.get(name, name)
      if primary not in self._benchmarks:
        raise KeyError(
            f"Cannot register override for unknown benchmark '{name}'."
        )
      public_cls = self._benchmarks[primary]
      if not issubclass(cls, public_cls):
        raise TypeError(f"{cls.__name__} must subclass the public '{name}'.")
      if cls.Config is not public_cls.Config:
        raise TypeError(
            f"{cls.__name__}.Config must be {public_cls.__name__}.Config."
        )
      if cls.name not in (public_cls.name, cls.__name__):
        raise ValueError(
            f"{cls.__name__}.name must be '{public_cls.name}', got"
            f" '{cls.name}'."
        )
      cls.name = public_cls.name
      gen_key = str(generation)
      existing = self._overrides.get((primary, gen_key))
      if existing is not None and existing is not cls:
        raise ValueError(
            f"Override for benchmark '{name}' on generation '{gen_key}' is"
            " already registered."
        )
      self._overrides[(primary, gen_key)] = cls
      return cls

    return wrapper

  def register_param_defaults(
      self, name: str, *, generation: str, **defaults: Any
  ) -> None:
    """Registers generation-scoped default parameter overrides for `name`."""
    primary = self._aliases.get(name, name)
    if primary not in self._benchmarks:
      raise KeyError(
          f"Cannot register param defaults for unknown benchmark '{name}'."
      )
    public_cls = self._benchmarks[primary]
    valid_fields = {
        f.name for f in dataclasses.fields(public_cls.Config) if f.init
    }
    invalid_keys = sorted(set(defaults) - valid_fields)
    if invalid_keys:
      raise ValueError(
          f"Unknown Config field(s) {invalid_keys} for benchmark '{name}'."
          f" Valid fields: {sorted(valid_fields)}"
      )
    gen_key = str(generation)
    if (primary, gen_key) in self._param_defaults:
      raise ValueError(
          f"Param defaults for benchmark '{name}' on generation '{gen_key}'"
          " are already registered."
      )
    self._param_defaults[(primary, gen_key)] = dict(defaults)

  def get_param_defaults(
      self, name: str, generation: str | None = None
  ) -> dict[str, Any]:
    """Returns generation-scoped default parameter overrides for `name`."""
    if generation is None:
      return {}
    primary = self._aliases.get(name, name)
    return dict(self._param_defaults.get((primary, str(generation)), {}))

  def get_benchmark(
      self, name: str, generation: str | None = None
  ) -> type[Any]:
    """Retrieve a benchmark class by name or alias."""
    primary_name = self._aliases.get(name, name)
    if (
        generation is not None
        and (primary_name, str(generation)) in self._overrides
    ):
      return self._overrides[(primary_name, str(generation))]
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
