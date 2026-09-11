"""Configuration management for JAX benchmarks."""

import dataclasses
import enum
import itertools
from typing import Any

from accelerator_microbenchmarks.core import csv_loader
from accelerator_microbenchmarks.core import model_configs
import yaml


class BenchmarkKey(str, enum.Enum):
  NAME = "name"
  MODEL_PRESET = "model_preset"
  PARAMS = "params"
  CASES = "cases"
  CASES_FROM_CSV = "cases_from_csv"
  SWEEP = "sweep"


_ALLOWED_BENCHMARK_KEYS = frozenset(BenchmarkKey)
_MAX_COMBINATIONS = 1000


def expand_sweep(
    base_params: dict[str, Any], sweep_spec: dict[str, Any] | None
) -> list[dict[str, Any]]:
  """Expands a parameter dictionary across all Cartesian sweep axes.

  Args:
    base_params: Baseline parameter dictionary to expand.
    sweep_spec: Mapping of parameter name to either a list of values, a single
      scalar value, or a range dictionary specifying 'start', 'end', and
      optionally 'multiplier' or 'increase_by'. If None or empty, returns a
      single-element list containing a copy of base_params.

  Returns:
    A list of resolved parameter dictionaries covering the Cartesian product
    of all sweep axes, capped at _MAX_COMBINATIONS.

  Raises:
    ValueError: If a range sweep specification defines start > end or
      non-positive progression (multiplier <= 1 and increase_by <= 0).
  """
  if not sweep_spec:
    return [base_params.copy()]

  keys = list(sweep_spec.keys())
  values = []
  for key in keys:
    val = sweep_spec[key]
    if isinstance(val, list):
      values.append(val)
    elif isinstance(val, dict) and "start" in val and "end" in val:
      start, end = val["start"], val["end"]
      if start > end:
        raise ValueError(
            f"Sweep range for key '{key}' has start ({start}) > end ({end})."
        )
      mult = val.get("multiplier", 1)
      inc = val.get("increase_by", 1) if mult == 1 else 0
      if mult <= 1 and inc <= 0:
        raise ValueError(
            f"Range sweep for '{key}' must have positive progression"
            f" (multiplier > 1 or increase_by > 0), but got multiplier={mult}"
            f" and increase_by={inc}."
        )
      curr, seq = start, []
      while curr <= end:
        seq.append(curr)
        curr = curr * mult if mult > 1 else curr + inc
      values.append(seq)
    else:
      values.append([val])

  # Generate Cartesian product of all sweep parameters
  combinations = []
  product_size = 1
  for val_list in values:
    product_size *= len(val_list)
  if product_size > _MAX_COMBINATIONS:
    print(
        f"Warning: Sweep generates {product_size} combinations, capping at"
        f" {_MAX_COMBINATIONS} to prevent explosion."
    )

  for combo in itertools.islice(itertools.product(*values), _MAX_COMBINATIONS):
    param_set = base_params.copy()
    param_set.update(dict(zip(keys, combo)))
    combinations.append(param_set)
  return combinations


def _validate_disjoint_sweep_keys(
    benchmark_name: str,
    sweep: dict[str, Any],
    params: dict[str, Any],
    cases_list: list[dict[str, Any]] | None,
    cases_source: BenchmarkKey,
) -> None:
  """Validates that sweep parameters do not collide with reserved, baseline, or case keys."""
  sweep_keys = set(sweep.keys())
  reserved_collision = sweep_keys & _ALLOWED_BENCHMARK_KEYS
  if reserved_collision:
    raise ValueError(
        f"Benchmark '{benchmark_name}' has reserved key(s)"
        f" {sorted(reserved_collision)} defined in"
        f" '{BenchmarkKey.SWEEP.value}'."
    )
  params_collision = sweep_keys & set(params.keys())
  if params_collision:
    raise ValueError(
        f"Benchmark '{benchmark_name}' has parameter(s)"
        f" {sorted(params_collision)} defined in '{BenchmarkKey.SWEEP.value}'"
        f" that collide with keys in '{BenchmarkKey.PARAMS.value}'."
    )
  if cases_list:
    case_keys = set()
    for c in cases_list:
      if isinstance(c, dict):
        case_keys.update(c.keys())
    case_collision = sweep_keys & case_keys
    if case_collision:
      raise ValueError(
          f"Benchmark '{benchmark_name}' has parameter(s)"
          f" {sorted(case_collision)} defined in '{BenchmarkKey.SWEEP.value}'"
          f" that collide with keys in '{cases_source.value}'."
      )


def load_config(path: str) -> list[dict[str, Any]]:
  """Load and expand a single-benchmark YAML configuration.

  Args:
    path: Path to the YAML configuration file.

  Returns:
    A list of resolved parameter dictionaries, each containing 'name':
      <benchmark_name>.

  Raises:
    ValueError: If the file is not a dictionary, lacks 'benchmark:', or violates
      schema constraints.
  """
  with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  if not isinstance(data, dict):
    raise ValueError(f"Config file at {path} must define a YAML dictionary.")

  if "benchmark" not in data or not isinstance(data["benchmark"], dict):
    raise ValueError(
        f"Config file at '{path}' must define a 'benchmark:' mapping."
    )

  extra_top_keys = set(data.keys()) - {"benchmark"}
  if extra_top_keys:
    raise ValueError(
        f"Unrecognized top-level key(s) in '{path}': {sorted(extra_top_keys)}."
        " Baseline execution parameters should be placed under 'benchmark:' ->"
        f" '{BenchmarkKey.PARAMS.value}:'."
    )

  benchmark_spec = data["benchmark"].copy()

  benchmark_name = benchmark_spec.get(BenchmarkKey.NAME)
  if not benchmark_name:
    raise ValueError(
        f"Config file at '{path}' must specify '{BenchmarkKey.NAME.value}:'"
        " inside the 'benchmark:' mapping."
    )

  extra_keys = set(benchmark_spec.keys()) - _ALLOWED_BENCHMARK_KEYS
  if extra_keys:
    raise ValueError(
        f"Unrecognized key(s) in 'benchmark' mapping of '{path}':"
        f" {sorted(extra_keys)}. Baseline execution parameters should be"
        f" placed under '{BenchmarkKey.PARAMS.value}:'."
    )

  if (
      BenchmarkKey.CASES in benchmark_spec
      and BenchmarkKey.CASES_FROM_CSV in benchmark_spec
  ):
    raise ValueError(
        f"Benchmark '{benchmark_name}' cannot specify both"
        f" '{BenchmarkKey.CASES.value}' and"
        f" '{BenchmarkKey.CASES_FROM_CSV.value}'. Choose one."
    )

  # Validate and extract params
  params = benchmark_spec.get(BenchmarkKey.PARAMS)
  if params is None:
    params = {}
  elif not isinstance(params, dict):
    raise ValueError(
        f"Expected '{BenchmarkKey.PARAMS.value}' in benchmark"
        f" '{benchmark_name}' to be a dict, but got {type(params).__name__}."
    )

  params_reserved = set(params.keys()) & _ALLOWED_BENCHMARK_KEYS
  if params_reserved:
    raise ValueError(
        f"Benchmark '{benchmark_name}' has reserved key(s)"
        f" {sorted(params_reserved)} defined in '{BenchmarkKey.PARAMS.value}'."
    )

  # Validate and extract model preset
  model_name = benchmark_spec.get(BenchmarkKey.MODEL_PRESET)
  if model_name is not None and model_name not in model_configs.MODELS:
    raise ValueError(
        f"Unknown model preset '{model_name}'. Valid presets are:"
        f" {sorted(model_configs.MODELS.keys())}."
    )
  model_defaults = (
      dataclasses.asdict(model_configs.MODELS[model_name]) if model_name else {}
  )

  # Validate and extract cases
  cases_list = None
  cases_source = BenchmarkKey.CASES
  if BenchmarkKey.CASES in benchmark_spec:
    cases_list = benchmark_spec.get(BenchmarkKey.CASES)
    if not isinstance(cases_list, list):
      raise ValueError(
          f"Expected '{BenchmarkKey.CASES.value}' in benchmark"
          f" '{benchmark_name}' to be a list, but got"
          f" {type(cases_list).__name__}."
      )
    for c in cases_list:
      if not isinstance(c, dict):
        raise ValueError(
            f"Expected each item in '{BenchmarkKey.CASES.value}' of benchmark"
            f" '{benchmark_name}' to be a dict, but got {type(c).__name__}."
        )
  elif BenchmarkKey.CASES_FROM_CSV in benchmark_spec:
    csv_path = benchmark_spec.get(BenchmarkKey.CASES_FROM_CSV)
    if not isinstance(csv_path, str) or not csv_path.strip():
      raise ValueError(
          f"Expected '{BenchmarkKey.CASES_FROM_CSV.value}' in benchmark"
          f" '{benchmark_name}' to be a non-empty str, but got"
          f" {type(csv_path).__name__}."
      )
    cases_list = csv_loader.load_cases_from_csv(csv_path)
    cases_source = BenchmarkKey.CASES_FROM_CSV
    for c in cases_list:
      if not isinstance(c, dict):
        raise ValueError(
            f"Expected each item in '{BenchmarkKey.CASES_FROM_CSV.value}' of"
            f" benchmark '{benchmark_name}' to be a dict, but got"
            f" {type(c).__name__}."
        )

  # Validate and extract sweep
  sweep = benchmark_spec.get(BenchmarkKey.SWEEP)
  if sweep is not None:
    if not isinstance(sweep, dict) or not sweep:
      raise ValueError(
          f"Expected '{BenchmarkKey.SWEEP.value}' in benchmark"
          f" '{benchmark_name}' to be a non-empty dict, but got"
          f" {type(sweep).__name__}."
      )
    _validate_disjoint_sweep_keys(
        benchmark_name=benchmark_name,
        sweep=sweep,
        params=params,
        cases_list=cases_list,
        cases_source=cases_source,
    )

  # Stage 1 (model_preset) & Stage 2 (params): Base parameters
  # params overrides model_defaults; benchmark name is injected.
  base_params = {**model_defaults, **params, "name": benchmark_name}

  # Stage 3: Cases (per-case parameter overrides over base_params)
  case_dicts = (
      [{**base_params, **case} for case in cases_list]
      if cases_list is not None
      else [base_params]
  )

  # Stage 4: Sweep Cartesian product expansion across combinations
  if sweep is not None:
    return list(
        itertools.chain.from_iterable(
            expand_sweep(case, sweep) for case in case_dicts
        )
    )
  return case_dicts
