"""Configuration management for JAX benchmarks."""

import dataclasses
import itertools
from typing import Any


from accelerator_microbenchmarks.core import csv_loader
from accelerator_microbenchmarks.core import model_configs
import yaml


def resolve_params(
    base_params: dict[str, Any], entry: dict[str, Any]
) -> list[dict[str, Any]]:
  """Resolve parameter sets from a config entry, supporting sweeps."""
  merged = base_params.copy()
  merged.update(entry)

  if "sweep" not in merged:
    return [merged]

  sweep_def = merged.pop("sweep")
  keys = list(sweep_def.keys())
  values = []

  for key in keys:
    val = sweep_def[key]
    if isinstance(val, list):
      values.append(val)
    elif isinstance(val, dict) and "start" in val and "end" in val:
      # Simple range/multiplier expansion
      start = val["start"]
      end = val["end"]
      mult = val.get("multiplier", 1)
      inc = val.get("increase_by", 1) if mult == 1 else 0

      curr = start
      seq = []
      while curr <= end:
        seq.append(curr)
        if mult > 1:
          curr *= mult
        else:
          curr += inc
      values.append(seq)
    else:
      values.append([val])

  # Generate Cartesian product of all sweep parameters
  combinations = []
  max_combinations = 1000  # Safeguard against combinatorial explosion
  product_size = 1
  for val_list in values:
    product_size *= len(val_list)
  if product_size > max_combinations:
    print(
        f"Warning: Sweep generates {product_size} combinations, capping at"
        f" {max_combinations} to prevent explosion."
    )

  for combo in itertools.islice(itertools.product(*values), max_combinations):
    param_set = merged.copy()
    param_set.update(dict(zip(keys, combo)))
    combinations.append(param_set)

  return combinations


def load_config(path: str) -> list[dict[str, Any]]:
  """Load and expand a single-benchmark YAML configuration.

  Args:
    path: Path to the YAML configuration file.

  Returns:
    A list of resolved parameter dictionaries, each containing 'name': <benchmark_name>.

  Raises:
    ValueError: If the file is not a dictionary, lacks 'benchmark:', or lacks
      a benchmark name.
  """
  with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

  if not isinstance(data, dict):
    raise ValueError(f"Config file at {path} must define a YAML dictionary.")

  if "benchmark" not in data or not isinstance(data["benchmark"], dict):
    raise ValueError(
        f"Config file at '{path}' must define a 'benchmark:' mapping."
    )

  # 1. Separate Top-Level Metadata from Benchmark Spec
  top_level = data.copy()
  benchmark_spec = top_level.pop("benchmark").copy()

  benchmark_name = benchmark_spec.pop("name", None)
  if not benchmark_name:
    raise ValueError(
        f"Config file at '{path}' must specify 'name:' inside the 'benchmark:' mapping."
    )

  global_params = top_level

  # 2. Expand Model Presets
  if "model" in benchmark_spec:
    model_name = benchmark_spec.pop("model")
    if model_name in model_configs.MODELS:
      model_params = dataclasses.asdict(model_configs.MODELS[model_name])
      for k, v in model_params.items():
        if k not in benchmark_spec:
          benchmark_spec[k] = v

  # 3. Expand Cases, CSV Shapes & Parameter Sweeps
  benchmark_spec["name"] = benchmark_name
  if "cases" in benchmark_spec:
    cases_list = benchmark_spec.pop("cases")
    if not isinstance(cases_list, list):
      raise ValueError(
          f"Expected 'cases' in benchmark '{benchmark_name}' to be a list, but"
          f" got {type(cases_list).__name__}."
      )
    fully_expanded = []
    for case_params in cases_list:
      if not isinstance(case_params, dict):
        raise ValueError(
            f"Expected each item in 'cases' of benchmark '{benchmark_name}' to"
            f" be a dict, but got {type(case_params).__name__}."
        )
      entry = benchmark_spec.copy()
      entry.update(case_params)
      fully_expanded.extend(resolve_params(global_params, entry))
    return fully_expanded

  if "csv_shapes" in benchmark_spec:
    csv_path = benchmark_spec.pop("csv_shapes")
    csv_entries = csv_loader.load_shapes_from_csv(csv_path)
    fully_expanded = []
    for row_params in csv_entries:
      entry = benchmark_spec.copy()
      entry.update(row_params)
      fully_expanded.extend(resolve_params(global_params, entry))
    return fully_expanded

  return resolve_params(global_params, benchmark_spec)
