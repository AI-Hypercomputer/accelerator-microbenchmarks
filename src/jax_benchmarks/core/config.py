"""Configuration management for JAX benchmarks."""

import itertools
from typing import Any, Dict, List, Optional
import yaml


def resolve_params(
    base_params: Dict[str, Any], entry: Dict[str, Any]
) -> List[Dict[str, Any]]:
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


def load_config(path: str) -> List[Dict[str, Any]]:
  """Load and expand hierarchical benchmark configuration."""
  from .model_configs import MODELS

  with open(path, "r") as f:
    data = yaml.safe_load(f)

  global_params = data.get("global", {})
  hardware_stats = data.get("hardware", {})
  if hardware_stats:
    global_params["hardware_stats"] = hardware_stats

  benchmarks_data = data.get("benchmarks", [])

  fully_expanded = []
  for b_entry in benchmarks_data:
    # 1. Handle Model Presets
    if "model" in b_entry:
      model_name = b_entry.pop("model")
      if model_name in MODELS:
        import dataclasses

        model_params = dataclasses.asdict(MODELS[model_name])
        for k, v in model_params.items():
          if k not in b_entry:
            b_entry[k] = v

    # 2. Handle CSV Shapes
    if "csv_shapes" in b_entry:
      from .csv_loader import load_shapes_from_csv

      csv_path = b_entry.pop("csv_shapes")
      # If path is relative, it should be relative to the config file or CWD
      # For simplicity, assuming relative to CWD or absolute
      csv_entries = load_shapes_from_csv(csv_path)

      for row_params in csv_entries:
        # Create a specific entry for this CSV row
        specific_entry = b_entry.copy()
        specific_entry.update(row_params)
        fully_expanded.extend(resolve_params(global_params, specific_entry))
    else:
      fully_expanded.extend(resolve_params(global_params, b_entry))

  return fully_expanded
