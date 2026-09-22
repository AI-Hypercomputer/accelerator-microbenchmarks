"""Execution runner engine for accelerator microbenchmarks."""

import dataclasses
import gc
import json
import os
import traceback
from typing import Any, List, Optional

from accelerator_microbenchmarks.benchmarks import benchmark_loader
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import config
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
import jax
import yaml

_REPO_ROOT = "third_party/py/accelerator_microbenchmarks"


def get_op_flags_and_env(
    benchmark_configs: List[dict[str, Any]],
    xla_flags_file_path: str | None = None,
) -> tuple[list[str], dict[str, str]]:
  """Resolves op_flags.yaml for benchmark configs and returns (flags_list, env_dict)."""
  benchmark_sets = set([conf["name"] for conf in benchmark_configs])
  if not benchmark_sets:
    raise ValueError("No benchmarks in config")
  if len(benchmark_sets) > 1:
    raise ValueError("Multiple benchmarks in config: %s" % benchmark_sets)
  benchmark_name = benchmark_sets.pop()
  if not benchmark_name:
    return [], {}

  flags_list: list[str] = []
  env_dict: dict[str, str] = {}

  try:
    if xla_flags_file_path is None:
      xla_flags_file_path = os.path.normpath(
          os.path.join(os.path.dirname(__file__), "..", "op_flags.yaml")
      )

    if xla_flags_file_path and os.path.exists(xla_flags_file_path):
      with open(xla_flags_file_path, "r") as f:
        op_flags = yaml.safe_load(f)

      if not isinstance(op_flags, dict):
        raise ValueError(
            f"Invalid format in '{xla_flags_file_path}': expected a dict, got"
            f" {type(op_flags).__name__}."
        )
      if benchmark_name in op_flags:
        flags_config = op_flags[benchmark_name]
        if not isinstance(flags_config, dict):
          raise ValueError(
              f"Invalid format for '{benchmark_name}' in"
              f" '{xla_flags_file_path}': expected a dict, got"
              f" {type(flags_config).__name__}."
          )
        unexpected_keys = set(flags_config.keys()) - {"flags", "env"}
        if unexpected_keys:
          raise ValueError(
              f"Unexpected keys for '{benchmark_name}' in"
              f" '{xla_flags_file_path}': {sorted(unexpected_keys)}."
          )
        raw_flags = flags_config.get("flags")
        if raw_flags is not None and not isinstance(raw_flags, list):
          raise ValueError(
              f"Invalid 'flags' for '{benchmark_name}' in"
              f" '{xla_flags_file_path}': expected a list, got"
              f" {type(raw_flags).__name__}."
          )
        raw_env = flags_config.get("env")
        if raw_env is not None and not isinstance(raw_env, dict):
          raise ValueError(
              f"Invalid 'env' for '{benchmark_name}' in"
              f" '{xla_flags_file_path}': expected a dict, got"
              f" {type(raw_env).__name__}."
          )
        flags_list = [str(flag) for flag in raw_flags or []]
        env_dict = {str(k): str(v) for k, v in (raw_env or {}).items()}
    else:
      print(
          f"Warning: op_flags.yaml not found at '{xla_flags_file_path}'. "
          "Default LIBTPU_INIT_ARGS will not be loaded."
      )
  except ValueError:
    raise
  except yaml.YAMLError as e:
    raise ValueError(
        f"Invalid YAML syntax in '{xla_flags_file_path}': {e}"
    ) from e
  except Exception as e:
    print(f"Warning: Failed to load op_flags.yaml: {e}")

  return flags_list, env_dict


def set_xla_flags(
    benchmark_configs: List[dict[str, Any]],
    xla_flags_file_path: str | None = None,
):
  """Set env vars based on first benchmark in config and op_flags.yaml."""
  flags_list, env_dict = get_op_flags_and_env(
      benchmark_configs, xla_flags_file_path=xla_flags_file_path
  )
  if flags_list:
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(flags_list)
    print(f"Set LIBTPU_INIT_ARGS: {os.environ['LIBTPU_INIT_ARGS']}")
  for k, v in env_dict.items():
    os.environ[k] = str(v)
    print(f"Set env {k}: {v}")

  print(f"RUNTIME_CFG: XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
  print(
      f"RUNTIME_CFG: LIBTPU_INIT_ARGS={os.environ.get('LIBTPU_INIT_ARGS', '')}"
  )


def init_jax_distributed():
  """Ensures JAX distributed coordinator is initialized."""
  print("Initializing JAX distributed system...")
  try:
    jax.distributed.initialize()
  except Exception as e:
    print(f"Note: jax.distributed.initialize() failed or not needed: {e}")
  try:
    devices = jax.devices()
    print(f"JAX devices: {len(devices)} (e.g. {devices[:4]}...)")
  except Exception as e:
    print(f"Error initializing JAX devices: {e}")


def run_benchmarks(
    tasks: List[tuple[str, base.BaseBenchmarkParams]],
    output_dir: str = "results",
    xprof_timing: bool = False,
    xprof_dir: str = "/tmp/tensorboard",
    config_path: Optional[str] = None,
    xla_flags_file_path: Optional[str] = None,
    print_table: bool = True,
) -> List[base.BenchmarkResult]:
  """Core execution engine for typed benchmark task configurations."""
  xprof_config = base.XprofConfig(
      xprof_timing=xprof_timing, xprof_dir=xprof_dir
  )

  benchmark_loader.load_all_benchmarks()

  # 1. Set Env Vars from op_flags.yaml
  set_xla_flags(
      [{"name": task_name} for task_name, _ in tasks],
      xla_flags_file_path=xla_flags_file_path,
  )

  # 2. Ensure JAX is initialized
  init_jax_distributed()

  # 3. Detect Platform and Resolve HardwareSpec
  platform_info = platform.get_platform_info()
  resolved_hardware_spec = system.get_hardware_spec(platform_info.tpu_type)

  all_results: List[base.BenchmarkResult] = []
  for task_name, config_obj in tasks:
    print(f"\n>>> Running Benchmark: {task_name} with {config_obj}")

    try:
      benchmark_cls = registry.benchmark_registry.get_benchmark(task_name)

      test_case_configs = config_obj.expand_test_cases()
      total_cases = len(test_case_configs)
      for idx, test_case_config in enumerate(test_case_configs, 1):
        benchmark_instance = benchmark_cls(
            config=test_case_config,
            hardware_spec=resolved_hardware_spec,
            xprof_config=xprof_config,
        )
        run_id = benchmark_instance.get_run_identifier()
        print(f"\nRunning [{idx}/{total_cases}] {task_name} ({run_id})...")
        result = benchmark_instance.run()
        all_results.append(result)
        print(
            f"Success [{idx}/{total_cases}] {task_name}. Metrics:"
            f" {result.metrics}"
        )
        gc.collect()
    except Exception as e:
      print(f"Benchmark '{task_name}' failed: {e}")
      traceback.print_exc()

  print(
      f"Process {jax.process_index()} reached end of run. all_results length:"
      f" {len(all_results)}"
  )

  if all_results:
    report.report_results(
        all_results,
        output_dir=output_dir,
        print_table=print_table,
    )

  return all_results
