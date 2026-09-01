"""Execution runner engine for accelerator microbenchmarks."""

import gc
import json
import os
import traceback
from typing import Any, List, Optional

from accelerator_microbenchmarks.benchmarks import benchmark_loader
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import config
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
import jax
import yaml

_REPO_ROOT = "third_party/py/accelerator_microbenchmarks"

# TODO: Revisit benchmark name mapping design.
_BENCHMARK_NAME_MAPPING = {
    "all_reduce": "all_reduce",
    "reduce_scatter": "psum_scatter",
    "hbm": "hbm_bandwidth",
}


def set_xla_flags(
    benchmark_configs: List[dict[str, Any]], flags_file_path: str | None = None
):
  """Set env vars based on first benchmark in config and op_flags.yaml."""
  benchmark_sets = set([conf["name"] for conf in benchmark_configs])
  if not benchmark_sets:
    raise ValueError("No benchmarks in config")
  if len(benchmark_sets) > 1:
    raise ValueError("Multiple benchmarks in config: %s" % benchmark_sets)
  benchmark_name = benchmark_sets.pop()
  if not benchmark_name:
    return

  op_key = _BENCHMARK_NAME_MAPPING.get(benchmark_name, benchmark_name)
  try:
    if flags_file_path is None:
      flags_file_path = os.path.join(
          os.path.dirname(__file__), "..", "op_flags.yaml"
      )

    if flags_file_path and os.path.exists(flags_file_path):
      with open(flags_file_path, "r") as f:
        op_flags = yaml.safe_load(f)

      if op_key in op_flags:
        flags_config = op_flags[op_key]
        if isinstance(flags_config, list):
          os.environ["LIBTPU_INIT_ARGS"] = " ".join(flags_config)
          print(f"Set LIBTPU_INIT_ARGS: {os.environ['LIBTPU_INIT_ARGS']}")
        elif isinstance(flags_config, dict):
          if "flags" in flags_config:
            os.environ["LIBTPU_INIT_ARGS"] = " ".join(flags_config["flags"])
            print(f"Set LIBTPU_INIT_ARGS: {os.environ['LIBTPU_INIT_ARGS']}")
          if "env" in flags_config:
            for k, v in flags_config["env"].items():
              os.environ[k] = str(v)
              print(f"Set env {k}: {v}")
  except Exception as e:
    print(f"Warning: Failed to load op_flags.yaml: {e}")

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
    hw: Optional[str] = None,
    xprof_dir: str = "/tmp/tensorboard",
    config_path: Optional[str] = None,
    print_table: bool = True,
) -> List[base.BenchmarkResult]:
  """Core execution engine for typed benchmark task configurations."""
  benchmark_loader.load_all_benchmarks()

  # 1. Set Env Vars from op_flags.yaml
  set_xla_flags([{"name": task_name} for task_name, _ in tasks])

  # 2. Ensure JAX is initialized
  init_jax_distributed()

  all_results: List[base.BenchmarkResult] = []
  for task_name, config_obj in tasks:
    print(f"\n>>> Running Benchmark: {task_name} with {config_obj}")

    try:
      benchmark_cls = registry.benchmark_registry.get_benchmark(task_name)

      if not config_obj.xprof_dir or config_obj.xprof_dir == "/tmp/tensorboard":
        config_obj.xprof_dir = xprof_dir

      sys_name = config_obj.system or hw
      if sys_name and not config_obj.hardware_stats:
        try:
          sys_config = system.get_system(sys_name)
          hardware_stats = {}
          if sys_config.tflops:
            hardware_stats["tflops"] = sys_config.tflops.peak_tflops_per_dtype
          if sys_config.hbm:
            hardware_stats["hbm_bw"] = sys_config.hbm.curve_gbps
          if sys_config.ici:
            hardware_stats["ici"] = {
                "peak_bw_gbps": sys_config.ici.peak_bw_gbps,
                "bidirectional": sys_config.ici.bidirectional,
            }
          if hardware_stats:
            config_obj.hardware_stats = hardware_stats
        except Exception as e:
          print(f"Warning: Could not load system config for {sys_name}: {e}")

      test_case_configs = config_obj.expand_test_cases()
      total_cases = len(test_case_configs)
      for idx, test_case_config in enumerate(test_case_configs, 1):
        benchmark_instance = benchmark_cls(test_case_config)
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
