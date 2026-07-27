"""Main entry point for JAX microbenchmarks."""

import dataclasses
import json
import os
import traceback
from typing import Any, List
from absl import app
from absl import flags
from accelerator_microbenchmarks.benchmarks import benchmark_loader
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import config
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import system
import jax
import pandas as pd
import yaml

_REPO_ROOT = "third_party/py/accelerator_microbenchmarks"

# Map g3 benchmark names to op_flags.yaml keys
_BENCHMARK_NAME_MAPPING = {
    "all_reduce_sum": "psum",
    "reduce_scatter": "psum_scatter",
}

FLAGS = flags.FLAGS

flags.DEFINE_string("config", None, "YAML config path")
flags.DEFINE_string("output", "results", "Output directory")
flags.DEFINE_string("hw", None, "Hardware target environment defined in config")
flags.DEFINE_string(
    "xprof_dir", "/tmp/tensorboard", "Directory for xprof traces"
)


def save_output(results: List[base.BenchmarkResult], output_dir: str):
  """Save results in both digestible CSV and detailed JSON formats."""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  flat_results = []
  for res in results:
    entry = {
        **res.metadata.params,
        **res.metrics,
        **res.metadata.device_info,
        "benchmark": res.metadata.benchmark_name,
        "test_name": res.metadata.test_name,
        "KET_ms": res.metrics.get(
            "avg_ms", 0.0
        ),  # Kernel Execution Time as requested
        "throughput": res.metrics.get(
            "tflops_per_sec", res.metrics.get("bandwidth_gb_s", 0.0)
        ),
        "start": res.metadata.start_time,
    }
    flat_results.append(entry)

  # 1. Summary CSV (Digestible)
  df = pd.DataFrame(flat_results)
  csv_path = os.path.join(output_dir, "summary.csv")
  df.to_csv(csv_path, index=False)
  print(f"Summary saved to: {csv_path}")

  # 2. Detailed JSON (Complete)
  json_path = os.path.join(output_dir, "detailed.json")
  with open(json_path, "w") as f:
    json.dump([dataclasses.asdict(r) for r in results], f, indent=2)


def set_xla_flags(
    benchmark_configs: List[dict[str, Any]], flags_file_path: str | None = None
):
  """Set env vars based on first benchmark in config and op_flags.yaml."""
  if not benchmark_configs:
    return
  benchmark_name = benchmark_configs[0].get("name")
  if not benchmark_name:
    return

  op_key = _BENCHMARK_NAME_MAPPING.get(benchmark_name, benchmark_name)
  try:
    if flags_file_path is None:
      flags_file_path = os.path.join(
          os.path.dirname(__file__), "op_flags.yaml"
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


def main(argv):
  if len(argv) > 1:
    print(f"Warning: Unexpected positional arguments: {argv[1:]}")

  # 1. Load Config
  try:
    config_path = FLAGS.config
    print(f"Loading config from: {config_path}")
    benchmark_configs = config.load_config(config_path)
  except Exception as e:
    print(f"Error loading config: {e}")
    return

  # 2. Set Env Vars from op_flags.yaml
  set_xla_flags(benchmark_configs)

  # 3. Ensure JAX is initialized
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

  # Dynamically load all benchmarks to register them.
  # This is needed to allow for registration in subpackages.
  try:
    benchmark_loader.load_all_benchmarks()
  except Exception as e:
    print(f"Error loading benchmarks: {e}")
    return

  all_results = []
  for cfg in benchmark_configs:
    name = cfg.pop("name")
    print(f"\n>>> Running Benchmark: {name} with {cfg}")

    try:
      # Inject flag value if not explicitly in config
      if "xprof_dir" not in cfg:
        cfg["xprof_dir"] = FLAGS.xprof_dir

      # Determine system from config or flag
      sys_name = cfg.get("system", FLAGS.hw)
      if sys_name and "hardware_stats" not in cfg:
        try:
          sys_config = system.get_system(sys_name)
          cfg["hardware_stats"] = {
              "tflops": sys_config.tflops.peak_tflops_per_dtype,
              "hbm_bw": sys_config.hbm.curve_gbps,
              "ici": {
                  "peak_bw_gbps": sys_config.ici.peak_bw_gbps,
                  "bidirectional": sys_config.ici.bidirectional,
              },
          }
        except Exception as e:
          print(f"Warning: Could not load system config for {sys_name}: {e}")

      benchmark_cls = registry.benchmark_registry.get_benchmark(name)
      benchmark_instance = benchmark_cls()
      result = benchmark_instance.run(**cfg)
      all_results.append(result)
      print(f"Success. Metrics: {result.metrics}")
    except Exception as e:
      print(f"Benchmark '{name}' failed: {e}")
      traceback.print_exc()

  print(
      f"Process {jax.process_index()} reached end of main. all_results length:"
      f" {len(all_results)}"
  )

  # Ensure output directory exists on all processes to avoid tar errors
  if not os.path.exists(FLAGS.output):
    os.makedirs(FLAGS.output)

  if all_results and jax.process_index() == 0:
    save_output(all_results, FLAGS.output)


def run_main():
  flags.mark_flag_as_required("config")
  app.run(main)


if __name__ == "__main__":
  run_main()
