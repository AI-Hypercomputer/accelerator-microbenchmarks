"""Main entry point for JAX microbenchmarks."""

import dataclasses
import json
import os
import traceback
from typing import List

from absl import app
from absl import flags
import jax
import pandas as pd



from jax_benchmarks import benchmarks  # pylint: disable=unused-import
from jax_benchmarks import core
from jax_benchmarks.core import base
from jax_benchmarks.core import config
from jax_benchmarks.core import system


_REPO_ROOT = "experimental/users/rahulasharma/jax_new"

FLAGS = flags.FLAGS

flags.DEFINE_string("config", None, "YAML config path", required=True)
flags.DEFINE_string("output", "results", "Output directory")
flags.DEFINE_string("hw", None, "Hardware target environment defined in config")
flags.DEFINE_string(
    "xprof_dir", "/tmp/tensorboard", "Directory for xprof traces"
)

# Mark flags as required where applicable
flags.mark_flag_as_required("config")


def save_output(results: List[base.BenchmarkResult], output_dir: str):
  """Save results in both digestible CSV and detailed JSON formats."""
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  flat_results = []
  for res in results:
    entry = {
        "benchmark": res.metadata.benchmark_name,
        "test_name": res.metadata.test_name,
        "KET_ms": res.metrics.get(
            "avg_ms", 0.0
        ),  # Kernel Execution Time as requested
        "throughput": res.metrics.get(
            "tflops_per_sec", res.metrics.get("bandwidth_gb_s", 0.0)
        ),
        "start": res.metadata.start_time,
        **res.metadata.params,
        **res.metrics,
        **res.metadata.device_info,
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




def main(argv):
  if len(argv) > 1:
    print(f"Warning: Unexpected positional arguments: {argv[1:]}")

  # Ensure JAX is initialized
  print("Initializing JAX distributed system...")
  try:
    jax.distributed.initialize()
  except Exception as e:
    print(f"Note: jax.distributed.initialize() failed or not needed: {e}")

  jax.config.update("jax_enable_x64", True)
  try:
    devices = jax.devices()
    print(f"JAX devices: {len(devices)} (e.g. {devices[:4]}...)")
  except Exception as e:
    print(f"Error initializing JAX devices: {e}")

  # Benchmarks are imported at the top

  config_path = FLAGS.config
  print(f"Loading config from: {config_path}")
  benchmark_configs = config.load_config(config_path)

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

      benchmark_cls = core.registry.get_benchmark(name)
      benchmark_instance = benchmark_cls()
      result = benchmark_instance.run(**cfg)
      all_results.append(result)
      print(f"Success. Metrics: {result.metrics}")
    except Exception as e:
      print(f"Benchmark '{name}' failed: {e}")
      traceback.print_exc()

  if all_results and jax.process_index() == 0:
    save_output(all_results, FLAGS.output)


if __name__ == "__main__":
  app.run(main)
