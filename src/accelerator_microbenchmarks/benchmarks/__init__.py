"""Initializes the benchmarks package and auto-registers all benchmarks."""

from accelerator_microbenchmarks.benchmarks import benchmark_loader

try:
  benchmark_loader.load_all_benchmarks()
except Exception as e:
  print(f"Early error loading benchmarks: {e}")
