"""Smoke tests for benchmarks to ensure they run correctly on CPU."""

import benchmarks  # Trigger registration
from core import registry
import jax
import pytest


@pytest.mark.parametrize(
    "benchmark_name", ["matmul_naive", "matmul_shard_map", "all_reduce_sum"]
)
def test_benchmark_smoke(benchmark_name):
  """Verify individual benchmarks compile and run with small inputs on CPU."""
  benchmark_cls = registry.get_benchmark(benchmark_name)
  instance = benchmark_cls()

  # Small parameters for quick smoke test
  params = {
      "m": 128,
      "n": 128,
      "k": 128,  # Matmul
      "matrix_dim": 64,  # Collectives
      "warmup_tries": 1,
      "num_runs": 1,
      "dtype": "float32",
  }

  result = instance.run(**params)

  assert result.metadata.benchmark_name == benchmark_cls.__name__
  assert "avg_ms" in result.metrics
  assert len(result.raw_times_ms) == 1
