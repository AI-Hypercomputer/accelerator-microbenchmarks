"""Unit tests for core benchmarking logic."""

import os
from core.base import BaseBenchmark
from core.config import load_config
from core.registry import registry
import pytest
import yaml


def test_registry():
  """Test benchmark registration and retrieval."""

  @registry.register("test_benchmark_unique")
  class TestBenchmark(BaseBenchmark):

    def run_op(self, *args, **kwargs):
      return None

  assert "test_benchmark_unique" in registry.list_benchmarks()
  cls = registry.get_benchmark("test_benchmark_unique")
  assert cls == TestBenchmark

  with pytest.raises(ValueError):

    @registry.register("test_benchmark_unique")
    class Duplicate(BaseBenchmark):
      pass


def test_config_expansion(tmp_path):
  """Test hierarchical config loading and sweep expansion."""
  config_file = tmp_path / "config.yaml"
  config_content = {
      "global": {"warmup": 10},
      "benchmarks": [
          {"name": "b1", "sweep": {"m": [128, 256]}},
          {"name": "b2", "k": 512},
      ],
  }
  with open(config_file, "w") as f:
    yaml.dump(config_content, f)

  configs = load_config(str(config_file))

  assert len(configs) == 3  # 2 for b1 (sweep), 1 for b2

  # Check b1 entries
  b1_entries = [c for c in configs if c["name"] == "b1"]
  assert len(b1_entries) == 2
  assert b1_entries[0]["m"] == 128
  assert b1_entries[1]["m"] == 256
  assert b1_entries[0]["warmup"] == 10

  # Check b2
  b2_entry = next(c for c in configs if c["name"] == "b2")
  assert b2_entry["k"] == 512
  assert b2_entry["warmup"] == 10
