"""Unit tests for config.py."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import config


class ConfigTest(absltest.TestCase):
  """Unit tests for config.py."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.TemporaryDirectory()

  def tearDown(self):
    self.test_dir.cleanup()
    super().tearDown()

  def test_resolve_params_no_sweep(self):
    """Simple merging without sweeps."""
    base_params = {"a": 1, "b": 2}
    entry = {"b": 3, "c": 4}
    resolved = config.resolve_params(base_params, entry)

    self.assertLen(resolved, 1)
    self.assertEqual(resolved[0], {"a": 1, "b": 3, "c": 4})

  def test_resolve_params_sweep_list(self):
    """Sweeps with explicit lists of values."""
    base_params = {"a": 1}
    entry = {"sweep": {"b": [2, 3]}}
    resolved = config.resolve_params(base_params, entry)

    self.assertLen(resolved, 2)
    self.assertEqual(resolved[0], {"a": 1, "b": 2})
    self.assertEqual(resolved[1], {"a": 1, "b": 3})

  def test_resolve_params_sweep_range_multiplier(self):
    """Sweeps with range using multiplier."""
    base_params = {"a": 1}
    entry = {"sweep": {"b": {"start": 2, "end": 8, "multiplier": 2}}}
    resolved = config.resolve_params(base_params, entry)

    # b will be [2, 4, 8]
    self.assertLen(resolved, 3)
    self.assertEqual(resolved[0], {"a": 1, "b": 2})
    self.assertEqual(resolved[1], {"a": 1, "b": 4})
    self.assertEqual(resolved[2], {"a": 1, "b": 8})

  def test_resolve_params_sweep_range_increase_by(self):
    """Sweeps with range using increase_by."""
    base_params = {"a": 1}
    entry = {"sweep": {"b": {"start": 2, "end": 5, "increase_by": 1}}}
    resolved = config.resolve_params(base_params, entry)

    # b will be [2, 3, 4, 5]
    self.assertLen(resolved, 4)
    self.assertEqual(resolved[0], {"a": 1, "b": 2})
    self.assertEqual(resolved[1], {"a": 1, "b": 3})
    self.assertEqual(resolved[2], {"a": 1, "b": 4})
    self.assertEqual(resolved[3], {"a": 1, "b": 5})

  def test_resolve_params_combinatorial_explosion(self):
    """Capping combinations to prevent explosion."""
    base_params = {}
    entry = {
        "sweep": {
            "a": list(range(10)),
            "b": list(range(10)),
            "c": list(range(20)),
        }
    }
    # Total combinations = 10 * 10 * 20 = 2000
    # Capped at 1000
    resolved = config.resolve_params(base_params, entry)
    self.assertLen(resolved, 1000)

  @mock.patch(
      "accelerator_microbenchmarks.core.csv_loader.load_shapes_from_csv"
  )
  def test_load_config_with_sweeps(self, mock_load_csv):
    """Test loading a single-benchmark config with sweeps and hardware."""
    del mock_load_csv
    yaml_content = """
system: ironwood
hardware:
  tflops: 50.0

benchmark:
  name: all_reduce
  param1: 100
  sweep:
    param2: [1, 2]
"""
    config_path = os.path.join(self.test_dir.name, "config_sweep.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    expanded = config.load_config(config_path)

    self.assertLen(expanded, 2)
    self.assertEqual(expanded[0]["name"], "all_reduce")
    self.assertEqual(expanded[0]["system"], "ironwood")
    self.assertEqual(expanded[0]["param1"], 100)
    self.assertEqual(expanded[0]["hardware_stats"], {"tflops": 50.0})
    self.assertEqual(expanded[0]["param2"], 1)

    self.assertEqual(expanded[1]["name"], "all_reduce")
    self.assertEqual(expanded[1]["param2"], 2)

  def test_load_config_with_model(self):
    """Test loading a config with model presets."""
    yaml_content = """
benchmark:
  name: swiglu
  model: LLM-36B
"""
    config_path = os.path.join(self.test_dir.name, "config_model.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    expanded = config.load_config(config_path)

    self.assertLen(expanded, 1)
    self.assertEqual(expanded[0]["name"], "swiglu")
    self.assertEqual(expanded[0]["layers"], 60)
    self.assertEqual(expanded[0]["model_dim"], 7168)

  @mock.patch(
      "accelerator_microbenchmarks.core.csv_loader.load_shapes_from_csv"
  )
  def test_load_config_with_csv_shapes(self, mock_load_csv):
    """Test loading a config with CSV shapes."""
    mock_load_csv.return_value = [
        {"m": 128, "n": 128, "k": 128},
        {"m": 256, "n": 256, "k": 256},
    ]
    yaml_content = """
benchmark:
  name: gemm_generalized
  in_dtype: float32
  csv_shapes: shapes.csv
"""
    config_path = os.path.join(self.test_dir.name, "config_csv.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    expanded = config.load_config(config_path)

    self.assertLen(expanded, 2)
    self.assertEqual(expanded[0]["name"], "gemm_generalized")
    self.assertEqual(expanded[0]["in_dtype"], "float32")
    self.assertEqual(expanded[0]["m"], 128)
    self.assertEqual(expanded[1]["name"], "gemm_generalized")
    self.assertEqual(expanded[1]["m"], 256)

  def test_load_config_missing_benchmark_raises_error(self):
    """Verifies that missing benchmark task name raises ValueError."""
    yaml_content = """
warmup_tries: 2
num_runs: 10
"""
    config_path = os.path.join(self.test_dir.name, "config_missing.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with self.assertRaises(ValueError) as ctx:
      config.load_config(config_path)
    self.assertIn("must define a 'benchmark:' mapping", str(ctx.exception))


if __name__ == "__main__":
  absltest.main()


