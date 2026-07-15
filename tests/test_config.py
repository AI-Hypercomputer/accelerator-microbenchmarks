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
  def test_load_config(self, mock_load_csv):
    """Test loading a hierarchical config."""
    yaml_content = """
global:
  param1: 100
hardware:
  tflops: 50.0
benchmarks:
  - name: bm1
    sweep:
      param2: [1, 2]
  - name: bm2
    model: LLM-36B
  - name: bm3
    csv_shapes: shapes.csv
"""
    config_path = os.path.join(self.test_dir.name, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    # Mock CSV loader to return one row
    mock_load_csv.return_value = [{"csv_param": "value"}]

    expanded = config.load_config(config_path)

    # bm1: 2 combinations (sweep)
    # bm2: 1 combination (model LLM-36B)
    # bm3: 1 combination (from mocked CSV)
    # Total = 2 + 1 + 1 = 4
    self.assertLen(expanded, 4)

    # Check bm1 expansion
    self.assertEqual(expanded[0]["name"], "bm1")
    self.assertEqual(expanded[0]["param1"], 100)
    self.assertEqual(expanded[0]["hardware_stats"], {"tflops": 50.0})
    self.assertEqual(expanded[0]["param2"], 1)

    self.assertEqual(expanded[1]["name"], "bm1")
    self.assertEqual(expanded[1]["param2"], 2)

    # Check bm2 expansion (model presets)
    self.assertEqual(expanded[2]["name"], "bm2")
    self.assertEqual(expanded[2]["layers"], 60)
    self.assertEqual(expanded[2]["model_dim"], 7168)

    # Check bm3 expansion (CSV shapes)
    self.assertEqual(expanded[3]["name"], "bm3")
    self.assertEqual(expanded[3]["csv_param"], "value")


if __name__ == "__main__":
  absltest.main()
