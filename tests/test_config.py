"""Unit tests for config.py."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.core import config
from accelerator_microbenchmarks.core import csv_loader


def _find_all_benchmark_configs() -> list[tuple[str, str]]:
  """Discovers all benchmark YAML config files packaged in test runfiles or repo."""
  configs_dir = os.path.normpath(
      os.path.join(os.path.dirname(__file__), "..", "configs")
  )
  if not os.path.isdir(configs_dir):
    return []

  discovered = []
  for root, _, files in os.walk(configs_dir):
    for f in sorted(files):
      if not f.endswith(".yaml") or "op_flags" in f:
        continue
      full_path = os.path.join(root, f)
      rel_path = os.path.relpath(full_path, configs_dir)
      test_name = rel_path.translate(
          str.maketrans(dict.fromkeys(f"{os.sep}.-", "_"))
      )
      discovered.append((test_name, full_path))
  return sorted(discovered, key=lambda x: x[0])


class ConfigTest(parameterized.TestCase):
  """Unit tests for config.py."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.TemporaryDirectory()

  def tearDown(self):
    self.test_dir.cleanup()
    super().tearDown()

  def test_expand_sweep_no_sweep(self):
    """Simple expansion without sweeps."""
    base_params = {"a": 1, "b": 2}
    expanded = config.expand_sweep(base_params, None)

    self.assertLen(expanded, 1)
    self.assertEqual(expanded[0], {"a": 1, "b": 2})

  def test_expand_sweep_sweep_list(self):
    """Sweeps with explicit lists of values."""
    base_params = {"a": 1}
    sweep_spec = {"b": [2, 3]}
    expanded = config.expand_sweep(base_params, sweep_spec)

    self.assertLen(expanded, 2)
    self.assertEqual(expanded[0], {"a": 1, "b": 2})
    self.assertEqual(expanded[1], {"a": 1, "b": 3})

  def test_expand_sweep_range_multiplier(self):
    """Sweeps with range using multiplier."""
    base_params = {"a": 1}
    sweep_spec = {"b": {"start": 2, "end": 8, "multiplier": 2}}
    expanded = config.expand_sweep(base_params, sweep_spec)

    # b will be [2, 4, 8]
    self.assertLen(expanded, 3)
    self.assertEqual(expanded[0], {"a": 1, "b": 2})
    self.assertEqual(expanded[1], {"a": 1, "b": 4})
    self.assertEqual(expanded[2], {"a": 1, "b": 8})

  def test_expand_sweep_range_increase_by(self):
    """Sweeps with range using increase_by."""
    base_params = {"a": 1}
    sweep_spec = {"b": {"start": 2, "end": 5, "increase_by": 1}}
    expanded = config.expand_sweep(base_params, sweep_spec)

    # b will be [2, 3, 4, 5]
    self.assertLen(expanded, 4)
    self.assertEqual(expanded[0], {"a": 1, "b": 2})
    self.assertEqual(expanded[1], {"a": 1, "b": 3})
    self.assertEqual(expanded[2], {"a": 1, "b": 4})
    self.assertEqual(expanded[3], {"a": 1, "b": 5})

  def test_expand_sweep_range_start_greater_than_end_raises_error(self):
    """Verifies that range sweep with start > end raises ValueError."""
    base_params = {"a": 1}
    sweep_spec = {"b": {"start": 5, "end": 2}}
    with self.assertRaises(ValueError) as ctx:
      config.expand_sweep(base_params, sweep_spec)
    self.assertIn("start (5) > end (2)", str(ctx.exception))

    # Also verify that start == end is valid and does not raise
    valid_sweep = {"b": {"start": 5, "end": 5}}
    expanded = config.expand_sweep(base_params, valid_sweep)
    self.assertLen(expanded, 1)
    self.assertEqual(expanded[0], {"a": 1, "b": 5})

  def test_expand_sweep_combinatorial_explosion(self):
    """Capping combinations to prevent explosion."""
    base_params = {}
    sweep_spec = {
        "a": list(range(10)),
        "b": list(range(10)),
        "c": list(range(20)),
    }
    # Total combinations = 10 * 10 * 20 = 2000
    # Capped at 1000
    expanded = config.expand_sweep(base_params, sweep_spec)
    self.assertLen(expanded, 1000)

  @mock.patch.object(csv_loader, "load_cases_from_csv", spec_set=True)
  def test_load_config_with_sweeps(self, mock_load_csv):
    """Test loading a single-benchmark config with sweeps."""
    del mock_load_csv
    yaml_content = """
benchmark:
  name: all_reduce
  params:
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
    self.assertEqual(expanded[0]["param1"], 100)
    self.assertEqual(expanded[0]["param2"], 1)

    self.assertEqual(expanded[1]["name"], "all_reduce")
    self.assertEqual(expanded[1]["param2"], 2)

  def test_load_config_with_model_preset(self):
    """Test loading a config with model presets."""
    yaml_content = """
benchmark:
  name: swiglu
  model_preset: LLM-36B
"""
    config_path = os.path.join(self.test_dir.name, "config_model_preset.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    expanded = config.load_config(config_path)

    self.assertLen(expanded, 1)
    self.assertEqual(expanded[0]["name"], "swiglu")
    self.assertEqual(expanded[0]["layers"], 60)
    self.assertEqual(expanded[0]["model_dim"], 7168)

  @mock.patch.object(csv_loader, "load_cases_from_csv", spec_set=True)
  def test_load_config_with_cases_from_csv(self, mock_load_csv):
    """Test loading a config with external cases from CSV."""
    mock_load_csv.return_value = [
        {"m": 128, "n": 128, "k": 128},
        {"m": 256, "n": 256, "k": 256},
    ]
    yaml_content = """
benchmark:
  name: gemm_generalized
  params:
    in_dtype: float32
  cases_from_csv: shapes.csv
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

  def test_load_config_with_cases(self):
    """Test loading a config with explicit cases list."""
    yaml_content = """
benchmark:
  name: gemm_generalized
  params:
    warmup_tries: 2
    dtype: bfloat16
  cases:
    - m: 4096
      n: 4096
      k: 4096
    - m: 8192
      n: 8192
      k: 8192
"""
    config_path = os.path.join(self.test_dir.name, "config_cases.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    expanded = config.load_config(config_path)

    self.assertLen(expanded, 2)
    self.assertEqual(expanded[0]["name"], "gemm_generalized")
    self.assertEqual(expanded[0]["warmup_tries"], 2)
    self.assertEqual(expanded[0]["dtype"], "bfloat16")
    self.assertEqual(expanded[0]["m"], 4096)
    self.assertEqual(expanded[0]["n"], 4096)
    self.assertEqual(expanded[0]["k"], 4096)

    self.assertEqual(expanded[1]["name"], "gemm_generalized")
    self.assertEqual(expanded[1]["warmup_tries"], 2)
    self.assertEqual(expanded[1]["dtype"], "bfloat16")
    self.assertEqual(expanded[1]["m"], 8192)
    self.assertEqual(expanded[1]["n"], 8192)
    self.assertEqual(expanded[1]["k"], 8192)

  def test_load_config_cases_not_a_list_raises_error(self):
    """Verifies that non-list cases raises ValueError."""
    yaml_content = """
benchmark:
  name: gemm_generalized
  cases: "not_a_list"
"""
    config_path = os.path.join(self.test_dir.name, "config_cases_invalid.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with self.assertRaises(ValueError) as ctx:
      config.load_config(config_path)
    self.assertIn("to be a list", str(ctx.exception))

  def test_load_config_cases_item_not_a_dict_raises_error(self):
    """Verifies that non-dict items in cases raise ValueError."""
    yaml_content = """
benchmark:
  name: gemm_generalized
  cases:
    - "not_a_dict"
"""
    config_path = os.path.join(
        self.test_dir.name, "config_cases_item_invalid.yaml"
    )
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with self.assertRaises(ValueError) as ctx:
      config.load_config(config_path)
    self.assertIn("to be a dict", str(ctx.exception))

  def test_load_config_four_stage_precedence(self):
    """Verifies deterministic 4-stage parameter precedence.

    Stage 1: Model Preset (LLM-36B defaults)
    Stage 2: params: (baseline & explicit overrides over preset)
    Stage 3: cases: (per-case parameter overrides)
    Stage 4: sweep: (Cartesian product variations across resolved cases)
    """
    yaml_content = """
benchmark:
  name: attention_flashed
  model_preset: LLM-36B
  params:
    warmup_tries: 2
    num_runs: 5
    dtype: bfloat16
    seq_len: 4096
  cases:
    - head_dim: 64
    - head_dim: 256
  sweep:
    causal: [true, false]
"""
    config_path = os.path.join(self.test_dir.name, "config_precedence.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    expanded = config.load_config(config_path)

    # 2 cases * 2 sweep choices = 4 total configurations
    self.assertLen(expanded, 4)

    # Point 0: Case 0 (head_dim=64), causal=True
    self.assertEqual(expanded[0]["name"], "attention_flashed")
    self.assertEqual(expanded[0]["layers"], 60)  # Stage 1: from LLM-36B
    self.assertEqual(
        expanded[0]["seq_len"], 4096
    )  # Stage 2: params override (was 8192)
    self.assertEqual(expanded[0]["warmup_tries"], 2)  # Stage 2: params
    self.assertEqual(
        expanded[0]["head_dim"], 64
    )  # Stage 3: case 0 override (was 128)
    self.assertIs(expanded[0]["causal"], True)  # Stage 4: sweep

    # Point 1: Case 0 (head_dim=64), causal=False
    self.assertEqual(expanded[1]["head_dim"], 64)
    self.assertIs(expanded[1]["causal"], False)

    # Point 2: Case 1 (head_dim=256), causal=True
    self.assertEqual(expanded[2]["head_dim"], 256)  # Stage 3: case 1 override
    self.assertIs(expanded[2]["causal"], True)

    # Point 3: Case 1 (head_dim=256), causal=False
    self.assertEqual(expanded[3]["head_dim"], 256)
    self.assertIs(expanded[3]["causal"], False)

  def test_load_config_mutual_exclusion_cases_and_cases_from_csv_raises_error(
      self,
  ):
    """Verifies that specifying both cases and cases_from_csv raises ValueError."""
    yaml_content = """
benchmark:
  name: gemm_generalized
  cases:
    - m: 1024
  cases_from_csv: shapes.csv
"""
    config_path = os.path.join(self.test_dir.name, "config_mutual_excl.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with self.assertRaises(ValueError) as ctx:
      config.load_config(config_path)
    self.assertIn(
        "cannot specify both 'cases' and 'cases_from_csv'", str(ctx.exception)
    )

  @parameterized.named_parameters(
      (
          "reserved_key",
          """
benchmark:
  name: gemm_generalized
  sweep:
    name: [gemm_a, gemm_b]
""",
          "reserved key(s)",
          None,
      ),
      (
          "params_collision",
          """
benchmark:
  name: gemm_generalized
  params:
    m: 1024
  sweep:
    m: [1024, 2048]
""",
          "collide with keys in 'params'",
          None,
      ),
      (
          "cases_collision",
          """
benchmark:
  name: gemm_generalized
  cases:
    - m: 1024
      n: 512
  sweep:
    m: [1024, 2048]
""",
          "collide with keys in 'cases'",
          None,
      ),
      (
          "cases_from_csv_collision",
          """
benchmark:
  name: gemm_generalized
  cases_from_csv: shapes.csv
  sweep:
    m: [1024, 2048]
""",
          "collide with keys in 'cases_from_csv'",
          [{"m": 1024, "n": 1024}],
      ),
  )
  def test_load_config_sweep_key_collision_raises_error(
      self, yaml_content, expected_error_msg, mock_csv_cases
  ):
    """Verifies sweep key collisions raise ValueError."""
    config_path = os.path.join(self.test_dir.name, "config_collision.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with mock.patch.object(
        csv_loader,
        "load_cases_from_csv",
        spec_set=True,
        return_value=mock_csv_cases or [],
    ):
      with self.assertRaises(ValueError) as ctx:
        config.load_config(config_path)
    self.assertIn(expected_error_msg, str(ctx.exception))

  def test_load_config_sweep_overrides_model_preset(self):
    """Verifies that sweep can override and sweep parameters defined in model_preset."""
    yaml_content = """
benchmark:
  name: attention_flashed
  model_preset: LLM-36B
  params:
    warmup_tries: 2
    num_runs: 5
    dtype: bfloat16
  sweep:
    seq_len: [2048, 4096]
"""
    config_path = os.path.join(
        self.test_dir.name, "config_sweep_model_preset.yaml"
    )
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    expanded = config.load_config(config_path)
    self.assertLen(expanded, 2)
    self.assertEqual(expanded[0]["seq_len"], 2048)
    self.assertEqual(expanded[1]["seq_len"], 4096)
    # Ensure other model preset defaults are preserved
    self.assertEqual(expanded[0]["model_dim"], 7168)
    self.assertEqual(expanded[1]["model_dim"], 7168)

  def test_load_config_unrecognized_key_in_benchmark_raises_error(self):
    """Verifies that unrecognized keys directly under benchmark raise ValueError."""
    yaml_content = """
benchmark:
  name: gemm_generalized
  warmup_tries: 2
"""
    config_path = os.path.join(
        self.test_dir.name, "config_unrecognized_key.yaml"
    )
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with self.assertRaises(ValueError) as ctx:
      config.load_config(config_path)
    self.assertIn(
        "Unrecognized key(s) in 'benchmark' mapping", str(ctx.exception)
    )
    self.assertIn(
        "Baseline execution parameters should be placed under 'params:'",
        str(ctx.exception),
    )

  def test_load_config_unknown_model_preset_raises_error(self):
    """Verifies that an unknown model preset raises ValueError."""
    yaml_content = """
benchmark:
  name: swiglu
  model_preset: NONEXISTENT_MODEL
"""
    config_path = os.path.join(
        self.test_dir.name, "config_unknown_model_preset.yaml"
    )
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with self.assertRaises(ValueError) as ctx:
      config.load_config(config_path)
    self.assertIn(
        "Unknown model preset 'NONEXISTENT_MODEL'", str(ctx.exception)
    )

  def test_benchmark_key_enum(self):
    """Verifies all BenchmarkKey enum values."""
    self.assertEqual(config.BenchmarkKey.NAME, "name")
    self.assertEqual(config.BenchmarkKey.MODEL_PRESET, "model_preset")
    self.assertEqual(config.BenchmarkKey.PARAMS, "params")
    self.assertEqual(config.BenchmarkKey.CASES, "cases")
    self.assertEqual(config.BenchmarkKey.CASES_FROM_CSV, "cases_from_csv")
    self.assertEqual(config.BenchmarkKey.SWEEP, "sweep")
    self.assertEqual(
        {k.value for k in config.BenchmarkKey},
        {"name", "model_preset", "params", "cases", "cases_from_csv", "sweep"},
    )

  def test_load_config_params_not_a_dict_raises_error(self):
    """Verifies that non-dict params raises ValueError."""
    yaml_content = """
benchmark:
  name: gemm_generalized
  params: "not_a_dict"
"""
    config_path = os.path.join(
        self.test_dir.name, "config_params_not_dict.yaml"
    )
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with self.assertRaises(ValueError) as ctx:
      config.load_config(config_path)
    self.assertIn("Expected 'params' in benchmark", str(ctx.exception))
    self.assertIn("to be a dict", str(ctx.exception))

  def test_load_config_params_reserved_key_raises_error(self):
    """Verifies that params containing reserved keys raises ValueError."""
    yaml_content = """
benchmark:
  name: gemm_generalized
  params:
    sweep:
      m: [1024]
"""
    config_path = os.path.join(
        self.test_dir.name, "config_params_reserved.yaml"
    )
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with self.assertRaises(ValueError) as ctx:
      config.load_config(config_path)
    self.assertIn("reserved key(s)", str(ctx.exception))

  def test_load_config_sweep_invalid_type_raises_error(self):
    """Verifies that non-dict sweep raises ValueError."""
    yaml_content = """
benchmark:
  name: gemm_generalized
  sweep: "not_a_dict"
"""
    config_path = os.path.join(
        self.test_dir.name, "config_sweep_invalid_type.yaml"
    )
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with self.assertRaises(ValueError) as ctx:
      config.load_config(config_path)
    self.assertIn("Expected 'sweep' in benchmark", str(ctx.exception))
    self.assertIn("to be a non-empty dict", str(ctx.exception))

  def test_load_config_sweep_empty_mapping_raises_error(self):
    """Verifies that empty sweep mapping raises ValueError."""
    yaml_content = """
benchmark:
  name: gemm_generalized
  sweep: {}
"""
    config_path = os.path.join(self.test_dir.name, "config_sweep_empty.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
      f.write(yaml_content)

    with self.assertRaises(ValueError) as ctx:
      config.load_config(config_path)
    self.assertIn("Expected 'sweep' in benchmark", str(ctx.exception))
    self.assertIn("to be a non-empty dict", str(ctx.exception))

  @parameterized.named_parameters(_find_all_benchmark_configs())
  def test_all_configs_load_successfully(self, config_path):
    """Verifies that every YAML config loads and expands successfully."""
    real_load_csv = csv_loader.load_cases_from_csv

    def _load_csv_for_test(csv_path):
      if not os.path.exists(csv_path) and not csv_path.startswith(
          ("http://", "https://")
      ):
        rel_resolved = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", csv_path)
        )
        if os.path.exists(rel_resolved):
          return real_load_csv(rel_resolved)
      return real_load_csv(csv_path)

    with mock.patch.object(
        csv_loader, "load_cases_from_csv", side_effect=_load_csv_for_test
    ):
      expanded = config.load_config(config_path)
    self.assertNotEmpty(expanded)
    for item in expanded:
      self.assertIn("name", item)


if __name__ == "__main__":
  absltest.main()
