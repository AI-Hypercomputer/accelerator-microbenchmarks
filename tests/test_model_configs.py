"""Unit tests for model_configs.py."""

import dataclasses

from absl.testing import absltest
from accelerator_microbenchmarks.core import model_configs


class ModelConfigsTest(absltest.TestCase):
  """Unit tests for model_configs.py."""

  def test_model_presets_exist(self):
    """Verify that presets exist and have correct types."""
    self.assertIn("LLM-36B", model_configs.MODELS)
    self.assertIn("LLM-100B", model_configs.MODELS)
    self.assertIn("LLM-200B", model_configs.MODELS)
    self.assertIn("LLM-400B", model_configs.MODELS)

    for name, config in model_configs.MODELS.items():
      self.assertEqual(name, config.name)
      self.assertIsInstance(config, model_configs.ModelConfig)
      self.assertGreater(config.layers, 0)
      self.assertGreater(config.seq_len, 0)
      self.assertGreater(config.model_dim, 0)

  def test_convert_to_dict(self):
    """Verify conversion to dictionary using dataclasses.asdict."""
    config = model_configs.MODELS["LLM-36B"]
    config_dict = dataclasses.asdict(config)

    self.assertEqual(config_dict["name"], "LLM-36B")
    self.assertEqual(config_dict["layers"], 60)
    self.assertEqual(config_dict["seq_len"], 8192)
    self.assertEqual(config_dict["attn_type"], "MHA")
    self.assertEqual(config_dict["num_q_heads"], 56)
    self.assertEqual(config_dict["num_kv_heads"], 56)
    self.assertEqual(config_dict["head_dim"], 128)
    self.assertEqual(config_dict["ffn_type"], "SwiGLU")
    self.assertEqual(config_dict["model_dim"], 7168)
    self.assertEqual(config_dict["shared_ffn_dim"], 2048)
    self.assertEqual(config_dict["routed_ffn_dim"], 2048)
    self.assertEqual(config_dict["num_experts"], 256)
    self.assertEqual(config_dict["experts_activated"], 8)


if __name__ == "__main__":
  absltest.main()
