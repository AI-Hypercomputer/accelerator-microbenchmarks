"""Model configurations for LLM benchmarks."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
  name: str
  layers: int
  seq_len: int
  attn_type: str
  num_q_heads: int
  num_kv_heads: int
  head_dim: int
  ffn_type: str
  model_dim: int
  shared_ffn_dim: int
  routed_ffn_dim: int
  num_experts: int
  experts_activated: int  # shared + top_k


# LLM Parameter Presets
MODELS = {
    "LLM-36B": ModelConfig(
        name="LLM-36B",
        layers=60,
        seq_len=8192,
        attn_type="MHA",
        num_q_heads=56,
        num_kv_heads=56,
        head_dim=128,
        ffn_type="SwiGLU",
        model_dim=7168,
        shared_ffn_dim=2048,
        routed_ffn_dim=2048,
        num_experts=256,
        experts_activated=8,  # + shared
    ),
    "LLM-100B": ModelConfig(
        name="LLM-100B",
        layers=96,
        seq_len=16384,
        attn_type="GQA",
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        ffn_type="SwiGLU",
        model_dim=8192,
        shared_ffn_dim=2048,
        routed_ffn_dim=2048,
        num_experts=512,
        experts_activated=16,
    ),
    "LLM-200B": ModelConfig(
        name="LLM-200B",
        layers=80,
        seq_len=16384,
        attn_type="GQA",
        num_q_heads=128,
        num_kv_heads=8,
        head_dim=128,
        ffn_type="SwiGLU",
        model_dim=16384,
        shared_ffn_dim=4096,
        routed_ffn_dim=4096,
        num_experts=256,
        experts_activated=8,
    ),
    "LLM-400B": ModelConfig(
        name="LLM-400B",
        layers=96,
        seq_len=16384,
        attn_type="GQA",
        num_q_heads=128,
        num_kv_heads=8,
        head_dim=128,
        ffn_type="SwiGLU",
        model_dim=16384,
        shared_ffn_dim=4096,
        routed_ffn_dim=4096,
        num_experts=512,
        experts_activated=16,
    ),
}
