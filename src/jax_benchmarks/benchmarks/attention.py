"""Attention benchmarks."""

from typing import Any, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from ..core.base import BaseBenchmark
from ..core.registry import registry
from ..core.constants import MARKER

@registry.register("attention_flashed")
class AttentionBenchmark(BaseBenchmark):
  """Attention benchmark simulating FlashAttention behavior.

  Supports:
  - MHA/GQA
  - BF16 compute
  - Causal masking
  """

  def setup(self, **params):
    @jax.jit
    def attention_fn(q, k, v, mask=None):
      with jax.named_scope(MARKER):
        # jax.nn.dot_product_attention is the standard recommended path for TPU
        # which leverages optimized Flash-like kernels under the hood.
        return jax.nn.dot_product_attention(
            q, k, v, mask=mask, is_causal=params.get("causal", True)
        )

    self._jit_fn = attention_fn

  def generate_inputs(self, **params) -> Tuple:
    batch = params.get("batch", 1)
    seq_len = params.get("seq_len", 8192)
    heads_q = params.get("num_q_heads", 56)
    heads_kv = params.get("num_kv_heads", 56)
    head_dim = params.get("head_dim", 128)
    dtype = jnp.bfloat16

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (batch, heads_q, seq_len, head_dim), dtype=dtype)
    k = jax.random.normal(k2, (batch, heads_kv, seq_len, head_dim), dtype=dtype)
    v = jax.random.normal(k3, (batch, heads_kv, seq_len, head_dim), dtype=dtype)

    # Parallelize across heads as per TPU best practices for MHA
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    sharding = NamedSharding(
        self.mesh, P(self.mesh.axis_names[0], None, None, None)
    )
    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)

    return q, k, v

  def run_op(self, q, k, v) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(q, k, v)

  def get_total_bytes(self, **params) -> float:
    batch = params.get("batch", 1)
    q_len = params.get("seq_len", 8192)
    kv_len = q_len
    heads_q = params.get("num_q_heads", 56)
    heads_kv = params.get("num_kv_heads", 56)
    head_dim = params.get("head_dim", 128)
    itemsize = jnp.dtype(jnp.bfloat16).itemsize

    # Bytes = Load(Q, K, V) + Store(Out)
    return batch * (
        (heads_q * q_len * head_dim * itemsize)  # Q
        + (heads_kv * kv_len * head_dim * itemsize)  # K
        + (heads_kv * kv_len * head_dim * itemsize)  # V
        + (heads_q * q_len * head_dim * itemsize)  # Out
    )

  def get_arithmetic_intensity(self, **params) -> float:
    q_len = params.get("seq_len", 8192)
    kv_len = q_len
    heads = params.get("num_q_heads", 56)
    head_dim = params.get("head_dim", 128)
    # (4 * Q * K - 2 * Q * Q) * Heads * HeadDim
    flops = (4 * q_len * kv_len - 2 * q_len * q_len) * heads * head_dim
    return flops / self.get_total_bytes(**params)

  def calculate_metrics(
      self, times_ms: List[float], **params
  ) -> Dict[str, Any]:
    metrics = super().calculate_metrics(times_ms, **params)
    q_len = params.get("seq_len", 8192)
    kv_len = q_len
    heads = params.get("num_q_heads", 56)
    head_dim = params.get("head_dim", 128)

    # Meta's Flops formula for causal attention:
    # (4 * Q * K - 2 * Q * Q) * Heads * HeadDim
    # Simplified for square q_len == kv_len:
    total_flops = (4 * q_len * kv_len - 2 * q_len * q_len) * heads * head_dim

    avg_latency_s = metrics["avg_ms"] / 1000.0
    tflops_per_sec = (total_flops / avg_latency_s) / 1e12

    metrics["tflops_per_sec"] = tflops_per_sec
    metrics["total_flops"] = total_flops
    metrics["intensity"] = self.get_arithmetic_intensity(**params)
    return metrics
