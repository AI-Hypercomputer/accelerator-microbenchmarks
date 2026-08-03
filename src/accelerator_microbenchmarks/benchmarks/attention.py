"""Attention benchmarks."""

import dataclasses
from typing import Any
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class AttentionParams(base.BaseBenchmarkParams):
  mode: str = "fwd"
  causal: bool = True
  batch: int = 1
  seq_len: int = 8192
  num_q_heads: int = 56
  num_kv_heads: int = 56
  head_dim: int = 128


@registry.benchmark_registry.register("attention_flashed")


class AttentionBenchmark(base.BaseBenchmark[AttentionParams]):
  Config = AttentionParams
  """Attention benchmark simulating FlashAttention behavior.

  Supports:
  - MHA/GQA
  - BF16 compute
  - Causal masking
  """

  def setup(self):
    mode = self.config.mode
    causal = self.config.causal

    @jax.jit
    def attention_fwd(q, k, v, mask=None):
      with jax.named_scope(constants.MARKER):
        # jax.nn.dot_product_attention is the standard recommended path for TPU
        # which leverages optimized Flash-like kernels under the hood.
        return jax.nn.dot_product_attention(
            q, k, v, mask=mask, is_causal=causal
        )

    if mode == "fwd":
      self._jit_fn = attention_fwd
    elif mode == "bwd":

      @jax.jit
      def attention_bwd(q, k, v, mask=None):
        with jax.named_scope(constants.MARKER):
          out, vjp_fn = jax.vjp(
              lambda q_, k_, v_: jax.nn.dot_product_attention(
                  q_, k_, v_, mask=mask, is_causal=causal
              ),
              q,
              k,
              v,
          )
          grad_out = jnp.ones_like(out)
          dq, dk, dv = vjp_fn(grad_out)
          return dq, dk, dv

      self._jit_fn = attention_bwd
    else:
      raise ValueError(f"Unknown mode: {mode}")

  def get_run_identifier(self) -> str:
    batch = self.config.batch
    seq_len = self.config.seq_len
    num_q_heads = self.config.num_q_heads
    num_kv_heads = self.config.num_kv_heads
    head_dim = self.config.head_dim
    if any(
        v is not None
        for v in (batch, seq_len, num_q_heads, num_kv_heads, head_dim)
    ):
      return (
          f"b_{batch or 1}_s_{seq_len or 8192}_hq_{num_q_heads or 56}_hkv_{num_kv_heads or 56}_d_{head_dim or 128}"
      )
    return ""

  def generate_inputs(self) -> tuple[Any, ...]:
    batch = self.config.batch
    seq_len = self.config.seq_len
    heads_q = self.config.num_q_heads
    heads_kv = self.config.num_kv_heads
    head_dim = self.config.head_dim
    dtype = utils.parse_dtype(self.config.dtype)

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    q = jax.random.normal(k1, (batch, heads_q, seq_len, head_dim), dtype=dtype)
    k = jax.random.normal(k2, (batch, heads_kv, seq_len, head_dim), dtype=dtype)
    v = jax.random.normal(k3, (batch, heads_kv, seq_len, head_dim), dtype=dtype)

    # Parallelize across heads as per TPU best practices for MHA.
    # Shard on head dimension if it is divisible by the number of devices.
    # Otherwise, replicate.
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")

    mesh_axis = self.mesh.axis_names[0]
    num_devices = self.mesh.shape[mesh_axis]

    if heads_q % num_devices == 0:
      q_spec = jax.sharding.PartitionSpec(None, mesh_axis, None, None)
    else:
      q_spec = jax.sharding.PartitionSpec(None, None, None, None)

    if heads_kv % num_devices == 0:
      kv_spec = jax.sharding.PartitionSpec(None, mesh_axis, None, None)
    else:
      kv_spec = jax.sharding.PartitionSpec(None, None, None, None)

    q_sharding = jax.sharding.NamedSharding(self.mesh, q_spec)
    kv_sharding = jax.sharding.NamedSharding(self.mesh, kv_spec)

    q = jax.device_put(q, q_sharding)
    k = jax.device_put(k, kv_sharding)
    v = jax.device_put(v, kv_sharding)

    return q, k, v

  def run_op(self, q, k, v) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(q, k, v)

  def get_total_bytes(self) -> float:
    batch = self.config.batch
    q_len = self.config.seq_len
    kv_len = q_len
    heads_q = self.config.num_q_heads
    heads_kv = self.config.num_kv_heads
    head_dim = self.config.head_dim
    itemsize = jnp.dtype(jnp.bfloat16).itemsize
    mode = self.config.mode

    if mode == "fwd":
      # Bytes = Load(Q, K, V) + Store(Out)
      return batch * (
          (heads_q * q_len * head_dim * itemsize)  # Q
          + (heads_kv * kv_len * head_dim * itemsize)  # K
          + (heads_kv * kv_len * head_dim * itemsize)  # V
          + (heads_q * q_len * head_dim * itemsize)  # Out
      )
    elif mode == "bwd":
      # Bytes = Load(Q, K, V, Out, dOut) + Store(dQ, dK, dV)
      return batch * (
          2 * (heads_q * q_len * head_dim * itemsize)  # Q + dQ
          + 2 * (heads_kv * kv_len * head_dim * itemsize)  # K + dK
          + 2 * (heads_kv * kv_len * head_dim * itemsize)  # V + dV
          + (heads_q * q_len * head_dim * itemsize)  # Out
          + (heads_q * q_len * head_dim * itemsize)  # dOut
      )
    else:
      raise ValueError(f"Unknown mode: {mode}")

  def get_arithmetic_intensity(self) -> float:
    q_len = self.config.seq_len
    kv_len = q_len
    heads = self.config.num_q_heads
    head_dim = self.config.head_dim
    causal = self.config.causal
    mode = self.config.mode

    if causal:
      # (4 * Q * K - 2 * Q * Q) * Heads * HeadDim
      flops = (4 * q_len * kv_len - 2 * q_len * q_len) * heads * head_dim
    else:
      flops = 4 * q_len * kv_len * heads * head_dim

    if mode == "bwd":
      flops *= 2

    return flops / self.get_total_bytes()

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    metrics = super().calculate_metrics(times_ms)
    q_len = self.config.seq_len
    kv_len = q_len
    heads = self.config.num_q_heads
    head_dim = self.config.head_dim
    causal = self.config.causal
    mode = self.config.mode

    if causal:
      total_flops = (4 * q_len * kv_len - 2 * q_len * q_len) * heads * head_dim
    else:
      total_flops = 4 * q_len * kv_len * heads * head_dim

    if mode == "bwd":
      total_flops *= 2

    avg_latency_s = metrics["avg_ms"] / 1000.0
    tflops_per_sec = (total_flops / avg_latency_s) / 1e12

    metrics["tflops_per_sec"] = tflops_per_sec
    metrics["total_flops"] = total_flops
    metrics["intensity"] = self.get_arithmetic_intensity()
    return metrics
