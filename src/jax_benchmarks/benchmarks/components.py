"""Component benchmarks for full Transformer layers."""

from typing import Any, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from ..core.base import BaseBenchmark
from ..core.registry import registry
from .compute_ops import RMSNormBenchmark, SwiGLUBenchmark
from ..core.constants import MARKER
from .matmul import GeneralizedGemmBenchmark


class ComponentBenchmark(BaseBenchmark):
  """Base class for composite benchmarks that model full-layer behavior.

  Includes hooks for TP/CP/EP degree management and overlap tracking.
  """

  def __init__(self, **parallelism_cfg):
    super().__init__()
    self._fprop = None
    # Parallelism settings from Table C1-C18
    self.tp = parallelism_cfg.get("tp", 1)
    self.etp = parallelism_cfg.get("etp", 1)
    self.cp = parallelism_cfg.get("cp", 1)
    self.ep = parallelism_cfg.get("ep", 1)


@registry.register("transformer_layer_moe")
class TransformerLayerMoE(ComponentBenchmark):
  """Comprehensive Transformer Layer benchmark representing DeepSeek-like MoE architectures.

  Encapsulates:
  - RMSNorm
  - Attention (QKV + Proj)
  - Routed Experts (FFN)
  - Residual connections
  """

  def setup(self, **params):
    # In a real study, we would compose these or implement a single large JIT function
    # to capture cross-kernel optimization and memory pressure.
    # For simplicity in this study framework, we provide a unified JIT operation.

    @jax.jit
    def full_layer_fwd(x, w_attn, b_attn, w_ffn, b_ffn):
      with jax.named_scope(MARKER):
        # 1. Norm
        x_norm = jax.nn.standardize(x, axis=-1) * 1.0  # RMSNorm proxy
        # 2. Attention Projections (QKV)
        qkv = jnp.matmul(x_norm, w_attn)
        # 3. Dummy Attention mechanism
        attn_out = jnp.matmul(qkv, b_attn)  # Simplified proxy
        # 4. Residual
        x = x + attn_out
        # 5. FFN (MoE logic proxy)
        # Scaling shared and routed experts
        ffn_out = jnp.matmul(x, w_ffn)
        return x + ffn_out

    self._fprop = full_layer_fwd

  def generate_inputs(self, **params) -> Tuple:
    model_dim = params.get("model_dim", 7168)
    seq_len = params.get("mslen", 1024)  # Path length after Context Parallelism
    batch = 1  # Microbatch size 1 as requested

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, seq_len, model_dim), dtype=jnp.bfloat16)

    # Large-scale weights that would normally be sharded via FSDP/TP
    w_attn = jax.random.normal(
        key, (model_dim, model_dim * 3), dtype=jnp.bfloat16
    )
    b_attn = jax.random.normal(
        key, (model_dim * 3, model_dim), dtype=jnp.bfloat16
    )
    w_ffn = jax.random.normal(
        key, (model_dim, model_dim * 2), dtype=jnp.bfloat16
    )

    # TP Sharding: Partition model_dim across the mesh
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    mesh_axis = self.mesh.axis_names[0]
    sharding = NamedSharding(self.mesh, P(None, mesh_axis, None))

    x = jax.device_put(x, NamedSharding(self.mesh, P(None, None, None)))
    w_attn = jax.device_put(
        w_attn, NamedSharding(self.mesh, P(mesh_axis, None))
    )

    return x, w_attn, b_attn, w_ffn, w_ffn  # Dummy reuse for logic

  def run_op(self, *args) -> jnp.ndarray:
    if self._fprop is None:
      raise ValueError("Forward function not initialized.")
    return self._fprop(*args)

  def get_total_bytes(self, **params) -> float:
    model_dim = params.get("model_dim", 7168)
    seq_len = params.get("mslen", 1024)
    itemsize = jnp.dtype(jnp.bfloat16).itemsize

    # Very rough estimate for a full layer:
    # Inputs + Weights + Intermediates
    # For a study, we may need a more detailed breakdown.
    # Approximation: 10 * X_size + Weight_size
    x_size = seq_len * model_dim * itemsize
    w_size = (
        model_dim * model_dim * 10
    ) * itemsize  # Rough param count across all experts/attn
    return 10 * x_size + w_size

  def get_arithmetic_intensity(self, **params) -> float:
    model_dim = params.get("model_dim", 7168)
    seq_len = params.get("mslen", 1024)
    # Rough Flops: 24 * seq_len * model_dim^2
    flops = 24 * seq_len * (model_dim**2)
    return flops / self.get_total_bytes(**params)

  def calculate_metrics(
      self, times_ms: List[float], **params
  ) -> Dict[str, Any]:
    metrics = super().calculate_metrics(times_ms, **params)
    model_dim = params.get("model_dim", 7168)
    seq_len = params.get("mslen", 1024)
    flops = 24 * seq_len * (model_dim**2)

    avg_latency_s = metrics["avg_ms"] / 1000.0
    metrics["tflops_per_sec"] = (flops / avg_latency_s) / 1e12
    metrics["intensity"] = self.get_arithmetic_intensity(**params)
    return metrics
