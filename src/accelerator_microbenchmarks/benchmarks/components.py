"""Component benchmarks for full Transformer layers."""

import dataclasses
from typing import Any

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class TransformerLayerParams(base.BaseBenchmarkParams):
  model_dim: int = 7168
  mslen: int = 1024


class ComponentBenchmark(base.BaseBenchmark[TransformerLayerParams]):
  """Base class for composite benchmarks that model full-layer behavior.

  Includes hooks for TP/CP/EP degree management and overlap tracking.
  """
  Config = TransformerLayerParams

  def __init__(
      self,
      config: TransformerLayerParams,
      hardware_spec: system.HardwareSpec,
      mesh: jax.sharding.Mesh | None = None,
      **parallelism_cfg,
  ):
    mesh = mesh or parallelism_cfg.pop("mesh", None)
    super().__init__(config=config, hardware_spec=hardware_spec, mesh=mesh)
    self._fprop = None
    # Parallelism settings from Table C1-C18
    self.tp = parallelism_cfg.get("tp", 1)
    self.etp = parallelism_cfg.get("etp", 1)
    self.cp = parallelism_cfg.get("cp", 1)
    self.ep = parallelism_cfg.get("ep", 1)


@registry.benchmark_registry.register(
    "transformer_layer_moe", is_experimental=True
)
class TransformerLayerMoE(ComponentBenchmark):
  """Comprehensive Transformer Layer benchmark representing DeepSeek-like MoE architectures.

  Encapsulates:
  - RMSNorm
  - Attention (QKV + Proj)
  - Routed Experts (FFN)
  - Residual connections
  """
  Config = TransformerLayerParams

  def setup(self):
    # In a real study, we would compose these or implement a single large JIT
    # function to capture cross-kernel optimization and memory pressure.
    # For simplicity in this study framework, we provide a unified JIT
    # operation.

    @jax.jit
    def full_layer_fwd(x, w_attn, b_attn, w_ffn, b_ffn):
      with jax.named_scope(constants.MARKER):
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
        ffn_out = jnp.matmul(ffn_out, b_ffn)
        return x + ffn_out

    self._fprop = full_layer_fwd

  def get_run_identifier(self) -> str:
    model_dim = self.config.model_dim
    mslen = self.config.mslen
    if model_dim is not None or mslen is not None:
      return f"dim_{model_dim or 7168}_len_{mslen or 1024}"
    return ""

  def generate_inputs(self) -> tuple[Any, ...]:
    model_dim = self.config.model_dim
    seq_len = self.config.mslen  # Path length after Context Parallelism
    batch = 1  # Microbatch size 1 as requested

    key = jax.random.PRNGKey(0)
    dtype = utils.parse_dtype(self.config.dtype)
    x = jax.random.normal(key, (batch, seq_len, model_dim), dtype=dtype)

    # Large-scale weights that would normally be sharded via FSDP/TP
    w_attn = jax.random.normal(key, (model_dim, model_dim * 3), dtype=dtype)
    b_attn = jax.random.normal(key, (model_dim * 3, model_dim), dtype=dtype)
    w_ffn = jax.random.normal(key, (model_dim, model_dim * 2), dtype=dtype)
    b_ffn = jax.random.normal(key, (model_dim * 2, model_dim), dtype=dtype)

    # TP Sharding: Partition model_dim across the mesh
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    mesh_axis = self.mesh.axis_names[0]
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, mesh_axis, None)
    )

    x = jax.device_put(x, sharding)
    w_attn = jax.device_put(
        w_attn,
        jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(mesh_axis, None)
        ),
    )

    return x, w_attn, b_attn, w_ffn, b_ffn

  def run_op(self, *args) -> jnp.ndarray:
    if self._fprop is None:
      raise ValueError("Forward function not initialized.")
    return self._fprop(*args)

  def get_total_bytes(self) -> float:
    model_dim = self.config.model_dim
    seq_len = self.config.mslen
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

  def get_arithmetic_intensity(self) -> float:
    model_dim = self.config.model_dim
    seq_len = self.config.mslen
    # Rough Flops: 24 * seq_len * model_dim^2
    flops = 24 * seq_len * (model_dim**2)
    return flops / self.get_total_bytes()

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    metrics = super().calculate_metrics(times_ms)
    model_dim = self.config.model_dim
    seq_len = self.config.mslen
    flops = 24 * seq_len * (model_dim**2)

    avg_latency_s = metrics["avg_ms"] / 1000.0
    metrics["tflops_per_sec"] = (flops / avg_latency_s) / 1e12
    metrics["intensity"] = self.get_arithmetic_intensity()
    return metrics
