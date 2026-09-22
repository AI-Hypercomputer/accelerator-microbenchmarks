"""Component benchmarks for full Transformer layers."""

import dataclasses
from typing import Any, Callable, Sequence

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class TransformerLayerParams(base.SingleDtypeBenchmarkParams):
  model_dim: int = dataclasses.field(
      default=7168,
      metadata={
          "min": 1,
          "help": "Model dimension size for Transformer layer.",
      },
  )
  mslen: int = dataclasses.field(
      default=1024,
      metadata={
          "min": 1,
          "help": "Micro-sequence length after Context Parallelism.",
      },
  )


class ComponentBenchmark(base.BaseBenchmark[TransformerLayerParams]):
  """Base class for composite benchmarks that model full-layer behavior.

  Includes hooks for TP/CP/EP degree management and overlap tracking.
  """
  Config = TransformerLayerParams
  roofline_mode: constants.RooflineMode = constants.RooflineMode.COMPUTE

  def __init__(
      self,
      config: TransformerLayerParams,
      hardware_spec: system.HardwareSpec,
      mesh: jax.sharding.Mesh | None = None,
      xprof_config: base.XprofConfig | None = None,
      **parallelism_cfg,
  ):
    mesh = mesh or parallelism_cfg.pop("mesh", None)
    super().__init__(
        config=config,
        hardware_spec=hardware_spec,
        mesh=mesh,
        xprof_config=xprof_config,
    )
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
  QKV_PROJ_FACTOR: int = 3
  FFN_EXPANSION_FACTOR: int = 2
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("model_dim", report.format_str),
      ("mslen", report.format_str),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_tflops_per_chip", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_tflops_per_chip", report.format_2f),
  )

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
    w_attn = jax.random.normal(
        key, (model_dim, model_dim * self.QKV_PROJ_FACTOR), dtype=dtype
    )
    b_attn = jax.random.normal(
        key, (model_dim * self.QKV_PROJ_FACTOR, model_dim), dtype=dtype
    )
    w_ffn = jax.random.normal(
        key, (model_dim, model_dim * self.FFN_EXPANSION_FACTOR), dtype=dtype
    )
    b_ffn = jax.random.normal(
        key, (model_dim * self.FFN_EXPANSION_FACTOR, model_dim), dtype=dtype
    )

    # Sequence & Tensor Parallel Sharding: partition seq_len for x and
    # model_dim for weights
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    mesh_axis = self.mesh.axis_names[0]
    x_spec = (
        jax.sharding.PartitionSpec(None, mesh_axis, None)
        if self._get_num_seq_sharded_devices() > 1
        else jax.sharding.PartitionSpec(None, None, None)
    )
    if self._get_num_sharded_devices() > 1:
      col_spec = jax.sharding.PartitionSpec(None, mesh_axis)
      row_spec = jax.sharding.PartitionSpec(mesh_axis, None)
    else:
      col_spec = jax.sharding.PartitionSpec(None, None)
      row_spec = jax.sharding.PartitionSpec(None, None)

    x = jax.device_put(x, jax.sharding.NamedSharding(self.mesh, x_spec))
    w_attn = jax.device_put(
        w_attn, jax.sharding.NamedSharding(self.mesh, col_spec)
    )
    b_attn = jax.device_put(
        b_attn, jax.sharding.NamedSharding(self.mesh, row_spec)
    )
    w_ffn = jax.device_put(
        w_ffn, jax.sharding.NamedSharding(self.mesh, col_spec)
    )
    b_ffn = jax.device_put(
        b_ffn, jax.sharding.NamedSharding(self.mesh, row_spec)
    )

    return x, w_attn, b_attn, w_ffn, b_ffn

  def run_op(self, *args) -> jnp.ndarray:
    if self._fprop is None:
      raise ValueError("Forward function not initialized.")
    return self._fprop(*args)

  def _get_num_sharded_devices(self) -> int:
    if self.mesh is None:
      return 1
    mesh_axis = self.mesh.axis_names[0]
    num_devices = int(self.mesh.shape[mesh_axis])
    if self.config.model_dim % num_devices == 0:
      return num_devices
    return 1

  def _get_num_seq_sharded_devices(self) -> int:
    if self.mesh is None:
      return 1
    mesh_axis = self.mesh.axis_names[0]
    num_devices = int(self.mesh.shape[mesh_axis])
    if self.config.mslen % num_devices == 0:
      return num_devices
    return 1

  def get_total_bytes(self) -> float:
    model_dim = self.config.model_dim
    seq_len = self.config.mslen
    dtype = utils.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    x_size = seq_len * model_dim * itemsize
    # Total weight elements across w_attn (3d^2), b_attn (3d^2),
    # w_ffn (2d^2), and b_ffn (2d^2): (3 + 3 + 2 + 2) * d^2 = 10 * d^2
    weight_dim_multiplier = 2 * (
        self.QKV_PROJ_FACTOR + self.FFN_EXPANSION_FACTOR
    )
    w_size = (model_dim * model_dim * weight_dim_multiplier) * itemsize
    x_bytes_per_dev = (10 * x_size) / self._get_num_seq_sharded_devices()
    w_bytes_per_dev = w_size / self._get_num_sharded_devices()
    return float(x_bytes_per_dev + w_bytes_per_dev)

  def get_total_flops(self) -> float:
    model_dim = self.config.model_dim
    seq_len = self.config.mslen
    # Sum of 2 * M * K * N across the 4 matmuls in full_layer_fwd (batch=1):
    # 1. qkv = x_norm @ w_attn (d -> 3d): 2 * seq_len * d * 3d = 6 * s * d^2
    # 2. attn_out = qkv @ b_attn (3d -> d): 2 * seq_len * 3d * d = 6 * s * d^2
    # 3. ffn_mid = x @ w_ffn (d -> 2d): 2 * seq_len * d * 2d = 4 * s * d^2
    # 4. ffn_out = ffn_mid @ b_ffn (2d -> d): 2 * seq_len * 2d * d = 4 * s * d^2
    # Total coefficient = 2 * (3 + 3 + 2 + 2) = 20
    qkv_proj_flops = 2 * self.QKV_PROJ_FACTOR * seq_len * (model_dim**2)
    attn_out_flops = 2 * self.QKV_PROJ_FACTOR * seq_len * (model_dim**2)
    ffn_up_flops = 2 * self.FFN_EXPANSION_FACTOR * seq_len * (model_dim**2)
    ffn_down_flops = 2 * self.FFN_EXPANSION_FACTOR * seq_len * (model_dim**2)
    global_flops = (
        qkv_proj_flops + attn_out_flops + ffn_up_flops + ffn_down_flops
    )
    divisor = max(
        self._get_num_sharded_devices(), self._get_num_seq_sharded_devices()
    )
    return float(global_flops) / divisor

  def get_arithmetic_intensity(self) -> float:
    total_bytes = self.get_total_bytes()
    return self.get_total_flops() / total_bytes if total_bytes > 0 else 0.0

  def get_workload_metadata(self) -> dict[str, Any]:
    return {
        "total_flops": self.get_total_flops(),
    }

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    flops = self.get_total_flops()

    latency_s = latency_ms / 1000.0
    if latency_s == 0:
      tflops_per_sec = float("inf")
    else:
      tflops_per_sec = (flops / latency_s) / 1e12
    return {
        f"{prefix}_tflops_per_device": tflops_per_sec,
    }
