"""Attention benchmarks."""

import dataclasses
from typing import Any, Callable, Sequence
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp


class AttentionMode(constants.ParamEnum):
  """Execution mode for attention benchmark."""

  FWD = "fwd"
  BWD = "bwd"


@dataclasses.dataclass
class AttentionParams(base.SingleDtypeBenchmarkParams):
  mode: AttentionMode = dataclasses.field(
      default=AttentionMode.FWD,
      metadata={
          "help": "Attention execution pass (forward or backward).",
      },
  )
  causal: bool = dataclasses.field(
      default=True,
      metadata={"help": "Whether to apply causal attention masking."},
  )
  batch: int = dataclasses.field(
      default=1,
      metadata={"min": 1, "help": "Batch size dimension."},
  )
  seq_len: int = dataclasses.field(
      default=8192,
      metadata={"min": 1, "help": "Sequence length dimension."},
  )
  num_q_heads: int = dataclasses.field(
      default=56,
      metadata={"min": 1, "help": "Number of query attention heads."},
  )
  num_kv_heads: int = dataclasses.field(
      default=56,
      metadata={
          "min": 1,
          "help": "Number of key/value attention heads (GQA/MHA).",
      },
  )
  head_dim: int = dataclasses.field(
      default=128,
      metadata={"min": 1, "help": "Dimension size per attention head."},
  )


@registry.benchmark_registry.register(
    "attention_flashed", is_experimental=True
)
class AttentionBenchmark(base.BaseBenchmark[AttentionParams]):
  """Attention benchmark simulating FlashAttention behavior.

  Supports:
  - MHA/GQA
  - BF16 compute
  - Causal masking
  """

  Config = AttentionParams
  roofline_mode: constants.RooflineMode = constants.RooflineMode.COMPUTE

  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("mode", report.format_str),
      ("causal", report.format_str),
      ("batch", report.format_str),
      ("seq_len", report.format_str),
      ("num_q_heads", report.format_str),
      ("num_kv_heads", report.format_str),
      ("head_dim", report.format_str),
      ("total_flops", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_tflops_per_chip", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_tflops_per_chip", report.format_2f),
  )

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

    if mode == AttentionMode.FWD:
      self._jit_fn = attention_fwd
    elif mode == AttentionMode.BWD:

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

  def _get_num_sharded_devices(self) -> int:
    """Returns number of devices across which query/output heads are partitioned."""
    if self.mesh is None:
      return 1
    mesh_axis = self.mesh.axis_names[0]
    num_devices = int(self.mesh.shape[mesh_axis])
    if self.config.num_q_heads % num_devices == 0:
      return num_devices
    return 1

  def _get_num_kv_sharded_devices(self) -> int:
    """Returns number of devices across which K/V heads are partitioned."""
    if self.mesh is None:
      return 1
    mesh_axis = self.mesh.axis_names[0]
    num_devices = int(self.mesh.shape[mesh_axis])
    if self.config.num_kv_heads % num_devices == 0:
      return num_devices
    return 1

  def get_total_bytes(self) -> float:
    batch = self.config.batch
    q_len = self.config.seq_len
    kv_len = q_len
    heads_q = self.config.num_q_heads
    heads_kv = self.config.num_kv_heads
    head_dim = self.config.head_dim
    dtype = utils.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize
    mode = self.config.mode
    q_shards = self._get_num_sharded_devices()
    kv_shards = self._get_num_kv_sharded_devices()

    q_bytes_per_dev = (batch * heads_q * q_len * head_dim * itemsize) / q_shards
    kv_bytes_per_dev = (
        batch * heads_kv * kv_len * head_dim * itemsize
    ) / kv_shards

    if mode == AttentionMode.FWD:
      # Per-device Bytes = Load(Q_shard, K_shard, V_shard) + Store(Out_shard)
      return float(2 * q_bytes_per_dev + 2 * kv_bytes_per_dev)
    elif mode == AttentionMode.BWD:
      # Per-device Bytes = Load(Q, K, V, Out, dOut) + Store(dQ, dK, dV)
      return float(4 * q_bytes_per_dev + 4 * kv_bytes_per_dev)
    else:
      raise ValueError(f"Unknown mode: {mode}")

  def get_total_flops(self) -> float:
    batch = self.config.batch
    q_len = self.config.seq_len
    kv_len = q_len
    heads = self.config.num_q_heads
    head_dim = self.config.head_dim
    causal = self.config.causal
    mode = self.config.mode

    if causal:
      global_flops = (
          batch * (4 * q_len * kv_len - 2 * q_len * q_len) * heads * head_dim
      )
    else:
      global_flops = batch * 4 * q_len * kv_len * heads * head_dim

    if mode == AttentionMode.BWD:
      # FlashAttention forward (1.0x) executes 2 matmuls (S = Q @ K^T,
      # O = P @ V). Because P is not materialized to HBM, backward (vjp_fn)
      # executes 5 matmuls: 1 rematerialization matmul (S = Q @ K^T) +
      # 4 gradient matmuls (dP = dO @ V^T, dV = P^T @ dO, dQ = dS @ K,
      # dK = dS^T @ Q), yielding 5 / 2 = 2.5x forward FLOPs.
      # Since the timed MARKER scope in setup() wraps both jax.vjp (1.0x fwd)
      # and vjp_fn (2.5x bwd), total timed FLOPs = (1.0 + 2.5) = 3.5x fwd.
      fwd_flops_mult = 1.0
      bwd_recompute_and_grad_flops_mult = 2.5
      global_flops *= fwd_flops_mult + bwd_recompute_and_grad_flops_mult

    return float(global_flops) / self._get_num_sharded_devices()

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
    total_flops = self.get_total_flops()
    latency_s = latency_ms / 1000.0
    if latency_s == 0:
      tflops_per_sec = float("inf")
    else:
      tflops_per_sec = (total_flops / latency_s) / 1e12

    return {
        f"{prefix}_tflops_per_device": tflops_per_sec,
    }
