"""Specialized compute benchmarks for LLM components."""

import dataclasses
import logging
from typing import Any

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class ComputeParams(base.SingleDtypeBenchmarkParams):
  dim: int = dataclasses.field(
      default=4096,
      metadata={"min": 1, "help": "Hidden dimension size."},
  )
  batch: int = dataclasses.field(
      default=1024,
      metadata={"min": 1, "help": "Batch size dimension."},
  )


@dataclasses.dataclass
class RoPEParams(base.SingleDtypeBenchmarkParams):
  seq_len: int = dataclasses.field(
      default=1024,
      metadata={"min": 1, "help": "Sequence length dimension."},
  )
  head_dim: int = dataclasses.field(
      default=128,
      metadata={"min": 1, "help": "Dimension size per attention head."},
  )
  batch: int = dataclasses.field(
      default=32,
      metadata={"min": 1, "help": "Batch size dimension."},
  )
  heads: int = dataclasses.field(
      default=32,
      metadata={"min": 1, "help": "Number of attention heads."},
  )


@dataclasses.dataclass
class QuantParams(base.SingleDtypeBenchmarkParams):
  """Parameters for the rowwise quantization microbenchmark."""

  m: int = dataclasses.field(
      default=4096,
      metadata={"min": 1, "help": "Matrix dimension M (rows)."},
  )
  n: int = dataclasses.field(
      default=4096,
      metadata={"min": 1, "help": "Matrix dimension N (columns)."},
  )
  quant_dtype: str = dataclasses.field(
      default="float8_e4m3fn",
      metadata={
          "help": (
              "Target quantized data type (e.g., 'float8_e4m3fn',"
              " 'float4_e2m1fn', 'int4')."
          )
      },
  )
  scale_dtype: str = dataclasses.field(
      default="bfloat16",
      metadata={
          "help": (
              "Data type for rowwise scaling factors (e.g., 'bfloat16',"
              " 'float32')."
          )
      },
  )


@dataclasses.dataclass
class AddParams(base.SingleDtypeBenchmarkParams):
  size: int = dataclasses.field(
      default=1024 * 1024,
      metadata={"min": 1, "help": "Number of elements in the array."},
  )


@registry.benchmark_registry.register("swiglu", is_experimental=True)
class SwiGLUBenchmark(base.BaseBenchmark[ComputeParams]):
  """SwiGLU activation benchmark.

  Y = Swish(A) * B, where [A, B] = Split(X, 2)
  """
  Config = ComputeParams

  def setup(self):
    @jax.jit
    def swiglu_fn(x):
      with jax.named_scope(constants.MARKER):
        a, b = jnp.split(x, 2, axis=-1)
        return (a * jax.nn.sigmoid(a)) * b

    self._jit_fn = swiglu_fn

  def get_run_identifier(self) -> str:
    dim = self.config.dim
    batch = self.config.batch
    if dim is not None or batch is not None:
      return f"batch_{batch or 1024}_dim_{dim or 4096}"
    return ""

  def generate_inputs(self) -> tuple[Any, ...]:
    dim = self.config.dim
    batch = self.config.batch
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, dim * 2), dtype=jnp.bfloat16)
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    x = jax.device_put(
        x,
        jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(None, None)
        ),
    )
    return (x,)

  def run_op(self, x) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x)

  def get_total_bytes(self) -> float:
    dim = self.config.dim
    batch = self.config.batch
    itemsize = jnp.dtype(jnp.bfloat16).itemsize
    # Read X (batch * dim * 2), Write Out (batch * dim)
    return (batch * dim * 2 * itemsize) + (batch * dim * itemsize)

  def get_arithmetic_intensity(self) -> float:
    dim = self.config.dim
    batch = self.config.batch
    # 1 sigmoid (~4-10 flops) + 2 multiplies per element (dim)
    # Approximation: 10 flops per 'dim' element
    total_flops = batch * dim * 10
    return total_flops / self.get_total_bytes()

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    return {}


@registry.benchmark_registry.register("rmsnorm", is_experimental=True)
class RMSNormBenchmark(base.BaseBenchmark[ComputeParams]):
  """RMSNorm benchmark: Y = X / rms(X) * weight."""
  Config = ComputeParams

  def setup(self):
    @jax.jit
    def rmsnorm_fn(x, w):
      with jax.named_scope(constants.MARKER):
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)
        return (x / rms) * w

    self._jit_fn = rmsnorm_fn

  def get_run_identifier(self) -> str:
    dim = self.config.dim
    batch = self.config.batch
    if dim is not None or batch is not None:
      return f"batch_{batch or 1024}_dim_{dim or 4096}"
    return ""

  def generate_inputs(self) -> tuple[jnp.ndarray, jnp.ndarray]:
    dim = self.config.dim
    batch = self.config.batch
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, dim), dtype=jnp.bfloat16)
    w = jnp.ones((dim,), dtype=jnp.bfloat16)

    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None)
    )
    x = jax.device_put(x, sharding)
    w = jax.device_put(
        w,
        jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None)),
    )  # Replicated weight
    return x, w

  def run_op(self, x, w) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x, w)

  def get_total_bytes(self) -> float:
    dim = self.config.dim
    batch = self.config.batch
    itemsize = jnp.dtype(jnp.bfloat16).itemsize
    # Read X, Read W (small), Write Out
    return (batch * dim * itemsize * 2) + (dim * itemsize)

  def get_arithmetic_intensity(self) -> float:
    dim = self.config.dim
    batch = self.config.batch
    # square, sum, sqrt, div, mul = ~5 flops per element
    total_flops = batch * dim * 5
    return total_flops / self.get_total_bytes()

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    return {}


@registry.benchmark_registry.register("rope", is_experimental=True)
class RoPEBenchmark(base.BaseBenchmark[RoPEParams]):
  """Rotary Positional Embedding (RoPE) benchmark."""
  Config = RoPEParams

  def setup(self):
    # Simplified RoPE logic for benchmarking
    @jax.jit
    def rope_fn(x, freq_cis):
      # Complex element-wise multiplication as per Meta doc
      # x is treated as complex64 internally
      with jax.named_scope(constants.MARKER):
        return x * freq_cis

    self._jit_fn = rope_fn

  def get_run_identifier(self) -> str:
    seq_len = self.config.seq_len
    head_dim = self.config.head_dim
    batch = self.config.batch
    heads = self.config.heads
    if any(v is not None for v in (seq_len, head_dim, batch, heads)):
      return f"b_{batch or 32}_h_{heads or 32}_s_{seq_len or 1024}_d_{head_dim or 128}"
    return ""

  def generate_inputs(self) -> tuple[jnp.ndarray, jnp.ndarray]:
    m = self.config.seq_len
    n = self.config.head_dim
    batch = self.config.batch
    heads = self.config.heads

    key = jax.random.PRNGKey(0)
    # Convert to complex64 for internal compute as requested
    x = jax.random.normal(
        key, (batch, heads, m, n // 2), dtype=jnp.float32
    ) + 1j * jax.random.normal(
        key, (batch, heads, m, n // 2), dtype=jnp.float32
    )
    freq_cis = jax.random.normal(
        key, (1, 1, m, n // 2), dtype=jnp.float32
    ) + 1j * jax.random.normal(key, (1, 1, m, n // 2), dtype=jnp.float32)

    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    sharding = jax.sharding.NamedSharding(
        self.mesh,
        jax.sharding.PartitionSpec(None, None, None, None),
    )
    x = jax.device_put(x, sharding)
    freq_cis = jax.device_put(
        freq_cis,
        jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(None, None, None, None)
        ),
    )
    return x, freq_cis

  def run_op(self, x, freq_cis) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x, freq_cis)

  def get_total_bytes(self) -> float:
    m = self.config.seq_len
    n = self.config.head_dim
    batch = self.config.batch
    heads = self.config.heads
    # complex64 = 8 bytes per element
    itemsize = 8
    return batch * heads * m * (n // 2) * itemsize * 2

  def get_arithmetic_intensity(self) -> float:
    m = self.config.seq_len
    n = self.config.head_dim
    batch = self.config.batch
    heads = self.config.heads
    # 1 complex mul = 6 flops (4 mul, 2 add)
    total_flops = batch * heads * m * (n // 2) * 6
    return total_flops / self.get_total_bytes()

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    return {}


@registry.benchmark_registry.register("quantization", is_experimental=True)
class QuantizationBenchmark(base.BaseBenchmark[QuantParams]):
  """Rowwise quantization: OUT = cast_quant(X * SF)."""

  Config = QuantParams
  derive_chip_bandwidth: bool = True
  roofline_mode: constants.RooflineMode = constants.RooflineMode.MEMORY_HBM

  def setup(self):
    out_dtype = utils.parse_dtype(self.config.quant_dtype)
    sf_dtype = utils.parse_dtype(self.config.scale_dtype)
    q_max = utils.get_dtype_max(out_dtype)

    @jax.jit
    def quant_fn(x):
      with jax.named_scope(constants.MARKER):
        # Rowwise scaling factor: Q_MAX / amax(row)
        sf = (q_max / jnp.max(jnp.abs(x), axis=-1, keepdims=True)).astype(
            sf_dtype
        )
        out = (x * sf).astype(out_dtype)
        return out, sf

    self._jit_fn = quant_fn

  def get_run_identifier(self) -> str:
    return (
        f"m_{self.config.m}_n_{self.config.n}_"
        f"{self.config.quant_dtype}_sf_{self.config.scale_dtype}"
    )

  def generate_inputs(self) -> tuple[jnp.ndarray, ...]:
    m, n = self.config.m, self.config.n
    in_dtype = utils.parse_dtype(self.config.dtype)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (m, n), dtype=in_dtype)
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    x = jax.device_put(
        x,
        jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(None, None)
        ),
    )
    return (x,)

  def run_op(self, x) -> tuple[jnp.ndarray, jnp.ndarray]:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x)

  def get_total_bytes(self) -> float:
    m, n = self.config.m, self.config.n
    in_itemsize = utils.get_dtype_bytes(self.config.dtype)
    out_itemsize = utils.get_dtype_bytes(self.config.quant_dtype)
    sf_itemsize = utils.get_dtype_bytes(self.config.scale_dtype)
    # Read X, Write Out, Write SF (rowwise)
    return float(
        (m * n * in_itemsize) + (m * n * out_itemsize) + (m * sf_itemsize)
    )

  def get_arithmetic_intensity(self) -> float:
    m, n = self.config.m, self.config.n
    # 1 abs, 1 max (per row), 1 div, 1 mul
    # Approximation: 4 flops per element
    return (m * n * 4) / self.get_total_bytes()

  def get_workload_metadata(self) -> dict[str, Any]:
    total_bytes = self.get_total_bytes()
    return {
        "total_bytes_mib": total_bytes / (1024 * 1024),
        "quant_dtype": self.config.quant_dtype,
        "scale_dtype": self.config.scale_dtype,
    }

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    total_bytes = self.get_total_bytes()
    latency_s = latency_ms / 1000.0
    if latency_s <= 0:
      logging.warning(
          "Non-positive latency_s (%s) encountered for %s in"
          " QuantizationBenchmark; returning inf bandwidth.",
          latency_s,
          prefix,
      )
      bandwidth_gb_s = float("inf")
    else:
      # X is replicated (PartitionSpec(None, None)), so every device runs the
      # full [M, N] op and moves total_bytes; do not divide by mesh size.
      bandwidth_gb_s = (total_bytes / latency_s) / 1e9
    return {
        f"{prefix}_bandwidth_per_device_gb_s": bandwidth_gb_s,
    }


@registry.benchmark_registry.register("simple_add", is_experimental=True)
class AddBenchmark(base.BaseBenchmark[AddParams]):
  """Simple Z = X + Y benchmark."""
  Config = AddParams

  def setup(self):
    @jax.jit
    def add_fn(x, y):
      with jax.named_scope(constants.MARKER):
        return x + y

    self._jit_fn = add_fn

  def get_run_identifier(self) -> str:
    size = self.config.size
    if size is not None:
      return f"size_{size}"
    return ""

  def generate_inputs(self) -> tuple[jnp.ndarray, jnp.ndarray]:
    size = self.config.size
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (size,), dtype=jnp.bfloat16)
    y = jax.random.normal(key, (size,), dtype=jnp.bfloat16)
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None)
    )
    return jax.device_put(x, sharding), jax.device_put(y, sharding)

  def run_op(self, x, y) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x, y)

  def get_total_bytes(self) -> float:
    size = self.config.size
    itemsize = jnp.dtype(jnp.bfloat16).itemsize
    # Read X, Read Y, Write Z
    return size * itemsize * 3

  def get_arithmetic_intensity(self) -> float:
    size = self.config.size
    # 1 add per element
    return size / self.get_total_bytes()

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    return {}
