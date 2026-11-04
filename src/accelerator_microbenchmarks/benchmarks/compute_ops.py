"""Specialized compute benchmarks for LLM components."""

import abc
from collections.abc import Callable
import dataclasses
import logging
from typing import Any

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import dtypes
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.kernels import quantize_transpose
import jax
import jax.numpy as jnp


class PassMode(constants.ParamEnum):
  """Which pass of an op to benchmark."""

  FWD = "fwd"  # Forward pass.
  BWD = "bwd"  # Backward pass (VJP w.r.t. the activations).


def _pass_mode_field() -> PassMode:
  """Returns a dataclass field selecting the forward or backward pass."""
  return dataclasses.field(
      default=PassMode.FWD,
      metadata={
          "help": (
              "Pass to benchmark: 'fwd' (forward) or 'bwd' (backward, VJP"
              " w.r.t. the activations)."
          )
      },
  )


@dataclasses.dataclass
class ComputeParams(base.SingleDtypeBenchmarkParams):
  """Parameters for the 2D activation and normalization microbenchmarks.

  `dtype` is the activation (input and output) dtype. See each benchmark for
  how `batch` and `dim` map to its tensor shapes.
  """

  # Activations are floats; get_total_bytes() counts itemsize bytes.
  DTYPE_CONSTRAINTS = {"dtype": (jnp.floating,)}
  ALLOW_SUB_BYTE_DTYPES = False

  dim: int = dataclasses.field(
      default=4096,
      metadata={"min": 1, "help": "Hidden dimension size."},
  )
  batch: int = dataclasses.field(
      default=1024,
      metadata={"min": 1, "help": "Batch size dimension."},
  )
  mode: PassMode = _pass_mode_field()


@dataclasses.dataclass
class RoPEParams(base.SingleDtypeBenchmarkParams):
  """Parameters for the RoPE microbenchmark.

  `dtype` is the activation (input and output) dtype; the cos/sin table and the
  rotation math are always float32. The activations are laid out as a 2D
  [batch * seq_len, heads * head_dim] matrix, so an (M, N) shape maps to
  `batch=1, seq_len=M, heads=N // head_dim`.
  """

  # Activations are floats; get_total_bytes() counts itemsize bytes.
  DTYPE_CONSTRAINTS = {"dtype": (jnp.floating,)}
  ALLOW_SUB_BYTE_DTYPES = False

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
  mode: PassMode = _pass_mode_field()

  def __post_init__(self):
    super().__post_init__()
    if self.head_dim % 2:
      raise ValueError(f"head_dim must be even, got {self.head_dim}")


# Input (`dtype`) and rowwise scale (`scale_dtype`) dtypes of the quantization
# benchmark.
_INPUT_AND_SCALE_DTYPES = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)


@dataclasses.dataclass
class QuantParams(base.SingleDtypeBenchmarkParams):
  """Parameters for the rowwise quantization microbenchmark."""

  # The input is sampled with jax.random.normal, which does not support
  # sub-byte floats, and the rowwise scale is a fractional factor (an integer
  # scale truncates). JAX has no implicit promotion between 4/6/8-bit floats
  # and other floats, so `x * sf` raises unless both are >= 16-bit floats.
  # The quantized output may be a float or an integer.
  DTYPE_CONSTRAINTS = {
      "dtype": _INPUT_AND_SCALE_DTYPES,
      "quant_dtype": (jnp.floating, jnp.integer),
      "scale_dtype": _INPUT_AND_SCALE_DTYPES,
  }

  m: int = dataclasses.field(
      default=4096,
      metadata={"min": 1, "help": "Matrix dimension M (rows)."},
  )
  n: int = dataclasses.field(
      default=4096,
      metadata={"min": 1, "help": "Matrix dimension N (columns)."},
  )
  quant_dtype: dtypes.DType = base.dtype_field(
      default=dtypes.DType.FLOAT8_E4M3FN,
      metadata={
          "help": (
              "Target quantized data type (e.g., 'float8_e4m3fn',"
              " 'float4_e2m1fn', 'int4')."
          )
      },
  )
  scale_dtype: dtypes.DType = base.dtype_field(
      default=dtypes.DType.BFLOAT16,
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


def _jit_pass(
    fwd_fn: Callable[..., jax.Array],
    mode: PassMode,
    num_grad_args: int = 1,
) -> Callable[..., Any]:
  """Returns a jitted function that runs the requested pass of `fwd_fn`.

  Args:
    fwd_fn: The forward op. Its first `num_grad_args` positional arguments are
      differentiated in the backward pass; the rest are treated as constants.
    mode: FWD returns `jit(fwd_fn)`. BWD returns a jitted function that takes
      `fwd_fn`'s arguments followed by the output cotangent dY, and returns the
      VJP w.r.t. the first `num_grad_args` arguments (a single array when
      `num_grad_args` is 1, else a tuple).
    num_grad_args: Number of leading arguments to differentiate in BWD mode.

  Returns:
    The jitted function, with the op wrapped in the benchmark marker scope.
  """
  if mode == PassMode.FWD:

    @jax.jit
    def fwd_pass(*args):
      with jax.named_scope(constants.MARKER):
        return fwd_fn(*args)

    return fwd_pass

  @jax.jit
  def bwd_pass(*args):
    *inputs, dy = args
    grad_args = inputs[:num_grad_args]
    const_args = inputs[num_grad_args:]
    with jax.named_scope(constants.MARKER):
      _, vjp_fn = jax.vjp(lambda *g: fwd_fn(*g, *const_args), *grad_args)
      grads = vjp_fn(dy)
      return grads[0] if num_grad_args == 1 else grads

  return bwd_pass


def _hbm_bandwidth_metrics(
    total_bytes: float,
    latency_ms: float,
    prefix: constants.TimingDomain,
    benchmark_name: str,
) -> dict[str, float]:
  """Returns `<prefix>_bandwidth_per_device_gb_s` for a replicated op.

  Inputs are replicated, so every device runs the full op and moves
  `total_bytes`; the bandwidth is not divided by the mesh size.

  Args:
    total_bytes: HBM bytes moved by one run of the op on one device.
    latency_ms: Latency of one run in milliseconds.
    prefix: Timing domain of `latency_ms`.
    benchmark_name: Benchmark class name, used in the warning logged when
      `latency_ms` is not positive (the bandwidth is then `inf`).
  """
  latency_s = latency_ms / 1000.0
  if latency_s <= 0:
    logging.warning(
        "Non-positive latency_s (%s) encountered for %s in %s; returning inf"
        " bandwidth.",
        latency_s,
        prefix,
        benchmark_name,
    )
    bandwidth_gb_s = float("inf")
  else:
    bandwidth_gb_s = (total_bytes / latency_s) / 1e9
  return {f"{prefix}_bandwidth_per_device_gb_s": bandwidth_gb_s}


class _ActivationBenchmark(base.BaseBenchmark[ComputeParams]):
  """Shared plumbing for the memory-bound 2D activation/normalization ops.

  Subclasses implement the op and its HBM byte and FLOP counts; this class
  provides fwd/bwd inputs, the run identifier, and HBM bandwidth metrics.
  Inputs are replicated, so every device runs the full op.
  """

  Config = ComputeParams
  roofline_mode: constants.RooflineMode = constants.RooflineMode.MEMORY_HBM

  def _input_shape(self) -> tuple[int, int]:
    """Returns the shape of the activation input X."""
    return (self.config.batch, self.config.dim)

  def _output_shape(self) -> tuple[int, int]:
    """Returns the shape of the activation output Y (and of dY)."""
    return (self.config.batch, self.config.dim)

  def get_run_identifier(self) -> str:
    return f"batch_{self.config.batch}_dim_{self.config.dim}_{self.config.mode}"

  def _generate_activations(self) -> tuple[jax.Array, jax.Array | None]:
    """Returns X and, in BWD mode, the output cotangent dY (else None)."""
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    dtype = dtypes.parse_dtype(self.config.dtype)
    x_key, dy_key = jax.random.split(jax.random.PRNGKey(0))
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None)
    )
    x = jax.device_put(
        jax.random.normal(x_key, self._input_shape(), dtype), sharding
    )
    if self.config.mode != PassMode.BWD:
      return x, None
    dy = jax.device_put(
        jax.random.normal(dy_key, self._output_shape(), dtype), sharding
    )
    return x, dy

  def generate_inputs(self) -> tuple[jax.Array, ...]:
    x, dy = self._generate_activations()
    return (x,) if dy is None else (x, dy)

  def run_op(self, *args) -> Any:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(*args)

  def _activation_itemsize(self) -> float:
    return dtypes.get_dtype_bytes(self.config.dtype)

  @abc.abstractmethod
  def _forward_flops(self) -> float:
    """Returns the approximate FLOPs of the forward pass."""

  def get_arithmetic_intensity(self) -> float:
    # Approximation: the backward pass costs about twice the forward FLOPs.
    flops = self._forward_flops()
    if self.config.mode == PassMode.BWD:
      flops *= 2
    return flops / self.get_total_bytes()

  def get_workload_metadata(self) -> dict[str, Any]:
    return {"total_bytes_mib": self.get_total_bytes() / (1024 * 1024)}

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    return _hbm_bandwidth_metrics(
        self.get_total_bytes(), latency_ms, prefix, type(self).__name__
    )


@registry.benchmark_registry.register("swiglu", is_experimental=True)
class SwiGLUBenchmark(_ActivationBenchmark):
  """SwiGLU activation benchmark.

  Y = Swish(A) * B, where [A, B] = Split(X, 2), X is [batch, 2 * dim], and Y is
  [batch, dim]. A v7x SwiGLU shape (M, N), where N is the width of X, maps to
  `batch=M, dim=N // 2`. `mode=bwd` benchmarks the VJP w.r.t. X, which reads X
  and dY [batch, dim] and writes dX [batch, 2 * dim].
  """

  def setup(self):
    def swiglu_fwd(x):
      a, b = jnp.split(x, 2, axis=-1)
      return (a * jax.nn.sigmoid(a)) * b

    self._jit_fn = _jit_pass(swiglu_fwd, self.config.mode)

  def _input_shape(self) -> tuple[int, int]:
    return (self.config.batch, 2 * self.config.dim)

  def get_total_bytes(self) -> float:
    out_elements = self.config.batch * self.config.dim
    # fwd: read X (2 * out), write Y (out).
    # bwd: read X (2 * out), read dY (out), write dX (2 * out).
    num_elements = 5 if self.config.mode == PassMode.BWD else 3
    return float(num_elements * out_elements * self._activation_itemsize())

  def _forward_flops(self) -> float:
    # 1 sigmoid (~4-10 flops) + 2 multiplies per output element.
    # Approximation: 10 flops per output element.
    return float(self.config.batch * self.config.dim * 10)


@registry.benchmark_registry.register("sigmoid", is_experimental=True)
class SigmoidBenchmark(_ActivationBenchmark):
  """Sigmoid activation benchmark: Y = sigmoid(X), with X and Y [batch, dim].

  `mode=bwd` benchmarks the VJP w.r.t. X, which reads X and dY and writes dX.
  """

  def setup(self):
    self._jit_fn = _jit_pass(jax.nn.sigmoid, self.config.mode)

  def get_total_bytes(self) -> float:
    # fwd: read X, write Y. bwd: read X, read dY, write dX.
    num_tensors = 3 if self.config.mode == PassMode.BWD else 2
    return float(
        num_tensors
        * self.config.batch
        * self.config.dim
        * self._activation_itemsize()
    )

  def _forward_flops(self) -> float:
    # negate, exp, add, reciprocal = ~4 flops per element.
    return float(self.config.batch * self.config.dim * 4)


@registry.benchmark_registry.register("rmsnorm", is_experimental=True)
class RMSNormBenchmark(_ActivationBenchmark):
  """RMSNorm benchmark: Y = X / rms(X) * weight.

  X and Y are [batch, dim] and the weight is [dim]. `mode=bwd` benchmarks the
  VJP w.r.t. both X and the weight, which reads X, the weight, and dY, and
  writes dX and dWeight.
  """

  def setup(self):
    def rmsnorm_fwd(x, w):
      rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)
      return (x / rms) * w

    self._jit_fn = _jit_pass(rmsnorm_fwd, self.config.mode, num_grad_args=2)

  def generate_inputs(self) -> tuple[jax.Array, ...]:
    x, dy = self._generate_activations()
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    w = jax.device_put(
        jnp.ones(
            (self.config.dim,), dtype=dtypes.parse_dtype(self.config.dtype)
        ),
        jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None)),
    )  # Replicated weight
    return (x, w) if dy is None else (x, w, dy)

  def get_total_bytes(self) -> float:
    activation_elements = self.config.batch * self.config.dim
    if self.config.mode == PassMode.BWD:
      # Read X, read dY, write dX; read W, write dW.
      num_elements = 3 * activation_elements + 2 * self.config.dim
    else:
      # Read X, write Y; read W.
      num_elements = 2 * activation_elements + self.config.dim
    return float(num_elements * self._activation_itemsize())

  def _forward_flops(self) -> float:
    # square, sum, sqrt, div, mul = ~5 flops per element.
    return float(self.config.batch * self.config.dim * 5)


@registry.benchmark_registry.register("rope", is_experimental=True)
class RoPEBenchmark(base.BaseBenchmark[RoPEParams]):
  """Rotary Positional Embedding (RoPE) benchmark.

  Matches the TPU v7x (Ironwood) RoPE microbenchmark: activations X are a
  [batch * seq_len, heads * head_dim] matrix in `dtype`, and the cos/sin table
  is a single float32 [seq_len, head_dim] array holding cos in the first half of
  the columns and sin in the second half. Each head is rotated with the
  half-split ("rotate_half") convention in float32:

    [x1, x2] -> [x1 * cos - x2 * sin, x2 * cos + x1 * sin]

  and the result is cast back to `dtype`. `mode=bwd` benchmarks the VJP w.r.t.
  X (the table is a constant), which reads dY instead of X and writes dX.
  """

  Config = RoPEParams
  roofline_mode: constants.RooflineMode = constants.RooflineMode.MEMORY_HBM

  def setup(self):
    batch = self.config.batch
    seq_len = self.config.seq_len
    heads = self.config.heads
    head_dim = self.config.head_dim

    def rope_fwd(x, table):
      xf = x.astype(jnp.float32).reshape(batch, seq_len, heads, head_dim)
      x1, x2 = jnp.split(xf, 2, axis=-1)
      cos, sin = jnp.split(table, 2, axis=-1)
      # Broadcast the per-position table over batch and heads.
      cos = cos[None, :, None, :]
      sin = sin[None, :, None, :]
      out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
      return out.reshape(x.shape).astype(x.dtype)

    self._jit_fn = _jit_pass(rope_fwd, self.config.mode)

  def get_run_identifier(self) -> str:
    return (
        f"b_{self.config.batch}_h_{self.config.heads}_"
        f"s_{self.config.seq_len}_d_{self.config.head_dim}_{self.config.mode}"
    )

  def _num_activation_elements(self) -> int:
    return (
        self.config.batch
        * self.config.seq_len
        * self.config.heads
        * self.config.head_dim
    )

  def generate_inputs(self) -> tuple[jnp.ndarray, ...]:
    batch = self.config.batch
    seq_len = self.config.seq_len
    heads = self.config.heads
    head_dim = self.config.head_dim
    dtype = dtypes.parse_dtype(self.config.dtype)

    x_key, dy_key = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(x_key, (batch * seq_len, heads * head_dim), dtype)
    # Standard RoPE angles: position * 10000^(-2i / head_dim).
    inv_freq = 1.0 / (
        10000.0 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    angles = jnp.arange(seq_len, dtype=jnp.float32)[:, None] * inv_freq
    table = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None)
    )
    x = jax.device_put(x, sharding)
    table = jax.device_put(table, sharding)
    if self.config.mode == PassMode.BWD:
      dy = jax.device_put(jax.random.normal(dy_key, x.shape, dtype), sharding)
      return x, table, dy
    return x, table

  def run_op(self, *args) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(*args)

  def get_total_bytes(self) -> float:
    itemsize = dtypes.get_dtype_bytes(self.config.dtype)
    table_itemsize = jnp.dtype(jnp.float32).itemsize
    # fwd: read X, read table, write Y. bwd: read dY, read table, write dX (the
    # VJP of a linear op does not read X).
    return float(
        2 * self._num_activation_elements() * itemsize
        + self.config.seq_len * self.config.head_dim * table_itemsize
    )

  def get_arithmetic_intensity(self) -> float:
    # Per rotated pair: 4 muls + 2 adds = 3 flops per activation element.
    total_flops = self._num_activation_elements() * 3
    return total_flops / self.get_total_bytes()

  def get_workload_metadata(self) -> dict[str, Any]:
    return {"total_bytes_mib": self.get_total_bytes() / (1024 * 1024)}

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    return _hbm_bandwidth_metrics(
        self.get_total_bytes(), latency_ms, prefix, type(self).__name__
    )


@registry.benchmark_registry.register("quantization", is_experimental=True)
class QuantizationBenchmark(base.BaseBenchmark[QuantParams]):
  """Rowwise quantization: OUT = cast_quant(X * SF).

  For fused transposed quantization, see `QuantizeTransposeBenchmark`
  (`quantize_transpose`).
  """

  Config = QuantParams
  derive_chip_bandwidth: bool = True
  roofline_mode: constants.RooflineMode = constants.RooflineMode.MEMORY_HBM

  def setup(self):
    out_dtype = dtypes.parse_dtype(self.config.quant_dtype)
    sf_dtype = dtypes.parse_dtype(self.config.scale_dtype)
    q_max = dtypes.get_dtype_max(out_dtype)

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
    in_dtype = dtypes.parse_dtype(self.config.dtype)
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
    """Returns 1-pass algorithmic HBM traffic (read X, write OUT, write SF).

    Like all other benchmarks in the suite (`RMSNormBenchmark`,
    `SwiGLUBenchmark`, `RoPEBenchmark`, `FlashAttentionBenchmark`,
    `MatmulBenchmark`, and rowwise `QuantizationBenchmark`), this reports the
    minimum theoretical 1-pass algorithmic I/O bytes rather than physical
    multi-pass or tile-rematerialization reads, even when multi-tile transposed
    quantization physically reads X twice (amax phase + quantize phase).
    """
    m, n = self.config.m, self.config.n
    in_itemsize = dtypes.get_dtype_bytes(self.config.dtype)
    out_itemsize = dtypes.get_dtype_bytes(self.config.quant_dtype)
    sf_itemsize = dtypes.get_dtype_bytes(self.config.scale_dtype)
    # Read X, Write Out, Write SF (one per output row)
    return float(
        (m * n * in_itemsize)
        + (m * n * out_itemsize)
        + (self._num_scales() * sf_itemsize)
    )

  def _num_scales(self) -> int:
    """Number of scaling factors: one per output row."""
    return self.config.m

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


@registry.benchmark_registry.register(
    "quantize_transpose", is_experimental=True
)
class QuantizeTransposeBenchmark(QuantizationBenchmark):
  """Transposed quantization: OUT<N, M> = cast_quant(X^T * SF).

  SF<N> holds one scaling factor per column of X. Runs as a fused Pallas kernel
  (`quantize_transpose.quantize_transpose`) that does not write X^T to HBM.
  """

  def setup(self):
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    out_dtype = dtypes.parse_dtype(self.config.quant_dtype)
    sf_dtype = dtypes.parse_dtype(self.config.scale_dtype)
    q_max = dtypes.get_dtype_max(out_dtype)
    bm, bn = self._get_transpose_blocks()

    def per_device_quant_fn(x):
      with jax.named_scope(constants.MARKER):
        return quantize_transpose.quantize_transpose(
            x,
            out_dtype=out_dtype,
            scale_dtype=sf_dtype,
            q_max=q_max,
            block_m=bm,
            block_n=bn,
        )

    # X is replicated, so every device runs the full [M, N] kernel, as in
    # QuantizationBenchmark.
    replicated = jax.sharding.PartitionSpec(None, None)
    self._jit_fn = jax.jit(
        jax.shard_map(
            per_device_quant_fn,
            mesh=self.mesh,
            in_specs=replicated,
            out_specs=(replicated, replicated),
            check_vma=False,
        )
    )

  def _num_scales(self) -> int:
    """One scaling factor per column of X (output row of X^T)."""
    return self.config.n

  def _get_transpose_blocks(self) -> tuple[int, int]:
    """Derives the `(block_m, block_n)` tile for the current config."""
    return quantize_transpose.default_transpose_blocks(
        self.config.m,
        self.config.n,
        in_dtype=dtypes.parse_dtype(self.config.dtype),
        out_dtype=dtypes.parse_dtype(self.config.quant_dtype),
        scale_dtype=dtypes.parse_dtype(self.config.scale_dtype),
    )

  def get_workload_metadata(self) -> dict[str, Any]:
    metadata = super().get_workload_metadata()
    bm, bn = self._get_transpose_blocks()
    metadata["block_m"] = bm
    metadata["block_n"] = bn
    return metadata


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
    return f"size_{self.config.size}"

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
