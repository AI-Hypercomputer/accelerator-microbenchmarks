"""Specialized compute benchmarks for LLM components."""

from typing import Any, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from ..core.base import BaseBenchmark
from ..core.registry import registry


@registry.register("swiglu")
class SwiGLUBenchmark(BaseBenchmark):
  """SwiGLU activation benchmark.

  Y = Swish(A) * B, where [A, B] = Split(X, 2)
  """

  def setup(self, **params):
    @jax.jit
    def swiglu_fn(x):
      a, b = jnp.split(x, 2, axis=-1)
      return (a * jax.nn.sigmoid(a)) * b

    self._jit_fn = swiglu_fn

  def generate_inputs(self, **params) -> Tuple[jnp.ndarray]:
    dim = params.get("dim", 4096)
    batch = params.get("batch", 1024)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, dim * 2), dtype=jnp.bfloat16)
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    x = jax.device_put(
        x, NamedSharding(self.mesh, P(self.mesh.axis_names[0], None))
    )
    return (x,)

  def run_op(self, x) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x)

  def get_total_bytes(self, **params) -> float:
    dim = params.get("dim", 4096)
    batch = params.get("batch", 1024)
    itemsize = jnp.dtype(jnp.bfloat16).itemsize
    # Read X (batch * dim * 2), Write Out (batch * dim)
    return (batch * dim * 2 * itemsize) + (batch * dim * itemsize)

  def get_arithmetic_intensity(self, **params) -> float:
    dim = params.get("dim", 4096)
    batch = params.get("batch", 1024)
    # 1 sigmoid (~4-10 flops) + 2 multiplies per element (dim)
    # Approximation: 10 flops per 'dim' element
    total_flops = batch * dim * 10
    return total_flops / self.get_total_bytes(**params)

  def calculate_metrics(
      self, times_ms: List[float], **params
  ) -> Dict[str, Any]:
    metrics = super().calculate_metrics(times_ms, **params)
    metrics["intensity"] = self.get_arithmetic_intensity(**params)
    return metrics


@registry.register("rmsnorm")
class RMSNormBenchmark(BaseBenchmark):
  """RMSNorm benchmark: Y = X / rms(X) * weight."""

  def setup(self, **params):
    @jax.jit
    def rmsnorm_fn(x, w):
      rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-6)
      return (x / rms) * w

    self._jit_fn = rmsnorm_fn

  def generate_inputs(self, **params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dim = params.get("dim", 4096)
    batch = params.get("batch", 1024)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, dim), dtype=jnp.bfloat16)
    w = jnp.ones((dim,), dtype=jnp.bfloat16)

    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    sharding = NamedSharding(self.mesh, P(self.mesh.axis_names[0], None))
    x = jax.device_put(x, sharding)
    w = jax.device_put(
        w, NamedSharding(self.mesh, P(None))
    )  # Replicated weight
    return x, w

  def run_op(self, x, w) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x, w)

  def get_total_bytes(self, **params) -> float:
    dim = params.get("dim", 4096)
    batch = params.get("batch", 1024)
    itemsize = jnp.dtype(jnp.bfloat16).itemsize
    # Read X, Read W (small), Write Out
    return (batch * dim * itemsize * 2) + (dim * itemsize)

  def get_arithmetic_intensity(self, **params) -> float:
    dim = params.get("dim", 4096)
    batch = params.get("batch", 1024)
    # square, sum, sqrt, div, mul = ~5 flops per element
    total_flops = batch * dim * 5
    return total_flops / self.get_total_bytes(**params)

  def calculate_metrics(
      self, times_ms: List[float], **params
  ) -> Dict[str, Any]:
    metrics = super().calculate_metrics(times_ms, **params)
    metrics["intensity"] = self.get_arithmetic_intensity(**params)
    return metrics


@registry.register("rope")
class RoPEBenchmark(BaseBenchmark):
  """Rotary Positional Embedding (RoPE) benchmark."""

  def setup(self, **params):
    # Simplified RoPE logic for benchmarking
    @jax.jit
    def rope_fn(x, freq_cis):
      # Complex element-wise multiplication as per Meta doc
      # x is treated as complex64 internally
      return x * freq_cis

    self._jit_fn = rope_fn

  def generate_inputs(self, **params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    m = params.get("seq_len", 1024)
    n = params.get("head_dim", 128)
    batch = params.get("batch", 32)
    heads = params.get("heads", 32)

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
    sharding = NamedSharding(
        self.mesh, P(self.mesh.axis_names[0], None, None, None)
    )
    x = jax.device_put(x, sharding)
    freq_cis = jax.device_put(
        freq_cis, NamedSharding(self.mesh, P(None, None, None, None))
    )
    return x, freq_cis

  def run_op(self, x, freq_cis) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x, freq_cis)

  def get_total_bytes(self, **params) -> float:
    m = params.get("seq_len", 1024)
    n = params.get("head_dim", 128)
    batch = params.get("batch", 32)
    heads = params.get("heads", 32)
    # complex64 = 8 bytes per element
    itemsize = 8
    return batch * heads * m * (n // 2) * itemsize * 2

  def get_arithmetic_intensity(self, **params) -> float:
    m = params.get("seq_len", 1024)
    n = params.get("head_dim", 128)
    batch = params.get("batch", 32)
    heads = params.get("heads", 32)
    # 1 complex mul = 6 flops (4 mul, 2 add)
    total_flops = batch * heads * m * (n // 2) * 6
    return total_flops / self.get_total_bytes(**params)

  def calculate_metrics(
      self, times_ms: List[float], **params
  ) -> Dict[str, Any]:
    metrics = super().calculate_metrics(times_ms, **params)
    metrics["intensity"] = self.get_arithmetic_intensity(**params)
    return metrics


@registry.register("quantization")
class QuantizationBenchmark(BaseBenchmark):
  """Rowwise quantization to FP8: OUT = cast_fp8(X / SF)."""

  def setup(self, **params):
    @jax.jit
    def quant_fn(x):
      # Rowwise scaling factor: FP8_MAX / amax(row)
      sf = 448.0 / jnp.max(jnp.abs(x), axis=-1, keepdims=True)
      out = (x * sf).astype(jnp.float8_e4m3fn)
      return out, sf

    self._jit_fn = quant_fn

  def generate_inputs(self, **params) -> Tuple[jnp.ndarray]:
    m, n = params.get("m", 4096), params.get("n", 4096)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (m, n), dtype=jnp.bfloat16)
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    x = jax.device_put(
        x, NamedSharding(self.mesh, P(self.mesh.axis_names[0], None))
    )
    return (x,)

  def run_op(self, x) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x)

  def get_total_bytes(self, **params) -> float:
    m, n = params.get("m", 4096), params.get("n", 4096)
    in_itemsize = jnp.dtype(jnp.bfloat16).itemsize
    out_itemsize = jnp.dtype(jnp.float8_e4m3fn).itemsize
    # Read X, Write Out, Write SF (rowwise)
    return (m * n * in_itemsize) + (m * n * out_itemsize) + (m * 4)

  def get_arithmetic_intensity(self, **params) -> float:
    m, n = params.get("m", 4096), params.get("n", 4096)
    # 1 abs, 1 max (per row), 1 div, 1 mul
    # Approximation: 4 flops per element
    return (m * n * 4) / self.get_total_bytes(**params)

  def calculate_metrics(
      self, times_ms: List[float], **params
  ) -> Dict[str, Any]:
    metrics = super().calculate_metrics(times_ms, **params)
    metrics["intensity"] = self.get_arithmetic_intensity(**params)
    return metrics


@registry.register("simple_add")
class AddBenchmark(BaseBenchmark):
  """Simple Z = X + Y benchmark."""

  def setup(self, **params):
    @jax.jit
    def add_fn(x, y):
      return x + y

    self._jit_fn = add_fn

  def generate_inputs(self, **params) -> Tuple[jnp.ndarray, jnp.ndarray]:
    size = params.get("size", 1024 * 1024)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (size,), dtype=jnp.bfloat16)
    y = jax.random.normal(key, (size,), dtype=jnp.bfloat16)
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    sharding = NamedSharding(self.mesh, P(self.mesh.axis_names[0]))
    return jax.device_put(x, sharding), jax.device_put(y, sharding)

  def run_op(self, x, y) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(x, y)

  def get_total_bytes(self, **params) -> float:
    size = params.get("size", 1024 * 1024)
    itemsize = jnp.dtype(jnp.bfloat16).itemsize
    # Read X, Read Y, Write Z
    return size * itemsize * 3

  def get_arithmetic_intensity(self, **params) -> float:
    size = params.get("size", 1024 * 1024)
    # 1 add per element
    return size / self.get_total_bytes(**params)

  def calculate_metrics(
      self, times_ms: List[float], **params
  ) -> Dict[str, Any]:
    metrics = super().calculate_metrics(times_ms, **params)
    metrics["intensity"] = self.get_arithmetic_intensity(**params)
    return metrics
