"""Matrix multiplication and GEMM benchmarks including FP8 support."""

from typing import Any
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
import jax
import jax.numpy as jnp


@registry.benchmark_registry.register("gemm_generalized")
class GeneralizedGemmBenchmark(base.BaseBenchmark):
  """Generalized GEMM benchmark supporting FP8 and throughput projection.

  Pseudo-code: OUT = matmul(IN0, IN1) * rescaling_factor
  """

  def setup(self, **params):
    out_dtype_str = params.get("out_dtype", "bfloat16")
    out_dtype = (
        getattr(jnp, out_dtype_str)
        if hasattr(jnp, out_dtype_str)
        else jnp.bfloat16
    )

    @jax.jit
    def gemm_fn(a, b, sf0=None, sf1=None):
      with jax.named_scope(constants.MARKER):
        # Standard matmul
        out = jnp.matmul(a, b)
        # Optional rescaling (row-wise scaling factors as requested)
        if sf0 is not None and sf1 is not None:
          # Assuming rowwise scaling factor SF0<M, 1> and SF1<1, N>
          out = out * (sf0 @ sf1)
        return out.astype(out_dtype)

    self._jit_fn = gemm_fn

  def get_run_identifier(self, **params) -> str:
    m = params.get("m")
    k = params.get("k")
    n = params.get("n")
    in_dtype = params.get("in_dtype")
    out_dtype = params.get("out_dtype")
    if any(v is not None for v in (m, k, n, in_dtype, out_dtype)):
      return (
          f"m_{m or 1024}_k_{k or 1024}_n_{n or 1024}_{in_dtype or 'float8_e4m3fn'}_to_{out_dtype or 'bfloat16'}"
      )
    return ""

  def generate_inputs(self, **params) -> tuple[Any, ...]:
    m, k, n = (
        params.get("m", 1024),
        params.get("k", 1024),
        params.get("n", 1024),
    )
    in_dtype_str = params.get("in_dtype", "float8_e4m3fn")

    # Resolve dtypes
    in_dtype = (
        getattr(jnp, in_dtype_str)
        if hasattr(jnp, in_dtype_str)
        else jnp.bfloat16
    )

    key = jax.random.PRNGKey(params.get("seed", 0))
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Data generation in HBM
    # Note: JAX might require intermediate conversion for random.normal if
    # dtypes aren't supported
    a = jax.random.normal(k1, (m, k)).astype(in_dtype)
    b = jax.random.normal(k2, (k, n)).astype(in_dtype)

    # Optional scaling factors
    use_sf = params.get("use_scaling_factors", False)
    sf0 = jax.random.normal(k3, (m, 1)).astype(jnp.float32) if use_sf else None
    sf1 = jax.random.normal(k4, (1, n)).astype(jnp.float32) if use_sf else None

    # Replicated sharding for computation ops to avoid discrepancies
    assert self.mesh is not None, "Mesh not initialized."
    replicated_sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None)
    )

    a = jax.device_put(a, replicated_sharding)
    b = jax.device_put(b, replicated_sharding)

    if use_sf:
      sf0 = jax.device_put(sf0, replicated_sharding)
      sf1 = jax.device_put(sf1, replicated_sharding)
      return a, b, sf0, sf1

    return a, b

  def run_op(self, *args) -> jnp.ndarray:
    assert self._jit_fn is not None
    return self._jit_fn(*args)

  def get_total_bytes(self, **params) -> float:
    m, k, n = (
        params.get("m", 1024),
        params.get("k", 1024),
        params.get("n", 1024),
    )
    in_dtype_str = params.get("in_dtype", "float8_e4m3fn")
    out_dtype_str = params.get("out_dtype", "bfloat16")

    in_itemsize = jnp.dtype(
        getattr(jnp, in_dtype_str)
        if hasattr(jnp, in_dtype_str)
        else jnp.bfloat16
    ).itemsize
    out_itemsize = jnp.dtype(
        getattr(jnp, out_dtype_str)
        if hasattr(jnp, out_dtype_str)
        else jnp.bfloat16
    ).itemsize

    # Bytes = Load(A) + Load(B) + Store(Out)
    # Scaling factors are small (row-wise), ignoring for intensity but could be
    # added.
    return (
        (m * k * in_itemsize) + (k * n * in_itemsize) + (m * n * out_itemsize)
    )

  def get_arithmetic_intensity(self, **params) -> float:
    m, k, n = (
        params.get("m", 1024),
        params.get("k", 1024),
        params.get("n", 1024),
    )
    flops = 2 * m * n * k
    bytes_moved = self.get_total_bytes(**params)
    return flops / bytes_moved if bytes_moved > 0 else 0.0

  def calculate_metrics(
      self, times_ms: list[float], **params
  ) -> dict[str, Any]:
    metrics = super().calculate_metrics(times_ms, **params)
    m, k, n = (
        params.get("m", 1024),
        params.get("k", 1024),
        params.get("n", 1024),
    )

    total_flops = 2 * m * n * k
    if params.get("use_scaling_factors", False):
      total_flops += m * n

    avg_latency_s = metrics["avg_ms"] / 1000.0
    if avg_latency_s == 0:
      tflops_per_sec = float("inf")
    else:
      tflops_per_sec = (total_flops / avg_latency_s) / 1e12

    metrics["tflops_per_sec"] = tflops_per_sec
    metrics["total_flops"] = total_flops
    metrics["intensity"] = self.get_arithmetic_intensity(**params)
    return metrics
