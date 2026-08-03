"""Matrix multiplication and GEMM benchmarks including FP8 support."""

import dataclasses
from typing import Any
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class GemmParams(base.BaseBenchmarkParams):
  m: int = 1024
  k: int = 1024
  n: int = 1024
  in_dtype: str = "float8_e4m3fn"
  out_dtype: str = "bfloat16"
  seed: int = 0
  use_scaling_factors: bool = False


@registry.benchmark_registry.register("gemm_generalized")
class GeneralizedGemmBenchmark(base.BaseBenchmark[GemmParams]):
  Config = GemmParams
  """Generalized GEMM benchmark supporting FP8 and throughput projection.

  Pseudo-code: OUT = matmul(IN0, IN1) * rescaling_factor
  """

  def get_compute_dtype(self) -> str:
    return self.config.in_dtype

  def match_xprof_op_fallback(self, event):
    args = event.get("args", {})
    hlo_category = args.get("hlo_category", "")
    return hlo_category.strip('"') == "convolution fusion"

  def setup(self):
    out_dtype = utils.parse_dtype(self.config.out_dtype)

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

  def get_run_identifier(self) -> str:
    return f"m_{self.config.m}_k_{self.config.k}_n_{self.config.n}_{self.config.in_dtype}_to_{self.config.out_dtype}"

  def generate_inputs(self) -> tuple[Any, ...]:
    m, k, n = self.config.m, self.config.k, self.config.n
    # Resolve dtypes
    in_dtype = utils.parse_dtype(self.config.in_dtype)

    key = jax.random.PRNGKey(self.config.seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Data generation in HBM
    # Note: JAX might require intermediate conversion for random.normal if
    # dtypes aren't supported
    a = jax.random.normal(k1, (m, k)).astype(in_dtype)
    b = jax.random.normal(k2, (k, n)).astype(in_dtype)

    # Optional scaling factors
    use_sf = self.config.use_scaling_factors
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

  def get_total_bytes(self) -> float:
    m, k, n = self.config.m, self.config.k, self.config.n
    in_itemsize = jnp.dtype(utils.parse_dtype(self.config.in_dtype)).itemsize
    out_itemsize = jnp.dtype(utils.parse_dtype(self.config.out_dtype)).itemsize

    # Bytes = Load(A) + Load(B) + Store(Out)
    # Scaling factors are small (row-wise), ignoring for intensity but could be
    # added.
    return (
        (m * k * in_itemsize) + (k * n * in_itemsize) + (m * n * out_itemsize)
    )

  def get_arithmetic_intensity(self) -> float:
    m, k, n = self.config.m, self.config.k, self.config.n
    flops = 2 * m * n * k
    bytes_moved = self.get_total_bytes()
    return flops / bytes_moved if bytes_moved > 0 else 0.0

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    metrics = super().calculate_metrics(times_ms)
    m, k, n = self.config.m, self.config.k, self.config.n

    total_flops = 2 * m * n * k
    if self.config.use_scaling_factors:
      total_flops += m * n

    avg_latency_s = metrics["avg_ms"] / 1000.0
    if avg_latency_s == 0:
      tflops_per_sec = float("inf")
    else:
      tflops_per_sec = (total_flops / avg_latency_s) / 1e12

    metrics["tflops_per_sec"] = tflops_per_sec
    metrics["total_flops"] = total_flops
    metrics["intensity"] = self.get_arithmetic_intensity()
    return metrics
