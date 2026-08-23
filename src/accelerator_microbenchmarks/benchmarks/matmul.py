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
  transpose_a: bool = False
  transpose_b: bool = False
  alpha: float = 1.0
  beta: float = 0.0


@registry.benchmark_registry.register("gemm", aliases=["gemm_generalized"])
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
    alpha = self.config.alpha
    beta = self.config.beta

    @jax.jit
    def gemm_fn(a, b, sf0=None, sf1=None, c=None):
      with jax.named_scope(constants.MARKER):
        # Standard matmul
        lhs_contracting_dim = (0,) if self.config.transpose_a else (1,)
        rhs_contracting_dim = (1,) if self.config.transpose_b else (0,)
        out = jax.lax.dot_general(
            a,
            b,
            dimension_numbers=(
                (lhs_contracting_dim, rhs_contracting_dim),
                ((), ()),
            ),
        )
        if alpha != 1.0:
          out = out * alpha
        if c is not None:
          out = out + (c * beta if beta != 1.0 else c)
        # Optional rescaling (row-wise scaling factors as requested)
        if sf0 is not None and sf1 is not None:
          # Assuming rowwise scaling factor SF0<M, 1> and SF1<1, N>
          out = out * (sf0 @ sf1)
        return out.astype(out_dtype)

    self._jit_fn = gemm_fn

  def get_run_identifier(self) -> str:
    run_identifier = (
        f"m_{self.config.m}_k_{self.config.k}_n_{self.config.n}_"
        f"{self.config.in_dtype}_to_{self.config.out_dtype}_"
        f"ta_{self.config.transpose_a}_tb_{self.config.transpose_b}_"
        f"alpha_{self.config.alpha}"
    )
    if self.config.beta != 0.0:
      run_identifier += f"_beta_{self.config.beta}"
    if self.config.use_scaling_factors:
      run_identifier += "_sf"
    return run_identifier

  def generate_inputs(self) -> tuple[Any, ...]:
    (
        m,
        k,
        n,
        transpose_a,
        transpose_b,
    ) = (
        self.config.m,
        self.config.k,
        self.config.n,
        self.config.transpose_a,
        self.config.transpose_b,
    )


    # Resolve dtypes
    in_dtype = utils.parse_dtype(self.config.in_dtype)

    key = jax.random.PRNGKey(self.config.seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # Data generation in HBM
    # Note: JAX might require intermediate conversion for random.normal if
    # dtypes aren't supported
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (n, k) if transpose_b else (k, n)
    a = jax.random.normal(k1, a_shape).astype(in_dtype)
    b = jax.random.normal(k2, b_shape).astype(in_dtype)

    # Optional scaling factors
    use_sf = self.config.use_scaling_factors
    sf0 = jax.random.normal(k3, (m, 1)).astype(jnp.float32) if use_sf else None
    sf1 = jax.random.normal(k4, (1, n)).astype(jnp.float32) if use_sf else None

    use_accumulator_matrix = True if self.config.beta != 0.0 else False
    c = (
        jax.random.normal(k5, (m, n)).astype(in_dtype)
        if use_accumulator_matrix
        else None
    )

    # Replicated sharding for computation ops to avoid discrepancies
    assert self.mesh is not None, "Mesh not initialized."
    replicated_sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None)
    )

    # Row and column scaling factor vectors (sf0: m x 1, sf1: 1 x n)
    # Unpacked as sf0, sf1 args into gemm_fn(a, b, sf0, sf1, c) if enabled,
    # else (None, None).
    if use_sf:
      sf_args = [
          jax.device_put(sf0, replicated_sharding),
          jax.device_put(sf1, replicated_sharding),
      ]
    else:
      sf_args = [None, None]

    # Accumulator matrix C (M x N) for fused GEMM addition when beta != 0.0.
    # Unpacked as c arg into gemm_fn(a, b, sf0, sf1, c) if enabled, else [].
    if use_accumulator_matrix:
      acc_args = [jax.device_put(c, replicated_sharding)]
    else:
      acc_args = []

    return (
        jax.device_put(a, replicated_sharding),
        jax.device_put(b, replicated_sharding),
        *sf_args,
        *acc_args,
    )


  def run_op(self, *args) -> jnp.ndarray:
    assert self._jit_fn is not None
    return self._jit_fn(*args)

  def get_total_bytes(self) -> float:
    m, k, n = self.config.m, self.config.k, self.config.n
    in_itemsize = jnp.dtype(utils.parse_dtype(self.config.in_dtype)).itemsize
    out_itemsize = jnp.dtype(utils.parse_dtype(self.config.out_dtype)).itemsize

    # Base memory traffic: Load A (M x K) and B (K x N), Store Out (M x N)
    bytes_a = m * k * in_itemsize
    bytes_b = k * n * in_itemsize
    bytes_out = m * n * out_itemsize

    # Optional memory traffic: Load C (M x N) when accumulator matrix is used (beta != 0.0).
    # Note: Row-wise scaling factors (sf0, sf1) are small and ignored.
    bytes_c = m * n * out_itemsize if self.config.beta != 0.0 else 0

    return float(bytes_a + bytes_b + bytes_out + bytes_c)

  def get_total_flops(self) -> float:
    m, k, n = self.config.m, self.config.k, self.config.n
    flops = 2 * m * n * k
    if self.config.use_scaling_factors:
      flops += 2 * m * n
    if self.config.alpha != 1.0:
      flops += m * n
    if self.config.beta != 0.0:
      flops += 2 * m * n if self.config.beta != 1.0 else m * n
    return float(flops)

  def get_arithmetic_intensity(self) -> float:
    flops = self.get_total_flops()
    bytes_moved = self.get_total_bytes()
    return flops / bytes_moved if bytes_moved > 0 else 0.0

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    metrics = super().calculate_metrics(times_ms)
    total_flops = self.get_total_flops()

    avg_latency_s = metrics["avg_ms"] / 1000.0
    if avg_latency_s == 0:
      tflops_per_sec = float("inf")
    else:
      tflops_per_sec = (total_flops / avg_latency_s) / 1e12

    metrics["tflops_per_sec"] = tflops_per_sec
    metrics["total_flops"] = total_flops
    metrics["intensity"] = self.get_arithmetic_intensity()
    return metrics
