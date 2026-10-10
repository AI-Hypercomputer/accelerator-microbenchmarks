"""Matrix multiplication and GEMM benchmarks including FP8 support."""

import dataclasses
from typing import Any, Callable, Sequence
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import config
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class GemmParams(base.BaseBenchmarkParams):
  """Configuration parameters for GEMM benchmark."""

  m: int = dataclasses.field(
      default=1024,
      metadata={
          "min": 1,
          "help": "Matrix dimension M (rows of operand A / output C).",
      },
  )
  k: int = dataclasses.field(
      default=1024,
      metadata={
          "min": 1,
          "help": "Contracting dimension K (columns of A / rows of B).",
      },
  )
  n: int = dataclasses.field(
      default=1024,
      metadata={
          "min": 1,
          "help": "Matrix dimension N (columns of operand B / output C).",
      },
  )
  in_dtype: str = dataclasses.field(
      default="bfloat16",
      metadata={
          "help": "Input operand data type (e.g. float8_e4m3fn, bfloat16)."
      },
  )
  out_dtype: str = dataclasses.field(
      default="bfloat16",
      metadata={
          "help": "Output accumulation/result data type (e.g. bfloat16)."
      },
  )

  seed: int = dataclasses.field(
      default=0,
      metadata={"min": 0, "help": "Random seed for tensor initialization."},
  )
  use_scaling_factors: bool = dataclasses.field(
      default=False,
      metadata={"help": "Apply row-wise/column-wise FP8 scaling factors."},
  )
  transpose_a: bool = dataclasses.field(
      default=False,
      metadata={"help": "Whether to transpose operand matrix A."},
  )
  transpose_b: bool = dataclasses.field(
      default=False,
      metadata={"help": "Whether to transpose operand matrix B."},
  )
  alpha: float = dataclasses.field(
      default=1.0,
      metadata={"help": "Scalar multiplier for GEMM product (alpha * A @ B)."},
  )
  beta: float = dataclasses.field(
      default=0.0,
      metadata={"help": "Scalar multiplier"
                        " for accumulator matrix C (beta * C)."},
  )


@registry.benchmark_registry.register("gemm", aliases=["gemm_generalized"])
class GeneralizedGemmBenchmark(base.BaseBenchmark[GemmParams]):
  """Generalized GEMM benchmark supporting FP8 and throughput projection.

  Pseudo-code: OUT = matmul(IN0, IN1) * rescaling_factor
  """

  Config = GemmParams
  roofline_mode: constants.RooflineMode = constants.RooflineMode.COMPUTE
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("in_dtype", report.format_str),
      ("out_dtype", report.format_str),
      ("m", report.format_str),
      ("k", report.format_str),
      ("n", report.format_str),
      ("transpose_a", report.format_str),
      ("transpose_b", report.format_str),
      ("alpha", report.format_str),
      ("beta", report.format_str),
      ("use_scaling_factors", report.format_str),
      ("total_flops", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_tflops_per_chip", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_tflops_per_chip", report.format_2f),
  )

  def get_compute_dtype(self) -> str:
    """Return the primary data type used for compute math, to determine peak TFLOPS."""
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


@dataclasses.dataclass
class GemmThrottlingParams(config.SoakingExecutionParamsMixin, GemmParams):
  """Parameters for GEMM throttling benchmark.

  Inherits soaking window batching (`samples_per_run`) and Phase 2 soak duration
  validation (`min_duration_s > 0.0`; `num_runs` is not used in soaking mode)
  from `SoakingExecutionParamsMixin`, `warmup_tries` and `min_duration_s` from
  `BaseBenchmarkParams`, and matrix shape/dtype parameters from `GemmParams`.
  """


@registry.benchmark_registry.register("gemm_throttling", is_experimental=True)
class GemmThrottlingBenchmark(GeneralizedGemmBenchmark):
  """[EXPERIMENTAL] GEMM throttling benchmark with 3-phase soaking execution and XProf thermal metrics."""

  Config = GemmThrottlingParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      tuple(GeneralizedGemmBenchmark.REPORT_SCHEMA)
      + (
          (constants.WALL_CLOCK_PRE_SOAKING_P50_MS, report.format_4f),
          (constants.WALL_CLOCK_POST_SOAKING_P50_MS, report.format_4f),
          (constants.WALL_CLOCK_SLOWDOWN_RATIO, report.format_2f),
          (
              constants.WALL_CLOCK_PRE_SOAKING_TFLOPS_PER_DEVICE,
              report.format_2f,
          ),
          (
              constants.WALL_CLOCK_POST_SOAKING_TFLOPS_PER_DEVICE,
              report.format_2f,
          ),
          (constants.XPROF_PRE_SOAKING_P50_MS, report.format_4f),
          (constants.XPROF_POST_SOAKING_P50_MS, report.format_4f),
          (constants.XPROF_SLOWDOWN_RATIO, report.format_2f),
          (constants.XPROF_PRE_SOAKING_TFLOPS_PER_DEVICE, report.format_2f),
          (constants.XPROF_POST_SOAKING_TFLOPS_PER_DEVICE, report.format_2f),
      )
      + tuple(
          (k, report.format_2f)
          for k in constants.THERMAL_PRE_SOAKING_METRIC_KEYS
      )
      + tuple(
          (k, report.format_2f)
          for k in constants.THERMAL_METRIC_KEYS
          if k != constants.PEAK_TEMP_SOURCE
      )
      + (
          (constants.PEAK_TEMP_SOURCE, report.format_str),
          (constants.PSTATE_REQUESTED, report.format_str),
      )
  )

  def derive_chip_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = super().derive_chip_metrics(metrics)
    total_flops = self.get_total_flops()

    for domain in (
        constants.TimingDomain.WALL_CLOCK,
        constants.TimingDomain.XPROF,
    ):
      pre_ms = metrics.get(f"{domain}_{constants.PRE_SOAKING_P50_MS}")
      if pre_ms is not None:
        pre_tflops = (
            (total_flops / (pre_ms / 1000.0)) / 1e12
            if pre_ms > 0
            else float("inf")
        )
        metrics[f"{domain}_{constants.PRE_SOAKING}_tflops_per_device"] = (
            pre_tflops
        )

      post_ms = metrics.get(f"{domain}_{constants.POST_SOAKING_P50_MS}")
      if post_ms is not None:
        post_tflops = (
            (total_flops / (post_ms / 1000.0)) / 1e12
            if post_ms > 0
            else float("inf")
        )
        metrics[f"{domain}_{constants.POST_SOAKING}_tflops_per_device"] = (
            post_tflops
        )

    return metrics
