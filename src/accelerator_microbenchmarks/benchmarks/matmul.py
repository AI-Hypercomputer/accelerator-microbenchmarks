"""Matrix multiplication and GEMM benchmarks including FP8 support."""

import abc
import dataclasses
from typing import Any, Callable, Optional, Sequence, TypeVar
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import checksum
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
  in_dtype: str = base.dtype_field(
      default="bfloat16",
      metadata={
          "help": "Input operand data type (e.g. float8_e4m3fn, bfloat16)."
      },
  )
  out_dtype: str = base.dtype_field(
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


TGemmConfig = TypeVar("TGemmConfig", bound=base.BaseBenchmarkParams)


def _sample_normal(key: jax.Array, shape: tuple[int, ...], dtype: Any):
  """Module-level so `jax.jit` reuses its compile cache across calls."""
  return jax.random.normal(key, shape).astype(dtype)


class BaseGemmBenchmark(base.BaseBenchmark[TGemmConfig]):
  """Base class for GEMM benchmarks.

  Holds the logic that does not depend on the params class: the compute
  roofline, the XProf op fallback, and the intensity and TFLOPS metrics derived
  from `get_total_flops()` and `get_total_bytes()`. Subclasses implement those
  two, and `get_compute_dtype()`, from their own params.
  """

  roofline_mode: constants.RooflineMode = constants.RooflineMode.COMPUTE

  @abc.abstractmethod
  def get_total_flops(self) -> float:
    """Returns the FLOPs of one `run_op` call."""

  def match_xprof_op_fallback(self, event):
    args = event.get("args", {})
    hlo_category = args.get("hlo_category", "")
    return hlo_category.strip('"') == "convolution fusion"

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


@registry.benchmark_registry.register("gemm", aliases=["gemm_generalized"])
class GeneralizedGemmBenchmark(BaseGemmBenchmark[GemmParams]):
  """Generalized GEMM benchmark supporting FP8 and throughput projection.

  Pseudo-code: OUT = matmul(IN0, IN1) * rescaling_factor
  """

  Config = GemmParams
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

  def _replicated_normal(
      self, key: jax.Array, shape: tuple[int, ...], dtype: Any
  ) -> jax.Array:
    """Returns standard-normal samples replicated across the mesh.

    Generated on every mesh device under `jit` with a replicated
    `out_shardings` (as in `hbm.py`), so no device-to-device broadcast is
    needed; required for `single_host_tpu_init`, where such transfers hang.
    """
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    replicated_sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(*([None] * len(shape)))
    )
    generate = jax.jit(
        _sample_normal,
        static_argnums=(1, 2),
        out_shardings=replicated_sharding,
    )
    return generate(key, shape, dtype)

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

    # Data generation in HBM, replicated across the mesh to avoid
    # discrepancies between devices.
    # Note: JAX might require intermediate conversion for random.normal if
    # dtypes aren't supported
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (n, k) if transpose_b else (k, n)
    a = self._replicated_normal(k1, a_shape, in_dtype)
    b = self._replicated_normal(k2, b_shape, in_dtype)

    # Row and column scaling factor vectors (sf0: m x 1, sf1: 1 x n)
    # Unpacked as sf0, sf1 args into gemm_fn(a, b, sf0, sf1, c) if enabled,
    # else (None, None).
    if self.config.use_scaling_factors:
      sf_args = [
          self._replicated_normal(k3, (m, 1), jnp.float32),
          self._replicated_normal(k4, (1, n), jnp.float32),
      ]
    else:
      sf_args = [None, None]

    # Accumulator matrix C (M x N) for fused GEMM addition when beta != 0.0.
    # Unpacked as c arg into gemm_fn(a, b, sf0, sf1, c) if enabled, else [].
    if self.config.beta != 0.0:
      acc_args = [self._replicated_normal(k5, (m, n), in_dtype)]
    else:
      acc_args = []

    return (a, b, *sf_args, *acc_args)

  def run_op(self, *args) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(*args)

  def get_compute_dtype(self) -> str:
    """Return the primary data type used for compute math, to determine peak TFLOPS."""
    return self.config.in_dtype

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


@dataclasses.dataclass
class GemmChainedParams(base.BaseBenchmarkParams):
  """Configuration parameters for chained GEMM determinism benchmark.

  Only the parameters of the chain `X <- X @ B` are exposed: X is M x K and B is
  a square K x K matrix, so there is no separate N.
  """

  # Integer dtypes are rejected: B ~ N(0, 1) / sqrt(k) truncates to all zeros,
  # so the chain collapses to 0 and every device trivially matches.
  DTYPE_CONSTRAINTS = {
      "in_dtype": (jnp.floating,),
      "out_dtype": (jnp.floating,),
  }

  # Overrides the inherited field, which keeps its place in the field order.
  # The checksums of every timed run are gathered across hosts, which needs
  # the same number of runs on every host (a mismatch can hang the gather). A
  # min_duration_s loop stops on each host's own clock and cannot guarantee
  # that, while num_runs alone does.
  min_duration_s: float = dataclasses.field(
      default=0.0,
      metadata={
          "min": 0.0,
          "max": 0.0,
          "help": (
              "Not supported; must be 0. Every host must run exactly num_runs"
              " timed runs so that the checksums of every run can be compared"
              " across hosts. Increase num_runs for more coverage."
          ),
      },
  )

  m: int = dataclasses.field(
      default=1024,
      metadata={"min": 1, "help": "Matrix dimension M (rows of X)."},
  )
  k: int = dataclasses.field(
      default=1024,
      metadata={
          "min": 1,
          "help": "Matrix dimension K (columns of X; B is K x K).",
      },
  )
  in_dtype: str = base.dtype_field(
      default="bfloat16",
      metadata={
          "help": (
              "Input operand data type of each matmul (e.g. float8_e4m3fn,"
              " bfloat16)."
          )
      },
  )
  out_dtype: str = base.dtype_field(
      default="bfloat16",
      metadata={"help": "Data type of X after each matmul (e.g. bfloat16)."},
  )
  seed: int = dataclasses.field(
      default=0,
      metadata={"min": 0, "help": "Random seed for X and B initialization."},
  )
  chain_length: int = dataclasses.field(
      default=1,
      metadata={
          "min": 1,
          "help": "Number of sequential matrix multiplications in the chain.",
      },
  )

  def __post_init__(self):
    super().__post_init__()
    for name in ("in_dtype", "out_dtype"):
      dtype_str = getattr(self, name)
      # The element width must be supported by the checksum.
      if not checksum.is_checksummable_dtype(utils.parse_dtype(dtype_str)):
        raise ValueError(
            f"gemm_chained requires a {name} with a bit width in"
            f" {checksum.SUPPORTED_BITS}, got {dtype_str!r}."
        )


def make_inputs_fn(
    mesh: jax.sharding.Mesh,
    m: int,
    k: int,
    in_dtype: Any,
    seed: int,
):
  """Returns the jitted `() -> (x0, b)` that generates the chain's inputs.

  X0 (M x K) and B (K x K, scaled by 1/sqrt(K)) are seeded Gaussians cast to
  `in_dtype` and replicated over `mesh`. Every device generates its own replica,
  from a PRNG key built inside the jitted function: nothing is computed on one
  device and copied to the others, so a fault during input generation only
  affects the faulty device's checksum.
  """
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

  def inputs_fn():
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed), 2)
    x0 = jax.random.normal(k1, (m, k)).astype(in_dtype)
    b = (jax.random.normal(k2, (k, k)) / jnp.sqrt(jnp.float32(k))).astype(
        in_dtype
    )
    return x0, b

  return jax.jit(inputs_fn, out_shardings=(replicated, replicated))


def make_chain_fn(
    mesh: jax.sharding.Mesh,
    chain_length: int,
    in_dtype: Any,
    out_dtype: Any,
):
  """Returns the jitted chain `(x0, b) -> X`, i.e. the timed `run_op`.

  Every device of `mesh` runs, on its own replicas and without collectives,
  `chain_length` steps of `X <- dot(X.astype(in_dtype), b).astype(out_dtype)`
  under the MARKER scope. Block `d` of X along axis 0 is the final X of mesh
  device `d`, i.e. the layout `checksum.make_checksum_fn` takes.

  Raises:
    ValueError: If `chain_length` < 1.
  """
  if chain_length < 1:
    raise ValueError(f"chain_length must be >= 1, got {chain_length}.")
  pspec = jax.sharding.PartitionSpec

  def per_device(x0, b):

    def step(_, x):
      return jax.lax.dot_general(
          x.astype(in_dtype), b, dimension_numbers=(((1,), (0,)), ((), ()))
      ).astype(out_dtype)

    with jax.named_scope(constants.MARKER):
      return jax.lax.fori_loop(0, chain_length, step, x0.astype(out_dtype))

  @jax.jit
  def chain_fn(x0, b):
    return jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(pspec(), pspec()),
        out_specs=pspec(tuple(mesh.axis_names)),
        check_vma=False,
    )(x0, b)

  return chain_fn


@registry.benchmark_registry.register("gemm_chained", is_experimental=True)
class GemmChainedBenchmark(BaseGemmBenchmark[GemmChainedParams]):
  """[EXPERIMENTAL] Chained GEMM determinism benchmark with cross-host per-device checksum comparison."""

  # Runtime: the params dataclass the CLI / YAML loader instantiates.
  Config = GemmChainedParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("in_dtype", report.format_str),
      ("out_dtype", report.format_str),
      ("m", report.format_str),
      ("k", report.format_str),
      ("chain_length", report.format_str),
      (checksum.ALL_DEVICES_VALID_CHECKSUM, report.format_str),
      (checksum.INTRA_HOST_CHECKSUM_MATCH, report.format_str),
      (checksum.GLOBAL_CHECKSUM_MATCH, report.format_str),
      ("total_flops", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_tflops_per_chip", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_tflops_per_chip", report.format_2f),
  )

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._checksum_fn: Optional[Callable[[jax.Array], jax.Array]] = None
    # This host's `checksum.checksum_local_rows` result of each timed run, in
    # run order.
    self._local_runs: list[Any] = []

  def setup(self):
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    in_dtype = utils.parse_dtype(self.config.in_dtype)
    out_dtype = utils.parse_dtype(self.config.out_dtype)
    self._jit_fn = make_chain_fn(
        mesh=self.mesh,
        chain_length=self.config.chain_length,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
    )
    self._checksum_fn = checksum.make_checksum_fn(self.mesh)
    self._local_runs = []

  def get_run_identifier(self) -> str:
    return (
        f"m_{self.config.m}_k_{self.config.k}_"
        f"{self.config.in_dtype}_to_{self.config.out_dtype}_"
        f"chain_{self.config.chain_length}"
    )

  def generate_inputs(self) -> tuple[Any, ...]:
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    inputs_fn = make_inputs_fn(
        mesh=self.mesh,
        m=self.config.m,
        k=self.config.k,
        in_dtype=utils.parse_dtype(self.config.in_dtype),
        seed=self.config.seed,
    )
    return inputs_fn()

  def run_op(self, *args) -> jax.Array:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    # Timed: runs only the chain. Each run's final X is checksummed in
    # `post_run_processing`, after the run's timer stops.
    return self._jit_fn(*args)

  def post_run_processing(self, outputs: jax.Array, run_index: int) -> None:
    del run_index  # The rows are kept in run order.
    if self._checksum_fn is None:
      raise ValueError("Checksum function not initialized.")
    # Blocks on the readback, so the checksum does not run in the next run's
    # timed window. Never raises once initialized.
    self._local_runs.append(
        checksum.checksum_local_rows(outputs, self._checksum_fn)
    )

  def get_total_bytes(self) -> float:
    m, k = self.config.m, self.config.k
    in_itemsize = jnp.dtype(utils.parse_dtype(self.config.in_dtype)).itemsize
    out_itemsize = jnp.dtype(utils.parse_dtype(self.config.out_dtype)).itemsize
    # Per step: load X (M x K) and B (K x K), store X (M x K).
    step_bytes = (m * k + k * k) * in_itemsize + m * k * out_itemsize
    return float(self.config.chain_length * step_bytes)

  def get_total_flops(self) -> float:
    m, k = self.config.m, self.config.k
    return float(self.config.chain_length * 2 * m * k * k)

  def get_compute_dtype(self) -> str:
    """Return the primary data type used for compute math, to determine peak TFLOPS."""
    return self.config.in_dtype

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    metrics = super().calculate_metrics(times_ms)
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    metrics.update(
        checksum.gather_and_compare_checksums(self._local_runs, self.mesh)
    )
    return metrics
