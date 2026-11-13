"""Matrix multiplication and GEMM benchmarks including FP8 support."""

import abc
import dataclasses
import time
from typing import Any, Callable, Optional, Sequence, TypeVar
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import checksum
from accelerator_microbenchmarks.core import config
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import dtypes
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

# Unquantized floating-point dtypes supported for inputs and outputs.
_UNQUANTIZED_FLOAT_DTYPES = (
    jnp.bfloat16,
    jnp.float32,
)


def _accumulation_dtype(lhs_dtype: str, rhs_dtype: str) -> Any:
  """MXU accumulation dtype: int32 if both inputs are integers, else float32."""
  if all(jnp.issubdtype(dtypes.parse_dtype(d), jnp.integer)
         for d in (lhs_dtype, rhs_dtype)):
    return jnp.int32
  return jnp.float32


def _resolve_rhs_dtype(params: Any) -> None:
  """Defaults `rhs_dtype` to `in_dtype` and validates it like `in_dtype`."""
  params.rhs_dtype = params._to_allowed_dtype(  # pylint: disable=protected-access
      "in_dtype", params.rhs_dtype or params.in_dtype
  )


def _compute_dtype(lhs_dtype: str, rhs_dtype: str) -> str:
  """The wider operand sets the MXU path (e.g. FP8 for FP8 x INT4)."""
  return str(max(lhs_dtype, rhs_dtype, key=dtypes.get_dtype_bits))


@dataclasses.dataclass
class MatmulShapeParams(base.BaseBenchmarkParams):
  """Shared matrix shape, dtype, and seed parameters for GEMM benchmarks."""

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
  in_dtype: dtypes.DType = base.dtype_field(
      default=dtypes.DType.BFLOAT16,
      metadata={
          "help": "Input operand data type (e.g. float8_e4m3fn, bfloat16)."
      },
  )
  out_dtype: dtypes.DType = base.dtype_field(
      default=dtypes.DType.BFLOAT16,
      metadata={
          "help": "Output accumulation/result data type (e.g. bfloat16)."
      },
  )
  seed: int = dataclasses.field(
      default=0,
      metadata={"min": 0, "help": "Random seed for tensor initialization."},
  )


@dataclasses.dataclass
class GemmParams(MatmulShapeParams):
  """Configuration parameters for GEMM benchmark."""

  rhs_dtype: str = dataclasses.field(
      default="",
      metadata={
          "help": (
              "Data type of B (weights); empty = in_dtype. E.g. in_dtype"
              " float8_e4m3fn with rhs_dtype int4 for an FP8 x INT4 GEMM."
          )
      },
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

  def __post_init__(self):
    super().__post_init__()
    _resolve_rhs_dtype(self)


TGemmConfig = TypeVar("TGemmConfig", bound=base.BaseBenchmarkParams)
TGemmParams = TypeVar("TGemmParams", bound=GemmParams)


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

  def _replicated_normal(
      self, key: jax.Array, shape: tuple[int, ...], dtype: Any
  ) -> jax.Array:
    """Samples a normal array replicated on every mesh device.

    Generating under `jit` with a replicated `out_shardings` (as in `hbm.py`)
    produces each shard on its own device, and the key is passed as a host
    array, so no data moves between devices. The result equals eager sampling
    with the same key.
    """
    # Replicated sharding for computation ops to avoid discrepancies
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    replicated_sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None)
    )
    generate = jax.jit(
        _sample_normal,
        static_argnums=(1, 2),
        out_shardings=replicated_sharding,
    )
    return generate(np.asarray(key), shape, dtype)


@registry.benchmark_registry.register("gemm", aliases=["gemm_generalized"])
class GeneralizedGemmBenchmark(BaseGemmBenchmark[TGemmParams]):
  """Generalized GEMM benchmark supporting FP8 and throughput projection.

  Pseudo-code: OUT = matmul(IN0, IN1) * rescaling_factor
  """

  Config = GemmParams
  # Per-device GEMM on replicated inputs; no inter-device (ICI) traffic.
  requires_ici: bool = False
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("in_dtype", report.format_str),
      ("rhs_dtype", report.format_str),
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
    out_dtype = dtypes.parse_dtype(self.config.out_dtype)
    acc_dtype = _accumulation_dtype(self.config.in_dtype, self.config.rhs_dtype)
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
            preferred_element_type=acc_dtype,
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
    if self.config.rhs_dtype != self.config.in_dtype:
      run_identifier += f"_rhs_{self.config.rhs_dtype}"
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
    in_dtype = dtypes.parse_dtype(self.config.in_dtype)

    key = jax.random.PRNGKey(self.config.seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # Data generation in HBM
    # Note: JAX might require intermediate conversion for random.normal if
    # dtypes aren't supported
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (n, k) if transpose_b else (k, n)
    a = self._replicated_normal(k1, a_shape, in_dtype)
    b = self._replicated_normal(
        k2, b_shape, dtypes.parse_dtype(self.config.rhs_dtype)
    )

    # Optional scaling factors
    use_sf = self.config.use_scaling_factors
    sf0 = self._replicated_normal(k3, (m, 1), jnp.float32) if use_sf else None
    sf1 = self._replicated_normal(k4, (1, n), jnp.float32) if use_sf else None

    use_accumulator_matrix = True if self.config.beta != 0.0 else False
    c = (
        self._replicated_normal(
            k5, (m, n), dtypes.parse_dtype(self.config.out_dtype)
        )
        if use_accumulator_matrix
        else None
    )

    # Row and column scaling factor vectors (sf0: m x 1, sf1: 1 x n)
    # Unpacked as sf0, sf1 args into gemm_fn(a, b, sf0, sf1, c) if enabled,
    # else (None, None).
    if use_sf:
      sf_args = [sf0, sf1]
    else:
      sf_args = [None, None]

    # Accumulator matrix C (M x N) for fused GEMM addition when beta != 0.0.
    # Unpacked as c arg into gemm_fn(a, b, sf0, sf1, c) if enabled, else [].
    if use_accumulator_matrix:
      acc_args = [c]
    else:
      acc_args = []

    return (a, b, *sf_args, *acc_args)

  def run_op(self, *args) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(*args)

  def get_compute_dtype(self) -> str:
    """Return the primary data type used for compute math, to determine peak TFLOPS."""
    return _compute_dtype(self.config.in_dtype, self.config.rhs_dtype)

  def get_total_bytes(self) -> float:
    m, k, n = self.config.m, self.config.k, self.config.n
    # Sub-byte dtypes (int4, float4_e2m1fn) are counted at their packed size.
    in_itemsize = dtypes.get_dtype_bytes(self.config.in_dtype)
    rhs_itemsize = dtypes.get_dtype_bytes(self.config.rhs_dtype)
    out_itemsize = dtypes.get_dtype_bytes(self.config.out_dtype)

    # Base memory traffic: Load A (M x K) and B (K x N), Store Out (M x N)
    bytes_a = m * k * in_itemsize
    bytes_b = k * n * rhs_itemsize
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


_SOAKING_THERMAL_REPORT_COLUMNS: tuple[
    tuple[str, Callable[[Any], str]], ...
] = (
    (
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
        (k, report.format_2f) for k in constants.THERMAL_PRE_SOAKING_METRIC_KEYS
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


@registry.benchmark_registry.register("gemm_throttling", is_experimental=True)
class GemmThrottlingBenchmark(GeneralizedGemmBenchmark[GemmThrottlingParams]):
  """[EXPERIMENTAL] GEMM throttling benchmark with 3-phase soaking execution and XProf thermal metrics."""

  Config = GemmThrottlingParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      tuple(GeneralizedGemmBenchmark.REPORT_SCHEMA)
      + _SOAKING_THERMAL_REPORT_COLUMNS
  )


# Dtypes of the chained GEMM's X and B: the floats whose bit width the checksum
# supports (checksum.SUPPORTED_BITS), so float64 and float4_e2m1fn are excluded.
_CHAIN_DTYPES = tuple(
    d.jnp_dtype.type
    for d in dtypes.dtypes_matching(jnp.floating)
    if checksum.is_checksummable_dtype(d.jnp_dtype)
)


@dataclasses.dataclass
class GemmChainedParams(base.BaseBenchmarkParams):
  """Configuration parameters for chained GEMM determinism benchmark.

  Only the parameters of the chain `X <- X @ B` are exposed: X is M x K and B is
  a square K x K matrix, so there is no separate N.
  """

  # Floats only (see _CHAIN_DTYPES): integer B ~ N(0, 1) / sqrt(k) truncates to
  # all zeros, so the chain collapses to 0 and every device trivially matches.
  DTYPE_CONSTRAINTS = {
      "in_dtype": _CHAIN_DTYPES,
      "out_dtype": _CHAIN_DTYPES,
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
  in_dtype: dtypes.DType = base.dtype_field(
      default=dtypes.DType.BFLOAT16,
      metadata={
          "help": (
              "Input operand data type of each matmul (e.g. float8_e4m3fn,"
              " bfloat16)."
          )
      },
  )
  out_dtype: dtypes.DType = base.dtype_field(
      default=dtypes.DType.BFLOAT16,
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


TGemmChainedConfig = TypeVar("TGemmChainedConfig", bound=GemmChainedParams)


@registry.benchmark_registry.register("gemm_chained", is_experimental=True)
class GemmChainedBenchmark(BaseGemmBenchmark[TGemmChainedConfig]):
  """[EXPERIMENTAL] Chained GEMM determinism benchmark with cross-host per-device checksum comparison."""

  # Runtime: the params dataclass the CLI / YAML loader instantiates.
  Config = GemmChainedParams
  # Per-device chained GEMM with host-side checksum comparison; no ICI.
  requires_ici: bool = False
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
    in_dtype = dtypes.parse_dtype(self.config.in_dtype)
    out_dtype = dtypes.parse_dtype(self.config.out_dtype)
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
        in_dtype=dtypes.parse_dtype(self.config.in_dtype),
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
    in_itemsize = jnp.dtype(dtypes.parse_dtype(self.config.in_dtype)).itemsize
    out_itemsize = jnp.dtype(dtypes.parse_dtype(self.config.out_dtype)).itemsize
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


@dataclasses.dataclass
class GemmPulseParams(config.PulseExecutionParamsMixin, GemmChainedParams):
  """Configuration parameters for multi-cycle chained GEMM pulse benchmark."""

  min_duration_s: float = dataclasses.field(
      default=0.0,
      metadata={
          "min": 0.0,
          "help": (
              "Minimum total duration in seconds of the untraced middle"
              " soaking phase between pre_soaking_window and"
              " post_soaking_window (>= 0.0)."
          ),
      },
  )
  enable_checksum: bool = dataclasses.field(
      default=True,
      metadata={
          "help": "Compute and verify per-device 64-bit checksums per window.",
      },
  )


@registry.benchmark_registry.register("gemm_pulse", is_experimental=True)
class GemmPulseBenchmark(GemmChainedBenchmark[GemmPulseParams]):
  """[EXPERIMENTAL] Chained GEMM multi-cycle determinism and pulse benchmark."""

  Config = GemmPulseParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      tuple(GemmChainedBenchmark.REPORT_SCHEMA)
      + (
          ("num_cycles", report.format_str),
          ("idle_s", report.format_str),
          ("enable_checksum", report.format_str),
      )
      + _SOAKING_THERMAL_REPORT_COLUMNS
  )

  def get_run_identifier(self) -> str:
    return (
        f"{super().get_run_identifier()}_cycles_{self.config.num_cycles}_"
        f"idle_{self.config.idle_s}_dur_{self.config.min_duration_s}_"
        f"checksum_{self.config.enable_checksum}"
    )

  def post_run_processing(self, outputs: jax.Array, run_index: int) -> None:
    if self.config.enable_checksum:
      super().post_run_processing(outputs, run_index)

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    if self.config.enable_checksum:
      return super().calculate_metrics(times_ms)
    return super(GemmChainedBenchmark, self).calculate_metrics(times_ms)


class QuantScaling(constants.ParamEnum):
  """Calibration method used to derive the quantization scaling factors.

  Attributes:
    DYNAMIC: Scales are computed on the fly from the absolute maximum of each
      row or column (`scale = amax / Q_MAX`).
    STATIC: Uses a fixed, compile-time clip bound `0.5 * Q_MAX` (`scale = 0.5`)
      with no `amax` reduction. A non-unit (`!= 1.0`) power-of-2 scale prevents
      XLA from folding away the scale/dequantize multiplications as identity
      no-ops while scaling inputs without mantissa rounding loss.
  """

  DYNAMIC = "dynamic"
  STATIC = "static"


def _quantize_per_channel(
    x: jax.Array,
    *,
    reduce_axis: int,
    quant_dtype: jax.typing.DTypeLike,
    scaling: QuantScaling,
) -> tuple[jax.Array, jax.Array]:
  """Quantizes `x` per channel along `reduce_axis`."""
  q_dtype = jnp.dtype(quant_dtype)
  q_max = float(dtypes.get_dtype_max(q_dtype))
  x_f32 = x.astype(jnp.float32)

  if scaling == QuantScaling.STATIC:
    scale_shape = list(x.shape)
    scale_shape[reduce_axis] = 1
    # Use a non-unit power-of-2 bound (`0.5 * q_max` -> `scale = 0.5`) so XLA
    # cannot elide `x * inv_scale` or `acc * (sf_a * sf_b)` as multiply-by-1
    # no-ops.
    static_absmax = 0.5 * q_max
    scale_f32 = jnp.full(scale_shape, static_absmax / q_max, dtype=jnp.float32)
    inv_scale_f32 = jnp.reciprocal(scale_f32)
  elif scaling == QuantScaling.DYNAMIC:
    amax = jnp.max(jnp.abs(x_f32), axis=reduce_axis, keepdims=True)
    clamped_amax = jnp.maximum(amax, 1e-7)
    scale_f32 = clamped_amax / q_max
    inv_scale_f32 = q_max / clamped_amax
  else:
    raise ValueError(
        f"Unsupported scaling mode '{scaling}'; expected 'dynamic' or 'static'."
    )

  q = x_f32 * inv_scale_f32
  if jnp.issubdtype(q_dtype, jnp.integer):
    q = jnp.clip(jnp.round(q), -q_max, q_max)
  else:
    q = jnp.clip(q, -q_max, q_max)
  return q.astype(q_dtype), scale_f32


@dataclasses.dataclass
class QuantizedGemmParams(MatmulShapeParams):
  """Configuration parameters for the `gemm_quantized` microbenchmark."""

  DTYPE_CONSTRAINTS = {
      "in_dtype": _UNQUANTIZED_FLOAT_DTYPES,
      "quant_dtype": (jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.int8),
      "rhs_quant_dtype": (
          jnp.float8_e4m3fn,
          jnp.float8_e5m2,
          jnp.int8,
          jnp.int4,
      ),
      "out_dtype": _UNQUANTIZED_FLOAT_DTYPES,
  }

  quant_dtype: dtypes.DType = base.dtype_field(
      default=dtypes.DType.FLOAT8_E4M3FN,
      metadata={
          "help": (
              "Quantized LHS (and default RHS) operand data type fed to the"
              " MXU (e.g. 'float8_e4m3fn', 'int8')."
          )
      },
  )
  rhs_quant_dtype: dtypes.DType = base.dtype_field(
      default=dtypes.DType.FLOAT8_E4M3FN,
      metadata={
          "help": (
              "Quantized RHS (B) operand data type. Defaults to `quant_dtype`;"
              " also supports 'int4' when `quant_dtype` is FP8"
              " ('float8_e4m3fn' or 'float8_e5m2')."
          )
      },
  )
  scaling: QuantScaling = dataclasses.field(
      default=QuantScaling.DYNAMIC,
      metadata={
          "help": (
              "Scale calibration: 'dynamic' (amax computed on the fly) or"
              " 'static' (fixed clip bound 0.5 * Q_MAX, i.e. scale = 0.5)."
          )
      },
  )
  dequantize: bool = dataclasses.field(
      default=False,
      metadata={
          "help": (
              "If True, multiply the accumulator by the row and column"
              " dequantization scales before casting to out_dtype."
          )
      },
  )

  def __post_init__(self):
    if "rhs_quant_dtype" not in self._explicit_fields:
      self.rhs_quant_dtype = self.quant_dtype
    super().__post_init__()
    if self.rhs_quant_dtype != self.quant_dtype and not (
        self.rhs_quant_dtype == dtypes.DType.INT4
        and self.quant_dtype in (
            dtypes.DType.FLOAT8_E4M3FN,
            dtypes.DType.FLOAT8_E5M2,
        )
    ):
      raise ValueError(
          "rhs_quant_dtype must match quant_dtype or be 'int4' paired with an"
          " FP8 quant_dtype ('float8_e4m3fn' or 'float8_e5m2'), got"
          f" quant_dtype='{self.quant_dtype}',"
          f" rhs_quant_dtype='{self.rhs_quant_dtype}'."
      )

  def with_generation_defaults(
      self, param_defaults: dict[str, Any]
  ) -> base.BaseBenchmarkParams:
    if (
        param_defaults
        and "quant_dtype" in param_defaults
        and "quant_dtype" not in self._explicit_fields
        and "rhs_quant_dtype" not in param_defaults
        and "rhs_quant_dtype" not in self._explicit_fields
    ):
      param_defaults = {
          **param_defaults,
          "rhs_quant_dtype": param_defaults["quant_dtype"],
      }
    return super().with_generation_defaults(param_defaults)


@registry.benchmark_registry.register("gemm_quantized", is_experimental=True)
class QuantizedGemmBenchmark(BaseGemmBenchmark[QuantizedGemmParams]):
  """[EXPERIMENTAL] GEMM with per-channel quantization of A and B."""

  Config = QuantizedGemmParams  # pylint: disable=invalid-name
  requires_ici: bool = False
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("in_dtype", report.format_str),
      ("quant_dtype", report.format_str),
      ("rhs_quant_dtype", report.format_str),
      ("out_dtype", report.format_str),
      ("m", report.format_str),
      ("k", report.format_str),
      ("n", report.format_str),
      ("scaling", report.format_str),
      ("dequantize", report.format_str),
      ("total_flops", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_tflops_per_chip", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_tflops_per_chip", report.format_2f),
  )

  def get_compute_dtype(self) -> str:
    """Returns the LHS quantized dtype, which determines the MXU peak TFLOPS."""
    return str(self.config.quant_dtype)

  def setup(self):
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    cfg = self.config
    quant_dtype = dtypes.parse_dtype(cfg.quant_dtype)
    rhs_quant_dtype = dtypes.parse_dtype(cfg.rhs_quant_dtype)
    out_dtype = dtypes.parse_dtype(cfg.out_dtype)
    both_int = jnp.issubdtype(quant_dtype, jnp.integer) and jnp.issubdtype(
        rhs_quant_dtype, jnp.integer
    )
    dot_acc_dtype = jnp.int32 if both_int else jnp.float32

    @jax.jit
    def quant_gemm_fn(a, b):
      with jax.named_scope(constants.MARKER):
        qa, sf_a = _quantize_per_channel(
            a,
            reduce_axis=1,
            quant_dtype=quant_dtype,
            scaling=cfg.scaling,
        )
        qb, sf_b = _quantize_per_channel(
            b,
            reduce_axis=0,
            quant_dtype=rhs_quant_dtype,
            scaling=cfg.scaling,
        )
        acc = jax.lax.dot_general(
            qa,
            qb,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=dot_acc_dtype,
        )
        if cfg.dequantize:
          acc = acc.astype(jnp.float32) * (sf_a * sf_b)
        return acc.astype(out_dtype)

    self._jit_fn = quant_gemm_fn

  def get_run_identifier(self) -> str:
    cfg = self.config
    quant_str = (
        f"{cfg.quant_dtype}"
        if cfg.rhs_quant_dtype == cfg.quant_dtype
        else f"{cfg.quant_dtype}_rhs_{cfg.rhs_quant_dtype}"
    )
    run_id = (
        f"m_{cfg.m}_k_{cfg.k}_n_{cfg.n}_"
        f"{cfg.in_dtype}_to_{quant_str}_{cfg.scaling}_out_{cfg.out_dtype}"
    )
    if cfg.dequantize:
      run_id += "_dequant"
    return run_id

  def generate_inputs(self) -> tuple[Any, ...]:
    cfg = self.config
    in_dtype = dtypes.parse_dtype(cfg.in_dtype)
    key_a, key_b = jax.random.split(jax.random.PRNGKey(cfg.seed))
    a = self._replicated_normal(key_a, (cfg.m, cfg.k), in_dtype)
    b = self._replicated_normal(key_b, (cfg.k, cfg.n), in_dtype)
    return (a, b)

  def run_op(self, *args) -> jax.Array:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(*args)

  def get_total_bytes(self) -> float:
    """Returns 1-pass algorithmic HBM bytes (read A, read B, write OUT)."""
    cfg = self.config
    m, k, n = cfg.m, cfg.k, cfg.n
    in_bytes = dtypes.get_dtype_bytes(cfg.in_dtype)
    out_bytes = dtypes.get_dtype_bytes(cfg.out_dtype)
    return float((m * k + k * n) * in_bytes + m * n * out_bytes)

  def get_total_flops(self) -> float:
    m, k, n = self.config.m, self.config.k, self.config.n
    return float(2 * m * n * k)


_DUTY_CYCLE_DELAY_SHAPE = (256, 256)  # f32 tensor updated by the idle steps.
_CALIBRATION_GEMMS = 20
_CALIBRATION_ELEM_OPS = 1000


@dataclasses.dataclass
class GemmDutyCycleParams(MatmulShapeParams):
  """Parameters for the GEMM duty-cycle benchmark."""

  warmup_tries: int = dataclasses.field(
      default=2, metadata={"min": 0, "help": "Warmup runs."}
  )
  num_runs: int = dataclasses.field(
      default=5, metadata={"min": 0, "help": "Timed runs."}
  )
  rhs_dtype: str = dataclasses.field(
      default="", metadata={"help": "Data type of B; empty = in_dtype."}
  )
  duty_cycle: float = dataclasses.field(
      default=1.0,
      metadata={
          "min": 0.01,
          "max": 1.0,
          "help": "Target fraction of time spent in the GEMM.",
      },
  )
  gemms_per_run: int = dataclasses.field(
      default=100,
      metadata={"min": 1, "help": "GEMM + idle iterations per timed run."},
  )
  num_elem_ops: int = dataclasses.field(
      default=-1,
      metadata={
          "min": -1,
          "help": "Idle steps after each GEMM; -1 calibrates from duty_cycle.",
      },
  )

  def __post_init__(self):
    super().__post_init__()
    _resolve_rhs_dtype(self)


def make_duty_cycle_fn(
    mesh: jax.sharding.Mesh, acc_dtype: Any, out_dtype: Any
) -> Callable[..., jax.Array]:
  """Returns jitted `(a, b, num_gemms, num_elem_ops) -> carry`.

  Each device runs `num_gemms` iterations of a GEMM followed by `num_elem_ops`
  elementwise steps seeded from the GEMM result. Optimization barriers keep XLA
  from hoisting the GEMM, overlapping it with the idle steps, or computing only
  the element the idle steps read. The loop bounds are traced, so calibration
  and timed runs share one executable.
  """
  pspec = jax.sharding.PartitionSpec

  def per_device(a, b, num_gemms, num_elem_ops):
    def gemm_step(state):
      i, carry = state
      a_i, b_i, carry = jax.lax.optimization_barrier((a, b, carry))
      res = jax.lax.dot_general(
          a_i,
          b_i,
          dimension_numbers=(((1,), (0,)), ((), ())),
          preferred_element_type=acc_dtype,
      ).astype(out_dtype)
      res = jax.lax.optimization_barrier(res)
      delay = (
          jnp.zeros(_DUTY_CYCLE_DELAY_SHAPE, jnp.float32)
          + res[0, 0].astype(jnp.float32) * 1e-20
          + carry
      )
      delay = jax.lax.fori_loop(0, num_elem_ops, lambda _, r: r + 1.0, delay)
      return i + 1, delay[0, 0] * 1e-20

    with jax.named_scope(constants.MARKER):
      _, carry = jax.lax.while_loop(
          lambda state: state[0] < num_gemms,
          gemm_step,
          (jnp.int32(0), jnp.float32(0.0)),
      )
    return carry.reshape(1)

  return jax.jit(
      jax.shard_map(
          per_device,
          mesh=mesh,
          in_specs=(pspec(),) * 4,
          out_specs=pspec(tuple(mesh.axis_names)),
          check_vma=False,
      )
  )


def calibrate_num_elem_ops(
    gemm_s: float, elem_op_s: float, duty_cycle: float
) -> int:
  """Idle steps per GEMM for `duty_cycle` (0 at 100%, else at least 1)."""
  if duty_cycle >= 1.0:
    return 0
  if elem_op_s <= 0:
    raise ValueError(f"elem_op_s must be positive, got {elem_op_s}.")
  return max(1, round(gemm_s * (1.0 - duty_cycle) / duty_cycle / elem_op_s))


@registry.benchmark_registry.register("gemm_duty_cycle", is_experimental=True)
class GemmDutyCycleBenchmark(BaseGemmBenchmark[GemmDutyCycleParams]):
  """[EXPERIMENTAL] GEMM at a target duty cycle (GEMM, then elementwise idle).

  `*_tflops_per_device` is the average over the run, `*_active_tflops_per_device`
  is that / `duty_cycle`, and `*_duty_cycle_pct` is the share of the run the
  GEMMs take at their calibrated speed (it drops if the run throttles).
  """

  Config = GemmDutyCycleParams
  requires_ici: bool = False
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("in_dtype", report.format_str),
      ("rhs_dtype", report.format_str),
      ("out_dtype", report.format_str),
      ("m", report.format_str),
      ("k", report.format_str),
      ("n", report.format_str),
      ("duty_cycle", report.format_2f),
      ("gemms_per_run", report.format_str),
      ("num_elem_ops", report.format_str),
      ("calibrated_gemm_us", report.format_2f),
      ("calibrated_elem_op_us", report.format_4f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_tflops_per_device", report.format_2f),
      ("wall_clock_duty_cycle_pct", report.format_2f),
      ("wall_clock_active_tflops_per_device", report.format_2f),
  )

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._num_elem_ops: int | None = None
    self._gemm_s: float | None = None
    self._elem_op_s: float | None = None

  def setup(self):
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    self._jit_fn = make_duty_cycle_fn(
        self.mesh,
        _accumulation_dtype(self.config.in_dtype, self.config.rhs_dtype),
        dtypes.parse_dtype(self.config.out_dtype),
    )

  def get_run_identifier(self) -> str:
    c = self.config
    return (
        f"m_{c.m}_k_{c.k}_n_{c.n}_{c.in_dtype}_x_{c.rhs_dtype}_to_"
        f"{c.out_dtype}_duty_{c.duty_cycle}_gemms_{c.gemms_per_run}"
    )

  def _scalar(self, value: int) -> jax.Array:
    assert self.mesh is not None
    return jax.device_put(
        jnp.int32(value),
        jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec()),
    )

  def _time_loop(self, a, b, num_gemms: int, num_elem_ops: int) -> float:
    """Best wall-clock seconds of 3 runs, process 0's value on every host."""
    assert self._jit_fn is not None
    args = (a, b, self._scalar(num_gemms), self._scalar(num_elem_ops))
    jax.block_until_ready(self._jit_fn(*args))
    best_s = float("inf")
    for _ in range(3):
      start_s = time.perf_counter()
      jax.block_until_ready(self._jit_fn(*args))
      best_s = min(best_s, time.perf_counter() - start_s)
    # Every host must run the same loop bounds.
    if jax.process_count() > 1:
      best_s = float(multihost_utils.broadcast_one_to_all(np.float64(best_s)))
    return best_s

  def _calibrate(self, a, b) -> None:
    """Sets `_gemm_s`, `_elem_op_s` and `_num_elem_ops`."""
    n = _CALIBRATION_GEMMS
    t_n = self._time_loop(a, b, n, 0)
    # The difference cancels the fixed dispatch overhead.
    self._gemm_s = max(self._time_loop(a, b, 2 * n, 0) - t_n, 0.0) / n
    if self.config.num_elem_ops >= 0 or self.config.duty_cycle >= 1.0:
      self._num_elem_ops = max(self.config.num_elem_ops, 0)
      return
    e = _CALIBRATION_ELEM_OPS
    self._elem_op_s = max(self._time_loop(a, b, n, e) - t_n, 0.0) / (n * e)
    k = calibrate_num_elem_ops(
        self._gemm_s, self._elem_op_s, self.config.duty_cycle
    )
    # Re-measure at k and 2k idle steps: with idle gaps the GEMM throttles less
    # than back to back, so solve both times in the target regime.
    t_k = self._time_loop(a, b, n, k)
    elem_op_s = (self._time_loop(a, b, n, 2 * k) - t_k) / (n * k)
    gemm_s = t_k / n - k * elem_op_s
    if elem_op_s > 0 and gemm_s > 0:
      self._gemm_s, self._elem_op_s = gemm_s, elem_op_s
      k = calibrate_num_elem_ops(gemm_s, elem_op_s, self.config.duty_cycle)
    self._num_elem_ops = k

  def generate_inputs(self) -> tuple[jax.Array, ...]:
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    c = self.config
    k1, k2 = jax.random.split(jax.random.PRNGKey(c.seed), 2)
    a = self._replicated_normal(
        k1, (c.m, c.k), dtypes.parse_dtype(c.in_dtype)
    )
    b = self._replicated_normal(
        k2, (c.k, c.n), dtypes.parse_dtype(c.rhs_dtype)
    )
    self._calibrate(a, b)
    assert self._num_elem_ops is not None
    return (
        a,
        b,
        self._scalar(c.gemms_per_run),
        self._scalar(self._num_elem_ops),
    )

  def run_op(self, *args) -> jax.Array:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(*args)

  def match_xprof_op_fallback(self, event):
    del event  # Only the MARKER-scoped loop includes the idle time.
    return False

  def get_compute_dtype(self) -> str:
    """Return the primary data type used for compute math, to determine peak TFLOPS."""
    return _compute_dtype(self.config.in_dtype, self.config.rhs_dtype)

  def get_total_bytes(self) -> float:
    c = self.config
    gemm_bytes = (
        c.m * c.k * dtypes.get_dtype_bytes(c.in_dtype)
        + c.k * c.n * dtypes.get_dtype_bytes(c.rhs_dtype)
        + c.m * c.n * dtypes.get_dtype_bytes(c.out_dtype)
    )
    return float(c.gemms_per_run * gemm_bytes)

  def get_total_flops(self) -> float:
    c = self.config
    return float(c.gemms_per_run * 2 * c.m * c.n * c.k)

  def get_workload_metadata(self) -> dict[str, Any]:
    us = lambda s: s * 1e6 if s is not None else None
    return {
        **super().get_workload_metadata(),
        "num_elem_ops": self._num_elem_ops,
        "calibrated_gemm_us": us(self._gemm_s),
        "calibrated_elem_op_us": us(self._elem_op_s),
    }

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    metrics = super().calculate_throughput_metrics(latency_ms, prefix)
    latency_s = latency_ms / 1000.0
    metrics[f"{prefix}_duty_cycle_pct"] = (
        self.config.gemms_per_run * self._gemm_s / latency_s * 100.0
        if self._gemm_s and latency_s > 0
        else None
    )
    metrics[f"{prefix}_active_tflops_per_device"] = (
        metrics[f"{prefix}_tflops_per_device"] / self.config.duty_cycle
    )
    return metrics
