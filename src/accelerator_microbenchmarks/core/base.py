"""Base class for all JAX benchmarks."""

import abc
import dataclasses
import datetime
import enum
import numbers
import os
import time
from typing import Any, Callable, ClassVar, Generic, Optional, Sequence, TypeVar

from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import executor
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import profiler
from accelerator_microbenchmarks.core import roofline
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.core import utils
import jax
import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class XprofConfig:
  """Configuration for XProf trace collection and device timing analysis."""

  xprof_timing: bool = False
  xprof_dir: str = "/tmp/tensorboard"
  device_timing_aggregation: str = (
      constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST
  )


@dataclasses.dataclass
class BaseBenchmarkParams:
  # jnp scalar types (e.g. `jnp.bfloat16`) or abstract dtype classes (e.g.
  # `jnp.floating`) the `dtype` field must be a sub-dtype of; None allows any
  # dtype.
  SUPPORTED_DTYPES: ClassVar[Sequence[jax.typing.DTypeLike] | None] = None

  warmup_tries: int = dataclasses.field(
      default=10,
      metadata={
          "min": 0,
          "help": "Number of warmup iterations before measurement.",
      },
  )
  num_runs: int = dataclasses.field(
      default=10,
      metadata={"min": 0, "help": "Number of measurement iterations."},
  )
  min_duration_s: float = dataclasses.field(
      default=0.0,
      metadata={
          "min": 0.0,
          "help": (
              "Minimum measurement duration in seconds. Also runs warmup for"
              " at least min_duration_s / 5 (capped at 1.0s)."
          ),
      },
  )

  def _validate_bounds(self) -> None:
    """Checks every field against its min/max metadata constraints.

    Raises:
      ValueError: If any field violates its `min`/`max` bound or integer type.
    """
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      minimum = field.metadata.get("min")
      maximum = field.metadata.get("max")

      if field.type is int:
        # bool is an Integral subclass, so reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
          bound_str = f" >= {minimum}" if minimum is not None else ""
          raise ValueError(
              f"{field.name} must be an integer{bound_str}, got {value!r}"
          )
        # Normalize integer-like values (e.g. numpy ints) so bounds apply.
        value = int(value)
        setattr(self, field.name, value)
      elif value is None:
        continue
      elif not isinstance(value, (int, float)) or isinstance(value, bool):
        continue

      if minimum is not None and value < minimum:
        raise ValueError(f"{field.name} must be >= {minimum}, got {value}")

      if maximum is not None and value > maximum:
        raise ValueError(f"{field.name} must be <= {maximum}, got {value}")

  def _coerce_enums(self) -> None:
    """Coerces every enum field in-place into a member of its Enum type.

    Raises:
      ValueError: If any field is not a valid member of its Enum type.
    """
    for field in dataclasses.fields(self):
      # Checks if the field is an enum by coercing raw values into their Enum
      # member. Only scalar enum fields are supported, not containers such as
      # list[Enum].
      if isinstance(field.type, type) and issubclass(field.type, enum.Enum):
        enum_type: type[enum.Enum] = field.type
      else:
        continue

      value = getattr(self, field.name)
      try:
        setattr(self, field.name, enum_type(value))
      except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid {field.name} '{value}'. Must be one of"
            f" {[member.value for member in enum_type]}."
        ) from exc

  def _validate_dtypes(self) -> None:
    """Validates every `dtype` / `*_dtype` field via `utils.validate_dtype`.

    The `dtype` field must also match the class's `SUPPORTED_DTYPES`, if set.

    Raises:
      ValueError: If any dtype field is invalid or unsupported.
    """
    for field in dataclasses.fields(self):
      if field.name != "dtype" and not field.name.endswith("_dtype"):
        continue
      value = getattr(self, field.name)
      try:
        utils.validate_dtype(
            value, self.SUPPORTED_DTYPES if field.name == "dtype" else None
        )
      except ValueError as exc:
        raise ValueError(f"{field.name}: {exc}") from exc

  def __post_init__(self):
    self._coerce_enums()
    self._validate_bounds()
    self._validate_dtypes()

  def expand_test_cases(self) -> Sequence["BaseBenchmarkParams"]:
    """Default 1-to-1 mapping: returns [self]."""
    return [self]


@dataclasses.dataclass
class SingleDtypeBenchmarkParams(BaseBenchmarkParams):
  """Base parameters for benchmarks using a single tensor data type."""

  dtype: str = dataclasses.field(
      default="bfloat16",
      metadata={
          "help": "Data type for tensor operations (e.g. bfloat16, float32)."
      },
  )


@dataclasses.dataclass
class BenchmarkMetadata:
  """Metadata for a single benchmark run."""

  benchmark_name: str
  test_name: str
  start_time: str
  end_time: str
  params: dict[str, Any]
  platform_info: platform.PlatformInfo
  hardware_spec: system.HardwareSpec
  xla_flags: str = ""
  libtpu_init_args: str = ""
  xprof_config: Optional[XprofConfig] = None


@dataclasses.dataclass
class BenchmarkResult:
  """Consolidated result of a benchmark run."""

  metadata: BenchmarkMetadata
  metrics: dict[str, Any]
  raw_times_ms: list[float]


TConfig = TypeVar("TConfig", bound=BaseBenchmarkParams)


class BaseBenchmark(Generic[TConfig], abc.ABC):
  """Abstract base class for microbenchmarks."""

  name: str = ""
  Config = BaseBenchmarkParams
  is_experimental: bool = False
  DEFAULT_LOCAL_DEVICE_ID: int = 0
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = ()
  # TODO(b/555967269): Redesign and decouple circular dependency between
  # BaseBenchmark and report formatters.
  REPORT_FORMATTERS: Optional[
      Sequence[Callable[[pd.DataFrame, type["BaseBenchmark"]], str]]
  ] = None
  derive_chip_bandwidth: bool = True
  roofline_mode: constants.RooflineMode = constants.RooflineMode.NONE

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if "name" not in cls.__dict__:
      cls.name = cls.__name__

  def __init__(
      self,
      config: TConfig,
      hardware_spec: system.HardwareSpec,
      mesh: Optional[jax.sharding.Mesh] = None,
      xprof_config: Optional[XprofConfig] = None,
  ):
    self.config: TConfig = config
    self.hardware_spec: system.HardwareSpec = hardware_spec
    self.mesh: Optional[jax.sharding.Mesh] = mesh
    self.xprof_config: XprofConfig = xprof_config or XprofConfig()
    self._jit_fn = None
    self._xprof_dir_actual: str = self.xprof_config.xprof_dir
    self._xprof_dir_cns: str = self._xprof_dir_actual
    self.executor: executor.BaseRunExecutor = executor.create_executor(self)

  def _create_default_mesh(self) -> jax.sharding.Mesh:
    """Create a default 1D mesh spanning all available devices."""
    devices = jax.devices()
    return jax.sharding.Mesh(np.array(devices), axis_names=("device",))

  @abc.abstractmethod
  def run_op(self, *args, **kwargs) -> Any:
    """The core operation intended for performance assessment."""

  def setup(self):
    """Perform setup such as JIT compilation or buffer pre-allocation."""
    # Mesh creation is deferred to the run method.
    pass

  def teardown(self) -> None:
    """Release resources acquired in setup() (e.g. pinned host buffers).

    Called by run() after every run, including when the run raises, so
    resources held by one sweep point are released before the next sweep
    point's benchmark instance calls setup(). Implementations must be
    idempotent and safe to call before setup().
    """
    pass

  def reset_data(self, *inputs, **kwargs) -> tuple[jax.Array, ...]:
    """Reset data for the next run."""
    return inputs

  def post_run_processing(self, outputs: Any, run_index: int) -> None:
    """Process the outputs of each timed run, outside the timer.

    Called only by BaseRunExecutor, not in warmup or by SoakingRunExecutor.
    Runs inside the profiler trace, so it must not emit MARKER named scopes,
    and must synchronize any device work it dispatches before returning.
    """
    del outputs, run_index  # Unused in base class.

  def get_run_identifier(self) -> str:
    """Return a string identifier for the current run parameters."""
    return ""

  @abc.abstractmethod
  def generate_inputs(self) -> tuple[Any, ...]:
    """Generate or retrieve inputs for the benchmark.

    Returns:
      A tuple of arguments passed to run_op.
    """
    pass

  def get_arithmetic_intensity(self) -> float:
    """Return operational intensity (FLOPs/Byte) for roofline analysis."""
    raise NotImplementedError(
        f"{self.__class__.__name__} must implement get_arithmetic_intensity() "
        "when roofline_mode is COMPUTE or MEMORY_HBM."
    )

  def get_peak_ici_bandwidth_per_chip(self) -> float | None:
    """Return the theoretical peak ICI bus bandwidth per chip (GB/s).

    Required when roofline_mode == RooflineMode.NETWORK_ICI. Returns None
    when the peak cannot be modeled for the current configuration.
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} must implement"
        " get_peak_ici_bandwidth_per_chip() when roofline_mode =="
        " RooflineMode.NETWORK_ICI."
    )

  @abc.abstractmethod
  def get_total_bytes(self) -> float:
    """Calculate total bytes moved to/from HBM."""
    pass

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    """Derive static workload metadata, host wall-clock latency statistics, and domain throughput metrics."""
    metrics = self.get_workload_metadata()
    if self.roofline_mode in (
        constants.RooflineMode.COMPUTE,
        constants.RooflineMode.MEMORY_HBM,
    ):
      metrics["intensity"] = self.get_arithmetic_intensity()
    latency_stats = self.calculate_latency_stats(
        times_ms, prefix=constants.TimingDomain.WALL_CLOCK
    )
    metrics.update(latency_stats)
    metrics.update(
        self.calculate_throughput_metrics(
            latency_ms=latency_stats[
                f"{constants.TimingDomain.WALL_CLOCK}_p50_ms"
            ],
            prefix=constants.TimingDomain.WALL_CLOCK,
        )
    )
    self.executor.augment_host_metrics(metrics)
    return metrics

  def get_workload_metadata(self) -> dict[str, Any]:
    """Return static, timing-invariant workload metadata (e.g., total_flops, data_size_mib)."""
    return {}

  def calculate_latency_stats(
      self, times_ms: list[float], prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    """Derive statistical latency metrics (avg, p50, p90, std) from raw timing data."""
    return self.executor.calculate_latency_stats(times_ms, prefix=prefix)

  @abc.abstractmethod
  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    """Calculate domain-prefixed throughput metrics from a representative latency (ms)."""

  def match_xprof_op_fallback(self, event: dict[str, Any]) -> bool:
    """Fallback to capture relevant xprof op when MARKER is not present."""
    del event  # Unused in base class.
    return False

  def apply_roofline_analysis(self, metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply roofline estimation to finalized metrics."""

    return roofline.apply_roofline_analysis(self, metrics)

  def get_compute_dtype(self) -> str:
    """Return the primary data type used for compute math, to determine peak TFLOPS."""
    if not hasattr(self.config, "dtype"):
      raise ValueError(
          f"Benchmark config {self.config} does not have a 'dtype' attribute. "
          "Please implement get_compute_dtype() in the subclass."
      )
    return self.config.dtype

  def get_device_to_measure(self) -> jax.Device:
    """Returns the local JAX Device to observe in XProf (default: local device 0)."""
    return jax.local_devices()[self.DEFAULT_LOCAL_DEVICE_ID]

  @property
  def requires_multihost_sync(self) -> bool:
    """Whether this benchmark targets a single remote host and requires broadcast to collect metrics.

    By default (symmetric workloads like matmul, collectives, hbm), each host
    independently parses its own local XProf trace without broadcasting.
    """
    return False

  @property
  def xprof_target_host_cpu(self) -> bool:
    """Whether XProf timing measures /host:CPU events instead of /device:TPU:{id}.

    Default is False (measures accelerator hardware kernels on
    /device:TPU:{id}). HostToDeviceBenchmark overrides this to True because
    H2D/D2H DMA transfers are recorded on the CPU host process.
    """
    return False

  def _apply_xprof_timing_and_sync(
      self, metrics: dict[str, Any]
  ) -> dict[str, Any]:
    """Parses XProf trace, syncs across hosts, and recalculates metrics."""
    xprof_dir = self._xprof_dir_actual
    cns_dir = self._xprof_dir_cns

    xprof_url = profiler.upload_xprof_trace(xprof_dir, cns_dir)
    if xprof_url:
      metrics["xprof_url"] = xprof_url

    all_devs_str = [f"{d.id}(p{d.process_index})" for d in jax.devices()]
    local_devs_str = [
        f"{d.id}(p{d.process_index})" for d in jax.local_devices()
    ]
    print(f"[Host {jax.process_index()}] All devices: {all_devs_str}")
    print(f"[Host {jax.process_index()}] Local devices: {local_devs_str}")
    durations: list[float] = []
    try:
      durations = self.executor.get_xprof_durations(xprof_dir)
    except ValueError:
      raise
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error parsing local XProf trace: {e}")

    xprof_avg, xprof_p50, xprof_p90, xprof_std = 0.0, 0.0, 0.0, 0.0
    if durations:
      print(
          f"Using XProf device timings ({len(durations)} runs)"
          " to calculate derived performance metrics."
      )
      xprof_stats = self.calculate_latency_stats(
          durations, prefix=constants.TimingDomain.XPROF
      )
      xprof_avg = xprof_stats[f"{constants.TimingDomain.XPROF}_avg_ms"]
      xprof_p50 = xprof_stats[f"{constants.TimingDomain.XPROF}_p50_ms"]
      xprof_p90 = xprof_stats[f"{constants.TimingDomain.XPROF}_p90_ms"]
      xprof_std = xprof_stats[f"{constants.TimingDomain.XPROF}_std_ms"]
    else:
      print("Warning: No XProf device timings found locally.")

    metrics["xprof_captured_runs"] = len(durations)

    # Apply XProf timings or fallback to None for unmeasured durations
    if xprof_p50 > 0:
      metrics.update({
          f"{constants.TimingDomain.XPROF}_avg_ms": xprof_avg,
          f"{constants.TimingDomain.XPROF}_p50_ms": xprof_p50,
          f"{constants.TimingDomain.XPROF}_p90_ms": xprof_p90,
          f"{constants.TimingDomain.XPROF}_std_ms": xprof_std,
      })
      self.executor.augment_xprof_metrics(metrics, has_xprof_timings=True)
      metrics.update(
          self.calculate_throughput_metrics(
              latency_ms=xprof_p50, prefix=constants.TimingDomain.XPROF
          )
      )
    else:
      print(
          "Warning: No valid XProf timings recorded; setting XProf metrics to"
          " None."
      )
      metrics.update({
          f"{constants.TimingDomain.XPROF}_avg_ms": None,
          f"{constants.TimingDomain.XPROF}_p50_ms": None,
          f"{constants.TimingDomain.XPROF}_p90_ms": None,
          f"{constants.TimingDomain.XPROF}_std_ms": None,
      })
      self.executor.augment_xprof_metrics(metrics, has_xprof_timings=False)
      for wall_clock_key, xprof_key in (
          (
              f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_device_gb_s",
              f"{constants.TimingDomain.XPROF}_bandwidth_per_device_gb_s",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_host_gb_s",
              f"{constants.TimingDomain.XPROF}_bandwidth_per_host_gb_s",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_chip_gb_s",
              f"{constants.TimingDomain.XPROF}_bandwidth_per_chip_gb_s",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_host_gbps",
              f"{constants.TimingDomain.XPROF}_bandwidth_per_host_gbps",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_tflops_per_device",
              f"{constants.TimingDomain.XPROF}_tflops_per_device",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_tflops_per_chip",
              f"{constants.TimingDomain.XPROF}_tflops_per_chip",
          ),
      ):
        if wall_clock_key in metrics:
          metrics[xprof_key] = None
    return metrics

  def derive_chip_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
    """Derives per-chip metrics from per-device metrics for available domains."""
    devices_per_chip = self.hardware_spec.devices_per_chip
    for prefix in constants.TimingDomain:
      tflops_dev_key = f"{prefix}_tflops_per_device"
      tflops_chip_key = f"{prefix}_tflops_per_chip"
      if tflops_dev_key in metrics and tflops_chip_key not in metrics:
        metrics[tflops_chip_key] = (
            metrics[tflops_dev_key] * devices_per_chip
            if metrics[tflops_dev_key] is not None
            else None
        )

      bw_dev_key = f"{prefix}_bandwidth_per_device_gb_s"
      bw_chip_key = f"{prefix}_bandwidth_per_chip_gb_s"
      if (
          self.derive_chip_bandwidth
          and bw_dev_key in metrics
          and bw_chip_key not in metrics
      ):
        metrics[bw_chip_key] = (
            metrics[bw_dev_key] * devices_per_chip
            if metrics[bw_dev_key] is not None
            else None
        )
    return metrics

  def run(self) -> BenchmarkResult:
    """Standard orchestration flow for a benchmark.

    Always calls teardown() afterwards, even if the run raises.

    Returns:
      The consolidated BenchmarkResult of the run.
    """
    try:
      return self._run()
    finally:
      self.teardown()

  def _run(self) -> BenchmarkResult:
    """Runs setup, warmup, measurement, and metric finalization."""
    if self.is_experimental:
      print(
          "===================================================================="
      )
      print(
          f"[WARNING] Running experimental benchmark '{self.name}'. Metrics"
          " and API are subject to change."
      )
      print(
          "===================================================================="
      )

    # 0. Initialize mesh if not provided
    if self.mesh is None:
      self.mesh = self._create_default_mesh()

    # 1. Setup & Initial Inputs
    self.setup()
    inputs = self.generate_inputs()

    # 2. Warmup & JIT Compilation
    inputs = self.executor.run_warmup(inputs)

    benchmark_name = self.name
    if self.xprof_config.xprof_timing:
      try:
        jax.profiler.stop_trace()
      except RuntimeError:
        # No active profiler trace is running; safe to ignore.
        pass
      xprof_base_dir = self.xprof_config.xprof_dir
      timestamp = int(time.time())

      run_id = self.get_run_identifier()
      dir_suffix = f"_{run_id}" if run_id else ""

      cns_xprof_dir = os.path.join(
          xprof_base_dir, f"{benchmark_name}{dir_suffix}_{timestamp}"
      )
      local_xprof_dir = f"/tmp/microbenchmarks_tmptrace/{benchmark_name}{dir_suffix}_{timestamp}"

      if not (
          cns_xprof_dir.startswith("/cns/")
          or cns_xprof_dir.startswith("/bigstore/")
      ):
        local_xprof_dir = cns_xprof_dir

      self._xprof_dir_actual = local_xprof_dir
      self._xprof_dir_cns = cns_xprof_dir
      print(
          f"Collecting xprof trace locally to {local_xprof_dir} across runs..."
      )

    start_ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    # 3. Measurement Loop
    loop_start = time.perf_counter()
    exec_result = self.executor.execute(
        inputs,
        self._xprof_dir_actual if self.xprof_config.xprof_timing else None,
    )

    if self.xprof_config.xprof_timing:
      print("Xprof trace collected.")

    end_ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    # 4. Finalize Results
    # Calculate host-side metrics
    metrics = self.calculate_metrics(exec_result.raw_times_ms)

    if self.xprof_config.xprof_timing:
      metrics = self._apply_xprof_timing_and_sync(metrics)
      expected = exec_result.xprof_expected_runs
      if expected is not None:
        metrics["xprof_expected_runs"] = expected
        captured = metrics["xprof_captured_runs"]
        if captured > expected:
          # Truncation can only lose samples; more durations than traced
          # iterations means the parser split steps incorrectly.
          raise RuntimeError(
              f"XProf parsed {captured} durations for {expected} traced runs;"
              " expected at most one duration per traced iteration. Check"
              " MARKER / match_xprof_op_fallback event matching."
          )
        if captured == 0:
          print(
              f"Warning: No XProf durations parsed for {expected} traced"
              " runs (trace parse failure, no MARKER / fallback events"
              " matched, or multihost broadcast failure)."
          )
        elif captured < expected:
          print(
              f"Warning: XProf captured {captured}/{expected} traced runs;"
              " xprof_* metrics are based on a truncated trace (profiler"
              " event limit likely exceeded)."
          )

    metrics = self.derive_chip_metrics(metrics)

    metrics = self.apply_roofline_analysis(metrics)

    metrics["total_duration_s"] = time.perf_counter() - loop_start
    metrics["actual_runs"] = exec_result.actual_runs

    metadata = BenchmarkMetadata(
        benchmark_name=benchmark_name,
        test_name=f"{benchmark_name}_{int(time.time())}",
        start_time=start_ts,
        end_time=end_ts,
        params=dataclasses.asdict(self.config)
        if dataclasses.is_dataclass(self.config)
        else {},
        # TODO(simonleesyuan): Platform description is independent of benchmark
        # execution logic; move platform_info capture out of BaseBenchmark into
        # the runner orchestrator.
        platform_info=platform.get_platform_info(),
        hardware_spec=self.hardware_spec,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""),
        xprof_config=(
            self.xprof_config if self.xprof_config.xprof_timing else None
        ),
    )

    return BenchmarkResult(
        metadata=metadata,
        metrics=metrics,
        raw_times_ms=exec_result.raw_times_ms,
    )
