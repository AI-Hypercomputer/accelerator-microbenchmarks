"""Base class for all JAX benchmarks."""

import abc
import contextlib
import dataclasses
import datetime
import enum
import os
import time
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import profiler
from accelerator_microbenchmarks.core import roofline
from accelerator_microbenchmarks.core import system
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class XprofConfig:
  """Configuration for XProf trace collection and device timing analysis."""

  xprof_timing: bool = False
  xprof_dir: str = "/tmp/tensorboard"


@dataclasses.dataclass
class BaseBenchmarkParams:
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
  use_trace_roofline: bool = dataclasses.field(
      default=False,
      metadata={
          "help": (
              "Override analytical arithmetic intensity with bottom-up FLOPs"
              " and HBM bytes extracted via jax.experimental.roofline."
          )
      },
  )

  def _validate_bounds(self) -> None:
    """Checks every field against its min/max metadata constraints.

    Raises:
      ValueError: If any field violates its `min`/`max` bound.
    """
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      if value is None:
        continue

      # Only validate numeric fields (int, float) for min/max bounds.
      # Exclude bool since bool is a subclass of int in Python.
      if not isinstance(value, (int, float)) or isinstance(value, bool):
        continue

      minimum = field.metadata.get("min")
      if minimum is not None and value < minimum:
        raise ValueError(f"{field.name} must be >= {minimum}, got {value}")

      maximum = field.metadata.get("max")
      if maximum is not None and value > maximum:
        raise ValueError(f"{field.name} must be <= {maximum}, got {value}")

  def _coerce_enums(self) -> None:
    """Coerces every enum field in-place into a member of its Enum type.

    Raises:
      ValueError: If any field is not a valid member of its Enum type.
    """
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      if value is None:
        continue

      # Checks if the field is an enum by coercing raw values into their Enum
      # member. Only scalar enum fields are supported, not containers such as
      # list[Enum].
      if isinstance(field.type, type) and issubclass(field.type, enum.Enum):
        enum_type: type[enum.Enum] = field.type
      else:
        continue

      try:
        setattr(self, field.name, enum_type(value))
      except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid {field.name} '{value}'. Must be one of"
            f" {[member.value for member in enum_type]}."
        ) from exc

  def __post_init__(self):
    self._coerce_enums()
    self._validate_bounds()

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

  Config = BaseBenchmarkParams
  DEFAULT_LOCAL_DEVICE_ID: int = 0
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = ()
  # TODO(b/555967269): Redesign and decouple circular dependency between
  # BaseBenchmark and report formatters.
  REPORT_FORMATTERS: Optional[
      Sequence[Callable[[pd.DataFrame, type["BaseBenchmark"]], str]]
  ] = None
  derive_chip_bandwidth: bool = True
  roofline_mode: constants.RooflineMode = constants.RooflineMode.NONE

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

  def reset_data(self, *inputs, **kwargs) -> tuple[jax.Array, ...]:
    """Reset data for the next run."""
    return inputs

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

  @abc.abstractmethod
  def get_arithmetic_intensity(self) -> float:
    """Calculate the arithmetic intensity (Flops / Bytes) for the operation.

    To be implemented by subclasses.

    Returns:
      The arithmetic intensity as a float.
    """
    pass

  @abc.abstractmethod
  def get_total_bytes(self) -> float:
    """Calculate total bytes moved to/from HBM."""
    pass

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    """Derive static workload metadata, host wall-clock latency statistics, and domain throughput metrics."""
    metrics = self.get_workload_metadata()
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
    return metrics

  def get_workload_metadata(self) -> dict[str, Any]:
    """Return static, timing-invariant workload metadata (e.g., total_flops, intensity)."""
    return {
        "intensity": self.get_arithmetic_intensity(),
    }

  def calculate_latency_stats(
      self, times_ms: list[float], prefix: constants.TimingDomain
  ) -> dict[str, float]:
    """Derive statistical latency metrics (avg, p50, p90, std) from raw timing data."""
    if not times_ms:
      return {
          f"{prefix}_avg_ms": 0.0,
          f"{prefix}_p50_ms": 0.0,
          f"{prefix}_p90_ms": 0.0,
          f"{prefix}_std_ms": 0.0,
      }

    # Filter outliers using Interquartile Range (IQR) if we have enough data
    # points
    if len(times_ms) > 3:
      q1 = np.percentile(times_ms, 25)
      q3 = np.percentile(times_ms, 75)
      iqr = q3 - q1
      lower_bound = q1 - 1.5 * iqr
      upper_bound = q3 + 1.5 * iqr
      filtered_times = [t for t in times_ms if lower_bound <= t <= upper_bound]

      # Fallback if filtering removes everything
      if not filtered_times:
        filtered_times = times_ms
    else:
      filtered_times = times_ms

    return {
        f"{prefix}_p50_ms": float(np.percentile(filtered_times, 50)),
        f"{prefix}_p90_ms": float(np.percentile(filtered_times, 90)),
        f"{prefix}_avg_ms": float(np.mean(filtered_times)),
        f"{prefix}_std_ms": float(np.std(filtered_times)),
    }

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

  def get_trace_metrics(self) -> Optional[dict[str, Any]]:
    """Extract Bottom-Up metrics using jax.experimental.roofline."""
    try:
      import jax.experimental.roofline as jax_roofline  # pylint: disable=g-import-not-at-top
      # We need the inputs to trace the function
      inputs = self.generate_inputs()
      # Trace the run_op function
      # Note: roofline() returns a wrapped function that returns
      # (out_shape, RooflineResult)
      roofline_fn = jax_roofline.roofline(self.run_op)
      _, result = roofline_fn(*inputs)

      return {
          "flops": result.flops,
          "hbm_bytes": result.hbm_bytes,
      }
    except (ImportError, AttributeError) as e:
      print(
          "Warning: jax.experimental.roofline or dependencies not"
          f" available: {e}"
      )
      return None
    except (TypeError, ValueError, RuntimeError) as e:
      print(f"Warning: Failed to trace roofline: {e}")
      return None

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

    measured_device = self.get_device_to_measure()
    is_owner = jax.process_index() == measured_device.process_index

    all_devs_str = [f"{d.id}(p{d.process_index})" for d in jax.devices()]
    local_devs_str = [
        f"{d.id}(p{d.process_index})" for d in jax.local_devices()
    ]
    print(f"[Host {jax.process_index()}] All devices: {all_devs_str}")
    print(f"[Host {jax.process_index()}] Local devices: {local_devs_str}")
    print(
        f"[Host {jax.process_index()}] Measured device:"
        f" {measured_device.id}(p{measured_device.process_index}),"
        f" owner_process={measured_device.process_index}, is_owner={is_owner}"
    )

    # 1. Parse local XProf trace on relevant hosts
    should_parse = not self.requires_multihost_sync or is_owner
    local_avg, local_p50, local_p90, local_std = 0.0, 0.0, 0.0, 0.0

    if should_parse:
      try:
        local_device_id = (
            None
            if self.xprof_target_host_cpu
            else measured_device.local_hardware_id
        )
        durations = profiler.parse_xprof_durations(
            xprof_dir,
            self.match_xprof_op_fallback,
            local_device_id=local_device_id,
        )
        if durations:
          print(
              f"Using XProf device timings ({len(durations)} runs)"
              " to calculate derived performance metrics."
          )
          xprof_stats = self.calculate_latency_stats(
              durations, prefix=constants.TimingDomain.XPROF
          )
          local_avg = xprof_stats[f"{constants.TimingDomain.XPROF}_avg_ms"]
          local_p50 = xprof_stats[f"{constants.TimingDomain.XPROF}_p50_ms"]
          local_p90 = xprof_stats[f"{constants.TimingDomain.XPROF}_p90_ms"]
          local_std = xprof_stats[f"{constants.TimingDomain.XPROF}_std_ms"]
        else:
          print("Warning: No XProf device timings found locally.")
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error parsing local XProf trace: {e}")
    else:
      print(
          f"Note: Measured device {measured_device.id} is on remote Host"
          f" {measured_device.process_index} (current Host is"
          f" {jax.process_index()}). Skipping local XProf parsing."
      )

    # 2. Optionally broadcast stats across hosts for asymmetric targets (D2D)
    synced_avg, synced_p50, synced_p90, synced_std = (
        local_avg,
        local_p50,
        local_p90,
        local_std,
    )
    if self.requires_multihost_sync:
      try:
        stats = jnp.array(
            [local_avg, local_p50, local_p90, local_std], dtype=jnp.float32
        )
        synced_stats = multihost_utils.broadcast_one_to_all(
            stats, is_source=is_owner
        )
        synced_avg = float(synced_stats[0])
        synced_p50 = float(synced_stats[1])
        synced_p90 = float(synced_stats[2])
        synced_std = float(synced_stats[3])
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Warning: multihost broadcast failed ({e}); using local values.")

    # 3. Apply XProf timings or fallback to None for unmeasured durations
    if synced_p50 > 0:
      metrics.update({
          f"{constants.TimingDomain.XPROF}_avg_ms": synced_avg,
          f"{constants.TimingDomain.XPROF}_p50_ms": synced_p50,
          f"{constants.TimingDomain.XPROF}_p90_ms": synced_p90,
          f"{constants.TimingDomain.XPROF}_std_ms": synced_std,
      })
      metrics.update(
          self.calculate_throughput_metrics(
              latency_ms=synced_p50, prefix=constants.TimingDomain.XPROF
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
      for wall_clock_key, xprof_key in (
          (
              f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_device_gb_s",
              f"{constants.TimingDomain.XPROF}_bandwidth_per_device_gb_s",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_chip_gb_s",
              f"{constants.TimingDomain.XPROF}_bandwidth_per_chip_gb_s",
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
    """Standard orchestration flow for a benchmark."""

    # 0. Initialize mesh if not provided
    if self.mesh is None:
      self.mesh = self._create_default_mesh()

    # 1. Setup & Initial Inputs
    self.setup()
    inputs = self.generate_inputs()

    # 2. Warmup & JIT Compilation
    # Run for at least warmup_tries OR a small duration if specified
    warmup_start = time.perf_counter()
    i = 0
    while i < self.config.warmup_tries or (
        time.perf_counter() - warmup_start
        < min(1.0, self.config.min_duration_s / 5)
    ):
      inputs = self.reset_data(*inputs)
      outputs = self.run_op(*inputs)
      jax.block_until_ready(outputs)
      i += 1

    if self.xprof_config.xprof_timing:
      try:
        jax.profiler.stop_trace()
      except RuntimeError:
        pass
      xprof_base_dir = self.xprof_config.xprof_dir
      benchmark_name = self.__class__.__name__
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
      ctx = jax.profiler.trace(local_xprof_dir, create_perfetto_link=False)
    else:
      ctx = contextlib.nullcontext()

    start_ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    # 3. Measurement Loop
    raw_times = []
    actual_runs = 0
    loop_start = time.perf_counter()

    # Ensure we run at least num_runs AND meet the min_duration_s requirement
    with ctx:
      while actual_runs < self.config.num_runs or (
          time.perf_counter() - loop_start < self.config.min_duration_s
      ):
        inputs = self.reset_data(*inputs)
        t0 = time.perf_counter()
        outputs = self.run_op(*inputs)
        jax.block_until_ready(outputs)
        t1 = time.perf_counter()
        actual_runs += 1
        if actual_runs < 1000:
          raw_times.append((t1 - t0) * 1000.0)

    if self.xprof_config.xprof_timing:
      print("Xprof trace collected.")

    end_ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    # 4. Finalize Results
    # Calculate host-side metrics
    metrics = self.calculate_metrics(raw_times)

    if self.xprof_config.xprof_timing:
      metrics = self._apply_xprof_timing_and_sync(metrics)

    metrics = self.derive_chip_metrics(metrics)

    metrics = self.apply_roofline_analysis(metrics)

    metrics["total_duration_s"] = time.perf_counter() - loop_start
    metrics["actual_runs"] = actual_runs

    metadata = BenchmarkMetadata(
        benchmark_name=self.__class__.__name__,
        test_name=f"{self.__class__.__name__}_{int(time.time())}",
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
        metadata=metadata, metrics=metrics, raw_times_ms=raw_times
    )
