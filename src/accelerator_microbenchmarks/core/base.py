"""Base class for all JAX benchmarks."""

import abc
import contextlib
import dataclasses
import datetime
import os
import time
from typing import Any, Generic, Optional, TypeVar

from accelerator_microbenchmarks.core import profiler
from accelerator_microbenchmarks.core import roofline
from accelerator_microbenchmarks.core import system
import jax
import numpy as np


@dataclasses.dataclass
class BaseBenchmarkParams:
  warmup_tries: int = 10
  num_runs: int = 10
  min_duration_s: float = 0.0
  xprof_timing: bool = False
  xprof_dir: str = "/tmp/tensorboard"
  system: str = ""
  use_trace_roofline: bool = False
  hardware_stats: dict[str, Any] = dataclasses.field(default_factory=dict)
  dtype: str = "bfloat16"


@dataclasses.dataclass
class BenchmarkMetadata:
  """Metadata for a single benchmark run."""

  benchmark_name: str
  test_name: str
  start_time: str
  end_time: str
  params: dict[str, Any]
  device_info: dict[str, Any]


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

  def __init__(self, config: TConfig, mesh: Optional[jax.sharding.Mesh] = None):
    if config is None:
      raise ValueError("A configuration object must be explicitly provided.")
    self.config = config
    self.mesh: Optional[jax.sharding.Mesh] = mesh
    self._jit_fn = None
    self._xprof_dir_actual: str = self.config.xprof_dir
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

    Args:
      **params: Additional parameters to customize input generation.

    Returns:
      A tuple of arguments passed to run_op.
    """
    pass

  @abc.abstractmethod
  def get_arithmetic_intensity(self) -> float:
    """Calculate the arithmetic intensity (Flops / Bytes) for the operation.

    To be implemented by subclasses.

    Args:
      **params: Additional parameters to customize intensity calculation.

    Returns:
      The arithmetic intensity as a float.
    """
    pass

  def get_roofline_performance(
      self, peak_tflops: float, hbm_bw_data: Any
  ) -> float:
    """Calculate the theoretical roofline performance ceiling (TFLOPS).

    Args:
      peak_tflops: The peak theoretical TFLOPS of the device.
      hbm_bw_data: HBM bandwidth data, either a float (peak GB/s) or a dict of
        {transfer_size_bytes: bandwidth_gb_s} for interpolation.
      **params: Additional parameters for intensity and byte calculation.

    Returns:
      The theoretical roofline performance ceiling in TFLOPS.
    """
    intensity = self.get_arithmetic_intensity()

    # Calculate total bytes moved for this op
    # Intensity = Flops / Bytes => Bytes = Flops / Intensity
    # But intensity might be 0 for memory-bound ops.
    # It's better to have a get_total_bytes method.
    total_bytes = self.get_total_bytes()
    bw = 0.0

    if isinstance(hbm_bw_data, (int, float)):
      bw = hbm_bw_data
    elif isinstance(hbm_bw_data, list):
      sorted_data = sorted(hbm_bw_data, key=lambda x: x[0])
      if not sorted_data:
        bw = 0.0
      elif total_bytes <= sorted_data[0][0]:
        bw = sorted_data[0][1]
      elif total_bytes >= sorted_data[-1][0]:
        bw = sorted_data[-1][1]
      else:
        bw = 0.0
        for i in range(len(sorted_data) - 1):
          s0, bw0 = sorted_data[i]
          s1, bw1 = sorted_data[i + 1]
          if s0 <= total_bytes <= s1:
            bw = bw0 + (bw1 - bw0) * (total_bytes - s0) / (s1 - s0)
            break
    elif isinstance(hbm_bw_data, dict):
      # Simple linear interpolation or nearest neighbor
      # Sorting by transfer size
      sorted_sizes = sorted(hbm_bw_data.keys())
      if total_bytes <= sorted_sizes[0]:
        bw = hbm_bw_data[sorted_sizes[0]]
      elif total_bytes >= sorted_sizes[-1]:
        bw = hbm_bw_data[sorted_sizes[-1]]
      else:
        # Find the bracket
        for i in range(len(sorted_sizes) - 1):
          s0, s1 = sorted_sizes[i], sorted_sizes[i + 1]
          if s0 <= total_bytes <= s1:
            bw0, bw1 = hbm_bw_data[s0], hbm_bw_data[s1]
            # Interpolate
            bw = bw0 + (bw1 - bw0) * (total_bytes - s0) / (s1 - s0)
            break
    else:
      bw = 0.0

    # Roofline = min(Peak Math, BW * Intensity)
    return min(peak_tflops, (intensity * bw) / 1000.0)

  @abc.abstractmethod
  def get_total_bytes(self) -> float:
    """Calculate total bytes moved to/from HBM."""
    pass

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    """Derive performance metrics from raw timing data."""
    if not times_ms:
      return {
          "avg_ms": 0.0,
          "p50_ms": 0.0,
          "p90_ms": 0.0,
          "std_ms": 0.0,
          "throughput": 0.0,
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
        "p50_ms": float(np.percentile(filtered_times, 50)),
        "p90_ms": float(np.percentile(filtered_times, 90)),
        "avg_ms": float(np.mean(filtered_times)),
        "std_ms": float(np.std(filtered_times)),
        "throughput": 0.0,  # To be overridden by subclasses
    }

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
      # We need the inputs to trace the function
      inputs = self.generate_inputs()
      # Trace the run_op function
      # Note: roofline() returns a wrapped function that returns
      # (out_shape, RooflineResult)
      roofline_fn = jax.experimental.roofline.roofline(self.run_op)
      _, result = roofline_fn(*inputs)

      return {
          "flops": result.flops,
          "hbm_bytes": result.hbm_bytes,
      }
    except ImportError as e:
      print(
          "Warning: jax.experimental.roofline or dependencies (absl-py) not"
          f" available: {e}"
      )
      return None
    except (TypeError, ValueError, RuntimeError) as e:
      print(f"Warning: Failed to trace roofline: {e}")
      return None

  def get_compute_dtype(self) -> str:
    """Return the primary data type used for compute math, to determine peak TFLOPS."""
    return self.config.dtype

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

    if self.config.xprof_timing:
      try:
        jax.profiler.stop_trace()
      except RuntimeError:
        pass
      xprof_base_dir = self.config.xprof_dir
      benchmark_name = self.__class__.__name__
      timestamp = int(time.time())

      run_id = self.get_run_identifier()
      dir_suffix = f"_{run_id}" if run_id else ""

      cns_xprof_dir = os.path.join(
          xprof_base_dir, f"{benchmark_name}{dir_suffix}_{timestamp}"
      )
      local_xprof_dir = (
          f"/tmp/microbenchmarks_tmptrace/{benchmark_name}{dir_suffix}_{timestamp}"
      )

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

    if self.config.xprof_timing:
      print("Xprof trace collected.")

    end_ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    # 4. Finalize Results
    # Calculate host-side metrics
    metrics = self.calculate_metrics(raw_times)

    xprof_url = None
    xprof_durations = None

    if self.config.xprof_timing:
      xprof_dir = self._xprof_dir_actual
      cns_dir = self._xprof_dir_cns
      xprof_url = profiler.upload_xprof_trace(xprof_dir, cns_dir)  # pyrefly: ignore[bad-argument-type]
      try:
        xprof_durations = profiler.parse_xprof_durations(
            # pyrefly: ignore[bad-argument-type]
            xprof_dir, self.match_xprof_op_fallback
        )
        if xprof_durations:
          print(
              f"Using XProf device timings ({len(xprof_durations)} runs) to"
              " calculate derived performance metrics."
          )
        else:
          print(
              "Warning: No XProf device timings found, falling back to host"
              " timings for derived metrics."
          )
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error parsing XProf trace: {e}. Falling back to host timings.")

    if self.config.xprof_timing:
      if xprof_url:
        metrics["xprof_url"] = xprof_url
      if xprof_durations:
        metrics["xprof_avg_ms"] = float(np.mean(xprof_durations))
        metrics["xprof_p50_ms"] = float(np.percentile(xprof_durations, 50))
        metrics["xprof_p90_ms"] = float(np.percentile(xprof_durations, 90))

        # Recalculate derived metrics (like bandwidth) using XProf timings
        device_metrics = self.calculate_metrics(xprof_durations)
        for key, value in device_metrics.items():
          if key not in ["p50_ms", "avg_ms", "p90_ms", "std_ms"]:
            metrics[key] = value

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
        device_info=system.get_runtime_device_info(),
    )

    return BenchmarkResult(
        metadata=metadata, metrics=metrics, raw_times_ms=raw_times
    )
