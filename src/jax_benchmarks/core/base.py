"""Base class for all JAX benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import datetime
import time
from typing import Any, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np


@dataclass
class BenchmarkMetadata:
  """Metadata for a single benchmark run."""

  benchmark_name: str
  test_name: str
  start_time: str
  end_time: str
  params: Dict[str, Any]
  device_info: Dict[str, Any]


@dataclass
class BenchmarkResult:
  """Consolidated result of a benchmark run."""

  metadata: BenchmarkMetadata
  metrics: Dict[str, Any]
  raw_times_ms: List[float]


class BaseBenchmark(ABC):
  """Abstract base class for microbenchmarks."""

  def __init__(self, mesh: Optional[Mesh] = None):
    self.mesh: Optional[Mesh] = mesh
    # Default settings that can be overridden by config
    self.warmup_tries = 10
    self.num_runs = 10
    self._jit_fn = None

  def _create_default_mesh(self) -> Mesh:
    """Create a default 1D mesh spanning all available devices."""
    devices = jax.devices()
    return Mesh(np.array(devices), axis_names=("device",))

  @abstractmethod
  def run_op(self, *args, **kwargs) -> Any:
    """The core operation intended for performance assessment."""
    pass

  def setup(self, **params):
    """Perform setup such as JIT compilation or buffer pre-allocation."""
    # Mesh creation is deferred to the run method.
    pass

  def generate_inputs(self, **params) -> Tuple:
    """Generate or retrieve inputs for the benchmark.

    Returns a tuple of arguments passed to run_op.
    """
    return ()

  def get_arithmetic_intensity(self, **params) -> float:
    """Calculate the arithmetic intensity (Flops / Bytes) for the operation.

    To be implemented by subclasses.
    """
    return 0.0

  def get_roofline_performance(
      self, peak_tflops: float, hbm_bw_data: Any, **params
  ) -> float:
    """Calculate the theoretical roofline performance ceiling (TFLOPS).

    hbm_bw_data can be:
    - A single float (peak GB/s)
    - A dict of {transfer_size_bytes: bandwidth_gb_s} for interpolation
    """
    intensity = self.get_arithmetic_intensity(**params)

    # Calculate total bytes moved for this op
    # Intensity = Flops / Bytes => Bytes = Flops / Intensity
    # But intensity might be 0 for memory-bound ops.
    # It's better to have a get_total_bytes method.
    total_bytes = self.get_total_bytes(**params)

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

  def get_total_bytes(self, **params) -> float:
    """Calculate total bytes moved to/from HBM."""
    return 0.0

  def calculate_metrics(
      self, times_ms: List[float], **params
  ) -> Dict[str, Any]:
    """Derive performance metrics from raw timing data."""
    if not times_ms:
      return {
          "avg_ms": 0.0,
          "p50_ms": 0.0,
          "p90_ms": 0.0,
          "std_ms": 0.0,
          "throughput": 0.0,
      }

    # Filter outliers using Interquartile Range (IQR) if we have enough data points
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
        "wall_clock_p50_ms": float(np.percentile(filtered_times, 50)),
        "wall_clock_p90_ms": float(np.percentile(filtered_times, 90)),
        "wall_clock_avg_ms": float(np.mean(filtered_times)),
        "wall_clock_std_ms": float(np.std(filtered_times)),
        "throughput": 0.0,  # To be overridden by subclasses
    }

  def apply_roofline_analysis(
      self, metrics: Dict[str, Any], **params
  ) -> Dict[str, Any]:
    """Apply roofline estimation to finalized metrics."""
    from .roofline import apply_roofline_analysis as ext_apply

    return ext_apply(self, metrics, **params)

  def _get_trace_metrics(self, **params) -> Optional[Dict[str, Any]]:
    """Extract Bottom-Up metrics using jax.experimental.roofline."""
    try:
      from jax.experimental import roofline
      # We need the inputs to trace the function
      inputs = self.generate_inputs(**params)
      # Trace the run_op function
      # Note: roofline() returns a wrapped function that returns (out_shape, RooflineResult)
      roofline_fn = roofline.roofline(self.run_op)
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
    except Exception as e:
      print(f"Warning: Failed to trace roofline: {e}")
      return None

  def run(self, **params) -> BenchmarkResult:
    """Standard orchestration flow for a benchmark."""
    self.warmup_tries = params.get("warmup_tries", self.warmup_tries)
    self.num_runs = params.get("num_runs", self.num_runs)
    self.min_duration_s = params.get("min_duration_s", 0.0)

    # 0. Initialize mesh if not provided
    if self.mesh is None:
      self.mesh = self._create_default_mesh()

    # 1. Setup & Initial Inputs
    self.setup(**params)
    inputs = self.generate_inputs(**params)

    # 2. Warmup & JIT Compilation
    # Run for at least warmup_tries OR a small duration if specified
    warmup_start = time.perf_counter()
    i = 0
    while i < self.warmup_tries or (
        time.perf_counter() - warmup_start < min(1.0, self.min_duration_s / 5)
    ):
      outputs = self.run_op(*inputs)
      jax.block_until_ready(outputs)
      i += 1

    import contextlib
    import os

    if params.get("xprof_timing", False):
      import jax.profiler as jax_profiler

      try:
        jax_profiler.stop_trace()
      except Exception:
        pass
      xprof_base_dir = params.get("xprof_dir", "/tmp/tensorboard")
      benchmark_name = self.__class__.__name__
      timestamp = int(time.time())

      cns_xprof_dir = os.path.join(
          xprof_base_dir, f"{benchmark_name}_{timestamp}"
      )
      local_xprof_dir = (
          f"/tmp/microbenchmarks_tmptrace/{benchmark_name}_{timestamp}"
      )

      if not (
          cns_xprof_dir.startswith("/cns/")
          or cns_xprof_dir.startswith("/bigstore/")
      ):
        local_xprof_dir = cns_xprof_dir

      params["xprof_dir_actual"] = local_xprof_dir
      params["xprof_dir_cns"] = cns_xprof_dir
      print(
          f"Collecting xprof trace locally to {local_xprof_dir} across runs..."
      )
      ctx = jax_profiler.trace(local_xprof_dir, create_perfetto_link=False)
    else:
      ctx = contextlib.nullcontext()

    start_ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    # 3. Measurement Loop
    raw_times = []
    loop_start = time.perf_counter()

    # Ensure we run at least num_runs AND meet the min_duration_s requirement
    with ctx:
      while len(raw_times) < self.num_runs or (
          time.perf_counter() - loop_start < self.min_duration_s
      ):
        t0 = time.perf_counter()
        outputs = self.run_op(*inputs)
        jax.block_until_ready(outputs)
        t1 = time.perf_counter()
        raw_times.append((t1 - t0) * 1000.0)

    if params.get("xprof_timing", False):
      print("Xprof trace collected.")

    end_ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    # 4. Finalize Results
    metrics = self.calculate_metrics(raw_times, **params)
    metrics = self.apply_roofline_analysis(metrics, **params)

    if params.get("xprof_timing", False):
      from .profiler import parse_xprof_results

      xprof_dir = params.get(
          "xprof_dir_actual", params.get("xprof_dir", "/tmp/tensorboard")
      )
      cns_dir = params.get("xprof_dir_cns", xprof_dir)
      metrics = parse_xprof_results(xprof_dir, cns_dir, metrics)

    metrics["total_duration_s"] = time.perf_counter() - loop_start
    metrics["actual_runs"] = len(raw_times)

    metadata = BenchmarkMetadata(
        benchmark_name=self.__class__.__name__,
        test_name=f"{self.__class__.__name__}_{int(time.time())}",
        start_time=start_ts,
        end_time=end_ts,
        params=params,
        device_info={
            "platform": str(jax.default_backend()),
            "device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
        },
    )

    return BenchmarkResult(
        metadata=metadata, metrics=metrics, raw_times_ms=raw_times
    )
