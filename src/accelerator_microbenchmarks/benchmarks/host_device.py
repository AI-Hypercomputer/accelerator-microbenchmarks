"""Host-to-Device and Device-to-Host transfer performance benchmarks."""

import dataclasses
from typing import Any, Callable, Sequence

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
import jax
import numpy as np


@dataclasses.dataclass
class HostDeviceParams(base.SingleDtypeBenchmarkParams):
  data_size_mib: int = dataclasses.field(
      default=64,
      metadata={"min": 1, "help": "Transfer payload size in Mebibytes (MiB)."},
  )

  @property
  def data_size_bytes(self) -> int:
    return self.data_size_mib * 1024 * 1024


@registry.benchmark_registry.register("host_to_device")
class HostToDeviceBenchmark(base.BaseBenchmark[HostDeviceParams]):
  """Benchmarks Host-to-Device transfer bandwidth."""

  Config = HostDeviceParams
  derive_chip_bandwidth: bool = False
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("data_size_mib", report.format_str),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_device_gb_s", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_device_gb_s", report.format_2f),
  )

  @property
  def xprof_target_host_cpu(self) -> bool:
    return True

  def setup(self):
    pass

  def get_run_identifier(self) -> str:
    return f"size_{self.config.data_size_mib}mib"

  def generate_inputs(self) -> tuple[np.ndarray, ...]:
    num_elements = self.config.data_size_bytes // np.dtype(np.float32).itemsize
    column = 128
    host_data = np.random.normal(size=(num_elements // column, column)).astype(
        np.float32
    )
    return (host_data,)

  def run_op(self, *args, **kwargs) -> jax.Array:
    # args[0] is host_data
    with jax.profiler.TraceAnnotation(constants.MARKER):
      # jax.device_put is asynchronous and returns immediately upon dispatch.
      # Calling block_until_ready() inside the TraceAnnotation context manager
      # forces the host thread to wait until the DMA transfer completes before
      # the profiler marker exits. Without this, the MARKER annotation only
      # measures the microsecond dispatch time rather than the transfer,
      # causing xprof_timing to calculate an artificially inflated bandwidth.
      arr = jax.device_put(args[0])
      return arr.block_until_ready()

  def get_total_bytes(self) -> float:
    return float(self.config.data_size_bytes)

  def get_arithmetic_intensity(self) -> float:
    # 100% memory-bound operation
    return 0.0

  def get_workload_metadata(self) -> dict[str, Any]:
    return {
        "total_bytes_mib": float(self.config.data_size_mib),
        "intensity": self.get_arithmetic_intensity(),
    }

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    total_bytes = self.get_total_bytes()
    latency_s = latency_ms / 1000.0
    if latency_s == 0:
      bandwidth_gb_s = float("inf")
    else:
      bandwidth_gb_s = total_bytes / (latency_s * 1e9)
    return {
        f"{prefix}_bandwidth_per_device_gb_s": bandwidth_gb_s,
    }


@registry.benchmark_registry.register("device_to_host")
class DeviceToHostBenchmark(base.BaseBenchmark[HostDeviceParams]):
  """Benchmarks Device-to-Host transfer bandwidth."""

  Config = HostDeviceParams
  derive_chip_bandwidth: bool = False
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("data_size_mib", report.format_str),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_device_gb_s", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_device_gb_s", report.format_2f),
  )

  host_data: np.ndarray
  device_array: jax.Array

  def setup(self):
    num_elements = self.config.data_size_bytes // np.dtype(np.float32).itemsize
    column = 128
    self.host_data = np.random.normal(size=(num_elements // column, column)).astype(
        np.float32
    )

  def reset_data(self, arr: jax.Array, **kwargs) -> tuple[jax.Array, ...]:
    if arr is not None:
      arr.delete()
    return self.generate_inputs()

  def get_run_identifier(self) -> str:
    return f"size_{self.config.data_size_mib}mib"

  def generate_inputs(self) -> tuple[Any, ...]:
    device_array = jax.device_put(self.host_data)
    device_array.block_until_ready()
    return (device_array,)

  def run_op(self, arr: jax.Array, **kwargs) -> np.ndarray:
    with jax.profiler.TraceAnnotation(constants.MARKER):
      return jax.device_get(arr)

  def get_total_bytes(self) -> float:
    return float(self.config.data_size_bytes)

  def get_arithmetic_intensity(self) -> float:
    # 100% memory-bound operation
    return 0.0

  def get_workload_metadata(self) -> dict[str, Any]:
    return {
        "total_bytes_mib": float(self.config.data_size_mib),
        "intensity": self.get_arithmetic_intensity(),
    }

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    total_bytes = self.get_total_bytes()
    latency_s = latency_ms / 1000.0
    if latency_s == 0:
      bandwidth_gb_s = float("inf")
    else:
      bandwidth_gb_s = total_bytes / (latency_s * 1e9)
    return {
        f"{prefix}_bandwidth_per_device_gb_s": bandwidth_gb_s,
    }
