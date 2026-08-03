"""Host-to-Device and Device-to-Host transfer performance benchmarks."""

import dataclasses
from typing import Any

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
import jax
import numpy as np


@dataclasses.dataclass
class HostDeviceParams(base.BaseBenchmarkParams):
  data_size_mib: int = 64

  @property
  def data_size_bytes(self) -> int:
    return self.data_size_mib * 1024 * 1024


@registry.benchmark_registry.register("host_to_device")
class HostToDeviceBenchmark(base.BaseBenchmark[HostDeviceParams]):
  Config = HostDeviceParams
  """Benchmarks Host-to-Device transfer bandwidth."""

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
      return jax.device_put(args[0])

  def get_total_bytes(self) -> float:
    return float(self.config.data_size_bytes)

  def get_arithmetic_intensity(self) -> float:
    # 100% memory-bound operation
    return 0.0

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    metrics = super().calculate_metrics(times_ms)
    avg_latency_s = metrics["avg_ms"] / 1000.0
    if avg_latency_s == 0:
      bandwidth_gb_s = float("inf")
    else:
      # Bandwidth in GiB/s
      bandwidth_gb_s = (self.config.data_size_mib / 1024.0) / avg_latency_s
    metrics["bandwidth_gb_s"] = bandwidth_gb_s
    metrics["total_bytes_mib"] = float(self.config.data_size_mib)
    return metrics


@registry.benchmark_registry.register("device_to_host")
class DeviceToHostBenchmark(base.BaseBenchmark[HostDeviceParams]):
  Config = HostDeviceParams
  """Benchmarks Device-to-Host transfer bandwidth."""

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

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    metrics = super().calculate_metrics(times_ms)
    avg_latency_s = metrics["avg_ms"] / 1000.0
    if avg_latency_s == 0:
      bandwidth_gb_s = float("inf")
    else:
      bandwidth_gb_s = (self.config.data_size_mib / 1024.0) / avg_latency_s
    metrics["bandwidth_gb_s"] = bandwidth_gb_s
    metrics["total_bytes_mib"] = float(self.config.data_size_mib)
    return metrics
