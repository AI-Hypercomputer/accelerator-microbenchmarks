"""Device-to-Device (D2D) transfer performance microbenchmark."""

import dataclasses
import enum
from typing import Any, Callable, Optional, Sequence

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.core import utils
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp


class TransferDirection(str, enum.Enum):
  UNI = "uni"
  BI = "bi"


@dataclasses.dataclass
class DeviceToDeviceParams(base.BaseBenchmarkParams):
  """YAML / CLI configuration specification (NO src/dst fields)."""

  data_size_mib: int = 1024
  direction: TransferDirection = TransferDirection.UNI
  seed: int = 0

  @property
  def data_size_bytes(self) -> int:
    return self.data_size_mib * 1024 * 1024

  def expand_test_cases(self) -> Sequence["DeviceToDeviceTestCaseParams"]:
    """1-to-many sweep: generates a DeviceToDeviceTestCaseParams for each (src, dst) pair."""
    devices = list(range(len(jax.devices())))
    base_kwargs = dataclasses.asdict(self)
    cases = []
    for src in devices:
      for dst in devices:
        if src == dst:
          continue
        cases.append(
            DeviceToDeviceTestCaseParams(
                **base_kwargs,
                src_device_index=src,
                dst_device_index=dst,
            )
        )
    return cases


@dataclasses.dataclass
class DeviceToDeviceTestCaseParams(DeviceToDeviceParams):
  """Concrete execution config for a single (src, dst) pair."""

  src_device_index: int = 0
  dst_device_index: int = 1


@registry.benchmark_registry.register("device_to_device")
class DeviceToDeviceBenchmark(base.BaseBenchmark[DeviceToDeviceTestCaseParams]):
  """Benchmarks Device-to-Device (D2D) transfer bandwidth using ppermute."""

  Config = DeviceToDeviceParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("direction", report.format_str),
      ("src_device_index", report.format_str),
      ("dst_device_index", report.format_str),
      ("data_size_mib", report.format_str),
      ("bandwidth_gb_s", report.format_2f),
      ("p50_ms", report.format_4f),
      ("xprof_p50_ms", report.format_4f),
  )
  REPORT_FORMATTERS = (
      report.format_standard_table,
      report.format_device_matrix,
  )

  def __init__(
      self,
      config: DeviceToDeviceTestCaseParams,
      hardware_spec: system.HardwareSpec,
      mesh: Optional[jax.sharding.Mesh] = None,
  ):
    super().__init__(config=config, hardware_spec=hardware_spec, mesh=mesh)
    self._jit_fn = None

  def get_device_to_measure(self) -> jax.Device:
    """Returns the destination JAX Device to observe in D2D transfers."""
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    dst_index = self.config.dst_device_index
    if dst_index >= self.mesh.devices.size:
      raise ValueError(
          f"dst_device_index ({dst_index}) exceeds mesh size"
          f" ({self.mesh.devices.size})."
      )
    return self.mesh.devices.flat[dst_index]

  @property
  def requires_multihost_sync(self) -> bool:
    """Device-to-Device transfers measure a specific target chip on one host."""
    return True

  def get_run_identifier(self) -> str:
    src = self.config.src_device_index
    dst = self.config.dst_device_index
    direction = str(self.config.direction)
    data_size_mib = self.config.data_size_mib
    return f"d2d_{src}_to_{dst}_{direction}_{data_size_mib}mib"

  def setup(self):
    num_devices = len(jax.devices())
    devices = mesh_utils.create_device_mesh((num_devices,))
    self.mesh = jax.sharding.Mesh(devices, ("x",))

    mapping_str = ", ".join(
        f"Index {idx} -> Device {dev.id} (Host {dev.process_index})"
        for idx, dev in enumerate(self.mesh.devices.flat)
    )
    print(f"[D2D Mesh Mapping] {mapping_str}")

    src = self.config.src_device_index
    dst = self.config.dst_device_index
    src_dev = self.mesh.devices.flat[src]
    dst_dev = self.mesh.devices.flat[dst]
    print(
        f"[D2D Transfer] src_index={src} -> Device {src_dev.id} (Host"
        f" {src_dev.process_index}) | dst_index={dst} ->"
        f" Device {dst_dev.id} (Host {dst_dev.process_index})"
    )
    is_bi = self.config.direction == TransferDirection.BI

    perm = [(src, dst), (dst, src)] if is_bi else [(src, dst)]

    def ppermute_kernel(val):
      with jax.named_scope(constants.MARKER):
        return jax.lax.ppermute(val, axis_name="x", perm=perm)

    sharded_kernel = jax.shard_map(
        ppermute_kernel,
        mesh=self.mesh,
        in_specs=jax.sharding.PartitionSpec("x", None),
        out_specs=jax.sharding.PartitionSpec("x", None),
        check_vma=False,
    )
    self._jit_fn = jax.jit(sharded_kernel)

  def generate_inputs(self) -> tuple[jnp.ndarray, ...]:
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")

    num_devices = len(jax.devices())
    size_bytes = int(self.config.data_size_bytes)
    dtype = utils.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    num_elements = size_bytes // itemsize
    shape = (num_devices, num_elements)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("x", None)
    )

    key = jax.random.PRNGKey(self.config.seed)

    generate_data = jax.jit(
        lambda k: jax.random.normal(k, shape, dtype=dtype),
        out_shardings=sharding,
    )
    data = generate_data(key)
    return (data,)

  def run_op(self, data) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(data)

  def get_total_bytes(self) -> float:
    size_bytes = self.config.data_size_bytes
    is_bi = self.config.direction == TransferDirection.BI
    data_factor = 2.0 if is_bi else 1.0
    return float(size_bytes * data_factor)

  def get_arithmetic_intensity(self) -> float:
    return 0.0

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    metrics = super().calculate_metrics(times_ms)
    total_bytes = self.get_total_bytes()
    avg_latency_s = metrics["avg_ms"] / 1000.0

    if avg_latency_s == 0:
      bandwidth_gb_s = float("inf")
    else:
      bandwidth_gb_s = total_bytes / (avg_latency_s * 1e9)

    metrics["bandwidth_gb_s"] = bandwidth_gb_s
    metrics["total_bytes_mib"] = total_bytes / (1024 * 1024)
    metrics["src_device_index"] = self.config.src_device_index
    metrics["dst_device_index"] = self.config.dst_device_index
    metrics["direction"] = self.config.direction
    return metrics
