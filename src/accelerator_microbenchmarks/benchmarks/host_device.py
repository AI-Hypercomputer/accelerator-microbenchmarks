"""Host-to-Device and Device-to-Host transfer performance benchmarks."""

import dataclasses
import gc
import numbers
from typing import Any, Callable, Optional, Sequence

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp
import numpy as np

from tpu_sync.api.jax import kv_cache_manager

# Minor (lane) dimension of the 2D/3D transfer buffers.
_COLUMNS = 128
# Mesh axis name used to shard the transfer buffer across local devices.
_DEVICE_AXIS = "device"
# `local_device` sentinel selecting every local device.
ALL_LOCAL_DEVICES = -1


class TransferBackend(constants.ParamEnum):
  """Host<->device transfer implementation under test.

  Attributes:
    JAX: Standard JAX transfers (`jax.device_put` / `jax.device_get`).
    TPU_SYNC: `tpu_sync` `KVCacheManager` DMA transfers between NUMA-local
      pinned host staging buffers and device HBM.
  """

  JAX = "jax"
  TPU_SYNC = "tpu_sync"


@dataclasses.dataclass
class HostDeviceParams(base.SingleDtypeBenchmarkParams):
  """Parameters for Host-to-Device and Device-to-Host benchmarks."""

  data_size_mib: int = dataclasses.field(
      default=64,
      metadata={
          "min": 1,
          "help": "Transfer payload size per device in Mebibytes (MiB).",
      },
  )
  transfer_backend: TransferBackend = dataclasses.field(
      default=TransferBackend.JAX,
      metadata={
          "help": (
              "Transfer implementation: 'jax' (jax.device_put/device_get) or"
              " 'tpu_sync' (tpu_sync KVCacheManager DMA over pinned host"
              " memory)."
          )
      },
  )
  local_device: int = dataclasses.field(
      default=ALL_LOCAL_DEVICES,
      metadata={
          "min": ALL_LOCAL_DEVICES,
          "help": (
              "Index into jax.local_devices() of the single device to"
              " transfer to/from; -1 transfers to/from all local devices"
              " concurrently."
          ),
      },
  )

  def __post_init__(self):
    if self.transfer_backend is None:
      raise ValueError(
          "Invalid transfer_backend 'None'. Must be one of"
          f" {[member.value for member in TransferBackend]}."
      )
    for name, minimum in (
        ("data_size_mib", 1),
        ("local_device", ALL_LOCAL_DEVICES),
    ):
      value = getattr(self, name)
      # bool is an Integral subclass, so reject it explicitly.
      if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{name} must be an integer >= {minimum}, got {value!r}"
        )
      # Normalize integer-like values (e.g. numpy ints) so bounds apply.
      setattr(self, name, int(value))
    # Coerces enums and checks min/max bounds.
    super().__post_init__()

  @property
  def data_size_bytes(self) -> int:
    return self.data_size_mib * 1024 * 1024


class _HostDeviceBenchmark(base.BaseBenchmark[HostDeviceParams]):
  """Shared buffer layout, metrics, and tpu_sync lifecycle for H2D/D2H."""

  Config = HostDeviceParams  # pylint: disable=invalid-name
  derive_chip_bandwidth: bool = False
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("transfer_backend", report.format_str),
      ("local_device", report.format_str),
      ("num_devices", report.format_str),
      ("data_size_mib", report.format_str),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_device_gb_s", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_device_gb_s", report.format_2f),
  )

  kv_manager: Optional[kv_cache_manager.KVCacheManager] = None
  device_array: Optional[jax.Array] = None
  sharding: Optional[jax.sharding.NamedSharding] = None

  @property
  def xprof_target_host_cpu(self) -> bool:
    # Host<->device DMA transfers are annotated on the /host:CPU plane.
    return True

  @property
  def use_tpu_sync(self) -> bool:
    return self.config.transfer_backend == TransferBackend.TPU_SYNC

  def get_transfer_devices(self) -> list[jax.Device]:
    """Returns the local devices that participate in the transfer."""
    local_devices = jax.local_devices()
    local_device = self.config.local_device
    if local_device == ALL_LOCAL_DEVICES:
      return list(local_devices)
    if local_device >= len(local_devices):
      raise ValueError(
          f"local_device ({local_device}) exceeds the number of local devices"
          f" ({len(local_devices)})."
      )
    return [local_devices[local_device]]

  @property
  def num_transfer_devices(self) -> int:
    return len(self.get_transfer_devices())

  @property
  def rows_per_device(self) -> int:
    itemsize = jnp.dtype(utils.parse_dtype(self.config.dtype)).itemsize
    return self.config.data_size_bytes // itemsize // _COLUMNS

  def get_buffer_shape(self) -> tuple[int, ...]:
    """Returns the global shape of the transfer buffer.

    For `tpu_sync`, `KVCacheManager` treats axis 0 of each device shard as the
    block axis (`block_bytes = shard_bytes / shard_shape[0]`), so the buffer
    gets a leading singleton axis and is sharded along axis 1: each device
    shard is `(1, rows_per_device, 128)` and a single block holds the full
    per-device payload.
    """
    rows = self.num_transfer_devices * self.rows_per_device
    if self.use_tpu_sync:
      return (1, rows, _COLUMNS)
    return (rows, _COLUMNS)

  def get_partition_spec(self) -> jax.sharding.PartitionSpec:
    if self.use_tpu_sync:
      return jax.sharding.PartitionSpec(None, _DEVICE_AXIS, None)
    return jax.sharding.PartitionSpec(_DEVICE_AXIS, None)

  def get_device_to_measure(self) -> jax.Device:
    return self.get_transfer_devices()[0]

  def get_run_identifier(self) -> str:
    local_device = self.config.local_device
    device_str = "all" if local_device == ALL_LOCAL_DEVICES else local_device
    return (
        f"{self.config.transfer_backend}_dev{device_str}"
        f"_size_{self.config.data_size_mib}mib"
    )

  def setup(self):
    # Release resources from any previous setup() so a re-run never holds two
    # pinned host staging pools at once.
    self.teardown()
    self.mesh = jax.sharding.Mesh(
        np.array(self.get_transfer_devices()), axis_names=(_DEVICE_AXIS,)
    )
    self.sharding = jax.sharding.NamedSharding(
        self.mesh, self.get_partition_spec()
    )
    if self.use_tpu_sync:
      self.device_array = self._make_device_array(self._make_host_shard())
      # One block per device shard covers the full per-device payload, so the
      # host staging pool is allocated, NUMA first-touched, and pinned once.
      self.kv_manager = kv_cache_manager.KVCacheManager(
          kv_caches=[self.device_array],
          local_control_port=0,
          host_blocks_to_allocate=1,
          unsafe_skip_buffer_lock=True,
      )

  def teardown(self) -> None:
    """Releases the pinned host staging pool and device buffer."""
    # Drop the KVCacheManager before the device buffer it references.
    self.kv_manager = None
    if self.device_array is not None:
      try:
        if not self.device_array.is_deleted():
          self.device_array.delete()
      except Exception as e:  # pylint: disable=broad-exception-caught
        # run() calls teardown() in a finally block; never mask the run's
        # original exception with a cleanup failure.
        print(f"Warning: failed to delete transfer buffer: {e}")
      finally:
        self.device_array = None
    gc.collect()

  def _make_host_shard(self) -> np.ndarray:
    """Returns a random host payload with the shape of one device shard."""
    dtype = utils.parse_dtype(self.config.dtype)
    shard_shape = (
        (1, self.rows_per_device, _COLUMNS)
        if self.use_tpu_sync
        else (self.rows_per_device, _COLUMNS)
    )
    return utils.random_bits_array(shard_shape, dtype)

  def _make_device_array(self, host_shard: np.ndarray) -> jax.Array:
    """Builds the sharded device buffer by placing `host_shard` on each device.

    Reusing a single host shard for every device keeps peak host memory at the
    per-device payload size instead of the full global buffer size.

    Args:
      host_shard: Host payload with the shape of one device shard.

    Returns:
      A ready, sharded device array of shape `get_buffer_shape()`.
    """
    if self.sharding is None:
      raise ValueError("setup() must be called before creating device arrays.")
    shards = [
        jax.device_put(host_shard, device)
        for device in self.get_transfer_devices()
    ]
    device_array = jax.make_array_from_single_device_arrays(
        self.get_buffer_shape(), self.sharding, shards
    )
    return device_array.block_until_ready()

  def _require_kv_manager(self) -> kv_cache_manager.KVCacheManager:
    if self.kv_manager is None:
      raise ValueError(
          "KVCacheManager is not initialized; call setup() before run_op()."
      )
    return self.kv_manager

  def get_total_bytes(self) -> float:
    """Returns the total bytes moved across all participating devices."""
    return float(self.config.data_size_bytes * self.num_transfer_devices)

  def get_workload_metadata(self) -> dict[str, Any]:
    return {"num_devices": self.num_transfer_devices}

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    bytes_per_device = self.get_total_bytes() / self.num_transfer_devices
    latency_s = latency_ms / 1000.0
    if latency_s == 0:
      bandwidth_gb_s = float("inf")
    else:
      bandwidth_gb_s = bytes_per_device / (latency_s * 1e9)
    return {
        f"{prefix}_bandwidth_per_device_gb_s": bandwidth_gb_s,
    }


@registry.benchmark_registry.register("host_to_device")
class HostToDeviceBenchmark(_HostDeviceBenchmark):
  """Benchmarks Host-to-Device transfer bandwidth."""

  def setup(self):
    super().setup()
    if self.use_tpu_sync:
      # Populate the pinned host staging buffer with the random payload.
      self._require_kv_manager().d2h([0], [0], [1]).Await()

  def generate_inputs(self) -> tuple[Any, ...]:
    if self.use_tpu_sync:
      return (self.device_array,)
    dtype = utils.parse_dtype(self.config.dtype)
    host_data = utils.random_bits_array(self.get_buffer_shape(), dtype)
    return (host_data,)

  def run_op(self, *args, **kwargs) -> Optional[jax.Array]:
    if self.use_tpu_sync:
      kv_manager = self._require_kv_manager()
      with jax.profiler.TraceAnnotation(constants.MARKER):
        # Await() blocks until the DMA into every device shard completes.
        kv_manager.h2d([0], [0], [1]).Await()
      return self.device_array

    # args[0] is host_data
    with jax.profiler.TraceAnnotation(constants.MARKER):
      # jax.device_put is asynchronous and returns immediately upon dispatch.
      # Calling block_until_ready() inside the TraceAnnotation context manager
      # forces the host thread to wait until the DMA transfer completes before
      # the profiler marker exits. Without this, the MARKER annotation only
      # measures the microsecond dispatch time rather than the transfer,
      # causing xprof_timing to calculate an artificially inflated bandwidth.
      arr = jax.device_put(args[0], self.sharding)
      return arr.block_until_ready()


@registry.benchmark_registry.register("device_to_host")
class DeviceToHostBenchmark(_HostDeviceBenchmark):
  """Benchmarks Device-to-Host transfer bandwidth."""

  host_data: Optional[np.ndarray] = None

  def setup(self):
    super().setup()
    if not self.use_tpu_sync:
      # Host payload reused to rebuild the source device array every run.
      self.host_data = self._make_host_shard()

  def teardown(self) -> None:
    self.host_data = None
    super().teardown()

  def reset_data(self, *inputs, **kwargs) -> tuple[Any, ...]:
    if self.use_tpu_sync:
      # KVCacheManager.d2h DMAs into its own host staging buffer and never
      # populates the jax.Array host-value cache, so the source is reusable.
      return inputs
    # jax.device_get caches the host copy on the array; transfer a fresh array
    # every run so each measurement performs a real D2H copy.
    arr = inputs[0] if inputs else None
    if arr is not None:
      arr.delete()
    return self.generate_inputs()

  def generate_inputs(self) -> tuple[Any, ...]:
    if self.use_tpu_sync:
      return (self.device_array,)
    if self.host_data is None:
      raise ValueError("setup() must be called before generate_inputs().")
    return (self._make_device_array(self.host_data),)

  def run_op(self, *args, **kwargs) -> Optional[np.ndarray]:
    if self.use_tpu_sync:
      kv_manager = self._require_kv_manager()
      with jax.profiler.TraceAnnotation(constants.MARKER):
        # Await() blocks until the DMA from every device shard completes.
        kv_manager.d2h([0], [0], [1]).Await()
      return None

    with jax.profiler.TraceAnnotation(constants.MARKER):
      return jax.device_get(args[0])
