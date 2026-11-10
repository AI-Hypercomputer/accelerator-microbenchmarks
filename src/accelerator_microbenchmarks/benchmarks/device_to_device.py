"""Device-to-Device (D2D) transfer performance microbenchmark."""

import dataclasses
import logging
from typing import Any, Callable, Optional, Sequence

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import dtypes
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.core import utils
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp


class TransferDirection(constants.ParamEnum):
  """Transfer direction for device-to-device benchmarks."""

  UNI = "uni"
  BI = "bi"


def classify_transfer_link(
    src: jax.Device, dst: jax.Device
) -> constants.TransferLink:
  """Returns the physical link that carries a `src -> dst` transfer.

  Devices on the same chip share physical chip coordinates and talk over the
  on-package die-to-die link; devices on different chips of the same slice
  talk over ICI.

  Args:
    src: Source device.
    dst: Destination device.

  Returns:
    `TransferLink.DIE_TO_DIE` if both devices are on the same chip, otherwise
    `TransferLink.ICI`.

  Raises:
    ValueError: If the devices have no physical chip coordinates (non-TPU) or
      are on different slices (the transfer then crosses DCN).
  """
  src_coords = getattr(src, "coords", None)
  dst_coords = getattr(dst, "coords", None)
  if src_coords is None or dst_coords is None:
    raise ValueError("devices have no physical chip coordinates (non-TPU)")
  if platform.get_device_slice_index(src) != platform.get_device_slice_index(
      dst
  ):
    raise ValueError("src and dst are on different slices (DCN)")
  if tuple(src_coords) == tuple(dst_coords):
    return constants.TransferLink.DIE_TO_DIE
  return constants.TransferLink.ICI


def device_to_device_peak_bandwidth_gb_s(
    link: constants.TransferLink,
    direction: TransferDirection,
    hardware_spec: system.HardwareSpec,
) -> float:
  """Returns the theoretical peak bandwidth of one D2D transfer (GB/s).

  A device-to-device transfer is bounded by a single physical link per
  direction. For ICI this holds independent of the hop count: a single
  transfer follows one route, so measured cross-chip transfers saturate one
  link whether the chips are neighbors or several hops apart. Links are
  full-duplex and `DeviceToDeviceBenchmark.get_total_bytes()` counts the
  bytes of both directions for `BI`, so the bidirectional peak is twice the
  per-direction peak.

  Scope: the result is the ceiling for the single measured `src -> dst`
  transfer, which runs alone (one device pair at a time) and so has the whole
  link to itself. It is not a per-device share of a chip resource: ICI links
  belong to the chip and are shared by all of its devices, so ICI peaks must
  not be summed across the devices of a chip.

  Args:
    link: The link carrying the transfer.
    direction: Unidirectional or bidirectional transfer.
    hardware_spec: Hardware spec holding the link bandwidths.

  Returns:
    The peak bandwidth in decimal GB/s, in the same units as
    `*_bandwidth_per_device_gb_s`.

  Raises:
    ValueError: If `hardware_spec` has no bandwidth for `link`.
  """
  if link == constants.TransferLink.DIE_TO_DIE:
    if hardware_spec.die_to_die is None:
      raise ValueError(f"no die-to-die spec for {hardware_spec.name}")
    per_direction_gb_s = hardware_spec.die_to_die.bandwidth_per_direction_gb_s
  else:
    if hardware_spec.ici is None:
      raise ValueError(f"no ICI spec for {hardware_spec.name}")
    per_direction_gb_s = hardware_spec.ici.link_bw_per_direction_gb_s
  num_directions = 2.0 if direction == TransferDirection.BI else 1.0
  return per_direction_gb_s * num_directions


@dataclasses.dataclass
class DeviceToDeviceParams(base.SingleDtypeBenchmarkParams):
  """YAML / CLI configuration specification (NO src/dst fields)."""

  # generate_inputs() sizes the payload with itemsize bytes per element.
  ALLOW_SUB_BYTE_DTYPES = False

  data_size_mib: int = dataclasses.field(
      default=1024,
      metadata={"min": 1, "help": "Transfer payload size in Mebibytes (MiB)."},
  )
  direction: TransferDirection = dataclasses.field(
      default=TransferDirection.UNI,
      metadata={
          "help": (
              "Transfer direction mode ('uni' for unidirectional, 'bi' for"
              " bidirectional)."
          )
      },
  )
  seed: int = dataclasses.field(
      default=0,
      metadata={"min": 0, "help": "Random seed for tensor initialization."},
  )

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

  src_device_index: int = dataclasses.field(
      default=0,
      metadata={"min": 0, "help": "Source local accelerator device index."},
  )
  dst_device_index: int = dataclasses.field(
      default=1,
      metadata={
          "min": 0,
          "help": "Destination local accelerator device index.",
      },
  )


@registry.benchmark_registry.register("device_to_device")
class DeviceToDeviceBenchmark(base.BaseBenchmark[DeviceToDeviceTestCaseParams]):
  """Benchmarks Device-to-Device (D2D) transfer bandwidth using ppermute."""

  Config = DeviceToDeviceParams
  derive_chip_bandwidth: bool = False
  roofline_mode: constants.RooflineMode = constants.RooflineMode.NETWORK_LINK
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("direction", report.format_str),
      ("src_device_index", report.format_str),
      ("dst_device_index", report.format_str),
      ("data_size_mib", report.format_str),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_device_gb_s", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_device_gb_s", report.format_2f),
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
      xprof_config: Optional[base.XprofConfig] = None,
  ):
    super().__init__(
        config=config,
        hardware_spec=hardware_spec,
        mesh=mesh,
        xprof_config=xprof_config,
    )
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
    dtype = dtypes.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    num_elements = size_bytes // itemsize
    shape = (num_devices, num_elements)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec("x", None)
    )

    key = jax.random.PRNGKey(self.config.seed)

    generate_data = jax.jit(
        lambda k: utils.random_normal_array(k, shape, dtype),
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

  def get_peak_link_bandwidth(
      self,
  ) -> tuple[float, constants.TransferLink] | None:
    """Returns the D2D peak and its bounding link, or None (with a warning).

    The peak is not modeled (and the roofline columns are omitted) for:
      - Non-TPU devices without chip coordinates.
      - Pairs on different slices (DCN).
      - Hardware specs without a bandwidth for the bounding link.
    """
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    # Indices refer to mesh positions, which need not follow chip order, so
    # classify the actual devices rather than the indices.
    src = self.mesh.devices.flat[self.config.src_device_index]
    dst = self.mesh.devices.flat[self.config.dst_device_index]
    try:
      link = classify_transfer_link(src, dst)
      peak_gb_s = device_to_device_peak_bandwidth_gb_s(
          link, self.config.direction, self.hardware_spec
      )
    except ValueError as e:
      logging.warning(
          "Skipping device_to_device *_roofline_efficiency_pct: %s.", e
      )
      return None
    return peak_gb_s, link
