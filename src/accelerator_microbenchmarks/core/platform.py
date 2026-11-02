"""Live platform and hardware discovery for accelerator microbenchmarks.

WARNING: Calling functions in this module (such as `get_platform_info`)
triggers JAX backend initialization (`jax.distributed.initialize()` and
`jax.devices()`), which instantiates the C++ PJRT/libtpu client singleton.
Any custom runtime configurations (e.g., `os.environ["LIBTPU_INIT_ARGS"]`)
MUST be set BEFORE invoking functions in this module, otherwise the PJRT
client will lock in default flags and silently ignore late environment
mutations.
"""

import dataclasses
import importlib
import importlib.metadata
import platform as py_platform
from typing import Sequence

from accelerator_microbenchmarks.core import system
import jax


@dataclasses.dataclass(frozen=True)
class PlatformInfo:
  """Structured runtime topology, process ranks, and software versions."""

  # Canonical TPU version, as resolved by `TpuVersion.from_device_kind`.
  tpu_type: system.TpuVersion
  # ICI topology of a single slice; every slice of a multi-slice run has the
  # same topology.
  topology: str
  total_devices: int
  local_devices: int
  process_count: int
  process_index: int
  # Number of slices connected over the Data Center Network (DCN); 1 for
  # single-slice runs.
  slice_count: int
  # Slice that this process's local devices belong to; 0 for single-slice runs.
  slice_index: int
  python_version: str
  jax_version: str
  jaxlib_version: str
  libtpu_version: str


def _get_package_version(module_name: str, *fallback_pkg_names: str) -> str:
  """Resolves package version across google3 monorepo and OSS/nightly wheels."""
  try:
    mod = importlib.import_module(module_name)
    version = getattr(mod, "__version__", None)
    if version:
      return str(version)
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  for pkg in (module_name, *fallback_pkg_names):
    try:
      return str(importlib.metadata.version(pkg))
    except Exception:  # pylint: disable=broad-exception-caught
      # Package metadata may be missing or unreadable in monorepo environments.
      continue
  return "unknown"


def bounding_box_dims(coords: Sequence[Sequence[int]]) -> tuple[int, ...]:
  """Returns the per-axis extent (max - min + 1) of chip coordinates.

  Args:
    coords: Non-empty sequence of equal-length chip coordinates.

  Returns:
    The extent of the coordinates' bounding box on each axis.

  Raises:
    ValueError: If `coords` is empty or its coordinates differ in length.
  """
  if not coords:
    raise ValueError("no chip coordinates to bound")
  num_axes = len(coords[0])
  if any(len(c) != num_axes for c in coords):
    raise ValueError("chip coordinates have different numbers of axes")
  return tuple(
      max(c[axis] for c in coords) - min(c[axis] for c in coords) + 1
      for axis in range(num_axes)
  )


def _get_topology(
    devices: Sequence[jax.Device],
    topology_dimension: int = 3,
) -> str:
  """Computes slice topology from device coordinates bounding box."""
  if not devices:
    return "unknown"
  # Multi-slice runs repeat per-slice chip coordinates across slices; bound a
  # single slice's devices so the result is always the per-slice ICI topology.
  first_slice = get_device_slice_index(devices[0])
  slice_devices = [
      d for d in devices if get_device_slice_index(d) == first_slice
  ]
  coords_list = [getattr(d, "coords", None) for d in slice_devices]
  if any(c is None for c in coords_list):
    return "unknown"
  coords_list = [c for c in coords_list if c is not None]
  try:
    coord_dim = len(coords_list[0])
    if any(len(c) != coord_dim for c in coords_list):
      return "unknown"
    # Compute the extent along each coordinate axis from the bounding box
    # of device coordinates: span = max(c[i]) - min(c[i]) + 1.
    # Example: For a 2x4 slice where x in [0, 1] and y in [0, 3],
    # axis 0 span is 1 - 0 + 1 = 2, axis 1 span is 3 - 0 + 1 = 4 -> [2, 4].
    dims = list(bounding_box_dims(coords_list))
  except (TypeError, ValueError):
    return "unknown"

  # Truncate trailing trivial dimension(s) when the physical architecture
  # dimension (topology_dimension) is lower than the coordinate tuple length.
  # Example: TPU v6e (Trillium) is a 2D mesh (topology_dimension = 2), but JAX
  # exposes 3D coordinates (x, y, 0), producing raw dims [2, 4, 1].
  # Since topology_dimension = 2 and dims[2] == 1, we slice dims[:2] -> "2x4".
  if coord_dim > topology_dimension and all(
      d == 1 for d in dims[topology_dimension:]
  ):
    dims = dims[:topology_dimension]

  return "x".join(str(d) for d in dims)


def get_device_slice_index(device: jax.Device) -> int:
  """Returns the device's DCN slice index, defaulting to 0 on single-slice/CPU."""
  return getattr(device, "slice_index", 0)


def _get_slice_count(devices: Sequence[jax.Device]) -> int:
  """Returns the number of distinct DCN slices across `devices`."""
  if not devices:
    raise ValueError("no devices provided to compute slice_count")
  return len({get_device_slice_index(d) for d in devices})


def _get_local_slice_index(local_devices: Sequence[jax.Device]) -> int:
  """Returns the slice index shared by this process's local devices.

  Args:
    local_devices: The devices attached to this process (`jax.local_devices()`).

  Returns:
    The integer slice index of this process.

  Raises:
    RuntimeError: If `local_devices` is empty or spans more than one slice.
  """
  if not local_devices:
    raise RuntimeError("No local TPU devices found for current process.")
  local_slice_indices = {get_device_slice_index(d) for d in local_devices}
  if len(local_slice_indices) > 1:
    raise RuntimeError(
        "Local devices of one process span multiple slices:"
        f" {sorted(local_slice_indices)}."
    )
  return local_slice_indices.pop()


def get_platform_info() -> PlatformInfo:
  """Extracts structured TPU hardware topology, process ranks, and software versions."""
  try:
    jax.distributed.initialize()
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  try:
    backend = jax.default_backend()
    devices = jax.devices()
  except Exception as e:
    raise RuntimeError(
        "TPU runtime environment is not properly initialized or missing"
        " required packages (JAX / libtpu). Please install prerequisites:"
        " pip install jax jaxlib libtpu-nightly"
    ) from e

  if backend != "tpu":
    raise RuntimeError(
        "TPUMS requires TPU accelerator hardware, but detected JAX backend:"
        f" '{backend}'. "
        "Accelerator microbenchmarks cannot execute on CPU. "
        "Please run on a Cloud TPU VM (e.g., v6e Trillium or v7x Ironwood)."
    )

  if not devices:
    raise RuntimeError(
        "JAX backend is 'tpu', but no TPU devices were discovered."
    )

  raw_device_kind = getattr(devices[0], "device_kind", None)
  if not raw_device_kind:
    raise RuntimeError(
        f"TPU device {devices[0]} is missing a valid 'device_kind'. "
        "The PJRT TPU runtime or libtpu may not be initialized correctly."
    )

  tpu_version = system.TpuVersion.from_device_kind(str(raw_device_kind))
  hw_spec = system.get_hardware_spec(tpu_version)
  topology = _get_topology(
      devices, topology_dimension=hw_spec.topology_dimension
  )

  total_devices = int(jax.device_count())
  local_devices = int(jax.local_device_count())
  process_count = int(jax.process_count())
  process_index = int(jax.process_index())
  slice_count = _get_slice_count(devices)
  slice_index = _get_local_slice_index(jax.local_devices())

  return PlatformInfo(
      tpu_type=tpu_version,
      topology=topology,
      total_devices=total_devices,
      local_devices=local_devices,
      process_count=process_count,
      process_index=process_index,
      slice_count=slice_count,
      slice_index=slice_index,
      python_version=py_platform.python_version(),
      jax_version=_get_package_version("jax"),
      jaxlib_version=_get_package_version("jaxlib"),
      libtpu_version=_get_package_version("libtpu", "libtpu-nightly"),
  )
