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
from typing import Any, Sequence

from accelerator_microbenchmarks.core import system
import jax


@dataclasses.dataclass(frozen=True)
class PlatformInfo:
  """Structured runtime topology, process ranks, and software versions."""

  tpu_type: system.TpuVersion
  topology: str
  total_devices: int
  local_devices: int
  process_count: int
  process_index: int
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
    except importlib.metadata.PackageNotFoundError:
      continue
    except Exception:  # pylint: disable=broad-exception-caught
      continue
  return "unknown"


def _get_topology(
    devices: Sequence[jax.Device],
    topology_dimension: int = 3,
) -> str:
  """Computes slice topology from device coordinates bounding box."""
  if not devices:
    return "unknown"
  coords_list = [getattr(d, "coords", None) for d in devices]
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
    dims = [
        max(c[i] for c in coords_list) - min(c[i] for c in coords_list) + 1
        for i in range(coord_dim)
    ]

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
  except Exception:  # pylint: disable=broad-exception-caught
    return "unknown"


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
        f"TPUMS requires TPU accelerator hardware, but detected JAX backend: '{backend}'. "
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

  tpu_version = system.TpuVersion.from_str(str(raw_device_kind))
  hw_spec = system.get_hardware_spec(tpu_version)
  topology = _get_topology(
      devices, topology_dimension=hw_spec.topology_dimension
  )

  total_devices = int(jax.device_count())
  local_devices = int(jax.local_device_count())
  process_count = int(jax.process_count())
  process_index = int(jax.process_index())

  return PlatformInfo(
      tpu_type=tpu_version,
      topology=topology,
      total_devices=total_devices,
      local_devices=local_devices,
      process_count=process_count,
      process_index=process_index,
      python_version=py_platform.python_version(),
      jax_version=_get_package_version("jax"),
      jaxlib_version=_get_package_version("jaxlib"),
      libtpu_version=_get_package_version("libtpu", "libtpu-nightly"),
  )

