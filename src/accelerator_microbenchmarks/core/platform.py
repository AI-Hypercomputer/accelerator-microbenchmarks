"""Live platform and hardware discovery for accelerator microbenchmarks.

WARNING: Calling functions in this module (such as `get_platform_description`)
triggers JAX backend initialization (`jax.distributed.initialize()` and
`jax.devices()`), which instantiates the C++ PJRT/libtpu client singleton.
Any custom runtime configurations (e.g., `os.environ["LIBTPU_INIT_ARGS"]`)
MUST be set BEFORE invoking functions in this module, otherwise the PJRT
client will lock in default flags and silently ignore late environment
mutations.
"""

import importlib
import importlib.metadata
import platform as py_platform
from typing import Any, Sequence

from accelerator_microbenchmarks.core import system
import jax


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
    device_kind: str | None = None,
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
    if not device_kind and devices:
      kind = getattr(devices[0], "device_kind", "")
      device_kind = str(kind) if kind else ""

    device_kind_normalized = device_kind.strip().lower() if device_kind else ""
    sys_config = system.SYSTEMS.get(device_kind_normalized)
    target_dim = sys_config.topology_dimension if sys_config else None
    # Truncate trailing trivial dimension(s) when the physical architecture
    # dimension (target_dim) is lower than the coordinate tuple length.
    # Example: TPU v6e (Trillium) is a 2D mesh (target_dim = 2), but JAX
    # exposes 3D coordinates (x, y, 0), producing raw dims [2, 4, 1].
    # Since target_dim = 2 and dims[2] == 1, we slice dims[:2] -> "2x4".
    if (
        target_dim
        and coord_dim > target_dim
        and all(d == 1 for d in dims[target_dim:])
    ):
      dims = dims[:target_dim]

    return "x".join(str(d) for d in dims)
  except Exception:  # pylint: disable=broad-exception-caught
    return "unknown"


def get_platform_description() -> dict[str, Any]:
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
    tpu_type = "none"
    topology = "none"
  else:
    tpu_type = str(getattr(devices[0], "device_kind", "unknown"))
    topology = _get_topology(devices, device_kind=tpu_type)

  total_devices = int(jax.device_count())
  local_devices = int(jax.local_device_count())
  process_count = int(jax.process_count())
  process_index = int(jax.process_index())

  return {
      "tpu_type": tpu_type,
      "topology": topology,
      "total_devices": total_devices,
      "local_devices": local_devices,
      "process_count": process_count,
      "process_index": process_index,
      "python_version": py_platform.python_version(),
      "jax_version": _get_package_version("jax"),
      "jaxlib_version": _get_package_version("jaxlib"),
      "libtpu_version": _get_package_version("libtpu", "libtpu-nightly"),
  }
