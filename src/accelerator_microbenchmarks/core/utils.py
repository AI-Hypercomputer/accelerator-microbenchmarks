"""Utility functions for accelerator microbenchmarks."""

from typing import Any
import jax
import jax.numpy as jnp
import numpy as np


def parse_dtype(dtype_str: str) -> jnp.dtype:
  """Parses a string into a jax.numpy dtype.

  Args:
    dtype_str: The string representation of the dtype (e.g., 'bfloat16').

  Returns:
    The corresponding jax.numpy dtype.

  Raises:
    ValueError: If the dtype string does not correspond to a valid jnp dtype.
  """
  if not hasattr(jnp, dtype_str):
    raise ValueError(
        f"Invalid dtype string: '{dtype_str}'. Could not find"
        f" jax.numpy.{dtype_str}."
    )
  return getattr(jnp, dtype_str)


def device_coordinate(device: Any) -> tuple[int, int, int, int]:
  """Extract hardware coordinate (x, y, z, core) for sorting devices."""
  coords = tuple(int(value) for value in getattr(device, "coords", ()))
  core = getattr(device, "core_on_chip", None)
  if len(coords) != 3 or core is None:
    process_idx = getattr(device, "process_index", 0)
    device_id = getattr(device, "id", 0)
    return (int(process_idx), int(device_id), 0, 0)
  return (*coords, int(core))


def devices_by_slice() -> list[list[Any]]:
  """Groups devices by slice_index (falling back to process_index)."""
  groups: dict[int, list[Any]] = {}
  for device in jax.devices():
    slice_index = getattr(device, "slice_index", None)
    if slice_index is None:
      slice_index = getattr(device, "process_index", 0)
    groups.setdefault(int(slice_index), []).append(device)
  for devices in groups.values():
    devices.sort(key=device_coordinate)
  return [groups[index] for index in sorted(groups)]


def build_multislice_mesh(
    num_slices: int,
    participants_per_slice: int,
    axis_names: tuple[str, ...] = ("dcn", "ici"),
) -> tuple[jax.sharding.Mesh, list[list[Any]]]:
  """Constructs a Mesh structured across multiple slices (DCN x ICI)."""
  groups = devices_by_slice()
  if len(groups) < num_slices:
    # If simulated on CPU/single-slice test environments, chunk devices
    all_devs = list(jax.devices())
    total_needed = num_slices * participants_per_slice
    if len(all_devs) >= total_needed:
      selected = [
          all_devs[
              i * participants_per_slice : (i + 1) * participants_per_slice
          ]
          for i in range(num_slices)
      ]
    else:
      raise RuntimeError(
          f"Need {num_slices} slices, but only found {len(groups)} slice groups"
          f" and {len(all_devs)} total devices."
      )
  else:
    selected = [
        devices[:participants_per_slice] for devices in groups[:num_slices]
    ]
    for slice_index, devices in enumerate(selected):
      if len(devices) != participants_per_slice:
        raise RuntimeError(
            f"Slice {slice_index} has {len(devices)} devices; "
            f"need {participants_per_slice}"
        )
  mesh = jax.sharding.Mesh(np.asarray(selected, dtype=object), axis_names)
  return mesh, selected
