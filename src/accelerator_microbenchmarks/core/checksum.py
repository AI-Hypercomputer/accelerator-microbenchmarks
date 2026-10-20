"""Bit-exact per-device checksums for the `gemm_chained` determinism benchmark.

After the timed loop, `make_checksum_fn` checksums each device's final X on that
device, and `gather_and_compare_checksums` all-gathers the checksums and
compares them within each host and across the slice.

Checksum (uint32, mod 2^32): with u_i the i-th element of X bitcast to uint,
s1 = sum(u_i), s2 = sum((i + 1) * u_i) and checksum = (s1 << 32) + s2. A
degenerate X (inf/NaN or all zeros) never passes.
"""

import collections
import logging
from typing import Any, Callable, NamedTuple, Optional, Sequence

from accelerator_microbenchmarks.core import utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

# Metric keys returned by `compare_gathered_checksums()`.
CHECKSUM = "checksum"
ALL_DEVICES_VALID_CHECKSUM = "all_devices_valid_checksum"
INTRA_HOST_CHECKSUM_MATCH = "intra_host_checksum_match"
GLOBAL_CHECKSUM_MATCH = "global_checksum_match"
DEVICE_CHECKSUMS = "device_checksums"
DIVERGING_DEVICES = "diverging_devices"

CHECKSUM_METRIC_KEYS = (
    CHECKSUM,
    ALL_DEVICES_VALID_CHECKSUM,
    INTRA_HOST_CHECKSUM_MATCH,
    GLOBAL_CHECKSUM_MATCH,
    DEVICE_CHECKSUMS,
    DIVERGING_DEVICES,
)

# Minimum number of devices to decide whether they all agree.
_MIN_DEVICES_FOR_MATCH = 2

# Device-side result row (uint32): (s1, s2, degenerate).
_NUM_CHECKSUM_WORDS = 2
_NUM_RESULT_WORDS = _NUM_CHECKSUM_WORDS + 1

# Gathered row (uint32): (device_id, reported, *result row). Other identity
# fields are looked up in the mesh, which spans all hosts.
_NUM_ROW_WORDS = 2 + _NUM_RESULT_WORDS
_ID, _REPORTED, _S1, _S2, _DEGENERATE = range(_NUM_ROW_WORDS)

# Bit width -> same-width uint to bitcast to. Wider types do not fit in uint32;
# sub-byte types cannot be bitcast element-wise.
_UINT_BY_BITS = {8: jnp.uint8, 16: jnp.uint16, 32: jnp.uint32}
SUPPORTED_BITS = tuple(_UINT_BY_BITS)


def is_checksummable_dtype(dtype: Any) -> bool:
  """Returns whether `checksum_words` supports elements of `dtype`."""
  try:
    return utils.get_dtype_bits(dtype) in _UINT_BY_BITS
  except ValueError:  # Not a numeric dtype.
    return False


def get_bitcast_uint_dtype(dtype: Any) -> Any:
  """Returns the same-width unsigned integer dtype used to bitcast `dtype`."""
  try:
    bits = utils.get_dtype_bits(dtype)
  except ValueError as e:
    raise ValueError(
        f"Checksum supports bit widths {SUPPORTED_BITS} only, got non-numeric"
        f" {jnp.dtype(dtype)}."
    ) from e
  if bits not in _UINT_BY_BITS:
    raise ValueError(
        f"Checksum supports bit widths {SUPPORTED_BITS} only, got"
        f" {jnp.dtype(dtype)} ({bits} bits)."
    )
  return _UINT_BY_BITS[bits]


def checksum_words(x: jax.Array) -> jax.Array:
  """Returns the uint32 `(s1, s2)` checksum words of `x` (jittable).

  The position weight in s2 catches swapped elements, which a plain sum misses.

  Raises:
    ValueError: If the element width is not in SUPPORTED_BITS or `x` has >= 2^32
      elements.
  """
  size = int(np.prod(x.shape))
  if size >= 2**32:
    raise ValueError(
        f"Checksum supports < 2^32 elements, got {size} (shape {x.shape})."
    )
  u = jax.lax.bitcast_convert_type(x, get_bitcast_uint_dtype(x.dtype))
  u = u.reshape(-1).astype(jnp.uint32)
  weights = jax.lax.iota(jnp.uint32, size) + jnp.uint32(1)
  s1 = jnp.sum(u, dtype=jnp.uint32)
  s2 = jnp.sum(weights * u, dtype=jnp.uint32)
  return jnp.stack([s1, s2])


def _is_degenerate(x: jax.Array) -> jax.Array:
  """Returns uint32 1 if `x` has inf/NaN or is all zeros, else 0."""
  has_nonfinite = jnp.any(jnp.logical_not(jnp.isfinite(x)))
  all_zero = jnp.all(x == 0)
  return jnp.logical_or(has_nonfinite, all_zero).astype(jnp.uint32)


def combine_checksum_words(s1: int, s2: int) -> int:
  """Returns the 64-bit checksum `(s1 << 32) + s2` of two uint32 words."""
  return ((int(s1) & 0xFFFFFFFF) << 32) | (int(s2) & 0xFFFFFFFF)


def format_checksum(value: Optional[int]) -> Optional[str]:
  """Formats a 64-bit checksum as a 16-digit lowercase hex string."""
  if value is None:
    return None
  return f"0x{value:016x}"


def make_checksum_fn(mesh: jax.sharding.Mesh):
  """Returns the jitted per-device checksum function for `mesh`.

  It maps X, whose block `d` along axis 0 is the final X of mesh device `d`, to
  uint32[num_devices, _NUM_RESULT_WORDS], whose row `d` is that device's
  `(s1, s2, degenerate)`, computed on that device (no collectives).
  """
  axes = tuple(mesh.axis_names)
  pspec = jax.sharding.PartitionSpec

  def per_device(x):
    row = jnp.concatenate([checksum_words(x), _is_degenerate(x)[None]])
    return row.reshape(1, _NUM_RESULT_WORDS)

  @jax.jit
  def checksum_fn(x):
    return jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=pspec(axes),
        out_specs=pspec(axes),
        check_vma=False,
    )(x)

  return checksum_fn


def get_device_identity(device: Any) -> dict[str, Any]:
  """Returns `device`'s identity; coords and core_on_chip are None off TPU."""
  coords = getattr(device, "coords", None)
  core_on_chip = getattr(device, "core_on_chip", None)
  return {
      "process_index": int(getattr(device, "process_index", 0)),
      "device_id": int(device.id),
      "coords": [int(c) for c in coords] if coords is not None else None,
      "core_on_chip": int(core_on_chip) if core_on_chip is not None else None,
  }


def _read_shard_words(shard: Any) -> np.ndarray:
  """Copies one device's result row to the host."""
  words = np.asarray(shard.data, dtype=np.uint32).reshape(-1)
  if words.size != _NUM_RESULT_WORDS:
    raise ValueError(
        f"Expected {_NUM_RESULT_WORDS} result words per device, got shape"
        f" {shard.data.shape} on {shard.device}."
    )
  return words


def read_local_rows(arr: Optional[jax.Array]) -> np.ndarray:
  """Returns this host's uint32 (local_devices, _NUM_ROW_WORDS) payload.

  One row per local device (sorted by id), so every host gathers the same shape.
  A device outside the mesh or whose readback fails is marked as not reported.

  Args:
    arr: Output of `make_checksum_fn`, or None if no checksums were computed.
  """
  devices = sorted(jax.local_devices(), key=lambda d: d.id)
  rows = np.zeros((len(devices), _NUM_ROW_WORDS), dtype=np.uint32)
  rows[:, _ID] = [d.id for d in devices]
  shards = {}
  if arr is not None:
    shards = {shard.device.id: shard for shard in arr.addressable_shards}
  for i, device in enumerate(devices):
    shard = shards.get(device.id)
    if shard is not None:
      try:
        rows[i, _S1:] = _read_shard_words(shard)
        rows[i, _REPORTED] = 1
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("Reading checksum of device %d failed: %s", device.id, e)
  return rows


def _invalid_result(
    device_checksums: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
  """Returns the result when the gather or comparison failed.

  As in every undecidable case, `diverging_devices` is `[]`; only
  `global_checksum_match` (None) says the result could not be decided.
  """
  return {
      CHECKSUM: None,
      ALL_DEVICES_VALID_CHECKSUM: False,
      INTRA_HOST_CHECKSUM_MATCH: None,
      GLOBAL_CHECKSUM_MATCH: None,
      DEVICE_CHECKSUMS: device_checksums or [],
      DIVERGING_DEVICES: [],
  }


class _Report(NamedTuple):
  """A reported device's result."""

  device: Any
  value: int  # 64-bit checksum.
  degenerate: bool


def _reports(rows: np.ndarray, devices: Sequence[Any]) -> list[_Report]:
  """Returns a `_Report` per reported row, looking devices up by id.

  Raises:
    ValueError: If a reported device is not in `devices`.
  """
  device_by_id = {d.id: d for d in devices}
  reports = []
  for row in rows.reshape(-1, _NUM_ROW_WORDS):
    if not row[_REPORTED]:
      continue
    device = device_by_id.get(int(row[_ID]))
    if device is None:
      raise ValueError(
          f"Device {row[_ID]} reported a checksum but is not a mesh device."
      )
    reports.append(
        _Report(
            device,
            combine_checksum_words(row[_S1], row[_S2]),
            bool(row[_DEGENERATE]),
        )
    )
  return reports


def _device_checksum_entry(report: _Report) -> dict[str, Any]:
  """Returns one device's entry in `device_checksums`."""
  identity = get_device_identity(report.device)
  return {
      "device_id": identity["device_id"],
      "coords": identity["coords"],
      "core_on_chip": identity["core_on_chip"],
      "checksum": format_checksum(report.value),
  }


def _match_flag(
    values: Sequence[int], *, complete: bool, degenerate: bool
) -> Optional[bool]:
  """Returns the three-state match flag over the reported `values` in scope.

  False if any two values differ, even with devices missing. True if all agree,
  `complete`, not `degenerate` and >= _MIN_DEVICES_FOR_MATCH. None otherwise.
  """
  if len(set(values)) > 1:
    return False
  if complete and not degenerate and len(values) >= _MIN_DEVICES_FOR_MATCH:
    return True
  return None


def _consensus(
    reports: Sequence[_Report], expected_num_devices: int
) -> Optional[int]:
  """Returns the checksum held by > half of the expected devices, or None.

  Missing devices count against it; a degenerate checksum is never the
  consensus.
  """
  counts = collections.Counter(r.value for r in reports)
  if not counts:
    return None
  value, count = counts.most_common(1)[0]
  if 2 * count <= expected_num_devices:
    return None
  if any(r.degenerate for r in reports if r.value == value):
    return None
  return value


def compare_gathered_checksums(
    rows: np.ndarray,
    process_index: int,
    *,
    mesh_devices: Sequence[Any],
) -> dict[str, Any]:
  """Returns the CHECKSUM_METRIC_KEYS metrics of the gathered rows (pure numpy).

  Mismatches are logged, not raised.

  Args:
    rows: uint32 rows of every host, of shape (..., _NUM_ROW_WORDS).
    process_index: Process index of the calling host.
    mesh_devices: Devices expected to report, on all hosts.
  """
  rows = np.asarray(rows, dtype=np.uint32)
  if rows.ndim < 2 or rows.shape[-1] != _NUM_ROW_WORDS:
    raise ValueError(
        f"rows must have shape (..., {_NUM_ROW_WORDS}), got {rows.shape}."
    )
  reports = _reports(rows, mesh_devices)
  local = sorted(
      (r for r in reports if r.device.process_index == process_index),
      key=lambda r: r.device.id,
  )
  num_local_mesh_devices = sum(
      d.process_index == process_index for d in mesh_devices
  )
  device_checksums = [_device_checksum_entry(r) for r in local]

  complete = len(reports) == len(mesh_devices)
  local_complete = len(local) == num_local_mesh_devices
  degenerate = [get_device_identity(r.device) for r in reports if r.degenerate]

  intra_match = _match_flag(
      [r.value for r in local],
      complete=local_complete,
      degenerate=any(r.degenerate for r in local),
  )
  global_match = _match_flag(
      [r.value for r in reports],
      complete=complete,
      degenerate=bool(degenerate),
  )

  # On a mismatch, every device that differs from the consensus (all if None).
  # Otherwise [], even if undecidable: only global_match says that.
  consensus = _consensus(reports, len(mesh_devices))
  diverging = [
      {
          **get_device_identity(r.device),
          "checksum": format_checksum(r.value),
          "consensus": format_checksum(consensus),
      }
      for r in reports
      if global_match is False and r.value != consensus
  ]
  diverging.sort(key=lambda d: (d["process_index"], d["device_id"]))

  if intra_match is False:
    logging.error(
        "CHECKSUM MISMATCH within host %d: local device checksums %s",
        process_index,
        device_checksums,
    )
  if global_match is False:
    logging.error(
        "CHECKSUM MISMATCH across slice (consensus=%s): diverging devices %s",
        format_checksum(consensus),
        diverging,
    )

  if not complete:
    logging.error(
        "Checksum check incomplete: %d of %d expected devices reported.",
        len(reports),
        len(mesh_devices),
    )
  if degenerate and consensus is not None:
    logging.error(
        "Degenerate result (inf/NaN or all zeros) on devices %s; likely a"
        " device fault, see diverging_devices.",
        degenerate,
    )
  elif degenerate:
    logging.error(
        "Degenerate chain result (inf/NaN or all zeros) on devices %s; the"
        " checksum check is invalid. Try a shorter chain_length or another"
        " dtype.",
        degenerate,
    )

  return {
      CHECKSUM: format_checksum(consensus),
      ALL_DEVICES_VALID_CHECKSUM: complete and not degenerate,
      INTRA_HOST_CHECKSUM_MATCH: intra_match,
      GLOBAL_CHECKSUM_MATCH: global_match,
      DEVICE_CHECKSUMS: device_checksums,
      DIVERGING_DEVICES: diverging,
  }


def gather_and_compare_checksums(
    x: Optional[jax.Array],
    checksum_fn: Callable[[jax.Array], jax.Array],
    mesh: jax.sharding.Mesh,
) -> dict[str, Any]:
  """Returns the CHECKSUM_METRIC_KEYS metrics of every device's final X.

  Call on every host after the timed loop. Never raises: a failing device is
  marked as not reported, so every host still joins the allgather.

  Args:
    x: Final X of every device (see `make_checksum_fn`), or None.
    checksum_fn: `make_checksum_fn(mesh)`.
    mesh: Mesh that computed `x`; its devices are the ones expected to report.
  """
  arr = None
  if x is not None:
    try:
      arr = checksum_fn(x)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.error("Computing device checksums failed: %s", e)
  local_rows = read_local_rows(arr)
  mesh_devices = list(mesh.devices.flat)
  if len(mesh_devices) < jax.device_count():
    logging.warning(
        "Checksum mesh covers %d of %d devices; devices outside the mesh"
        " are not checked.",
        len(mesh_devices),
        jax.device_count(),
    )
  try:
    # uint32, since uint64 is truncated without jax_enable_x64. Assumes every
    # host has the same number of local devices.
    gathered = multihost_utils.process_allgather(local_rows)
    return compare_gathered_checksums(
        gathered, jax.process_index(), mesh_devices=mesh_devices
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error("Checksum gather/compare failed: %s", e)
    # Every reported local row is a local device, so this lookup cannot fail.
    return _invalid_result([
        _device_checksum_entry(r)
        for r in _reports(local_rows, jax.local_devices())
    ])
