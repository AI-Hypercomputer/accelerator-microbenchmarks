"""Bit-exact per-device checksums for the `gemm_chained` determinism benchmark.

After each timed run, `checksum_local_rows` checksums each device's final X on
that device and reads back this host's rows. After the timed loop,
`gather_and_compare_checksums` all-gathers the rows of every run and compares
them within each host and across the slice.

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

# Row (uint32) of one local device in one run: (device_id, reported, *result
# row). Other identity fields are looked up in the mesh, which spans all hosts.
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


def get_device_identity(device: jax.Device) -> dict[str, Any]:
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


def checksum_local_rows(
    x: jax.Array, checksum_fn: Callable[[jax.Array], jax.Array]
) -> np.ndarray:
  """Checksums one run's final X on every device; reads back this host's rows.

  Call after each timed run. Blocks until the rows are on the host, so no
  checksum work is left to run in the next run's timed window. Never raises:
  if the checksum fails, every local device is marked as not reported.

  Args:
    x: Final X of every device (see `make_checksum_fn`).
    checksum_fn: `make_checksum_fn(mesh)`.

  Returns:
    This run's `read_local_rows` payload.
  """
  arr = None
  try:
    arr = checksum_fn(x)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error("Computing device checksums failed: %s", e)
  return read_local_rows(arr)


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


class _RunChecksum(NamedTuple):
  """A device's checksum in one timed run."""

  run: int
  checksum: int  # 64-bit checksum.
  degenerate: bool


class _DeviceRuns(NamedTuple):
  """A device's checksums over the timed runs."""

  device: jax.Device
  num_runs: int  # Timed runs, the same on every host.
  # One per run whose checksum was read back.
  run_checksums: list[_RunChecksum]

  @property
  def reported(self) -> bool:
    return bool(self.run_checksums)

  @property
  def complete(self) -> bool:
    """Whether every timed run has a checksum."""
    return self.num_runs > 0 and len(self.run_checksums) == self.num_runs

  @property
  def degenerate(self) -> bool:
    return any(rc.degenerate for rc in self.run_checksums)

  @property
  def majority_checksum(self) -> Optional[int]:
    """The checksum of more than half of the timed runs, or None.

    Runs without a checksum count against it.
    """
    counts = collections.Counter(rc.checksum for rc in self.run_checksums)
    if not counts:
      return None
    checksum, count = counts.most_common(1)[0]
    return checksum if 2 * count > self.num_runs else None


def _device_runs(
    gathered: np.ndarray, devices: Sequence[jax.Device]
) -> list[_DeviceRuns]:
  """Returns the `_DeviceRuns` of each of `devices`, looking devices up by id.

  Args:
    gathered: uint32 rows of every host and run, of shape (hosts, runs,
      local_devices, _NUM_ROW_WORDS).
    devices: Devices expected to report.

  Raises:
    ValueError: If a reported device is not in `devices`.
  """
  device_by_id = {d.id: d for d in devices}
  run_checksums = {d.id: [] for d in devices}
  for local_runs in gathered:
    for run, rows in enumerate(local_runs):
      for row in rows:
        if not row[_REPORTED]:
          continue
        device = device_by_id.get(int(row[_ID]))
        if device is None:
          raise ValueError(
              f"Device {row[_ID]} reported a checksum but is not a mesh device."
          )
        run_checksums[device.id].append(
            _RunChecksum(
                run,
                combine_checksum_words(row[_S1], row[_S2]),
                bool(row[_DEGENERATE]),
            )
        )
  return [
      _DeviceRuns(d, gathered.shape[1], run_checksums[d.id]) for d in devices
  ]


def _device_checksum_entry(runs: _DeviceRuns) -> dict[str, Any]:
  """Returns one device's entry in `device_checksums`."""
  identity = get_device_identity(runs.device)
  return {
      "device_id": identity["device_id"],
      "coords": identity["coords"],
      "core_on_chip": identity["core_on_chip"],
      "checksum": format_checksum(runs.majority_checksum),
  }


def _match_flag(scope: Sequence[_DeviceRuns]) -> Optional[bool]:
  """Returns the three-state match flag over every checksum in `scope`.

  False if any two checksums differ, across devices or timed runs, even with
  checksums missing. True if all agree, every device is complete, none is
  degenerate and there are >= _MIN_DEVICES_FOR_MATCH devices. None otherwise.
  """
  checksums = set()
  for runs in scope:
    checksums.update(rc.checksum for rc in runs.run_checksums)
  if len(checksums) > 1:
    return False
  if (
      all(runs.complete for runs in scope)
      and not any(runs.degenerate for runs in scope)
      and len(scope) >= _MIN_DEVICES_FOR_MATCH
  ):
    return True
  return None


def _consensus(device_runs: Sequence[_DeviceRuns]) -> Optional[int]:
  """Returns the majority checksum of > half of the expected devices, or None.

  Missing devices and devices without a majority checksum count against it; a
  degenerate checksum is never the consensus.
  """
  counts = collections.Counter(
      checksum
      for checksum in (runs.majority_checksum for runs in device_runs)
      if checksum is not None
  )
  if not counts:
    return None
  checksum, count = counts.most_common(1)[0]
  if 2 * count <= len(device_runs):
    return None
  if any(
      rc.degenerate
      for runs in device_runs
      for rc in runs.run_checksums
      if rc.checksum == checksum
  ):
    return None
  return checksum


def _failed_runs(
    runs: _DeviceRuns, consensus: Optional[int]
) -> list[dict[str, Any]]:
  """Returns the runs whose checksum differs from `consensus` (all if None)."""
  return [
      {
          "run": rc.run,
          "checksum": format_checksum(rc.checksum),
          "degenerate": rc.degenerate,
      }
      for rc in runs.run_checksums
      if consensus is None or rc.checksum != consensus
  ]


def compare_gathered_checksums(
    gathered: np.ndarray,
    process_index: int,
    *,
    mesh_devices: Sequence[jax.Device],
) -> dict[str, Any]:
  """Returns the CHECKSUM_METRIC_KEYS metrics of the gathered rows (pure numpy).

  Every device should have the same checksum in every timed run. Mismatches are
  logged, not raised.

  Args:
    gathered: uint32 rows of every host and run, of shape (hosts, runs,
      local_devices, _NUM_ROW_WORDS).
    process_index: Process index of the calling host.
    mesh_devices: Devices expected to report, on all hosts.
  """
  gathered = np.asarray(gathered, dtype=np.uint32)
  if gathered.ndim != 4 or gathered.shape[-1] != _NUM_ROW_WORDS:
    raise ValueError(
        "gathered must have shape (hosts, runs, local_devices,"
        f" {_NUM_ROW_WORDS}), got {gathered.shape}."
    )
  global_device_runs = _device_runs(gathered, mesh_devices)
  local_device_runs = sorted(
      (
          r
          for r in global_device_runs
          if r.device.process_index == process_index
      ),
      key=lambda r: r.device.id,
  )
  local_device_checksums = [
      _device_checksum_entry(r) for r in local_device_runs if r.reported
  ]

  num_complete = sum(r.complete for r in global_device_runs)
  # Each with the runs whose result is degenerate.
  degenerate_devices = [
      {
          **get_device_identity(r.device),
          "degenerate_runs": [
              rc.run for rc in r.run_checksums if rc.degenerate
          ],
      }
      for r in global_device_runs
      if r.degenerate
  ]

  intra_match = _match_flag(local_device_runs)
  global_match = _match_flag(global_device_runs)

  # On a mismatch, every device with a run that differs from the consensus,
  # with those runs (all devices and runs if None). Otherwise [], even if
  # undecidable: only global_match says that.
  global_consensus = _consensus(global_device_runs)
  diverging = []
  if global_match is False:
    for r in global_device_runs:
      failed_runs = _failed_runs(r, global_consensus)
      if failed_runs:
        diverging.append({
            **get_device_identity(r.device),
            "consensus": format_checksum(global_consensus),
            "failed_runs": failed_runs,
        })
  diverging.sort(key=lambda d: (d["process_index"], d["device_id"]))

  if intra_match is False:
    logging.error(
        "CHECKSUM MISMATCH within host %d: local device checksums %s; local"
        " diverging devices %s",
        process_index,
        local_device_checksums,
        [d for d in diverging if d["process_index"] == process_index],
    )
  if global_match is False:
    logging.error(
        "CHECKSUM MISMATCH across slice (consensus=%s): diverging devices %s",
        format_checksum(global_consensus),
        diverging,
    )

  if num_complete < len(global_device_runs):
    logging.error(
        "Checksum check incomplete: %d of %d expected devices have a checksum"
        " for every timed run.",
        num_complete,
        len(global_device_runs),
    )
  if degenerate_devices and global_consensus is not None:
    logging.error(
        "Degenerate result (inf/NaN or all zeros) on devices %s; likely a"
        " device fault, see diverging_devices.",
        degenerate_devices,
    )
  elif degenerate_devices:
    logging.error(
        "Degenerate chain result (inf/NaN or all zeros) on devices %s; the"
        " checksum check is invalid. Try a shorter chain_length or another"
        " dtype.",
        degenerate_devices,
    )

  return {
      CHECKSUM: format_checksum(global_consensus),
      ALL_DEVICES_VALID_CHECKSUM: (
          num_complete == len(global_device_runs) and not degenerate_devices
      ),
      INTRA_HOST_CHECKSUM_MATCH: intra_match,
      GLOBAL_CHECKSUM_MATCH: global_match,
      DEVICE_CHECKSUMS: local_device_checksums,
      DIVERGING_DEVICES: diverging,
  }


def _local_payload(local_runs: Sequence[np.ndarray]) -> np.ndarray:
  """Returns this host's rows of every timed run, stacked in run order.

  Never raises: invalid rows are logged and replaced by unreported ones, so the
  host still joins the allgather with the shape every host expects. Without a
  timed run, returns one unreported run, so every host still gathers once.

  Args:
    local_runs: This host's rows of every timed run, in run order.
  """
  unreported = read_local_rows(None)
  try:
    run_rows = [np.asarray(rows, dtype=np.uint32) for rows in local_runs]
    for rows in run_rows:
      if rows.shape != unreported.shape:
        raise ValueError(
            f"Expected rows of shape {unreported.shape}, got {rows.shape}."
        )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error("Invalid checksum rows; no run is reported: %s", e)
    run_rows = [unreported] * len(local_runs)
  return np.stack(run_rows or [unreported])


def gather_and_compare_checksums(
    local_runs: Sequence[np.ndarray], mesh: jax.sharding.Mesh
) -> dict[str, Any]:
  """Returns the CHECKSUM_METRIC_KEYS metrics of every timed run's final X.

  Call on every host after the timed loop. The rows of every run are gathered
  in one allgather, which needs the same shape, i.e. the same number of runs,
  on every host; a mismatch can hang it. Nothing in the timed loop keeps the
  hosts in step (the chain has no collectives), so `GemmChainedParams` requires
  `min_duration_s == 0`: every host then runs exactly `num_runs` timed runs.
  Never raises: a failing device is marked as not reported, so every host still
  joins the allgather.

  Args:
    local_runs: This host's rows of every timed run, in run order: one
      `checksum_local_rows` result per run.
    mesh: Mesh that ran the chain; its devices are the ones expected to report.
  """
  mesh_devices = list(mesh.devices.flat)
  if len(mesh_devices) < jax.device_count():
    logging.warning(
        "Checksum mesh covers %d of %d devices; devices outside the mesh"
        " are not checked.",
        len(mesh_devices),
        jax.device_count(),
    )
  payload = _local_payload(local_runs)
  try:
    # uint32, since uint64 is truncated without jax_enable_x64. Assumes every
    # host has the same number of local devices.
    gathered = multihost_utils.process_allgather(payload)
    return compare_gathered_checksums(
        gathered, jax.process_index(), mesh_devices=mesh_devices
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error("Checksum gather/compare failed: %s", e)
    # Every reported local row is a local device, so this lookup cannot fail.
    local_device_runs = _device_runs(
        payload[np.newaxis], sorted(jax.local_devices(), key=lambda d: d.id)
    )
    return _invalid_result(
        [_device_checksum_entry(r) for r in local_device_runs if r.reported]
    )
