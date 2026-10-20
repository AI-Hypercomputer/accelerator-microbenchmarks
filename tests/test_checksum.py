"""Unit tests for core/checksum.py."""

import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.core import checksum
import jax
import jax.numpy as jnp
import numpy as np

_NUM_CPU_DEVICES = 8

# Force multiple virtual CPU devices so per-device checksums and their
# comparison can be tested without TPUs. Must run before the JAX backend is
# initialized; JAX raises an error if the backend has already been initialized.
jax.config.update("jax_num_cpu_devices", _NUM_CPU_DEVICES)
# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


def _flip_bit(x, flat_index, bit):
  """Returns `x` with bit `bit` of element `flat_index` (row-major) flipped."""
  uint_dtype = checksum.get_bitcast_uint_dtype(x.dtype)
  u = jax.lax.bitcast_convert_type(x, uint_dtype).reshape(-1)
  u = u.at[flat_index].set(u[flat_index] ^ jnp.asarray(1 << bit, uint_dtype))
  return jax.lax.bitcast_convert_type(u.reshape(x.shape), x.dtype)


def _device(process_index, device_id, coords=None, core_on_chip=None):
  """Returns a fake device with the attributes that core/checksum.py reads."""
  return types.SimpleNamespace(
      process_index=process_index,
      id=device_id,
      coords=coords,
      core_on_chip=core_on_chip,
  )


def _devices(num_hosts, num_local):
  """Fake mesh devices with device_id = host * num_local + local index."""
  return [
      _device(h, h * num_local + l)
      for h in range(num_hosts)
      for l in range(num_local)
  ]


def _rows(words, reported=None, degenerate=None):
  """Builds the gathered rows of `compare_gathered_checksums`.

  Device `l` of host `h` has device_id `h * local_devices + l`, as in
  `_devices`.

  Args:
    words: (hosts, local_devices, 2) `(s1, s2)` per device.
    reported: Optional (hosts, local_devices) 0/1 flags; default all reported.
    degenerate: Optional (hosts, local_devices) 0/1 flags; default none.

  Returns:
    uint32 rows `(device_id, reported, s1, s2, degenerate)` of shape
    (hosts, local_devices, 5).
  """
  words = np.asarray(words, dtype=np.uint32)
  shape = words.shape[:2]
  if reported is None:
    reported = np.ones(shape, dtype=np.uint32)
  if degenerate is None:
    degenerate = np.zeros(shape, dtype=np.uint32)
  device_ids = np.arange(np.prod(shape)).reshape(shape)
  return np.stack(
      [device_ids, reported, words[..., 0], words[..., 1], degenerate], axis=-1
  ).astype(np.uint32)


def _mesh(num_devices):
  return jax.sharding.Mesh(
      np.array(jax.devices()[:num_devices]), axis_names=("device",)
  )


def _per_device_x(mesh, blocks):
  """Returns X whose block `d` along axis 0 is `blocks[d]`, on mesh device `d`.

  This is the layout `make_checksum_fn` takes, where block `d` is the final X of
  mesh device `d`.

  Args:
    mesh: Mesh whose devices each hold one block.
    blocks: One equally shaped block per mesh device, in mesh order.
  """
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(tuple(mesh.axis_names))
  )
  return jax.device_put(jnp.concatenate(blocks), sharding)


def _gather_args(mesh, value=1.0):
  """Returns the arguments of `gather_and_compare_checksums` for `mesh`.

  Args:
    mesh: Mesh whose devices report a checksum.
    value: Value of every element of each device's 8 x 8 bfloat16 final X.

  Returns:
    `(x, checksum_fn, mesh)`: every device's final X (see `_per_device_x`), the
    checksum function for `mesh`, and `mesh`.
  """
  block = jnp.full((8, 8), value, dtype=jnp.bfloat16)
  x = _per_device_x(mesh, [block] * mesh.devices.size)
  return x, checksum.make_checksum_fn(mesh), mesh


class ChecksumWordsTest(parameterized.TestCase):
  """Tests for the checksum formula, dtype widths and device identities."""

  @parameterized.named_parameters(
      ("uint32", jnp.uint32),
      ("float32", jnp.float32),
      ("bfloat16", jnp.bfloat16),
      ("float8_e4m3fn", jnp.float8_e4m3fn),
      ("int8", jnp.int8),
  )
  def test_checksum_words_matches_hand_computed_reference(self, dtype):
    """Verifies s1, s2, combine, and hex formatting against manual formula."""
    x = jnp.arange(1, 7, dtype=jnp.float32).reshape(2, 3).astype(dtype)
    words = checksum.checksum_words(x)
    self.assertEqual(words.shape, (2,))
    self.assertEqual(words.dtype, jnp.uint32)

    # Compute reference manually in Python int
    uint_dtype = checksum.get_bitcast_uint_dtype(dtype)
    raw_u = np.asarray(
        jax.lax.bitcast_convert_type(x, uint_dtype).reshape(-1),
        dtype=np.uint64,
    )
    expected_s1 = int(np.sum(raw_u)) & 0xFFFFFFFF
    expected_s2 = (
        sum((idx + 1) * int(val) for idx, val in enumerate(raw_u)) & 0xFFFFFFFF
    )
    self.assertEqual(int(words[0]), expected_s1)
    self.assertEqual(int(words[1]), expected_s2)

    combined = checksum.combine_checksum_words(int(words[0]), int(words[1]))
    self.assertEqual(combined, (expected_s1 << 32) | expected_s2)
    formatted = checksum.format_checksum(combined)
    self.assertEqual(formatted, f"0x{combined:016x}")
    self.assertLen(formatted, 18)
    self.assertIsNone(checksum.format_checksum(None))

  def test_checksum_sensitivity_bit_flip_swap_and_permutation(self):
    """Verifies bit flips, swaps and permutations change the checksum."""
    x = jnp.array([[1.5, -2.25, 3.75], [0.125, 16.0, -0.5]], dtype=jnp.bfloat16)
    w_base = checksum.checksum_words(x)
    c_base = checksum.combine_checksum_words(int(w_base[0]), int(w_base[1]))

    # 1. Single-bit flip changes both s1 and s2
    x_flipped = _flip_bit(x, flat_index=2, bit=0)
    w_flipped = checksum.checksum_words(x_flipped)
    c_flipped = checksum.combine_checksum_words(
        int(w_flipped[0]), int(w_flipped[1])
    )
    self.assertNotEqual(c_base, c_flipped)
    self.assertNotEqual(int(w_base[0]), int(w_flipped[0]))
    self.assertNotEqual(int(w_base[1]), int(w_flipped[1]))

    # 2. Swapping two distinct elements keeps s1 equal (plain sum would miss!)
    #    but changes position-weighted s2 and the combined 64-bit checksum.
    flat = x.reshape(-1)
    x_swapped = flat.at[0].set(flat[5]).at[5].set(flat[0]).reshape(x.shape)
    w_swapped = checksum.checksum_words(x_swapped)
    c_swapped = checksum.combine_checksum_words(
        int(w_swapped[0]), int(w_swapped[1])
    )
    self.assertEqual(int(w_base[0]), int(w_swapped[0]))
    self.assertNotEqual(int(w_base[1]), int(w_swapped[1]))
    self.assertNotEqual(c_base, c_swapped)

    # 3. Row permutation changes s2 and combined checksum
    x_permuted = x[::-1, :]
    w_permuted = checksum.checksum_words(x_permuted)
    c_permuted = checksum.combine_checksum_words(
        int(w_permuted[0]), int(w_permuted[1])
    )
    self.assertEqual(int(w_base[0]), int(w_permuted[0]))
    self.assertNotEqual(c_base, c_permuted)

  def test_checksum_words_wraparound_mod_2_32(self):
    """Verifies uint32 wraparound and 64-bit combine without truncation."""
    x = jnp.array([0xFFFFFFFF, 0x00000002], dtype=jnp.uint32)
    words = checksum.checksum_words(x)
    # s1 = (0xFFFFFFFF + 2) mod 2^32 = 1
    # s2 = (1 * 0xFFFFFFFF + 2 * 2) mod 2^32 = 3
    self.assertEqual(int(words[0]), 1)
    self.assertEqual(int(words[1]), 3)
    self.assertEqual(
        checksum.combine_checksum_words(int(words[0]), int(words[1])),
        (1 << 32) | 3,
    )

  @parameterized.named_parameters(
      ("float8_e4m3fn", jnp.float8_e4m3fn, jnp.uint8),
      ("int8", jnp.int8, jnp.uint8),
      ("bfloat16", jnp.bfloat16, jnp.uint16),
      ("float32", jnp.float32, jnp.uint32),
      # Rejected: sub-byte dtypes have itemsize 1 but are 4 bits wide.
      ("int4", jnp.int4, None),
      ("float4_e2m1fn", jnp.float4_e2m1fn, None),
      ("float64", jnp.float64, None),
      ("complex64", jnp.complex64, None),
      ("bool", jnp.bool_, None),
  )
  def test_get_bitcast_uint_dtype_by_true_bit_width(self, dtype, expected):
    """Verifies the same-width uint mapping and rejection of other widths."""
    self.assertEqual(
        checksum.is_checksummable_dtype(dtype), expected is not None
    )
    if expected is None:
      with self.assertRaisesRegex(ValueError, r"bit widths \(8, 16, 32\) only"):
        checksum.get_bitcast_uint_dtype(dtype)
    else:
      self.assertEqual(
          jnp.dtype(checksum.get_bitcast_uint_dtype(dtype)),
          jnp.dtype(expected),
      )

  def test_get_device_identity(self):
    """Verifies get_device_identity on TPU and CPU devices."""
    # 1. Simulated TPU device with coords and core_on_chip
    tpu_dev = types.SimpleNamespace(
        process_index=1, id=5, coords=(0, 1, 0), core_on_chip=1
    )
    self.assertEqual(
        checksum.get_device_identity(tpu_dev),
        {
            "process_index": 1,
            "device_id": 5,
            "coords": [0, 1, 0],
            "core_on_chip": 1,
        },
    )

    # 2. CPU device lacking coords and core_on_chip -> None
    cpu_id = checksum.get_device_identity(jax.devices()[0])
    self.assertIsNone(cpu_id["coords"])
    self.assertIsNone(cpu_id["core_on_chip"])


class MakeChecksumFnTest(absltest.TestCase):
  """Tests for `make_checksum_fn` on simulated CPU devices."""

  def test_each_device_checksums_its_own_block(self):
    """Verifies row `d` is the checksum of mesh device `d`'s own block.

    Device `bad` holds a corrupted block, so only its row may differ from the
    other devices'.
    """
    num_devices, m, k, bad = 4, 8, 16, 2
    mesh = _mesh(num_devices)
    healthy = jax.random.normal(jax.random.PRNGKey(0), (m, k)).astype(
        jnp.bfloat16
    )
    blocks = [
        _flip_bit(healthy, flat_index=3, bit=6) if d == bad else healthy
        for d in range(num_devices)
    ]
    rows = checksum.make_checksum_fn(mesh)(_per_device_x(mesh, blocks))

    # One `(s1, s2, degenerate)` row per device.
    self.assertEqual(rows.shape, (num_devices, 3))
    self.assertEqual(rows.dtype, jnp.uint32)

    row_blocks = {s.device: np.asarray(s.data) for s in rows.addressable_shards}
    for block, device in zip(blocks, mesh.devices.flat):
      # The row on each device is the checksum of that device's own block.
      expected_row = np.asarray(checksum.checksum_words(block)).tolist() + [0]
      self.assertEqual(row_blocks[device].reshape(-1).tolist(), expected_row)
    self.assertNotEqual(
        row_blocks[mesh.devices.flat[bad]].tolist(),
        row_blocks[mesh.devices.flat[0]].tolist(),
    )


class CompareGatheredChecksumsTest(parameterized.TestCase):
  """Tests for `compare_gathered_checksums` (pure numpy)."""

  def test_two_hosts_four_devices_single_diverging_matches_doc(self):
    """Tests the design doc's 2 hosts x 4 devices scenario (device 5 bad)."""
    good_s1, good_s2 = 0x3F2A9C1E, 0x7B4D0E58
    bad_s1, bad_s2 = 0x91C07D22, 0xA8E3F4B1

    words = np.tile(np.array([good_s1, good_s2], dtype=np.uint32), (2, 4, 1))
    # Device 5 is local index 1 on host 1
    words[1, 1] = [bad_s1, bad_s2]

    host0_coords = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]]
    host1_coords = [[0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0]]
    mesh_devices = [
        _device(h, 4 * h + l, coords=coords, core_on_chip=l % 2)
        for h, host_coords in enumerate([host0_coords, host1_coords])
        for l, coords in enumerate(host_coords)
    ]

    # Host 0 view: intra=True, global=False
    res_h0 = checksum.compare_gathered_checksums(
        _rows(words), process_index=0, mesh_devices=mesh_devices
    )
    self.assertEqual(res_h0["checksum"], "0x3f2a9c1e7b4d0e58")
    self.assertTrue(res_h0["all_devices_valid_checksum"])
    self.assertTrue(res_h0["intra_host_checksum_match"])
    self.assertFalse(res_h0["global_checksum_match"])
    self.assertEqual(
        res_h0["device_checksums"],
        [
            {
                "device_id": 0,
                "coords": [0, 0, 0],
                "core_on_chip": 0,
                "checksum": "0x3f2a9c1e7b4d0e58",
            },
            {
                "device_id": 1,
                "coords": [0, 0, 0],
                "core_on_chip": 1,
                "checksum": "0x3f2a9c1e7b4d0e58",
            },
            {
                "device_id": 2,
                "coords": [1, 0, 0],
                "core_on_chip": 0,
                "checksum": "0x3f2a9c1e7b4d0e58",
            },
            {
                "device_id": 3,
                "coords": [1, 0, 0],
                "core_on_chip": 1,
                "checksum": "0x3f2a9c1e7b4d0e58",
            },
        ],
    )
    expected_diverging = [{
        "process_index": 1,
        "device_id": 5,
        "coords": [0, 1, 0],
        "core_on_chip": 1,
        "checksum": "0x91c07d22a8e3f4b1",
        "consensus": "0x3f2a9c1e7b4d0e58",
    }]
    self.assertEqual(res_h0["diverging_devices"], expected_diverging)

    # Host 1 view: intra=False, global=False, same slice-wide diverging_devices
    res_h1 = checksum.compare_gathered_checksums(
        _rows(words), process_index=1, mesh_devices=mesh_devices
    )
    self.assertEqual(res_h1["checksum"], "0x3f2a9c1e7b4d0e58")
    self.assertTrue(res_h1["all_devices_valid_checksum"])
    self.assertFalse(res_h1["intra_host_checksum_match"])
    self.assertFalse(res_h1["global_checksum_match"])
    self.assertEqual(res_h1["diverging_devices"], expected_diverging)

  def test_consensus_tie(self):
    """Verifies a tie: no consensus, every device listed as diverging."""
    words = np.array(
        [
            [[1, 10], [1, 10]],
            [[2, 20], [2, 20]],
        ],
        dtype=np.uint32,
    )
    res = checksum.compare_gathered_checksums(
        _rows(words), process_index=0, mesh_devices=_devices(2, 2)
    )
    self.assertIsNone(res["checksum"])
    self.assertTrue(res["all_devices_valid_checksum"])
    self.assertIs(res["intra_host_checksum_match"], True)
    self.assertIs(res["global_checksum_match"], False)
    self.assertLen(res["diverging_devices"], 4)
    for entry in res["diverging_devices"]:
      self.assertIsNone(entry["consensus"])

  @parameterized.named_parameters(
      # (hosts, local devices per host, expected intra on the last host)
      ("one_host", 1, 3, False),
      # One local device per host: intra is undeterminable.
      ("one_device_per_host", 3, 1, None),
  )
  def test_three_devices_one_diverging_reaches_consensus(
      self, num_hosts, num_local, expected_intra
  ):
    """Verifies 3 devices (2 agree) is enough to identify the diverging one."""
    words = np.array([[1, 2], [1, 2], [9, 9]], dtype=np.uint32).reshape(
        num_hosts, num_local, 2
    )
    res = checksum.compare_gathered_checksums(
        _rows(words),
        process_index=num_hosts - 1,
        mesh_devices=_devices(num_hosts, num_local),
    )
    expected_consensus = checksum.format_checksum(
        checksum.combine_checksum_words(1, 2)
    )
    self.assertEqual(res["checksum"], expected_consensus)
    self.assertTrue(res["all_devices_valid_checksum"])
    self.assertIs(res["intra_host_checksum_match"], expected_intra)
    self.assertIs(res["global_checksum_match"], False)
    self.assertLen(res["diverging_devices"], 1)
    self.assertEqual(res["diverging_devices"][0]["device_id"], 2)
    self.assertEqual(
        res["diverging_devices"][0]["consensus"], expected_consensus
    )

  @parameterized.named_parameters(
      # (words (1 host, local, 2), expected match flags, expected checksum,
      #  expected number of diverging devices)
      ("single_device", [[[1, 2]]], None, "0x0000000100000002", 0),
      ("two_devices_agree", [[[1, 2], [1, 2]]], True, "0x0000000100000002", 0),
      # 1 of 2 devices is not a majority: neither can be blamed.
      ("two_devices_disagree", [[[1, 2], [3, 4]]], False, None, 2),
  )
  def test_one_or_two_devices(
      self, words, expected_match, expected_checksum, expected_num_diverging
  ):
    """Verifies one or two devices: consensus needs a strict majority."""
    words = np.array(words, dtype=np.uint32)
    res = checksum.compare_gathered_checksums(
        _rows(words),
        process_index=0,
        mesh_devices=_devices(*words.shape[:2]),
    )
    self.assertEqual(res["checksum"], expected_checksum)
    self.assertTrue(res["all_devices_valid_checksum"])
    self.assertIs(res["intra_host_checksum_match"], expected_match)
    self.assertIs(res["global_checksum_match"], expected_match)
    self.assertLen(res["device_checksums"], words.shape[1])
    self.assertLen(res["diverging_devices"], expected_num_diverging)

  def test_skips_unreported_devices(self):
    """Verifies a missing mesh device makes agreeing matches None, not True."""
    # Rows 3 and 4 carry garbage words that must not affect the comparison.
    words = np.array(
        [[[1, 2], [1, 2], [1, 2], [7, 7], [9, 9]]], dtype=np.uint32
    )
    # Mesh = devices 0..3 (device 4 is outside it); device 3 did not report.
    mesh_devices = _devices(1, 5)[:4]
    res_missing = checksum.compare_gathered_checksums(
        _rows(words, reported=[[1, 1, 1, 0, 0]]),
        process_index=0,
        mesh_devices=mesh_devices,
    )
    self.assertFalse(res_missing["all_devices_valid_checksum"])
    # The reported devices agree, but the missing one could differ.
    self.assertIsNone(res_missing["intra_host_checksum_match"])
    self.assertIsNone(res_missing["global_checksum_match"])
    self.assertEqual(res_missing["diverging_devices"], [])
    # 3 of 4 expected devices are still a majority.
    self.assertEqual(res_missing["checksum"], "0x0000000100000002")
    self.assertEqual(
        [d["device_id"] for d in res_missing["device_checksums"]], [0, 1, 2]
    )

    # A disagreement among the reported devices is a definite mismatch.
    words[0, 2] = [5, 5]
    res_mismatch = checksum.compare_gathered_checksums(
        _rows(words, reported=[[1, 1, 1, 0, 0]]),
        process_index=0,
        mesh_devices=mesh_devices,
    )
    self.assertIs(res_mismatch["intra_host_checksum_match"], False)
    self.assertIs(res_mismatch["global_checksum_match"], False)
    # 2 of 4 is not a majority: the missing device could tie the vote.
    self.assertIsNone(res_mismatch["checksum"])
    self.assertLen(res_mismatch["diverging_devices"], 3)

    # All mesh devices reported: device 4, outside the mesh, is ignored.
    words[0, 2] = [1, 2]
    words[0, 3] = [1, 2]
    res_complete = checksum.compare_gathered_checksums(
        _rows(words, reported=[[1, 1, 1, 1, 0]]),
        process_index=0,
        mesh_devices=mesh_devices,
    )
    self.assertTrue(res_complete["all_devices_valid_checksum"])
    self.assertIs(res_complete["intra_host_checksum_match"], True)
    self.assertIs(res_complete["global_checksum_match"], True)

  @parameterized.named_parameters(
      # (degenerate flags, device 2's words, expected match flags,
      #  expected checksum, expected diverging device ids)
      ("one_device_agreeing", [[0, 0, 1, 0]], [1, 2], None, None, []),
      ("every_device_agreeing", [[1, 1, 1, 1]], [1, 2], None, None, []),
      # A degenerate device that also diverges (e.g. an SDC turned into NaN):
      # the other devices still form the consensus.
      (
          "one_device_diverging",
          [[0, 0, 1, 0]],
          [9, 9],
          False,
          "0x0000000100000002",
          [2],
      ),
  )
  def test_degenerate_result_never_passes(
      self,
      degenerate,
      device_2_words,
      expected_match,
      expected_checksum,
      expected_diverging,
  ):
    """Verifies a degenerate X is never True nor the consensus."""
    words = np.tile(np.array([1, 2], dtype=np.uint32), (1, 4, 1))
    words[0, 2] = device_2_words
    res = checksum.compare_gathered_checksums(
        _rows(words, degenerate=degenerate),
        process_index=0,
        mesh_devices=_devices(1, 4),
    )
    self.assertFalse(res["all_devices_valid_checksum"])
    self.assertIs(res["intra_host_checksum_match"], expected_match)
    self.assertIs(res["global_checksum_match"], expected_match)
    self.assertEqual(res["checksum"], expected_checksum)
    self.assertEqual(
        [d["device_id"] for d in res["diverging_devices"]], expected_diverging
    )


class GatherAndCompareChecksumsTest(parameterized.TestCase):
  """Tests for `gather_and_compare_checksums` on simulated CPU devices."""

  @parameterized.named_parameters(
      # (device whose readback fails, expected reported column,
      #  expected all_devices_valid_checksum, expected intra/global match)
      ("all_mesh_devices_read", None, [1, 1, 1, 1, 0, 0, 0, 0], True, True),
      ("one_read_failure", 2, [1, 1, 0, 1, 0, 0, 0, 0], False, None),
  )
  def test_partial_mesh_gathers_one_fixed_shape_row_per_local_device(
      self, failing_device_id, expected_reported, expected_valid, expected_match
  ):
    """Verifies every host gathers once, even if a local readback fails.

    Skipping the collective on one host would leave the other hosts waiting in
    `process_allgather`, so every host contributes one row per local device
    (here 8, while the mesh has 4) exactly once. Devices outside the mesh do
    not count as missing for the intra-host match.

    Args:
      failing_device_id: Device whose readback raises, or None.
      expected_reported: Expected `reported` column of the local payload.
      expected_valid: Expected `all_devices_valid_checksum`.
      expected_match: Expected intra-host and global match.
    """
    x, checksum_fn, mesh = _gather_args(_mesh(4))
    read_shard_words = checksum._read_shard_words  # pylint: disable=protected-access

    def maybe_fail(shard):
      if shard.device.id == failing_device_id:
        raise RuntimeError("simulated D2H failure")
      return read_shard_words(shard)

    with mock.patch.object(
        checksum, "_read_shard_words", side_effect=maybe_fail
    ), mock.patch.object(
        checksum.multihost_utils,
        "process_allgather",
        wraps=checksum.multihost_utils.process_allgather,
    ) as spy_gather, self.assertLogs(
        level="WARNING"
    ) as logs:
      res = checksum.gather_and_compare_checksums(x, checksum_fn, mesh)

    spy_gather.assert_called_once()
    (local_rows,) = spy_gather.call_args.args
    # Rows are (device_id, reported, s1, s2, degenerate).
    self.assertEqual(local_rows.shape, (8, 5))
    device_ids, reported = local_rows[:, :2].T
    self.assertEqual(device_ids.tolist(), list(range(8)))
    self.assertEqual(reported.tolist(), expected_reported)
    self.assertTrue(
        any("covers 4 of" in line for line in logs.output), logs.output
    )
    self.assertIs(res["all_devices_valid_checksum"], expected_valid)
    self.assertIs(res["intra_host_checksum_match"], expected_match)
    self.assertIs(res["global_checksum_match"], expected_match)
    self.assertEqual(res["diverging_devices"], [])
    self.assertEqual(
        [d["device_id"] for d in res["device_checksums"]],
        [i for i, r in enumerate(expected_reported) if r],
    )

  def test_never_raises_on_failure(self):
    """Verifies an invalid result (no raise) on a None array or gather error."""
    unused_checksum_fn = mock.Mock()
    res_none = checksum.gather_and_compare_checksums(
        None, unused_checksum_fn, _mesh(_NUM_CPU_DEVICES)
    )
    unused_checksum_fn.assert_not_called()
    self.assertIsNone(res_none["checksum"])
    self.assertFalse(res_none["all_devices_valid_checksum"])
    self.assertIsNone(res_none["intra_host_checksum_match"])
    self.assertIsNone(res_none["global_checksum_match"])
    self.assertEqual(res_none["device_checksums"], [])
    self.assertEqual(res_none["diverging_devices"], [])

    x, checksum_fn, mesh = _gather_args(_mesh(4))
    with mock.patch.object(
        checksum.multihost_utils,
        "process_allgather",
        side_effect=RuntimeError("simulated gather failure"),
    ):
      res_err = checksum.gather_and_compare_checksums(x, checksum_fn, mesh)
    self.assertIsNone(res_err["checksum"])
    self.assertFalse(res_err["all_devices_valid_checksum"])
    self.assertIsNone(res_err["intra_host_checksum_match"])
    self.assertIsNone(res_err["global_checksum_match"])
    # Local device checksums read prior to allgather are still preserved
    self.assertLen(res_err["device_checksums"], 4)
    self.assertEqual(res_err["diverging_devices"], [])

  def test_checksum_failure_still_joins_allgather(self):
    """Verifies a failing checksum computation does not skip the collective.

    Skipping `process_allgather` on one host would leave the other hosts
    waiting, so a host whose checksum computation fails still contributes its
    fixed-shape payload, with every local device marked as not reported.
    """
    x, _, mesh = _gather_args(_mesh(_NUM_CPU_DEVICES))
    failing_checksum_fn = mock.Mock(
        side_effect=RuntimeError("simulated checksum failure")
    )
    with mock.patch.object(
        checksum.multihost_utils,
        "process_allgather",
        wraps=checksum.multihost_utils.process_allgather,
    ) as spy_gather, self.assertLogs(level="ERROR") as logs:
      res = checksum.gather_and_compare_checksums(x, failing_checksum_fn, mesh)

    failing_checksum_fn.assert_called_once()
    spy_gather.assert_called_once()
    (local_rows,) = spy_gather.call_args.args
    # Rows are (device_id, reported, s1, s2, degenerate).
    _, reported = local_rows[:, :2].T
    self.assertEqual(reported.tolist(), [0] * _NUM_CPU_DEVICES)
    self.assertIsNone(res["checksum"])
    self.assertFalse(res["all_devices_valid_checksum"])
    self.assertIsNone(res["intra_host_checksum_match"])
    self.assertIsNone(res["global_checksum_match"])
    self.assertEqual(res["device_checksums"], [])
    self.assertEqual(res["diverging_devices"], [])
    self.assertTrue(
        any(
            "Computing device checksums failed" in line for line in logs.output
        ),
        logs.output,
    )

  @parameterized.named_parameters(
      ("all_zero", 0.0),
      ("nan", float("nan")),
  )
  def test_degenerate_chain_is_invalid(self, value):
    """Verifies an all-zero or NaN chain result fails the check end to end."""
    with self.assertLogs(level="ERROR") as logs:
      res = checksum.gather_and_compare_checksums(
          *_gather_args(_mesh(8), value)
      )
    self.assertFalse(res["all_devices_valid_checksum"])
    # Every device computed the same degenerate result: undeterminable.
    self.assertIsNone(res["intra_host_checksum_match"])
    self.assertIsNone(res["global_checksum_match"])
    self.assertEqual(res["diverging_devices"], [])
    self.assertLen(res["device_checksums"], 8)
    self.assertTrue(
        any("Degenerate chain result" in line for line in logs.output),
        logs.output,
    )


if __name__ == "__main__":
  absltest.main()
