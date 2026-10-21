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
    words: (hosts, runs, local_devices, 2) `(s1, s2)` per run and device, or
      (hosts, local_devices, 2) for a single run.
    reported: Optional 0/1 flags shaped like `words[..., 0]`; default all
      reported.
    degenerate: Optional 0/1 flags shaped like `words[..., 0]`; default none.

  Returns:
    uint32 rows `(device_id, reported, s1, s2, degenerate)` of shape
    (hosts, runs, local_devices, 5).
  """
  words = np.asarray(words, dtype=np.uint32)
  if words.ndim == 3:  # A single run.
    words = words[:, np.newaxis]
  shape = words.shape[:3]
  num_hosts, _, num_local = shape
  device_ids = np.arange(num_hosts * num_local).reshape(num_hosts, 1, num_local)
  if reported is None:
    reported = np.ones(shape)
  if degenerate is None:
    degenerate = np.zeros(shape)
  return np.stack(
      [
          np.broadcast_to(device_ids, shape),
          np.reshape(reported, shape),
          words[..., 0],
          words[..., 1],
          np.reshape(degenerate, shape),
      ],
      axis=-1,
  ).astype(np.uint32)


# Checksum words of the run results in `_pattern_rows`; "N" is degenerate.
_PATTERN_WORDS = {
    "A": (1, 2),
    "B": (3, 4),
    "C": (5, 6),
    "D": (7, 8),
    "N": (9, 9),
}


def _pattern_rows(patterns):
  """Builds gathered rows from one string of run results per device.

  Args:
    patterns: `patterns[h][l]` has one character per timed run of local device
      `l` of host `h`: a key of `_PATTERN_WORDS`, or "-" for a run whose
      checksum was not read back. Every string has the same length.

  Returns:
    `_rows` of shape (hosts, runs, local_devices, 5).
  """
  results = np.array([[list(p) for p in host] for host in patterns])
  # (hosts, local_devices, runs) -> (hosts, runs, local_devices).
  results = results.swapaxes(1, 2)
  words = [_PATTERN_WORDS.get(r, (0, 0)) for r in results.flat]
  return _rows(
      np.reshape(words, results.shape + (2,)),
      reported=results != "-",
      degenerate=results == "N",
  )


def _checksum_of(result):
  """Returns the formatted checksum of a `_pattern_rows` result, or None."""
  if result is None:
    return None
  return checksum.format_checksum(
      checksum.combine_checksum_words(*_PATTERN_WORDS[result])
  )


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


def _local_runs(mesh, value=1.0, num_runs=1, checksum_fn=None):
  """Returns this host's rows of each of `num_runs` runs (`local_runs`).

  Args:
    mesh: Mesh whose devices report a checksum.
    value: Value of every element of each device's 8 x 8 bfloat16 final X.
    num_runs: Number of timed runs.
    checksum_fn: Checksum function; defaults to `make_checksum_fn(mesh)`.
  """
  block = jnp.full((8, 8), value, dtype=jnp.bfloat16)
  x = _per_device_x(mesh, [block] * mesh.devices.size)
  checksum_fn = checksum_fn or checksum.make_checksum_fn(mesh)
  return [checksum.checksum_local_rows(x, checksum_fn) for _ in range(num_runs)]


def _spy_allgather():
  """Returns a patch that spies on `process_allgather` without changing it."""
  return mock.patch.object(
      checksum.multihost_utils,
      "process_allgather",
      wraps=checksum.multihost_utils.process_allgather,
  )


def _gathered_payload(spy_gather):
  """Returns this host's payload of the only `process_allgather` call."""
  spy_gather.assert_called_once()
  (payload,) = spy_gather.call_args.args
  return payload


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
        "consensus": "0x3f2a9c1e7b4d0e58",
        "failed_runs": [{
            "run": 0,
            "checksum": "0x91c07d22a8e3f4b1",
            "degenerate": False,
        }],
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
    """Verifies a tie: no consensus, every run of every device is listed."""
    res = checksum.compare_gathered_checksums(
        _pattern_rows([["AAA", "AAA"], ["BBB", "BBB"]]),
        process_index=0,
        mesh_devices=_devices(2, 2),
    )
    self.assertIsNone(res["checksum"])
    self.assertTrue(res["all_devices_valid_checksum"])
    self.assertIs(res["intra_host_checksum_match"], True)
    self.assertIs(res["global_checksum_match"], False)
    self.assertEqual(
        [d["device_id"] for d in res["diverging_devices"]], [0, 1, 2, 3]
    )
    for entry in res["diverging_devices"]:
      result = "A" if entry["process_index"] == 0 else "B"
      self.assertIsNone(entry["consensus"])
      self.assertEqual(
          entry["failed_runs"],
          [
              {
                  "run": run,
                  "checksum": _checksum_of(result),
                  "degenerate": False,
              }
              for run in range(3)
          ],
      )

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


class MultiRunCompareTest(parameterized.TestCase):
  """Tests `compare_gathered_checksums` over several timed runs (pure numpy)."""

  @parameterized.named_parameters(
      # (runs of the device, expected device checksum; None if no majority)
      ("one_run", "A", "A"),
      ("two_runs_agree", "AA", "A"),
      ("two_runs_disagree", "AB", None),
      # A run without a checksum counts against every value.
      ("two_runs_one_missing", "A-", None),
      ("three_runs_one_missing", "AA-", "A"),
      ("three_runs_all_differ", "ABC", None),
      ("four_runs_tie", "AABB", None),
      ("five_runs_one_transient", "AABAA", "A"),
      ("five_runs_no_majority", "AABCD", None),
      # A degenerate checksum can represent a device, but is never consensus.
      ("five_runs_degenerate", "NNNNN", "N"),
  )
  def test_device_checksum_is_strict_majority_of_its_runs(self, runs, expected):
    """Verifies a device's checksum needs more than half of its timed runs."""
    res = checksum.compare_gathered_checksums(
        _pattern_rows([[runs]]), 0, mesh_devices=_devices(1, 1)
    )
    (entry,) = res["device_checksums"]
    self.assertEqual(entry["checksum"], _checksum_of(expected))

  @parameterized.named_parameters(
      # (runs of device 5, expected intra-host match of host 1 and global
      #  match, device 5's checksum, all_devices_valid_checksum)
      ("healthy", "AAAAA", True, "A", True),
      ("transient_run_2", "AABAA", False, "A", True),
      ("transient_run_0", "BAAAA", False, "A", True),
      ("persistent", "BBBBB", False, "B", True),
      ("wrong_from_run_2_on", "AABBB", False, "B", True),
      ("several_wrong_values", "AABCD", False, None, True),
      ("degenerate_run_2", "AANAA", False, "A", False),
      ("missing_run_2", "AA-AA", None, "A", False),
  )
  def test_compares_every_timed_run_of_every_device(
      self,
      device_5_runs,
      expected_match,
      expected_device_5_checksum,
      expected_valid,
  ):
    """Verifies 2 hosts x 4 devices x 5 runs, where device 5 varies."""
    rows = _pattern_rows(
        [["AAAAA"] * 4, ["AAAAA", device_5_runs, "AAAAA", "AAAAA"]]
    )
    # Device 5's runs whose checksum differs from the consensus "A".
    failed_runs = [
        {
            "run": run,
            "checksum": _checksum_of(result),
            "degenerate": result == "N",
        }
        for run, result in enumerate(device_5_runs)
        if result not in "A-"
    ]
    expected_diverging = (
        [{
            "process_index": 1,
            "device_id": 5,
            "coords": None,
            "core_on_chip": None,
            "consensus": _checksum_of("A"),
            "failed_runs": failed_runs,
        }]
        if failed_runs
        else []
    )
    # Host 0's devices all agree; only host 1 holds device 5.
    for process_index, expected_intra in ((0, True), (1, expected_match)):
      res = checksum.compare_gathered_checksums(
          rows, process_index, mesh_devices=_devices(2, 4)
      )
      self.assertIs(res["intra_host_checksum_match"], expected_intra)
      self.assertIs(res["global_checksum_match"], expected_match)
      self.assertEqual(res["checksum"], _checksum_of("A"))
      self.assertIs(res["all_devices_valid_checksum"], expected_valid)
      self.assertEqual(res["diverging_devices"], expected_diverging)
      if process_index == 1:
        device_5 = res["device_checksums"][1]
        self.assertEqual(device_5["device_id"], 5)
        self.assertEqual(
            device_5["checksum"], _checksum_of(expected_device_5_checksum)
        )


class GatherAndCompareChecksumsTest(parameterized.TestCase):
  """Tests `checksum_local_rows` and `gather_and_compare_checksums` on CPU."""

  def _assert_undecided(self, res, num_device_checksums=0):
    """Asserts `res` decides nothing and lists `num_device_checksums`."""
    self.assertIsNone(res["checksum"])
    self.assertFalse(res["all_devices_valid_checksum"])
    self.assertIsNone(res["intra_host_checksum_match"])
    self.assertIsNone(res["global_checksum_match"])
    self.assertLen(res["device_checksums"], num_device_checksums)
    self.assertEqual(res["diverging_devices"], [])

  def _assert_logged(self, logs, message):
    self.assertTrue(any(message in line for line in logs.output), logs.output)

  @parameterized.named_parameters(
      # (device whose readback fails, expected reported column,
      #  expected all_devices_valid_checksum, expected intra/global match)
      ("all_mesh_devices_read", None, [1, 1, 1, 1, 0, 0, 0, 0], True, True),
      ("one_read_failure", 2, [1, 1, 0, 1, 0, 0, 0, 0], False, None),
  )
  def test_partial_mesh_gathers_fixed_shape_rows_per_run_and_local_device(
      self, failing_device_id, expected_reported, expected_valid, expected_match
  ):
    """Verifies every host joins the allgather, even if a readback fails.

    Skipping a collective on one host would leave the other hosts waiting in
    `process_allgather`, so every host contributes one row per run and local
    device (here 8, while the mesh has 4). Devices outside the mesh do not
    count as missing for the intra-host match.

    Args:
      failing_device_id: Device whose readback raises, or None.
      expected_reported: Expected `reported` column of each run's rows.
      expected_valid: Expected `all_devices_valid_checksum`.
      expected_match: Expected intra-host and global match.
    """
    mesh = _mesh(4)
    read_shard_words = checksum._read_shard_words  # pylint: disable=protected-access

    def maybe_fail(shard):
      if shard.device.id == failing_device_id:
        raise RuntimeError("simulated D2H failure")
      return read_shard_words(shard)

    with mock.patch.object(
        checksum, "_read_shard_words", side_effect=maybe_fail
    ), _spy_allgather() as spy_gather, self.assertLogs(level="WARNING") as logs:
      local_runs = _local_runs(mesh, num_runs=2)
      res = checksum.gather_and_compare_checksums(local_runs, mesh)

    # The rows of every run, gathered once.
    payload = _gathered_payload(spy_gather)
    # Rows are (device_id, reported, s1, s2, degenerate).
    self.assertEqual(payload.shape, (2, 8, 5))
    for rows in payload:
      device_ids, reported = rows[:, :2].T
      self.assertEqual(device_ids.tolist(), list(range(8)))
      self.assertEqual(reported.tolist(), expected_reported)
    self._assert_logged(logs, "covers 4 of")
    self.assertIs(res["all_devices_valid_checksum"], expected_valid)
    self.assertIs(res["intra_host_checksum_match"], expected_match)
    self.assertIs(res["global_checksum_match"], expected_match)
    self.assertEqual(res["diverging_devices"], [])
    self.assertEqual(
        [d["device_id"] for d in res["device_checksums"]],
        [i for i, r in enumerate(expected_reported) if r],
    )

  def test_never_raises_on_failure(self):
    """Verifies an invalid result (no raise) with no runs or a gather error."""
    with _spy_allgather() as spy_gather:
      res_none = checksum.gather_and_compare_checksums(
          [], _mesh(_NUM_CPU_DEVICES)
      )
    # Without a timed run, every host still gathers one unreported run.
    payload = _gathered_payload(spy_gather)
    self.assertEqual(payload[..., 1].tolist(), [[0] * _NUM_CPU_DEVICES])
    self._assert_undecided(res_none)

    mesh = _mesh(4)
    local_runs = _local_runs(mesh)
    with mock.patch.object(
        checksum.multihost_utils,
        "process_allgather",
        side_effect=RuntimeError("simulated gather failure"),
    ):
      res_err = checksum.gather_and_compare_checksums(local_runs, mesh)
    # Local device checksums read prior to allgather are still preserved
    self._assert_undecided(res_err, num_device_checksums=4)

  def test_checksum_failure_still_joins_the_allgather(self):
    """Verifies a failing checksum computation does not skip a collective.

    Skipping `process_allgather` on one host would leave the other hosts
    waiting, so a host whose checksum computation fails still contributes its
    fixed-shape payload, with every local device marked as not reported.
    """
    mesh = _mesh(_NUM_CPU_DEVICES)
    failing_checksum_fn = mock.Mock(
        side_effect=RuntimeError("simulated checksum failure")
    )
    with _spy_allgather() as spy_gather, self.assertLogs(level="ERROR") as logs:
      local_runs = _local_runs(mesh, checksum_fn=failing_checksum_fn)
      res = checksum.gather_and_compare_checksums(local_runs, mesh)

    failing_checksum_fn.assert_called_once()
    payload = _gathered_payload(spy_gather)
    # Rows are (device_id, reported, s1, s2, degenerate).
    self.assertEqual(payload[..., 1].tolist(), [[0] * _NUM_CPU_DEVICES])
    self._assert_undecided(res)
    self._assert_logged(logs, "Computing device checksums failed")

  def test_invalid_rows_still_join_the_allgather(self):
    """Verifies rows of the wrong shape are replaced by unreported rows."""
    with _spy_allgather() as spy_gather, self.assertLogs(level="ERROR") as logs:
      res = checksum.gather_and_compare_checksums(
          [np.zeros((3, 5), dtype=np.uint32)], _mesh(4)
      )

    payload = _gathered_payload(spy_gather)
    self.assertEqual(payload.shape, (1, _NUM_CPU_DEVICES, 5))
    self.assertEqual(payload[..., 1].tolist(), [[0] * _NUM_CPU_DEVICES])
    self._assert_undecided(res)
    self._assert_logged(logs, "Invalid checksum rows")

  @parameterized.named_parameters(
      ("all_zero", 0.0),
      ("nan", float("nan")),
  )
  def test_degenerate_chain_is_invalid(self, value):
    """Verifies an all-zero or NaN chain result fails the check end to end."""
    mesh = _mesh(8)
    with self.assertLogs(level="ERROR") as logs:
      res = checksum.gather_and_compare_checksums(
          _local_runs(mesh, value), mesh
      )
    self.assertFalse(res["all_devices_valid_checksum"])
    # Every device computed the same degenerate result: undeterminable.
    self.assertIsNone(res["intra_host_checksum_match"])
    self.assertIsNone(res["global_checksum_match"])
    self.assertEqual(res["diverging_devices"], [])
    self.assertLen(res["device_checksums"], 8)
    self._assert_logged(logs, "Degenerate chain result")
    self._assert_logged(logs, "'degenerate_runs': [0]")


if __name__ == "__main__":
  absltest.main()
