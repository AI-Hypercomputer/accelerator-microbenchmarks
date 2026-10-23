"""Unit tests for host_device.py."""

import contextlib
import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import host_device
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
import jax
import jax.numpy as jnp
import numpy as np

# Number of virtual CPU devices, so multi-device sharding is exercised on CPU.
_NUM_DEVICES = 4

# Set CPU backend for fast testing without TPU requirements.
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_num_cpu_devices", _NUM_DEVICES)

_P = jax.sharding.PartitionSpec
_TPU_SYNC = host_device.TransferBackend.TPU_SYNC
_JAX = host_device.TransferBackend.JAX
_H2D = host_device.HostToDeviceBenchmark
_D2H = host_device.DeviceToHostBenchmark

# 4 MiB of bfloat16 is 2Mi elements, i.e. 16384 rows of 128 columns per device.
_DATA_SIZE_MIB = 4
_DATA_SIZE_BYTES = _DATA_SIZE_MIB * 1024 * 1024
_ROWS_PER_DEVICE = 16384

_BENCHMARK_CLASSES = (("h2d", _H2D), ("d2h", _D2H))

# Covers every backend x device-selection pair across both directions without
# the full 8-way cross product.
_LAYOUT_CASES = (
    ("h2d_jax_all", _H2D, "jax", -1),
    ("d2h_jax_dev0", _D2H, "jax", 0),
    ("h2d_tpu_sync_dev0", _H2D, "tpu_sync", 0),
    ("d2h_tpu_sync_all", _D2H, "tpu_sync", -1),
)

# KVCacheManager method each direction times in run_op().
_MEASURED_OP = {_H2D: "h2d", _D2H: "d2h"}


def _expected_num_devices(device_id: int) -> int:
  return _NUM_DEVICES if device_id == -1 else 1


def _expected_devices(device_id: int) -> list[jax.Device]:
  if device_id == -1:
    return list(jax.local_devices())
  return [jax.local_devices()[device_id]]


def _cross(*param_lists):
  """Returns the named cross product of `named_parameters`-style tuples."""
  cases = []
  for combo in itertools.product(*param_lists):
    name = "_".join(params[0] for params in combo)
    values = sum((tuple(params[1:]) for params in combo), ())
    cases.append((name,) + values)
  return cases


class _RecordingTraceAnnotation:
  """Fake TraceAnnotation that records enter/exit events into a shared log."""

  def __init__(self, events: list[str]):
    self._events = events

  def __call__(self, name: str, **kwargs):
    del kwargs  # Unused.
    events = self._events

    @contextlib.contextmanager
    def _ctx():
      events.append(f"enter:{name}")
      try:
        yield
      finally:
        events.append(f"exit:{name}")

    return _ctx()


class _HostDeviceTestBase(parameterized.TestCase):
  """Shared fixtures: CPU platform info, mocked KVCacheManager, factories."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            platform,
            "get_platform_info",
            return_value=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
        )
    )
    # KVCacheManager requires TPU hardware; replace it with a mock.
    self.mock_kv_cls = self.enter_context(
        mock.patch.object(
            host_device.kv_cache_manager, "KVCacheManager", autospec=True
        )
    )
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  @property
  def mock_kv(self) -> mock.MagicMock:
    return self.mock_kv_cls.return_value

  def _make_benchmark(self, bm_cls, **kwargs):
    config_params = {"data_size_mib": _DATA_SIZE_MIB}
    config_params.update(kwargs)
    config = host_device.HostDeviceParams(**config_params)
    bm = bm_cls(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.addCleanup(bm.teardown)
    return bm

  def _setup_benchmark(self, bm_cls, **kwargs):
    bm = self._make_benchmark(bm_cls, **kwargs)
    bm.setup()
    return bm

  def _assert_row_sharded(self, arr: jax.Array, device_id: int):
    """Asserts `arr` is a 2D jax buffer row-sharded over the transfer devices."""
    self.assertIsInstance(arr, jax.Array)
    self.assertEqual(arr.sharding.spec, _P("device", None))
    self.assertEqual(arr.devices(), set(_expected_devices(device_id)))
    self.assertLen(arr.addressable_shards, _expected_num_devices(device_id))


class HostDeviceParamsValidationTest(parameterized.TestCase):
  """Verifies the bounds and enums declared on HostDeviceParams fields."""

  def test_defaults(self):
    params = host_device.HostDeviceParams()
    self.assertEqual(params.transfer_backend, _TPU_SYNC)
    self.assertEqual(params.device_id, host_device.ALL_LOCAL_DEVICES)
    self.assertEqual(params.data_size_mib, 64)
    self.assertEqual(params.data_size_bytes, 64 * 1024 * 1024)

  @parameterized.named_parameters(
      ("jax", "jax", _JAX), ("tpu_sync", "tpu_sync", _TPU_SYNC)
  )
  def test_transfer_backend_coerced_to_enum(self, value, expected):
    params = host_device.HostDeviceParams(transfer_backend=value)
    self.assertIsInstance(params.transfer_backend, host_device.TransferBackend)
    self.assertEqual(params.transfer_backend, expected)

  @parameterized.parameters("cuda", "TPU_SYNC", None)
  def test_invalid_transfer_backend_raises_error(self, value):
    with self.assertRaisesRegex(
        ValueError,
        rf"Invalid transfer_backend '{value}'\. Must be one of \['jax',"
        r" 'tpu_sync'\]",
    ):
      host_device.HostDeviceParams(transfer_backend=value)

  @parameterized.parameters(-1, 0, 1000)
  def test_valid_device_id_accepted_by_params(self, device_id):
    # Upper bound depends on the runtime device count and is checked in
    # setup(); only the lower bound is a static params constraint.
    params = host_device.HostDeviceParams(device_id=device_id)
    self.assertEqual(params.device_id, device_id)

  @parameterized.named_parameters(
      ("data_size_zero", "data_size_mib", 0, "data_size_mib must be >= 1"),
      (
          "device_id_minus_two",
          "device_id",
          -2,
          "device_id must be >= -1",
      ),
      (
          "device_id_numpy",
          "device_id",
          np.int64(-2),
          "device_id must be >= -1",
      ),
  )
  def test_out_of_bounds_param_raises_error(self, name, value, regex):
    """Values below a field's declared minimum are rejected."""
    with self.assertRaisesRegex(ValueError, regex):
      host_device.HostDeviceParams(**{name: value})

  @parameterized.named_parameters(
      *_cross(
          (
              ("device_id", "device_id", -1),
              ("data_size_mib", "data_size_mib", 1),
          ),
          (("none", None), ("bool", True), ("float", 4.0), ("str", "4")),
      )
  )
  def test_non_int_param_raises_error(self, name, minimum, value):
    """Non-integer values (including bools) are rejected."""
    with self.assertRaisesRegex(
        ValueError, rf"{name} must be an integer >= {minimum}"
    ):
      host_device.HostDeviceParams(**{name: value})

  def test_numpy_int_params_normalized_to_int(self):
    params = host_device.HostDeviceParams(
        data_size_mib=np.int64(8), device_id=np.int32(0)
    )
    self.assertIs(type(params.data_size_mib), int)
    self.assertIs(type(params.device_id), int)
    self.assertEqual(params.data_size_mib, 8)
    self.assertEqual(params.device_id, 0)


class HostDeviceCommonTest(_HostDeviceTestBase):
  """Behavior shared by HostToDeviceBenchmark and DeviceToHostBenchmark."""

  def test_cpu_device_count(self):
    # Guards the multi-device assumptions of every test in this file.
    self.assertLen(jax.local_devices(), _NUM_DEVICES)

  @parameterized.named_parameters(
      ("h2d", "host_to_device", _H2D), ("d2h", "device_to_host", _D2H)
  )
  def test_benchmark_registered(self, name, bm_cls):
    self.assertEqual(registry.benchmark_registry.get_benchmark(name), bm_cls)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_xprof_target_host_cpu_and_no_chip_bandwidth(self, bm_cls):
    bm = self._make_benchmark(bm_cls)
    self.assertTrue(bm.xprof_target_host_cpu)
    derived = bm.derive_chip_metrics(
        {"wall_clock_bandwidth_per_device_gb_s": 10.0}
    )
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", derived)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", derived)

  # --- device_id selection ----------------------------------------------

  @parameterized.named_parameters(("all", -1), ("dev0", 0), ("dev3", 3))
  def test_get_transfer_devices(self, device_id):
    bm = self._make_benchmark(_H2D, device_id=device_id)
    expected = _expected_devices(device_id)
    self.assertEqual(bm.get_transfer_devices(), expected)
    self.assertLen(expected, bm.num_transfer_devices)
    self.assertEqual(bm.get_device_to_measure(), expected[0])

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, (("jax", "jax"), ("tpu_sync", "tpu_sync")))
  )
  def test_out_of_bounds_device_id_raises_error(self, bm_cls, backend):
    """device_id >= len(jax.local_devices()) is rejected before allocation."""
    bm = self._make_benchmark(
        bm_cls, transfer_backend=backend, device_id=_NUM_DEVICES
    )
    with self.assertRaisesRegex(
        ValueError, "exceeds the number of local devices"
    ):
      bm.setup()
    # A failed setup() must not leave a KVCacheManager behind.
    self.mock_kv_cls.assert_not_called()
    self.assertIsNone(bm.kv_manager)

  # --- Buffer shape, sharding and mesh ----------------------------------

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_setup_builds_buffer_layout(self, bm_cls, backend, device_id):
    """jax uses a 2D row-sharded buffer; tpu_sync a block-major 3D buffer."""
    bm = self._setup_benchmark(
        bm_cls, transfer_backend=backend, device_id=device_id
    )
    n = _expected_num_devices(device_id)
    self.assertEqual(bm.rows_per_device, _ROWS_PER_DEVICE)
    if backend == "tpu_sync":
      self.assertEqual(bm.get_buffer_shape(), (1, n * _ROWS_PER_DEVICE, 128))
      self.assertEqual(bm.get_partition_spec(), _P(None, "device", None))
    else:
      self.assertEqual(bm.get_buffer_shape(), (n * _ROWS_PER_DEVICE, 128))
      self.assertEqual(bm.get_partition_spec(), _P("device", None))
    # setup() replaces the mesh with one spanning only the transfer devices.
    self.assertEqual(bm.mesh.axis_names, ("device",))
    self.assertEqual(list(bm.mesh.devices.flat), _expected_devices(device_id))
    self.assertIsInstance(bm.sharding, jax.sharding.NamedSharding)
    self.assertIs(bm.sharding.mesh, bm.mesh)
    self.assertEqual(bm.sharding.spec, bm.get_partition_spec())

  @parameterized.named_parameters(
      ("bfloat16", "bfloat16", 16384),
      ("float32", "float32", 8192),
      ("float8_e4m3fn", "float8_e4m3fn", 32768),
  )
  def test_rows_per_device_honors_dtype(self, dtype, expected_rows):
    """Rows per device shrink as the dtype itemsize grows."""
    bm = self._make_benchmark(_H2D, dtype=dtype, transfer_backend="tpu_sync")
    self.assertEqual(bm.rows_per_device, expected_rows)
    self.assertEqual(
        bm.get_buffer_shape(), (1, _NUM_DEVICES * expected_rows, 128)
    )

  # --- Byte accounting, metrics, run identifier --------------------------

  @parameterized.named_parameters(("all", -1), ("dev0", 0))
  def test_calculate_metrics(self, device_id):
    """Bandwidth is reported per device and per host, with num_devices."""
    bm = self._setup_benchmark(_H2D, device_id=device_id)
    n = _expected_num_devices(device_id)
    self.assertAlmostEqual(bm.get_total_bytes(), float(_DATA_SIZE_BYTES * n))

    metrics = bm.calculate_metrics([10.0, 10.0, 10.0])
    # Per-device bandwidth is 4194304 bytes / (0.01 s * 1e9) = 0.4194304 GB/s;
    # per-host bandwidth scales by the number of participating devices.
    self.assertAlmostEqual(metrics["wall_clock_p50_ms"], 10.0)
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], 0.4194304
    )
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_host_gb_s"], 0.4194304 * n
    )
    self.assertEqual(metrics["num_devices"], n)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", metrics)

    xprof = bm.calculate_throughput_metrics(1.0, constants.TimingDomain.XPROF)
    self.assertAlmostEqual(
        xprof["xprof_bandwidth_per_device_gb_s"], _DATA_SIZE_BYTES / 1e6
    )
    self.assertAlmostEqual(
        xprof["xprof_bandwidth_per_host_gb_s"], _DATA_SIZE_BYTES / 1e6 * n
    )

  def test_zero_latency_gives_infinite_bandwidth(self):
    bm = self._make_benchmark(_H2D)
    metrics = bm.calculate_throughput_metrics(
        0.0, constants.TimingDomain.WALL_CLOCK
    )
    self.assertEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], float("inf")
    )
    self.assertEqual(
        metrics["wall_clock_bandwidth_per_host_gb_s"], float("inf")
    )

  @parameterized.named_parameters(
      ("jax_all", "jax", -1, "jax_devall_size_4mib"),
      ("jax_dev3", "jax", 3, "jax_dev3_size_4mib"),
      ("tpu_sync_all", "tpu_sync", -1, "tpu_sync_devall_size_4mib"),
      ("tpu_sync_dev0", "tpu_sync", 0, "tpu_sync_dev0_size_4mib"),
  )
  def test_get_run_identifier(self, backend, device_id, expected):
    """Run identifiers encode backend, device selection, and payload size."""
    bm = self._make_benchmark(
        _H2D, transfer_backend=backend, device_id=device_id
    )
    self.assertEqual(bm.get_run_identifier(), expected)

  # --- Report schema ------------------------------------------------------

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_schema_coverage(self, bm_cls, backend, device_id):
    """Verify REPORT_SCHEMA matches output keys and covers all metrics."""
    bm = self._setup_benchmark(
        bm_cls, transfer_backend=backend, device_id=device_id
    )
    test_report_utils.assert_schema_matches_output(self, bm)

  @parameterized.named_parameters(
      ("h2d_tpu_sync_dev0", _H2D, "tpu_sync", 0, 1),
      ("d2h_jax_all", _D2H, "jax", -1, 8),
  )
  def test_format_benchmark_table(
      self, bm_cls, backend, device_id, num_devices
  ):
    """Tests formatting of host<->device benchmark tables."""
    name = bm_cls.__name__
    res = base.BenchmarkResult(
        metadata=base.BenchmarkMetadata(
            benchmark_name=name,
            test_name=f"{name}_test",
            start_time="2026-08-18T10:00:00",
            end_time="2026-08-18T10:01:00",
            params={
                "dtype": "float32",
                "data_size_mib": 256,
                "transfer_backend": backend,
                "device_id": device_id,
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "num_devices": num_devices,
            "wall_clock_bandwidth_per_device_gb_s": 18.25,
            "wall_clock_bandwidth_per_host_gb_s": 18.25 * num_devices,
            "xprof_bandwidth_per_device_gb_s": 18.50,
            "xprof_bandwidth_per_host_gb_s": 18.50 * num_devices,
            "wall_clock_p50_ms": 14.0274,
            "xprof_p50_ms": 14.0100,
        },
        raw_times_ms=[1.0],
    )
    df = report.results_to_dataframe([res])
    table = report.format_benchmark_table(
        df, schema=bm_cls.REPORT_SCHEMA, title=name
    )
    self.assertIn(f"Benchmark Results ({name})", table)
    for col, _ in bm_cls.REPORT_SCHEMA:
      self.assertIn(col, table)
    for value in (
        "float32",
        backend,
        str(device_id),
        "256",
        "18.25",
        f"{18.25 * num_devices:.2f}",
        "18.50",
        f"{18.50 * num_devices:.2f}",
        "14.0274",
        "14.0100",
    ):
      self.assertIn(value, table)

  # --- jax backend generate_inputs / run_op -------------------------------

  @parameterized.named_parameters(
      *_cross(
          _BENCHMARK_CLASSES,
          (
              ("bfloat16_all", "bfloat16", jnp.bfloat16, -1),
              ("float32_dev2", "float32", jnp.float32, 2),
              ("int8_all", "int8", jnp.int8, -1),
              ("float8_e5m2_dev0", "float8_e5m2", jnp.float8_e5m2, 0),
          ),
      )
  )
  def test_jax_generate_inputs_and_run_op(
      self, bm_cls, dtype, jnp_dtype, device_id
  ):
    """Inputs and outputs honor dtype, payload size, and device selection."""
    bm = self._setup_benchmark(
        bm_cls, dtype=dtype, transfer_backend="jax", device_id=device_id
    )
    n = _expected_num_devices(device_id)
    expected_dtype = jnp.dtype(jnp_dtype)
    expected_shape = (
        n * _DATA_SIZE_BYTES // expected_dtype.itemsize // 128,
        128,
    )
    (inp,) = bm.generate_inputs()
    out = bm.run_op(inp)
    for arr in (inp, out):
      self.assertEqual(arr.dtype, expected_dtype)
      self.assertEqual(arr.shape, expected_shape)
      self.assertEqual(arr.nbytes, bm.get_total_bytes())
    # The host side is a NumPy array; the device side is row-sharded.
    host_arr, device_arr = (inp, out) if bm_cls is _H2D else (out, inp)
    self.assertIsInstance(host_arr, np.ndarray)
    self._assert_row_sharded(device_arr, device_id)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_jax_run_op_preserves_data(self, bm_cls):
    bm = self._setup_benchmark(bm_cls, transfer_backend="jax")
    (inp,) = bm.generate_inputs()
    expected_bits = np.asarray(inp).view(np.uint16)
    out = bm.run_op(inp)
    # Compare raw bits: random payloads contain NaN bit patterns.
    np.testing.assert_array_equal(
        np.asarray(out).view(np.uint16), expected_bits
    )

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_invalid_dtype_raises_error(self, bm_cls):
    bm = self._make_benchmark(
        bm_cls, dtype="invalid_dtype", transfer_backend="jax"
    )
    with self.assertRaisesRegex(
        ValueError, "Invalid dtype string: 'invalid_dtype'"
    ):
      bm.setup()
      bm.generate_inputs()

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_jax_run_op_inside_marker(self, bm_cls):
    bm = self._setup_benchmark(bm_cls, transfer_backend="jax")
    inputs = bm.generate_inputs()
    events = []
    with mock.patch.object(
        jax.profiler, "TraceAnnotation", new=_RecordingTraceAnnotation(events)
    ):
      bm.run_op(*inputs)
    self.assertEqual(
        events, [f"enter:{constants.MARKER}", f"exit:{constants.MARKER}"]
    )

  # --- tpu_sync KVCacheManager lifecycle ----------------------------------

  @parameterized.named_parameters(
      ("h2d_all", _H2D, -1, True),
      ("h2d_dev3", _H2D, 3, True),
      ("d2h_all", _D2H, -1, False),
      ("d2h_dev3", _D2H, 3, False),
  )
  def test_tpu_sync_setup_creates_block_major_kv_manager(
      self, bm_cls, device_id, primes_host_staging
  ):
    """setup() allocates one KVCacheManager whose shards are single blocks."""
    bm = self._setup_benchmark(
        bm_cls, transfer_backend="tpu_sync", device_id=device_id
    )
    self.mock_kv_cls.assert_called_once_with(
        kv_caches=[bm.device_array],
        local_control_port=0,
        host_blocks_to_allocate=1,
        unsafe_skip_buffer_lock=True,
    )
    self.assertIs(bm.kv_manager, self.mock_kv)

    n = _expected_num_devices(device_id)
    kv_cache = bm.device_array
    self.assertIsInstance(kv_cache, jax.Array)
    self.assertEqual(kv_cache.shape, (1, n * _ROWS_PER_DEVICE, 128))
    self.assertEqual(kv_cache.dtype, jnp.bfloat16)
    self.assertEqual(kv_cache.sharding.spec, _P(None, "device", None))
    self.assertEqual(kv_cache.devices(), set(_expected_devices(device_id)))
    self.assertLen(kv_cache.addressable_shards, n)
    for shard in kv_cache.addressable_shards:
      # Axis 0 is the KVCacheManager block axis: one block per shard.
      self.assertEqual(shard.data.shape, (1, _ROWS_PER_DEVICE, 128))

    # H2D fills the pinned host staging buffer with a D2H before timing.
    self.mock_kv.h2d.assert_not_called()
    if primes_host_staging:
      self.mock_kv.d2h.assert_called_once_with([0], [0], [1])
      self.mock_kv.d2h.return_value.Await.assert_called_once_with()
    else:
      self.mock_kv.d2h.assert_not_called()

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_jax_backend_does_not_create_kv_manager(self, bm_cls):
    bm = self._setup_benchmark(bm_cls, transfer_backend="jax")
    bm.run_op(*bm.generate_inputs())
    self.mock_kv_cls.assert_not_called()
    self.assertIsNone(bm.kv_manager)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_tpu_sync_run_op_before_setup_raises_error(self, bm_cls):
    bm = self._make_benchmark(bm_cls, transfer_backend="tpu_sync")
    with self.assertRaisesRegex(
        ValueError, "KVCacheManager is not initialized"
    ):
      bm.run_op(None)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_tpu_sync_generate_inputs_returns_kv_buffer(self, bm_cls):
    bm = self._setup_benchmark(bm_cls, transfer_backend="tpu_sync")
    (inp,) = bm.generate_inputs()
    self.assertIs(inp, bm.device_array)
    # Inputs are stable across calls; the KVCacheManager buffer is reused.
    self.assertIs(bm.generate_inputs()[0], bm.device_array)

  @parameterized.named_parameters(
      ("h2d", _H2D, "h2d", "d2h", True),
      ("d2h", _D2H, "d2h", "h2d", False),
  )
  def test_tpu_sync_run_op(self, bm_cls, op_name, other_op, returns_buffer):
    """One full-block transfer is issued and awaited inside the MARKER."""
    bm = self._setup_benchmark(bm_cls, transfer_backend="tpu_sync")
    inputs = bm.generate_inputs()
    self.mock_kv.reset_mock()
    events = []
    op = getattr(self.mock_kv, op_name)
    op.side_effect = lambda *a: events.append(op_name) or op.return_value
    op.return_value.Await.side_effect = lambda: events.append("await")
    with (
        mock.patch.object(
            jax.profiler,
            "TraceAnnotation",
            new=_RecordingTraceAnnotation(events),
        ),
        mock.patch.object(jax, "device_put") as mock_device_put,
        mock.patch.object(jax, "device_get") as mock_device_get,
    ):
      out = bm.run_op(*inputs)
    op.assert_called_once_with([0], [0], [1])
    getattr(self.mock_kv, other_op).assert_not_called()
    self.assertEqual(
        events,
        [
            f"enter:{constants.MARKER}",
            op_name,
            "await",
            f"exit:{constants.MARKER}",
        ],
    )
    mock_device_put.assert_not_called()
    mock_device_get.assert_not_called()
    self.assertIs(out, bm.device_array if returns_buffer else None)

  # --- teardown -----------------------------------------------------------

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_teardown_releases_kv_manager_and_buffer(self, bm_cls):
    bm = self._setup_benchmark(bm_cls, transfer_backend="tpu_sync")
    device_array = bm.device_array
    self.assertIsNotNone(bm.kv_manager)
    bm.teardown()
    self.assertIsNone(bm.kv_manager)
    self.assertIsNone(bm.device_array)
    self.assertTrue(device_array.is_deleted())

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_teardown_is_idempotent_and_safe_before_setup(
      self, bm_cls, backend, device_id
  ):
    """teardown() may be called before setup() and more than once."""
    bm = self._make_benchmark(
        bm_cls, transfer_backend=backend, device_id=device_id
    )
    bm.teardown()  # Before setup().
    bm.setup()
    bm.teardown()
    bm.teardown()  # Twice.
    self.assertIsNone(bm.kv_manager)
    self.assertIsNone(bm.device_array)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_teardown_tolerates_already_deleted_buffer(self, bm_cls):
    bm = self._setup_benchmark(bm_cls, transfer_backend="tpu_sync")
    bm.device_array.delete()
    bm.teardown()
    self.assertIsNone(bm.device_array)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_teardown_propagates_delete_error_and_clears_state(self, bm_cls):
    """A failing buffer delete is surfaced, but references are still dropped."""
    bm = self._setup_benchmark(bm_cls, transfer_backend="tpu_sync")
    bm.device_array.delete()
    bad_array = mock.MagicMock()
    bad_array.is_deleted.return_value = False
    bad_array.delete.side_effect = RuntimeError("delete failed")
    bm.device_array = bad_array
    with self.assertRaisesRegex(RuntimeError, "delete failed"):
      bm.teardown()
    bad_array.delete.assert_called_once()
    self.assertIsNone(bm.kv_manager)
    self.assertIsNone(bm.device_array)

  # --- run() --------------------------------------------------------------

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_run_calls_teardown(self, bm_cls, backend, device_id):
    """run() releases benchmark resources exactly once after measurement."""
    bm = self._make_benchmark(
        bm_cls,
        transfer_backend=backend,
        device_id=device_id,
        num_runs=2,
        warmup_tries=1,
    )
    with mock.patch.object(bm, "teardown", wraps=bm.teardown) as mock_teardown:
      bm.run()
    self.assertEqual(mock_teardown.call_count, 1)
    self.assertIsNone(bm.kv_manager)
    self.assertIsNone(bm.device_array)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_run_tears_down_when_measurement_raises(self, bm_cls):
    """run() releases the KVCacheManager even when a transfer fails."""
    bm = self._make_benchmark(
        bm_cls, transfer_backend="tpu_sync", num_runs=2, warmup_tries=1
    )
    getattr(self.mock_kv, _MEASURED_OP[bm_cls]).side_effect = RuntimeError(
        "DMA failed"
    )
    with self.assertRaisesRegex(RuntimeError, "DMA failed"):
      bm.run()
    self.assertIsNone(bm.kv_manager)
    self.assertIsNone(bm.device_array)

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_run_e2e(self, bm_cls, backend, device_id):
    """run() reports per-device and per-host bandwidth, never per-chip."""
    bm = self._make_benchmark(
        bm_cls,
        transfer_backend=backend,
        device_id=device_id,
        num_runs=3,
        warmup_tries=2,
    )
    result = bm.run()
    n = _expected_num_devices(device_id)
    metrics = result.metrics
    self.assertGreater(metrics["wall_clock_bandwidth_per_device_gb_s"], 0.0)
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_host_gb_s"],
        metrics["wall_clock_bandwidth_per_device_gb_s"] * n,
    )
    self.assertEqual(metrics["num_devices"], n)
    self.assertEqual(metrics["actual_runs"], 3)
    for key in (
        "wall_clock_bandwidth_per_chip_gb_s",
        "roofline_tflops_limit",
        "compute_roofline_efficiency_pct",
        "peak_hbm_bw_gb_s",
        "memory_roofline_efficiency_pct",
    ):
      self.assertNotIn(key, metrics)
    self.assertEqual(result.metadata.params["transfer_backend"], backend)
    self.assertEqual(result.metadata.params["device_id"], device_id)
    if backend == "tpu_sync":
      measured = getattr(self.mock_kv, _MEASURED_OP[bm_cls])
      # At least warmup_tries + num_runs transfers, all of the full block.
      self.assertGreaterEqual(measured.call_count, 5)
      for call in measured.call_args_list:
        self.assertEqual(call, mock.call([0], [0], [1]))
    else:
      self.mock_kv_cls.assert_not_called()


class DeviceToHostBenchmarkTest(_HostDeviceTestBase):
  """Unit tests specific to DeviceToHostBenchmark."""

  _BM = _D2H

  def test_jax_generate_inputs_before_setup_raises_error(self):
    bm = self._make_benchmark(self._BM, transfer_backend="jax")
    with self.assertRaisesRegex(ValueError, r"setup\(\) must be called"):
      bm.generate_inputs()

  def test_jax_generate_inputs_without_sharding_raises_error(self):
    bm = self._make_benchmark(self._BM, transfer_backend="jax")
    # A host payload alone is not enough; the sharding is built in setup().
    bm.host_data = np.zeros((_ROWS_PER_DEVICE, 128), dtype=jnp.bfloat16)
    with self.assertRaisesRegex(
        ValueError, r"setup\(\) must be called before creating device arrays"
    ):
      bm.generate_inputs()

  def test_jax_reset_data_replaces_and_deletes_array(self):
    """jax mode must transfer a fresh array so device_get is never cached."""
    bm = self._setup_benchmark(self._BM, transfer_backend="jax")
    (old_array,) = bm.generate_inputs()
    bm.run_op(old_array)  # Populates the host-value cache of old_array.
    (new_array,) = bm.reset_data(old_array)
    self.assertTrue(old_array.is_deleted())
    self.assertIsNot(new_array, old_array)
    self.assertFalse(new_array.is_deleted())
    self.assertEqual(new_array.shape, old_array.shape)
    self.assertEqual(new_array.sharding.spec, _P("device", None))

  def test_jax_reset_data_without_input_generates_array(self):
    bm = self._setup_benchmark(self._BM, transfer_backend="jax")
    (new_array,) = bm.reset_data(None)
    self.assertIsInstance(new_array, jax.Array)
    self.assertFalse(new_array.is_deleted())

  def test_tpu_sync_reset_data_is_noop(self):
    """KVCacheManager.d2h bypasses the host-value cache; reuse the buffer."""
    bm = self._setup_benchmark(self._BM, transfer_backend="tpu_sync")
    inputs = bm.generate_inputs()
    with mock.patch.object(
        bm, "generate_inputs", wraps=bm.generate_inputs
    ) as mock_generate:
      new_inputs = bm.reset_data(*inputs)
    self.assertEqual(tuple(new_inputs), tuple(inputs))
    self.assertIs(new_inputs[0], bm.device_array)
    self.assertFalse(bm.device_array.is_deleted())
    mock_generate.assert_not_called()

  def test_tpu_sync_run_keeps_kv_buffer_until_teardown(self):
    """reset_data must never delete the buffer registered with KVCacheManager."""
    bm = self._make_benchmark(
        self._BM, transfer_backend="tpu_sync", num_runs=3, warmup_tries=1
    )
    seen = []
    original_run_op = bm.run_op

    def recording_run_op(*args, **kwargs):
      arr = args[0]
      seen.append(arr)
      self.assertFalse(arr.is_deleted())
      return original_run_op(*args, **kwargs)

    bm.run_op = recording_run_op
    bm.run()
    self.assertGreaterEqual(len(seen), 4)
    # Every measured transfer used the same registered buffer.
    self.assertTrue(all(arr is seen[0] for arr in seen))

  def test_teardown_releases_host_data(self):
    bm = self._setup_benchmark(self._BM, transfer_backend="jax")
    self.assertIsNotNone(bm.host_data)
    bm.teardown()
    self.assertIsNone(bm.host_data)


if __name__ == "__main__":
  absltest.main()
