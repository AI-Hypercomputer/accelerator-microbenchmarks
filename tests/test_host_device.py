"""Unit tests for host_device.py."""

import contextlib
import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import host_device
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
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

# 4 MiB of bfloat16 is 2Mi elements, i.e. 16384 rows of 128 columns per device.
_DATA_SIZE_MIB = 4
_DATA_SIZE_BYTES = _DATA_SIZE_MIB * 1024 * 1024
_ROWS_PER_DEVICE = 16384

_PLATFORM_INFO_TARGET = (
    "accelerator_microbenchmarks.core.platform"
    ".get_platform_info"
)

_BENCHMARK_CLASSES = (
    ("h2d", host_device.HostToDeviceBenchmark),
    ("d2h", host_device.DeviceToHostBenchmark),
)

_BACKENDS_AND_DEVICES = (
    ("jax_all", "jax", -1),
    ("jax_dev0", "jax", 0),
    ("tpu_sync_all", "tpu_sync", -1),
    ("tpu_sync_dev0", "tpu_sync", 0),
)


def _expected_num_devices(local_device: int) -> int:
  return _NUM_DEVICES if local_device == -1 else 1


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
        mock.patch(
            _PLATFORM_INFO_TARGET,
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


class HostDeviceParamsValidationTest(parameterized.TestCase):
  """Verifies the bounds and enums declared on HostDeviceParams fields."""

  @parameterized.parameters(0, -10)
  def test_non_positive_data_size_raises_error(self, data_size_mib):
    with self.assertRaisesRegex(ValueError, "data_size_mib must be >= 1"):
      host_device.HostDeviceParams(data_size_mib=data_size_mib)

  def test_defaults(self):
    params = host_device.HostDeviceParams()
    self.assertEqual(params.transfer_backend, _JAX)
    self.assertEqual(params.local_device, 0)
    self.assertEqual(host_device.ALL_LOCAL_DEVICES, -1)
    self.assertEqual(params.data_size_mib, 64)
    self.assertEqual(params.data_size_bytes, 64 * 1024 * 1024)

  @parameterized.named_parameters(
      ("jax_str", "jax", _JAX),
      ("tpu_sync_str", "tpu_sync", _TPU_SYNC),
      ("jax_enum", _JAX, _JAX),
      ("tpu_sync_enum", _TPU_SYNC, _TPU_SYNC),
  )
  def test_transfer_backend_coerced_to_enum(self, value, expected):
    params = host_device.HostDeviceParams(transfer_backend=value)
    self.assertIsInstance(params.transfer_backend, host_device.TransferBackend)
    self.assertEqual(params.transfer_backend, expected)

  @parameterized.parameters("cuda", "TPU_SYNC", "tpu-sync", "", None)
  def test_invalid_transfer_backend_raises_error(self, value):
    with self.assertRaisesRegex(
        ValueError,
        rf"Invalid transfer_backend '{value}'\. Must be one of \['jax',"
        r" 'tpu_sync'\]",
    ):
      host_device.HostDeviceParams(transfer_backend=value)

  def test_transfer_backend_values(self):
    self.assertEqual(
        [member.value for member in host_device.TransferBackend],
        ["jax", "tpu_sync"],
    )

  @parameterized.parameters(-1, 0, 1, 3, 1000)
  def test_valid_local_device_accepted_by_params(self, local_device):
    # Upper bound depends on the runtime device count and is checked in
    # setup(); only the lower bound is a static params constraint.
    params = host_device.HostDeviceParams(local_device=local_device)
    self.assertEqual(params.local_device, local_device)

  @parameterized.parameters(-2, -100)
  def test_local_device_below_minus_one_raises_error(self, local_device):
    with self.assertRaisesRegex(ValueError, "local_device must be >= -1"):
      host_device.HostDeviceParams(local_device=local_device)

  @parameterized.parameters(None, True, False, 0.0, "0")
  def test_non_int_local_device_raises_error(self, local_device):
    with self.assertRaisesRegex(
        ValueError, "local_device must be an integer >= -1"
    ):
      host_device.HostDeviceParams(local_device=local_device)

  @parameterized.parameters(None, True, False, 4.0, "4")
  def test_non_int_data_size_raises_error(self, data_size_mib):
    with self.assertRaisesRegex(
        ValueError, "data_size_mib must be an integer >= 1"
    ):
      host_device.HostDeviceParams(data_size_mib=data_size_mib)

  def test_numpy_int_params_normalized_to_int(self):
    params = host_device.HostDeviceParams(
        data_size_mib=np.int64(8), local_device=np.int32(0)
    )
    self.assertIs(type(params.data_size_mib), int)
    self.assertIs(type(params.local_device), int)
    self.assertEqual(params.data_size_mib, 8)
    self.assertEqual(params.local_device, 0)

  def test_numpy_int_bounds_still_enforced(self):
    with self.assertRaisesRegex(ValueError, "local_device must be >= -1"):
      host_device.HostDeviceParams(local_device=np.int64(-2))


class HostDeviceCommonTest(_HostDeviceTestBase):
  """Behavior shared by HostToDeviceBenchmark and DeviceToHostBenchmark."""

  def test_cpu_device_count(self):
    # Guards the multi-device assumptions of every test in this file.
    self.assertLen(jax.local_devices(), _NUM_DEVICES)

  @parameterized.named_parameters(
      ("h2d", "host_to_device", host_device.HostToDeviceBenchmark),
      ("d2h", "device_to_host", host_device.DeviceToHostBenchmark),
  )
  def test_benchmark_registered(self, name, bm_cls):
    self.assertEqual(registry.benchmark_registry.get_benchmark(name), bm_cls)

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, _BACKENDS_AND_DEVICES)
  )
  def test_xprof_target_host_cpu(self, bm_cls, backend, local_device):
    bm = self._make_benchmark(
        bm_cls, transfer_backend=backend, local_device=local_device
    )
    self.assertTrue(bm.xprof_target_host_cpu)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_no_chip_bandwidth_derivation(self, bm_cls):
    self.assertFalse(bm_cls.derive_chip_bandwidth)
    bm = self._make_benchmark(bm_cls)
    derived = bm.derive_chip_metrics(
        {"wall_clock_bandwidth_per_device_gb_s": 10.0}
    )
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", derived)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", derived)

  # --- local_device selection -------------------------------------------

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_all_local_devices(self, bm_cls):
    bm = self._make_benchmark(bm_cls, local_device=-1)
    self.assertEqual(bm.get_transfer_devices(), list(jax.local_devices()))
    self.assertEqual(bm.num_transfer_devices, _NUM_DEVICES)
    self.assertEqual(bm.get_device_to_measure(), jax.local_devices()[0])

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, [(f"dev{i}", i) for i in range(_NUM_DEVICES)])
  )
  def test_single_local_device(self, bm_cls, local_device):
    bm = self._make_benchmark(bm_cls, local_device=local_device)
    expected = jax.local_devices()[local_device]
    self.assertEqual(bm.get_transfer_devices(), [expected])
    self.assertEqual(bm.num_transfer_devices, 1)
    self.assertEqual(bm.get_device_to_measure(), expected)

  @parameterized.named_parameters(
      *_cross(
          _BENCHMARK_CLASSES,
          (("jax", "jax"), ("tpu_sync", "tpu_sync")),
          (("n", _NUM_DEVICES), ("n_plus_10", _NUM_DEVICES + 10)),
      )
  )
  def test_out_of_bounds_local_device_raises_error(
      self, bm_cls, backend, local_device
  ):
    """local_device >= len(jax.local_devices()) is rejected before allocation."""
    bm = self._make_benchmark(
        bm_cls, transfer_backend=backend, local_device=local_device
    )
    with self.assertRaisesRegex(
        ValueError, "exceeds the number of local devices"
    ):
      bm.get_transfer_devices()
    with self.assertRaisesRegex(
        ValueError, "exceeds the number of local devices"
    ):
      bm.setup()
    # A failed setup() must not leave a KVCacheManager behind.
    self.mock_kv_cls.assert_not_called()
    self.assertIsNone(bm.kv_manager)

  # --- Buffer shape, sharding and mesh ----------------------------------

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, _BACKENDS_AND_DEVICES)
  )
  def test_buffer_shape_and_partition_spec(self, bm_cls, backend, local_device):
    """jax uses a 2D row-sharded buffer; tpu_sync a block-major 3D buffer."""
    bm = self._make_benchmark(
        bm_cls, transfer_backend=backend, local_device=local_device
    )
    n = _expected_num_devices(local_device)
    self.assertEqual(bm.rows_per_device, _ROWS_PER_DEVICE)
    if backend == "tpu_sync":
      self.assertEqual(bm.get_buffer_shape(), (1, n * _ROWS_PER_DEVICE, 128))
      self.assertEqual(bm.get_partition_spec(), _P(None, "device", None))
    else:
      self.assertEqual(bm.get_buffer_shape(), (n * _ROWS_PER_DEVICE, 128))
      self.assertEqual(bm.get_partition_spec(), _P("device", None))

  @parameterized.named_parameters(
      ("bfloat16", "bfloat16", 16384),
      ("float32", "float32", 8192),
      ("int8", "int8", 32768),
      ("float8_e4m3fn", "float8_e4m3fn", 32768),
  )
  def test_rows_per_device_honors_dtype(self, dtype, expected_rows):
    """Rows per device shrink as the dtype itemsize grows."""
    bm = self._make_benchmark(
        host_device.HostToDeviceBenchmark,
        dtype=dtype,
        transfer_backend="tpu_sync",
        local_device=-1,
    )
    self.assertEqual(bm.rows_per_device, expected_rows)
    self.assertEqual(
        bm.get_buffer_shape(), (1, _NUM_DEVICES * expected_rows, 128)
    )

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, _BACKENDS_AND_DEVICES)
  )
  def test_setup_builds_mesh_over_transfer_devices(
      self, bm_cls, backend, local_device
  ):
    """setup() replaces the mesh with one spanning only the transfer devices."""
    bm = self._setup_benchmark(
        bm_cls, transfer_backend=backend, local_device=local_device
    )
    expected_devices = (
        list(jax.local_devices())
        if local_device == -1
        else [jax.local_devices()[local_device]]
    )
    self.assertEqual(bm.mesh.axis_names, ("device",))
    self.assertEqual(list(bm.mesh.devices.flat), expected_devices)
    self.assertIsInstance(bm.sharding, jax.sharding.NamedSharding)
    self.assertIs(bm.sharding.mesh, bm.mesh)
    self.assertEqual(bm.sharding.spec, bm.get_partition_spec())

  # --- Byte accounting, metrics, run identifier --------------------------

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, _BACKENDS_AND_DEVICES)
  )
  def test_get_total_bytes(self, bm_cls, backend, local_device):
    bm = self._setup_benchmark(
        bm_cls, transfer_backend=backend, local_device=local_device
    )
    self.assertAlmostEqual(
        bm.get_total_bytes(),
        float(_DATA_SIZE_BYTES * _expected_num_devices(local_device)),
    )

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, _BACKENDS_AND_DEVICES)
  )
  def test_calculate_metrics(self, bm_cls, backend, local_device):
    """Bandwidth is reported per device and in total, with num_devices metadata."""
    bm = self._setup_benchmark(
        bm_cls, transfer_backend=backend, local_device=local_device
    )
    metrics = bm.calculate_metrics([10.0, 10.0, 10.0])
    expected_num_devices = _expected_num_devices(local_device)
    # Per-device bandwidth is 4194304 bytes / (0.01 s * 1e9) = 0.4194304 GB/s;
    # total bandwidth scales by the number of participating devices.
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)
    self.assertAlmostEqual(metrics["wall_clock_p50_ms"], 10.0)
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], 0.4194304
    )
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_total_gb_s"],
        0.4194304 * expected_num_devices,
    )
    self.assertEqual(metrics["num_devices"], expected_num_devices)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", metrics)
    self.assertNotIn("total_bytes_mib", metrics)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_zero_latency_gives_infinite_bandwidth(self, bm_cls):
    bm = self._make_benchmark(bm_cls)
    metrics = bm.calculate_throughput_metrics(
        0.0, constants.TimingDomain.WALL_CLOCK
    )
    self.assertEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], float("inf")
    )
    self.assertEqual(metrics["wall_clock_bandwidth_total_gb_s"], float("inf"))

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, (("all", -1), ("dev0", 0)))
  )
  def test_xprof_prefix_bandwidth(self, bm_cls, local_device):
    bm = self._make_benchmark(bm_cls, local_device=local_device)
    metrics = bm.calculate_throughput_metrics(1.0, constants.TimingDomain.XPROF)
    expected_per_device = _DATA_SIZE_BYTES / 1e6
    self.assertAlmostEqual(
        metrics["xprof_bandwidth_per_device_gb_s"], expected_per_device
    )
    self.assertAlmostEqual(
        metrics["xprof_bandwidth_total_gb_s"],
        expected_per_device * _expected_num_devices(local_device),
    )

  @parameterized.named_parameters(
      *_cross(
          _BENCHMARK_CLASSES,
          (
              ("jax_all", "jax", -1, "jax_devall_size_4mib"),
              ("jax_dev0", "jax", 0, "jax_dev0_size_4mib"),
              ("jax_dev3", "jax", 3, "jax_dev3_size_4mib"),
              ("tpu_sync_all", "tpu_sync", -1, "tpu_sync_devall_size_4mib"),
              ("tpu_sync_dev0", "tpu_sync", 0, "tpu_sync_dev0_size_4mib"),
          ),
      )
  )
  def test_get_run_identifier(self, bm_cls, backend, local_device, expected):
    """Run identifiers encode backend, device selection, and payload size."""
    bm = self._make_benchmark(
        bm_cls, transfer_backend=backend, local_device=local_device
    )
    self.assertEqual(bm.get_run_identifier(), expected)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_run_identifier_unique_across_sweep(self, bm_cls):
    identifiers = {
        self._make_benchmark(
            bm_cls, transfer_backend=backend, local_device=local_device
        ).get_run_identifier()
        for _, backend, local_device in _BACKENDS_AND_DEVICES
    }
    self.assertLen(identifiers, len(_BACKENDS_AND_DEVICES))

  # --- Report schema ------------------------------------------------------

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_report_schema_columns(self, bm_cls):
    self.assertEqual(
        [col for col, _ in bm_cls.REPORT_SCHEMA],
        [
            "dtype",
            "transfer_backend",
            "local_device",
            "num_devices",
            "data_size_mib",
            "wall_clock_p50_ms",
            "wall_clock_bandwidth_per_device_gb_s",
            "wall_clock_bandwidth_total_gb_s",
            "xprof_p50_ms",
            "xprof_bandwidth_per_device_gb_s",
            "xprof_bandwidth_total_gb_s",
        ],
    )

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, _BACKENDS_AND_DEVICES)
  )
  def test_schema_coverage(self, bm_cls, backend, local_device):
    """Verify REPORT_SCHEMA matches output keys and covers all metrics."""
    bm = self._setup_benchmark(
        bm_cls, transfer_backend=backend, local_device=local_device
    )
    test_report_utils.assert_schema_matches_output(self, bm)

  @parameterized.named_parameters(
      (
          "h2d_tpu_sync_dev0",
          host_device.HostToDeviceBenchmark,
          "tpu_sync",
          0,
          1,
      ),
      (
          "d2h_jax_all",
          host_device.DeviceToHostBenchmark,
          "jax",
          -1,
          8,
      ),
  )
  def test_format_benchmark_table(
      self, bm_cls, backend, local_device, num_devices
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
                "local_device": local_device,
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "num_devices": num_devices,
            "wall_clock_bandwidth_per_device_gb_s": 18.25,
            "wall_clock_bandwidth_total_gb_s": 18.25 * num_devices,
            "xprof_bandwidth_per_device_gb_s": 18.50,
            "xprof_bandwidth_total_gb_s": 18.50 * num_devices,
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
        str(local_device),
        "256",
        "18.25",
        f"{18.25 * num_devices:.2f}",
        "18.50",
        f"{18.50 * num_devices:.2f}",
        "14.0274",
        "14.0100",
    ):
      self.assertIn(value, table)

  # --- tpu_sync KVCacheManager lifecycle ----------------------------------

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, (("all", -1), ("dev0", 0), ("dev2", 2)))
  )
  def test_tpu_sync_setup_creates_kv_manager(self, bm_cls, local_device):
    """setup() pre-allocates one KVCacheManager with a single host block."""
    bm = self._setup_benchmark(
        bm_cls, transfer_backend="tpu_sync", local_device=local_device
    )
    self.mock_kv_cls.assert_called_once_with(
        kv_caches=[bm.device_array],
        local_control_port=0,
        host_blocks_to_allocate=1,
        unsafe_skip_buffer_lock=True,
    )
    self.assertIs(bm.kv_manager, self.mock_kv)

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, (("all", -1), ("dev0", 0), ("dev3", 3)))
  )
  def test_tpu_sync_kv_cache_is_block_major(self, bm_cls, local_device):
    """Each device shard is a single (1, rows, 128) block."""
    bm = self._setup_benchmark(
        bm_cls,
        transfer_backend="tpu_sync",
        local_device=local_device,
    )
    n = _expected_num_devices(local_device)
    (kv_cache,) = self.mock_kv_cls.call_args.kwargs["kv_caches"]
    self.assertIsInstance(kv_cache, jax.Array)
    self.assertEqual(kv_cache.shape, (1, n * _ROWS_PER_DEVICE, 128))
    self.assertEqual(kv_cache.dtype, jnp.bfloat16)
    self.assertEqual(kv_cache.nbytes, n * _DATA_SIZE_BYTES)
    self.assertEqual(kv_cache.sharding.spec, _P(None, "device", None))
    self.assertEqual(kv_cache.devices(), set(bm.get_transfer_devices()))
    shards = kv_cache.addressable_shards
    self.assertLen(shards, n)
    for shard in shards:
      self.assertEqual(shard.data.shape, (1, _ROWS_PER_DEVICE, 128))
      # Axis 0 is the KVCacheManager block axis: one block per shard.
      self.assertEqual(
          shard.data.nbytes // shard.data.shape[0], _DATA_SIZE_BYTES
      )

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
    inputs = bm.generate_inputs()
    self.assertLen(inputs, 1)
    self.assertIs(inputs[0], bm.device_array)
    # Inputs are stable across calls; the KVCacheManager buffer is reused.
    self.assertIs(bm.generate_inputs()[0], bm.device_array)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_teardown_releases_kv_manager_and_buffer(self, bm_cls):
    bm = self._setup_benchmark(bm_cls, transfer_backend="tpu_sync")
    device_array = bm.device_array
    self.assertIsNotNone(bm.kv_manager)
    bm.teardown()
    self.assertIsNone(bm.kv_manager)
    self.assertIsNone(bm.device_array)
    self.assertTrue(device_array.is_deleted())

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, _BACKENDS_AND_DEVICES)
  )
  def test_teardown_is_idempotent_and_safe_before_setup(
      self, bm_cls, backend, local_device
  ):
    """teardown() may be called before setup() and more than once."""
    bm = self._make_benchmark(
        bm_cls, transfer_backend=backend, local_device=local_device
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
  def test_teardown_tolerates_failing_buffer_delete(self, bm_cls):
    bm = self._setup_benchmark(bm_cls, transfer_backend="tpu_sync")
    bm.device_array.delete()
    bad_array = mock.MagicMock()
    bad_array.is_deleted.return_value = False
    bad_array.delete.side_effect = RuntimeError("delete failed")
    bm.device_array = bad_array
    bm.teardown()  # Must not raise.
    bad_array.delete.assert_called_once()
    self.assertIsNone(bm.kv_manager)
    self.assertIsNone(bm.device_array)

  @parameterized.named_parameters(*_BENCHMARK_CLASSES)
  def test_repeated_setup_releases_previous_resources(self, bm_cls):
    """run() calls setup() again; the first staging pool must be dropped."""
    first_kv, second_kv = mock.MagicMock(), mock.MagicMock()
    self.mock_kv_cls.side_effect = [first_kv, second_kv]
    bm = self._setup_benchmark(bm_cls, transfer_backend="tpu_sync")
    first_array = bm.device_array
    self.assertIs(bm.kv_manager, first_kv)

    bm.setup()

    self.assertEqual(self.mock_kv_cls.call_count, 2)
    self.assertIs(bm.kv_manager, second_kv)
    self.assertTrue(first_array.is_deleted())
    self.assertIsNot(bm.device_array, first_array)
    self.assertFalse(bm.device_array.is_deleted())

  @parameterized.named_parameters(
      *_cross(_BENCHMARK_CLASSES, _BACKENDS_AND_DEVICES)
  )
  def test_run_calls_teardown(self, bm_cls, backend, local_device):
    """run() releases benchmark resources once measurement completes."""
    bm = self._make_benchmark(
        bm_cls,
        transfer_backend=backend,
        local_device=local_device,
        num_runs=2,
        warmup_tries=1,
    )
    with mock.patch.object(bm, "teardown", wraps=bm.teardown) as mock_teardown:
      bm.run()
    self.assertGreaterEqual(mock_teardown.call_count, 1)
    # The final call must be the one issued by run() after measurement.
    self.assertIsNone(bm.kv_manager)
    self.assertIsNone(bm.device_array)

  @parameterized.named_parameters(
      ("h2d", host_device.HostToDeviceBenchmark, "h2d"),
      ("d2h", host_device.DeviceToHostBenchmark, "d2h"),
  )
  def test_run_tears_down_when_measurement_raises(self, bm_cls, op_name):
    """run() releases the KVCacheManager even when a transfer fails."""
    bm = self._make_benchmark(
        bm_cls, transfer_backend="tpu_sync", num_runs=2, warmup_tries=1
    )
    getattr(self.mock_kv, op_name).side_effect = RuntimeError("DMA failed")
    with self.assertRaisesRegex(RuntimeError, "DMA failed"):
      bm.run()
    self.assertIsNone(bm.kv_manager)
    self.assertIsNone(bm.device_array)

  @parameterized.named_parameters(
      ("h2d_all", host_device.HostToDeviceBenchmark, "h2d", -1),
      ("h2d_dev0", host_device.HostToDeviceBenchmark, "h2d", 0),
      ("d2h_all", host_device.DeviceToHostBenchmark, "d2h", -1),
      ("d2h_dev0", host_device.DeviceToHostBenchmark, "d2h", 0),
  )
  def test_tpu_sync_run_e2e(self, bm_cls, op_name, local_device):
    """run() drives every tpu_sync transfer through a full-block copy."""
    bm = self._make_benchmark(
        bm_cls,
        transfer_backend="tpu_sync",
        local_device=local_device,
        num_runs=3,
        warmup_tries=2,
    )
    result = bm.run()
    expected_num_devices = _expected_num_devices(local_device)
    self.mock_kv_cls.assert_called()
    measured = getattr(self.mock_kv, op_name)
    # At least warmup_tries + num_runs transfers, all of the full block.
    self.assertGreaterEqual(measured.call_count, 5)
    for call in measured.call_args_list:
      self.assertEqual(call, mock.call([0], [0], [1]))
    self.assertGreater(
        result.metrics["wall_clock_bandwidth_per_device_gb_s"], 0.0
    )
    self.assertAlmostEqual(
        result.metrics["wall_clock_bandwidth_total_gb_s"],
        result.metrics["wall_clock_bandwidth_per_device_gb_s"]
        * expected_num_devices,
    )
    self.assertEqual(result.metrics["num_devices"], expected_num_devices)
    self.assertEqual(result.metrics["actual_runs"], 3)
    self.assertEqual(result.metadata.params["transfer_backend"], "tpu_sync")
    self.assertEqual(result.metadata.params["local_device"], local_device)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", result.metrics)
    self.assertIsNone(bm.kv_manager)


class HostToDeviceBenchmarkTest(_HostDeviceTestBase):
  """Unit tests for HostToDeviceBenchmark."""

  _BM = host_device.HostToDeviceBenchmark

  @parameterized.named_parameters(("all", -1), ("dev0", 0))
  def test_generate_inputs(self, local_device):
    """Verify input shape and default dtype across selected devices."""
    bm = self._setup_benchmark(self._BM, local_device=local_device)
    n = _expected_num_devices(local_device)
    inputs = bm.generate_inputs()
    self.assertLen(inputs, 1)
    host_data = inputs[0]
    self.assertIsInstance(host_data, np.ndarray)
    self.assertEqual(host_data.shape, (n * _ROWS_PER_DEVICE, 128))
    self.assertEqual(host_data.dtype, jnp.bfloat16)
    self.assertEqual(host_data.nbytes, bm.get_total_bytes())

  @parameterized.named_parameters(("all", -1), ("dev0", 0), ("dev2", 2))
  def test_run_op(self, local_device):
    """Verify run_op places a sharded JAX array on the transfer devices."""
    bm = self._setup_benchmark(self._BM, local_device=local_device)
    n = _expected_num_devices(local_device)
    out = bm.run_op(*bm.generate_inputs())
    self.assertIsInstance(out, jax.Array)
    self.assertEqual(out.shape, (n * _ROWS_PER_DEVICE, 128))
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertEqual(out.nbytes, n * _DATA_SIZE_BYTES)
    self.assertEqual(out.sharding.spec, _P("device", None))
    self.assertEqual(out.devices(), set(bm.get_transfer_devices()))
    for shard in out.addressable_shards:
      self.assertEqual(shard.data.shape, (_ROWS_PER_DEVICE, 128))

  def test_run_op_preserves_data(self):
    bm = self._setup_benchmark(self._BM)
    (host_data,) = bm.generate_inputs()
    out = bm.run_op(host_data)
    # Compare raw bits: random payloads contain NaN bit patterns.
    np.testing.assert_array_equal(
        np.asarray(out).view(np.uint16), host_data.view(np.uint16)
    )

  @parameterized.named_parameters(
      ("bfloat16", "bfloat16", jnp.bfloat16, 2, 16384),
      ("float32", "float32", jnp.float32, 4, 8192),
      ("float16", "float16", jnp.float16, 2, 16384),
      ("int32", "int32", jnp.int32, 4, 8192),
      ("int8", "int8", jnp.int8, 1, 32768),
      ("float8_e4m3fn", "float8_e4m3fn", jnp.float8_e4m3fn, 1, 32768),
      ("float8_e5m2", "float8_e5m2", jnp.float8_e5m2, 1, 32768),
  )
  def test_dtypes_generate_inputs_and_run_op(
      self, dtype_str, expected_dtype, expected_itemsize, rows_per_device
  ):
    """Verify generate_inputs and run_op honor config.dtype and payload size."""
    bm = self._setup_benchmark(self._BM, dtype=dtype_str, local_device=0)
    expected_shape = (rows_per_device, 128)
    inputs = bm.generate_inputs()
    self.assertLen(inputs, 1)
    host_data = inputs[0]
    self.assertIsInstance(host_data, np.ndarray)
    self.assertEqual(host_data.dtype, expected_dtype)
    self.assertEqual(host_data.dtype.itemsize, expected_itemsize)
    self.assertEqual(host_data.shape, expected_shape)
    self.assertEqual(host_data.nbytes, bm.config.data_size_bytes)

    out = bm.run_op(*inputs)
    self.assertIsInstance(out, jax.Array)
    self.assertEqual(out.dtype, expected_dtype)
    self.assertEqual(out.dtype.itemsize, expected_itemsize)
    self.assertEqual(out.shape, expected_shape)
    self.assertEqual(out.nbytes, bm.config.data_size_bytes)

  def test_invalid_dtype_raises_error(self):
    """Verify generate_inputs raises ValueError for invalid dtype."""
    with self.assertRaisesRegex(
        ValueError, "Invalid dtype string: 'invalid_dtype'"
    ):
      bm = self._setup_benchmark(self._BM, dtype="invalid_dtype")
      bm.generate_inputs()

  @parameterized.named_parameters(("all", -1), ("dev0", 0))
  def test_tpu_sync_setup_primes_host_staging(self, local_device):
    """setup() fills the pinned host buffer with a D2H before timing H2D."""
    bm = self._setup_benchmark(
        self._BM, transfer_backend="tpu_sync", local_device=local_device
    )
    self.mock_kv.d2h.assert_called_once_with([0], [0], [1])
    self.mock_kv.d2h.return_value.Await.assert_called_once_with()
    self.mock_kv.h2d.assert_not_called()
    self.assertIs(bm.kv_manager, self.mock_kv)

  @parameterized.named_parameters(("all", -1), ("dev0", 0))
  def test_tpu_sync_run_op(self, local_device):
    bm = self._setup_benchmark(
        self._BM, transfer_backend="tpu_sync", local_device=local_device
    )
    self.mock_kv.reset_mock()
    out = bm.run_op(*bm.generate_inputs())
    self.mock_kv.h2d.assert_called_once_with([0], [0], [1])
    self.mock_kv.h2d.return_value.Await.assert_called_once_with()
    self.mock_kv.d2h.assert_not_called()
    self.assertIs(out, bm.device_array)

  def test_tpu_sync_run_op_awaits_inside_marker(self):
    """The transfer is issued and awaited inside the MARKER annotation."""
    bm = self._setup_benchmark(self._BM, transfer_backend="tpu_sync")
    events = []
    self.mock_kv.h2d.side_effect = lambda *a: events.append("h2d") or (
        self.mock_kv.h2d.return_value
    )
    self.mock_kv.h2d.return_value.Await.side_effect = lambda: events.append(
        "await"
    )
    with mock.patch.object(
        jax.profiler,
        "TraceAnnotation",
        new=_RecordingTraceAnnotation(events),
    ):
      bm.run_op(*bm.generate_inputs())
    self.assertEqual(
        events,
        [
            f"enter:{constants.MARKER}",
            "h2d",
            "await",
            f"exit:{constants.MARKER}",
        ],
    )

  def test_jax_run_op_blocks_inside_marker(self):
    bm = self._setup_benchmark(self._BM)
    events = []
    with mock.patch.object(
        jax.profiler,
        "TraceAnnotation",
        new=_RecordingTraceAnnotation(events),
    ):
      bm.run_op(*bm.generate_inputs())
    self.assertEqual(
        events, [f"enter:{constants.MARKER}", f"exit:{constants.MARKER}"]
    )

  def test_tpu_sync_does_not_use_device_put(self):
    bm = self._setup_benchmark(self._BM, transfer_backend="tpu_sync")
    inputs = bm.generate_inputs()
    with mock.patch.object(jax, "device_put") as mock_device_put:
      bm.run_op(*inputs)
    mock_device_put.assert_not_called()

  def test_run_e2e(self):
    """Verify run() produces per-device and total bandwidth without per-chip bandwidth."""
    bm = self._make_benchmark(self._BM, num_runs=2, warmup_tries=1)
    result = bm.run()
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", result.metrics)
    self.assertIn("wall_clock_bandwidth_total_gb_s", result.metrics)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", result.metrics)
    self.assertGreater(
        result.metrics["wall_clock_bandwidth_per_device_gb_s"], 0.0
    )
    self.assertAlmostEqual(
        result.metrics["wall_clock_bandwidth_total_gb_s"],
        result.metrics["wall_clock_bandwidth_per_device_gb_s"],
    )
    self.assertIn("wall_clock_avg_ms", result.metrics)
    self.assertGreater(result.metrics["wall_clock_avg_ms"], 0.0)
    self.assertEqual(result.metrics["num_devices"], 1)
    self.assertEqual(result.metadata.params["local_device"], 0)
    self.assertNotIn("roofline_tflops_limit", result.metrics)
    self.assertNotIn("compute_roofline_efficiency_pct", result.metrics)
    self.assertNotIn("peak_hbm_bw_gb_s", result.metrics)
    self.assertNotIn("memory_roofline_efficiency_pct", result.metrics)
    self.mock_kv_cls.assert_not_called()


class DeviceToHostBenchmarkTest(_HostDeviceTestBase):
  """Unit tests for DeviceToHostBenchmark."""

  _BM = host_device.DeviceToHostBenchmark

  @parameterized.named_parameters(("all", -1), ("dev0", 0), ("dev3", 3))
  def test_generate_inputs(self, local_device):
    """Verify input is a sharded JAX Array on the transfer devices."""
    bm = self._setup_benchmark(self._BM, local_device=local_device)
    n = _expected_num_devices(local_device)
    inputs = bm.generate_inputs()
    self.assertLen(inputs, 1)
    device_array = inputs[0]
    self.assertIsInstance(device_array, jax.Array)
    self.assertEqual(device_array.shape, (n * _ROWS_PER_DEVICE, 128))
    self.assertEqual(device_array.dtype, jnp.bfloat16)
    self.assertEqual(device_array.nbytes, n * _DATA_SIZE_BYTES)
    self.assertEqual(device_array.sharding.spec, _P("device", None))
    self.assertEqual(device_array.devices(), set(bm.get_transfer_devices()))
    for shard in device_array.addressable_shards:
      self.assertEqual(shard.data.shape, (_ROWS_PER_DEVICE, 128))

  def test_jax_generate_inputs_before_setup_raises_error(self):
    bm = self._make_benchmark(self._BM)
    with self.assertRaisesRegex(ValueError, r"setup\(\) must be called"):
      bm.generate_inputs()

  def test_jax_generate_inputs_without_sharding_raises_error(self):
    bm = self._make_benchmark(self._BM)
    # A host payload alone is not enough; the sharding is built in setup().
    bm.host_data = np.zeros((_ROWS_PER_DEVICE, 128), dtype=jnp.bfloat16)
    with self.assertRaisesRegex(
        ValueError, r"setup\(\) must be called before creating device arrays"
    ):
      bm.generate_inputs()

  @parameterized.named_parameters(("all", -1), ("dev0", 0))
  def test_run_op(self, local_device):
    """Verify run_op returns a numpy array."""
    bm = self._setup_benchmark(self._BM, local_device=local_device)
    n = _expected_num_devices(local_device)
    inputs = bm.generate_inputs()
    out = bm.run_op(*inputs)
    self.assertIsInstance(out, np.ndarray)
    self.assertEqual(out.shape, (n * _ROWS_PER_DEVICE, 128))
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertEqual(out.nbytes, bm.get_total_bytes())

  @parameterized.named_parameters(
      ("bfloat16", "bfloat16", jnp.bfloat16, 2, 16384),
      ("float32", "float32", jnp.float32, 4, 8192),
      ("float16", "float16", jnp.float16, 2, 16384),
      ("int32", "int32", jnp.int32, 4, 8192),
      ("int8", "int8", jnp.int8, 1, 32768),
      ("float8_e4m3fn", "float8_e4m3fn", jnp.float8_e4m3fn, 1, 32768),
      ("float8_e5m2", "float8_e5m2", jnp.float8_e5m2, 1, 32768),
  )
  def test_dtypes_generate_inputs_and_run_op(
      self, dtype_str, expected_dtype, expected_itemsize, rows_per_device
  ):
    """Verify generate_inputs and run_op honor config.dtype and payload size."""
    bm = self._setup_benchmark(self._BM, dtype=dtype_str, local_device=0)
    expected_shape = (rows_per_device, 128)
    inputs = bm.generate_inputs()
    self.assertLen(inputs, 1)
    device_array = inputs[0]
    self.assertIsInstance(device_array, jax.Array)
    self.assertEqual(device_array.dtype, expected_dtype)
    self.assertEqual(device_array.dtype.itemsize, expected_itemsize)
    self.assertEqual(device_array.shape, expected_shape)
    self.assertEqual(device_array.nbytes, bm.config.data_size_bytes)

    out = bm.run_op(*inputs)
    self.assertIsInstance(out, np.ndarray)
    self.assertEqual(out.dtype, expected_dtype)
    self.assertEqual(out.dtype.itemsize, expected_itemsize)
    self.assertEqual(out.shape, expected_shape)
    self.assertEqual(out.nbytes, bm.config.data_size_bytes)

  def test_invalid_dtype_raises_error(self):
    """Verify setup raises ValueError for invalid dtype."""
    with self.assertRaisesRegex(
        ValueError, "Invalid dtype string: 'invalid_dtype'"
    ):
      self._setup_benchmark(self._BM, dtype="invalid_dtype")

  # --- reset_data ---------------------------------------------------------

  @parameterized.named_parameters(("all", -1), ("dev0", 0))
  def test_jax_reset_data_replaces_and_deletes_array(self, local_device):
    """jax mode must transfer a fresh array so device_get is never cached."""
    bm = self._setup_benchmark(self._BM, local_device=local_device)
    (old_array,) = bm.generate_inputs()
    bm.run_op(old_array)  # Populates the host-value cache of old_array.
    new_inputs = bm.reset_data(old_array)
    self.assertLen(new_inputs, 1)
    new_array = new_inputs[0]
    self.assertTrue(old_array.is_deleted())
    self.assertIsNot(new_array, old_array)
    self.assertFalse(new_array.is_deleted())
    self.assertEqual(new_array.shape, old_array.shape)
    self.assertEqual(new_array.sharding.spec, _P("device", None))

  def test_jax_reset_data_without_input_generates_array(self):
    bm = self._setup_benchmark(self._BM)
    (new_array,) = bm.reset_data(None)
    self.assertIsInstance(new_array, jax.Array)
    self.assertFalse(new_array.is_deleted())

  @parameterized.named_parameters(("all", -1), ("dev0", 0))
  def test_tpu_sync_reset_data_is_noop(self, local_device):
    """KVCacheManager.d2h bypasses the host-value cache; reuse the buffer."""
    bm = self._setup_benchmark(
        self._BM, transfer_backend="tpu_sync", local_device=local_device
    )
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

  # --- tpu_sync run_op ----------------------------------------------------

  @parameterized.named_parameters(("all", -1), ("dev0", 0))
  def test_tpu_sync_setup_does_not_transfer(self, local_device):
    self._setup_benchmark(
        self._BM, transfer_backend="tpu_sync", local_device=local_device
    )
    self.mock_kv.d2h.assert_not_called()
    self.mock_kv.h2d.assert_not_called()

  @parameterized.named_parameters(("all", -1), ("dev0", 0))
  def test_tpu_sync_run_op(self, local_device):
    bm = self._setup_benchmark(
        self._BM, transfer_backend="tpu_sync", local_device=local_device
    )
    out = bm.run_op(*bm.generate_inputs())
    self.mock_kv.d2h.assert_called_once_with([0], [0], [1])
    self.mock_kv.d2h.return_value.Await.assert_called_once_with()
    self.mock_kv.h2d.assert_not_called()
    self.assertIsNone(out)

  def test_tpu_sync_run_op_awaits_inside_marker(self):
    """The transfer is issued and awaited inside the MARKER annotation."""
    bm = self._setup_benchmark(self._BM, transfer_backend="tpu_sync")
    events = []
    self.mock_kv.d2h.side_effect = lambda *a: events.append("d2h") or (
        self.mock_kv.d2h.return_value
    )
    self.mock_kv.d2h.return_value.Await.side_effect = lambda: events.append(
        "await"
    )
    with mock.patch.object(
        jax.profiler,
        "TraceAnnotation",
        new=_RecordingTraceAnnotation(events),
    ):
      bm.run_op(*bm.generate_inputs())
    self.assertEqual(
        events,
        [
            f"enter:{constants.MARKER}",
            "d2h",
            "await",
            f"exit:{constants.MARKER}",
        ],
    )

  def test_tpu_sync_does_not_use_device_get(self):
    bm = self._setup_benchmark(self._BM, transfer_backend="tpu_sync")
    inputs = bm.generate_inputs()
    with mock.patch.object(jax, "device_get") as mock_device_get:
      bm.run_op(*inputs)
    mock_device_get.assert_not_called()

  def test_teardown_releases_host_data(self):
    bm = self._setup_benchmark(self._BM)
    self.assertIsNotNone(bm.host_data)
    bm.teardown()
    self.assertIsNone(bm.host_data)

  def test_run_e2e(self):
    """Verify run() produces per-device and total bandwidth without per-chip bandwidth."""
    bm = self._make_benchmark(self._BM, num_runs=2, warmup_tries=1)
    result = bm.run()
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", result.metrics)
    self.assertIn("wall_clock_bandwidth_total_gb_s", result.metrics)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", result.metrics)
    self.assertGreater(
        result.metrics["wall_clock_bandwidth_per_device_gb_s"], 0.0
    )
    self.assertAlmostEqual(
        result.metrics["wall_clock_bandwidth_total_gb_s"],
        result.metrics["wall_clock_bandwidth_per_device_gb_s"],
    )
    self.assertIn("wall_clock_avg_ms", result.metrics)
    self.assertGreater(result.metrics["wall_clock_avg_ms"], 0.0)
    self.assertEqual(result.metrics["num_devices"], 1)
    self.assertEqual(result.metadata.params["local_device"], 0)
    self.assertNotIn("roofline_tflops_limit", result.metrics)
    self.assertNotIn("compute_roofline_efficiency_pct", result.metrics)
    self.assertNotIn("peak_hbm_bw_gb_s", result.metrics)
    self.assertNotIn("memory_roofline_efficiency_pct", result.metrics)
    self.assertIsNone(bm.host_data)
    self.mock_kv_cls.assert_not_called()


if __name__ == "__main__":
  absltest.main()
