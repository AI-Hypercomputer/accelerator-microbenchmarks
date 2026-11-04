"""Unit tests for host_device.py."""

import contextlib
import csv
import itertools
import json
import os
import tempfile
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

# Covers every backend x scope pair across both directions.
_LAYOUT_CASES = (
    ("h2d_jax_host", _H2D, "jax", "host"),
    ("d2h_jax_device", _D2H, "jax", "device"),
    ("h2d_tpu_sync_device", _H2D, "tpu_sync", "device"),
    ("d2h_tpu_sync_chip", _D2H, "tpu_sync", "chip"),
    ("d2h_tpu_sync_host", _D2H, "tpu_sync", "host"),
)

# KVCacheManager method each direction times in run_op().
_MEASURED_OP = {_H2D: "h2d", _D2H: "d2h"}


def _expected_num_devices(
    scope: str,
    hw_spec: system.HardwareSpec = system.TPU7X_HARDWARE_SPEC,
) -> int:
  resolved = constants.Scope(scope)
  if resolved == constants.Scope.DEVICE:
    return 1
  if resolved == constants.Scope.CHIP:
    return hw_spec.devices_per_chip
  return _NUM_DEVICES


def _expected_devices(
    scope: str,
    hw_spec: system.HardwareSpec = system.TPU7X_HARDWARE_SPEC,
    local_index: int = 0,
) -> list[jax.Device]:
  """Returns the expected CPU devices for `scope` and `local_index`."""
  local_devices = list(jax.local_devices())
  resolved = constants.Scope(scope)
  if resolved == constants.Scope.DEVICE:
    return [local_devices[local_index]]
  if resolved == constants.Scope.CHIP:
    n = hw_spec.devices_per_chip
    return local_devices[local_index * n : (local_index + 1) * n]
  return local_devices


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
    # Attach simulated chip `coords` to virtual CPU devices matching
    # `devices_per_chip` so multi-core chip scope validation is exercised.
    self._devices_per_chip = system.TPU7X_HARDWARE_SPEC.devices_per_chip
    self.enter_context(
        mock.patch.object(
            type(jax.local_devices()[0]),
            "coords",
            new=property(lambda d: (d.id // self._devices_per_chip, 0, 0)),
            create=True,
        )
    )
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  @property
  def mock_kv(self) -> mock.MagicMock:
    return self.mock_kv_cls.return_value

  def _make_benchmark(
      self,
      bm_cls,
      hardware_spec: system.HardwareSpec = system.TPU7X_HARDWARE_SPEC,
      **kwargs,
  ):
    """Instantiates `bm_cls` with CPU test fixtures and registers teardown."""
    self._devices_per_chip = hardware_spec.devices_per_chip
    config_params = {"data_size_mib": _DATA_SIZE_MIB}
    config_params.update(kwargs)
    config = host_device.HostDeviceParams(**config_params)
    bm = bm_cls(
        config=config,
        hardware_spec=hardware_spec,
        mesh=self.mock_mesh,
    )
    self.addCleanup(bm.teardown)
    return bm

  def _setup_benchmark(self, bm_cls, **kwargs):
    bm = self._make_benchmark(bm_cls, **kwargs)
    bm.setup()
    return bm

  def _assert_row_sharded(self, arr: jax.Array, scope: str):
    """Asserts `arr` is a 2D jax buffer row-sharded over the transfer devices."""
    self.assertIsInstance(arr, jax.Array)
    self.assertEqual(arr.sharding.spec, _P("device", None))
    self.assertEqual(arr.devices(), set(_expected_devices(scope)))
    self.assertLen(arr.addressable_shards, _expected_num_devices(scope))


class HostDeviceParamsValidationTest(parameterized.TestCase):
  """Verifies the bounds and enums declared on HostDeviceParams fields."""

  def test_defaults(self):
    self.assertTrue(
        issubclass(host_device.HostDeviceParams, base.ScopeParamsMixin)
    )
    params = host_device.HostDeviceParams()
    self.assertEqual(params.transfer_backend, _TPU_SYNC)
    self.assertEqual(params.scope, constants.Scope.HOST)
    self.assertEqual(params.local_index, 0)
    self.assertEqual(params.data_size_mib, 64)
    self.assertEqual(params.data_size_bytes, 64 * 1024 * 1024)

  def test_local_index_validation(self):
    """Verifies non-negative integer local_index validation on HostDeviceParams."""
    params = host_device.HostDeviceParams(scope="device", local_index=3)
    self.assertEqual(params.scope, constants.Scope.DEVICE)
    self.assertEqual(params.local_index, 3)

    with self.assertRaisesRegex(ValueError, "local_index must be >= 0"):
      host_device.HostDeviceParams(scope="device", local_index=-1)

    with self.assertRaisesRegex(
        ValueError, "local_index must be an integer >= 0"
    ):
      host_device.HostDeviceParams(scope="device", local_index=True)

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

  @parameterized.named_parameters(
      ("device", "device", constants.Scope.DEVICE),
      ("chip", "chip", constants.Scope.CHIP),
      ("host", "host", constants.Scope.HOST),
  )
  def test_scope_coerced_to_enum(self, value, expected):
    params = host_device.HostDeviceParams(scope=value)
    self.assertIsInstance(params.scope, constants.Scope)
    self.assertEqual(params.scope, expected)

  @parameterized.parameters("rack", "HOST", "DEVICE", 0, -1, None)
  def test_invalid_scope_raises_error(self, value):
    with self.assertRaisesRegex(
        ValueError,
        rf"Invalid scope '{value}'\. Must be one of \['device', 'chip',"
        r" 'host'\]",
    ):
      host_device.HostDeviceParams(scope=value)

  @parameterized.named_parameters(
      ("data_size_zero", "data_size_mib", 0, "data_size_mib must be >= 1"),
      (
          "data_size_negative",
          "data_size_mib",
          -2,
          "data_size_mib must be >= 1",
      ),
  )
  def test_out_of_bounds_param_raises_error(self, name, value, regex):
    """Values below a field's declared minimum are rejected."""
    with self.assertRaisesRegex(ValueError, regex):
      host_device.HostDeviceParams(**{name: value})

  @parameterized.named_parameters(
      ("none", "data_size_mib", 1, None),
      ("bool", "data_size_mib", 1, True),
      ("float", "data_size_mib", 1, 4.0),
      ("str", "data_size_mib", 1, "4"),
  )
  def test_non_int_param_raises_error(self, name, minimum, value):
    """Non-integer values (including bools) are rejected."""
    with self.assertRaisesRegex(
        ValueError, rf"{name} must be an integer >= {minimum}"
    ):
      host_device.HostDeviceParams(**{name: value})

  @parameterized.parameters("int4", "float4_e2m1fn")
  def test_sub_byte_dtype_raises_error(self, dtype):
    """Verifies sub-byte dtypes are rejected (bytes count itemsize)."""
    with self.assertRaisesRegex(
        ValueError, f"^dtype: Unsupported sub-byte dtype '{dtype}'"
    ):
      host_device.HostDeviceParams(dtype=dtype)

  def test_numpy_int_params_normalized_to_int(self):
    params = host_device.HostDeviceParams(data_size_mib=np.int64(8))
    self.assertIs(type(params.data_size_mib), int)
    self.assertEqual(params.data_size_mib, 8)


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

  # --- scope selection ---------------------------------------------------

  @parameterized.named_parameters(
      ("device", "device", 0),
      ("device_nonzero", "device", 3),
      ("chip", "chip", 0),
      ("chip_nonzero", "chip", 1),
      ("host", "host", 0),
  )
  def test_get_devices_to_measure(self, scope, local_index):
    """Resolves expected transfer devices across scopes and local_index."""
    bm = self._make_benchmark(_H2D, scope=scope, local_index=local_index)
    expected = _expected_devices(scope, local_index=local_index)
    self.assertEqual(bm.get_devices_to_measure(), expected)
    self.assertLen(expected, bm.num_transfer_devices)
    self.assertEqual(bm.get_device_to_measure(), expected[0])

  def test_get_devices_to_measure_out_of_range_local_index_raises(self):
    """Out-of-range local_index raises ValueError without clamping."""
    bm_dev = self._make_benchmark(_H2D, scope="device", local_index=4)
    with self.assertRaisesRegex(
        ValueError, "local_index 4 out of range for scope='device'"
    ):
      bm_dev.get_devices_to_measure()

    bm_chip = self._make_benchmark(_H2D, scope="chip", local_index=2)
    with self.assertRaisesRegex(
        ValueError, "local_index 2 out of range for scope='chip'"
    ):
      bm_chip.get_devices_to_measure()

    bm_host = self._make_benchmark(_H2D, scope="host", local_index=1)
    with self.assertRaisesRegex(
        ValueError, "local_index must be 0 when scope='host', got 1"
    ):
      bm_host.get_devices_to_measure()

  def test_get_devices_to_measure_chip_scope_single_die(self):
    """On single-die chips (v6e), chip scope selects 1 device."""
    bm = self._make_benchmark(
        _H2D, hardware_spec=system.V6E_HARDWARE_SPEC, scope="chip"
    )
    self.assertEqual(bm.get_devices_to_measure(), [jax.local_devices()[0]])
    self.assertEqual(bm.num_transfer_devices, 1)

  # --- Buffer shape, sharding and mesh ----------------------------------

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_setup_builds_buffer_layout(self, bm_cls, backend, scope):
    """jax uses a 2D row-sharded buffer; tpu_sync a block-major 3D buffer."""
    bm = self._setup_benchmark(bm_cls, transfer_backend=backend, scope=scope)
    n = _expected_num_devices(scope)
    self.assertEqual(bm.rows_per_device, _ROWS_PER_DEVICE)
    if backend == "tpu_sync":
      self.assertEqual(bm.get_buffer_shape(), (1, n * _ROWS_PER_DEVICE, 128))
      self.assertEqual(bm.get_partition_spec(), _P(None, "device", None))
    else:
      self.assertEqual(bm.get_buffer_shape(), (n * _ROWS_PER_DEVICE, 128))
      self.assertEqual(bm.get_partition_spec(), _P("device", None))
    # setup() replaces the mesh with one spanning only the transfer devices.
    self.assertEqual(bm.mesh.axis_names, ("device",))
    self.assertEqual(list(bm.mesh.devices.flat), _expected_devices(scope))
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

  @parameterized.named_parameters(
      ("device_tpu7x", "device", system.TPU7X_HARDWARE_SPEC),
      ("chip_tpu7x", "chip", system.TPU7X_HARDWARE_SPEC),
      ("host_tpu7x", "host", system.TPU7X_HARDWARE_SPEC),
      ("chip_v6e", "chip", system.V6E_HARDWARE_SPEC),
      ("host_v6e", "host", system.V6E_HARDWARE_SPEC),
  )
  def test_calculate_metrics(self, scope, hw_spec):
    """Smaller scopes are derived as average shares; larger scopes remain None."""
    bm = self._setup_benchmark(_H2D, hardware_spec=hw_spec, scope=scope)
    n = _expected_num_devices(scope, hw_spec=hw_spec)
    self.assertAlmostEqual(bm.get_total_bytes(), float(_DATA_SIZE_BYTES * n))

    metrics = bm.calculate_metrics([10.0, 10.0, 10.0])
    # Per-device bandwidth is 4194304 bytes / (0.01 s * 1e9) = 0.4194304 GB/s;
    # active scope bandwidth scales by the number of participating devices.
    wc_per_dev = 0.4194304
    self.assertAlmostEqual(metrics["wall_clock_p50_ms"], 10.0)
    self.assertAlmostEqual(
        metrics[f"wall_clock_bandwidth_per_{scope}_gb_s"], wc_per_dev * n
    )
    self.assertEqual(metrics["num_devices"], n)

    xprof = bm.calculate_throughput_metrics(1.0, constants.TimingDomain.XPROF)
    xp_per_dev = _DATA_SIZE_BYTES / 1e6
    self.assertAlmostEqual(
        xprof[f"xprof_bandwidth_per_{scope}_gb_s"], xp_per_dev * n
    )

    for domain_metrics, per_dev_bw, prefix in (
        (metrics, wc_per_dev, "wall_clock"),
        (xprof, xp_per_dev, "xprof"),
    ):
      self.assertAlmostEqual(
          domain_metrics[f"{prefix}_bandwidth_per_device_gb_s"], per_dev_bw
      )
      if scope == "device":
        self.assertIsNone(domain_metrics[f"{prefix}_bandwidth_per_chip_gb_s"])
        self.assertIsNone(domain_metrics[f"{prefix}_bandwidth_per_host_gb_s"])
      elif scope == "chip":
        self.assertAlmostEqual(
            domain_metrics[f"{prefix}_bandwidth_per_chip_gb_s"],
            per_dev_bw * hw_spec.devices_per_chip,
        )
        self.assertIsNone(domain_metrics[f"{prefix}_bandwidth_per_host_gb_s"])
      else:
        self.assertAlmostEqual(
            domain_metrics[f"{prefix}_bandwidth_per_chip_gb_s"],
            per_dev_bw * hw_spec.devices_per_chip,
        )
        self.assertAlmostEqual(
            domain_metrics[f"{prefix}_bandwidth_per_host_gb_s"],
            per_dev_bw * n,
        )

  @parameterized.named_parameters(
      ("device", "device"),
      ("chip", "chip"),
      ("host", "host"),
  )
  def test_zero_latency_gives_infinite_bandwidth(self, scope):
    """Zero latency yields inf on active and smaller scopes and None on larger."""
    bm = self._make_benchmark(_H2D, scope=scope)
    metrics = bm.calculate_throughput_metrics(
        0.0, constants.TimingDomain.WALL_CLOCK
    )
    self.assertEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], float("inf")
    )
    if scope == "device":
      self.assertIsNone(metrics["wall_clock_bandwidth_per_chip_gb_s"])
      self.assertIsNone(metrics["wall_clock_bandwidth_per_host_gb_s"])
    elif scope == "chip":
      self.assertEqual(
          metrics["wall_clock_bandwidth_per_chip_gb_s"], float("inf")
      )
      self.assertIsNone(metrics["wall_clock_bandwidth_per_host_gb_s"])
    else:
      self.assertEqual(
          metrics["wall_clock_bandwidth_per_chip_gb_s"], float("inf")
      )
      self.assertEqual(
          metrics["wall_clock_bandwidth_per_host_gb_s"], float("inf")
      )

  @parameterized.named_parameters(
      ("jax_host", "jax", "host", 0, "jax_scope_host_idx_0_size_4mib"),
      ("jax_chip", "jax", "chip", 1, "jax_scope_chip_idx_1_size_4mib"),
      (
          "tpu_sync_host",
          "tpu_sync",
          "host",
          0,
          "tpu_sync_scope_host_idx_0_size_4mib",
      ),
      (
          "tpu_sync_device",
          "tpu_sync",
          "device",
          3,
          "tpu_sync_scope_device_idx_3_size_4mib",
      ),
  )
  def test_get_run_identifier(self, backend, scope, local_index, expected):
    """Run identifiers encode backend, scope, local_index, and payload size."""
    bm = self._make_benchmark(
        _H2D, transfer_backend=backend, scope=scope, local_index=local_index
    )
    self.assertEqual(bm.get_run_identifier(), expected)

  # --- Report schema ------------------------------------------------------

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_schema_coverage(self, bm_cls, backend, scope):
    """Verify REPORT_SCHEMA matches output keys and covers all metrics."""
    bm = self._setup_benchmark(bm_cls, transfer_backend=backend, scope=scope)
    test_report_utils.assert_schema_matches_output(self, bm)

  @parameterized.named_parameters(
      ("h2d_tpu_sync_device", _H2D, "tpu_sync", "device", 1),
      ("h2d_tpu_sync_chip", _H2D, "tpu_sync", "chip", 2),
      ("d2h_jax_host", _D2H, "jax", "host", 8),
  )
  def test_format_benchmark_table(self, bm_cls, backend, scope, num_devices):
    """Tests formatting of host<->device benchmark tables with all scope columns."""
    name = bm_cls.__name__
    metrics = {
        "num_devices": num_devices,
        "wall_clock_p50_ms": 14.0274,
        "wall_clock_bandwidth_per_device_gb_s": 18.25,
        "wall_clock_bandwidth_per_chip_gb_s": (
            None if scope == "device" else 36.50
        ),
        "wall_clock_bandwidth_per_host_gb_s": (
            18.25 * num_devices if scope == "host" else None
        ),
        "xprof_p50_ms": 14.0100,
        "xprof_bandwidth_per_device_gb_s": 18.50,
        "xprof_bandwidth_per_chip_gb_s": None if scope == "device" else 37.00,
        "xprof_bandwidth_per_host_gb_s": (
            18.50 * num_devices if scope == "host" else None
        ),
    }
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
                "scope": scope,
                "local_index": 0,
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics=metrics,
        timings=base.BenchmarkTimings(
            wall_clock=base.WallClockTimings(durations_ms=[1.0])
        ),
    )
    df = report.results_to_dataframe([res])
    table = report.format_benchmark_table(
        df, schema=bm_cls.REPORT_SCHEMA, title=name
    )
    self.assertIn(f"Benchmark Results ({name})", table)
    for col_name, _ in bm_cls.REPORT_SCHEMA:
      self.assertIn(col_name, table)
    if scope == "host":
      self.assertNotIn(" - ", table)
    else:
      self.assertIn("-", table)
    self.assertNotIn("None", table)
    for value in (
        "float32",
        backend,
        scope,
        "256",
        "18.25",
        "18.50",
        f"{18.25 * num_devices:.2f}",
        f"{18.50 * num_devices:.2f}",
        "14.0274",
        "14.0100",
    ):
      self.assertIn(value, table)

  def test_host_scope_console_csv_and_json_outputs(self):
    """Verifies console table, summary.csv, and detailed.json for host-scope H2D."""
    bm = self._setup_benchmark(_H2D, transfer_backend="tpu_sync", scope="host")
    metrics = bm.calculate_metrics([10.0, 10.0, 10.0])
    metrics["xprof_p50_ms"] = 9.5000
    metrics.update(
        bm.calculate_throughput_metrics(9.5, constants.TimingDomain.XPROF)
    )
    res = base.BenchmarkResult(
        metadata=base.BenchmarkMetadata(
            benchmark_name=_H2D.__name__,
            test_name="HostToDeviceBenchmark_host_test",
            start_time="2026-08-18T10:00:00",
            end_time="2026-08-18T10:01:00",
            params={
                "dtype": bm.config.dtype,
                "transfer_backend": str(bm.config.transfer_backend),
                "scope": str(bm.config.scope),
                "local_index": bm.config.local_index,
                "data_size_mib": bm.config.data_size_mib,
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics=metrics,
        timings=base.BenchmarkTimings(
            wall_clock=base.WallClockTimings(durations_ms=[10.0, 10.0, 10.0])
        ),
    )
    df = report.results_to_dataframe([res])
    table = report.format_standard_table(df, _H2D)
    for col_name, _ in _H2D.REPORT_SCHEMA:
      self.assertIn(col_name, table)
    self.assertNotIn("None", table)
    self.assertNotIn(" - ", table)

    with tempfile.TemporaryDirectory() as tmpdir:
      report.save_output([res], tmpdir)
      csv_path = os.path.join(tmpdir, "summary.csv")
      json_path = os.path.join(tmpdir, "detailed.json")
      with open(csv_path, "r") as f:
        csv_text = f.read()
        f.seek(0)
        rows = list(csv.DictReader(f))
      self.assertNotIn("None", csv_text)
      self.assertLen(rows, 1)
      wc_host = float(rows[0]["wall_clock_bandwidth_per_host_gb_s"])
      xp_host = float(rows[0]["xprof_bandwidth_per_host_gb_s"])
      self.assertAlmostEqual(
          float(rows[0]["wall_clock_bandwidth_per_device_gb_s"]),
          wc_host / _NUM_DEVICES,
      )
      self.assertAlmostEqual(
          float(rows[0]["wall_clock_bandwidth_per_chip_gb_s"]),
          (wc_host / _NUM_DEVICES) * 2,
      )
      self.assertAlmostEqual(
          float(rows[0]["xprof_bandwidth_per_device_gb_s"]),
          xp_host / _NUM_DEVICES,
      )
      self.assertAlmostEqual(
          float(rows[0]["xprof_bandwidth_per_chip_gb_s"]),
          (xp_host / _NUM_DEVICES) * 2,
      )

      with open(json_path, "r") as f:
        json_text = f.read()
      self.assertNotIn("None", json_text)
      self.assertNotIn("NaN", json_text)
      parsed = json.loads(json_text)
      json_metrics = parsed[0]["metrics"]
      self.assertAlmostEqual(
          json_metrics["wall_clock_bandwidth_per_device_gb_s"],
          json_metrics["wall_clock_bandwidth_per_host_gb_s"] / _NUM_DEVICES,
      )
      self.assertAlmostEqual(
          json_metrics["wall_clock_bandwidth_per_chip_gb_s"],
          (json_metrics["wall_clock_bandwidth_per_host_gb_s"] / _NUM_DEVICES)
          * 2,
      )
      self.assertGreater(
          json_metrics["wall_clock_bandwidth_per_host_gb_s"], 0.0
      )
      self.assertAlmostEqual(
          json_metrics["xprof_bandwidth_per_device_gb_s"],
          json_metrics["xprof_bandwidth_per_host_gb_s"] / _NUM_DEVICES,
      )
      self.assertAlmostEqual(
          json_metrics["xprof_bandwidth_per_chip_gb_s"],
          (json_metrics["xprof_bandwidth_per_host_gb_s"] / _NUM_DEVICES) * 2,
      )
      self.assertGreater(json_metrics["xprof_bandwidth_per_host_gb_s"], 0.0)
      undeclared_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
      if undeclared_dir:
        out_file = os.path.join(undeclared_dir, "h2d_host_renderings.txt")
        with open(out_file, "w") as out_f:
          out_f.write(
              f"=== CONSOLE ===\n{table}\n"
              f"=== CSV ===\n{csv_text}\n"
              f"=== JSON ===\n{json_text}\n"
          )

  # --- jax backend generate_inputs / run_op -------------------------------

  @parameterized.named_parameters(
      *_cross(
          _BENCHMARK_CLASSES,
          (
              ("bfloat16_host", "bfloat16", jnp.bfloat16, "host"),
              ("float32_chip", "float32", jnp.float32, "chip"),
              ("int8_host", "int8", jnp.int8, "host"),
              ("float8_e5m2_device", "float8_e5m2", jnp.float8_e5m2, "device"),
          ),
      )
  )
  def test_jax_generate_inputs_and_run_op(
      self, bm_cls, dtype, jnp_dtype, scope
  ):
    """Inputs and outputs honor dtype, payload size, and scope selection."""
    bm = self._setup_benchmark(
        bm_cls, dtype=dtype, transfer_backend="jax", scope=scope
    )
    n = _expected_num_devices(scope)
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
    self._assert_row_sharded(device_arr, scope)

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
    """Verify config construction raises ValueError for invalid dtype."""
    with self.assertRaisesRegex(
        ValueError,
        r"^dtype: Invalid dtype 'invalid_dtype'\. Supported dtypes for dtype:",
    ):
      self._make_benchmark(
          bm_cls, dtype="invalid_dtype", transfer_backend="jax"
      )

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
      ("h2d_host", _H2D, "host", True),
      ("h2d_chip", _H2D, "chip", True),
      ("h2d_device", _H2D, "device", True),
      ("d2h_host", _D2H, "host", False),
      ("d2h_chip", _D2H, "chip", False),
      ("d2h_device", _D2H, "device", False),
  )
  def test_tpu_sync_setup_creates_block_major_kv_manager(
      self, bm_cls, scope, primes_host_staging
  ):
    """setup() allocates one KVCacheManager whose shards are single blocks."""
    bm = self._setup_benchmark(bm_cls, transfer_backend="tpu_sync", scope=scope)
    self.mock_kv_cls.assert_called_once_with(
        kv_caches=[bm.device_array],
        local_control_port=0,
        host_blocks_to_allocate=1,
        unsafe_skip_buffer_lock=True,
    )
    self.assertIs(bm.kv_manager, self.mock_kv)

    n = _expected_num_devices(scope)
    kv_cache = bm.device_array
    self.assertIsInstance(kv_cache, jax.Array)
    self.assertEqual(kv_cache.shape, (1, n * _ROWS_PER_DEVICE, 128))
    self.assertEqual(kv_cache.dtype, jnp.bfloat16)
    self.assertEqual(kv_cache.sharding.spec, _P(None, "device", None))
    self.assertEqual(kv_cache.devices(), set(_expected_devices(scope)))
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
      self, bm_cls, backend, scope
  ):
    """teardown() may be called before setup() and more than once."""
    bm = self._make_benchmark(bm_cls, transfer_backend=backend, scope=scope)
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
  def test_run_calls_teardown(self, bm_cls, backend, scope):
    """run() releases benchmark resources exactly once after measurement."""
    bm = self._make_benchmark(
        bm_cls,
        transfer_backend=backend,
        scope=scope,
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
  def test_run_e2e(self, bm_cls, backend, scope):
    """run() populates active and smaller scopes and leaves larger scopes None."""
    bm = self._make_benchmark(
        bm_cls,
        transfer_backend=backend,
        scope=scope,
        num_runs=3,
        warmup_tries=2,
    )
    result = bm.run()
    n = _expected_num_devices(scope)
    metrics = result.metrics
    active_bw = metrics[f"wall_clock_bandwidth_per_{scope}_gb_s"]
    self.assertGreater(active_bw, 0.0)
    self.assertEqual(metrics["num_devices"], n)
    self.assertEqual(metrics["actual_runs"], 3)
    if scope == "device":
      self.assertIsNone(metrics["wall_clock_bandwidth_per_chip_gb_s"])
      self.assertIsNone(metrics["wall_clock_bandwidth_per_host_gb_s"])
    elif scope == "chip":
      self.assertAlmostEqual(
          metrics["wall_clock_bandwidth_per_device_gb_s"], active_bw / 2.0
      )
      self.assertIsNone(metrics["wall_clock_bandwidth_per_host_gb_s"])
    else:
      self.assertAlmostEqual(
          metrics["wall_clock_bandwidth_per_device_gb_s"], active_bw / n
      )
      self.assertAlmostEqual(
          metrics["wall_clock_bandwidth_per_chip_gb_s"], (active_bw / n) * 2.0
      )
    for key in (
        "roofline_tflops_limit",
        "compute_roofline_efficiency_pct",
        "peak_hbm_bw_gb_s",
        "memory_roofline_efficiency_pct",
    ):
      self.assertNotIn(key, metrics)
    self.assertEqual(result.metadata.params["transfer_backend"], backend)
    self.assertEqual(result.metadata.params["scope"], scope)
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
