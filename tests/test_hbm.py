"""Unit tests for hbm.py."""

import argparse
import contextlib
import io
import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks import cli
from accelerator_microbenchmarks.benchmarks import hbm
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import profiler
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
import jax
import jax.numpy as jnp
import numpy as np

_NUM_CPU_DEVICES = 8

# Force multiple virtual CPU devices so multi-device scopes ('chip', 'host')
# can be tested without TPUs. Must run before the JAX backend is initialized;
# JAX raises an error if the backend has already been initialized.
jax.config.update("jax_num_cpu_devices", _NUM_CPU_DEVICES)
# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class HBMBandwidthBenchmarkTest(parameterized.TestCase):
  """Unit tests for hbm.py."""

  def setUp(self):
    super().setUp()
    self.patcher = mock.patch(
        "accelerator_microbenchmarks.core.platform.get_platform_info",
        return_value=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
    )
    self.patcher.start()
    self.addCleanup(self.patcher.stop)
    # Attach simulated chip `coords` to virtual CPU devices matching
    # `devices_per_chip` so multi-core chip scope validation is exercised.
    self._devices_per_chip = system.TPU7X_HARDWARE_SPEC.devices_per_chip
    self.coords_patcher = mock.patch.object(
        type(jax.local_devices()[0]),
        "coords",
        new=property(lambda d: (d.id // self._devices_per_chip, 0, 0)),
        create=True,
    )
    self.coords_patcher.start()
    self.addCleanup(self.coords_patcher.stop)
    # Create a dummy mesh for testing on CPU
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    self.params = {
        "num_elements": 1024,
        "dtype": "bfloat16",
    }

  def _setup_benchmark(
      self,
      op_type: hbm.HBMKernelOp | str = hbm.HBMKernelOp.COPY,
      hardware_spec: system.HardwareSpec = system.TPU7X_HARDWARE_SPEC,
      **kwargs,
  ):
    """Initializes the HBMBandwidthBenchmark with given params."""
    self._devices_per_chip = hardware_spec.devices_per_chip
    params = self.params.copy()
    params["op_type"] = op_type
    params.update(kwargs)
    config = hbm.HBMBandwidthParams(**params)
    self.bm = hbm.HBMBandwidthBenchmark(
        config=config, hardware_spec=hardware_spec, mesh=self.mock_mesh
    )
    self.bm.setup()

  def test_benchmark_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("hbm")
    self.assertEqual(bm_class, hbm.HBMBandwidthBenchmark)

  @parameterized.named_parameters(
      ("copy", "copy", 2048, "device", 0, "copy_dim_2048_scope_device_idx_0"),
      (
          "scale",
          "scale",
          2048,
          "device",
          0,
          "scale_dim_2048_scope_device_idx_0",
      ),
      ("add", "add", 2048, "device", 0, "add_dim_2048_scope_device_idx_0"),
      (
          "triad",
          "triad",
          2048,
          "device",
          0,
          "triad_dim_2048_scope_device_idx_0",
      ),
      (
          "read_only",
          "read_only",
          2048,
          "device",
          0,
          "read_only_dim_2048_scope_device_idx_0",
      ),
      (
          "write_only",
          "write_only",
          2048,
          "device",
          0,
          "write_only_dim_2048_scope_device_idx_0",
      ),
      (
          "read_alias",
          "read",
          2048,
          "device",
          0,
          "read_dim_2048_scope_device_idx_0",
      ),
      (
          "write_alias",
          "write",
          2048,
          "device",
          0,
          "write_dim_2048_scope_device_idx_0",
      ),
      (
          "nonzero_device_idx",
          "write_only",
          2048,
          "device",
          7,
          "write_only_dim_2048_scope_device_idx_7",
      ),
      (
          "custom_num_elements",
          "copy",
          4096,
          "chip",
          2,
          "copy_dim_4096_scope_chip_idx_2",
      ),
      (
          "large_num_elements",
          "add",
          134217728,
          "host",
          0,
          "add_dim_134217728_scope_host_idx_0",
      ),
  )
  def test_get_run_identifier(
      self, op_type, num_elements, scope, local_index, expected_identifier
  ):
    """Verify run identifier generation for all STREAM ops, scopes, and local_index."""
    config = hbm.HBMBandwidthParams(
        op_type=op_type,
        num_elements=num_elements,
        scope=scope,
        local_index=local_index,
    )
    self.bm = hbm.HBMBandwidthBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.bm.setup()
    self.assertEqual(self.bm.get_run_identifier(), expected_identifier)

  def test_get_devices_to_measure_and_get_device_to_measure(self):
    """Verify get_devices_to_measure and get_device_to_measure across scopes and local_index."""
    local_devices = list(jax.local_devices())
    self.assertLen(local_devices, _NUM_CPU_DEVICES)

    # 1. device scope -> 1 device (default 0 and non-zero local_index)
    self._setup_benchmark(num_elements=64, scope="device")
    self.assertEqual(self.bm.get_devices_to_measure(), local_devices[:1])
    self.assertEqual(self.bm.get_device_to_measure(), local_devices[0])

    self._setup_benchmark(num_elements=64, scope="device", local_index=5)
    self.assertEqual(self.bm.get_devices_to_measure(), [local_devices[5]])
    self.assertEqual(self.bm.get_device_to_measure(), local_devices[5])

    # 2. chip scope on dual-die TPU7x -> 2 devices (default chip 0 and chip 2)
    self._setup_benchmark(num_elements=64, scope="chip")
    self.assertEqual(self.bm.get_devices_to_measure(), local_devices[:2])
    self.assertEqual(self.bm.get_device_to_measure(), local_devices[0])

    self._setup_benchmark(num_elements=64, scope="chip", local_index=2)
    self.assertEqual(self.bm.get_devices_to_measure(), local_devices[4:6])
    self.assertEqual(self.bm.get_device_to_measure(), local_devices[4])

    # 3. chip scope on single-die v6e -> 1 device
    self._setup_benchmark(
        num_elements=64,
        scope="chip",
        local_index=3,
        hardware_spec=system.V6E_HARDWARE_SPEC,
    )
    self.assertEqual(self.bm.get_devices_to_measure(), [local_devices[3]])

    # 4. host scope -> all local devices
    self._setup_benchmark(num_elements=64, scope="host")
    self.assertEqual(self.bm.get_devices_to_measure(), local_devices)
    self.assertEqual(self.bm.get_device_to_measure(), local_devices[0])

    # 5. Out-of-range local_index raises ValueError during setup()
    with self.assertRaisesRegex(
        ValueError, "local_index 8 out of range for scope='device'"
    ):
      self._setup_benchmark(num_elements=64, scope="device", local_index=8)
    with self.assertRaisesRegex(
        ValueError, "local_index 4 out of range for scope='chip'"
    ):
      self._setup_benchmark(num_elements=64, scope="chip", local_index=4)
    with self.assertRaisesRegex(
        ValueError, "local_index must be 0 when scope='host', got 1"
    ):
      self._setup_benchmark(num_elements=64, scope="host", local_index=1)

  def test_get_xprof_device_timing_aggregation_by_scope(self):
    """Multi-device scopes promote default XProf aggregation to slowest device."""
    # Single-device scope keeps measured_device_or_first.
    self._setup_benchmark(num_elements=64, scope="device")
    self.assertEqual(
        self.bm.get_xprof_device_timing_aggregation(),
        constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST,
    )

    # Single-die chip scope (v6e: devices_per_chip=1) has 1 device -> unchanged.
    self._setup_benchmark(
        num_elements=64,
        scope="chip",
        hardware_spec=system.V6E_HARDWARE_SPEC,
    )
    self.assertEqual(
        self.bm.get_xprof_device_timing_aggregation(),
        constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST,
    )

    # Dual-die chip scope (TPU7x: devices_per_chip=2) promotes to slowest.
    self._setup_benchmark(num_elements=64, scope="chip")
    self.assertEqual(
        self.bm.get_xprof_device_timing_aggregation(),
        constants.XprofDeviceTimingAggregation.LOCAL_SLOWEST_DEVICE,
    )

    # Multi-device host scope promotes to slowest.
    self._setup_benchmark(num_elements=64, scope="host")
    self.assertEqual(
        self.bm.get_xprof_device_timing_aggregation(),
        constants.XprofDeviceTimingAggregation.LOCAL_SLOWEST_DEVICE,
    )

    # Explicit global_slowest_per_step is preserved.
    explicit_bm = hbm.HBMBandwidthBenchmark(
        config=hbm.HBMBandwidthParams(num_elements=64, scope="chip"),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
        xprof_config=base.XprofConfig(
            xprof_timing=True,
            device_timing_aggregation=(
                constants.XprofDeviceTimingAggregation.GLOBAL_SLOWEST_PER_STEP
            ),
        ),
    )
    self.assertEqual(
        explicit_bm.get_xprof_device_timing_aggregation(),
        constants.XprofDeviceTimingAggregation.GLOBAL_SLOWEST_PER_STEP,
    )

  def test_xprof_timing_uses_slowest_device_within_scope(self):
    """Multi-device HBM XProf bandwidth uses the slowest in-scope device."""
    # copy with 1024 bf16 elements moves 4096 bytes per device.
    single_dev_bytes = 4096.0
    real_local_devices = list(jax.local_devices())
    fake_devices = []
    for idx, real_dev in enumerate(real_local_devices):
      proxy = mock.MagicMock(wraps=real_dev)
      proxy.id = idx
      proxy.process_index = 0
      proxy.local_hardware_id = idx
      proxy.coords = (idx // 2, 0, 0)
      fake_devices.append(proxy)

    per_device_durations_ms = {
        0: [0.20, 0.20, 0.20],  # Fast first die on chip 0
        1: [0.40, 0.40, 0.40],  # Throttled second die on chip 0
        2: [0.30, 0.30, 0.30],
        3: [0.35, 0.35, 0.35],
        4: [0.25, 0.25, 0.25],
        5: [0.50, 0.50, 0.50],  # Slowest die on host (out of scope for chip 0)
        6: [0.28, 0.28, 0.28],
        7: [0.32, 0.32, 0.32],
    }

    test_report_utils.patch_xprof_trace_loading(self)
    queried_ids: list[int | None] = []

    def parse_for_scope(unused_trace, local_device_id=0):
      queried_ids.append(local_device_id)
      return list(per_device_durations_ms[local_device_id])

    with (
        mock.patch("jax.profiler.trace", return_value=contextlib.nullcontext()),
        mock.patch(
            "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
            return_value=None,
        ),
        mock.patch.object(
            profiler.XprofTrace,
            "device_step_durations_ms",
            autospec=True,
            side_effect=parse_for_scope,
        ),
    ):
      # 1. scope=device: parses only device 0 (0.20 ms).
      bm_device = hbm.HBMBandwidthBenchmark(
          config=hbm.HBMBandwidthParams(
              op_type="copy",
              num_elements=1024,
              dtype="bfloat16",
              scope="device",
              warmup_tries=0,
              num_runs=3,
          ),
          hardware_spec=system.TPU7X_HARDWARE_SPEC,
          mesh=self.mock_mesh,
          xprof_config=base.XprofConfig(xprof_timing=True),
      )
      with (
          mock.patch.object(bm_device, "setup"),
          mock.patch.object(bm_device, "generate_inputs", return_value=()),
          mock.patch.object(bm_device, "run_op", return_value=jnp.zeros((1,))),
          mock.patch.object(jax, "local_devices", return_value=fake_devices),
      ):
        bm_device.spec = hbm.HBM_KERNELS[hbm.HBMKernelOp.COPY]
        res_device = bm_device.run()

      self.assertEqual(queried_ids, [0])
      self.assertAlmostEqual(res_device.metrics["xprof_p50_ms"], 0.20)
      self.assertAlmostEqual(
          res_device.metrics["xprof_bandwidth_per_device_gb_s"],
          (single_dev_bytes / 0.0002) / 1e9,
      )
      self.assertEqual(
          res_device.metadata.xprof_config.device_timing_aggregation,
          constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST,
      )

      # 2. scope=chip: parses only chip 0 devices [0, 1], picks slower die 1
      # (0.40 ms), and ignores out-of-scope device 5 (0.50 ms).
      queried_ids.clear()
      bm_chip = hbm.HBMBandwidthBenchmark(
          config=hbm.HBMBandwidthParams(
              op_type="copy",
              num_elements=1024,
              dtype="bfloat16",
              scope="chip",
              warmup_tries=0,
              num_runs=3,
          ),
          hardware_spec=system.TPU7X_HARDWARE_SPEC,
          mesh=self.mock_mesh,
          xprof_config=base.XprofConfig(xprof_timing=True),
      )
      with (
          mock.patch.object(bm_chip, "setup"),
          mock.patch.object(bm_chip, "generate_inputs", return_value=()),
          mock.patch.object(bm_chip, "run_op", return_value=jnp.zeros((1,))),
          mock.patch.object(jax, "local_devices", return_value=fake_devices),
      ):
        bm_chip.spec = hbm.HBM_KERNELS[hbm.HBMKernelOp.COPY]
        res_chip = bm_chip.run()

      self.assertEqual(queried_ids, [0, 1])
      self.assertAlmostEqual(res_chip.metrics["xprof_p50_ms"], 0.40)
      expected_chip_bw = (2 * single_dev_bytes / 0.0004) / 1e9
      self.assertAlmostEqual(
          res_chip.metrics["xprof_bandwidth_per_chip_gb_s"], expected_chip_bw
      )
      self.assertAlmostEqual(
          res_chip.metrics["xprof_bandwidth_per_device_gb_s"],
          expected_chip_bw / 2.0,
      )
      self.assertIsNone(res_chip.metrics["xprof_bandwidth_per_host_gb_s"])
      self.assertEqual(
          res_chip.metadata.xprof_config.device_timing_aggregation,
          constants.XprofDeviceTimingAggregation.LOCAL_SLOWEST_DEVICE,
      )

      # 3. scope=host: parses all 8 devices [0..7] and picks slowest die 5
      # (0.50 ms).
      queried_ids.clear()
      bm_host = hbm.HBMBandwidthBenchmark(
          config=hbm.HBMBandwidthParams(
              op_type="copy",
              num_elements=1024,
              dtype="bfloat16",
              scope="host",
              warmup_tries=0,
              num_runs=3,
          ),
          hardware_spec=system.TPU7X_HARDWARE_SPEC,
          mesh=self.mock_mesh,
          xprof_config=base.XprofConfig(xprof_timing=True),
      )
      with (
          mock.patch.object(bm_host, "setup"),
          mock.patch.object(bm_host, "generate_inputs", return_value=()),
          mock.patch.object(bm_host, "run_op", return_value=jnp.zeros((1,))),
          mock.patch.object(jax, "local_devices", return_value=fake_devices),
      ):
        bm_host.spec = hbm.HBM_KERNELS[hbm.HBMKernelOp.COPY]
        res_host = bm_host.run()

      self.assertEqual(queried_ids, list(range(8)))
      self.assertAlmostEqual(res_host.metrics["xprof_p50_ms"], 0.50)
      expected_host_bw = (8 * single_dev_bytes / 0.0005) / 1e9
      self.assertAlmostEqual(
          res_host.metrics["xprof_bandwidth_per_host_gb_s"], expected_host_bw
      )
      self.assertAlmostEqual(
          res_host.metrics["xprof_bandwidth_per_chip_gb_s"],
          (expected_host_bw / 8.0) * 2.0,
      )
      self.assertAlmostEqual(
          res_host.metrics["xprof_bandwidth_per_device_gb_s"],
          expected_host_bw / 8.0,
      )
      self.assertEqual(
          res_host.metadata.xprof_config.device_timing_aggregation,
          constants.XprofDeviceTimingAggregation.LOCAL_SLOWEST_DEVICE,
      )

  @parameterized.parameters(
      ("copy", "device", 5),
      ("write_only", "device", 7),
      ("read_only", "chip", 2),
      ("triad", "chip", 3),
  )
  def test_nonzero_local_index_sharding_and_xprof_dir(
      self, op_type, scope, local_index
  ):
    """Non-zero local_index places shards on the selected unit and tags XProf dir."""
    local_devices = list(jax.local_devices())
    expected_devices = (
        [local_devices[local_index]]
        if scope == "device"
        else local_devices[local_index * 2 : (local_index + 1) * 2]
    )
    temp_dir = self.create_tempdir().full_path
    config = hbm.HBMBandwidthParams(
        op_type=op_type,
        num_elements=1024,
        scope=scope,
        local_index=local_index,
        warmup_tries=1,
        num_runs=2,
    )
    bm = hbm.HBMBandwidthBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
        xprof_config=base.XprofConfig(xprof_timing=True, xprof_dir=temp_dir),
    )
    bm.setup()
    self.assertEqual(bm.get_devices_to_measure(), expected_devices)
    self.assertEqual(bm.get_device_to_measure(), expected_devices[0])

    inputs = bm.generate_inputs()
    for inp in inputs:
      self.assertEqual(inp.devices(), set(expected_devices))
    out = bm.run_op(*inputs)
    out.block_until_ready()
    self.assertEqual(out.devices(), set(expected_devices))

    mock_load_trace, _ = test_report_utils.patch_xprof_trace_loading(self)
    with (
        mock.patch("jax.profiler.trace", return_value=contextlib.nullcontext()),
        mock.patch("jax.profiler.stop_trace"),
        mock.patch(
            "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
            return_value="http://mock_xprof_url",
        ),
        mock.patch.object(
            profiler.XprofTrace,
            "device_step_durations_ms",
            autospec=True,
            return_value=[0.5, 0.5],
        ) as mock_durations,
    ):
      result = bm.run()

    mock_load_trace.assert_called_once_with(
        bm._xprof_dir_actual,  # pylint: disable=protected-access
        bm.match_xprof_op_fallback,
    )
    self.assertEqual(
        [
            call.kwargs["local_device_id"]
            for call in mock_durations.call_args_list
        ],
        [d.local_hardware_id for d in expected_devices],
    )

    self.assertEqual(result.metadata.params["scope"], scope)
    self.assertEqual(result.metadata.params["local_index"], local_index)
    self.assertIn(
        f"_scope_{scope}_idx_{local_index}_",
        bm._xprof_dir_actual,  # pylint: disable=protected-access
    )

  @parameterized.parameters(
      itertools.product(
          ("copy", "scale", "add", "triad", "read_only", "write_only"),
          ("device", "chip", "host"),
      )
  )
  def test_scope_sharding_all_ops(self, op_type, scope):
    """All kernels execute and shard across the scope's devices."""
    self._setup_benchmark(
        op_type=op_type,
        num_elements=1024,
        scope=scope,
    )
    expected_devices = self.bm.get_devices_to_measure()
    num_devices = len(expected_devices)
    expected_spec = jax.sharding.PartitionSpec("device")

    inputs = self.bm.generate_inputs()
    for inp in inputs:
      self.assertEqual(inp.shape, (num_devices * 1024,))
      self.assertEqual(inp.addressable_shards[0].data.shape, (1024,))
      self.assertEqual(inp.devices(), set(expected_devices))
      self.assertEqual(inp.sharding.spec, expected_spec)

    out = self.bm.run_op(*inputs)
    out.block_until_ready()
    self.assertEqual(out.devices(), set(expected_devices))
    self.assertEqual(out.sharding.spec, expected_spec)
    if op_type == "read_only":
      self.assertEqual(out.shape, (num_devices,))
      self.assertEqual(out.addressable_shards[0].data.shape, (1,))
    else:
      self.assertEqual(out.shape, (num_devices * 1024,))
      self.assertEqual(out.addressable_shards[0].data.shape, (1024,))

  @parameterized.parameters(
      "copy", "scale", "add", "triad", "read_only", "write_only"
  )
  def test_stream_ops_execution(self, op_type):
    """Verify that all operations generate correct inputs and execute."""
    self._setup_benchmark(op_type)
    inputs = self.bm.generate_inputs()

    if op_type in ("add", "triad"):
      self.assertLen(inputs, 2)
      self.assertEqual(inputs[0].shape, (1024,))
      self.assertEqual(inputs[1].shape, (1024,))
    elif op_type == "write_only":
      self.assertEmpty(inputs)
    else:
      self.assertLen(inputs, 1)
      self.assertEqual(inputs[0].shape, (1024,))

  @parameterized.product(
      op_type=("copy", "scale", "add", "triad", "write_only"),
      dtype=("int8", "uint8"),
  )
  def test_integer_dtypes_keep_output_dtype(self, op_type, dtype):
    """Verify integer inputs are generated and outputs keep the dtype."""
    self._setup_benchmark(op_type, dtype=dtype)
    out = self.bm.run_op(*self.bm.generate_inputs())
    self.assertEqual(out.dtype, jnp.dtype(dtype))

  @parameterized.parameters("int8", "uint8", "int32", "bfloat16", "float32")
  def test_copy_adds_one_in_same_dtype(self, dtype):
    """Verify copy computes x + 1 without promotion (x + 0 would fold away)."""
    self._setup_benchmark("copy", dtype=dtype)

    out = self.bm.run_op(jnp.array([0, 3, 5], dtype=dtype))

    self.assertEqual(out.dtype, jnp.dtype(dtype))
    np.testing.assert_array_equal(out, np.array([1, 4, 6], dtype=dtype))

  @parameterized.parameters("int8", "uint8", "int32", "bfloat16", "float32")
  def test_scalar_draw_range(self, dtype):
    """Verify the scalar is drawn from [2.0, 5.0]: `x * 1` would fold away."""
    with mock.patch.object(
        hbm.random, "uniform", autospec=True, return_value=2.0
    ) as mock_uniform:
      self._setup_benchmark("scale", dtype=dtype)
    mock_uniform.assert_called_once_with(2.0, 5.0)
    self.assertEqual(self.bm.scalar.dtype, jnp.dtype(dtype))
    self.assertEqual(float(self.bm.scalar), 2.0)

  @parameterized.parameters("int8", "uint8", "int32")
  def test_integer_scalar_truncates_to_at_least_two(self, dtype):
    """Verify integer scalars truncate to [2, 5], never to 1."""
    for draw, expected in ((2.9, 2), (5.0, 5)):
      with mock.patch.object(
          hbm.random, "uniform", autospec=True, return_value=draw
      ):
        self._setup_benchmark("scale", dtype=dtype)
      self.assertEqual(int(self.bm.scalar), expected)

    with mock.patch.object(
        hbm.random, "uniform", autospec=True, return_value=2.9
    ):
      self._setup_benchmark("scale", dtype=dtype)
    out = self.bm.run_op(jnp.array([0, 1, 3], dtype=dtype))

    np.testing.assert_array_equal(out, np.array([0, 2, 6], dtype=dtype))

  def test_bool_dtype_raises_error(self):
    """Verify bool is rejected, as copy/scale would constant-fold."""
    with self.assertRaisesRegex(
        ValueError,
        r"^dtype: Unsupported dtype 'bool'\. Supported dtypes for dtype:",
    ):
      hbm.HBMBandwidthParams(dtype="bool")

  def test_read_and_write_values(self):
    """Verify numerical correctness of read_only and write_only ops."""
    # Test write_only: should produce array filled with scalar
    self._setup_benchmark("write_only")
    scalar = self.bm.scalar
    self.assertIsNotNone(scalar)
    out_write = self.bm.run_op()
    self.assertEqual(out_write.shape, (1024,))
    self.assertEqual(out_write.dtype, jnp.bfloat16)
    np.testing.assert_allclose(out_write, np.full((1024,), scalar), rtol=1e-2)

    # Test read_only: should perform per-device reduction of input array
    self._setup_benchmark("read_only")
    x_ones = jnp.ones((1024,), dtype=jnp.bfloat16)
    out_read = self.bm.run_op(x_ones)
    self.assertEqual(out_read.shape, (1,))
    self.assertTrue(bool(out_read[0]))

    x_zeros = jnp.zeros((1024,), dtype=jnp.bfloat16)
    out_read_zeros = self.bm.run_op(x_zeros)
    self.assertEqual(out_read_zeros.shape, (1,))
    self.assertFalse(bool(out_read_zeros[0]))

    # Multi-device read_only (chip scope, 2 devices): each device reduces its
    # own 1D shard independently without cross-device reduction.
    self._setup_benchmark("read_only", scope="chip")
    x_mixed = jax.device_put(
        jnp.concatenate([
            jnp.zeros(1024, dtype=jnp.bfloat16),
            jnp.ones(1024, dtype=jnp.bfloat16),
        ]),
        self.bm.sharding,
    )
    out_mixed = self.bm.run_op(x_mixed)
    self.assertEqual(out_mixed.shape, (2,))
    np.testing.assert_array_equal(np.asarray(out_mixed), [False, True])

  def test_hlo_lowering_read_and_write(self):
    """Verify HLO lowering: 0 all-reduces for read, 0 reads for write."""
    # 1. READ-only: verify 1D per-partition reduce with zero all-reduce
    scopes_and_counts = (
        ("device", 1),
        ("chip", 2),
        ("host", _NUM_CPU_DEVICES),
    )
    for scope, num_devs in scopes_and_counts:
      self._setup_benchmark("read_only", scope=scope)
      (x,) = self.bm.generate_inputs()
      hlo_read = (
          self.bm._jit_fn.lower(x)  # pylint: disable=protected-access
          .compile()
          .as_text()
      )
      self.assertIn("reduce", hlo_read.lower())
      self.assertNotIn("all-reduce", hlo_read.lower())
      self.assertIn("bf16[1024]", hlo_read)
      self.assertNotIn("bf16[1,1024]", hlo_read)
      self.assertIn("ROOT", hlo_read)
      self.assertRegex(
          hlo_read,
          r"ENTRY\s+%[a-zA-Z0-9_.]+\s+\([^)]+\)\s+->\s+pred\[1\]",
      )
      if num_devs > 1:
        self.assertIn(f"num_partitions={num_devs}", hlo_read)

    # 2. WRITE-only: verify broadcast/constant generation with 0 inputs
    self._setup_benchmark("write_only")
    hlo_write = (
        self.bm._jit_fn.lower()  # pylint: disable=protected-access
        .compile()
        .as_text()
    )
    self.assertIn("broadcast", hlo_write.lower())
    # Entry computation layout and signature must take 0 inputs: () -> ...
    self.assertIn("entry_computation_layout={()->", hlo_write)
    self.assertRegex(hlo_write, r"ENTRY\s+%[a-zA-Z0-9_.]+\s+\(\)\s+->")

  def test_run_op(self):
    """Verify that running the op returns the expected shape."""
    self._setup_benchmark()
    inputs = self.bm.generate_inputs()
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (1024,))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_random_scalar(self):
    """Verify internal random scalar works as expected for scale and triad ops."""
    x = jnp.ones((1024,), dtype=jnp.bfloat16)

    # Test scale: y = scalar * x
    self._setup_benchmark("scale")
    scalar = self.bm.scalar
    self.assertIsNotNone(scalar)
    out_scale = self.bm.run_op(x)
    np.testing.assert_allclose(out_scale, np.ones((1024,)) * scalar, rtol=1e-2)

    # Test triad: z = x + scalar * y
    x_zeros = jnp.zeros((1024,), dtype=jnp.bfloat16)
    y_ones = jnp.ones((1024,), dtype=jnp.bfloat16)
    self._setup_benchmark("triad")
    scalar = self.bm.scalar
    self.assertIsNotNone(scalar)
    out_triad = self.bm.run_op(x_zeros, y_ones)
    np.testing.assert_allclose(out_triad, np.ones((1024,)) * scalar, rtol=1e-2)

  def test_run_op_uninitialized(self):
    """Verify calling run_op before setup raises a ValueError."""
    config = hbm.HBMBandwidthParams()
    uninitialized_bm = hbm.HBMBandwidthBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC
    )
    with self.assertRaisesRegex(ValueError, "JIT function not initialized."):
      uninitialized_bm.run_op(jnp.ones((1024,)))

  @parameterized.named_parameters(
      ("device", "device", 1),
      ("chip", "chip", 2),
      ("host", "host", _NUM_CPU_DEVICES),
  )
  def test_setup_preserves_base_mesh_and_sets_scope_mesh(
      self, scope, expected_num_devices
  ):
    """Verify setup() keeps BaseBenchmark.mesh intact and populates scope_mesh."""
    config = hbm.HBMBandwidthParams(
        op_type="copy",
        num_elements=64,
        warmup_tries=1,
        num_runs=1,
        scope=scope,
    )
    bm = hbm.HBMBandwidthBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.assertIs(bm.mesh, self.mock_mesh)
    self.assertIsNone(bm.scope_mesh)
    self.assertIsNone(bm.sharding)

    bm.setup()
    self.assertIs(bm.mesh, self.mock_mesh)
    self.assertIsNotNone(bm.scope_mesh)
    self.assertEqual(
        list(bm.scope_mesh.devices.flat), bm.get_devices_to_measure()
    )
    self.assertLen(list(bm.scope_mesh.devices.flat), expected_num_devices)
    self.assertIsNotNone(bm.sharding)
    self.assertEqual(bm.sharding.mesh, bm.scope_mesh)
    self.assertEqual(bm.sharding.spec, jax.sharding.PartitionSpec("device"))

    # Also verify BaseBenchmark._run()'s default global mesh is not overwritten.
    bm_default_mesh = hbm.HBMBandwidthBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC
    )
    bm_default_mesh.run()
    self.assertIsNotNone(bm_default_mesh.mesh)
    self.assertEqual(
        list(bm_default_mesh.mesh.devices.flat), list(jax.devices())
    )
    self.assertIsNotNone(bm_default_mesh.scope_mesh)
    self.assertEqual(
        list(bm_default_mesh.scope_mesh.devices.flat),
        bm_default_mesh.get_devices_to_measure(),
    )

  def test_generate_inputs_requires_sharding(self):
    """Verify generate_inputs() asserts when setup() / sharding is missing."""
    config = hbm.HBMBandwidthParams(num_elements=64)
    uninitialized_bm = hbm.HBMBandwidthBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC
    )
    with self.assertRaises(AssertionError):
      uninitialized_bm.generate_inputs()

    self._setup_benchmark("copy", num_elements=64, scope="chip")
    (x,) = self.bm.generate_inputs()
    self.assertEqual(x.sharding, self.bm.sharding)

    self.bm.sharding = None
    with self.assertRaises(AssertionError):
      self.bm.generate_inputs()

  def test_get_total_bytes(self):
    """Verify the byte calculation for 1-input vs 2-input ops and scopes."""
    # Copy/Scale: 1024 * 2 bytes/element * 2 arrays = 4096 bytes/device
    self._setup_benchmark("copy")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 4096.0)

    self._setup_benchmark("copy", scope="chip")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 4096.0 * 2)

    self._setup_benchmark("copy", scope="host")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 4096.0 * _NUM_CPU_DEVICES)

    self._setup_benchmark("scale")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 4096.0)

    # Add/Triad: num_elements 1024 * 2 bytes/element * 3 arrays = 6144 bytes
    self._setup_benchmark("add")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 6144.0)

    self._setup_benchmark("triad")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 6144.0)

    # Read/Write only: 1024 * 2 bytes/element * 1 array = 2048 bytes
    self._setup_benchmark("read_only")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 2048.0)

    self._setup_benchmark("write_only")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 2048.0)

  def test_get_arithmetic_intensity(self):
    """Verify arithmetic intensity calculations across all STREAM ops."""
    self._setup_benchmark("copy")
    # 1 FLOP / 2 bytes
    self.assertAlmostEqual(self.bm.get_arithmetic_intensity(), 0.25)

    self._setup_benchmark("scale")
    # 1 FLOP / 2 bytes
    self.assertAlmostEqual(self.bm.get_arithmetic_intensity(), 0.25)

    self._setup_benchmark("add")
    # 1 FLOP / (2 bytes * 3 arrays)
    self.assertAlmostEqual(self.bm.get_arithmetic_intensity(), 1.0 / 6.0)

    self._setup_benchmark("triad")
    # 2 FLOP / (2 bytes * 3 arrays)
    self.assertAlmostEqual(self.bm.get_arithmetic_intensity(), 1.0 / 3.0)

    self._setup_benchmark("read_only")
    # 1 FLOP / (2 bytes * 1 array)
    self.assertAlmostEqual(self.bm.get_arithmetic_intensity(), 0.5)

    self._setup_benchmark("write_only")
    # 0 FLOP / (2 bytes * 1 array)
    self.assertAlmostEqual(self.bm.get_arithmetic_intensity(), 0.0)

  @parameterized.parameters(
      ("copy", "device", system.TPU7X_HARDWARE_SPEC, 1),
      ("scale", "device", system.TPU7X_HARDWARE_SPEC, 1),
      ("add", "chip", system.TPU7X_HARDWARE_SPEC, 2),
      ("triad", "host", system.TPU7X_HARDWARE_SPEC, _NUM_CPU_DEVICES),
      ("read_only", "chip", system.TPU7X_HARDWARE_SPEC, 2),
      ("read_only", "chip", system.V6E_HARDWARE_SPEC, 1),
      ("triad", "host", system.V6E_HARDWARE_SPEC, _NUM_CPU_DEVICES),
      ("write_only", "device", system.TPU7X_HARDWARE_SPEC, 1),
  )
  def test_calculate_metrics(
      self, op_type, scope, hw_spec, expected_num_devices
  ):
    """Verify that metrics are correctly calculated across ops and scopes."""
    times_ms = [10.0, 10.0, 10.0]
    self._setup_benchmark(op_type, hardware_spec=hw_spec, scope=scope)
    metrics = self.bm.calculate_metrics(times_ms)

    if op_type in ("copy", "scale"):
      single_dev_bytes = 4096.0
    elif op_type in ("add", "triad"):
      single_dev_bytes = 6144.0
    else:
      single_dev_bytes = 2048.0
    total_bytes = single_dev_bytes * expected_num_devices
    expected_bw_gb_s = (total_bytes / 0.01) / 1e9

    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)
    self.assertEqual(metrics["op_type"], op_type)
    self.assertEqual(metrics["num_devices"], expected_num_devices)
    self.assertAlmostEqual(
        metrics[f"wall_clock_bandwidth_per_{scope}_gb_s"], expected_bw_gb_s
    )
    if scope == "device":
      self.assertIsNone(metrics["wall_clock_bandwidth_per_chip_gb_s"])
      self.assertIsNone(metrics["wall_clock_bandwidth_per_host_gb_s"])
    elif scope == "chip":
      self.assertAlmostEqual(
          metrics["wall_clock_bandwidth_per_device_gb_s"],
          expected_bw_gb_s / hw_spec.devices_per_chip,
      )
      self.assertIsNone(metrics["wall_clock_bandwidth_per_host_gb_s"])
    else:
      expected_per_dev = expected_bw_gb_s / expected_num_devices
      self.assertAlmostEqual(
          metrics["wall_clock_bandwidth_per_device_gb_s"], expected_per_dev
      )
      self.assertAlmostEqual(
          metrics["wall_clock_bandwidth_per_chip_gb_s"],
          expected_per_dev * hw_spec.devices_per_chip,
      )
    self.assertAlmostEqual(
        metrics["data_size_mib"], single_dev_bytes / (1024 * 1024)
    )
    self.assertNotIn("total_bytes_mib", metrics)

  def test_format_benchmark_table(self):
    """Tests formatting of HBM bandwidth benchmark tables."""
    res = base.BenchmarkResult(
        metadata=base.BenchmarkMetadata(
            benchmark_name="HBMBandwidthBenchmark",
            test_name="HBMBandwidthBenchmark_test",
            start_time="2026-08-18T10:00:00",
            end_time="2026-08-18T10:01:00",
            params={
                "num_elements": 134217728,
                "dtype": "bfloat16",
                "op_type": "copy",
                "scope": "device",
                "local_index": 0,
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "num_devices": 1,
            "data_size_mib": 256.00,
            "wall_clock_p50_ms": 0.07112,
            "wall_clock_bandwidth_per_device_gb_s": 3769.107,
            "wall_clock_bandwidth_per_chip_gb_s": None,
            "wall_clock_bandwidth_per_host_gb_s": None,
            "xprof_bandwidth_per_device_gb_s": 3800.000,
            "xprof_bandwidth_per_chip_gb_s": None,
            "xprof_bandwidth_per_host_gb_s": None,
            "xprof_p50_ms": 0.06543,
        },
        timings=base.BenchmarkTimings(
            wall_clock=base.WallClockTimings(durations_ms=[1.0])
        ),
    )
    expected_cols = [
        "dtype",
        "op_type",
        "scope",
        "local_index",
        "num_devices",
        "num_elements",
        "data_size_mib",
        "wall_clock_p50_ms",
        "wall_clock_bandwidth_per_device_gb_s",
        "wall_clock_bandwidth_per_chip_gb_s",
        "wall_clock_bandwidth_per_host_gb_s",
        "xprof_p50_ms",
        "xprof_bandwidth_per_device_gb_s",
        "xprof_bandwidth_per_chip_gb_s",
        "xprof_bandwidth_per_host_gb_s",
    ]
    schema_cols = [col for col, _ in hbm.HBMBandwidthBenchmark.REPORT_SCHEMA]
    self.assertEqual(schema_cols, expected_cols)

    df = report.results_to_dataframe([res])
    table = report.format_benchmark_table(
        df,
        schema=hbm.HBMBandwidthBenchmark.REPORT_SCHEMA,
        title="HBMBandwidthBenchmark",
    )
    self.assertIn("Benchmark Results (HBMBandwidthBenchmark)", table)
    for col in expected_cols:
      self.assertIn(col, table)
    self.assertIn("bfloat16", table)
    self.assertIn("copy", table)
    self.assertIn("device", table)
    self.assertIn("134217728", table)
    self.assertIn("256.00", table)
    self.assertNotIn("total_bytes_mib", table)
    self.assertIn("3769.11", table)
    self.assertIn("3800.00", table)
    self.assertIn("0.0711", table)
    self.assertIn("0.0654", table)
    self.assertIn("-", table)
    self.assertNotIn("None", table)
    self.assertLen(table.splitlines()[0], 304)

  @parameterized.named_parameters(
      ("device", "device", 1),
      ("chip", "chip", 2),
      ("host", "host", _NUM_CPU_DEVICES),
  )
  def test_run_e2e_roofline_metrics(self, scope, expected_num_devices):
    """Verifies that HBMBandwidthBenchmark emits memory roofline metrics across scopes."""
    self._setup_benchmark("copy", scope=scope)
    result = self.bm.run()
    self.assertEqual(result.metrics["num_devices"], expected_num_devices)
    active_bw = result.metrics[f"wall_clock_bandwidth_per_{scope}_gb_s"]
    self.assertGreater(active_bw, 0.0)
    if scope == "device":
      self.assertIsNone(result.metrics["wall_clock_bandwidth_per_chip_gb_s"])
      self.assertIsNone(result.metrics["wall_clock_bandwidth_per_host_gb_s"])
    elif scope == "chip":
      self.assertAlmostEqual(
          result.metrics["wall_clock_bandwidth_per_device_gb_s"],
          active_bw / 2.0,
      )
      self.assertIsNone(result.metrics["wall_clock_bandwidth_per_host_gb_s"])
    else:
      self.assertAlmostEqual(
          result.metrics["wall_clock_bandwidth_per_device_gb_s"],
          active_bw / _NUM_CPU_DEVICES,
      )
      self.assertAlmostEqual(
          result.metrics["wall_clock_bandwidth_per_chip_gb_s"],
          (active_bw / _NUM_CPU_DEVICES) * 2.0,
      )
    self.assertEqual(result.metrics["peak_hbm_bw_per_device_gb_s"], 3690.0)
    expected_eff_pct = (active_bw / (expected_num_devices * 3690.0)) * 100.0
    self.assertAlmostEqual(
        result.metrics["wall_clock_memory_roofline_efficiency_pct"],
        expected_eff_pct,
    )
    self.assertLessEqual(
        result.metrics["wall_clock_memory_roofline_efficiency_pct"], 100.0
    )
    self.assertNotIn("roofline_tflops_limit_per_device", result.metrics)
    self.assertNotIn(
        "wall_clock_compute_roofline_efficiency_pct", result.metrics
    )

  @parameterized.parameters("device", "chip", "host")
  def test_schema_coverage(self, scope):
    """Verify REPORT_SCHEMA matches output keys and covers all metrics."""
    self._setup_benchmark("copy", scope=scope)
    test_report_utils.assert_schema_matches_output(self, self.bm)

  @parameterized.parameters(
      constants.XprofDeviceTimingAggregation.LOCAL_SLOWEST_DEVICE,
      constants.XprofDeviceTimingAggregation.GLOBAL_SLOWEST_PER_STEP,
  )
  def test_rejects_unsupported_device_timing_aggregation(self, aggregation):
    """Single-device scopes reject multi-device aggregations; multi-device scopes allow all."""
    for scope, hw_spec in (
        ("device", system.TPU7X_HARDWARE_SPEC),
        ("chip", system.V6E_HARDWARE_SPEC),
    ):
      bm = hbm.HBMBandwidthBenchmark(
          config=hbm.HBMBandwidthParams(num_elements=64, scope=scope),
          hardware_spec=hw_spec,
          mesh=self.mock_mesh,
          xprof_config=base.XprofConfig(
              xprof_timing=True, device_timing_aggregation=aggregation
          ),
      )
      self.assertEqual(
          bm.supported_device_timing_aggregations,
          frozenset(
              {constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST}
          ),
      )
      with (
          mock.patch.object(bm, "setup") as mock_setup,
          self.assertRaisesRegex(
              ValueError,
              r"HBMBandwidthBenchmark does not support"
              r" xprof_device_timing_aggregation=.*; supported:"
              r" measured_device_or_first\.",
          ),
      ):
        bm.run()
      mock_setup.assert_not_called()

    for scope in ("chip", "host"):
      bm_multi = hbm.HBMBandwidthBenchmark(
          config=hbm.HBMBandwidthParams(num_elements=64, scope=scope),
          hardware_spec=system.TPU7X_HARDWARE_SPEC,
          mesh=self.mock_mesh,
          xprof_config=base.XprofConfig(
              xprof_timing=True, device_timing_aggregation=aggregation
          ),
      )
      self.assertEqual(
          bm_multi.supported_device_timing_aggregations,
          frozenset(constants.XprofDeviceTimingAggregation),
      )
      bm_multi._validate_device_timing_aggregation()  # pylint: disable=protected-access


class HBMBandwidthParamsValidationTest(parameterized.TestCase):
  """Verifies the bounds declared on HBMBandwidthParams fields."""

  @parameterized.parameters(0, -1)
  def test_non_positive_num_elements_raises_error(self, num_elements):
    with self.assertRaisesRegex(ValueError, "num_elements must be >= 1"):
      hbm.HBMBandwidthParams(num_elements=num_elements)

  @parameterized.parameters("rack", "DEVICE", 0, -1, None)
  def test_invalid_scope_raises_error(self, scope):
    with self.assertRaisesRegex(
        ValueError,
        rf"Invalid scope '{scope}'\. Must be one of \['device', 'chip',"
        r" 'host'\]",
    ):
      hbm.HBMBandwidthParams(scope=scope)

  def test_scope_coerces_to_enum(self):
    self.assertTrue(issubclass(hbm.HBMBandwidthParams, base.ScopeParamsMixin))
    default_params = hbm.HBMBandwidthParams()
    self.assertEqual(default_params.scope, constants.Scope.DEVICE)
    self.assertEqual(default_params.local_index, 0)

    params = hbm.HBMBandwidthParams(scope="chip", local_index=2)
    self.assertEqual(params.scope, constants.Scope.CHIP)
    self.assertIsInstance(params.scope, constants.Scope)
    self.assertEqual(params.local_index, 2)

  def test_invalid_local_index_raises_error(self):
    with self.assertRaisesRegex(ValueError, "local_index must be >= 0"):
      hbm.HBMBandwidthParams(local_index=-1)
    with self.assertRaisesRegex(
        ValueError, "local_index must be an integer >= 0"
    ):
      hbm.HBMBandwidthParams(local_index=True)

  def test_invalid_op_type_raises_error(self):
    with self.assertRaisesRegex(ValueError, "Invalid op_type 'invalid_kernel'"):
      hbm.HBMBandwidthParams(op_type="invalid_kernel")

  def test_complex_dtype_raises_error(self):
    """Verify complex is rejected, as it reports above the HBM peak."""
    with self.assertRaisesRegex(
        ValueError,
        r"^dtype: Unsupported dtype 'complex64'\. Supported dtypes for dtype:",
    ):
      hbm.HBMBandwidthParams(dtype="complex64")

  @parameterized.parameters("int4", "float4_e2m1fn")
  def test_sub_byte_dtype_raises_error(self, dtype):
    """Verify sub-byte dtypes are rejected, as bytes count 1 per element."""
    with self.assertRaisesRegex(
        ValueError, f"^dtype: Unsupported sub-byte dtype '{dtype}'"
    ):
      hbm.HBMBandwidthParams(dtype=dtype)

  def test_op_type_coerces_to_enum(self):
    params = hbm.HBMBandwidthParams(op_type="scale")
    self.assertEqual(params.op_type, hbm.HBMKernelOp.SCALE)
    self.assertIsInstance(params.op_type, hbm.HBMKernelOp)

  @parameterized.named_parameters(
      (op.name.lower(), op.value) for op in hbm.HBMKernelOp
  )
  def test_op_type_help_message_and_default_value(self, choice: str):
    """Verifies help message includes choice and matching default."""
    parser = argparse.ArgumentParser()
    cli.add_dataclass_arguments(parser, hbm.HBMBandwidthParams)
    s = io.StringIO()
    parser.print_help(file=s)
    help_msg = s.getvalue()

    normalized_help = " ".join(help_msg.split())
    self.assertIn(
        choice,
        help_msg,
        f"Enum value '{choice}' missing from help message: {help_msg}",
    )

    default_op = hbm.HBMBandwidthParams().op_type.value
    self.assertIn(
        f"(default: {default_op})",
        normalized_help,
        f"Displayed default '{default_op}' missing from help: {help_msg}",
    )


if __name__ == "__main__":
  absltest.main()
