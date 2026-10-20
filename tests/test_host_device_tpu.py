"""Unit tests for host_device.py that requires TPU devices."""

import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import host_device
from accelerator_microbenchmarks.core import (
    base,
)
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp
import numpy as np


class DeviceToHostBenchmarkTest(parameterized.TestCase):
  """Unit tests for DeviceToHostBenchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    # Use small runs for testing to limit memory. data_size_mib is per device,
    # so pin a single device to keep the expected shapes below.
    self.params = {
        "data_size_mib": 4,
        "num_runs": 2,
        "warmup_tries": 1,
        "local_device": 0,
    }

  def is_array_on_tpu(self, my_array):
    """Checks if any device associated with the array is a TPU."""
    device_set = my_array.devices()

    if not device_set:
      # Array has no associated devices, so not on TPU.
      return False

    for device in device_set:
      # Check for TPU using various attributes
      if hasattr(device, "platform") and "tpu" in device.platform.lower():
        return True
      if "Tpu" in type(device).__name__:
        return True
      if hasattr(device, "device_kind") and "tpu" in device.device_kind.lower():
        return True

    return False

  def test_reset_data(self):
    """Verify reset_data deletes the old array and allocates a new one."""
    if "tpu" not in jax.devices()[0].platform:
      self.fail("This test is meaninglesss on CPU.")

    d2h_params = {
        k: v
        for k, v in self.params.items()
        if k not in ("num_runs", "warmup_tries")
    }
    config = host_device.HostDeviceParams(**d2h_params)
    self.bm = host_device.DeviceToHostBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.bm.setup()
    self.bm.inputs = list(self.bm.generate_inputs())

    old_array = self.bm.inputs[0]
    self.assertFalse(old_array.is_deleted())

    new_inputs = self.bm.reset_data(*self.bm.inputs)

    # The old array should be deleted
    self.assertTrue(old_array.is_deleted())

    # The new array should be a valid, active JAX Array in HBM
    new_array = new_inputs[0]
    self.assertIsInstance(new_array, jax.Array)
    self.assertFalse(new_array.is_deleted())
    self.assertEqual(new_array.shape, (16384, 128))
    self.assertEqual(new_array.dtype, jnp.bfloat16)
    if not self.is_array_on_tpu(new_array):
      self.fail("New array is not on TPU.")

  def test_run_resets_data(self):
    """Verify that run() actually calls reset_data() to replace and delete arrays."""
    if "tpu" not in jax.devices()[0].platform:
      self.fail("This test is meaninglesss on CPU.")

    # Capture the array returned by generate_inputs
    d2h_params = {
        k: v
        for k, v in self.params.items()
        if k not in ("num_runs", "warmup_tries")
    }
    config = host_device.HostDeviceParams(**d2h_params)
    self.bm = host_device.DeviceToHostBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    original_generate_inputs = self.bm.generate_inputs
    generated_arrays = []

    def wrapped_generate_inputs(**params):
      res = original_generate_inputs(**params)
      generated_arrays.extend(res)
      return res

    self.bm.generate_inputs = wrapped_generate_inputs
    # Run the benchmark
    result = self.bm.run()
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", result.metrics)
    self.assertIn("wall_clock_bandwidth_total_gb_s", result.metrics)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", result.metrics)

    # If reset_data() was executed, the generated array must have been deleted
    self.assertGreater(len(generated_arrays), 1)
    old_array = generated_arrays[0]
    self.assertTrue(old_array.is_deleted(), "Old array was not reset.")

  # 32768 is not included as it will run into timeout errors.
  # Forge tests have limited time for each test.
  @parameterized.named_parameters([
      {"testcase_name": "8GB", "size_mib": 8192},
      {"testcase_name": "16GB", "size_mib": 16384},
  ])
  def test_large_data_sizes(self, size_mib):
    """Verify setup, generate_inputs, and run_op with large sizes (8GB, 16GB)."""
    devices = jax.devices()
    if devices[0].device_kind != "TPU7x" or jax.devices()[0].platform != "tpu":
      print("Skipping test for non-TPU7x devices.")
      self.skipTest("This test is a corner case for TPU v7x.")

    params = {
        "data_size_mib": size_mib,
        "num_runs": 20,
        "warmup_tries": 2,
        "local_device": 0,
    }
    d2h_params = {
        k: v for k, v in params.items() if k not in ("num_runs", "warmup_tries")
    }
    config = host_device.HostDeviceParams(**d2h_params)
    self.bm = host_device.DeviceToHostBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.bm.setup()
    inputs = self.bm.generate_inputs()

    dtype = utils.parse_dtype(config.dtype)
    itemsize = jnp.dtype(dtype).itemsize
    num_elements = (1024 * 1024 * size_mib) // itemsize
    expected_shape = (num_elements // 128, 128)

    out = self.bm.run_op(*inputs)
    self.assertIsInstance(out, np.ndarray)
    self.assertEqual(out.shape, expected_shape)
    self.assertEqual(out.dtype, dtype)


class HostToDeviceBenchmarkTest(parameterized.TestCase):
  """Unit tests for HostToDeviceBenchmark on physical TPU hardware."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    self.patcher = mock.patch(
        "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
        return_value="http://mock_xprof_url",
    )
    self.patcher.start()
    self.addCleanup(self.patcher.stop)

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()

  def test_h2d_bandwidth_below_theoretical_max_v7x(self):
    """Verify H2D bandwidth from XProf timing on TPU v7x is < 60 GB/s."""
    devices = jax.devices()
    if (
        not devices
        or jax.devices()[0].platform != "tpu"
        or devices[0].device_kind != "TPU7x"
    ):
      self.skipTest("This test requires TPU v7x hardware.")

    params = {
        "data_size_mib": 8192,
        "num_runs": 10,
        "warmup_tries": 2,
        "dtype": "float32",
        "local_device": 0,
    }
    config = host_device.HostDeviceParams(**params)
    bm = host_device.HostToDeviceBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
        xprof_config=base.XprofConfig(xprof_timing=True),
    )
    bm.setup()
    result = bm.run()

    self.assertIn("xprof_bandwidth_per_device_gb_s", result.metrics)
    self.assertIn("xprof_bandwidth_total_gb_s", result.metrics)
    self.assertNotIn("xprof_bandwidth_per_chip_gb_s", result.metrics)
    print(f"result: {result}")
    bandwidth_gb_s = result.metrics["xprof_bandwidth_per_device_gb_s"]
    self.assertAlmostEqual(
        result.metrics["xprof_bandwidth_total_gb_s"], bandwidth_gb_s
    )
    print(f"bandwidth_gb_s: {bandwidth_gb_s}")
    self.assertGreater(bandwidth_gb_s, 0.0)
    # Theoretical maximum PCIe Gen5 bandwidth for v7x is 60 GB/s
    self.assertLess(
        bandwidth_gb_s,
        60.0,
        f"Measured H2D bandwidth {bandwidth_gb_s:.2f} GB/s exceeds theoretical"
        " max 60 GB/s.",
    )
    self.assertGreater(
        bandwidth_gb_s,
        0.0,
        f"Measured H2D bandwidth {bandwidth_gb_s:.2f} GB/s is below"
        " min 0.0 GB/s.",
    )


# Theoretical maximum per-device PCIe Gen5 x16 bandwidth on TPU v7x.
_PCIE_MAX_GB_S = 60.0

_TRANSFER_CASES = tuple(
    {
        "testcase_name": f"{direction}_{backend}_dev{dev_name}",
        "bm_cls": bm_cls,
        "transfer_backend": backend,
        "local_device": local_device,
    }
    for (direction, bm_cls), backend, (
        dev_name,
        local_device,
    ) in itertools.product(
        (
            ("h2d", host_device.HostToDeviceBenchmark),
            ("d2h", host_device.DeviceToHostBenchmark),
        ),
        ("tpu_sync", "jax"),
        (("all", -1), ("0", 0)),
    )
)

_TPU_SYNC_CASES = tuple(
    case for case in _TRANSFER_CASES if case["transfer_backend"] == "tpu_sync"
)


class TransferBackendTpuTest(parameterized.TestCase):
  """Covers transfer_backend x local_device for H2D and D2H on TPU."""

  def setUp(self):
    super().setUp()
    if jax.devices()[0].platform != "tpu":
      self.skipTest("This test requires TPU hardware.")
    self.enter_context(
        mock.patch(
            "accelerator_microbenchmarks.core.profiler.upload_xprof_trace",
            return_value="http://mock_xprof_url",
        )
    )

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()

  def _make_benchmark(self, bm_cls, xprof_timing=False, **params):
    config = host_device.HostDeviceParams(**params)
    bm = bm_cls(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        xprof_config=base.XprofConfig(xprof_timing=xprof_timing),
    )
    self.addCleanup(bm.teardown)
    return bm

  def _expected_devices(self, local_device):
    if local_device == -1:
      return list(jax.local_devices())
    return [jax.local_devices()[local_device]]

  def _assert_bandwidth_in_range(self, metrics, key):
    self.assertIn(key, metrics)
    bandwidth_gb_s = metrics[key]
    self.assertGreater(bandwidth_gb_s, 0.0)
    self.assertLess(
        bandwidth_gb_s,
        _PCIE_MAX_GB_S,
        f"{key}={bandwidth_gb_s:.2f} GB/s exceeds the theoretical PCIe max"
        f" {_PCIE_MAX_GB_S} GB/s.",
    )

  @parameterized.named_parameters(*_TPU_SYNC_CASES)
  def test_tpu_sync_setup_allocates_block_major_kv_cache(
      self, bm_cls, transfer_backend, local_device
  ):
    """setup() builds a real KVCacheManager over (1, rows, 128) shards."""
    bm = self._make_benchmark(
        bm_cls,
        data_size_mib=64,
        transfer_backend=transfer_backend,
        local_device=local_device,
    )
    bm.setup()
    devices = self._expected_devices(local_device)
    self.assertIsInstance(
        bm.kv_manager, host_device.kv_cache_manager.KVCacheManager
    )
    self.assertEqual(list(bm.mesh.devices.flat), devices)
    rows = bm.rows_per_device
    self.assertEqual(bm.device_array.shape, (1, len(devices) * rows, 128))
    self.assertEqual(
        bm.device_array.sharding.spec,
        jax.sharding.PartitionSpec(None, "device", None),
    )
    self.assertEqual(bm.device_array.devices(), set(devices))
    for shard in bm.device_array.addressable_shards:
      self.assertEqual(shard.data.shape, (1, rows, 128))
      self.assertIn("tpu", shard.device.platform.lower())

  @parameterized.named_parameters(*_TPU_SYNC_CASES)
  def test_tpu_sync_run_op_repeats(
      self, bm_cls, transfer_backend, local_device
  ):
    """Repeated transfers reuse the registered buffer without deleting it."""
    bm = self._make_benchmark(
        bm_cls,
        data_size_mib=64,
        transfer_backend=transfer_backend,
        local_device=local_device,
    )
    bm.setup()
    inputs = bm.generate_inputs()
    for _ in range(3):
      inputs = bm.reset_data(*inputs)
      jax.block_until_ready(bm.run_op(*inputs))
    self.assertIs(inputs[0], bm.device_array)
    self.assertFalse(bm.device_array.is_deleted())

  @parameterized.named_parameters(*_TRANSFER_CASES)
  def test_run_wall_clock(self, bm_cls, transfer_backend, local_device):
    """run() reports sane per-device bandwidth and releases resources."""
    bm = self._make_benchmark(
        bm_cls,
        data_size_mib=256,
        num_runs=5,
        warmup_tries=2,
        transfer_backend=transfer_backend,
        local_device=local_device,
    )
    result = bm.run()
    expected_devices = self._expected_devices(local_device)
    self._assert_bandwidth_in_range(
        result.metrics, "wall_clock_bandwidth_per_device_gb_s"
    )
    self.assertAlmostEqual(
        result.metrics["wall_clock_bandwidth_total_gb_s"],
        result.metrics["wall_clock_bandwidth_per_device_gb_s"]
        * len(expected_devices),
    )
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", result.metrics)
    self.assertLen(expected_devices, result.metrics["num_devices"])
    self.assertEqual(
        result.metadata.params["transfer_backend"], transfer_backend
    )
    self.assertEqual(result.metadata.params["local_device"], local_device)
    # run() must always release the pinned host staging pool.
    self.assertIsNone(bm.kv_manager)
    self.assertIsNone(bm.device_array)

  @parameterized.named_parameters(*_TPU_SYNC_CASES)
  def test_tpu_sync_run_xprof(self, bm_cls, transfer_backend, local_device):
    """XProf parses the /host:CPU MARKER events for tpu_sync transfers."""
    devices = jax.devices()
    if devices[0].device_kind != "TPU7x":
      self.skipTest("This test requires TPU v7x hardware.")
    bm = self._make_benchmark(
        bm_cls,
        xprof_timing=True,
        data_size_mib=1024,
        num_runs=10,
        warmup_tries=2,
        transfer_backend=transfer_backend,
        local_device=local_device,
    )
    self.assertTrue(bm.xprof_target_host_cpu)
    result = bm.run()
    print(f"{bm.get_run_identifier()}: {result.metrics}")
    self._assert_bandwidth_in_range(
        result.metrics, "xprof_bandwidth_per_device_gb_s"
    )
    self.assertAlmostEqual(
        result.metrics["xprof_bandwidth_total_gb_s"],
        result.metrics["xprof_bandwidth_per_device_gb_s"]
        * len(self._expected_devices(local_device)),
    )
    self.assertNotIn("xprof_bandwidth_per_chip_gb_s", result.metrics)
    self.assertIsNone(bm.kv_manager)

  @parameterized.named_parameters(
      ("h2d", host_device.HostToDeviceBenchmark),
      ("d2h", host_device.DeviceToHostBenchmark),
  )
  def test_tpu_sync_sweep_points_release_pinned_memory(self, bm_cls):
    """Back-to-back sweep points must not accumulate pinned host memory."""
    for _ in range(3):
      bm = self._make_benchmark(
          bm_cls,
          data_size_mib=1024,
          num_runs=2,
          warmup_tries=1,
          transfer_backend="tpu_sync",
          local_device=-1,
      )
      bm.run()
      self.assertIsNone(bm.kv_manager)
      self.assertIsNone(bm.device_array)


if __name__ == "__main__":
  absltest.main()
