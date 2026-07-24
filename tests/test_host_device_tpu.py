"""Unit tests for host_device.py that requires TPU devices."""

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import host_device
import jax
import numpy as np


class DeviceToHostBenchmarkTest(parameterized.TestCase):
  """Unit tests for DeviceToHostBenchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    self.bm = host_device.DeviceToHostBenchmark(mesh=self.mock_mesh)
    # Use small runs for testing to limit memory
    self.params = {
        "data_size_mib": 4,
        "num_runs": 2,
        "warmup_tries": 1,
    }

  def is_array_on_tpu(self, my_array):
    """Checks if any device associated with the array is a TPU."""
    device_set = my_array.devices()

    if not device_set:
      # Array has no associated devices, so not on TPU.
      return False

    for device in device_set:
      # Check for TPU using various attributes
      if hasattr(device, 'platform') and 'tpu' in device.platform.lower():
        return True
      if "Tpu" in type(device).__name__:
        return True
      if hasattr(device, 'device_kind') and 'tpu' in device.device_kind.lower():
        return True

    return False

  def test_reset_data(self):
    """Verify reset_data deletes the old array and allocates a new one."""
    if "tpu" not in jax.devices()[0].platform:
      self.fail("This test is meaninglesss on CPU.")

    self.bm.setup(**self.params)
    self.bm.inputs = list(self.bm.generate_inputs(**self.params))

    old_array = self.bm.inputs[0]
    self.assertFalse(old_array.is_deleted())

    new_inputs = self.bm.reset_data(*self.bm.inputs)

    # The old array should be deleted
    self.assertTrue(old_array.is_deleted())

    # The new array should be a valid, active JAX Array in HBM
    new_array = new_inputs[0]
    self.assertIsInstance(new_array, jax.Array)
    self.assertFalse(new_array.is_deleted())
    self.assertEqual(new_array.shape, (8192, 128))
    if not self.is_array_on_tpu(new_array):
      self.fail("New array is not on TPU.")

  def test_run_resets_data(self):
    """Verify that run() actually calls reset_data() to replace and delete arrays."""
    if "tpu" not in jax.devices()[0].platform:
      self.fail("This test is meaninglesss on CPU.")

    # Capture the array returned by generate_inputs
    original_generate_inputs = self.bm.generate_inputs
    generated_arrays = []

    def wrapped_generate_inputs(**params):
      res = original_generate_inputs(**params)
      generated_arrays.extend(res)
      return res

    self.bm.generate_inputs = wrapped_generate_inputs
    # Run the benchmark
    self.bm.run(**self.params)

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
      self.skipTest("This test is a corner case for Ghostfish.")

    params = {"data_size_mib": size_mib, "num_runs": 20, "warmup_tries": 2}
    self.bm.setup(**params)
    inputs = self.bm.generate_inputs(**params)

    # 1M elements * size_mib / float32_size (4 bytes)
    num_elements = (1024 * 1024 * size_mib) // np.dtype(np.float32).itemsize
    expected_shape = (num_elements // 128, 128)

    out = self.bm.run_op(*inputs)
    self.assertIsInstance(out, np.ndarray)
    self.assertEqual(out.shape, expected_shape)


if __name__ == "__main__":
  absltest.main()