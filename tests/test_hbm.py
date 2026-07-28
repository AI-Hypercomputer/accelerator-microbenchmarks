"""Unit tests for hbm.py."""

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import hbm
from accelerator_microbenchmarks.core import registry
import jax
import jax.numpy as jnp
import numpy as np


# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class HBMBandwidthBenchmarkTest(parameterized.TestCase):
  """Unit tests for hbm.py."""

  def setUp(self):
    super().setUp()
    # Create a dummy mesh for testing on CPU
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    self.bm = hbm.HBMBandwidthBenchmark(mesh=self.mock_mesh)
    self.params = {
        "size": 1024,
        "dtype": "bfloat16",
    }

  def test_benchmark_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("hbm_bandwidth")
    self.assertEqual(bm_class, hbm.HBMBandwidthBenchmark)

  @parameterized.parameters("copy", "scale", "add", "triad")
  def test_get_run_identifier(self, op_type):
    """Verify run identifier generation for all STREAM ops with and without size."""
    params_with_size = {"op_type": op_type, "size": 2048}
    self.assertEqual(
        self.bm.get_run_identifier(**params_with_size), f"{op_type}_dim_2048"
    )

    params_no_size = {"op_type": op_type}
    self.assertEqual(self.bm.get_run_identifier(**params_no_size), f"{op_type}")

  @parameterized.parameters("copy", "scale", "add", "triad")
  def test_stream_ops_execution(self, op_type):
    """Verify that all STREAM operations generate correct inputs and execute."""
    params = dict(self.params, op_type=op_type)
    self.bm.setup(**params)
    inputs = self.bm.generate_inputs(**params)

    if op_type in ("add", "triad"):
      self.assertEqual(len(inputs), 2)
      self.assertEqual(inputs[0].shape, (1024,))
      self.assertEqual(inputs[1].shape, (1024,))
    else:
      self.assertEqual(len(inputs), 1)
      self.assertEqual(inputs[0].shape, (1024,))

    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (1024,))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_random_scalar(self):
    """Verify internal random scalar works as expected for scale and triad ops."""
    x = jnp.ones((1024,), dtype=jnp.bfloat16)

    # Test scale: y = scalar * x
    self.bm.setup(op_type="scale")
    scalar = self.bm.scalar
    self.assertIsNotNone(scalar)
    out_scale = self.bm.run_op(x)
    np.testing.assert_allclose(out_scale, np.ones(1024) * scalar, rtol=1e-2)

    # Test triad: z = x + scalar * y
    x_zeros = jnp.zeros((1024,), dtype=jnp.bfloat16)
    y_ones = jnp.ones((1024,), dtype=jnp.bfloat16)
    self.bm.setup(op_type="triad")
    scalar = self.bm.scalar
    self.assertIsNotNone(scalar)
    out_triad = self.bm.run_op(x_zeros, y_ones)
    np.testing.assert_allclose(out_triad, np.ones(1024) * scalar, rtol=1e-2)

  def test_unsupported_op_type(self):
    """Verify an unsupported op_type raises a ValueError when setup is called."""
    with self.assertRaisesRegex(
        ValueError, "Unsupported op_type: 'invalid_kernel'"
    ):
      self.bm.setup(op_type="invalid_kernel")

  def test_unconfigured_benchmark_access(self):
    """Verify accessing properties on an unconfigured benchmark raises ValueError."""
    unconfigured_bm = hbm.HBMBandwidthBenchmark()
    with self.assertRaisesRegex(
        ValueError, "HBMBandwidthBenchmark is not configured"
    ):
      unconfigured_bm.get_total_bytes()

  def test_run_op_uninitialized(self):
    """Verify calling run_op before setup raises a ValueError."""
    uninitialized_bm = hbm.HBMBandwidthBenchmark()
    with self.assertRaisesRegex(ValueError, "JIT function not initialized."):
      uninitialized_bm.run_op(jnp.ones((1024,)))

  def test_get_total_bytes(self):
    """Verify the byte calculation for 1-input vs 2-input ops."""
    # Copy/Scale: size 1024 * 2 bytes/element * 2 arrays = 4096 bytes
    copy_params = dict(self.params, op_type="copy")
    scale_params = dict(self.params, op_type="scale")
    self.assertAlmostEqual(self.bm.get_total_bytes(**copy_params), 4096.0)
    self.assertAlmostEqual(self.bm.get_total_bytes(**scale_params), 4096.0)

    # Add/Triad: size 1024 * 2 bytes/element * 3 arrays = 6144 bytes
    add_params = dict(self.params, op_type="add")
    triad_params = dict(self.params, op_type="triad")
    self.assertAlmostEqual(self.bm.get_total_bytes(**add_params), 6144.0)
    self.assertAlmostEqual(self.bm.get_total_bytes(**triad_params), 6144.0)

  def test_get_arithmetic_intensity(self):
    """Verify arithmetic intensity calculations across all STREAM ops."""
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(**dict(self.params, op_type="copy")),
        # 1 FLOP / 2 bytes
        0.25,
    )
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(**dict(self.params, op_type="scale")),
        # 1 FLOP / 2 bytes
        0.25,
    )
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(**dict(self.params, op_type="add")),
        # 1 FLOP / (2 bytes * 3 arrays)
        1.0 / 6.0,
    )
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(**dict(self.params, op_type="triad")),
        # 2 FLOP / (2 bytes * 3 arrays)
        1.0 / 3.0,
    )

  @parameterized.parameters("copy", "scale", "add", "triad")
  def test_calculate_metrics(self, op_type):
    """Verify that metrics are correctly calculated across all STREAM ops."""
    # total_bytes = 4096 for copy/scale, 6144 for add/triad
    # avg_ms = 10.0 -> avg_latency_s = 0.01s
    times_ms = [10.0, 10.0, 10.0]
    params = dict(self.params, op_type=op_type)
    metrics = self.bm.calculate_metrics(times_ms, **params)

    expected_bytes = 4096.0 if op_type in ("copy", "scale") else 6144.0
    # total_bytes / avg_latency_s / 1e9
    expected_bw_gb_s = (expected_bytes / 0.01) / 1e9

    self.assertAlmostEqual(metrics["avg_ms"], 10.0)
    self.assertEqual(metrics["op_type"], op_type)
    self.assertAlmostEqual(metrics["bandwidth_gb_s"], expected_bw_gb_s)
    self.assertAlmostEqual(metrics["total_bytes_mb"], expected_bytes / 1e6)


if __name__ == "__main__":
  absltest.main()
