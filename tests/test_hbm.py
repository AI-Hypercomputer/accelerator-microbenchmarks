"""Unit tests for hbm.py."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import hbm
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
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
    self.params = {
        "size": 1024,
        "dtype": "bfloat16",
    }

  def _setup_benchmark(self, op_type: str = "copy", **kwargs):
    params = self.params.copy()
    params["op_type"] = op_type
    params.update(kwargs)
    config = hbm.HBMBandwidthParams(**params)
    self.bm = hbm.HBMBandwidthBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()

  def test_benchmark_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("hbm_bandwidth")
    self.assertEqual(bm_class, hbm.HBMBandwidthBenchmark)

  @parameterized.named_parameters(
      ("copy", "copy", 2048, 0, "copy_dim_2048_dev_0"),
      ("scale", "scale", 2048, 0, "scale_dim_2048_dev_0"),
      ("add", "add", 2048, 0, "add_dim_2048_dev_0"),
      ("triad", "triad", 2048, 0, "triad_dim_2048_dev_0"),
      ("custom_size", "copy", 4096, 0, "copy_dim_4096_dev_0"),
      ("large_size", "add", 134217728, 0, "add_dim_134217728_dev_0"),
  )
  def test_get_run_identifier(
      self, op_type, size, device_id, expected_identifier
  ):
    """Verify run identifier generation for all STREAM ops."""
    config = hbm.HBMBandwidthParams(
        op_type=op_type, size=size, device_id=device_id
    )
    self.bm = hbm.HBMBandwidthBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    self.assertEqual(self.bm.get_run_identifier(), expected_identifier)

  def test_device_id_validation(self):
    """Verify setup validates device_id is within range of local devices."""
    num_devices = len(jax.devices())

    # Invalid negative device_id
    config = hbm.HBMBandwidthParams(device_id=-1)
    bm = hbm.HBMBandwidthBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    with self.assertRaisesRegex(ValueError, "Invalid device_id: -1"):
      bm.setup()

    # Invalid out of range device_id
    config = hbm.HBMBandwidthParams(device_id=num_devices)
    bm = hbm.HBMBandwidthBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    with self.assertRaisesRegex(
        ValueError, f"Invalid device_id: {num_devices}"
    ):
      bm.setup()

  def test_get_device_to_measure(self):
    """Verify get_device_to_measure returns targeted local device."""
    mock_devices = [mock.MagicMock() for _ in range(8)]
    with mock.patch.object(jax, "devices", return_value=mock_devices):
      config = hbm.HBMBandwidthParams(device_id=7)
      bm = hbm.HBMBandwidthBenchmark(
          config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
      )
      bm.setup()
      self.assertEqual(bm.get_device_to_measure(), mock_devices[7])

  @parameterized.parameters("copy", "scale", "add", "triad")
  def test_stream_ops_execution(self, op_type):
    """Verify that all STREAM operations generate correct inputs and execute."""
    self._setup_benchmark(op_type)
    inputs = self.bm.generate_inputs()

    if op_type in ("add", "triad"):
      self.assertEqual(len(inputs), 2)
      self.assertEqual(inputs[0].shape, (1024,))
      self.assertEqual(inputs[1].shape, (1024,))
    else:
      self.assertEqual(len(inputs), 1)
      self.assertEqual(inputs[0].shape, (1024,))

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
    np.testing.assert_allclose(out_scale, np.ones(1024) * scalar, rtol=1e-2)

    # Test triad: z = x + scalar * y
    x_zeros = jnp.zeros((1024,), dtype=jnp.bfloat16)
    y_ones = jnp.ones((1024,), dtype=jnp.bfloat16)
    self._setup_benchmark("triad")
    scalar = self.bm.scalar
    self.assertIsNotNone(scalar)
    out_triad = self.bm.run_op(x_zeros, y_ones)
    np.testing.assert_allclose(out_triad, np.ones(1024) * scalar, rtol=1e-2)

  def test_unsupported_op_type(self):
    """Verify an unsupported op_type raises a ValueError when setup is called."""
    with self.assertRaisesRegex(
        ValueError, "Unsupported op_type: 'invalid_kernel'"
    ):
      self._setup_benchmark("invalid_kernel")

  def test_run_op_uninitialized(self):
    """Verify calling run_op before setup raises a ValueError."""
    config = hbm.HBMBandwidthParams()
    uninitialized_bm = hbm.HBMBandwidthBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC
    )
    with self.assertRaisesRegex(ValueError, "JIT function not initialized."):
      uninitialized_bm.run_op(jnp.ones((1024,)))

  def test_get_total_bytes(self):
    """Verify the byte calculation for 1-input vs 2-input ops."""
    # Copy/Scale: size 1024 * 2 bytes/element * 2 arrays = 4096 bytes
    self._setup_benchmark("copy")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 4096.0)

    self._setup_benchmark("scale")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 4096.0)

    # Add/Triad: size 1024 * 2 bytes/element * 3 arrays = 6144 bytes
    self._setup_benchmark("add")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 6144.0)

    self._setup_benchmark("triad")
    self.assertAlmostEqual(self.bm.get_total_bytes(), 6144.0)

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

  @parameterized.parameters("copy", "scale", "add", "triad")
  def test_calculate_metrics(self, op_type):
    """Verify that metrics are correctly calculated across all STREAM ops."""
    # total_bytes = 4096 for copy/scale, 6144 for add/triad
    # avg_ms = 10.0 -> avg_latency_s = 0.01s
    times_ms = [10.0, 10.0, 10.0]
    self._setup_benchmark(op_type)
    metrics = self.bm.calculate_metrics(times_ms)

    expected_bytes = 4096.0 if op_type in ("copy", "scale") else 6144.0
    # total_bytes / avg_latency_s / 1e9
    expected_bw_gb_s = (expected_bytes / 0.01) / 1e9

    self.assertAlmostEqual(metrics["avg_ms"], 10.0)
    self.assertEqual(metrics["op_type"], op_type)
    self.assertAlmostEqual(metrics["bandwidth_gb_s"], expected_bw_gb_s)
    self.assertAlmostEqual(
        metrics["total_bytes_mib"], expected_bytes / (1024 * 1024)
    )

  def test_format_benchmark_table(self):
    """Tests formatting of HBM bandwidth benchmark tables."""
    res = base.BenchmarkResult(
        metadata=base.BenchmarkMetadata(
            benchmark_name="HBMBandwidthBenchmark",
            test_name="HBMBandwidthBenchmark_test",
            start_time="2026-08-18T10:00:00",
            end_time="2026-08-18T10:01:00",
            params={
                "size": 134217728,
                "dtype": "bfloat16",
                "op_type": "copy",
                "device_id": 63,
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "total_bytes_mib": 256.00,
            "p50_ms": 0.07112,
            "bandwidth_gb_s": 7538.214,
            "xprof_p50_ms": 0.06543,
        },
        raw_times_ms=[1.0],
    )
    expected_cols = [
        "dtype",
        "op_type",
        "device_id",
        "size",
        "total_bytes_mib",
        "bandwidth_gb_s",
        "p50_ms",
        "xprof_p50_ms",
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
    self.assertIn("63", table)
    self.assertIn("134217728", table)
    self.assertIn("256.00", table)
    self.assertIn("7538.21", table)
    self.assertIn("0.0711", table)
    self.assertIn("0.0654", table)

  def test_schema_coverage(self):
    """Verify REPORT_SCHEMA matches output keys and covers all metrics."""
    self._setup_benchmark("copy")
    test_report_utils.assert_schema_matches_output(self, self.bm)


if __name__ == "__main__":
  absltest.main()
