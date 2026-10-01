"""Unit tests for compute_ops.py."""

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import compute_ops
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import system
import jax
import jax.numpy as jnp
import numpy as np

# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class SwiGLUBenchmarkTest(absltest.TestCase):
  """Unit tests for SwiGLU benchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def test_benchmark_registered(self):
    bm_class = registry.benchmark_registry.get_benchmark("swiglu")
    self.assertEqual(bm_class, compute_ops.SwiGLUBenchmark)

  def test_generate_inputs(self):
    params = {"dim": 128, "batch": 32}
    config = compute_ops.ComputeParams(**params)
    self.bm = compute_ops.SwiGLUBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    (x,) = self.bm.generate_inputs()
    self.assertEqual(x.shape, (32, 128 * 2))
    self.assertEqual(x.dtype, jnp.bfloat16)

  def test_run_op(self):
    params = {"dim": 128, "batch": 32}
    config = compute_ops.ComputeParams(**params)
    self.bm = compute_ops.SwiGLUBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    (x,) = self.bm.generate_inputs()
    out = self.bm.run_op(x)
    self.assertEqual(out.shape, (32, 128))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_get_total_bytes(self):
    params = {"dim": 128, "batch": 32}
    config = compute_ops.ComputeParams(**params)
    self.bm = compute_ops.SwiGLUBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    # Read X (32 * 128 * 2 * 2) + Write Out (32 * 128 * 2)
    # 16384 + 8192 = 24576
    expected_bytes = 24576.0
    self.assertAlmostEqual(self.bm.get_total_bytes(), expected_bytes)

  def test_get_arithmetic_intensity(self):
    params = {"dim": 128, "batch": 32}
    config = compute_ops.ComputeParams(**params)
    self.bm = compute_ops.SwiGLUBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    # flops = 32 * 128 * 10 = 40960
    # intensity = 40960 / 24576 = 1.6666666666666667
    expected_intensity = 40960 / 24576
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(), expected_intensity
    )
    metrics = self.bm.calculate_metrics([10.0])
    self.assertNotIn("intensity", metrics)
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)


class RMSNormBenchmarkTest(absltest.TestCase):
  """Unit tests for RMSNorm benchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def test_benchmark_registered(self):
    bm_class = registry.benchmark_registry.get_benchmark("rmsnorm")
    self.assertEqual(bm_class, compute_ops.RMSNormBenchmark)

  def test_generate_inputs(self):
    params = {"dim": 128, "batch": 32}
    config = compute_ops.ComputeParams(**params)
    self.bm = compute_ops.RMSNormBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    x, w = self.bm.generate_inputs()
    self.assertEqual(x.shape, (32, 128))
    self.assertEqual(w.shape, (128,))
    self.assertEqual(x.dtype, jnp.bfloat16)
    self.assertEqual(w.dtype, jnp.bfloat16)

  def test_run_op(self):
    params = {"dim": 128, "batch": 32}
    config = compute_ops.ComputeParams(**params)
    self.bm = compute_ops.RMSNormBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    x, w = self.bm.generate_inputs()
    out = self.bm.run_op(x, w)
    self.assertEqual(out.shape, (32, 128))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_get_total_bytes(self):
    params = {"dim": 128, "batch": 32}
    config = compute_ops.ComputeParams(**params)
    self.bm = compute_ops.RMSNormBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    # Read X (32 * 128 * 2), Read W (128 * 2), Write Out (32 * 128 * 2)
    # 8192 + 256 + 8192 = 16640
    expected_bytes = 16640.0
    self.assertAlmostEqual(self.bm.get_total_bytes(), expected_bytes)
    metrics = self.bm.calculate_metrics([10.0])
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)


class RoPEBenchmarkTest(absltest.TestCase):
  """Unit tests for RoPE benchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def test_benchmark_registered(self):
    bm_class = registry.benchmark_registry.get_benchmark("rope")
    self.assertEqual(bm_class, compute_ops.RoPEBenchmark)

  def test_generate_inputs(self):
    params = {"seq_len": 64, "head_dim": 64, "batch": 8, "heads": 16}
    config = compute_ops.RoPEParams(**params)
    self.bm = compute_ops.RoPEBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    x, freq_cis = self.bm.generate_inputs()
    self.assertEqual(x.shape, (8, 16, 64, 32))
    self.assertEqual(freq_cis.shape, (1, 1, 64, 32))
    self.assertEqual(x.dtype, jnp.complex64)
    self.assertEqual(freq_cis.dtype, jnp.complex64)

  def test_run_op(self):
    params = {"seq_len": 64, "head_dim": 64, "batch": 8, "heads": 16}
    config = compute_ops.RoPEParams(**params)
    self.bm = compute_ops.RoPEBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    x, freq_cis = self.bm.generate_inputs()
    out = self.bm.run_op(x, freq_cis)
    self.assertEqual(out.shape, (8, 16, 64, 32))
    self.assertEqual(out.dtype, jnp.complex64)

  def test_get_total_bytes(self):
    params = {"seq_len": 64, "head_dim": 64, "batch": 8, "heads": 16}
    config = compute_ops.RoPEParams(**params)
    self.bm = compute_ops.RoPEBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    # batch * heads * seq_len * head_dim//2 * 8 * 2
    # 8 * 16 * 64 * 32 * 8 * 2 = 4194304
    expected_bytes = 4194304.0
    self.assertAlmostEqual(self.bm.get_total_bytes(), expected_bytes)
    metrics = self.bm.calculate_metrics([10.0])
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)


class QuantizationBenchmarkTest(parameterized.TestCase):
  """Unit tests for Quantization benchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def test_benchmark_registered(self):
    bm_class = registry.benchmark_registry.get_benchmark("quantization")
    self.assertEqual(bm_class, compute_ops.QuantizationBenchmark)

  def test_generate_inputs(self):
    params = {"m": 64, "n": 128}
    config = compute_ops.QuantParams(**params)
    self.bm = compute_ops.QuantizationBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    (x,) = self.bm.generate_inputs()
    self.assertEqual(x.shape, (64, 128))
    self.assertEqual(x.dtype, jnp.bfloat16)

  def test_run_op(self):
    params = {"m": 64, "n": 128}
    config = compute_ops.QuantParams(**params)
    self.bm = compute_ops.QuantizationBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    (x,) = self.bm.generate_inputs()
    out, sf = self.bm.run_op(x)
    self.assertEqual(out.shape, (64, 128))
    self.assertEqual(out.dtype, jnp.float8_e4m3fn)
    self.assertEqual(sf.shape, (64, 1))
    self.assertEqual(sf.dtype, jnp.bfloat16)

  @parameterized.named_parameters(
      (
          "fp8_bf16_sf",
          "float8_e4m3fn",
          "bfloat16",
          jnp.float8_e4m3fn,
          jnp.bfloat16,
          24704.0,
      ),
      (
          "fp4_bf16_sf",
          "float4_e2m1fn",
          "bfloat16",
          jnp.float4_e2m1fn,
          jnp.bfloat16,
          20608.0,
      ),
      ("int4_bf16_sf", "int4", "bfloat16", jnp.int4, jnp.bfloat16, 20608.0),
      (
          "fp8_fp32_sf",
          "float8_e4m3fn",
          "float32",
          jnp.float8_e4m3fn,
          jnp.float32,
          24832.0,
      ),
  )
  def test_quant_dtypes_and_bytes(
      self,
      quant_dtype,
      scale_dtype,
      expected_out_dtype,
      expected_sf_dtype,
      expected_bytes,
  ):
    """Verifies output dtype, scale dtype, and total bytes across FP8, FP4, and INT4."""
    config = compute_ops.QuantParams(
        m=64, n=128, quant_dtype=quant_dtype, scale_dtype=scale_dtype
    )
    self.bm = compute_ops.QuantizationBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.bm.setup()
    (x,) = self.bm.generate_inputs()
    out, sf = self.bm.run_op(x)
    self.assertEqual(out.shape, (64, 128))
    self.assertEqual(out.dtype, expected_out_dtype)
    self.assertEqual(sf.shape, (64, 1))
    self.assertEqual(sf.dtype, expected_sf_dtype)
    self.assertAlmostEqual(self.bm.get_total_bytes(), expected_bytes)
    self.assertEqual(
        self.bm.get_run_identifier(),
        f"m_64_n_128_{quant_dtype}_sf_{scale_dtype}",
    )

  def test_get_total_bytes(self):
    """Verifies default total bytes and derived device bandwidth metrics."""
    params = {"m": 64, "n": 128}
    config = compute_ops.QuantParams(**params)
    self.bm = compute_ops.QuantizationBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    # Read X (64 * 128 * 2), Write Out (64 * 128 * 1), Write SF (64 * 2)
    # 16384 + 8192 + 128 = 24704
    expected_bytes = 24704.0
    self.assertAlmostEqual(self.bm.get_total_bytes(), expected_bytes)
    metrics = self.bm.calculate_metrics([10.0])
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", metrics)
    self.assertGreater(metrics["wall_clock_bandwidth_per_device_gb_s"], 0.0)

  def test_calculate_throughput_metrics_zero_latency_warns(self):
    """Verifies zero/negative latency logs a warning and returns inf bandwidth."""
    config = compute_ops.QuantParams(m=64, n=128)
    bm = compute_ops.QuantizationBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    with self.assertLogs(level="WARNING") as cm:
      metrics = bm.calculate_throughput_metrics(
          0.0, constants.TimingDomain.WALL_CLOCK
      )
    self.assertEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], float("inf")
    )
    self.assertTrue(
        any("Non-positive latency_s" in msg for msg in cm.output)
    )


class AddBenchmarkTest(absltest.TestCase):
  """Unit tests for Add benchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def test_benchmark_registered(self):
    bm_class = registry.benchmark_registry.get_benchmark("simple_add")
    self.assertEqual(bm_class, compute_ops.AddBenchmark)

  def test_generate_inputs(self):
    params = {"size": 1024}
    config = compute_ops.AddParams(**params)
    self.bm = compute_ops.AddBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    x, y = self.bm.generate_inputs()
    self.assertEqual(x.shape, (1024,))
    self.assertEqual(y.shape, (1024,))
    self.assertEqual(x.dtype, jnp.bfloat16)
    self.assertEqual(y.dtype, jnp.bfloat16)

  def test_run_op(self):
    params = {"size": 1024}
    config = compute_ops.AddParams(**params)
    self.bm = compute_ops.AddBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    x, y = self.bm.generate_inputs()
    out = self.bm.run_op(x, y)
    self.assertEqual(out.shape, (1024,))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_get_total_bytes(self):
    params = {"size": 1024}
    config = compute_ops.AddParams(**params)
    self.bm = compute_ops.AddBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    # Read X, Read Y, Write Z
    # 1024 * 2 * 3 = 6144
    expected_bytes = 6144.0
    self.assertAlmostEqual(self.bm.get_total_bytes(), expected_bytes)
    metrics = self.bm.calculate_metrics([10.0])
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)


class ComputeOpsParamsValidationTest(parameterized.TestCase):
  """Verifies the bounds declared on the compute_ops params classes."""

  @parameterized.parameters(
      (compute_ops.ComputeParams, "dim", 0),
      (compute_ops.ComputeParams, "batch", -1),
      (compute_ops.RoPEParams, "seq_len", 0),
      (compute_ops.RoPEParams, "head_dim", -1),
      (compute_ops.RoPEParams, "batch", 0),
      (compute_ops.RoPEParams, "heads", 0),
      (compute_ops.QuantParams, "m", 0),
      (compute_ops.QuantParams, "n", -1),
      (compute_ops.AddParams, "size", 0),
  )
  def test_out_of_range_values_raise_error(self, params_cls, field, value):
    """Verifies that out-of-range numeric values raise a ValueError."""
    with self.assertRaisesRegex(ValueError, f"{field} must be >= 1"):
      params_cls(**{field: value})


if __name__ == "__main__":
  absltest.main()
