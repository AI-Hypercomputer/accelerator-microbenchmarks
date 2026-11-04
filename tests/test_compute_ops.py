"""Unit tests for compute_ops.py."""

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import compute_ops
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.kernels import quantize_transpose
import jax
import jax.numpy as jnp
import numpy as np


def setUpModule():  # pylint: disable=invalid-name
  # Set CPU backend for fast testing without TPU requirements
  jax.config.update("jax_platform_name", "cpu")


class _ActivationTestBase(parameterized.TestCase):
  """Shared setup for the activation/normalization benchmark tests."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def _make_benchmark(self, benchmark_cls, **params):
    config = compute_ops.ComputeParams(**{"dim": 128, "batch": 32, **params})
    bm = benchmark_cls(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    bm.setup()
    return bm


class ComputeParamsTest(absltest.TestCase):
  """Unit tests for ComputeParams."""

  def test_default_mode_is_fwd(self):
    self.assertEqual(compute_ops.ComputeParams().mode, compute_ops.PassMode.FWD)

  def test_mode_is_coerced_from_string(self):
    config = compute_ops.ComputeParams(mode="bwd")
    self.assertEqual(config.mode, compute_ops.PassMode.BWD)

  def test_invalid_mode_raises_error(self):
    with self.assertRaisesRegex(ValueError, "Invalid mode"):
      compute_ops.ComputeParams(mode="sideways")

  def test_non_float_dtype_raises_error(self):
    with self.assertRaises(ValueError):
      compute_ops.ComputeParams(dtype="int8")


class SwiGLUBenchmarkTest(_ActivationTestBase):
  """Unit tests for SwiGLU benchmark."""

  def test_benchmark_registered(self):
    bm_class = registry.benchmark_registry.get_benchmark("swiglu")
    self.assertEqual(bm_class, compute_ops.SwiGLUBenchmark)

  def test_generate_inputs_fwd(self):
    bm = self._make_benchmark(compute_ops.SwiGLUBenchmark)
    (x,) = bm.generate_inputs()
    self.assertEqual(x.shape, (32, 128 * 2))
    self.assertEqual(x.dtype, jnp.bfloat16)

  def test_generate_inputs_bwd(self):
    bm = self._make_benchmark(compute_ops.SwiGLUBenchmark, mode="bwd")
    x, dy = bm.generate_inputs()
    self.assertEqual(x.shape, (32, 128 * 2))
    self.assertEqual(dy.shape, (32, 128))
    self.assertEqual(dy.dtype, jnp.bfloat16)

  def test_generate_inputs_uses_dtype(self):
    bm = self._make_benchmark(compute_ops.SwiGLUBenchmark, dtype="float32")
    (x,) = bm.generate_inputs()
    self.assertEqual(x.dtype, jnp.float32)

  def test_run_op_fwd(self):
    bm = self._make_benchmark(compute_ops.SwiGLUBenchmark, dtype="float32")
    (x,) = bm.generate_inputs()
    out = bm.run_op(x)
    self.assertEqual(out.shape, (32, 128))
    a, b = np.split(np.asarray(x, np.float64), 2, axis=-1)
    expected = a / (1.0 + np.exp(-a)) * b
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

  def test_run_op_bwd_matches_grad(self):
    bm = self._make_benchmark(
        compute_ops.SwiGLUBenchmark, dtype="float32", mode="bwd"
    )
    x, dy = bm.generate_inputs()
    dx = bm.run_op(x, dy)
    self.assertEqual(dx.shape, x.shape)

    def swiglu(x_):
      a, b = jnp.split(x_, 2, axis=-1)
      return jnp.sum(a * jax.nn.sigmoid(a) * b * dy)

    np.testing.assert_allclose(dx, jax.grad(swiglu)(x), rtol=1e-5, atol=1e-5)

  @parameterized.parameters(
      # Read X (32 * 256 * 2) + write Y (32 * 128 * 2) = 24576.
      ("fwd", 24576.0),
      # Read X (16384) + read dY (8192) + write dX (16384) = 40960.
      ("bwd", 40960.0),
  )
  def test_get_total_bytes(self, mode, expected_bytes):
    bm = self._make_benchmark(compute_ops.SwiGLUBenchmark, mode=mode)
    self.assertAlmostEqual(bm.get_total_bytes(), expected_bytes)

  def test_get_total_bytes_matches_v7x_shape(self):
    # v7x (M=4096, N=1024) moved 12 MiB.
    bm = self._make_benchmark(compute_ops.SwiGLUBenchmark, batch=4096, dim=512)
    self.assertAlmostEqual(bm.get_total_bytes() / (1024 * 1024), 12.0)

  @parameterized.parameters(("fwd", 1), ("bwd", 2))
  def test_get_arithmetic_intensity(self, mode, flops_multiplier):
    bm = self._make_benchmark(compute_ops.SwiGLUBenchmark, mode=mode)
    self.assertAlmostEqual(
        bm.get_arithmetic_intensity(),
        32 * 128 * 10 * flops_multiplier / bm.get_total_bytes(),
    )

  def test_calculate_metrics(self):
    bm = self._make_benchmark(compute_ops.SwiGLUBenchmark)
    metrics = bm.calculate_metrics([10.0])
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], 24576.0 / 0.01 / 1e9
    )
    self.assertAlmostEqual(metrics["total_bytes_mib"], 24576.0 / (1024 * 1024))
    self.assertIn("intensity", metrics)

  def test_run_identifier_includes_mode(self):
    bm = self._make_benchmark(compute_ops.SwiGLUBenchmark, mode="bwd")
    self.assertEqual(bm.get_run_identifier(), "batch_32_dim_128_bwd")


class SigmoidBenchmarkTest(_ActivationTestBase):
  """Unit tests for Sigmoid benchmark."""

  def test_benchmark_registered(self):
    bm_class = registry.benchmark_registry.get_benchmark("sigmoid")
    self.assertEqual(bm_class, compute_ops.SigmoidBenchmark)

  def test_generate_inputs_bwd(self):
    bm = self._make_benchmark(compute_ops.SigmoidBenchmark, mode="bwd")
    x, dy = bm.generate_inputs()
    self.assertEqual(x.shape, (32, 128))
    self.assertEqual(dy.shape, (32, 128))
    self.assertEqual(x.dtype, jnp.bfloat16)

  def test_run_op_fwd(self):
    bm = self._make_benchmark(compute_ops.SigmoidBenchmark, dtype="float32")
    (x,) = bm.generate_inputs()
    out = bm.run_op(x)
    expected = 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

  def test_run_op_bwd(self):
    bm = self._make_benchmark(
        compute_ops.SigmoidBenchmark, dtype="float32", mode="bwd"
    )
    x, dy = bm.generate_inputs()
    dx = bm.run_op(x, dy)
    y = 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))
    np.testing.assert_allclose(
        dx, np.asarray(dy) * y * (1.0 - y), rtol=1e-5, atol=1e-5
    )

  @parameterized.parameters(
      # Read X + write Y: 2 * 32 * 128 * 2 = 16384.
      ("fwd", 16384.0),
      # Read X + read dY + write dX: 3 * 32 * 128 * 2 = 24576.
      ("bwd", 24576.0),
  )
  def test_get_total_bytes(self, mode, expected_bytes):
    bm = self._make_benchmark(compute_ops.SigmoidBenchmark, mode=mode)
    self.assertAlmostEqual(bm.get_total_bytes(), expected_bytes)

  def test_calculate_metrics(self):
    bm = self._make_benchmark(compute_ops.SigmoidBenchmark)
    metrics = bm.calculate_metrics([10.0])
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], 16384.0 / 0.01 / 1e9
    )
    self.assertAlmostEqual(metrics["intensity"], 32 * 128 * 4 / 16384.0)


class RMSNormBenchmarkTest(_ActivationTestBase):
  """Unit tests for RMSNorm benchmark."""

  def test_benchmark_registered(self):
    bm_class = registry.benchmark_registry.get_benchmark("rmsnorm")
    self.assertEqual(bm_class, compute_ops.RMSNormBenchmark)

  def test_generate_inputs_fwd(self):
    bm = self._make_benchmark(compute_ops.RMSNormBenchmark)
    x, w = bm.generate_inputs()
    self.assertEqual(x.shape, (32, 128))
    self.assertEqual(w.shape, (128,))
    self.assertEqual(x.dtype, jnp.bfloat16)
    self.assertEqual(w.dtype, jnp.bfloat16)

  def test_generate_inputs_bwd(self):
    bm = self._make_benchmark(compute_ops.RMSNormBenchmark, mode="bwd")
    x, w, dy = bm.generate_inputs()
    self.assertEqual(dy.shape, x.shape)
    self.assertEqual(w.shape, (128,))

  def test_run_op_fwd(self):
    bm = self._make_benchmark(compute_ops.RMSNormBenchmark)
    x, w = bm.generate_inputs()
    out = bm.run_op(x, w)
    self.assertEqual(out.shape, (32, 128))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_run_op_bwd_returns_input_and_weight_grads(self):
    bm = self._make_benchmark(
        compute_ops.RMSNormBenchmark, dtype="float32", mode="bwd"
    )
    x, w, dy = bm.generate_inputs()
    dx, dw = bm.run_op(x, w, dy)
    self.assertEqual(dx.shape, x.shape)
    self.assertEqual(dw.shape, w.shape)

    def rmsnorm(x_, w_):
      rms = jnp.sqrt(jnp.mean(jnp.square(x_), axis=-1, keepdims=True) + 1e-6)
      return jnp.sum(x_ / rms * w_ * dy)

    expected_dx, expected_dw = jax.grad(rmsnorm, argnums=(0, 1))(x, w)
    np.testing.assert_allclose(dx, expected_dx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dw, expected_dw, rtol=1e-4, atol=1e-4)

  @parameterized.parameters(
      # Read X (8192) + write Y (8192) + read W (256) = 16640.
      ("fwd", 16640.0),
      # Read X, dY + write dX (3 * 8192) + read W, write dW (2 * 256) = 25088.
      ("bwd", 25088.0),
  )
  def test_get_total_bytes(self, mode, expected_bytes):
    bm = self._make_benchmark(compute_ops.RMSNormBenchmark, mode=mode)
    self.assertAlmostEqual(bm.get_total_bytes(), expected_bytes)

  def test_calculate_metrics(self):
    bm = self._make_benchmark(compute_ops.RMSNormBenchmark)
    metrics = bm.calculate_metrics([10.0])
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], 16640.0 / 0.01 / 1e9
    )


class RoPEBenchmarkTest(parameterized.TestCase):
  """Unit tests for RoPE benchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def _make_benchmark(self, **params):
    config = compute_ops.RoPEParams(
        **{"seq_len": 16, "head_dim": 8, "batch": 2, "heads": 3, **params}
    )
    bm = compute_ops.RoPEBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    bm.setup()
    return bm

  @staticmethod
  def _reference_rope(x, table, batch, seq_len, heads, head_dim):
    """Per-head rotate_half RoPE in float64 numpy."""
    x = np.asarray(x, np.float64).reshape(batch, seq_len, heads, head_dim)
    table = np.asarray(table, np.float64)
    half = head_dim // 2
    cos = table[None, :, None, :half]
    sin = table[None, :, None, half:]
    x1, x2 = x[..., :half], x[..., half:]
    out = np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return out.reshape(batch * seq_len, heads * head_dim)

  def test_benchmark_registered(self):
    bm_class = registry.benchmark_registry.get_benchmark("rope")
    self.assertEqual(bm_class, compute_ops.RoPEBenchmark)

  def test_default_mode_is_fwd(self):
    self.assertEqual(compute_ops.RoPEParams().mode, compute_ops.PassMode.FWD)

  def test_mode_is_coerced_from_string(self):
    config = compute_ops.RoPEParams(mode="bwd")
    self.assertEqual(config.mode, compute_ops.PassMode.BWD)

  def test_invalid_mode_raises_error(self):
    with self.assertRaisesRegex(ValueError, "Invalid mode"):
      compute_ops.RoPEParams(mode="sideways")

  def test_odd_head_dim_raises_error(self):
    with self.assertRaisesRegex(ValueError, "head_dim must be even"):
      compute_ops.RoPEParams(head_dim=7)

  def test_non_float_dtype_raises_error(self):
    with self.assertRaises(ValueError):
      compute_ops.RoPEParams(dtype="int8")

  def test_generate_inputs_fwd(self):
    bm = self._make_benchmark()
    x, table = bm.generate_inputs()
    self.assertEqual(x.shape, (2 * 16, 3 * 8))
    self.assertEqual(x.dtype, jnp.bfloat16)
    self.assertEqual(table.shape, (16, 8))
    self.assertEqual(table.dtype, jnp.float32)
    # Position 0 has angle 0: cos = 1, sin = 0.
    np.testing.assert_allclose(table[0], [1.0] * 4 + [0.0] * 4)

  def test_generate_inputs_bwd(self):
    bm = self._make_benchmark(mode="bwd")
    x, table, dy = bm.generate_inputs()
    self.assertEqual(dy.shape, x.shape)
    self.assertEqual(dy.dtype, jnp.bfloat16)
    self.assertEqual(table.dtype, jnp.float32)

  def test_run_op_fwd_matches_reference(self):
    bm = self._make_benchmark(dtype="float32")
    x, table = bm.generate_inputs()
    out = bm.run_op(x, table)
    self.assertEqual(out.shape, x.shape)
    self.assertEqual(out.dtype, jnp.float32)
    expected = self._reference_rope(x, table, 2, 16, 3, 8)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

  def test_run_op_fwd_keeps_bfloat16_output(self):
    bm = self._make_benchmark()
    x, table = bm.generate_inputs()
    out = bm.run_op(x, table)
    self.assertEqual(out.dtype, jnp.bfloat16)
    expected = self._reference_rope(x, table, 2, 16, 3, 8)
    np.testing.assert_allclose(
        np.asarray(out, np.float32), expected, rtol=2e-2, atol=2e-2
    )

  def test_run_op_bwd_is_inverse_rotation(self):
    bm = self._make_benchmark(dtype="float32", mode="bwd")
    x, table, dy = bm.generate_inputs()
    dx = bm.run_op(x, table, dy)
    self.assertEqual(dx.shape, x.shape)
    self.assertEqual(dx.dtype, jnp.float32)
    # The VJP of a rotation is the rotation by the negated angle.
    half = 4
    inverse_table = np.concatenate(
        [np.asarray(table[:, :half]), -np.asarray(table[:, half:])], axis=-1
    )
    expected = self._reference_rope(dy, inverse_table, 2, 16, 3, 8)
    np.testing.assert_allclose(dx, expected, rtol=1e-5, atol=1e-5)

  @parameterized.parameters("fwd", "bwd")
  def test_get_total_bytes(self, mode):
    bm = self._make_benchmark(mode=mode)
    # Activations in + out: 2 * (2 * 16 * 3 * 8) * 2 bytes = 3072.
    # f32 table: 16 * 8 * 4 bytes = 512.
    self.assertAlmostEqual(bm.get_total_bytes(), 3584.0)

  def test_get_total_bytes_matches_v7x_shape(self):
    # v7x (M=1024, N=1792) moved 7.5 MiB.
    bm = self._make_benchmark(batch=1, seq_len=1024, heads=14, head_dim=128)
    self.assertAlmostEqual(bm.get_total_bytes() / (1024 * 1024), 7.5)

  def test_get_arithmetic_intensity(self):
    bm = self._make_benchmark()
    self.assertAlmostEqual(
        bm.get_arithmetic_intensity(), (2 * 16 * 3 * 8 * 3) / 3584.0
    )

  def test_calculate_metrics(self):
    bm = self._make_benchmark()
    metrics = bm.calculate_metrics([10.0])
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], 3584.0 / 0.01 / 1e9
    )
    self.assertAlmostEqual(metrics["total_bytes_mib"], 3584.0 / (1024 * 1024))

  def test_run_identifier_includes_mode(self):
    bm = self._make_benchmark(mode="bwd")
    self.assertEqual(bm.get_run_identifier(), "b_2_h_3_s_16_d_8_bwd")


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
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.bm.setup()
    (x,) = self.bm.generate_inputs()
    self.assertEqual(x.shape, (64, 128))
    self.assertEqual(x.dtype, jnp.bfloat16)

  def test_run_op(self):
    params = {"m": 64, "n": 128}
    config = compute_ops.QuantParams(**params)
    self.bm = compute_ops.QuantizationBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
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

  @parameterized.named_parameters(
      ("int_input", "dtype", "int8"),
      ("fp8_input", "dtype", "float8_e4m3fn"),
      ("fp4_input", "dtype", "float4_e2m1fn"),
      ("bool_quant", "quant_dtype", "bool"),
      ("complex_quant", "quant_dtype", "complex64"),
      ("int_scale", "scale_dtype", "int8"),
      ("complex_scale", "scale_dtype", "complex64"),
      ("fp8_scale", "scale_dtype", "float8_e4m3fn"),
      ("fp4_scale", "scale_dtype", "float4_e2m1fn"),
  )
  def test_params_reject_unsupported_dtypes(self, field, dtype):
    """Verifies the input, scale and quant dtype constraints are enforced."""
    with self.assertRaisesRegex(
        ValueError,
        rf"^{field}: Unsupported dtype '{dtype}'\. Supported dtypes for"
        rf" {field}:",
    ):
      compute_ops.QuantParams(**{field: dtype})

  @parameterized.product(
      field=("dtype", "scale_dtype"),
      dtype=("bfloat16", "float16", "float32"),
  )
  def test_params_accept_16_and_32_bit_floats(self, field, dtype):
    """Verifies >= 16-bit float inputs and scales are accepted."""
    config = compute_ops.QuantParams(**{field: dtype})
    self.assertEqual(getattr(config, field), dtype)

  @parameterized.parameters("dtype", "scale_dtype")
  def test_params_float64_requires_x64(self, field):
    """Verifies a float64 input or scale is accepted only with jax_enable_x64."""
    with jax.enable_x64(True):
      config = compute_ops.QuantParams(**{field: "float64"})
      self.assertEqual(getattr(config, field), "float64")
    with jax.enable_x64(False), self.assertRaisesRegex(
        ValueError, f"^{field}: JAX canonicalizes dtype 'float64'"
    ):
      compute_ops.QuantParams(**{field: "float64"})

  @parameterized.product(
      dtype=("bfloat16", "float16", "float32", "float64"),
      scale_dtype=("bfloat16", "float16", "float32", "float64"),
  )
  def test_run_op_accepted_input_and_scale_dtypes(self, dtype, scale_dtype):
    """Verifies every accepted input x scale dtype pair runs end to end."""
    with jax.enable_x64("float64" in (dtype, scale_dtype)):
      config = compute_ops.QuantParams(
          m=64, n=128, dtype=dtype, scale_dtype=scale_dtype
      )
      self.bm = compute_ops.QuantizationBenchmark(
          config=config,
          hardware_spec=system.TPU7X_HARDWARE_SPEC,
          mesh=self.mock_mesh,
      )
      self.bm.setup()
      (x,) = self.bm.generate_inputs()
      out, sf = self.bm.run_op(x)
      self.assertEqual(x.dtype, jnp.dtype(dtype))
      self.assertEqual(out.dtype, jnp.float8_e4m3fn)
      self.assertEqual(sf.dtype, jnp.dtype(scale_dtype))

  @parameterized.parameters("int8", "uint8", "float8_e5m2")
  def test_params_accept_float_or_int_quant_dtype(self, quant_dtype):
    """Verifies quant_dtype accepts integer and floating-point dtypes."""
    config = compute_ops.QuantParams(quant_dtype=quant_dtype)
    self.assertEqual(config.quant_dtype, quant_dtype)

  def test_get_total_bytes(self):
    """Verifies default total bytes and derived device bandwidth metrics."""
    params = {"m": 64, "n": 128}
    config = compute_ops.QuantParams(**params)
    self.bm = compute_ops.QuantizationBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
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
    """Verifies non-positive latency logs a warning and returns inf."""
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
    self.assertIn("Non-positive latency_s", "\n".join(cm.output))


class QuantizeTransposeBenchmarkTest(parameterized.TestCase):
  """Unit tests for the transposed quantization (Pallas kernel) benchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def test_benchmark_registered(self):
    bm_class = registry.benchmark_registry.get_benchmark("quantize_transpose")
    self.assertEqual(bm_class, compute_ops.QuantizeTransposeBenchmark)

  def test_setup_without_mesh_raises(self):
    bm = compute_ops.QuantizeTransposeBenchmark(
        config=compute_ops.QuantParams(m=64, n=128),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )
    with self.assertRaisesRegex(ValueError, "Mesh not initialized"):
      bm.setup()

  @parameterized.named_parameters(
      ("fp8_bf16_sf", "float8_e4m3fn", "bfloat16"),
      ("fp4_bf16_sf", "float4_e2m1fn", "bfloat16"),
      ("int4_bf16_sf", "int4", "bfloat16"),
      ("fp8_fp32_sf", "float8_e4m3fn", "float32"),
      ("int4_fp32_sf", "int4", "float32"),
  )
  def test_matches_rowwise_quantize_of_transposed_input(
      self, quant_dtype, scale_dtype
  ):
    """Verifies the kernel on X equals QuantizationBenchmark on X^T."""
    bm_t = compute_ops.QuantizeTransposeBenchmark(
        config=compute_ops.QuantParams(
            m=64, n=128, quant_dtype=quant_dtype, scale_dtype=scale_dtype
        ),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    bm_ref = compute_ops.QuantizationBenchmark(
        config=compute_ops.QuantParams(
            m=128, n=64, quant_dtype=quant_dtype, scale_dtype=scale_dtype
        ),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    bm_t.setup()
    bm_ref.setup()

    (x,) = bm_t.generate_inputs()
    with mock.patch.object(
        quantize_transpose,
        "quantize_transpose",
        functools.partial(
            quantize_transpose.quantize_transpose, interpret=True
        ),
    ):
      out_t, sf_t = bm_t.run_op(x)
    out_ref, sf_ref = bm_ref.run_op(x.T)

    self.assertEqual(out_t.shape, (128, 64))
    self.assertEqual(out_t.dtype, jnp.dtype(quant_dtype))
    self.assertEqual(sf_t.shape, (128, 1))
    self.assertEqual(sf_t.dtype, jnp.dtype(scale_dtype))
    np.testing.assert_array_equal(
        np.asarray(out_t, dtype=np.float32),
        np.asarray(out_ref, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(sf_t, dtype=np.float32),
        np.asarray(sf_ref, dtype=np.float32),
    )

  @parameterized.named_parameters(
      # Read X (64 * 128 * 2) + Write Out (64 * 128 * 1) + Write SF (128 * 2)
      # 16384 + 8192 + 256 = 24832
      ("fp8_bf16_sf", "float8_e4m3fn", "bfloat16", 24832.0),
      # 16384 + 8192 + (128 * 4) = 25088
      ("fp8_fp32_sf", "float8_e4m3fn", "float32", 25088.0),
      # 16384 + (64 * 128 * 0.5) + (128 * 2) = 20736
      ("int4_bf16_sf", "int4", "bfloat16", 20736.0),
  )
  def test_total_bytes_identifier_and_metadata(
      self, quant_dtype, scale_dtype, expected_bytes
  ):
    """Verifies N-length scales in bytes, the identifier, and metadata."""
    bm = compute_ops.QuantizeTransposeBenchmark(
        config=compute_ops.QuantParams(
            m=64, n=128, quant_dtype=quant_dtype, scale_dtype=scale_dtype
        ),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.assertAlmostEqual(bm.get_total_bytes(), expected_bytes)
    self.assertEqual(
        bm.get_run_identifier(), f"m_64_n_128_{quant_dtype}_sf_{scale_dtype}"
    )
    metadata = bm.get_workload_metadata()
    self.assertAlmostEqual(
        metadata["total_bytes_mib"], expected_bytes / (1024 * 1024)
    )
    self.assertEqual(metadata["block_m"], 64)
    self.assertEqual(metadata["block_n"], 128)


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
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
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
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
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
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
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
