"""Unit tests for matlmul.py."""

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import matmul
from accelerator_microbenchmarks.core import registry
import jax
import jax.numpy as jnp
import numpy as np

# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class GeneralizedGemmBenchmarkTest(parameterized.TestCase):
  """Unit tests for Generalized GEMM benchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def _setup_benchmark(self, **kwargs):
    params = {
        "m": 64,
        "k": 64,
        "n": 64,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
    }
    params.update(kwargs)
    config = matmul.GemmParams(**params)
    self.bm = matmul.GeneralizedGemmBenchmark(
        config=config, mesh=self.mock_mesh
    )
    self.bm.setup()

  def test_benchmark_registered(self):
    """Test that the benchmark is properly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("gemm_generalized")
    self.assertEqual(bm_class, matmul.GeneralizedGemmBenchmark)

  def test_generate_inputs(self):
    """Test generating inputs without scaling factors."""
    self._setup_benchmark()
    inputs = self.bm.generate_inputs()
    self.assertLen(inputs, 4)
    a, b = inputs[0], inputs[1]
    self.assertEqual(a.shape, (64, 64))
    self.assertEqual(b.shape, (64, 64))
    self.assertEqual(a.dtype, jnp.bfloat16)
    self.assertEqual(b.dtype, jnp.bfloat16)

  def test_generate_inputs_with_scaling_factors(self):
    """Test generating inputs with scaling factors."""
    self._setup_benchmark(use_scaling_factors=True)
    inputs = self.bm.generate_inputs()
    self.assertLen(inputs, 4)
    a, b, sf0, sf1 = inputs
    self.assertEqual(a.shape, (64, 64))
    self.assertEqual(b.shape, (64, 64))
    self.assertEqual(sf0.shape, (64, 1))
    self.assertEqual(sf1.shape, (1, 64))
    self.assertEqual(a.dtype, jnp.bfloat16)
    self.assertEqual(b.dtype, jnp.bfloat16)
    self.assertEqual(sf0.dtype, jnp.float32)
    self.assertEqual(sf1.dtype, jnp.float32)

  def test_run_op(self):
    """Test run op without scaling factors."""
    self._setup_benchmark()
    inputs = self.bm.generate_inputs()
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (64, 64))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_run_op_with_scaling_factors(self):
    """Test run op with scaling factors."""
    self._setup_benchmark(use_scaling_factors=True)
    inputs = self.bm.generate_inputs()
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (64, 64))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_get_total_bytes(self):
    """Test calculating total bytes."""
    self._setup_benchmark()
    # Read A (64 * 64 * 2), Read B (64 * 64 * 2), Write Out (64 * 64 * 2)
    # 8192 * 3 = 24576
    expected_bytes = 24576.0
    self.assertAlmostEqual(self.bm.get_total_bytes(), expected_bytes)

  def test_get_arithmetic_intensity(self):
    """Test calculating arithmetic intensity."""
    self._setup_benchmark()
    # flops = 2 * 64 * 64 * 64 = 524288
    # bytes = 24576
    # intensity = 524288 / 24576 = 21.333333333333332
    expected_intensity = 524288 / 24576
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(), expected_intensity
    )

  def test_calculate_metrics(self):
    """Test calculating performance metrics."""
    self._setup_benchmark()
    metrics = self.bm.calculate_metrics([1.0, 1.5, 2.0])
    self.assertIn("avg_ms", metrics)
    self.assertIn("tflops_per_sec", metrics)
    self.assertIn("total_flops", metrics)
    self.assertIn("intensity", metrics)
    self.assertEqual(metrics["total_flops"], 524288)
    self.assertAlmostEqual(metrics["intensity"], 524288 / 24576)

  def test_is_xprof_op(self):
    """Test that is_xprof_op correctly identifies convolution fusion events."""
    config = matmul.GemmParams(m=128, n=128, k=128)
    self.bm = matmul.GeneralizedGemmBenchmark(
        config=config, mesh=self.mock_mesh
    )
    self.assertTrue(
        self.bm.match_xprof_op_fallback(
            {"args": {"hlo_category": "convolution fusion"}}
        )
    )
    self.assertFalse(
        self.bm.match_xprof_op_fallback(
            {"args": {"hlo_category": "other fusion"}}
        )
    )
    self.assertFalse(self.bm.match_xprof_op_fallback({}))

  @parameterized.named_parameters(
      ("normal", False, False, (32, 64), (64, 128)),
      ("transposed_a", True, False, (64, 32), (64, 128)),
      ("transposed_b", False, True, (32, 64), (128, 64)),
      ("both_transposed", True, True, (64, 32), (128, 64)),
  )
  def test_generate_inputs_transposed(
      self, ta, tb, expected_a_shape, expected_b_shape
  ):
    """Test shape generation for transposed inputs with non-square matrices."""
    m, k, n = 32, 64, 128

    params = {
        "m": m,
        "k": k,
        "n": n,
        "in_dtype": "float32",
        "out_dtype": "float32",
        "transpose_a": ta,
        "transpose_b": tb,
    }
    self._setup_benchmark(**params)
    inputs = self.bm.generate_inputs()
    a, b = inputs[0], inputs[1]
    self.assertEqual(
        a.shape,
        expected_a_shape,
        f"Failed a.shape for ta={ta}, tb={tb} using transpose_a={ta}, "
        f"transpose_b={tb}",
    )
    self.assertEqual(
        b.shape,
        expected_b_shape,
        f"Failed b.shape for ta={ta}, tb={tb} using transpose_a={ta}, "
        f"transpose_b={tb}",
    )

  @parameterized.named_parameters(
      ("normal", False, False),
      ("transposed_a", True, False),
      ("transposed_b", False, True),
      ("both_transposed", True, True),
  )
  def test_run_op_transposed(self, ta, tb):
    """Test output shape and numerical correctness for transposed matmuls."""
    m, k, n = 32, 64, 128
    params = {
        "m": m,
        "k": k,
        "n": n,
        "in_dtype": "float32",
        "out_dtype": "float32",
        "transpose_a": ta,
        "transpose_b": tb,
    }
    self._setup_benchmark(**params)
    inputs = self.bm.generate_inputs()
    a, b = inputs[0], inputs[1]
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (32, 128))

    a_ref = a.T if ta else a
    b_ref = b.T if tb else b
    expected_out = jnp.matmul(a_ref, b_ref)
    np.testing.assert_allclose(out, expected_out, rtol=1e-4, atol=1e-4)

  def test_run_op_transposed_with_scaling_factors(self):
    """Test run op transposed with scaling factors."""
    m, k, n = 32, 64, 128
    params = {
        "m": m,
        "k": k,
        "n": n,
        "in_dtype": "float32",
        "out_dtype": "float32",
        "transpose_a": True,
        "transpose_b": True,
        "use_scaling_factors": True,
    }
    self._setup_benchmark(**params)
    inputs = self.bm.generate_inputs()
    self.assertLen(inputs, 4)
    a, b, sf0, sf1 = inputs[:4]
    self.assertEqual(a.shape, (k, m))
    self.assertEqual(b.shape, (n, k))
    self.assertEqual(sf0.shape, (m, 1))
    self.assertEqual(sf1.shape, (1, n))

    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (m, n))

    expected_out = jnp.matmul(a.T, b.T) * (sf0 @ sf1)
    np.testing.assert_allclose(out, expected_out, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      (
          "default",
          {},
          "m_32_k_64_n_128_bfloat16_to_bfloat16_ta_False_tb_False_alpha_1.0",
      ),
      (
          "transposed_a",
          {"transpose_a": True},
          "m_32_k_64_n_128_bfloat16_to_bfloat16_ta_True_tb_False_alpha_1.0",
      ),
      (
          "transposed_b",
          {"transpose_b": True},
          "m_32_k_64_n_128_bfloat16_to_bfloat16_ta_False_tb_True_alpha_1.0",
      ),
      (
          "both_transposed",
          {"transpose_a": True, "transpose_b": True},
          "m_32_k_64_n_128_bfloat16_to_bfloat16_ta_True_tb_True_alpha_1.0",
      ),
      (
          "with_alpha_not_equal_to_1",
          {"alpha": 2.0},
          "m_32_k_64_n_128_bfloat16_to_bfloat16_ta_False_tb_False_alpha_2.0",
      ),
      (
          "with_beta",
          {"beta": 2.0},
          "m_32_k_64_n_128_bfloat16_to_bfloat16_ta_False_tb_False_alpha_1.0_beta_2.0",
      ),
      (
          "with_scaling_factors",
          {"use_scaling_factors": True},
          "m_32_k_64_n_128_bfloat16_to_bfloat16_ta_False_tb_False_alpha_1.0_sf",
      ),
      (
          "with_beta_and_scaling_factors",
          {"beta": 1.5, "use_scaling_factors": True},
          "m_32_k_64_n_128_bfloat16_to_bfloat16_ta_False_tb_False_alpha_1.0_beta_1.5_sf",
      ),
      (
          "all_options",
          {
              "transpose_a": True,
              "transpose_b": True,
              "alpha": 0.5,
              "beta": 1.5,
              "use_scaling_factors": True,
          },
          "m_32_k_64_n_128_bfloat16_to_bfloat16_ta_True_tb_True_alpha_0.5_beta_1.5_sf",
      ),
  )
  def test_get_run_identifier(self, kwargs_override, expected_identifier):
    """Test run identifier formatting across various configuration options."""
    params = {
        "m": 32,
        "k": 64,
        "n": 128,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
    }
    params.update(kwargs_override)
    self._setup_benchmark(**params)
    self.assertEqual(self.bm.get_run_identifier(), expected_identifier)

  @parameterized.named_parameters(
      ("no_scaling_factors_alpha_1", False, 1.0),
      ("scaling_factors_alpha_1", True, 1.0),
      ("no_scaling_factors_alpha_2", False, 2.0),
      ("scaling_factors_alpha_2", True, 2.0),
  )
  def test_run_op_with_alpha_scaling_factor(
      self, use_scaling_factors, alpha
  ):
    """Test run op with scalar multiplier alpha and scaling factors."""
    m, k, n = 32, 64, 128
    params = {
        "m": m,
        "k": k,
        "n": n,
        "in_dtype": "float32",
        "out_dtype": "float32",
        "use_scaling_factors": use_scaling_factors,
        "alpha": alpha,
    }
    self._setup_benchmark(**params)
    inputs = self.bm.generate_inputs()
    a, b = inputs[0], inputs[1]
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (m, n))

    expected_out = alpha * jnp.matmul(a, b)
    if use_scaling_factors:
      sf0, sf1 = inputs[2], inputs[3]
      expected_out = expected_out * (sf0 @ sf1)
    np.testing.assert_allclose(out, expected_out, rtol=1e-4, atol=1e-4)

  def test_calculate_metrics_with_alpha(self):
    """Test FLOPs and arithmetic intensity calculation when alpha != 1.0."""
    self._setup_benchmark(alpha=0.5)
    # m=64, k=64, n=64
    # base_flops = 2 * 64 * 64 * 64 = 524288
    # alpha_flops = 64 * 64 = 4096
    # total_flops = 528384
    # bytes = 24576
    # intensity = 528384 / 24576 = 21.5
    metrics = self.bm.calculate_metrics([1.0, 1.5, 2.0])
    self.assertEqual(metrics["total_flops"], 528384)
    self.assertAlmostEqual(metrics["intensity"], 528384 / 24576)
    self.assertAlmostEqual(self.bm.get_arithmetic_intensity(), 528384 / 24576)

  def test_calculate_metrics_with_scaling_factors(self):
    """Test FLOPs and arithmetic intensity calculation when use_scaling_factors=True."""
    self._setup_benchmark(use_scaling_factors=True)
    # m=64, k=64, n=64
    # base_flops = 2 * 64 * 64 * 64 = 524288
    # scaling_flops = 2 * 64 * 64 = 8192
    # total_flops = 532480
    # bytes = 24576
    # intensity = 532480 / 24576
    metrics = self.bm.calculate_metrics([1.0, 1.5, 2.0])
    self.assertEqual(metrics["total_flops"], 532480)
    self.assertAlmostEqual(metrics["intensity"], 532480 / 24576)
    self.assertAlmostEqual(self.bm.get_arithmetic_intensity(), 532480 / 24576)

  def test_generate_inputs_with_accumulator_matrix(self):
    """Test generating inputs with accumulator matrix (beta != 0.0)."""
    self._setup_benchmark(beta=1.0)
    inputs = self.bm.generate_inputs()
    self.assertLen(inputs, 5)
    a, b, sf0, sf1, c = inputs
    self.assertEqual(a.shape, (64, 64))
    self.assertEqual(b.shape, (64, 64))
    self.assertIsNone(sf0)
    self.assertIsNone(sf1)
    self.assertEqual(c.shape, (64, 64))

  def test_run_op_with_accumulator_matrix(self):
    """Test run op with accumulator matrix and beta scaling factor."""
    m, k, n = 32, 64, 128
    params = {
        "m": m,
        "k": k,
        "n": n,
        "in_dtype": "float32",
        "out_dtype": "float32",
        "beta": 2.0,
    }
    self._setup_benchmark(**params)
    inputs = self.bm.generate_inputs()
    a, b, _, _, c = inputs
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (m, n))

    expected_out = jnp.matmul(a, b) + 2.0 * c
    np.testing.assert_allclose(out, expected_out, rtol=1e-4, atol=1e-4)

  def test_calculate_metrics_with_accumulator_matrix(self):
    """Test FLOPs and bytes calculation when beta != 0.0."""
    self._setup_benchmark(beta=2.0)
    # m=64, k=64, n=64
    # base_flops = 2 * 64 * 64 * 64 = 524288
    # beta_flops (beta != 1.0) = 2 * 64 * 64 = 8192
    # total_flops = 532480
    # bytes = 24576 + (64 * 64 * 2) = 32768
    # intensity = 532480 / 32768 = 16.25
    metrics = self.bm.calculate_metrics([1.0, 1.5, 2.0])
    self.assertEqual(metrics["total_flops"], 532480)
    self.assertEqual(self.bm.get_total_bytes(), 32768.0)
    self.assertAlmostEqual(metrics["intensity"], 532480 / 32768)
    self.assertAlmostEqual(self.bm.get_arithmetic_intensity(), 532480 / 32768)



if __name__ == "__main__":
  absltest.main()


