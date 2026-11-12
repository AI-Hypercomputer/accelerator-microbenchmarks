"""Unit tests for matlmul.py."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import matmul
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


class GeneralizedGemmBenchmarkTest(parameterized.TestCase):
  """Unit tests for Generalized GEMM benchmark."""

  def setUp(self):
    super().setUp()
    self._platform_patcher = mock.patch(
        "accelerator_microbenchmarks.core.platform.get_platform_info",
        return_value=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
    )
    self._platform_patcher.start()
    self.addCleanup(self._platform_patcher.stop)
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
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.bm.setup()

  def test_benchmark_registered(self):
    """Test that the benchmark is properly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("gemm")
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

  def test_replicated_normal_passes_key_as_host_array(self):
    """The PRNG key reaches the jitted sampler as a host array.

    Every input has to arrive via the host->device path, including the key,
    so that no data moves between devices during input generation.
    """
    self._setup_benchmark()
    seen = {}
    real_jit = jax.jit

    def capturing_jit(fn, **kwargs):
      jitted = real_jit(fn, **kwargs)

      def call(key, *args):
        seen["key"] = key
        return jitted(key, *args)

      return call

    with mock.patch.object(jax, "jit", side_effect=capturing_jit):
      out = self.bm._replicated_normal(
          jax.random.PRNGKey(3), (8, 8), jnp.float32
      )
    self.assertIsInstance(seen["key"], np.ndarray)
    self.assertTrue(out.sharding.is_fully_replicated)
    np.testing.assert_array_equal(
        np.asarray(out),
        np.asarray(jax.random.normal(jax.random.PRNGKey(3), (8, 8))),
    )

  def test_generate_inputs_replicated_on_every_device_without_device_put(
      self,
  ):
    """Operands are generated on every local device, not broadcast from one.

    They must be replicated, identical on each device, match eager sampling
    with the same key, and use no cross-device `jax.device_put`.
    """
    self._setup_benchmark(seed=7, use_scaling_factors=True, beta=1.0)
    with mock.patch.object(
        jax, "device_put", wraps=jax.device_put
    ) as device_put:
      inputs = self.bm.generate_inputs()
    device_put.assert_not_called()
    self.assertLen(inputs, 5)
    for arr in inputs:
      self.assertTrue(arr.sharding.is_fully_replicated)
      self.assertEqual(
          set(arr.sharding.device_set), set(self.mock_mesh.devices.flat)
      )
      self.assertLen(arr.addressable_shards, len(self.mock_mesh.devices.flat))
      first = np.asarray(arr.addressable_shards[0].data)
      for shard in arr.addressable_shards[1:]:
        np.testing.assert_array_equal(np.asarray(shard.data), first)

    key = jax.random.PRNGKey(7)
    k1, _, _, _, k5 = jax.random.split(key, 5)
    np.testing.assert_array_equal(
        np.asarray(inputs[0]),
        np.asarray(jax.random.normal(k1, (64, 64)).astype(jnp.bfloat16)),
    )
    np.testing.assert_array_equal(
        np.asarray(inputs[4]),
        np.asarray(jax.random.normal(k5, (64, 64)).astype(jnp.bfloat16)),
    )

  def test_reset_data_reuses_inputs(self):
    """Inputs are generated once and reused across timed runs."""
    self._setup_benchmark()
    inputs = self.bm.generate_inputs()
    for new, old in zip(self.bm.reset_data(*inputs), inputs, strict=True):
      self.assertIs(new, old)

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
    self.assertIn("wall_clock_avg_ms", metrics)
    self.assertIn("wall_clock_tflops_per_device", metrics)
    self.assertIn("total_flops", metrics)
    self.assertIn("intensity", metrics)
    self.assertEqual(metrics["total_flops"], 524288)
    self.assertAlmostEqual(metrics["intensity"], 524288 / 24576)

  def test_is_xprof_op(self):
    """Test that is_xprof_op correctly identifies convolution fusion events."""
    config = matmul.GemmParams(m=128, n=128, k=128)
    self.bm = matmul.GeneralizedGemmBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
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
  def test_run_op_with_alpha_scaling_factor(self, use_scaling_factors, alpha):
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

  def test_format_benchmark_table(self):
    """Tests formatting of GEMM benchmark tables."""
    res = base.BenchmarkResult(
        metadata=base.BenchmarkMetadata(
            benchmark_name="GeneralizedGemmBenchmark",
            test_name="GeneralizedGemmBenchmark_test",
            start_time="2026-08-18T10:00:00",
            end_time="2026-08-18T10:01:00",
            params={
                "in_dtype": "bfloat16",
                "out_dtype": "float32",
                "m": 4096,
                "k": 2048,
                "n": 8192,
                "transpose_a": True,
                "transpose_b": False,
                "alpha": 1.5,
                "beta": 0.5,
                "use_scaling_factors": True,
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "total_flops": 137438953472.0,
            "wall_clock_tflops_per_device": 331.25,
            "wall_clock_tflops_per_chip": 662.50,
            "xprof_tflops_per_device": 340.00,
            "xprof_tflops_per_chip": 680.00,
            "wall_clock_p50_ms": 0.4200,
            "xprof_p50_ms": 0.4100,
        },
        timings=base.BenchmarkTimings(
            wall_clock=base.WallClockTimings(durations_ms=[1.0])
        ),
    )
    expected_cols = [
        "in_dtype",
        "out_dtype",
        "m",
        "k",
        "n",
        "transpose_a",
        "transpose_b",
        "alpha",
        "beta",
        "use_scaling_factors",
        "total_flops",
        "wall_clock_p50_ms",
        "wall_clock_tflops_per_chip",
        "xprof_p50_ms",
        "xprof_tflops_per_chip",
    ]
    schema_cols = [
        col for col, _ in matmul.GeneralizedGemmBenchmark.REPORT_SCHEMA
    ]
    self.assertEqual(schema_cols, expected_cols)

    df = report.results_to_dataframe([res])
    table = report.format_benchmark_table(
        df,
        schema=matmul.GeneralizedGemmBenchmark.REPORT_SCHEMA,
        title="GeneralizedGemmBenchmark",
    )
    self.assertIn("Benchmark Results (GeneralizedGemmBenchmark)", table)
    for col in expected_cols:
      self.assertIn(col, table)
    self.assertIn("bfloat16", table)
    self.assertIn("float32", table)
    self.assertIn("4096", table)
    self.assertIn("2048", table)
    self.assertIn("8192", table)
    self.assertIn("True", table)
    self.assertIn("False", table)
    self.assertIn("1.5", table)
    self.assertIn("0.5", table)
    self.assertIn("137438953472.00", table)
    self.assertIn("662.50", table)
    self.assertIn("680.00", table)
    self.assertIn("0.4200", table)
    self.assertIn("0.4100", table)

  def test_schema_coverage(self):
    """Verify REPORT_SCHEMA matches output keys and covers all metrics."""
    self._setup_benchmark()
    test_report_utils.assert_schema_matches_output(
        self,
        self.bm,
    )

  def test_gemm_params_default_dtypes(self):
    """Test GemmParams defaults in_dtype and out_dtype to bfloat16 and excludes dtype."""
    # 1. Default instance without explicit dtypes
    params = matmul.GemmParams()
    self.assertFalse(hasattr(params, "dtype"))
    self.assertEqual(params.in_dtype, "bfloat16")
    self.assertEqual(params.out_dtype, "bfloat16")

    # 2. Explicit in_dtype and out_dtype are preserved
    params_custom = matmul.GemmParams(
        in_dtype="float8_e4m3fn", out_dtype="float32"
    )
    self.assertFalse(hasattr(params_custom, "dtype"))
    self.assertEqual(params_custom.in_dtype, "float8_e4m3fn")
    self.assertEqual(params_custom.out_dtype, "float32")

    # 3. Passing dtype raises TypeError
    with self.assertRaises(TypeError):
      matmul.GemmParams(dtype="bfloat16")  # pytype: disable=wrong-keyword-args

  def test_get_compute_dtype(self):
    """Verifies GeneralizedGemmBenchmark.get_compute_dtype returns in_dtype."""
    self._setup_benchmark(in_dtype="float8_e4m3fn", out_dtype="bfloat16")
    self.assertEqual(self.bm.get_compute_dtype(), "float8_e4m3fn")

  def test_run_orchestration_roofline_metrics(self):
    """Verifies end-to-end run emits compute roofline metrics."""
    params = {
        "m": 64,
        "k": 64,
        "n": 64,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
        "warmup_tries": 1,
        "num_runs": 2,
    }
    config = matmul.GemmParams(**params)
    bm = matmul.GeneralizedGemmBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    result = bm.run()
    self.assertIn("roofline_tflops_limit_per_device", result.metrics)
    self.assertIn("peak_hbm_bw_per_device_gb_s", result.metrics)
    self.assertIn("wall_clock_compute_roofline_efficiency_pct", result.metrics)
    self.assertGreater(result.metrics["roofline_tflops_limit_per_device"], 0.0)
    self.assertGreater(result.metrics["peak_hbm_bw_per_device_gb_s"], 0.0)
    self.assertGreaterEqual(
        result.metrics["wall_clock_compute_roofline_efficiency_pct"], 0.0
    )


class GemmParamsValidationTest(parameterized.TestCase):
  """Verifies the bounds declared on GemmParams fields."""

  @parameterized.parameters(
      ("m", -1, 1),
      ("m", 0, 1),
      ("k", 0, 1),
      ("n", -5, 1),
      ("seed", -1, 0),
  )
  def test_out_of_range_values_raise_error(self, field, value, bound):
    with self.assertRaisesRegex(ValueError, f"{field} must be >= {bound}"):
      matmul.GemmParams(**{field: value})


class GemmThrottlingBenchmarkTest(parameterized.TestCase):
  """Unit tests for GemmThrottlingBenchmark."""

  def setUp(self):
    super().setUp()
    self._platform_patcher = mock.patch(
        "accelerator_microbenchmarks.core.platform.get_platform_info",
        return_value=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
    )
    self._platform_patcher.start()
    self.addCleanup(self._platform_patcher.stop)
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def _setup_benchmark(self, **kwargs):
    """Initializes a GemmThrottlingBenchmark instance for testing."""
    params = {
        "m": 64,
        "k": 64,
        "n": 64,
        "in_dtype": "bfloat16",
        "out_dtype": "bfloat16",
        "min_duration_s": 0.01,
        "samples_per_run": 5,
    }
    params.update(kwargs)
    config = matmul.GemmThrottlingParams(**params)
    self.bm = matmul.GemmThrottlingBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.bm.setup()

  def test_benchmark_registered(self):
    """Verifies that gemm_throttling is registered in the benchmark registry."""
    bm_class = registry.benchmark_registry.get_benchmark("gemm_throttling")
    self.assertEqual(bm_class, matmul.GemmThrottlingBenchmark)

  def test_gemm_throttling_params_defaults_and_validation(self):
    """Verifies GemmThrottlingParams defaults samples_per_run=25 and validates min_duration_s > 0.0."""
    default_cfg = matmul.GemmThrottlingParams(min_duration_s=120.0)
    self.assertEqual(default_cfg.min_duration_s, 120.0)
    self.assertEqual(default_cfg.samples_per_run, 25)

    with self.assertRaisesRegex(
        ValueError, "min_duration_s must be > 0.0 for soaking benchmarks."
    ):
      matmul.GemmThrottlingParams()

    cfg = matmul.GemmThrottlingParams(
        m=128,
        k=128,
        n=128,
        in_dtype="bfloat16",
        out_dtype="bfloat16",
        min_duration_s=60.0,
        samples_per_run=4,
    )
    cases = cfg.expand_test_cases()
    self.assertLen(cases, 1)
    self.assertEqual((cases[0].m, cases[0].k, cases[0].n), (128, 128, 128))
    self.assertEqual(cases[0].min_duration_s, 60.0)
    self.assertEqual(cases[0].samples_per_run, 4)

    with self.assertRaisesRegex(
        ValueError, "min_duration_s must be > 0.0 for soaking benchmarks."
    ):
      matmul.GemmThrottlingParams(min_duration_s=0.0)

  def test_generate_inputs_and_run_op_replicated(self):
    """Verifies replicated input generation and matmul execution."""
    self._setup_benchmark()
    inputs = self.bm.generate_inputs()
    self.assertLen(inputs, 4)
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (64, 64))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_schema_coverage(self):
    """Verifies REPORT_SCHEMA bidirectional contract for GemmThrottlingBenchmark."""
    self._setup_benchmark()
    test_report_utils.assert_schema_matches_output(
        self,
        self.bm,
    )


class QuantizedGemmBenchmarkTest(parameterized.TestCase):
  """Unit tests for QuantizedGemmBenchmark."""

  def setUp(self):
    super().setUp()
    self._platform_patcher = mock.patch(
        "accelerator_microbenchmarks.core.platform.get_platform_info",
        return_value=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
    )
    self._platform_patcher.start()
    self.addCleanup(self._platform_patcher.stop)
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def _setup_benchmark(self, **kwargs):
    """Initializes a QuantizedGemmBenchmark instance for testing."""
    params = {
        "m": 64,
        "k": 128,
        "n": 64,
        "in_dtype": "bfloat16",
        "quant_dtype": "float8_e4m3fn",
        "out_dtype": "bfloat16",
        "warmup_tries": 1,
        "num_runs": 2,
    }
    params.update(kwargs)
    cfg = matmul.QuantizedGemmParams(**params)
    self.bm = matmul.QuantizedGemmBenchmark(
        config=cfg,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.bm.setup()

  def test_benchmark_registered(self):
    """Tests that gemm_quantized is registered in the benchmark registry."""
    bm_class = registry.benchmark_registry.get_benchmark("gemm_quantized")
    self.assertEqual(bm_class, matmul.QuantizedGemmBenchmark)

  def test_generate_inputs_and_run_op(self):
    """Tests input generation and basic execution of QuantizedGemmBenchmark."""
    self._setup_benchmark()
    inputs = self.bm.generate_inputs()
    self.assertLen(inputs, 2)
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (64, 64))
    self.assertEqual(out.dtype, jnp.bfloat16)

  @parameterized.parameters(
      (
          "dynamic",
          False,
          "m_64_k_128_n_64_bfloat16_to_float8_e4m3fn_dynamic_out_bfloat16",
          25690112.0,
      ),
      (
          "static",
          False,
          "m_64_k_128_n_64_bfloat16_to_float8_e4m3fn_static_out_bfloat16",
          3211264.0,
      ),
      (
          "dynamic",
          True,
          "m_64_k_128_n_64_bfloat16_to_float8_e4m3fn_dynamic_out_bfloat16_dequant",
          802816.0,
      ),
      (
          "static",
          True,
          "m_64_k_128_n_64_bfloat16_to_float8_e4m3fn_static_out_bfloat16_dequant",
          802816.0,
      ),
  )
  def test_scaling_and_dequantize_variants(
      self, scaling, dequantize, expected_run_id, expected_val
  ):
    """Tests run IDs and numerical outputs across scaling and dequantize modes."""
    self._setup_benchmark(scaling=scaling, dequantize=dequantize)
    self.assertEqual(self.bm.get_run_identifier(), expected_run_id)
    inputs = self.bm.generate_inputs()
    out = self.bm.run_op(*inputs)
    self.assertEqual(out.shape, (64, 64))
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

    # Verify exact numerical quantization and dequantization with dyadic inputs:
    # A = 112.0 (dynamic sf_a = 0.25, qa = 448; static sf_a = 0.5, qa = 224)
    # B = 56.0  (dynamic sf_b = 0.125, qb = 448; static sf_b = 0.5, qb = 112)
    a = jnp.full((64, 128), 112.0, dtype=jnp.bfloat16)
    b = jnp.full((128, 64), 56.0, dtype=jnp.bfloat16)
    exact_out = self.bm.run_op(a, b)
    np.testing.assert_allclose(
        np.asarray(exact_out, dtype=np.float32),
        np.full((64, 64), expected_val, dtype=np.float32),
    )

  def test_int8_quant_dtype_rounds_and_accumulates_int32(self):
    """Tests that int8 quantization rounds to nearest integer and uses int32."""
    self._setup_benchmark(
        m=2,
        k=2,
        n=2,
        quant_dtype="int8",
        scaling="dynamic",
        dequantize=True,
    )
    # Row 0 of A has amax = 127.0 -> scale = 1.0; 0.6 rounds to 1 (not 0).
    a = jnp.array([[127.0, 0.6], [63.5, -63.5]], dtype=jnp.bfloat16)
    b = jnp.array([[127.0, 63.5], [127.0, -63.5]], dtype=jnp.bfloat16)
    out = self.bm.run_op(a, b)
    self.assertEqual(out.shape, (2, 2))
    self.assertEqual(out.dtype, jnp.bfloat16)
    # Row 0 dot Col 0: q_a=[127, 1], q_b=[127, 127], sf_a=1.0, sf_b=1.0
    # -> (127*127 + 1*127) * 1.0 = 16256.0 (would be 16129.0 without round).
    self.assertEqual(float(out[0, 0]), 16256.0)

  def test_static_scaling_clips_outliers_to_qmax(self):
    """Tests that static scaling clips values exceeding 0.5 * q_max to q_max."""
    self._setup_benchmark(
        m=2,
        k=2,
        n=2,
        quant_dtype="float8_e4m3fn",
        scaling="static",
        dequantize=False,
    )
    # Values > 224.0 exceed q_max (448.0) after dividing by static scale 0.5;
    # without clipping to [-q_max, q_max], casting 1000.0 to float8_e4m3fn
    # overflows to NaN.
    a = jnp.array([[500.0, -500.0], [112.0, 0.0]], dtype=jnp.bfloat16)
    b = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.bfloat16)
    out = self.bm.run_op(a, b)
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    # q_a[0] clips to [448.0, -448.0] and q_b is [[2.0, 0.0], [0.0, 2.0]].
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float32),
        np.array([[896.0, -896.0], [448.0, 0.0]], dtype=np.float32),
    )

  def test_params_validation(self):
    """Tests that invalid scaling values raise ValueError."""
    with self.assertRaisesRegex(ValueError, "Invalid scaling"):
      matmul.QuantizedGemmParams(**{"scaling": "invalid"})

  def test_flops_bytes_and_compute_dtype(self):
    """Tests FLOPs, bytes, and compute dtype reporting."""
    self._setup_benchmark(m=64, k=128, n=64)
    self.assertEqual(self.bm.get_compute_dtype(), "float8_e4m3fn")
    self.assertEqual(self.bm.get_total_flops(), float(2 * 64 * 128 * 64))
    # 1-pass algorithmic HBM bytes: read A (64x128 bf16) + read B (128x64 bf16)
    # + write OUT (64x64 bf16).
    expected_bytes = float((64 * 128 + 128 * 64) * 2 + 64 * 64 * 2)
    self.assertEqual(self.bm.get_total_bytes(), expected_bytes)

  def test_schema_coverage(self):
    """Tests that benchmark output matches the report schema."""
    self._setup_benchmark()
    test_report_utils.assert_schema_matches_output(self, self.bm)


if __name__ == "__main__":
  absltest.main()
