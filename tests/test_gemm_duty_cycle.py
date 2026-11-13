"""Unit tests for the gemm_duty_cycle benchmark (CPU)."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import matmul
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import dtypes
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")


class GemmDutyCycleBenchmarkTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    for patcher in (
        mock.patch(
            "accelerator_microbenchmarks.core.platform.get_platform_info",
            return_value=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
        ),
        # Few, long idle loops so CPU timing noise stays small.
        mock.patch.object(matmul, "_CALIBRATION_GEMMS", 2),
        mock.patch.object(matmul, "_CALIBRATION_ELEM_OPS", 2000),
    ):
      patcher.start()
      self.addCleanup(patcher.stop)
    self.mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def _make(self, **kwargs) -> matmul.GemmDutyCycleBenchmark:
    params = dict(m=64, k=64, n=64, gemms_per_run=4, warmup_tries=1, num_runs=2)
    params.update(kwargs)
    return matmul.GemmDutyCycleBenchmark(
        config=matmul.GemmDutyCycleParams(**params),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mesh,
    )

  @parameterized.parameters(
      (1e-3, 1e-6, 1.0, 0),
      (1e-3, 1e-6, 0.5, 1000),
      (1e-3, 1e-6, 0.2, 4000),
      (1e-3, 1.0, 0.99, 1),
  )
  def test_calibrate_num_elem_ops(self, gemm_s, elem_op_s, duty, expected):
    self.assertEqual(
        matmul.calibrate_num_elem_ops(gemm_s, elem_op_s, duty), expected
    )

  @parameterized.parameters(
      {"duty_cycle": 0.0}, {"duty_cycle": 1.5}, {"rhs_dtype": "bf16"}
  )
  def test_rejects_invalid_params(self, **kwargs):
    with self.assertRaises(ValueError):
      self._make(**kwargs)

  @parameterized.parameters(
      ("bfloat16", ""),
      ("float8_e4m3fn", ""),
      ("int8", ""),
      ("float8_e4m3fn", "float4_e2m1fn"),
      ("float8_e4m3fn", "int4"),
  )
  def test_loop_dtypes(self, in_dtype, rhs_dtype):
    bm = self._make(in_dtype=in_dtype, rhs_dtype=rhs_dtype)
    bm.setup()
    b_dtype = rhs_dtype or in_dtype
    args = (
        jnp.ones((64, 64), in_dtype),
        jnp.ones((64, 64), b_dtype),
        jnp.int32(3),
        jnp.int32(5),
    )
    if dtypes.is_sub_byte_dtype(b_dtype):
      # XLA:CPU cannot compile sub-byte matmuls; check that it lowers.
      bm._jit_fn.lower(*args)  # pylint: disable=protected-access
    else:
      out = bm.run_op(*args)
      self.assertEqual(out.shape, (len(jax.devices()),))
      self.assertTrue(np.all(np.isfinite(np.asarray(out))))

  def test_bytes_and_compute_dtype(self):
    bm = self._make(in_dtype="float8_e4m3fn", rhs_dtype="int4")
    self.assertEqual(bm.get_compute_dtype(), "float8_e4m3fn")
    self.assertEqual(bm.get_total_flops(), 4 * 2 * 64**3)
    self.assertEqual(bm.get_total_bytes(), 4 * 64 * 64 * (1 + 0.5 + 2))

  def test_throughput_metrics(self):
    bm = self._make(duty_cycle=0.5)
    bm._gemm_s = 1e-3  # pylint: disable=protected-access
    metrics = bm.calculate_throughput_metrics(
        latency_ms=8.0, prefix=constants.TimingDomain.WALL_CLOCK
    )
    avg = 4 * 2 * 64**3 / 8e-3 / 1e12
    self.assertAlmostEqual(metrics["wall_clock_tflops_per_device"], avg)
    self.assertAlmostEqual(metrics["wall_clock_duty_cycle_pct"], 50.0)
    self.assertAlmostEqual(
        metrics["wall_clock_active_tflops_per_device"], 2 * avg
    )

  @parameterized.parameters(
      {"duty_cycle": 1.0},
      {"duty_cycle": 0.5},
      {"duty_cycle": 0.5, "num_elem_ops": 7},
      {"in_dtype": "int8", "out_dtype": "int32"},
  )
  def test_run(self, **kwargs):
    result = self._make(**kwargs).run()
    self.assertGreater(result.metrics["wall_clock_tflops_per_device"], 0.0)
    expected_ops = kwargs.get("num_elem_ops")
    if expected_ops is not None:
      self.assertEqual(result.metrics["num_elem_ops"], expected_ops)
    elif kwargs.get("duty_cycle", 1.0) < 1.0:
      self.assertGreaterEqual(result.metrics["num_elem_ops"], 1)
    else:
      self.assertEqual(result.metrics["num_elem_ops"], 0)


if __name__ == "__main__":
  absltest.main()
