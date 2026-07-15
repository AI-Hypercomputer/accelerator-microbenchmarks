"""Unit tests for the BaseBenchmark execution loop on CPU."""

import contextlib
import unittest

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
import jax
import jax.numpy as jnp
import numpy as np


# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class DummyBenchmark(base.BaseBenchmark):
  """A dummy benchmark for testing the BaseBenchmark class."""

  def run_op(self, x):
    return x * 2.0

  def generate_inputs(self, **_params):
    return (jnp.ones((10, 10)),)

  def get_arithmetic_intensity(self, **_params):
    return 1.0

  def get_total_bytes(self, **_params):
    return 400.0


class BaseBenchmarkTest(absltest.TestCase):
  """Tests for the BaseBenchmark class."""

  def test_calculate_metrics_iqr(self):
    bm = DummyBenchmark()
    # Deliberately introduce outliers
    times_ms = [10.0, 11.0, 10.5, 9.5, 100.0, 0.1, 10.2]
    metrics = bm.calculate_metrics(times_ms)

    # 100.0 and 0.1 should be filtered out by IQR
    # Left with [10.0, 11.0, 10.5, 9.5, 10.2] -> mean should be ~10.24
    self.assertGreater(metrics["avg_ms"], 9.0)
    self.assertLess(metrics["avg_ms"], 12.0)

  def test_init_without_mesh(self):
    bm = DummyBenchmark()
    self.assertIsNone(bm.mesh)
    self.assertEqual(bm.warmup_tries, 10)
    self.assertEqual(bm.num_runs, 10)
    self.assertIsNone(bm._jit_fn)  # pylint: disable=protected-access

  def test_init_with_mesh(self):
    mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    bm = DummyBenchmark(mesh=mock_mesh)
    self.assertEqual(bm.mesh, mock_mesh)

  def test_create_default_mesh(self):
    bm = DummyBenchmark()
    mesh = bm._create_default_mesh()  # pylint: disable=protected-access
    self.assertIsInstance(mesh, jax.sharding.Mesh)
    self.assertEqual(mesh.axis_names, ("device",))

  def test_get_roofline_performance_float(self):
    bm = DummyBenchmark()
    perf = bm.get_roofline_performance(peak_tflops=100.0, hbm_bw_data=200.0)
    self.assertAlmostEqual(perf, 0.2)

  def test_get_roofline_performance_list(self):
    bm = DummyBenchmark()
    hbm_bw_data = [(100, 50.0), (500, 250.0)]
    perf = bm.get_roofline_performance(
        peak_tflops=100.0, hbm_bw_data=hbm_bw_data
    )
    self.assertAlmostEqual(perf, 0.2)

  def test_get_roofline_performance_dict(self):
    bm = DummyBenchmark()
    hbm_bw_data = {100: 50.0, 500: 250.0}
    perf = bm.get_roofline_performance(
        peak_tflops=100.0, hbm_bw_data=hbm_bw_data
    )
    self.assertAlmostEqual(perf, 0.2)

  def test_calculate_metrics_empty(self):
    bm = DummyBenchmark()
    metrics = bm.calculate_metrics([])
    self.assertEqual(metrics["avg_ms"], 0.0)
    self.assertEqual(metrics["throughput"], 0.0)

  @unittest.mock.patch("jax.experimental.roofline.roofline")
  def test_get_trace_metrics(self, mock_roofline):
    mock_result = unittest.mock.Mock()
    mock_result.flops = 1000
    mock_result.hbm_bytes = 500
    mock_roofline.return_value = lambda *args: (None, mock_result)

    bm = DummyBenchmark()
    metrics = bm.get_trace_metrics()
    self.assertIsNotNone(metrics)
    self.assertEqual(metrics["flops"], 1000)
    self.assertEqual(metrics["hbm_bytes"], 500)

  def test_run_orchestration(self):
    """Tests the full run orchestration of the BaseBenchmark."""
    bm = DummyBenchmark()

    params = {
        "warmup_tries": 2,
        "num_runs": 5,
        "hardware_stats": {
            "tflops": {"float32": 100.0},
            "hbm_bw": [(1024, 100.0), (1048576, 200.0)],
            "ici": {"peak_bw_gbps": 50.0, "bidirectional": True},
        },
        "dtype": "float32",
        "xprof_timing": False,
    }

    result = bm.run(**params)

    self.assertEqual(result.metadata.benchmark_name, "DummyBenchmark")
    self.assertEqual(result.metrics["actual_runs"], 5)

    # Validate Roofline values computed correctly
    self.assertIn("roofline_tflops_limit", result.metrics)
    self.assertIn("peak_bw_at_size_gb_s", result.metrics)
    self.assertEqual(
        result.metrics["peak_bw_at_size_gb_s"], 100.0
    )  # size is 400 bytes, < 1024

  @unittest.mock.patch("jax.profiler.trace")
  def test_xprof_naming_with_identifier(self, mock_trace):
    class BenchmarkWithId(DummyBenchmark):

      def get_run_identifier(self, **params) -> str:
        return f"test_id_{params.get('val')}"

    bm = BenchmarkWithId()
    mock_trace.return_value = contextlib.nullcontext()

    params = {
        "warmup_tries": 1,
        "num_runs": 1,
        "xprof_timing": True,
        "xprof_dir": "/tmp/test_xprof",
        "val": 42,
    }
    bm.run(**params)

    mock_trace.assert_called_once()
    called_path = mock_trace.call_args[0][0]
    self.assertIn("BenchmarkWithId_test_id_42_", called_path)
    self.assertTrue(called_path.startswith("/tmp/test_xprof/"))

  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @unittest.mock.patch("jax.profiler.trace")
  def test_run_with_xprof_timing(
      self, mock_trace, mock_upload, mock_parse_durations
  ):
    """Tests that benchmark uses XProf durations for derived metrics only."""
    bm = DummyBenchmark()
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = "http://mock_xprof_url"
    # Mock XProf durations to be exactly 5.0 ms
    mock_parse_durations.return_value = [5.0, 5.0, 5.0, 5.0, 5.0]

    params = {
        "warmup_tries": 1,
        "num_runs": 5,
        "xprof_timing": True,
        "xprof_dir": "/tmp/test_xprof",
    }
    result = bm.run(**params)

    # Assert that the base metrics retain the host timings (not 5.0 ms)
    self.assertNotEqual(result.metrics.get("avg_ms"), 5.0)
    self.assertNotEqual(result.metrics.get("p50_ms"), 5.0)

    # Assert XProf metrics correctly reflect the mocked XProf durations
    self.assertEqual(result.metrics["xprof_avg_ms"], 5.0)
    self.assertEqual(result.metrics["xprof_url"], "http://mock_xprof_url")

    mock_upload.assert_called_once()
    mock_parse_durations.assert_called_once()


if __name__ == "__main__":
  absltest.main()
