"""Unit tests for the BaseBenchmark execution loop on CPU."""

import contextlib
import dataclasses
import unittest

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
import jax
import jax.numpy as jnp
import numpy as np


# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class DummyBenchmark(base.BaseBenchmark):
  """A dummy benchmark for testing the BaseBenchmark class."""

  def __init__(self, config=None, hardware_spec=None, mesh=None):
    if config is None:
      config = base.BaseBenchmarkParams()
    if hardware_spec is None:
      hardware_spec = system.get_hardware_spec(system.TpuVersion.TPU7X)
    super().__init__(config=config, hardware_spec=hardware_spec, mesh=mesh)

  def run_op(self, x):
    return x * 2.0

  def generate_inputs(self, **_params):
    return (jnp.ones((10, 10)),)

  def get_arithmetic_intensity(self, **_params):
    return 1.0

  def get_total_bytes(self, **_params):
    return 400.0


class BaseBenchmarkParamsTest(absltest.TestCase):
  """Tests for BaseBenchmarkParams configuration and expansion contract."""

  def test_expand_test_cases_default_identity(self):
    """Verifies default 1-to-1 identity mapping: returns [self]."""
    params = base.BaseBenchmarkParams(warmup_tries=5, num_runs=20)
    cases = params.expand_test_cases()
    self.assertLen(cases, 1)
    self.assertIs(cases[0], params)
    self.assertEqual(cases[0].warmup_tries, 5)
    self.assertEqual(cases[0].num_runs, 20)

  def test_expand_test_cases_subclass_inheritance(self):
    """Verifies subclasses without override inherit the [self] identity contract."""

    @dataclasses.dataclass
    class CustomConfig(base.BaseBenchmarkParams):
      custom_flag: str = "test"

    cfg = CustomConfig(warmup_tries=3, custom_flag="hello")
    cases = cfg.expand_test_cases()
    self.assertLen(cases, 1)
    self.assertIs(cases[0], cfg)
    self.assertEqual(cases[0].custom_flag, "hello")


class BaseBenchmarkTest(absltest.TestCase):
  """Tests for the BaseBenchmark class."""

  def setUp(self):
    super().setUp()
    self._platform_patcher = unittest.mock.patch(
        "accelerator_microbenchmarks.core.platform.get_platform_info",
        return_value=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
    )
    self._platform_patcher.start()

  def tearDown(self):
    self._platform_patcher.stop()
    super().tearDown()

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
    self.assertEqual(bm.config.warmup_tries, 10)
    self.assertEqual(bm.config.num_runs, 10)
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

    params = {
        "warmup_tries": 2,
        "num_runs": 5,
        "dtype": "float32",
        "xprof_timing": False,
    }

    config = base.BaseBenchmarkParams(**params)
    hw_spec = system.get_hardware_spec(system.TpuVersion.TPU7X)
    bm = DummyBenchmark(config=config, hardware_spec=hw_spec)
    result = bm.run()

    self.assertEqual(result.metadata.benchmark_name, "DummyBenchmark")
    self.assertEqual(result.metrics["actual_runs"], 5)
    self.assertIsNotNone(result.metadata.platform_info)
    self.assertEqual(result.metadata.hardware_spec, hw_spec)

    # Validate Roofline values computed correctly
    self.assertIn("roofline_tflops_limit", result.metrics)
    self.assertIn("peak_bw_at_size_gb_s", result.metrics)
    self.assertEqual(
        result.metrics["peak_bw_at_size_gb_s"], 100.0
    )  # size is 400 bytes, <= 1024

  @unittest.mock.patch("jax.profiler.trace")
  def test_xprof_naming_with_identifier(self, mock_trace):
    class BenchmarkWithId(DummyBenchmark):

      def get_run_identifier(self) -> str:
        return "test_id_42"

    mock_trace.return_value = contextlib.nullcontext()

    params = {
        "warmup_tries": 1,
        "num_runs": 1,
        "xprof_timing": True,
        "xprof_dir": "/tmp/test_xprof",
    }

    config = base.BaseBenchmarkParams(**params)
    bm = BenchmarkWithId(config=config)
    bm.run()

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

    config = base.BaseBenchmarkParams(**params)
    bm = DummyBenchmark(config=config)
    result = bm.run()

    # Assert that the base metrics retain the host timings (not 5.0 ms)
    self.assertNotEqual(result.metrics.get("avg_ms"), 5.0)
    self.assertNotEqual(result.metrics.get("p50_ms"), 5.0)

    # Assert XProf metrics correctly reflect the mocked XProf durations
    self.assertEqual(result.metrics["xprof_avg_ms"], 5.0)
    self.assertEqual(result.metrics["xprof_url"], "http://mock_xprof_url")

    mock_upload.assert_called_once()
    mock_parse_durations.assert_called_once()

  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @unittest.mock.patch("jax.profiler.trace")
  def test_apply_xprof_timing_and_sync_host_cpu_override(
      self, mock_trace, mock_upload, mock_parse_durations
  ):
    """Tests that xprof_target_host_cpu=True passes local_device_id=None."""

    class HostCpuBenchmark(DummyBenchmark):

      @property
      def xprof_target_host_cpu(self) -> bool:
        return True

    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = "http://mock_xprof_url"
    mock_parse_durations.return_value = [2.0]

    params = {
        "warmup_tries": 1,
        "num_runs": 1,
        "xprof_timing": True,
        "xprof_dir": "/tmp/test_xprof",
    }
    config = base.BaseBenchmarkParams(**params)
    bm = HostCpuBenchmark(config=config)
    bm.run()

    mock_parse_durations.assert_called_once()
    self.assertIsNone(
        mock_parse_durations.call_args.kwargs.get("local_device_id")
    )

  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @unittest.mock.patch("jax.profiler.trace")
  def test_apply_xprof_timing_and_sync_empty_durations_fallback(
      self, mock_trace, mock_upload, mock_parse_durations
  ):
    """Tests that empty XProf durations set xprof metrics to None while keeping host timings."""
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = None
    mock_parse_durations.return_value = []  # Empty durations

    params = {
        "warmup_tries": 1,
        "num_runs": 1,
        "xprof_timing": True,
        "xprof_dir": "/tmp/test_xprof",
    }
    config = base.BaseBenchmarkParams(**params)
    bm = DummyBenchmark(config=config)
    result = bm.run()

    mock_parse_durations.assert_called_once()
    self.assertIsNone(result.metrics["xprof_avg_ms"])
    self.assertIsNone(result.metrics["xprof_p50_ms"])
    self.assertGreater(result.metrics["avg_ms"], 0.0)

  @unittest.mock.patch("jax.experimental.multihost_utils.broadcast_one_to_all")
  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @unittest.mock.patch("jax.profiler.trace")
  @unittest.mock.patch("jax.process_index")
  def test_apply_xprof_timing_and_sync_with_multihost_sync(
      self,
      mock_process_index,
      mock_trace,
      mock_upload,
      mock_parse_durations,
      mock_broadcast,
  ):
    """Tests requires_multihost_sync=True (e.g., asymmetric D2D): non-owner hosts use broadcast metrics."""

    class MultihostBenchmark(DummyBenchmark):

      @property
      def requires_multihost_sync(self) -> bool:
        return True

      def get_device_to_measure(self):
        mock_device = unittest.mock.Mock()
        mock_device.id = 0
        mock_device.process_index = 0  # Host 0 is the owner
        mock_device.local_hardware_id = 0
        return mock_device

    mock_process_index.return_value = 1  # Current host is Host 1 (non-owner)
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = "http://mock_xprof_url"
    mock_broadcast.return_value = jnp.array([4.2, 4.1, 4.5], dtype=jnp.float32)

    params = {
        "warmup_tries": 1,
        "num_runs": 1,
        "xprof_timing": True,
        "xprof_dir": "/tmp/test_xprof",
    }
    config = base.BaseBenchmarkParams(**params)
    bm = MultihostBenchmark(config=config)
    result = bm.run()

    # When requires_multihost_sync=True (e.g. Device-to-Device asymmetric
    # benchmarks), only the owner host parses the trace; non-owner hosts
    # skip local trace parsing and use broadcast metrics.
    mock_parse_durations.assert_not_called()
    self.assertAlmostEqual(result.metrics["xprof_avg_ms"], 4.2, places=4)
    self.assertAlmostEqual(result.metrics["xprof_p50_ms"], 4.1, places=4)

  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.parse_xprof_durations"
  )
  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  @unittest.mock.patch("jax.profiler.trace")
  @unittest.mock.patch("jax.process_index")
  def test_apply_xprof_timing_and_sync_without_multihost_sync(
      self,
      mock_process_index,
      mock_trace,
      mock_upload,
      mock_parse_durations,
  ):
    """Tests requires_multihost_sync=False (e.g., Collectives): each host parses its trace locally."""

    class MultihostBenchmark(DummyBenchmark):

      @property
      def requires_multihost_sync(self) -> bool:
        # Each host parses its local trace without cross-host sync
        return False

    mock_process_index.return_value = 1  # Current host is Host 1
    mock_trace.return_value = contextlib.nullcontext()
    mock_upload.return_value = "http://mock_xprof_url"
    mock_parse_durations.return_value = [3.5, 3.5, 3.5]

    params = {
        "warmup_tries": 1,
        "num_runs": 3,
        "xprof_timing": True,
        "xprof_dir": "/tmp/test_xprof",
    }
    config = base.BaseBenchmarkParams(**params)
    bm = MultihostBenchmark(config=config)
    result = bm.run()

    # When requires_multihost_sync=False (e.g., Collective symmetric benchmarks),
    # even though hosts exchange data during execution, each host parses its own local
    # XProf trace for its local device without invoking cross-host metric broadcast.
    mock_parse_durations.assert_called_once()
    self.assertEqual(result.metrics["xprof_avg_ms"], 3.5)


if __name__ == "__main__":
  absltest.main()
