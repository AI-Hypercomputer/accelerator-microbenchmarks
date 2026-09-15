"""Unit tests for roofline.py."""

import dataclasses
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import roofline
from accelerator_microbenchmarks.core import system


@dataclasses.dataclass
class MockConfig(base.BaseBenchmarkParams):
  dtype: str = "bfloat16"


class RooflineTest(absltest.TestCase):
  """Unit tests for roofline.py."""

  def setUp(self):
    super().setUp()
    self.mock_benchmark = mock.MagicMock()
    self.mock_benchmark.get_arithmetic_intensity.return_value = 1.0
    self.mock_benchmark.get_total_bytes.return_value = 1000.0
    self.mock_benchmark.get_compute_dtype.return_value = "bfloat16"
    self.mock_benchmark.config = MockConfig()
    self.mock_benchmark.hardware_spec = system.TPU7X_HARDWARE_SPEC

  def test_apply_roofline_analysis_none_mode(self):
    """Test that RooflineMode.NONE returns metrics unchanged without roofline keys."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.NONE
    metrics = {"tflops_per_device": 50.0, "bandwidth_per_device_gb_s": 80.0}
    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)
    self.assertEqual(
        result,
        {"tflops_per_device": 50.0, "bandwidth_per_device_gb_s": 80.0},
    )
    self.assertNotIn("roofline_tflops_limit", result)
    self.assertNotIn("compute_roofline_efficiency_pct", result)
    self.assertNotIn("peak_hbm_bw_gb_s", result)
    self.assertNotIn("memory_roofline_efficiency_pct", result)

  def test_apply_roofline_analysis_compute_mode(self):
    """Test that RooflineMode.COMPUTE computes tflops roofline and omits memory efficiency."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    metrics = {"tflops_per_device": 50.0, "bandwidth_per_device_gb_s": 80.0}
    mock_hw = mock.MagicMock(spec=system.HardwareSpec)
    mock_hw.name = system.TpuVersion.TPU7X
    mock_hw.tflops = system.TflopsSpec(
        peak_tflops_per_device={"bfloat16": 100.0}
    )
    mock_hw.hbm = system.HbmSpec(peak_bw_gbps=200.0)
    mock_hw.peak_hbm_bandwidth_per_device = 200.0
    self.mock_benchmark.hardware_spec = mock_hw

    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)
    self.assertAlmostEqual(result["roofline_tflops_limit"], 0.2)
    self.assertAlmostEqual(result["compute_roofline_efficiency_pct"], 25000.0)
    self.assertEqual(result["peak_hbm_bw_gb_s"], 200.0)
    self.assertNotIn("memory_roofline_efficiency_pct", result)

  def test_apply_roofline_analysis_memory_hbm_mode(self):
    """Test that RooflineMode.MEMORY_HBM computes HBM bw efficiency and omits compute efficiency."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.MEMORY_HBM
    metrics = {"tflops_per_device": 50.0, "bandwidth_per_device_gb_s": 80.0}
    mock_hw = mock.MagicMock(spec=system.HardwareSpec)
    mock_hw.name = system.TpuVersion.TPU7X
    mock_hw.tflops = None
    mock_hw.hbm = system.HbmSpec(peak_bw_gbps=200.0)
    mock_hw.peak_hbm_bandwidth_per_device = 200.0
    self.mock_benchmark.hardware_spec = mock_hw
    self.mock_benchmark.get_compute_dtype.side_effect = AssertionError(
        "get_compute_dtype should not be called in MEMORY_HBM mode"
    )

    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)
    self.assertEqual(result["peak_hbm_bw_gb_s"], 200.0)
    self.assertAlmostEqual(result["memory_roofline_efficiency_pct"], 40.0)
    self.assertNotIn("roofline_tflops_limit", result)
    self.assertNotIn("compute_roofline_efficiency_pct", result)

  def test_apply_roofline_analysis_zero_tflops_yields_zero_efficiency(self):
    """Test that actual_tflops=0.0 yields compute_roofline_efficiency_pct=0.0."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    metrics = {"tflops_per_device": 0.0}
    mock_hw = mock.MagicMock(spec=system.HardwareSpec)
    mock_hw.name = system.TpuVersion.TPU7X
    mock_hw.tflops = system.TflopsSpec(
        peak_tflops_per_device={"bfloat16": 100.0}
    )
    mock_hw.hbm = system.HbmSpec(peak_bw_gbps=200.0)
    mock_hw.peak_hbm_bandwidth_per_device = 200.0
    self.mock_benchmark.hardware_spec = mock_hw

    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)
    self.assertIn("compute_roofline_efficiency_pct", result)
    self.assertEqual(result["compute_roofline_efficiency_pct"], 0.0)

  def test_apply_roofline_analysis_flat_peak_bandwidth_invariant_to_size(self):
    """Test roofline analysis yields constant flat peak bandwidth across transfer sizes."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.MEMORY_HBM
    self.mock_benchmark.hardware_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        devices_per_chip=1,
        hbm=system.HbmSpec(peak_bw_gbps=500.0),
    )
    for size in (1, 1024, 1024 * 1024, 1024 * 1024 * 1024):
      self.mock_benchmark.get_total_bytes.return_value = float(size)
      metrics = {"bandwidth_per_device_gb_s": 250.0}
      result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)
      self.assertEqual(result["peak_hbm_bw_gb_s"], 500.0)
      self.assertAlmostEqual(result["memory_roofline_efficiency_pct"], 50.0)

  def test_apply_roofline_analysis_chip_tflops_no_roofline_efficiency(self):
    """Test that chip-level tflops does not trigger device-level roofline efficiency."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    metrics = {"tflops_per_chip": 100.0}
    self.mock_benchmark.hardware_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        tflops=system.TflopsSpec(peak_tflops_per_device={"bfloat16": 100.0}),
        hbm=system.HbmSpec(peak_bw_gbps=200.0),
    )
    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)
    self.assertNotIn("compute_roofline_efficiency_pct", result)
    self.assertNotIn("roofline_efficiency", result)

  def test_apply_roofline_analysis_memory_roofline_efficiency_pct(self):
    """Test roofline analysis calculates bandwidth efficiency strictly at device level."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.MEMORY_HBM
    self.mock_benchmark.hardware_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        devices_per_chip=2,
        tflops=system.TflopsSpec(peak_tflops_per_device={"bfloat16": 100.0}),
        hbm=system.HbmSpec(peak_bw_gbps=200.0),
    )
    # Per-device peak HBM is 200.0 / 2 = 100.0 GB/s.
    # 1. Using bandwidth_per_device_gb_s: 80.0 / 100.0 = 80.0%
    metrics_dev = {"bandwidth_per_device_gb_s": 80.0}
    res_dev = roofline.apply_roofline_analysis(
        self.mock_benchmark, metrics_dev
    )
    self.assertAlmostEqual(res_dev["memory_roofline_efficiency_pct"], 80.0)

    # 2. Chip-only metric (e.g. from collectives) does NOT compute HBM
    # memory_roofline_efficiency_pct.
    metrics_chip = {"bandwidth_per_chip_gb_s": 100.0}
    res_chip = roofline.apply_roofline_analysis(
        self.mock_benchmark, metrics_chip
    )
    self.assertNotIn("memory_roofline_efficiency_pct", res_chip)

  def test_apply_roofline_analysis_zero_peak_bw(self):
    """Test roofline analysis when peak HBM bandwidth is zero."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.MEMORY_HBM
    self.mock_benchmark.hardware_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        devices_per_chip=1,
        hbm=system.HbmSpec(peak_bw_gbps=0.0),
    )
    metrics = {"bandwidth_per_device_gb_s": 50.0}
    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)
    self.assertEqual(result["peak_hbm_bw_gb_s"], 0.0)
    self.assertNotIn("memory_roofline_efficiency_pct", result)

  def test_apply_roofline_analysis_hbm_none_leaves_metrics_unchanged(self):
    """Verify apply_roofline_analysis gracefully skips when HBM spec is None."""
    hw_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        devices_per_chip=2,
        tflops=system.TflopsSpec(peak_tflops_per_device={"bfloat16": 100.0}),
        hbm=None,
    )
    self.mock_benchmark.hardware_spec = hw_spec

    # In MEMORY_HBM mode
    self.mock_benchmark.roofline_mode = constants.RooflineMode.MEMORY_HBM
    res_mem = roofline.apply_roofline_analysis(
        self.mock_benchmark, {"bandwidth_per_device_gb_s": 50.0}
    )
    self.assertNotIn("peak_hbm_bw_gb_s", res_mem)
    self.assertNotIn("memory_roofline_efficiency_pct", res_mem)

    # In COMPUTE mode
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    res_comp = roofline.apply_roofline_analysis(
        self.mock_benchmark, {"tflops_per_device": 50.0}
    )
    self.assertNotIn("roofline_tflops_limit", res_comp)
    self.assertNotIn("compute_roofline_efficiency_pct", res_comp)

  def test_apply_roofline_analysis_use_trace(self):
    """Test roofline analysis when using trace metrics."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    self.mock_benchmark.get_trace_metrics.return_value = {
        "flops": 2000,
        "hbm_bytes": 1000,
    }
    metrics = {}
    self.mock_benchmark.config = MockConfig(use_trace_roofline=True)
    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)

    self.assertEqual(result["trace_flops"], 2000)
    self.assertEqual(result["trace_hbm_bytes"], 1000)
    self.assertAlmostEqual(result["intensity"], 2.0)

  def test_apply_roofline_analysis_missing_dtype_warns_and_falls_back(self):
    """Test roofline analysis logs warning when compute dtype peak tflops is missing."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    self.mock_benchmark.get_compute_dtype.return_value = "float32"
    self.mock_benchmark.hardware_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        tflops=system.TflopsSpec(peak_tflops_per_device={"bfloat16": 100.0}),
        hbm=system.HbmSpec(peak_bw_gbps=200.0),
    )
    with self.assertLogs(level="WARNING") as log_cm:
      result = roofline.apply_roofline_analysis(self.mock_benchmark, {})
      self.assertAlmostEqual(result["roofline_tflops_limit"], 0.2)
      self.assertTrue(
          any(
              "Peak TFLOPS setup is missing for dtype 'float32'" in log
              for log in log_cm.output
          )
      )

  def test_apply_roofline_analysis_missing_fallback_dtype_raises_key_error(
      self,
  ):
    """Test roofline analysis raises KeyError when fallback dtype is missing from HardwareSpec."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    self.mock_benchmark.get_compute_dtype.return_value = "float32"
    self.mock_benchmark.hardware_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        tflops=system.TflopsSpec(peak_tflops_per_device={"int8": 100.0}),
        hbm=system.HbmSpec(peak_bw_gbps=200.0),
    )
    with self.assertRaisesRegex(
        KeyError, "missing peak TFLOPS for canonical fallback dtype"
    ):
      roofline.apply_roofline_analysis(self.mock_benchmark, {})

  def test_apply_roofline_analysis_tpu7x_flat_peak_compute_mode(self):
    """Verify live TPU7X_HARDWARE_SPEC in COMPUTE mode for memory and compute bound shapes."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    self.mock_benchmark.hardware_spec = system.TPU7X_HARDWARE_SPEC
    self.mock_benchmark.get_compute_dtype.return_value = "bfloat16"

    # Memory-bound shape (intensity=1.0, bfloat16, tflops_per_device=3.0)
    self.mock_benchmark.get_arithmetic_intensity.return_value = 1.0
    metrics_mem = {"tflops_per_device": 3.0}
    res_mem = roofline.apply_roofline_analysis(self.mock_benchmark, metrics_mem)
    self.assertEqual(res_mem["peak_hbm_bw_gb_s"], 3690.0)
    self.assertAlmostEqual(res_mem["roofline_tflops_limit"], 3.69)
    self.assertAlmostEqual(
        res_mem["compute_roofline_efficiency_pct"], (3.0 / 3.69) * 100.0
    )

    # Compute-bound shape (intensity=400.0, bfloat16, tflops_per_device=1000.0)
    self.mock_benchmark.get_arithmetic_intensity.return_value = 400.0
    metrics_compute = {"tflops_per_device": 1000.0}
    res_compute = roofline.apply_roofline_analysis(
        self.mock_benchmark, metrics_compute
    )
    self.assertAlmostEqual(res_compute["roofline_tflops_limit"], 1153.5)
    self.assertAlmostEqual(
        res_compute["compute_roofline_efficiency_pct"],
        (1000.0 / 1153.5) * 100.0,
    )

  def test_apply_roofline_analysis_tpu7x_flat_peak_memory_mode(self):
    """Verify live TPU7X_HARDWARE_SPEC in MEMORY_HBM mode for small and large transfers."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.MEMORY_HBM
    self.mock_benchmark.hardware_spec = system.TPU7X_HARDWARE_SPEC

    # 2 MiB transfer (bandwidth_per_device_gb_s=903.75)
    self.mock_benchmark.get_total_bytes.return_value = 2 * 1024 * 1024
    metrics_2mib = {"bandwidth_per_device_gb_s": 903.75}
    res_2mib = roofline.apply_roofline_analysis(
        self.mock_benchmark, metrics_2mib
    )
    self.assertEqual(res_2mib["peak_hbm_bw_gb_s"], 3690.0)
    self.assertAlmostEqual(
        res_2mib["memory_roofline_efficiency_pct"],
        (903.75 / 3690.0) * 100.0,
    )
    self.assertAlmostEqual(
        res_2mib["memory_roofline_efficiency_pct"], 24.49, places=2
    )

    # 2048 MiB transfer (bandwidth_per_device_gb_s=3604.0)
    self.mock_benchmark.get_total_bytes.return_value = 2048 * 1024 * 1024
    metrics_2048mib = {"bandwidth_per_device_gb_s": 3604.0}
    res_2048mib = roofline.apply_roofline_analysis(
        self.mock_benchmark, metrics_2048mib
    )
    self.assertEqual(res_2048mib["peak_hbm_bw_gb_s"], 3690.0)
    self.assertAlmostEqual(
        res_2048mib["memory_roofline_efficiency_pct"],
        (3604.0 / 3690.0) * 100.0,
    )
    self.assertAlmostEqual(
        res_2048mib["memory_roofline_efficiency_pct"], 97.67, places=2
    )
    self.assertLessEqual(res_2048mib["memory_roofline_efficiency_pct"], 100.0)

  def test_apply_roofline_analysis_none_hardware_spec(self):
    """Verify that None hardware_spec safely returns metrics unchanged in all modes."""
    self.mock_benchmark.hardware_spec = None
    metrics = {"tflops_per_device": 50.0, "bandwidth_per_device_gb_s": 80.0}

    # MEMORY_HBM mode with hw_spec=None
    self.mock_benchmark.roofline_mode = constants.RooflineMode.MEMORY_HBM
    res_mem = roofline.apply_roofline_analysis(
        self.mock_benchmark, metrics.copy()
    )
    self.assertEqual(res_mem, metrics)

    # COMPUTE mode with hw_spec=None
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    res_comp = roofline.apply_roofline_analysis(
        self.mock_benchmark, metrics.copy()
    )
    self.assertEqual(res_comp, metrics)

  def test_apply_roofline_analysis_fallback_dtype(self):
    """Verify fallback dtype lookup and warning when requested dtype is not in spec."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    self.mock_benchmark.get_compute_dtype.return_value = "custom_unknown_dtype"
    mock_hw = mock.MagicMock(spec=system.HardwareSpec)
    mock_hw.name = system.TpuVersion.TPU7X
    mock_hw.tflops = system.TflopsSpec(
        peak_tflops_per_device={system.DEFAULT_FALLBACK_DTYPE: 1000.0}
    )
    mock_hw.hbm = system.HbmSpec(peak_bw_gbps=200.0)
    mock_hw.peak_hbm_bandwidth_per_device = 200.0
    self.mock_benchmark.hardware_spec = mock_hw

    metrics = {"tflops_per_device": 100.0}
    with self.assertLogs(level="WARNING") as log_output:
      res = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)

    self.assertIn("roofline_tflops_limit", res)
    self.assertTrue(
        any("Falling back to canonical" in log for log in log_output.output)
    )

  def test_apply_roofline_analysis_missing_fallback_dtype_raises_keyerror(self):
    """Verify KeyError is raised when even canonical fallback dtype is missing."""
    self.mock_benchmark.roofline_mode = constants.RooflineMode.COMPUTE
    self.mock_benchmark.get_compute_dtype.return_value = "unknown_dtype"
    mock_hw = mock.MagicMock(spec=system.HardwareSpec)
    mock_hw.name = system.TpuVersion.TPU7X
    mock_hw.tflops = system.TflopsSpec(
        peak_tflops_per_device={"other_dtype": 500.0}
    )
    mock_hw.hbm = system.HbmSpec(peak_bw_gbps=200.0)
    mock_hw.peak_hbm_bandwidth_per_device = 200.0
    self.mock_benchmark.hardware_spec = mock_hw

    metrics = {"tflops_per_device": 100.0}
    with self.assertRaises(KeyError):
      roofline.apply_roofline_analysis(self.mock_benchmark, metrics)


if __name__ == "__main__":
  absltest.main()
