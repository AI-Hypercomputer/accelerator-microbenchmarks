"""Unit tests for roofline.py."""

import dataclasses
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import roofline


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

  def test_apply_roofline_analysis_scalar_bw(self):
    """Test roofline analysis with scalar bandwidth."""
    metrics = {"tflops_per_sec": 50.0}
    params = {
        "hardware_stats": {"tflops": {"bfloat16": 100.0}, "hbm_bw": 200.0},
        "dtype": "bfloat16",
    }

    # setup config instead of passing params
    self.mock_benchmark.config = MockConfig(**params)
    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)

    # intensity = 1.0, bw = 200.0
    # roofline = min(100.0, 1.0 * 200.0 / 1000.0) = min(100.0, 0.2) = 0.2
    self.assertAlmostEqual(result["roofline_tflops_limit"], 0.2)
    self.assertEqual(result["peak_bw_at_size_gb_s"], 200.0)
    # efficiency = (50.0 / 0.2) * 100 = 25000%
    self.assertAlmostEqual(result["roofline_efficiency"], 25000.0)

  def test_apply_roofline_analysis_list_bw_interpolation(self):
    """Test roofline analysis with list-based bandwidth interpolation."""
    self.mock_benchmark.get_total_bytes.return_value = 300.0
    metrics = {}
    params = {
        "hardware_stats": {
            "tflops": {"bfloat16": 100.0},
            "hbm_bw": [(100, 50.0), (500, 250.0)],
        },
        "dtype": "bfloat16",
    }

    self.mock_benchmark.config = MockConfig(**params)
    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)

    # total_bytes = 300, between 100 and 500
    # bw = 50 + (250 - 50) * (300 - 100) / (500 - 100)
    #    = 50 + 200 * 200 / 400 = 50 + 100 = 150
    self.assertAlmostEqual(result["peak_bw_at_size_gb_s"], 150.0)

  def test_apply_roofline_analysis_dict_bw_interpolation(self):
    """Test roofline analysis with dict-based bandwidth interpolation."""
    self.mock_benchmark.get_total_bytes.return_value = 300.0
    metrics = {}
    params = {
        "hardware_stats": {
            "tflops": {"bfloat16": 100.0},
            "hbm_bw": {"100": 50.0, "500": 250.0},
        },
        "dtype": "bfloat16",
    }

    self.mock_benchmark.config = MockConfig(**params)
    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)

    self.assertAlmostEqual(result["peak_bw_at_size_gb_s"], 150.0)

  def test_apply_roofline_analysis_use_trace(self):
    """Test roofline analysis when using trace metrics."""
    self.mock_benchmark.get_trace_metrics.return_value = {
        "flops": 2000,
        "hbm_bytes": 1000,
    }
    metrics = {}
    params = {"use_trace_roofline": True}
    self.mock_benchmark.config = MockConfig(**params)
    result = roofline.apply_roofline_analysis(self.mock_benchmark, metrics)

    self.assertEqual(result["trace_flops"], 2000)
    self.assertEqual(result["trace_hbm_bytes"], 1000)
    self.assertAlmostEqual(result["intensity"], 2.0)


if __name__ == "__main__":
  absltest.main()
