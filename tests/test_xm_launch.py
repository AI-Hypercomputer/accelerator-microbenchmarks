"""Unit tests for MLCOMPASS_SELECT_TESTS parsing in xm_launch.py."""

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.orchestration import xm_launch


class TestXmLaunchSelectTests(parameterized.TestCase):
  """Tests for MLCOMPASS_SELECT_TESTS parsing and benchmark selection."""

  @parameterized.parameters(
      (None, None),
      ("", None),
      ("   ", None),
      ("{MLCOMPASS_SELECT_TESTS}", None),
  )
  def test_none_or_template_placeholder(self, select_tests_str, expected):
    self.assertEqual(
        xm_launch._parse_selected_benchmarks(select_tests_str), expected
    )

  @parameterized.parameters(
      (
          "//third_party/py/accelerator_microbenchmarks/src/accelerator_microbenchmarks:main/gfl_2x2x2/AllGatherBenchmark",
          {"all_gather"},
      ),
      (
          "//third_party/py/accelerator_microbenchmarks/src/accelerator_microbenchmarks:main/gfl_2x2x1/GeneralizedGemmBenchmark",
          {"gemm", "gemm_generalized"},
      ),
      (
          "//third_party/py/accelerator_microbenchmarks/src/accelerator_microbenchmarks:main/gfl_2x2x1/HBMBandwidthBenchmark",
          {"hbm", "hbm_bandwidth"},
      ),
      (
          "//third_party/py/accelerator_microbenchmarks/src/accelerator_microbenchmarks:main/gfl_2x2x2/AllReduceBenchmark",
          {"all_reduce"},
      ),
      (
          "//third_party/py/accelerator_microbenchmarks/src/accelerator_microbenchmarks:main/gfl_2x2x2/AllToAllBenchmark",
          {"all_to_all"},
      ),
      (
          "//third_party/py/accelerator_microbenchmarks/src/accelerator_microbenchmarks:main/gfl_2x2x2/ReduceScatterBenchmark",
          {"reduce_scatter"},
      ),
  )
  def test_single_full_target(self, select_tests_str, expected):
    self.assertEqual(
        xm_launch._parse_selected_benchmarks(select_tests_str), expected
    )

  @parameterized.parameters(
      (
          "tpums_gfl_2x2x2_AllGatherBenchmark",
          {"all_gather"},
      ),
      (
          "tpums_gfl_2x2x1_GeneralizedGemmBenchmark",
          {"gemm", "gemm_generalized"},
      ),
      (
          "tpums_gfl_2x2x1_HBMBandwidthBenchmark",
          {"hbm", "hbm_bandwidth"},
      ),
      (
          "tpums_gfl_2x2x2_AllReduceBenchmark",
          {"all_reduce"},
      ),
      (
          "tpums_gfl_2x2x2_AllToAllBenchmark",
          {"all_to_all"},
      ),
      (
          "tpums_gfl_2x2x2_ReduceScatterBenchmark",
          {"reduce_scatter"},
      ),
  )
  def test_shortened_target(self, select_tests_str, expected):
    self.assertEqual(
        xm_launch._parse_selected_benchmarks(select_tests_str), expected
    )

  def test_multiple_comma_separated_targets(self):
    select_tests_str = (
        "//third_party/py/accelerator_microbenchmarks/src/accelerator_microbenchmarks:main/gfl_2x2x2/AllGatherBenchmark,"
        "//third_party/py/accelerator_microbenchmarks/src/accelerator_microbenchmarks:main/gfl_2x2x2/AllReduceBenchmark"
    )
    self.assertEqual(
        xm_launch._parse_selected_benchmarks(select_tests_str),
        {"all_gather", "all_reduce"},
    )

  def test_multiple_shortened_targets(self):
    select_tests_str = (
        "tpums_gfl_2x2x2_AllGatherBenchmark,tpums_gfl_2x2x2_AllReduceBenchmark"
    )
    self.assertEqual(
        xm_launch._parse_selected_benchmarks(select_tests_str),
        {"all_gather", "all_reduce"},
    )

  @parameterized.parameters(
      ("AllGatherBenchmark", {"all_gather"}),
      ("GeneralizedGemmBenchmark", {"gemm", "gemm_generalized"}),
      ("HBMBandwidthBenchmark", {"hbm", "hbm_bandwidth"}),
      ("AllReduceBenchmark", {"all_reduce"}),
      ("AllToAllBenchmark", {"all_to_all"}),
      ("ReduceScatterBenchmark", {"reduce_scatter"}),
  )
  def test_direct_class_name(self, select_tests_str, expected):
    self.assertEqual(
        xm_launch._parse_selected_benchmarks(select_tests_str), expected
    )

  @parameterized.parameters(
      ("all_gather", {"all_gather"}),
      ("gemm_generalized", {"gemm", "gemm_generalized"}),
      ("gemm", {"gemm", "gemm_generalized"}),
      ("hbm", {"hbm", "hbm_bandwidth"}),
      ("hbm_bandwidth", {"hbm", "hbm_bandwidth"}),
      ("all_reduce", {"all_reduce"}),
      ("all_to_all", {"all_to_all"}),
      ("reduce_scatter", {"reduce_scatter"}),
  )
  def test_direct_op_or_alias_name(self, select_tests_str, expected):
    self.assertEqual(
        xm_launch._parse_selected_benchmarks(select_tests_str), expected
    )

  @parameterized.parameters(
      "//third_party/py/accelerator_microbenchmarks/src/accelerator_microbenchmarks:main/gfl_2x2x2/FakeBenchmark",
      "non_existent_op",
      "AllGatherBenchmark,InvalidBenchmark",
  )
  def test_unrecognized_benchmark_raises_value_error(self, select_tests_str):
    with self.assertRaises(ValueError) as ctx:
      xm_launch._parse_selected_benchmarks(select_tests_str)
    self.assertIn("Unrecognized benchmark identifier(s)", str(ctx.exception))


if __name__ == "__main__":
  absltest.main()
