"""Unit tests for device_to_device.py."""

import argparse
import io
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks import cli
from accelerator_microbenchmarks.benchmarks import device_to_device
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.core import units
from accelerator_microbenchmarks.tests import test_report_utils
import jax
import jax.numpy as jnp


def setUpModule():  # pylint: disable=invalid-name
  # Force 4 CPU devices for testing multi-device D2D transfers
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=4"
  )
  # Set CPU backend for fast testing without hardware requirements
  jax.config.update("jax_platform_name", "cpu")


class DeviceToDeviceParamsTest(parameterized.TestCase):
  """Unit tests for DeviceToDeviceParams parameter expansion and properties."""

  def test_expand_test_cases_pair_generation(self):
    """Verifies N * (N - 1) sweep pairs where src != dst."""
    cfg = device_to_device.DeviceToDeviceParams(
        data_size="1MiB", direction="uni"
    )
    cases = cfg.expand_test_cases()
    num_devices = len(jax.devices())
    expected_pairs = num_devices * (num_devices - 1)
    if num_devices <= 1:
      self.assertEmpty(cases)
    else:
      self.assertLen(cases, expected_pairs)
      for c in cases:
        self.assertIsInstance(c, device_to_device.DeviceToDeviceTestCaseParams)
        self.assertNotEqual(c.src_device_index, c.dst_device_index)

  def test_expand_test_cases_attribute_propagation(self):
    """Verifies config flags propagate correctly into generated test cases."""
    cfg = device_to_device.DeviceToDeviceParams(
        data_size="16MiB",
        direction="bi",
        seed=42,
        warmup_tries=3,
        num_runs=7,
    )
    cases = cfg.expand_test_cases()
    num_devices = len(jax.devices())
    if num_devices > 1:
      first_case = cases[0]
      self.assertEqual(first_case.data_size, "16MiB")
      self.assertIsInstance(first_case.data_size, units.ByteSize)
      self.assertEqual(first_case.direction, "bi")
      self.assertEqual(first_case.seed, 42)
      self.assertEqual(first_case.warmup_tries, 3)
      self.assertEqual(first_case.num_runs, 7)

  @parameterized.parameters(
      ("4B", 4),
      ("4KiB", 4096),
      ("4MiB", 4 * 1024 * 1024),
      ("1024KiB", 1024 * 1024),
  )
  def test_data_size_bytes_property(self, data_size, expected_bytes):
    """Verifies data_size labels convert to bytes and are kept as written."""
    cfg = device_to_device.DeviceToDeviceParams(
        data_size=data_size, dtype="float32"
    )
    self.assertEqual(cfg.data_size, data_size)
    self.assertEqual(cfg.data_size_bytes, expected_bytes)

  def test_default_data_size(self):
    """Verifies the default payload is 1GiB."""
    cfg = device_to_device.DeviceToDeviceParams()
    self.assertEqual(cfg.data_size, "1GiB")
    self.assertEqual(cfg.data_size_bytes, 1 << 30)

  @parameterized.parameters(
      ("2B", "float32"),
      ("6B", "float32"),
      ("1B", "bfloat16"),
      ("0B", "int8"),
  )
  def test_partial_element_data_size_raises_error(self, data_size, dtype):
    """Verifies data_size must be a positive whole number of elements."""
    with self.assertRaisesRegex(
        ValueError, f"whole number of {dtype} elements"
    ):
      device_to_device.DeviceToDeviceParams(data_size=data_size, dtype=dtype)

  @parameterized.parameters(
      (4096, "string with a unit"),
      ("4096", "has no unit"),
      ("4K", "Did you mean '4KiB'"),
      ("1.5KiB", "whole number"),
  )
  def test_invalid_data_size_raises_error(self, data_size, message):
    """Verifies invalid data_size values fail with the ByteSize hint."""
    with self.assertRaisesRegex(ValueError, f"data_size: .*{message}"):
      device_to_device.DeviceToDeviceParams(data_size=data_size)

  def test_data_size_survives_replace_and_expand(self):
    """Verifies the label is unchanged through dataclasses round trips."""
    cfg = device_to_device.DeviceToDeviceParams(
        data_size="4KiB", dtype="float32"
    ).with_generation_defaults({"num_runs": 3})
    self.assertEqual(cfg.data_size, "4KiB")
    for case in cfg.expand_test_cases():
      self.assertEqual(case.data_size, "4KiB")
      self.assertEqual(case.data_size_bytes, 4096)

  @parameterized.named_parameters(
      (d.name.lower(), d.value) for d in device_to_device.TransferDirection
  )
  def test_direction_help_message_and_default_value(self, choice: str):
    """Verifies help message includes choice and matching default."""
    parser = argparse.ArgumentParser()
    cli.add_dataclass_arguments(
        parser, device_to_device.DeviceToDeviceParams
    )
    s = io.StringIO()
    parser.print_help(file=s)
    help_msg = s.getvalue()

    normalized_help = " ".join(help_msg.split())
    self.assertIn(
        choice,
        help_msg,
        f"Enum value '{choice}' missing from help message: {help_msg}",
    )

    default_direction = (
        device_to_device.DeviceToDeviceParams().direction.value
    )
    self.assertIn(
        f"(default: {default_direction})",
        normalized_help,
        f"Displayed default '{default_direction}' missing from help:"
        f" {help_msg}",
    )

  @parameterized.parameters(
      ("uni", device_to_device.TransferDirection.UNI),
      ("bi", device_to_device.TransferDirection.BI),
  )
  def test_direction_is_normalized_to_enum(self, value, expected):
    """Verifies raw direction strings are coerced to TransferDirection."""
    cfg = device_to_device.DeviceToDeviceParams(direction=value)
    self.assertEqual(cfg.direction, expected)

  @parameterized.parameters("TESTING", "BI")
  def test_unknown_direction_raises_error(self, value):
    """Verifies unknown directions are rejected, including wrong casing."""
    with self.assertRaisesRegex(ValueError, f"Invalid direction '{value}'"):
      device_to_device.DeviceToDeviceParams(direction=value)

  @parameterized.parameters(
      (device_to_device.DeviceToDeviceParams, "seed", -1, 0),
      (
          device_to_device.DeviceToDeviceTestCaseParams,
          "src_device_index",
          -1,
          0,
      ),
      (
          device_to_device.DeviceToDeviceTestCaseParams,
          "dst_device_index",
          -1,
          0,
      ),
  )
  def test_out_of_range_values_raise_error(
      self, params_cls, field, value, bound
  ):
    """Verifies the bounds declared on the device_to_device params fields."""
    with self.assertRaisesRegex(ValueError, f"{field} must be >= {bound}"):
      params_cls(**{field: value})

  @parameterized.product(
      params_cls=(
          device_to_device.DeviceToDeviceParams,
          device_to_device.DeviceToDeviceTestCaseParams,
      ),
      dtype=("int4", "float4_e2m1fn"),
  )
  def test_sub_byte_dtype_raises_error(self, params_cls, dtype):
    """Verifies sub-byte dtypes are rejected (the payload counts itemsize)."""
    with self.assertRaisesRegex(
        ValueError, f"^dtype: Unsupported sub-byte dtype '{dtype}'"
    ):
      params_cls(dtype=dtype)

  @parameterized.parameters("int8", "uint8", "bool")
  def test_byte_sized_non_float_dtypes_accepted(self, dtype):
    """Verifies 1-byte integer and bool payloads are accepted."""
    cfg = device_to_device.DeviceToDeviceParams(dtype=dtype)
    self.assertEqual(cfg.dtype, dtype)


class DeviceToDeviceBenchmarkTest(parameterized.TestCase):
  """Unit tests for DeviceToDeviceBenchmark."""

  def setUp(self):
    super().setUp()
    self.patcher = mock.patch(
        "accelerator_microbenchmarks.core.platform.get_platform_info",
        return_value=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
    )
    self.patcher.start()
    self.addCleanup(self.patcher.stop)
    self.bm_class = registry.benchmark_registry.get_benchmark(
        "device_to_device"
    )
    self.bm = self.bm_class(
        device_to_device.DeviceToDeviceTestCaseParams(
            data_size="1MiB",
            direction="uni",
        ),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )

  def test_benchmark_registered(self):
    """Verify registration."""
    self.assertEqual(self.bm_class, device_to_device.DeviceToDeviceBenchmark)

  def test_get_total_bytes(self):
    """Verify byte calculation."""
    self.bm.config.data_size = units.ByteSize("1MiB")
    self.bm.config.direction = "uni"
    expected_bytes_uni = 1.0 * 1024 * 1024
    self.assertEqual(
        self.bm.get_total_bytes(),
        expected_bytes_uni,
    )
    self.bm.config.direction = "bi"
    expected_bytes_bi = 2.0 * 1024 * 1024
    self.assertEqual(
        self.bm.get_total_bytes(),
        expected_bytes_bi,
    )

  def test_calculate_metrics(self):
    """Verify bandwidth metrics calculation."""
    times_ms = [10.0, 10.0]
    self.bm.config.data_size = units.ByteSize("1MiB")
    self.bm.config.direction = "uni"
    self.bm.config.src_device_index = 0
    self.bm.config.dst_device_index = 1
    metrics = self.bm.calculate_metrics(times_ms)
    self.assertAlmostEqual(metrics["wall_clock_avg_ms"], 10.0)
    # total_bytes = 1048576 bytes
    # avg_latency_s = 0.01 s
    # wall_clock_bandwidth_per_device_gb_s = 1048576 / (0.01 * 1e9) = 0.1048576 GB/s
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", metrics)
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], 0.1048576
    )
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", metrics)
    self.assertNotIn("src_device_index", metrics)
    self.assertNotIn("dst_device_index", metrics)
    self.assertNotIn("direction", metrics)
    self.assertNotIn("total_bytes_mib", metrics)

  def test_derive_chip_metrics_no_chip_bandwidth(self):
    """Verify derive_chip_metrics does not derive wall_clock_bandwidth_per_chip_gb_s."""
    metrics = {"wall_clock_bandwidth_per_device_gb_s": 10.0}
    derived = self.bm.derive_chip_metrics(metrics)
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", derived)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", derived)

  def test_run_e2e(self):
    """Verify run() produces valid wall_clock_bandwidth_per_device_gb_s without per_chip bandwidth."""
    self.bm.config.src_device_index = 0
    self.bm.config.dst_device_index = 1
    self.bm.config.data_size = units.ByteSize("1MiB")
    self.bm.config.warmup_tries = 1
    self.bm.config.num_runs = 2
    result = self.bm.run()
    self.assertIn("wall_clock_bandwidth_per_device_gb_s", result.metrics)
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", result.metrics)
    self.assertGreater(
        result.metrics["wall_clock_bandwidth_per_device_gb_s"], 0.0
    )
    self.assertEqual(result.metadata.params["src_device_index"], 0)
    self.assertEqual(result.metadata.params["dst_device_index"], 1)
    self.assertEqual(result.metadata.params["direction"], "uni")
    self.assertNotIn("src_device_index", result.metrics)
    self.assertNotIn("dst_device_index", result.metrics)
    self.assertNotIn("direction", result.metrics)
    self.assertNotIn("roofline_tflops_limit", result.metrics)
    self.assertNotIn("compute_roofline_efficiency_pct", result.metrics)
    self.assertNotIn("peak_hbm_bw_gb_s", result.metrics)
    self.assertNotIn("memory_roofline_efficiency_pct", result.metrics)

  def test_get_arithmetic_intensity(self):
    """Verify DeviceToDeviceBenchmark does not implement get_arithmetic_intensity."""
    with self.assertRaises(NotImplementedError):
      self.bm.get_arithmetic_intensity()

  def test_calculate_metrics_zero_latency(self):
    """Verify bandwidth calculation handles zero latency cleanly (inf)."""
    metrics = self.bm.calculate_metrics([0.0, 0.0])
    self.assertEqual(
        metrics["wall_clock_bandwidth_per_device_gb_s"], float("inf")
    )

  def test_get_run_identifier(self):
    """Verify run identifier string generation."""
    self.bm.config.src_device_index = 0
    self.bm.config.dst_device_index = 1
    self.bm.config.direction = device_to_device.TransferDirection.UNI
    self.bm.config.data_size = units.ByteSize("1MiB")
    self.assertEqual(self.bm.get_run_identifier(), "d2d_0_to_1_uni_1MiB")
    self.bm.config.direction = device_to_device.TransferDirection.BI
    self.assertEqual(self.bm.get_run_identifier(), "d2d_0_to_1_bi_1MiB")
    self.bm.config.data_size = units.ByteSize("4KiB")
    self.assertEqual(self.bm.get_run_identifier(), "d2d_0_to_1_bi_4KiB")

  def test_workload_metadata_reports_data_size_bytes(self):
    """Verifies data_size_bytes is emitted next to the data_size param."""
    self.bm.config.data_size = units.ByteSize("4KiB")
    self.assertEqual(
        self.bm.get_workload_metadata(), {"data_size_bytes": 4096}
    )

  def test_requires_multihost_sync(self):
    """Verify D2D requires multihost sync for asymmetric target device."""
    self.assertTrue(self.bm.requires_multihost_sync)

  def test_get_device_to_measure_uninitialized(self):
    """Verify get_device_to_measure before setup raises ValueError."""
    uninitialized_bm = self.bm_class(
        device_to_device.DeviceToDeviceTestCaseParams(),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )
    with self.assertRaisesRegex(ValueError, "Mesh not initialized."):
      uninitialized_bm.get_device_to_measure()

  def test_get_device_to_measure_initialized(self):
    """Verify get_device_to_measure resolves destination device after setup."""
    self.bm.config.src_device_index = 0
    self.bm.config.dst_device_index = 1
    self.bm.setup()
    dev = self.bm.get_device_to_measure()
    self.assertEqual(dev, self.bm.mesh.devices.flat[1])

  def test_get_device_to_measure_out_of_bounds(self):
    """Verify get_device_to_measure with invalid dst_device_index raises ValueError."""
    self.bm.setup()
    self.bm.config.dst_device_index = 9999
    with self.assertRaisesRegex(ValueError, "exceeds mesh size"):
      self.bm.get_device_to_measure()

  def test_run_op_uninitialized(self):
    """Verify run_op before setup raises ValueError."""
    uninitialized_bm = self.bm_class(
        device_to_device.DeviceToDeviceTestCaseParams(),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )
    with self.assertRaisesRegex(ValueError, "JIT function not initialized."):
      uninitialized_bm.run_op(None)

  def test_generate_inputs_uninitialized(self):
    """Verify generate_inputs before setup raises ValueError."""
    uninitialized_bm = self.bm_class(
        device_to_device.DeviceToDeviceTestCaseParams(),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )
    with self.assertRaisesRegex(ValueError, "Mesh not initialized."):
      uninitialized_bm.generate_inputs()

  # int8 / bool cover non-float input generation (jax.random.normal is
  # float-only).
  @parameterized.parameters("bfloat16", "int8", "bool")
  def test_generate_inputs_and_run_op(self, dtype):
    """Verify setup, input generation, and kernel execution."""
    bm = self.bm_class(
        device_to_device.DeviceToDeviceTestCaseParams(
            dtype=dtype,
            data_size="1MiB",
            direction="uni",
            src_device_index=0,
            dst_device_index=1,
        ),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )
    bm.setup()
    inputs = bm.generate_inputs()
    self.assertLen(inputs, 1)
    self.assertEqual(inputs[0].dtype, jnp.dtype(dtype))
    self.assertEqual(
        inputs[0].shape,
        (len(jax.devices()), 1024 * 1024 // jnp.dtype(dtype).itemsize),
    )
    out = bm.run_op(*inputs)
    self.assertEqual(out.shape, inputs[0].shape)
    self.assertEqual(out.dtype, inputs[0].dtype)

  def test_format_benchmark_table(self):
    """Tests formatting of Device-to-Device benchmark tables."""
    res = base.BenchmarkResult(
        metadata=base.BenchmarkMetadata(
            benchmark_name="DeviceToDeviceBenchmark",
            test_name="DeviceToDeviceBenchmark_test",
            start_time="2026-08-18T10:00:00",
            end_time="2026-08-18T10:01:00",
            params={
                "dtype": "float32",
                "direction": "uni",
                "src_device_index": 7783,
                "data_size": units.ByteSize("1GiB"),
                "dst_device_index": 153,
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "data_size_bytes": 1 << 30,
            "wall_clock_bandwidth_per_device_gb_s": 85.50,
            "xprof_bandwidth_per_device_gb_s": 86.00,
            "wall_clock_p50_ms": 12.3456,
            "xprof_p50_ms": 12.3400,
        },
        timings=base.BenchmarkTimings(
            wall_clock=base.WallClockTimings(durations_ms=[1.0])
        ),
    )
    expected_cols = [
        "dtype",
        "direction",
        "src_device_index",
        "dst_device_index",
        "data_size",
        "data_size_bytes",
        "wall_clock_p50_ms",
        "wall_clock_bandwidth_per_device_gb_s",
        "xprof_p50_ms",
        "xprof_bandwidth_per_device_gb_s",
    ]
    schema_cols = [
        col for col, _ in device_to_device.DeviceToDeviceBenchmark.REPORT_SCHEMA
    ]
    self.assertEqual(schema_cols, expected_cols)

    df = report.results_to_dataframe([res])
    table = report.format_benchmark_table(
        df,
        schema=device_to_device.DeviceToDeviceBenchmark.REPORT_SCHEMA,
        title="DeviceToDeviceBenchmark",
    )
    self.assertIn("Benchmark Results (DeviceToDeviceBenchmark)", table)
    for col in expected_cols:
      self.assertIn(col, table)
    self.assertIn("float32", table)
    self.assertIn("uni", table)
    self.assertIn("7783", table)
    self.assertIn("153", table)
    self.assertIn("1GiB", table)
    self.assertIn("1073741824", table)
    self.assertIn("85.50", table)
    self.assertIn("86.00", table)
    self.assertIn("12.3456", table)
    self.assertIn("12.3400", table)

  def test_schema_coverage(self):
    """Verify REPORT_SCHEMA matches output keys and covers all metrics."""
    self.bm.setup()
    test_report_utils.assert_schema_matches_output(self, self.bm)

  def test_report_formatters_declaration(self):
    """Verifies that DeviceToDeviceBenchmark declares both standard table and matrix formatters."""
    self.assertEqual(
        device_to_device.DeviceToDeviceBenchmark.REPORT_FORMATTERS,
        (
            report.format_standard_table,
            report.format_device_matrix,
        ),
    )

  def test_generate_benchmark_report_dual_representation(self):
    """Verifies that generate_benchmark_report produces both standard table and pairwise matrix."""
    res1 = base.BenchmarkResult(
        metadata=base.BenchmarkMetadata(
            benchmark_name="DeviceToDeviceBenchmark",
            test_name="DeviceToDeviceBenchmark_0_1",
            start_time="2026-08-18T10:00:00",
            end_time="2026-08-18T10:01:00",
            params={
                "dtype": "bfloat16",
                "direction": "uni",
                "src_device_index": 0,
                "dst_device_index": 1,
                "data_size": units.ByteSize("1GiB"),
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "data_size_bytes": 1 << 30,
            "wall_clock_bandwidth_per_device_gb_s": 85.50,
            "wall_clock_p50_ms": 12.3456,
            "xprof_p50_ms": 12.3400,
        },
        timings=base.BenchmarkTimings(
            wall_clock=base.WallClockTimings(durations_ms=[12.3456])
        ),
    )
    res2 = base.BenchmarkResult(
        metadata=base.BenchmarkMetadata(
            benchmark_name="DeviceToDeviceBenchmark",
            test_name="DeviceToDeviceBenchmark_1_0",
            start_time="2026-08-18T10:00:00",
            end_time="2026-08-18T10:01:00",
            params={
                "dtype": "bfloat16",
                "direction": "uni",
                "src_device_index": 1,
                "dst_device_index": 0,
                "data_size": units.ByteSize("1GiB"),
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "data_size_bytes": 1 << 30,
            "wall_clock_bandwidth_per_device_gb_s": 86.20,
            "wall_clock_p50_ms": 12.1000,
            "xprof_p50_ms": 12.0000,
        },
        timings=base.BenchmarkTimings(
            wall_clock=base.WallClockTimings(durations_ms=[12.1000])
        ),
    )
    df = report.results_to_dataframe([res1, res2])
    rep = report.generate_benchmark_report(df)

    # 1. Verify standard flat table is present
    self.assertIn("Benchmark Results (DeviceToDeviceBenchmark)", rep)
    self.assertIn("85.50", rep)
    self.assertIn("86.20", rep)

    # 2. Verify pairwise matrix is present
    self.assertIn(
        "Device-to-Device Bandwidth Matrix (Wall Clock GB/s)", rep
    )
    self.assertIn("D0", rep)
    self.assertIn("D1", rep)
    self.assertIn("X", rep)
    self.assertIn("data_size=1GiB", rep)


if __name__ == "__main__":
  absltest.main()
