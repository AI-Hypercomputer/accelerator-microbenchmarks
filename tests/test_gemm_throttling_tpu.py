"""Unit tests for GEMM throttling benchmark on physical TPU hardware."""

import json
import os
import tempfile


class GemmThrottlingTPUTest(absltest.TestCase):
  """Unit tests for GEMM throttling benchmark on physical TPU hardware."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("This test requires real TPU devices.")

  def tearDown(self):
    super().tearDown()

  def test_gemm_throttling_end_to_end_on_tpu(self):
    """Tests the full GemmThrottlingBenchmark execution on TPU."""
    config = matmul.GemmThrottlingParams(
        m=4096,
        k=4096,
        n=4096,
        in_dtype="bfloat16",
        out_dtype="bfloat16",
        min_duration_s=3.0,
        samples_per_run=5,
    )
    benchmark = matmul.GemmThrottlingBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        xprof_config=base.XprofConfig(xprof_timing=True),
    )
    result = benchmark.run()
    print("BORG_TPU7X_METRICS: " + json.dumps(result.metrics, indent=2))
    self.assertIsNotNone(result.metrics)

    # Verify core performance metrics
    required_keys = [
        "xprof_pre_soaking_p50_ms",
        "xprof_post_soaking_p50_ms",
        "xprof_pre_soaking_tflops_per_device",
        "xprof_post_soaking_tflops_per_device",
        "xprof_slowdown_ratio",
        "wall_clock_pre_soaking_p50_ms",
        "wall_clock_post_soaking_p50_ms",
    ]
    for key in required_keys:
      self.assertIn(key, result.metrics)
      self.assertGreater(result.metrics[key], 0.0)

    # Verify thermal and firmware telemetry
    thermal_keys = [
        "hbm_peak_temp_c",
        "vdd_core_power_mean_w",
        "hbm_power_mean_w",
        "total_power_mean_w",
        "pstate_min",
        "pstate_max",
    ]
    for key in thermal_keys:
      self.assertIn(key, result.metrics)
      self.assertIsNotNone(result.metrics[key])

    # Check report generation
    with tempfile.TemporaryDirectory() as tmpdir:
      report.save_output([result], tmpdir)
      summary_path = os.path.join(tmpdir, "summary.csv")
      detailed_path = os.path.join(tmpdir, "detailed.json")
      self.assertTrue(os.path.exists(summary_path))
      self.assertTrue(os.path.exists(detailed_path))

      with open(detailed_path, "r") as f:
        details_list = json.load(f)
      self.assertLen(details_list, 1)

      # Verify thermal_per_device matches device count
      thermal_per_device = benchmark.executor.get_result_details().get(
          "thermal_per_device", {}
      )
      self.assertLen(thermal_per_device, jax.local_device_count())

  def _run_pulse(self, **params) -> base.BenchmarkResult:
    benchmark = matmul.GemmPulseBenchmark(
        config=matmul.GemmPulseParams(
            m=4096, k=4096, chain_length=10, idle_s=0.02, **params
        ),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        xprof_config=base.XprofConfig(xprof_timing=True),
    )
    return benchmark.run()

  def test_gemm_pulse_e2e_on_tpu(self):
    """Tests multi-cycle GemmPulseBenchmark execution on TPU."""
    result = self._run_pulse(num_cycles=4)
    self.assertEqual(result.metrics["actual_runs"], 4)
    self.assertTrue(result.metrics["all_devices_valid_checksum"])
    self.assertTrue(result.metrics["intra_host_checksum_match"])
    self.assertTrue(result.metrics["global_checksum_match"])
    for key in (
        "xprof_pre_soaking_p50_ms",
        "xprof_post_soaking_p50_ms",
        "xprof_pre_soaking_tflops_per_device",
        "xprof_post_soaking_tflops_per_device",
        "xprof_slowdown_ratio",
        "wall_clock_pre_soaking_p50_ms",
        "wall_clock_post_soaking_p50_ms",
        "wall_clock_pre_soaking_tflops_per_device",
        "wall_clock_post_soaking_tflops_per_device",
        "wall_clock_slowdown_ratio",
    ):
      self.assertGreater(result.metrics[key], 0.0)
    for key in (
        "hbm_peak_temp_c",
        "total_power_mean_w",
        "pstate_min",
        "pstate_max",
    ):
      self.assertIsNotNone(result.metrics[key])

    tmpdir = self.create_tempdir().full_path
    report.save_output([result], tmpdir)
    self.assertTrue(os.path.exists(os.path.join(tmpdir, "summary.csv")))
    self.assertTrue(os.path.exists(os.path.join(tmpdir, "detailed.json")))

  def test_gemm_pulse_checksum_disabled_on_tpu(self):
    """Tests GemmPulseBenchmark with checksum disabled on TPU."""
    result = self._run_pulse(num_cycles=3, enable_checksum=False)
    self.assertEqual(result.metrics["actual_runs"], 3)
    self.assertGreater(result.metrics["xprof_pre_soaking_p50_ms"], 0.0)
    self.assertGreater(result.metrics["xprof_post_soaking_p50_ms"], 0.0)
    for key in (
        "checksum",
        "all_devices_valid_checksum",
        "intra_host_checksum_match",
        "global_checksum_match",
    ):
      self.assertNotIn(key, result.metrics)

  def test_gemm_pulse_soak_mode_on_tpu(self):
    """Tests multi-cycle soak GemmPulseBenchmark execution on TPU."""
    result = self._run_pulse(num_cycles=2, min_duration_s=0.5)
    self.assertGreater(result.metrics["actual_runs"], 2)
    self.assertTrue(result.metrics["all_devices_valid_checksum"])
    self.assertTrue(result.metrics["intra_host_checksum_match"])
    self.assertTrue(result.metrics["global_checksum_match"])


if __name__ == "__main__":
  absltest.main()
