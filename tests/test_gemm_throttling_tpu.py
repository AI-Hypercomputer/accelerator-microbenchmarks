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

      # Verify thermal_per_device matches device count
      thermal_per_device = benchmark.executor.get_result_details().get(
          "thermal_per_device", {}
      )
      self.assertLen(thermal_per_device, jax.local_device_count())


if __name__ == "__main__":
  absltest.main()
