"""Tests for HBM bandwidth utilizing TPU Ghostfish."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import hbm
from accelerator_microbenchmarks.core import system
import jax

_UTILIZATION_THRESHOLD = 0.8


class HBMBandwidthTPUTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.bm = hbm.HBMBandwidthBenchmark()

    # Use Xprof to pull exact on-device execution time, meaning we don't need
    # massive arrays to dilute python overhead anymore.
    self.params = {
        "size": 1024 * 1024 * 128,
        "warmup_tries": 3,
        "num_runs": 10,
        "xprof_timing": True,
    }

  def tearDown(self):
    super().tearDown()
    jax.clear_caches()

  def _get_peak_bandwidth_per_core(self):
    # We only need ghostfish (v7x)
    gfc = system.get_system("gfc")
    # Peak BW is the highest value in the curve.
    # The curve specifies the peak bandwidth for an entire chip (2 TensorCores).
    # Since this microbenchmark forces execution on a single local device,
    # the theoretical peak for our test is half of the chip's peak.
    chip_peak_bw = gfc.hbm.curve_gbps[-1][1]
    return chip_peak_bw / 2.0

  @parameterized.parameters(
      ("copy",),
      ("scale",),
      ("add",),
      ("triad",),
  )
  def test_hbm_utilization_above_80_percent(self, op_type):
    """Verify that HBM utilization is >80% for the given operation."""
    if not jax.devices() or jax.devices()[0].platform != "tpu":
      self.skipTest("This test requires a TPU backend.")

    params = dict(self.params, op_type=op_type)
    result = self.bm.run(**params)

    # Use xprof_avg_ms for precise device-level timing if available
    xprof_avg_ms = result.metrics.get("xprof_avg_ms", None)
    if xprof_avg_ms is not None:
      total_bytes = self.bm.get_total_bytes(**params)
      bw_gb_s = (total_bytes / (xprof_avg_ms / 1000.0)) / 1e9
    else:
      bw_gb_s = result.metrics.get("bandwidth_gb_s", 0)

    peak_bw = self._get_peak_bandwidth_per_core()
    utilization = bw_gb_s / peak_bw

    print(f"HBM utilization for '{op_type}' was {utilization*100:.2f}%")
    print(f"Measured {bw_gb_s:.2f} GB/s, Peak {peak_bw} GB/s")
    self.assertGreaterEqual(
        utilization,
        _UTILIZATION_THRESHOLD,
        f"HBM utilization for '{op_type}' was {utilization*100:.2f}%, "
        f"which is below the {_UTILIZATION_THRESHOLD*100:.2f}% threshold.",
    )


if __name__ == "__main__":
  absltest.main()
