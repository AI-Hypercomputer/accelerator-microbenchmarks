"""Unit tests for system.py."""

from absl.testing import absltest
from accelerator_microbenchmarks.core import system


class SystemTest(absltest.TestCase):
  """Unit tests for system.py."""

  def test_get_system_valid(self):
    """Verify that get_system returns the correct SystemConfig."""
    sys_config = system.get_system("ironwood")
    self.assertEqual(sys_config.name, "ironwood")
    self.assertIsInstance(sys_config, system.SystemConfig)

    # Test alias
    sys_config_alias = system.get_system("gfc")
    self.assertEqual(sys_config_alias.name, "ironwood")
    self.assertEqual(sys_config_alias, sys_config)

  def test_get_system_case_insensitive(self):
    """Verify that get_system is case insensitive."""
    sys_config = system.get_system("IRONWOOD")
    self.assertEqual(sys_config.name, "ironwood")

  def test_get_system_invalid(self):
    """Verify that get_system raises ValueError for invalid names."""
    with self.assertRaises(ValueError):
      system.get_system("nonexistent")

  def test_system_config_presets(self):
    """Verify that the IRONWOOD preset has correct values."""
    sys_config = system.IRONWOOD
    self.assertEqual(sys_config.name, "ironwood")

    # Test compute stats
    self.assertEqual(
        sys_config.tflops.peak_tflops_per_dtype["bfloat16"], 2307.0
    )
    self.assertEqual(sys_config.tflops.peak_tflops_per_dtype["float32"], 1153.5)
    self.assertEqual(
        sys_config.tflops.peak_tflops_per_dtype["float8_e5m2"], 4614.0
    )
    self.assertEqual(
        sys_config.tflops.peak_tflops_per_dtype["float8_e4m3fn"], 4614.0
    )
    self.assertEqual(sys_config.tflops.peak_tflops_per_dtype["int8"], 4614.0)

    # Test ICI stats
    self.assertEqual(sys_config.ici.peak_bw_gbps, 1200.0)
    self.assertTrue(sys_config.ici.bidirectional)

    # Test HBM stats
    self.assertEqual(sys_config.hbm.curve_gbps[0], (1024, 100.0))
    self.assertEqual(sys_config.hbm.curve_gbps[1], (1048576, 2000.0))
    self.assertEqual(sys_config.hbm.curve_gbps[2], (104857600, 5000.0))
    self.assertEqual(sys_config.hbm.curve_gbps[3], (1073741824, 7380.0))


if __name__ == "__main__":
  absltest.main()
