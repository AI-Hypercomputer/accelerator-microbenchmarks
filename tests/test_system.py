"""Unit tests for system.py."""

from unittest import mock

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

  def test_get_runtime_device_info_success(self):
    with mock.patch("jax.default_backend", return_value="cpu"):
      with mock.patch("jax.device_count", return_value=1):
        with mock.patch("jax.local_device_count", return_value=1):
          with mock.patch("jax.__version__", "0.4.1"):
            with mock.patch("importlib.metadata.version") as mock_version:
              mock_version.side_effect = (
                  lambda pkg: "1.0" if pkg == "libtpu" else None
              )
              with mock.patch("jax.devices") as mock_devices:
                mock_device = mock.MagicMock()
                mock_device.device_kind = "CPU"
                mock_devices.return_value = [mock_device]

                info = system.get_runtime_device_info()

                self.assertEqual(info["platform"], "cpu")
                self.assertEqual(info["device_count"], 1)
                self.assertEqual(info["local_device_count"], 1)
                self.assertEqual(info["jax_version"], "0.4.1")
                self.assertEqual(info["libtpu_version"], "1.0")
                self.assertEqual(info["chip_version"], "CPU")

  def test_get_runtime_device_info_fallback(self):
    with mock.patch("jax.default_backend", side_effect=Exception("mock err")):
      # If default_backend fails, does it crash?
      pass

    with mock.patch("jax.default_backend", return_value="cpu"):
      with mock.patch("jax.device_count", return_value=1):
        with mock.patch("jax.local_device_count", return_value=1):
          with mock.patch(
              "importlib.metadata.version", side_effect=Exception("no package")
          ):
            with mock.patch(
                "jax.devices", side_effect=Exception("no generic context")
            ):
              info = system.get_runtime_device_info()

              self.assertNotIn("libtpu_version", info)
              self.assertNotIn("chip_version", info)


if __name__ == "__main__":
  absltest.main()
