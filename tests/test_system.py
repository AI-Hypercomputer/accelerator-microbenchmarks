"""Unit tests for system.py."""

from absl.testing import absltest
from accelerator_microbenchmarks.core import system


class SystemTest(absltest.TestCase):
  """Unit tests for system.py."""

  def test_tpu_version_from_str_valid(self):
    """Verify that TpuVersion.from_str correctly normalizes aliases."""
    for alias in (
        "tpu v7x",
        "tpu7x",
        "TPU7x",
        "ironwood",
        "tpu v7",
        "tpu7",
        "TPU7",
        "v7",
    ):
      self.assertEqual(system.TpuVersion.from_str(alias), system.TpuVersion.TPU7X)

    for alias in ("v6e", "tpu v6 lite", "trillium", "6e"):
      self.assertEqual(system.TpuVersion.from_str(alias), system.TpuVersion.V6E)

    # Idempotent on enum instance
    self.assertEqual(
        system.TpuVersion.from_str(system.TpuVersion.TPU7X), system.TpuVersion.TPU7X
    )

  def test_tpu_version_str_behavior(self):
    """Verify that TpuVersion inherits from str and stringifies to its value."""
    self.assertIsInstance(system.TpuVersion.TPU7X, str)
    self.assertIsInstance(system.TpuVersion.V6E, str)
    self.assertEqual(str(system.TpuVersion.TPU7X), "tpu7x")
    self.assertEqual(str(system.TpuVersion.V6E), "v6e")
    self.assertEqual(f"{system.TpuVersion.TPU7X}", "tpu7x")
    self.assertEqual(f"{system.TpuVersion.V6E}", "v6e")

  def test_default_fallback_dtype(self):
    """Verify default fallback compute dtype is bfloat16 and present in all specs."""
    self.assertEqual(system.DEFAULT_FALLBACK_DTYPE, "bfloat16")
    for tpu_ver, hw_spec in system.HARDWARE_SPECS.items():
      self.assertIsNotNone(hw_spec.tflops)
      self.assertIn(
          system.DEFAULT_FALLBACK_DTYPE,
          hw_spec.tflops.peak_tflops_per_device,
          f"HardwareSpec for {tpu_ver} is missing peak TFLOPS for"
          f" {system.DEFAULT_FALLBACK_DTYPE}",
      )
    self.assertEqual(system.TpuVersion.TPU7X, "tpu7x")
    self.assertEqual(system.TpuVersion.V6E, "v6e")

  def test_tpu_version_from_str_invalid(self):
    """Verify that TpuVersion.from_str raises ValueError for unsupported hardware."""
    with self.assertRaises(ValueError):
      system.TpuVersion.from_str("unsupported_chip")

    with self.assertRaises(ValueError):
      system.TpuVersion.from_str("")

  def test_get_hardware_spec_valid(self):
    """Verify that get_hardware_spec returns the correct HardwareSpec."""
    v7x_spec = system.get_hardware_spec(system.TpuVersion.TPU7X)
    self.assertEqual(v7x_spec.name, system.TpuVersion.TPU7X)
    self.assertEqual(v7x_spec, system.TPU7X_HARDWARE_SPEC)
    self.assertIsInstance(v7x_spec, system.HardwareSpec)

    v6e_spec = system.get_hardware_spec(system.TpuVersion.V6E)
    self.assertEqual(v6e_spec.name, system.TpuVersion.V6E)
    self.assertEqual(v6e_spec, system.V6E_HARDWARE_SPEC)

    # Test string aliases
    self.assertEqual(
        system.get_hardware_spec("ironwood"), system.TPU7X_HARDWARE_SPEC
    )
    self.assertEqual(
        system.get_hardware_spec("tpu v7"), system.TPU7X_HARDWARE_SPEC
    )
    self.assertEqual(
        system.get_hardware_spec("tpu7"), system.TPU7X_HARDWARE_SPEC
    )
    self.assertEqual(
        system.get_hardware_spec("v7"), system.TPU7X_HARDWARE_SPEC
    )
    self.assertEqual(
        system.get_hardware_spec("trillium"), system.V6E_HARDWARE_SPEC
    )
    self.assertEqual(
        system.get_hardware_spec("v6e"), system.V6E_HARDWARE_SPEC
    )

  def test_get_hardware_spec_case_insensitive(self):
    """Verify that get_hardware_spec is case insensitive."""
    hw_spec = system.get_hardware_spec("IRONWOOD")
    self.assertEqual(hw_spec.name, system.TpuVersion.TPU7X)
    hw_spec_v6e = system.get_hardware_spec("V6E")
    self.assertEqual(hw_spec_v6e.name, system.TpuVersion.V6E)

  def test_get_hardware_spec_invalid(self):
    """Verify that get_hardware_spec raises ValueError for invalid names."""
    with self.assertRaises(ValueError):
      system.get_hardware_spec("nonexistent")

  def test_hardware_spec_presets(self):
    """Verify that the TPU7X_HARDWARE_SPEC preset has correct values."""
    hw_spec = system.TPU7X_HARDWARE_SPEC
    self.assertEqual(hw_spec.name, system.TpuVersion.TPU7X)
    self.assertEqual(hw_spec.topology_dimension, 3)

    # Test compute stats
    self.assertEqual(
        hw_spec.tflops.peak_tflops_per_device["bfloat16"], 1153.5
    )
    self.assertEqual(hw_spec.tflops.peak_tflops_per_device["float32"], 576.75)
    self.assertEqual(
        hw_spec.tflops.peak_tflops_per_device["float8_e5m2"], 2307.0
    )
    self.assertEqual(
        hw_spec.tflops.peak_tflops_per_device["float8_e4m3fn"], 2307.0
    )
    self.assertEqual(hw_spec.tflops.peak_tflops_per_device["int8"], 2307.0)

    # Test ICI stats
    self.assertEqual(hw_spec.ici.peak_bw_gbps, 1200.0)
    self.assertTrue(hw_spec.ici.bidirectional)

    # Test HBM stats
    self.assertEqual(hw_spec.hbm.curve_gbps[0], (1024, 100.0))
    self.assertEqual(hw_spec.hbm.curve_gbps[1], (1048576, 2000.0))
    self.assertEqual(hw_spec.hbm.curve_gbps[2], (104857600, 5000.0))
    self.assertEqual(hw_spec.hbm.curve_gbps[3], (1073741824, 7380.0))


  def test_hardware_spec_v6e_presets(self):
    """Verify that the V6E_HARDWARE_SPEC preset has correct values."""
    hw_spec = system.V6E_HARDWARE_SPEC
    self.assertEqual(hw_spec.name, system.TpuVersion.V6E)
    self.assertEqual(hw_spec.topology_dimension, 2)

    # Test compute stats
    self.assertEqual(
        hw_spec.tflops.peak_tflops_per_device["bfloat16"], 918.0
    )
    self.assertEqual(hw_spec.tflops.peak_tflops_per_device["float32"], 459.0)
    self.assertEqual(
        hw_spec.tflops.peak_tflops_per_device["float8_e5m2"], 918.0
    )
    self.assertEqual(
        hw_spec.tflops.peak_tflops_per_device["float8_e4m3fn"], 918.0
    )
    self.assertEqual(hw_spec.tflops.peak_tflops_per_device["int8"], 1836.0)
    self.assertEqual(hw_spec.tflops.peak_tflops_per_device["int4"], 3672.0)

    # Test ICI stats
    self.assertEqual(hw_spec.ici.peak_bw_gbps, 800.0)
    self.assertTrue(hw_spec.ici.bidirectional)

    # Test HBM stats
    self.assertEqual(hw_spec.hbm.curve_gbps[0], (1024, 50.0))
    self.assertEqual(hw_spec.hbm.curve_gbps[1], (1048576, 800.0))
    self.assertEqual(hw_spec.hbm.curve_gbps[2], (104857600, 1400.0))
    self.assertEqual(hw_spec.hbm.curve_gbps[3], (1073741824, 1638.4))


if __name__ == "__main__":
  absltest.main()
