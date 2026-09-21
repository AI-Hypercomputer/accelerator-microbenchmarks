"""Unit tests for system.py."""

import os
import pkgutil

from absl.testing import absltest
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.specs import schema

# Modules in the `specs` package that hold shared infrastructure rather than a
# TPU generation definition. Every other module is expected to be named after
# the canonical version id whose spec it defines.
_INFRASTRUCTURE_MODULES = frozenset({"schema"})


class SystemTest(absltest.TestCase):
  """Unit tests for system.py."""

  def test_tpu_version_from_device_kind_valid(self):
    """Verify that from_device_kind maps JAX device kinds to generations."""
    for device_kind in ("tpu7x", "TPU7x", "TPU v7x", "  tpu7x  "):
      self.assertEqual(
          system.TpuVersion.from_device_kind(device_kind),
          system.TpuVersion.TPU7X,
      )

    for device_kind in ("v6e", "V6E", "TPU v6e", "TPU v6 lite"):
      self.assertEqual(
          system.TpuVersion.from_device_kind(device_kind),
          system.TpuVersion.V6E,
      )

  def test_tpu_version_from_device_kind_is_idempotent_on_enum(self):
    """Verify that an already resolved TpuVersion is returned unchanged."""
    self.assertIs(
        system.TpuVersion.from_device_kind(system.TpuVersion.TPU7X),
        system.TpuVersion.TPU7X,
    )

  def test_tpu_version_from_device_kind_rejects_partial_match(self):
    """Verify that device kinds are matched exactly rather than by substring."""
    for device_kind in ("tpu7xyz", "v6", "7", "7x", "v7x"):
      with self.assertRaises(ValueError):
        system.TpuVersion.from_device_kind(device_kind)

  def test_tpu_version_from_device_kind_rejects_marketing_names(self):
    """Verify that only device kinds, not marketing names, are accepted.

    Marketing and codenames live in a different namespace than the PJRT
    `device_kind` strings, so they are deliberately not resolvable here.
    """
    for name in ("ironwood", "trillium", "6e"):
      with self.assertRaises(ValueError):
        system.TpuVersion.from_device_kind(name)

  def test_tpu_version_from_device_kind_invalid(self):
    """Verify that from_device_kind rejects unsupported hardware."""
    with self.assertRaises(ValueError):
      system.TpuVersion.from_device_kind("unsupported_chip")

    with self.assertRaises(ValueError):
      system.TpuVersion.from_device_kind("")

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

  def test_public_generations_are_always_available(self):
    """Verify that the publicly announced generations are never stripped."""
    self.assertContainsSubset(
        [system.TpuVersion.TPU7X, system.TpuVersion.V6E],
        system.HARDWARE_SPECS,
    )

  def test_canonical_version_ids_are_lowercase(self):
    """Verify that every canonical version id is lowercase."""
    for version in system.HARDWARE_SPECS:
      self.assertEqual(str(version), str(version).lower())

  def test_specs_declare_their_own_version_as_name(self):
    """Verify that HARDWARE_SPECS keys agree with the specs they map to."""
    for version, spec in system.HARDWARE_SPECS.items():
      self.assertEqual(spec.name, version)

  def test_every_spec_module_in_the_package_is_aggregated(self):
    """Verify that no spec module is missing from HARDWARE_SPECS.

    `HARDWARE_SPECS` is maintained by hand, so this guards against adding a
    spec module and forgetting to register it. Spec modules must be named
    after the canonical version id they define.
    """
    package_dir = os.path.dirname(schema.__file__)
    discovered = {
        module_name
        for _, module_name, _ in pkgutil.iter_modules([package_dir])
        if module_name not in _INFRASTRUCTURE_MODULES
    }
    self.assertNotEmpty(discovered)
    self.assertContainsSubset(
        discovered, {str(version) for version in system.HARDWARE_SPECS}
    )

  def test_get_hardware_spec_valid(self):
    """Verify that get_hardware_spec returns the correct HardwareSpec."""
    v7x_spec = system.get_hardware_spec(system.TpuVersion.TPU7X)
    self.assertEqual(v7x_spec.name, system.TpuVersion.TPU7X)
    self.assertEqual(v7x_spec, system.TPU7X_HARDWARE_SPEC)
    self.assertIsInstance(v7x_spec, system.HardwareSpec)

    v6e_spec = system.get_hardware_spec(system.TpuVersion.V6E)
    self.assertEqual(v6e_spec.name, system.TpuVersion.V6E)
    self.assertEqual(v6e_spec, system.V6E_HARDWARE_SPEC)

  def test_get_hardware_spec_from_jax_device_kind(self):
    """Verify that JAX device_kind strings resolve to the right generation."""
    self.assertEqual(
        system.get_hardware_spec(system.TpuVersion.from_device_kind("TPU7x")),
        system.TPU7X_HARDWARE_SPEC,
    )
    self.assertEqual(
        system.get_hardware_spec(
            system.TpuVersion.from_device_kind("TPU v6 lite")
        ),
        system.V6E_HARDWARE_SPEC,
    )

  def test_get_hardware_spec_invalid(self):
    """Verify that get_hardware_spec raises ValueError for invalid versions."""
    with self.assertRaisesRegex(ValueError, "Unsupported TPU hardware"):
      system.get_hardware_spec("nonexistent")

  def test_get_hardware_spec_keys_on_the_canonical_version(self):
    """Verify lookups resolve by canonical version, not by arbitrary spelling.

    `TpuVersion` subclasses `str`, so a member hashes like its value and the
    exact canonical id resolves as well as the enum member itself. Every
    other spelling, including JAX device kinds and marketing names, must go
    through `TpuVersion.from_device_kind` first.
    """
    self.assertEqual(
        system.get_hardware_spec("tpu7x"), system.TPU7X_HARDWARE_SPEC
    )
    for target in ("TPU7x", "TPU v7x", "IRONWOOD", "ironwood", "7x"):
      with self.assertRaisesRegex(ValueError, "Unsupported TPU hardware"):
        system.get_hardware_spec(target)

  def test_hardware_spec_tpu7x_presets(self):
    """Verify that the TPU7X_HARDWARE_SPEC preset has correct values."""
    hw_spec = system.TPU7X_HARDWARE_SPEC
    self.assertEqual(hw_spec.name, system.TpuVersion.TPU7X)
    self.assertEqual(hw_spec.topology_dimension, 3)
    self.assertEqual(hw_spec.devices_per_chip, 2)

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
    self.assertEqual(hw_spec.ici.peak_bw_gbps_per_chip, 1200.0)
    self.assertTrue(hw_spec.ici.bidirectional)

    # Test HBM stats
    self.assertEqual(hw_spec.hbm.peak_bw_gbps, 7380.0)
    self.assertEqual(hw_spec.hbm.peak_bw_gbps_per_chip, 7380.0)
    self.assertEqual(hw_spec.peak_hbm_bandwidth_per_device, 3690.0)

  def test_hardware_spec_v6e_presets(self):
    """Verify that the V6E_HARDWARE_SPEC preset has correct values."""
    hw_spec = system.V6E_HARDWARE_SPEC
    self.assertEqual(hw_spec.name, system.TpuVersion.V6E)
    self.assertEqual(hw_spec.topology_dimension, 2)
    self.assertEqual(hw_spec.devices_per_chip, 1)

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
    self.assertEqual(hw_spec.ici.peak_bw_gbps_per_chip, 800.0)
    self.assertTrue(hw_spec.ici.bidirectional)

    # Test HBM stats
    self.assertEqual(hw_spec.hbm.peak_bw_gbps, 1638.4)
    self.assertEqual(hw_spec.hbm.peak_bw_gbps_per_chip, 1638.4)
    self.assertEqual(hw_spec.peak_hbm_bandwidth_per_device, 1638.4)

  def test_hardware_spec_hbm_none(self):
    """Verify peak_hbm_bandwidth_per_device returns 0.0 when HBM is None or devices_per_chip <= 0."""
    hw_spec = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        devices_per_chip=2,
        hbm=None,
    )
    self.assertEqual(hw_spec.peak_hbm_bandwidth_per_device, 0.0)

    hw_spec_zero_dev = system.HardwareSpec(
        name=system.TpuVersion.TPU7X,
        devices_per_chip=0,
        hbm=system.HbmSpec(peak_bw_gbps=100.0),
    )
    self.assertEqual(hw_spec_zero_dev.peak_hbm_bandwidth_per_device, 0.0)


if __name__ == "__main__":
  absltest.main()
