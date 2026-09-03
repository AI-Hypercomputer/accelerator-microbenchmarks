"""Integration tests for core/platform.py requiring TPU hardware."""

from absl.testing import absltest
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import system
import jax


class PlatformTpuTest(absltest.TestCase):
  """Hardware-in-the-loop tests for platform.py using live JAX on TPU."""

  def test_get_platform_info_live_tpu(self):
    """Verify get_platform_info returns valid hardware metadata on real TPU."""
    if jax.default_backend() != "tpu":
      self.skipTest("This test requires real TPU devices.")

    desc = platform.get_platform_info()

    self.assertIsInstance(desc, platform.PlatformInfo)
    self.assertIsInstance(desc.tpu_type, system.TpuVersion)
    self.assertIn(
        desc.tpu_type, (system.TpuVersion.TPU7X, system.TpuVersion.V6E)
    )
    self.assertNotEqual(desc.topology, "unknown")
    self.assertRegex(desc.topology, r"^\d+(x\d+)*$")
    self.assertGreater(desc.total_devices, 0)
    self.assertGreater(desc.local_devices, 0)
    self.assertGreater(desc.process_count, 0)
    self.assertGreaterEqual(desc.process_index, 0)
    self.assertIsInstance(desc.python_version, str)
    self.assertNotEqual(desc.python_version, "unknown")
    self.assertIsInstance(desc.jax_version, str)
    self.assertNotEqual(desc.jax_version, "unknown")
    # In hermetic builds (e.g. Google3), pip package metadata may be absent,
    # causing jaxlib or libtpu versions to resolve to "unknown".
    self.assertIsInstance(desc.jaxlib_version, str)
    self.assertTrue(desc.jaxlib_version)
    self.assertIsInstance(desc.libtpu_version, str)
    self.assertTrue(desc.libtpu_version)


if __name__ == "__main__":
  absltest.main()
