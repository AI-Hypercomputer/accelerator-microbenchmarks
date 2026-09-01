"""Unit tests for core/platform.py."""

from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import platform


class PlatformTest(absltest.TestCase):
  """Unit tests for platform.py."""

  def test_get_package_version_from_module_attribute(self):
    """Verify _get_package_version resolves version from module __version__."""
    mock_mod = mock.MagicMock()
    mock_mod.__version__ = "0.4.30"
    with mock.patch.object(
        platform.importlib, "import_module", return_value=mock_mod
    ):
      version = platform._get_package_version("jax")
      self.assertEqual(version, "0.4.30")

  def test_get_package_version_from_metadata(self):
    """Verify _get_package_version falls back to importlib.metadata."""
    mock_mod = mock.MagicMock(spec=[])
    with mock.patch.object(
        platform.importlib, "import_module", return_value=mock_mod
    ), mock.patch.object(
        platform.importlib.metadata, "version", return_value="0.4.30"
    ) as mock_meta:
      version = platform._get_package_version("jax")
      self.assertEqual(version, "0.4.30")
      mock_meta.assert_called_once_with("jax")

  def test_get_package_version_with_fallback_pkg_names(self):
    """Verify _get_package_version tries fallback package names."""
    with mock.patch.object(
        platform.importlib,
        "import_module",
        side_effect=ImportError("No module"),
    ):

      def mock_metadata_version(pkg):
        if pkg == "libtpu":
          raise platform.importlib.metadata.PackageNotFoundError()
        if pkg == "libtpu-nightly":
          return "0.1.dev20240101"
        raise platform.importlib.metadata.PackageNotFoundError()

      with mock.patch.object(
          platform.importlib.metadata,
          "version",
          side_effect=mock_metadata_version,
      ):
        version = platform._get_package_version("libtpu", "libtpu-nightly")
        self.assertEqual(version, "0.1.dev20240101")

  def test_get_package_version_unknown_fallback(self):
    """Verify _get_package_version returns 'unknown' when all resolution fails."""
    with mock.patch.object(
        platform.importlib,
        "import_module",
        side_effect=ImportError("No module"),
    ), mock.patch.object(
        platform.importlib.metadata,
        "version",
        side_effect=platform.importlib.metadata.PackageNotFoundError(),
    ):
      version = platform._get_package_version("nonexistent_pkg")
      self.assertEqual(version, "unknown")

  def test_get_package_version_broad_exception_handling(self):
    """Verify _get_package_version handles unexpected exceptions gracefully."""
    with mock.patch.object(
        platform.importlib, "import_module", side_effect=Exception("unexpected")
    ), mock.patch.object(
        platform.importlib.metadata,
        "version",
        side_effect=Exception("unexpected"),
    ):
      version = platform._get_package_version("faulty_pkg")
      self.assertEqual(version, "unknown")

  def test_get_platform_description_success(self):
    """Verify get_platform_description returns complete metadata dictionary."""
    mock_device = mock.MagicMock()
    mock_device.platform = "tpu"
    mock_device.device_kind = "TPU v7x"
    mock_device.coords = (0, 0, 0)

    with mock.patch("jax.distributed.initialize") as mock_dist_init, mock.patch(
        "jax.default_backend", return_value="tpu"
    ), mock.patch("jax.devices", return_value=[mock_device]), mock.patch(
        "jax.device_count", return_value=4
    ), mock.patch(
        "jax.local_device_count", return_value=4
    ), mock.patch(
        "jax.process_count", return_value=1
    ), mock.patch(
        "jax.process_index", return_value=0
    ), mock.patch.object(
        platform.py_platform, "python_version", return_value="3.11.0"
    ), mock.patch.object(
        platform, "_get_package_version"
    ) as mock_pkg_version:
      mock_pkg_version.side_effect = lambda mod, *fallbacks: {
          "jax": "0.4.30",
          "jaxlib": "0.4.30",
          "libtpu": "0.1.dev20240101",
      }.get(mod, "unknown")

      desc = platform.get_platform_description()

      mock_dist_init.assert_called_once()
      expected_keys = {
          "tpu_type",
          "topology",
          "total_devices",
          "local_devices",
          "process_count",
          "process_index",
          "python_version",
          "jax_version",
          "jaxlib_version",
          "libtpu_version",
      }
      self.assertEqual(set(desc.keys()), expected_keys)
      self.assertEqual(desc["tpu_type"], "TPU v7x")
      self.assertEqual(desc["topology"], "1x1x1")
      self.assertEqual(desc["total_devices"], 4)
      self.assertEqual(desc["local_devices"], 4)
      self.assertEqual(desc["process_count"], 1)
      self.assertEqual(desc["process_index"], 0)
      self.assertEqual(desc["python_version"], "3.11.0")
      self.assertEqual(desc["jax_version"], "0.4.30")
      self.assertEqual(desc["jaxlib_version"], "0.4.30")
      self.assertEqual(desc["libtpu_version"], "0.1.dev20240101")

  def test_get_platform_description_distributed_init_failure_ignored(self):
    """Verify get_platform_description proceeds if distributed.initialize fails."""
    mock_device = mock.MagicMock()
    mock_device.platform = "tpu"
    mock_device.device_kind = "TPU v6e"
    mock_device.coords = (0, 0, 0)

    with mock.patch(
        "jax.distributed.initialize",
        side_effect=Exception("Not in distributed cluster"),
    ), mock.patch("jax.default_backend", return_value="tpu"), mock.patch(
        "jax.devices", return_value=[mock_device]
    ), mock.patch(
        "jax.device_count", return_value=1
    ), mock.patch(
        "jax.local_device_count", return_value=1
    ), mock.patch(
        "jax.process_count", return_value=1
    ), mock.patch(
        "jax.process_index", return_value=0
    ):
      desc = platform.get_platform_description()
      self.assertEqual(desc["tpu_type"], "TPU v6e")
      self.assertEqual(desc["total_devices"], 1)

  def test_get_platform_description_cpu_fallback(self):
    """Verify get_platform_description returns none for tpu_type and topology on CPU."""
    mock_device = mock.MagicMock()
    mock_device.platform = "cpu"
    mock_device.device_kind = "cpu"

    with mock.patch("jax.distributed.initialize"), mock.patch(
        "jax.default_backend", return_value="cpu"
    ), mock.patch("jax.devices", return_value=[mock_device]), mock.patch(
        "jax.device_count", return_value=8
    ), mock.patch(
        "jax.local_device_count", return_value=8
    ), mock.patch(
        "jax.process_count", return_value=1
    ), mock.patch(
        "jax.process_index", return_value=0
    ):
      desc = platform.get_platform_description()
      self.assertEqual(desc["tpu_type"], "none")
      self.assertEqual(desc["topology"], "none")
      self.assertEqual(desc["total_devices"], 8)
      self.assertEqual(desc["local_devices"], 8)

  def test_get_platform_description_error_not_masked(self):
    """Verify unexpected exceptions outside jax.devices() are not masked."""
    mock_device = mock.MagicMock()
    mock_device.platform = "tpu"
    mock_device.device_kind = "TPU v7x"
    mock_device.coords = (0, 0, 0)

    with mock.patch("jax.distributed.initialize"), mock.patch(
        "jax.default_backend", return_value="tpu"
    ), mock.patch("jax.devices", return_value=[mock_device]), mock.patch(
        "jax.device_count", side_effect=ValueError("Simulated device failure")
    ):
      with self.assertRaises(ValueError) as cm:
        platform.get_platform_description()
      self.assertIn("Simulated device failure", str(cm.exception))

  def test_get_platform_description_device_exception_raises_runtime_error(self):
    """Verify get_platform_description wraps general exceptions in RuntimeError."""
    with mock.patch("jax.distributed.initialize"), mock.patch(
        "jax.default_backend", return_value="tpu"
    ), mock.patch(
        "jax.devices", side_effect=Exception("PJRT initialization error")
    ):
      with self.assertRaises(RuntimeError) as cm:
        platform.get_platform_description()
      self.assertIn(
          "TPU runtime environment is not properly initialized",
          str(cm.exception),
      )

  def test_get_platform_description_backend_exception_raises_runtime_error(
      self,
  ):
    """Verify get_platform_description wraps default_backend exceptions in RuntimeError."""
    with mock.patch("jax.distributed.initialize"), mock.patch(
        "jax.default_backend",
        side_effect=Exception("PJRT initialization error"),
    ):
      with self.assertRaises(RuntimeError) as cm:
        platform.get_platform_description()
      self.assertIn(
          "TPU runtime environment is not properly initialized",
          str(cm.exception),
      )

  def test_get_topology_with_explicit_device_kind(self):
    """Verify _get_topology respects explicit device_kind parameter."""
    devices = []
    for x in range(2):
      for y in range(4):
        dev = mock.MagicMock()
        dev.coords = (x, y, 0)
        devices.append(dev)
    topology = platform._get_topology(devices, device_kind="TPU v6 lite")
    self.assertEqual(topology, "2x4")

  def test_get_topology_v6e(self):
    """Verify _get_topology strips trivial 3rd dimension for TPU v6."""
    for kind in ("TPU v6 lite", "v6e"):
      devices = []
      for x in range(2):
        for y in range(4):
          dev = mock.MagicMock()
          dev.device_kind = kind
          dev.coords = (x, y, 0)
          devices.append(dev)
      self.assertEqual(platform._get_topology(devices), "2x4")

  def test_get_topology_v6e_large_simulated(self):
    """Verify _get_topology handles simulated 16x16 v6e mesh."""
    devices = []
    for x in range(16):
      for y in range(16):
        dev = mock.MagicMock()
        dev.device_kind = "v6e"
        dev.coords = (x, y, 0)
        devices.append(dev)
    self.assertEqual(platform._get_topology(devices), "16x16")

  def test_get_topology_v7x(self):
    """Verify _get_topology preserves 3D format for TPU v7x and Ironwood."""
    for kind in ("TPU7x", "tpu7x", "TPU v7x", "tpu v7x", "ironwood", "gfc"):
      devices = []
      for x in range(2):
        for y in range(2):
          for _ in range(2):
            dev = mock.MagicMock()
            dev.device_kind = kind
            dev.coords = (x, y, 0)
            devices.append(dev)
      self.assertEqual(platform._get_topology(devices), "2x2x1")

  def test_get_topology_v7x_large_simulated(self):
    """Verify _get_topology handles simulated 4x4x4 v7x mesh."""
    devices = []
    for x in range(4):
      for y in range(4):
        for z in range(4):
          dev = mock.MagicMock()
          dev.device_kind = "ironwood"
          dev.coords = (x, y, z)
          devices.append(dev)
    self.assertEqual(platform._get_topology(devices), "4x4x4")

  def test_get_topology_unconfigured_platform_graceful(self):
    """Verify _get_topology formats coordinates even if platform is unconfigured in SYSTEMS."""
    for kind in ["CPU", "GPU", "TPU v4", "", None]:
      dev = mock.MagicMock()
      dev.coords = (0, 0, 0)
      dev.device_kind = kind
      self.assertEqual(platform._get_topology([dev]), "1x1x1")

  def test_get_topology_fallback_unknown(self):
    """Verify _get_topology returns unknown for invalid/missing coordinates."""
    # Empty devices list
    self.assertEqual(platform._get_topology([]), "unknown")

    # Missing coords attribute
    dev_no_coords = mock.MagicMock(spec=[])
    self.assertEqual(platform._get_topology([dev_no_coords]), "unknown")

    # None coords
    dev_none_coords = mock.MagicMock()
    dev_none_coords.coords = None
    self.assertEqual(platform._get_topology([dev_none_coords]), "unknown")

    # Mismatched dimension lengths
    dev_2d = mock.MagicMock()
    dev_2d.device_kind = "tpu v7x"
    dev_2d.coords = (0, 0)
    dev_3d = mock.MagicMock()
    dev_3d.device_kind = "tpu v7x"
    dev_3d.coords = (0, 0, 0)
    self.assertEqual(platform._get_topology([dev_2d, dev_3d]), "unknown")


if __name__ == "__main__":
  absltest.main()
