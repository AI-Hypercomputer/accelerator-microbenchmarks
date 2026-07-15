"""Unit tests for csv_loader.py."""

import io
import os
import tempfile
from unittest import mock
import urllib.error as urllib_error

from absl.testing import absltest
from accelerator_microbenchmarks.core import csv_loader


class CsvLoaderTest(absltest.TestCase):
  """Unit tests for csv_loader.py."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.TemporaryDirectory()

  def tearDown(self):
    self.test_dir.cleanup()
    super().tearDown()

  def test_load_shapes_from_csv_valid_local(self):
    """Test loading from a valid local CSV file with various types."""
    csv_content = """param1,param2,param3,param4
10,3.14,hello,
20,6.28,world,
"""
    csv_path = os.path.join(self.test_dir.name, "test.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
      f.write(csv_content)

    shapes = csv_loader.load_shapes_from_csv(csv_path)

    self.assertLen(shapes, 2)
    self.assertEqual(
        shapes[0],
        {"param1": 10, "param2": 3.14, "param3": "hello", "param4": None},
    )
    self.assertEqual(
        shapes[1],
        {"param1": 20, "param2": 6.28, "param3": "world", "param4": None},
    )

  def test_load_shapes_from_csv_empty_path(self):
    """Test loading from an empty path."""
    shapes = csv_loader.load_shapes_from_csv("")
    self.assertEqual(shapes, [])

  def test_load_shapes_from_csv_missing_file(self):
    """Test loading from a missing file."""
    shapes = csv_loader.load_shapes_from_csv(
        os.path.join(self.test_dir.name, "nonexistent.csv")
    )
    self.assertEqual(shapes, [])

  @mock.patch("urllib.request.urlopen")
  def test_load_shapes_from_csv_valid_remote(self, mock_urlopen):
    """Test loading from a valid remote CSV file."""
    csv_content = b"""param1,param2
100,200
"""
    mock_response = io.BytesIO(csv_content)
    mock_urlopen.return_value.__enter__.return_value = mock_response

    shapes = csv_loader.load_shapes_from_csv("https://example.com/test.csv")

    self.assertLen(shapes, 1)
    self.assertEqual(shapes[0], {"param1": 100, "param2": 200})
    mock_urlopen.assert_called_once_with("https://example.com/test.csv")

  @mock.patch("urllib.request.urlopen")
  def test_load_shapes_from_csv_remote_error(self, mock_urlopen):
    """Test loading from a remote CSV file with a URL error."""
    mock_urlopen.side_effect = urllib_error.URLError("URL error")

    shapes = csv_loader.load_shapes_from_csv("https://example.com/test.csv")

    self.assertEqual(shapes, [])
    mock_urlopen.assert_called_once_with("https://example.com/test.csv")

  def test_load_shapes_from_csv_type_inference(self):
    """Test type inference for all supported types."""
    csv_content = """int_param,float_param,str_param,none_param
1,1.1,one,
-2,-2.2,two,
"""
    csv_path = os.path.join(self.test_dir.name, "test_types.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
      f.write(csv_content)

    shapes = csv_loader.load_shapes_from_csv(csv_path)

    self.assertLen(shapes, 2)
    # Row 1
    self.assertIsInstance(shapes[0]["int_param"], int)
    self.assertEqual(shapes[0]["int_param"], 1)
    self.assertIsInstance(shapes[0]["float_param"], float)
    self.assertEqual(shapes[0]["float_param"], 1.1)
    self.assertIsInstance(shapes[0]["str_param"], str)
    self.assertEqual(shapes[0]["str_param"], "one")
    self.assertIsNone(shapes[0]["none_param"])

    # Row 2
    self.assertIsInstance(shapes[1]["int_param"], int)
    self.assertEqual(shapes[1]["int_param"], -2)
    self.assertIsInstance(shapes[1]["float_param"], float)
    self.assertEqual(shapes[1]["float_param"], -2.2)
    self.assertIsInstance(shapes[1]["str_param"], str)
    self.assertEqual(shapes[1]["str_param"], "two")
    self.assertIsNone(shapes[1]["none_param"])


if __name__ == "__main__":
  absltest.main()
