"""Unit tests for csv_loader.py."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import csv_loader
import pandas as pd


class CsvLoaderTest(absltest.TestCase):
  """Unit tests for csv_loader.py."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.TemporaryDirectory()

  def tearDown(self):
    self.test_dir.cleanup()
    super().tearDown()

  def test_load_cases_from_csv_valid_local(self):
    """Test loading from a valid local CSV file with various types."""
    csv_content = """param1,param2,param3,param4
10,3.14,hello,
20,6.28,world,
"""
    csv_path = os.path.join(self.test_dir.name, "test.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
      f.write(csv_content)

    cases = csv_loader.load_cases_from_csv(csv_path)

    self.assertLen(cases, 2)
    self.assertEqual(
        cases[0],
        {"param1": 10, "param2": 3.14, "param3": "hello"},
    )
    self.assertNotIn("param4", cases[0])
    self.assertEqual(
        cases[1],
        {"param1": 20, "param2": 6.28, "param3": "world"},
    )
    self.assertNotIn("param4", cases[1])

  def test_load_cases_from_csv_empty_path(self):
    """Test loading from an empty or whitespace-only path."""
    with self.assertRaises(ValueError):
      csv_loader.load_cases_from_csv("")
    with self.assertRaises(ValueError):
      csv_loader.load_cases_from_csv("   ")

  def test_load_cases_from_csv_missing_file(self):
    """Test loading from a missing file."""
    with self.assertRaises(FileNotFoundError):
      csv_loader.load_cases_from_csv(
          os.path.join(self.test_dir.name, "nonexistent.csv")
      )

  @mock.patch.object(pd, "read_csv", spec_set=True)
  def test_load_cases_from_csv_valid_remote(self, mock_read_csv):
    """Test loading from a valid remote CSV file."""
    mock_read_csv.return_value = pd.DataFrame({
        "param1": [100],
        "param2": [200],
    })

    cases = csv_loader.load_cases_from_csv("https://example.com/test.csv")

    self.assertLen(cases, 1)
    self.assertEqual(cases[0], {"param1": 100, "param2": 200})
    mock_read_csv.assert_called_once_with("https://example.com/test.csv")

  @mock.patch.object(pd, "read_csv", spec_set=True)
  def test_load_cases_from_csv_os_error_raises_value_error(self, mock_read_csv):
    """Verifies that underlying OSErrors from pd.read_csv are translated to ValueError."""
    mock_read_csv.side_effect = OSError("Simulated I/O failure")

    with self.assertRaisesRegex(
        ValueError, "Failed to parse CSV.*Simulated I/O failure"
    ):
      csv_loader.load_cases_from_csv("https://example.com/test.csv")
    mock_read_csv.assert_called_once_with("https://example.com/test.csv")

  def test_load_cases_from_csv_type_inference(self):
    """Test type inference for all supported types."""
    csv_content = """int_param,float_param,str_param,none_param
1,1.1,one,
-2,-2.2,two,
"""
    csv_path = os.path.join(self.test_dir.name, "test_types.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
      f.write(csv_content)

    cases = csv_loader.load_cases_from_csv(csv_path)

    self.assertLen(cases, 2)
    # Row 1
    self.assertIsInstance(cases[0]["int_param"], int)
    self.assertEqual(cases[0]["int_param"], 1)
    self.assertIsInstance(cases[0]["float_param"], float)
    self.assertEqual(cases[0]["float_param"], 1.1)
    self.assertIsInstance(cases[0]["str_param"], str)
    self.assertEqual(cases[0]["str_param"], "one")
    self.assertNotIn("none_param", cases[0])

    # Row 2
    self.assertIsInstance(cases[1]["int_param"], int)
    self.assertEqual(cases[1]["int_param"], -2)
    self.assertIsInstance(cases[1]["float_param"], float)
    self.assertEqual(cases[1]["float_param"], -2.2)
    self.assertIsInstance(cases[1]["str_param"], str)
    self.assertEqual(cases[1]["str_param"], "two")
    self.assertNotIn("none_param", cases[1])

  def test_load_cases_from_csv_parsing_and_normalization(self):
    """Test standard RFC 4180 parsing with stripped headers and type preservation."""
    csv_content = (
        "  name , in_dtype  ,  m  , b1, b2, sparse_col, rate \n"
        "all_reduce,bfloat16,1024,True,False,512,1.5\n"
        "matmul,float32,2048,False,True,,2.5\n"
        "swiglu,bfloat16,0,False,False,0,0.0\n"
    )
    csv_path = os.path.join(self.test_dir.name, "test_normalize.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
      f.write(csv_content)

    cases = csv_loader.load_cases_from_csv(csv_path)

    self.assertLen(cases, 3)

    # Row 0: headers stripped, int/bool/float/str parsed correctly
    self.assertEqual(
        cases[0],
        {
            "name": "all_reduce",
            "in_dtype": "bfloat16",
            "m": 1024,
            "b1": True,
            "b2": False,
            "sparse_col": 512,
            "rate": 1.5,
        },
    )
    self.assertIsInstance(cases[0]["name"], str)
    self.assertIsInstance(cases[0]["in_dtype"], str)
    self.assertIsInstance(cases[0]["m"], int)
    self.assertIsInstance(cases[0]["sparse_col"], int)
    self.assertIsInstance(cases[0]["rate"], float)
    self.assertIs(cases[0]["b1"], True)
    self.assertIs(cases[0]["b2"], False)

    # Row 1: sparse cell omitted, int/bool/float/str preserved
    self.assertEqual(
        cases[1],
        {
            "name": "matmul",
            "in_dtype": "float32",
            "m": 2048,
            "b1": False,
            "b2": True,
            "rate": 2.5,
        },
    )
    self.assertNotIn("sparse_col", cases[1])
    self.assertIsInstance(cases[1]["m"], int)
    self.assertIs(cases[1]["b1"], False)
    self.assertIs(cases[1]["b2"], True)
    self.assertIsInstance(cases[1]["rate"], float)

    # Row 2: 0 and False values preserved (not discarded as falsy)
    self.assertEqual(
        cases[2],
        {
            "name": "swiglu",
            "in_dtype": "bfloat16",
            "m": 0,
            "b1": False,
            "b2": False,
            "sparse_col": 0,
            "rate": 0.0,
        },
    )
    self.assertEqual(cases[2]["m"], 0)
    self.assertIsInstance(cases[2]["m"], int)
    self.assertEqual(cases[2]["sparse_col"], 0)
    self.assertIsInstance(cases[2]["sparse_col"], int)
    self.assertEqual(cases[2]["rate"], 0.0)
    self.assertIsInstance(cases[2]["rate"], float)
    self.assertIs(cases[2]["b1"], False)
    self.assertIs(cases[2]["b2"], False)

  def test_load_cases_from_csv_empty_and_malformed(self):
    """Test handling of 0-byte, headers-only, all-empty rows, and malformed CSVs."""
    # 0-byte file
    empty_path = os.path.join(self.test_dir.name, "empty.csv")
    with open(empty_path, "w", encoding="utf-8") as f:
      f.write("")
    with self.assertRaises(ValueError):
      csv_loader.load_cases_from_csv(empty_path)

    # Headers only
    headers_path = os.path.join(self.test_dir.name, "headers_only.csv")
    with open(headers_path, "w", encoding="utf-8") as f:
      f.write("m,n,k\n")
    with self.assertRaises(ValueError):
      csv_loader.load_cases_from_csv(headers_path)

    # All-empty rows
    empty_rows_path = os.path.join(self.test_dir.name, "empty_rows.csv")
    with open(empty_rows_path, "w", encoding="utf-8") as f:
      f.write("param1,param2\n,\n,\n")
    with self.assertRaises(ValueError):
      csv_loader.load_cases_from_csv(empty_rows_path)

    # Malformed CSV
    malformed_path = os.path.join(self.test_dir.name, "malformed.csv")
    with open(malformed_path, "w", encoding="utf-8") as f:
      f.write('col1,col2\n"unclosed quote,value\nanother line\n')
    with self.assertRaises(ValueError):
      csv_loader.load_cases_from_csv(malformed_path)

    # Mixed empty rows: valid rows extracted
    mixed_rows_path = os.path.join(self.test_dir.name, "mixed_rows.csv")
    with open(mixed_rows_path, "w", encoding="utf-8") as f:
      f.write("param1,param2\n,\n100,200\n,\n")
    cases = csv_loader.load_cases_from_csv(mixed_rows_path)
    self.assertLen(cases, 1)
    self.assertEqual(cases[0], {"param1": 100, "param2": 200})


if __name__ == "__main__":
  absltest.main()
