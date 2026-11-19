"""Unit tests for core/units.py."""

import copy
import dataclasses
import json

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import units


class ParseByteSizeTest(parameterized.TestCase):
  """Tests for parse_byte_size()."""

  @parameterized.parameters(
      ("0B", 0),
      ("4B", 4),
      ("1536B", 1536),
      ("4KiB", 4096),
      ("1024KiB", 1 << 20),
      ("256MiB", 256 << 20),
      ("1GiB", 1 << 30),
      ("2TiB", 2 << 40),
  )
  def test_parses_iec_units(self, value, expected):
    """Verifies every IEC unit converts to the exact byte count."""
    self.assertEqual(units.parse_byte_size(value), expected)

  @parameterized.parameters(4096, 4.0, True, None)
  def test_rejects_non_strings(self, value):
    """Verifies non-string values (e.g. YAML ints) are rejected."""
    with self.assertRaisesRegex(ValueError, "string with a unit"):
      units.parse_byte_size(value)

  def test_rejects_missing_unit_with_hint(self):
    """Verifies a bare number suggests adding a unit."""
    with self.assertRaisesRegex(ValueError, "has no unit: write '4096B'"):
      units.parse_byte_size("4096")

  @parameterized.parameters(
      ("4K", "4KiB"),
      ("4KB", "4KiB"),
      ("4kB", "4KiB"),
      ("4kib", "4KiB"),
      ("4KIb", "4KiB"),
      ("4M", "4MiB"),
      ("4MB", "4MiB"),
      ("4G", "4GiB"),
      ("4GB", "4GiB"),
      ("4TB", "4TiB"),
      ("4b", "4B"),
  )
  def test_rejects_ambiguous_units_with_suggestion(self, value, suggestion):
    """Verifies ambiguous or mis-cased units suggest the IEC spelling."""
    with self.assertRaisesRegex(
        ValueError, f"Unsupported unit .* Did you mean '{suggestion}'"
    ):
      units.parse_byte_size(value)

  def test_rejects_unknown_unit_without_suggestion(self):
    """Verifies an unrelated unit lists the supported units only."""
    with self.assertRaisesRegex(ValueError, "Unsupported unit 'PiB'") as cm:
      units.parse_byte_size("4PiB")
    self.assertNotIn("Did you mean", str(cm.exception))

  @parameterized.parameters(
      "1.5KiB", "4 KiB", " 4KiB", "4KiB ", "-4KiB", "+4KiB", "KiB", "", "4_096B"
  )
  def test_rejects_malformed_values(self, value):
    """Verifies decimals, spaces, signs and missing numbers are rejected."""
    with self.assertRaisesRegex(ValueError, "Invalid byte size"):
      units.parse_byte_size(value)


@dataclasses.dataclass
class _Holder:
  size: units.ByteSize


class ByteSizeTest(parameterized.TestCase):
  """Tests for the ByteSize value type."""

  def test_keeps_label_as_written(self):
    """Verifies the label is not normalized."""
    size = units.ByteSize("1024KiB")
    self.assertEqual(size, "1024KiB")
    self.assertEqual(str(size), "1024KiB")
    self.assertEqual(size.num_bytes, 1 << 20)

  def test_is_a_str(self):
    """Verifies ByteSize behaves as a plain string in formatting."""
    size = units.ByteSize("4KiB")
    self.assertIsInstance(size, str)
    self.assertEqual(f"run_{size}", "run_4KiB")

  def test_equality_is_string_equality(self):
    """Verifies equal sizes with different labels compare unequal."""
    self.assertNotEqual(units.ByteSize("1024KiB"), units.ByteSize("1MiB"))
    self.assertEqual(
        units.ByteSize("1024KiB").num_bytes, units.ByteSize("1MiB").num_bytes
    )

  def test_construction_is_idempotent(self):
    """Verifies wrapping a ByteSize returns the same object."""
    size = units.ByteSize("4KiB")
    self.assertIs(units.ByteSize(size), size)

  @parameterized.parameters(4096, "4096", "4K", "1.5KiB")
  def test_rejects_invalid_values(self, value):
    """Verifies construction validates the label."""
    with self.assertRaises(ValueError):
      units.ByteSize(value)

  def test_survives_dataclass_and_copy_round_trips(self):
    """Verifies asdict / replace / deepcopy keep a ByteSize, not a dict."""
    holder = _Holder(size=units.ByteSize("4KiB"))
    as_dict = dataclasses.asdict(holder)
    self.assertIsInstance(as_dict["size"], units.ByteSize)
    self.assertEqual(as_dict["size"], "4KiB")
    self.assertIsInstance(dataclasses.replace(holder).size, units.ByteSize)
    self.assertIsInstance(copy.deepcopy(holder.size), units.ByteSize)

  def test_parse_names_the_parameter(self):
    """Verifies ByteSize.parse() prefixes errors with the parameter name."""
    self.assertEqual(
        units.ByteSize.parse("4KiB", param_name="data_size"), "4KiB"
    )
    with self.assertRaisesRegex(
        ValueError, r"^data_size: Unsupported unit .* Did you mean '4KiB'"
    ):
      units.ByteSize.parse("4K", param_name="data_size")

  def test_is_a_param_type(self):
    """Verifies ByteSize opts in to params parsing."""
    self.assertTrue(issubclass(units.ByteSize, constants.ParamType))

  def test_serializes_as_json_string(self):
    """Verifies JSON writes the label as a plain string."""
    self.assertEqual(
        json.dumps({"data_size": units.ByteSize("4KiB")}),
        '{"data_size": "4KiB"}',
    )


if __name__ == "__main__":
  absltest.main()
