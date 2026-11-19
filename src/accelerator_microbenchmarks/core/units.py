"""Byte sizes written with IEC binary units, e.g. '4KiB'."""

import re
from typing import Any, Final, Mapping, Self

from accelerator_microbenchmarks.core import constants

# IEC binary units accepted in size strings (1 KiB = 1024 B).
IEC_UNITS: Final[Mapping[str, int]] = {
    "B": 1,
    "KiB": 1 << 10,
    "MiB": 1 << 20,
    "GiB": 1 << 30,
    "TiB": 1 << 40,
}
_SIZE_RE: Final = re.compile(r"(?P<num>[0-9]+)(?P<unit>[A-Za-z]+)")
# Unit letter -> IEC unit to suggest for ambiguous or misspelled units, e.g.
# 'K', 'KB' or 'kib' -> 'KiB'. 'K'/'KB' mean 1024 in some tools and 1000 in
# others.
_SUGGESTED_UNITS: Final[Mapping[str, str]] = {
    "": "B",
    "k": "KiB",
    "m": "MiB",
    "g": "GiB",
    "t": "TiB",
}
_UNIT_NAMES: Final = ", ".join(IEC_UNITS)


def parse_byte_size(value: str) -> int:
  """Returns the number of bytes in `value`, a size with a required IEC unit.

  Args:
    value: A whole number followed by one of `IEC_UNITS`, with no spaces, e.g.
      '4096B', '4KiB' or '256MiB'. Units are case-sensitive.

  Returns:
    The size in bytes.

  Raises:
    ValueError: If `value` is not a string (e.g. an int), has no unit, uses an
      ambiguous or unknown unit (e.g. '4K', '4KB', '4kib'), or is not a whole
      number without spaces or a sign (e.g. '1.5KiB', '4 KiB', '-4KiB').
  """
  if not isinstance(value, str):
    raise ValueError(
        f"Byte size must be a string with a unit (e.g. '4KiB'), got {value!r}."
    )
  if value.isascii() and value.isdigit():
    raise ValueError(
        f"Byte size {value!r} has no unit: write '{value}B', or use a larger"
        " unit such as 'KiB' or 'MiB'."
    )
  match = _SIZE_RE.fullmatch(value)
  if match is None:
    raise ValueError(
        f"Invalid byte size {value!r}: expected a whole number followed by one"
        f" of {_UNIT_NAMES} with no spaces, e.g. '4KiB' or '1536B'."
    )
  num, unit = int(match["num"]), match["unit"]
  if unit not in IEC_UNITS:
    suggestion = _suggest_unit(unit)
    hint = f" Did you mean '{num}{suggestion}'?" if suggestion else ""
    raise ValueError(
        f"Unsupported unit {unit!r} in byte size {value!r}: use one of"
        f" {_UNIT_NAMES} (1 KiB = 1024 B).{hint}"
    )
  return num * IEC_UNITS[unit]


def _suggest_unit(unit: str) -> str | None:
  """Returns the IEC unit to suggest for a misspelled `unit`, if any."""
  return _SUGGESTED_UNITS.get(unit.lower().removesuffix("ib").removesuffix("b"))


class ByteSize(constants.ParamType, str):
  """A byte size with a required IEC unit, kept exactly as written.

  A `str` subclass, so it prints and serializes (CSV, JSON, YAML, run IDs) as
  the label the user wrote, e.g. '4KiB' or '1024KiB'. Construction validates
  the label (see `parse_byte_size()`). Equality is string equality, so
  '1024KiB' != '1MiB'; compare sizes with `num_bytes`.

  Deliberately not a dataclass: `dataclasses.asdict()` would turn a nested
  dataclass into a dict.
  """

  __slots__ = ()

  def __new__(cls, value: str) -> Self:
    if isinstance(value, cls):
      return value
    parse_byte_size(value)
    return super().__new__(cls, value)

  @classmethod
  def parse(cls, raw: Any, *, param_name: str) -> Self:
    """Returns `raw` as a ByteSize, with the parameter name in errors."""
    try:
      return cls(raw)
    except ValueError as exc:
      raise ValueError(f"{param_name}: {exc}") from exc

  @property
  def num_bytes(self) -> int:
    """The size in bytes."""
    return parse_byte_size(self)
