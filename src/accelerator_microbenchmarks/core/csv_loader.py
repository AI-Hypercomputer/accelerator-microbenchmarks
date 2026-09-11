"""Utility to load benchmark parameters from CSV files."""

from typing import Any

import pandas as pd


def _is_valid_value(val: Any) -> bool:
  """Returns True if val is not NA/null and not an empty string."""
  if pd.isna(val):
    return False
  if isinstance(val, str) and not val:
    return False
  return True


def load_cases_from_csv(path: str) -> list[dict[str, Any]]:
  """Reads a CSV (local or remote URL) and returns a list of row dicts.

  Supports Google Sheets export links:
  https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=<GID>

  Args:
    path: The path to the CSV file, can be a local path or a URL.

  Returns:
    A list of dictionaries, where each dictionary represents a row in the CSV.
    Column headers are stripped, unpopulated (empty/NaN) cells are omitted,
    and values have inferred types (int, float, bool, or str).

  Raises:
    ValueError: If path is empty, malformed, or fails during fetch/parsing, or
      contains no valid benchmark cases.
    FileNotFoundError: If the local CSV file does not exist.
  """
  if not path or not path.strip():
    raise ValueError(
        f"Invalid path '{path}': path cannot be empty or whitespace only."
    )
  path = path.strip()

  try:
    df = pd.read_csv(path).convert_dtypes()
  except FileNotFoundError:
    raise
  except (
      pd.errors.EmptyDataError,
      pd.errors.ParserError,
      OSError,
      UnicodeDecodeError,
  ) as e:
    raise ValueError(f"Failed to parse CSV from '{path}': {e}") from e

  if df.empty:
    raise ValueError(f"CSV file at '{path}' contains no data rows.")

  # Clean column headers
  df.columns = df.columns.str.strip()

  cases = []
  for row in df.to_dict(orient="records"):
    if any(_is_valid_value(v) for v in row.values()):
      cases.append({
          k: v.item() if hasattr(v, "item") else v
          for k, v in row.items()
          if _is_valid_value(v)
      })

  if not cases:
    raise ValueError(f"CSV file at '{path}' contains no valid benchmark cases.")

  return cases
