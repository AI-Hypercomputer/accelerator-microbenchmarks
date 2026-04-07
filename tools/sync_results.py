"""Upload benchmark results to Google Sheets using gcloud credentials."""

import argparse
import json
import os
import subprocess
import pandas as pd
import requests


def get_access_token():
  """Use gcloud to get an access token for the current environment."""
  try:
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()
    return token
  except Exception as e:
    print(f"Error getting gcloud token: {e}")
    return None


def sync_to_sheet(
    csv_path: str, spreadsheet_id: str, range_name: str = "Results!A1"
):
  """Append CSV data to a Google Sheet."""
  token = get_access_token()
  if not token:
    print(
        "Failed to get auth token. Ensure you've run 'gcloud auth login' or"
        " your VM has the correct scopes."
    )
    return

  # Load data
  df = pd.read_csv(csv_path)
  # Convert to list of lists for Sheets API
  values = [df.columns.values.tolist()] + df.values.tolist()

  url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}:append"
  params = {
      "valueInputOption": "USER_ENTERED",
      "insertDataOption": "INSERT_ROWS",
  }
  headers = {"Authorization": f"Bearer {token}", "Content-Type": "json"}
  body = {"values": values}

  try:
    response = requests.post(url, headers=headers, params=params, json=body)
    if response.status_code == 200:
      print(
          f"Successfully synced {len(values)-1} rows to Google Sheet:"
          f" {spreadsheet_id}"
      )
    else:
      print(f"Failed to sync: {response.status_code} - {response.text}")
  except Exception as e:
    print(f"Error during sync: {e}")


def main():
  parser = argparse.ArgumentParser(description="Sync Results to Google Sheets")
  parser.add_argument(
      "--csv", type=str, default="results/summary.csv", help="Source CSV path"
  )
  parser.add_argument(
      "--sheet_id", type=str, required=True, help="Google Spreadsheet ID"
  )
  parser.add_argument(
      "--range",
      type=str,
      default="Sheet1!A1",
      help="Target range (e.g. 'Results!A1')",
  )
  args = parser.parse_args()

  if not os.path.exists(args.csv):
    print(f"CSV file not found: {args.csv}")
    return

  sync_to_sheet(args.csv, args.sheet_id, args.range)


if __name__ == "__main__":
  main()
