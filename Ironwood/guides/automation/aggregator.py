import argparse
import os
import glob
import pandas as pd
import gcsfs

def download_from_gcs(bucket_path: str, local_dir: str):
    """
    Downloads the content of the GCS bucket path to a local directory.
    """
    fs = gcsfs.GCSFileSystem()
    gcs_path = bucket_path.replace("gs://", "")
    
    print(f"Downloading from gs://{gcs_path} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    fs.get(gcs_path, local_dir, recursive=True)

def aggregate_results(local_dir: str):
    categories = ["collectives", "hbm", "host_device"]
    directories = {}
    for category in categories:
        directories[category] = glob.glob(f"{local_dir}/*/{category}/*", recursive=True)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download from GCS and aggregate results locally.")
    parser.add_argument("--bucket_path", type=str, required=True, help="The GCS bucket path (gs://...)")
    parser.add_argument("--local_dir", type=str, default="./results", help="Local directory to download and aggregate results.")
    args = parser.parse_args()
    
    download_from_gcs(args.bucket_path, args.local_dir)
    aggregate_results(args.local_dir)
