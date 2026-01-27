# Ironwood Automation Tool

This directory contains the automation scripts for running TPU microbenchmarks. The tool simplifies the process of launching multiple benchmark jobs, waiting for their completion, and aggregating the results into a unified format.

## Prerequisites

Before running the automation script, ensure the following requirements are met:

1.  **Node Pool Topology**: The script expects specific TPU node pools to be available in your cluster.
    *   The `check_node_pool_setup.sh` script validates this.
2.  **GCS Bucket**: You must have a Google Cloud Storage (GCS) bucket for the intermediate and final results.
    *   This can be setup by `gcloud storage buckets create gs://my-unique-bucket-name --location=us-central1`
3.  **Kubectl**: Ensure `kubectl` is configured and connected to your GKE cluster.

## User Journey

1.  **Clone & Checkout Branch**.
    ```bash
    git clone https://github.com/google/accelerator-microbenchmarks.git
    cd accelerator-microbenchmarks
    git checkout tpu7x-auto
    ```

2.  **Setup Environment**: Ensure your node pools are set up and you have prepared a GCS bucket.

3.  **Run Automation Script**:
    The main script is `automation_launch.sh`. You need to set the `GCS_BUCKET_ROOT_DIR` environment variable before running it.

    ```bash
    # Replace with your actual bucket path (must start with gs://)
    export GCS_BUCKET_ROOT_DIR="gs://your-bucket-name/automation_results"
    
    # Run the launch script
    bash Ironwood/guides/automation/automation_launch.sh
    ```

4.  **Retrieve Results**:
    After the script completes, the final aggregated TSV files will be available in your GCS bucket. The script generates a timestamped directory for each run.
    *   Check the script output for the exact path: `The intermediate result will be written to gs://...`
    *   Look for the `final` directory under that path (e.g., `gs://your-bucket/automation_results/<timestamp>/final`).
