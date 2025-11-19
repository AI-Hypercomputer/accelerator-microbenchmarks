# JAX Multi-Host TPU Microbenchmark Runner

This script automates the deployment and execution of JAX-based microbenchmarks on Google Cloud TPU multi-host environments. It fetches benchmark source code from a Git repository, sets up dependencies including a Python virtual environment on each TPU worker, runs the benchmark suite in parallel, and copies results back to the local machine.

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and authenticated on your local machine or Cloudtop.
- A Google Cloud project with a multi-host TPU slice provisioned (e.g., v4-8, v5p-16).
- SSH access to the TPU nodes configured. The script uses `~/.ssh/google_compute_engine` by default for authentication with `gcloud compute tpus tpu-vm ssh`.
- If `GIT_REPO_URL` points to a private repository, ensure authentication is configured. For SSH URLs (`git@host:repo.git`), ensure your SSH key is added to the remote Git service and agent forwarding is working. For HTTPS, use a URL with an embedded token or ensure the TPU service account has access permissions.

## Configuration

All configuration is handled by environment variables set in the header of the `run_jax_git.sh` script. Modify this section to match your GCP environment and benchmark repository.

```bash
#!/bin/bash

# --- TPU Configuration ---
export PROJECT="my-gcp-project-id"
export ZONE="us-central2-b"
export TPU_NAME="my-tpu-vm-name"

# --- Git Configuration ---
export GIT_REPO_URL="https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git"
export GIT_BRANCH="reproduce"

# --- Benchmark Configuration ---
export LIBTPU_VERSION="0.0.26.dev20251022"

```

## Usage

1.  Ensure your SSH key for GCP is loaded into `ssh-agent`:
    ```bash
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/google_compute_engine
    ```
2.  Make the script executable:
    ```bash
    chmod +x jax_multihost.sh
    ```
3.  Execute the script:
    ```bash
    ./jax_multihost.sh
    ```

## How it Works

The script performs the following steps sequentially across all TPU workers:

1.  **Git Clone**: Clones the repository specified by `GIT_REPO_URL` and `GIT_BRANCH` into `~/accelerator-microbenchmarks` on each worker.
    ```bash
    export GIT_TERMINAL_PROMPT=0 && GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone --depth 1 -b ${GIT_BRANCH} ${GIT_REPO_URL} accelerator-microbenchmarks
    ```
2.  **Environment Setup**:
    *   Installs `python3-venv` using `apt-get`.
    *   Creates a Python virtual environment named `micro_venv`.
    *   Activates `micro_venv` and installs dependencies from `requirements.txt`.
    *   Installs the specified `libtpu` version via `pip`.
    ```bash
    cd accelerator-microbenchmarks && \
    sudo apt-get update && sudo apt-get install -y python3-venv && \
    python3 -m venv micro_venv && \
    source micro_venv/bin/activate && \
    pip install -r requirements.txt && \
    pip install -U --pre libtpu==${LIBTPU_VERSION} -f https://storage.googleapis.com/libtpu-wheels/index.html
    ```
3.  **Benchmark Execution**: Activates the `micro_venv` environment and runs `Ironwood/scripts/run_ici_microbenchmark.sh` with appropriate environment variables for JAX and XLA debugging.
    ```bash
    cd accelerator-microbenchmarks && \
    source micro_venv/bin/activate && \
    export JAX_PLATFORMS=tpu,cpu && \
    export TPU_VMODULE=singleton_tpu_system_manager=10,tpu_version_flag=10,device_util=10,device_scanner=10,mesh_builder=10,master=10 && \
    export XLA_IR_DEBUG=1 && \
    export XLA_HLO_DEBUG=1 && \
    sh Ironwood/scripts/run_ici_microbenchmark.sh
    ```
