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


GCLOUD_COMMAND="gcloud" # or gcloud alpha

run_on_all_workers() {
  local command_script="$1"
  echo "--- Executing on all workers ---"
  echo "${command_script}"

  if ! "${GCLOUD_COMMAND}" compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --worker=all \
    --command="${command_script}" \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    --ssh-flag="-o LogLevel=ERROR" \
    --ssh-flag="-o ForwardAgent=yes" \
    --ssh-flag="-o ExitOnForwardFailure=yes" \
    --ssh-flag="-n" \
    --quiet 2>&1; then
    echo "ERROR: Command failed on one or more workers. See output above for details. Exiting."
    exit 1
  fi
}
# --- SSH Key Setup ---
echo "--- Ensuring ssh-agent is running ---"
eval "$(ssh-agent -s)" || { echo "ERROR: Failed to start ssh-agent. Exiting."; exit 1; }
echo "--- Adding SSH key to agent ---"
ssh-add ~/.ssh/google_compute_engine || { echo "ERROR: Failed to add SSH key to agent. Exiting."; exit 1; }


echo "--- Cloning repository on all workers ---"
GIT_CLONE_COMMAND=$(cat <<EOF
export GIT_TERMINAL_PROMPT=0 && GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone --depth 1 -b ${GIT_BRANCH} ${GIT_REPO_URL} accelerator-microbenchmarks
EOF
)
run_on_all_workers "${GIT_CLONE_COMMAND}"

echo "--- Install dependencies ---"
BENCHMARK_SETUP_COMMAND=$(cat <<EOF
cd accelerator-microbenchmarks && \
sudo apt-get update && sudo apt-get install -y python3-venv && \
python3 -m venv micro_venv && \
source micro_venv/bin/activate && \
pip install -r requirements.txt && \
pip install -U --pre libtpu==${LIBTPU_VERSION} -f https://storage.googleapis.com/libtpu-wheels/index.html && \
ls && \
pwd
EOF
)
run_on_all_workers "${BENCHMARK_SETUP_COMMAND}"

echo "--- Running benchmarks ---"
BENCHMARK_RUN_COMMAND=$(cat <<EOF
cd accelerator-microbenchmarks && \
source micro_venv/bin/activate && \
export JAX_PLATFORMS=tpu,cpu && \
export TPU_VMODULE=singleton_tpu_system_manager=10,tpu_version_flag=10,device_util=10,device_scanner=10,mesh_builder=10,master=10 && \
pwd && \
ls && \
export XLA_IR_DEBUG=1 && \
export XLA_HLO_DEBUG=1 && \
sh Ironwood/scripts/run_ici_microbenchmark.sh
EOF
)
run_on_all_workers "${BENCHMARK_RUN_COMMAND}"
