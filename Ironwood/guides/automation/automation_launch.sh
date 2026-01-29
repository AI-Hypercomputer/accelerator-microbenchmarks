#!/usr/bin/env bash

######################################################################
#                            USER INPUT
######################################################################
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
export GCS_BUCKET_ROOT_DIR=""

MAX_RETRIES=3
TIMEOUT_SECOND=3600

yaml_names=(
    "tpu7x-2x2x1-hbm.yaml" "tpu7x-2x2x1-host_device.yaml" "tpu7x-2x2x1-gemm.yaml" "tpu7x-2x2x1-collectives.yaml"
    "tpu7x-2x2x2-hbm.yaml" "tpu7x-2x2x2-host_device.yaml" "tpu7x-2x2x2-gemm.yaml" "tpu7x-2x2x2-collectives.yaml"
    "tpu7x-2x2x4-hbm.yaml" "tpu7x-2x2x4-host_device.yaml" "tpu7x-2x2x4-gemm.yaml" "tpu7x-2x2x4-collectives.yaml"
    "tpu7x-2x4x4-hbm.yaml" "tpu7x-2x4x4-host_device.yaml" "tpu7x-2x4x4-gemm.yaml" "tpu7x-2x4x4-collectives.yaml"
)

######################################################################
#                        VALIDATION & SETUP
######################################################################

if [[ -z "${GCS_BUCKET_ROOT_DIR}" || "${GCS_BUCKET_ROOT_DIR}" != "gs://"* ]]; then
  echo "Error: GCS_BUCKET_ROOT_DIR must be set and start with gs://"
  exit 1
fi

echo "The intermediate result will be written to ${GCS_BUCKET_ROOT_DIR}"

required_topologies=($(printf "%s\n" "${yaml_names[@]}" | grep -oE '[0-9]+x[0-9]+x[0-9]+' | sort -u))

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if ! bash "${SCRIPT_DIR}/check_node_pool_setup.sh" "${required_topologies[@]}"; then
  exit 1
fi

for topology in "${required_topologies[@]}"; do
    export TOPOLOGY="${topology}"
    export TPUS=$(echo "${TOPOLOGY}" | sed 's/x/*/g' | bc)
    envsubst '${TOPOLOGY} ${TPUS}' < ${SCRIPT_DIR}/job-queue.yaml | kubectl apply -f -
done

######################################################################
#                 LAUNCH JOBS & WAIT FOR COMPLETION
######################################################################


# Function to wait for a job to complete or fail
wait_for_job_completion() {
    local job_name="$1"
    local timeout="$2"
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))

    while true; do
        current_time=$(date +%s)
        if [[ $current_time -gt $end_time ]]; then
            echo "Timeout waiting for job ${job_name}"
            return 2
        fi

        # Check for Complete condition
        if kubectl get job "${job_name}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null | grep -q "True"; then
            echo "Job ${job_name} completed successfully!"
            return 0
        fi

        # Check for Failed condition
        if kubectl get job "${job_name}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True"; then
            echo "Job ${job_name} FAILED!"
            return 1
        fi

        sleep 5
    done
}

# Function to apply jobs and wait for them to complete
# Returns a list of failed yaml files in the variable FAILED_JOBS
apply_and_wait() {
    local yaml_files=("$@")
    local pids=()
    local job_names_in_batch=()
    FAILED_JOBS=()

    echo "Processing batch of ${#yaml_files[@]} jobs..."

    # Launch all jobs
    for yaml_file in "${yaml_files[@]}"; do
        local filepath="${SCRIPT_DIR}/${yaml_file}"
        # Derive job name: remove .yaml, lowercase, replace _ with -
        local job_name=$(basename "${yaml_file}" .yaml | tr '[:upper:]' '[:lower:]' | tr '_' '-')
        export JOB_NAME="${job_name}"
        export GCS_PATH="${GCS_BUCKET_ROOT_DIR}/${job_name}"
        
        echo "Launching job: ${filepath} (name: ${JOB_NAME})"
        envsubst '${JOB_NAME} ${GCS_PATH}' < "${filepath}" | kubectl apply -f -
        job_names_in_batch+=("${JOB_NAME}")
    done

    # Wait for completion in background
    for i in "${!yaml_files[@]}"; do
        local yaml_file="${yaml_files[$i]}"
        local filepath="${SCRIPT_DIR}/${yaml_file}"
        local job_name="${job_names_in_batch[$i]}"
        export GCS_PATH="${GCS_BUCKET_ROOT_DIR}/${job_name}"
        (
            wait_for_job_completion "${job_name}" ${TIMEOUT_SECOND}
            wait_status=$?

            export JOB_NAME="${job_name}"
            envsubst '${JOB_NAME} ${GCS_PATH}' < "${filepath}" | kubectl delete -f - &> /dev/null
            exit $wait_status
        ) &
        pids+=($!)
    done

    # Collect results
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        if [[ $? -ne 0 ]]; then
            FAILED_JOBS+=("${yaml_files[$i]}")
        fi
    done
}

# Retry loop
current_batch=("${yaml_names[@]}")

for (( retry=1; retry<=MAX_RETRIES; retry++ )); do
    apply_and_wait "${current_batch[@]}"
    
    if [[ ${#FAILED_JOBS[@]} -eq 0 ]]; then
        echo "All jobs completed successfully in Round ${retry}!"
        break
    fi
    
    echo "Round ${retry} finished. ${#FAILED_JOBS[@]} jobs failed."
    current_batch=("${FAILED_JOBS[@]}")
    
    if [[ ${retry} -lt ${MAX_RETRIES} ]]; then
        echo "Retrying failed jobs..."
        echo "========================================"
        echo "$((retry + 1)) / ${MAX_RETRIES}" max retries
        echo "========================================"
    else
        echo "Max retries reached. ¯\_(ツ)_/¯"
    fi
done

echo ""
echo "Jobs completed. Aggregating results..."
echo ""

envsubst '${GCS_BUCKET_ROOT_DIR}' < ${SCRIPT_DIR}/aggregator.yaml | kubectl apply -f -

# Print the failed jobs at the end for better visibility.

if [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
    echo "The following jobs finally failed after ${MAX_RETRIES} rounds:"
    printf '%s\n' "${FAILED_JOBS[@]}"

    echo -e "\nTo retry manually, run:"
    for yaml_file in "${FAILED_JOBS[@]}"; do
        job_name=$(basename "${yaml_file}" .yaml | tr '[:upper:]' '[:lower:]' | tr '_' '-')
        GCS_PATH="${GCS_BUCKET_ROOT_DIR}/${job_name}"
        echo "JOB_NAME=\"${job_name}\" GCS_PATH=\"${GCS_PATH}\" envsubst '\${JOB_NAME} \${GCS_PATH}' < \"${SCRIPT_DIR}/${yaml_file}\" | kubectl apply -f -"
    done
else
    echo "Success! All jobs finished."
fi