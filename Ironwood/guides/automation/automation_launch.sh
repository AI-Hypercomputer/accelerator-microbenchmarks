#!/usr/bin/env bash

######################################################################
#                            USER INPUT
######################################################################
export GCS_PATH=""

yaml_names=("tpu7x-2x2x1-hbm.yaml" "tpu7x-2x2x2-hbm.yaml")
job_names=("tpu7x-8-hbm" "tpu7x-16-hbm")

######################################################################
#                        VALIDATION & SETUP
######################################################################

if [[ -z "${GCS_PATH}" || "${GCS_PATH}" != "gs://"* ]]; then
  echo "Error: GCS_PATH must be set and start with gs://"
  exit 1
fi

echo "The intermediate result will be written to ${GCS_PATH}"

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if ! bash "${SCRIPT_DIR}/check_node_pool_setup.sh"; then
  exit 1
fi

######################################################################
#                 LAUNCH JOBS & WAIT FOR COMPLETION
######################################################################

length=${#yaml_names[@]}
for (( i=0; i<length; i++ )); do
    yaml_file=${yaml_names[${i}]}
    export JOB_NAME=${job_names[${i}]}
    echo "Launch job: ${yaml_file}"
    envsubst '${JOB_NAME} ${GCS_PATH}' < ${yaml_file} | kubectl apply -f -
    (
        echo "Job launched successfully!"
        if kubectl wait --for=condition=complete job/${JOB_NAME} --timeout=1800s &> /dev/null; then
            echo "Job from ${yaml_file} is completed!"
        else
            echo "Job from ${yaml_file} failed!"
            
        fi
        envsubst '${JOB_NAME} ${GCS_PATH}' < ${yaml_file} | kubectl delete -f -
    ) &
done

echo "All jobs dispatched. Waiting for results..."
wait
echo "All processing done."

kubectl apply -f aggregator.yaml