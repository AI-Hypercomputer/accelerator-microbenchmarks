#!/usr/bin/env bash

######################################################################
#                            USER INPUT
######################################################################
export GCS_BUCKET_ROOT_DIR=""

yaml_names=("tpu7x-16-hbm.yaml")
job_names=("tpu7x-16-hbm")

######################################################################
#                        VALIDATION & SETUP
######################################################################

if [[ -z "${GCS_BUCKET_ROOT_DIR}" || "${GCS_BUCKET_ROOT_DIR}" != "gs://"* ]]; then
  echo "Error: GCS_BUCKET_ROOT_DIR must be set and start with gs://"
  exit 1
fi
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
export GCS_PATH="${GCS_BUCKET_ROOT_DIR}/${TIMESTAMP}"
echo "The intermediate result will be written to ${GCS_PATH}"

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
if ! bash "${SCRIPT_DIR}/check_node_pool_setup.sh"; then
  exit 1
fi

######################################################################
#                 LAUNCH JOBS & WAIT FOR COMPLETION
######################################################################

for yaml_file in "${yaml_names[@]}"; do
    echo "Launch job: ${yaml_file}"
    envsubst '${GCS_PATH}' < ${yaml_file} | kubectl apply -f -
done

for job_name in "${job_names[@]}"; do
    kubectl wait --for=condition=complete job/${job_name} --timeout=1800s
    kubectl delete job ${job_name}
done
kubectl apply -f aggregator.yaml