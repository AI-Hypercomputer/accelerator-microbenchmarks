#!/usr/bin/env bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

yaml_names=("tpu7x-16-hbm.yaml")
job_names=("tpu7x-16-hbm")

# Fill the target GCS bucket path.
export GCS_BUCKET_ROOT_DIR=""
export GCS_PATH="${GCS_BUCKET_ROOT_DIR}/${TIMESTAMP}"

for yaml_file in "${yaml_names[@]}"; do
    echo "Launch job: ${yaml_file}"
    envsubst '${GCS_PATH}' < ${yaml_file} | kubectl apply -f -
done

for job_name in "${job_names[@]}"; do
    kubectl wait --for=condition=complete job/${job_name} --timeout=1800s
    kubectl delete job ${job_name}
done
kubectl apply -f aggregator.yaml