#!/bin/bash

required_chip="tpu7x"
required_topologies=("2x2x1")

echo "Checking for required GKE TPU configurations..."
echo "Required TPU Type: ${required_chip}"
echo "-----------------------------------------------------------------"

all_found=true

for topology in "${required_topologies[@]}"; do
  echo -n "Checking for TPU topology '${topology}' with type '${required_chip}': "

  matching_nodes=$(kubectl get nodes -l cloud.google.com/gke-tpu-topology=${topology},cloud.google.com/gke-tpu-accelerator=${required_chip} -o custom-columns=NAME:.metadata.name --no-headers 2>/dev/null)

  if [[ -n "${matching_nodes}" ]]; then
    echo "FOUND"
  else
    echo "MISSING"
    all_found=false
  fi
done

echo "-----------------------------------------------------------------"

if [[ "${all_found}" = true ]]; then
  echo "SUCCESS: All required TPU configurations (topology + type) are present in the cluster."
  exit 0
else
  echo "FAILURE: One or more required TPU configurations are missing."
  exit 1
fi
