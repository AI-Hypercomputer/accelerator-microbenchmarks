#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_throttling_microbenchmark.sh

CONFIG_NAMES="throttling_gemm_large_bf16"

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="Ironwood/configs/training/${CONFIG}.yaml"
  
  echo "--- Starting training benchmark for ${CONFIG} ---"
  
  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"
  
  echo "--- Finished training benchmark for ${CONFIG} ---"
done