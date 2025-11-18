#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_inference_compute_microbenchmark.sh

CONFIG_NAMES="add_bf16 add_fp8 rmsnorm_bf16 sigmoid_bf16 sigmoid_fp8 silu_mul_bf16 silu_mul_fp8"

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="Ironwood/configs/inference/${CONFIG}.yaml"
  
  echo "--- Starting inference benchmark for ${CONFIG} ---"
  
  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"
  
  echo "--- Finished benchmark for ${CONFIG} ---"
done