#!/bin/bash

CONFIG_NAMES="add_bf16 add_fp8 silu_mul_bf16 silu_mul_fp8 rmsnorm_bf16 sigmoid_bf16 sigmoid_fp8"

# Loop through each config name in the list
for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="configs/inference_2d/${CONFIG}.yaml"
  
  echo "--- Starting benchmark for ${CONFIG} ---"
  
  # Run the python script and wait for it to complete
  python src/run_benchmark.py --config="${CONFIG_FILE}"
  
  echo "--- Finished benchmark for ${CONFIG} ---"
done