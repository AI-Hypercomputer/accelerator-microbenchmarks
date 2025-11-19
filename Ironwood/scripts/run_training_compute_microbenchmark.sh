#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_training_compute_microbenchmark.sh

CONFIG_NAMES="gemm_multiple_run_more gemm_simple gemm gemm_accum gemm_fp8_rowwise add quantization transpose_quantization"

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="Ironwood/configs/training/${CONFIG}.yaml"
  
  echo "--- Starting benchmark for ${CONFIG} ---"
  
  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"
  
  echo "--- Finished benchmark for ${CONFIG} ---"
done