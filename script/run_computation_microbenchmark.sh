#!/bin/bash

# Run command: sh script/run_computation_mircrobenchmark.sh 

# CONFIG_NAMES="gemm_simple gemm"
# CONFIG_NAMES="all_gather_1d all_gather_2d all_gather_3d"
# CONFIG_NAMES="all_gather_1d all_gather_2d all_gather_3d"

# CONFIG_NAMES="gemm_simple gemm gemm_accum quantization transpose_quantization swiglu_fwd swiglu_bwd rmsnorm_fwd rmsnorm_bwd add gemm_fp8_rowwise"

# Loop through each config name in the list

CONFIG_NAMES="gemm"

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="configs/ironwood/${CONFIG}.yaml"
  
  echo "--- Starting benchmark for ${CONFIG} ---"
  
  # Run the python script and wait for it to complete
  python src/run_benchmark.py --config="${CONFIG_FILE}"
  
  echo "--- Finished benchmark for ${CONFIG} ---"
done