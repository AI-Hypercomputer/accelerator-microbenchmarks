#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_ici_microbenchmark.sh

CONFIG_NAMES="all_gather_1d all_reduce_1d"

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="Ironwood/configs/collectives/${CONFIG}.yaml"
  
  echo "--- Starting benchmark for ${CONFIG} ---"
  
  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"
  # python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"  --output_path=gs://rahulasharma-bucket/jax_mb

  wait 
#   --output_path=gs://rahulasharma-bucket/jax_mb
  
  echo "--- Finished benchmark for ${CONFIG} ---"
done