#!/bin/bash

# Run command: sh script/run_ici_microbenchmark.sh 

# Loop through each config name in the list

# CONFIG_NAMES="all_gather_1d all_gather_2d all_gather_3d"
CONFIG_NAMES="all_gather_2d "

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="configs/ironwood/ici_collectives/${CONFIG}.yaml"
  
  echo "--- Starting benchmark for ${CONFIG} ---"
  
  # Run the python script and wait for it to complete
  python src/run_benchmark.py --config="${CONFIG_FILE}"
  
  echo "--- Finished benchmark for ${CONFIG} ---"
done