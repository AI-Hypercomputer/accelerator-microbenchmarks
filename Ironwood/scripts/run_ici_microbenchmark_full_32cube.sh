#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_ici_microbenchmark.sh



CONFIG_DIR="Ironwood/configs/collectives/8x16x16"

for CONFIG_FILE in "$CONFIG_DIR"/*.yaml
do
  CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
  
  echo "--- Starting benchmark for ${CONFIG_NAME} ---"
  
  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"

  wait 
  
  echo "--- Finished benchmark for ${CONFIG_NAME} ---"
done