#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_ici_microbenchmark.sh


topology=$1
demo=$2

CONFIG_DIR="Ironwood/configs/collectives/$topology"

for CONFIG_FILE in "$CONFIG_DIR"/*.yaml
do
  CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
  
  echo "--- Starting benchmark for ${CONFIG_NAME} ---"
  
  # Run the python script and wait for it to complete
  if $demo
  then
    python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}" --demo
  else
    python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"
  fi

  wait 
  
  echo "--- Finished benchmark for ${CONFIG_NAME} ---"
done