#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_ici_microbenchmark.sh


topology=$1
demo=$2

if [ -z "$topology" ]; then
  echo "Error: Topology argument is missing."
  echo "Usage: $0 <topology> [true|false]"
  exit 1
fi

CONFIG_DIR="Ironwood/configs/collectives/$topology"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Error: Configuration directory '$CONFIG_DIR' does not exist."
  exit 1
fi

for CONFIG_FILE in "$CONFIG_DIR"/*.yaml
do
  CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
  
  echo "--- Starting benchmark for ${CONFIG_NAME} ---"
  
  # Run the python script and wait for it to complete
  if [ "$demo" = "true" ]; then
    python3 Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}" --demo
  else
    python3 Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"
  fi
  
  echo "--- Finished benchmark for ${CONFIG_NAME} ---"
done