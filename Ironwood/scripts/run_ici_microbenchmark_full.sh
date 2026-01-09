#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_ici_microbenchmark.sh




# CONFIG_NAMES='reduce_scatter_1d reduce_scatter_2d all_gather_3d all_reduce_3d all_to_all_3d all_gather_2d all_reduce_2d all_to_all_2d all_gather_1d all_reduce_1d all_to_all_1d'

CONFIG_NAMES='all_gather_1d all_reduce_1d all_to_all_1d reduce_scatter_1d'

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="Ironwood/configs/collectives/8x8x16/${CONFIG}.yaml"
  
  echo "--- Starting benchmark for ${CONFIG} ---"
  
  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"

  wait 
  
  echo "--- Finished benchmark for ${CONFIG} ---"
done