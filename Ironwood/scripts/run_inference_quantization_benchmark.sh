#!/bin/bash

# Run command: sh ./Ironwood/scripts/run_inference_quantization_benchmark.sh
# NOTE: currently the improved FP4 quantization logic isn't available in the stable release of libtpu yet
pip install --pre libtpu==0.0.31.dev20251124+nightly -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
CONFIG_NAMES="quantization"

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="Ironwood/configs/inference/${CONFIG}.yaml"

  echo "--- Starting benchmark for ${CONFIG} ---"

  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"

  echo "--- Finished benchmark for ${CONFIG} ---"
done