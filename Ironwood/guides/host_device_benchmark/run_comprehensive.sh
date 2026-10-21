#!/bin/bash
set -e

# Change to the root of the Ironwood directory assuming this script is run from anywhere
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IRONWOOD_DIR="$(dirname $(dirname "$DIR"))"
cd "$IRONWOOD_DIR"

echo "Running comprehensive benchmarks across 1, 2, and 8 devices..."
python3 src/run_benchmark.py --config configs/host_device/comprehensive_experiments.yaml

echo "Running 8-device benchmark with numactl interleaving..."
numactl --interleave=all python3 src/run_benchmark.py --config configs/host_device/comprehensive_8dev_experiments.yaml

echo "Benchmarks completed successfully."
