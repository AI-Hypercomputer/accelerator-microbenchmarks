#!/bin/bash

# Default values
CONFIG_DIR="Ironwood/configs/host_device"
SPECIFIC_CONFIG=""
INTERLEAVED=false
NUMACTL_BIND=false

# Helper function for usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --config <path>       Path to specific config file (optional)"
    echo "  --numactl_bind        Run with numactl --cpunodebind=0 --membind=0"
    echo "  --interleaved         Run with numactl --interleave=all"
    echo "  --help                Show this help message"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) SPECIFIC_CONFIG="$2"; shift ;;
        --interleaved) INTERLEAVED=true ;;
        --numactl_bind) NUMACTL_BIND=true ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "--- Starting Host-Device Transfer Benchmark (H2D/D2H) ---"
echo "********************************************************"
echo "WARNING: This benchmark is currently a WORK IN PROGRESS"
echo "********************************************************"
echo ""
echo "Configuration:"
echo "    Interleaved: $INTERLEAVED"
echo "    numactl Bound: $NUMACTL_BIND"
echo ""

if [ -n "$SPECIFIC_CONFIG" ]; then
    CONFIGS=("$SPECIFIC_CONFIG")
else
    # Use nullglob to handle case where no files match (though unlikely here)
    shopt -s nullglob
    CONFIGS=("$CONFIG_DIR"/*.yaml)
    shopt -u nullglob
fi

if [[ "$INTERLEAVED" = true && "$NUMACTL_BIND" = true ]]; then
    echo "Only one of --interleaved and --numactl_bind is allowed to be set at once."
    exit 1
fi

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No configuration files found!"
    exit 1
fi

for CONFIG_FILE in "${CONFIGS[@]}"; do
    echo "--- Running Config: $CONFIG_FILE ---"
    CMD="python Ironwood/src/run_benchmark.py --config=${CONFIG_FILE}"

    if [ "$INTERLEAVED" = true ]; then
        if command -v numactl &> /dev/null; then
            echo "Running with numactl --interleave=all"
            numactl --interleave=all $CMD
        else
            echo "Warning: numactl not found. Running without interleaving."
            $CMD
        fi
    elif [ "$NUMACTL_BIND" = true ]; then
        if command -v numactl &> /dev/null; then
            echo "Running with numactl --cpunodebind=0 --membind=0"
            numactl --cpunodebind=0 --membind=0 $CMD
        else
            echo "Warning: numactl not found. Running without binding."
            $CMD
        fi
    else
        $CMD
    fi
    echo "--- Finished Config: $CONFIG_FILE ---"
    echo ""
done

echo "--- All Benchmarks Finished ---"
