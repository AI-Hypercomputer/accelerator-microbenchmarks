# Host-Device Benchmark Guide

This directory contains instructions and a script to run comprehensive Host-to-Device (H2D) and Device-to-Host (D2H) microbenchmarks on Cloud TPUs.

## Overview

The benchmarks measure the transfer bandwidth for various configurations:
- **Transfer Strategies**: `simple`, `pipelined`, `pinned_memory`
- **Data Sizes**: Ranging from 1 MiB to 16,384 MiB.
- **Input Types**: Replicating inputs using `numpy` arrays or pre-allocated `jax` arrays.
- **Device Counts**: Scaling across 1, 2, and 8 TPU devices.
- **NUMA Settings**: Testing the impact of `--interleave=all` with `numactl`.

## How to Run

A convenience script `run_comprehensive.sh` is provided. It executes two suites of tests sequentially from the `Ironwood/` directory:

1. **Comprehensive Suite**: Sweeps through all transfer strategies, data sizes, input types, and device configurations (1, 2, and 8 devices) without any specific NUMA configuration.
2. **8-Device NUMA Suite**: Executes an 8-device specific sweep while enforcing `numactl --interleave=all` at the process level to balance memory allocations across NUMA nodes, heavily improving the pipelined D2H transfer bottleneck on multi-chip architectures.

### Execution

Simply execute the script on your TPU VM:

```bash
bash run_comprehensive.sh
```

## Configuration Files

The executed configurations are located at:
- `Ironwood/configs/host_device/comprehensive_experiments.yaml`
- `Ironwood/configs/host_device/comprehensive_8dev_experiments.yaml`

Refer to these files to adjust the tested parameters or trace directories.

## Analyzing Results

Resulting logs and TSV files will be exported to the directory specified within the configurations (default typically output to the console and/or a timestamped TSV). You can use simple Pandas scripts to analyze and extract the max P50 bandwidths.
