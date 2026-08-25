# JAX Benchmarks

A comprehensive, extensible framework for profiling and benchmarking JAX
operations on TPUs and other hardware accelerators.

## Overview

The `accelerator_microbenchmarks` package provides a structured way to measure
the performance (latency, throughput, memory bandwidth) of various JAX
primitives and composite operations. It includes built-in benchmarks for:

*   **Compute Operations:** Generalized GEMMs, Matrix Multiplications,
Attention mechanisms.
*   **Collective Communications:** `psum`, `all_gather`, `all_to_all`, `reduce_scatter` (using `shard_map`).
*   **Memory Bandwidth:** HBM bandwidth profiling.

The framework is highly configurable via YAML files, allowing users to define
parameter sweeps, warm-up iterations, and matrix shapes without modifying Python
code.

## Directory Structure

```text
accelerator_microbenchmarks/
├── configs/            # YAML configuration files (e.g., sample.yaml, hbm_sweep.yaml)
├── docs/               # Documentation (README, DEVELOPERS, DESIGN, RATIONALE)
│   ├── DESIGN.md
│   ├── DEVELOPERS.md
│   └── RATIONALE.md
├── pyproject.toml
├── results/            # Can create output directory for benchmark metrics (JSON, CSV)
├── src/
│   └── accelerator_microbenchmarks/
│       ├── benchmarks/ # Concrete benchmark implementations (collectives, matmul, etc.)
│       ├── core/       # Framework core (BaseBenchmark, registry, config parsing)
│       └── cli.py      # Entry point for running benchmarks (tpums)
├── README.md
<!-- copybara:strip_begin(internal) -->
└── tools/              # Utility scripts (e.g., syncing results)
<!-- copybara:strip_end -->
```

## How It Works

1.  **Configuration:** A YAML file (e.g., `configs/sample.yaml`) defines global
settings (number of runs, warmup tries) and a list of benchmarks to execute. It
supports parameter "sweeps" to automatically test a range of dimensions or mesh
shapes.
2.  **Registry:** The `tpums` CLI parses the YAML and looks up the
requested benchmark names in a central registry.
3.  **Execution:** For each configuration permutation, the framework
instantiates the benchmark, calls its `setup()`, runs `warmup_tries` iterations,
and then executes `num_runs` iterations while capturing precise timing metrics.

## Installation

You can install the package locally via `pip`. It is recommended to do this in a
dedicated virtual environment:

```bash
pip install .
```

For editable mode (useful when developing custom benchmarks):

```bash
pip install -e .
```

## Running Benchmarks

<!-- copybara:strip_begin(internal) -->
### Using XManager (Recommended for TPUs)

A launch script is provided one level up to deploy benchmarks to a TPU slice via
XManager.

```bash
# From third_party/py/accelerator_microbenchmarks/
bash orchestration/launch.sh
```

You can customize `launch.sh` to change the `TOPOLOGY` (e.g., `4x4x4`), the Borg
cell, or the YAML config path.
<!-- copybara:strip_end -->

### Running via CLI

Once installed, you can run benchmarks directly using the `tpums` CLI:

```bash
tpums benchmark run-config configs/sample.yaml
```

<!-- copybara:strip_begin(internal) -->
### Running Locally with Bazel

If you are on a machine with available accelerators or want to test
functionality on CPU, you can run the binary directly via Bazel:

```bash
bazel run //src/accelerator_microbenchmarks:tpums -- \
  benchmark run-config configs/sample.yaml
```
<!-- copybara:strip_end -->

## Adding a New Benchmark

To add a new benchmark, please refer to the detailed instructions in [DEVELOPERS.md](docs/DEVELOPERS.md).


## Configuration Guide (YAML)

The YAML configuration supports discrete values and sweep definitions.

```yaml
global:
  warmup_tries: 2
  num_runs: 5
  dtype: "bfloat16"

benchmarks:
  # 1. Fixed parameters
  - name: my_custom_op
    size: 2048

  # 2. List sweep
  - name: my_custom_op
    sweep:
      size: [1024, 2048, 4096]

  # 3. Geometric/Range sweep
  - name: hbm_bandwidth
    sweep:
      size: 
        start: 1024
        end: 8192
        multiplier: 2  # Will test 1024, 2048, 4096, 8192
```

## Reviewing Results

By default, the benchmark runner aggregates results and writes them to the
`results/` directory as `detailed.json` and `summary.csv`.
<!-- copybara:strip_begin(internal) -->
You can use `tools/sync_results.py` to copy these results out of the XManager
execution environment.
<!-- copybara:strip_end -->

## Note for old users

This code has gone through significant refactoring. In case you are heavily
dependent on the old version of the code, you can pin your dependencies to this
tag (v1.1-legacy)