# TPUMS: Design Document

## 1. Introduction & Motivation

This document details the design and architecture of the
`accelerator-microbenchmarks` (`tpums`) repository, a framework for
high-fidelity, modular, and scalable JAX microbenchmarks. The primary motivation
is to replace ad-hoc benchmarking scripts with a standardized, extensible, and
insight-rich platform for evaluating compute, memory, and interconnect
performance on Cloud TPUs (`tpu7x`, `v6e`).

(See [RATIONALE.md](RATIONALE.md) for a detailed breakdown of motivations and
gaps addressed).

## 2. Core Features

-   **Modular Registry Architecture**: Decorator-based registry
    (`core/registry.py`) and dynamic loader
    (`benchmarks/benchmark_loader.py`) for automatic discovery of benchmarks and
    aliases.
-   **Typed Dataclass Configuration**: Each benchmark binds a typed
    `BaseBenchmarkParams` (or `SingleDtypeBenchmarkParams`) dataclass via
    `Config = <ParamsClass>`, providing automatic bounds validation, enum
    coercion, and dynamic CLI flag generation.
-   **Standardized Benchmark Lifecycle**: `BaseBenchmark` (`core/base.py`)
    enforces a unified execution pipeline across compilation warmup, device
    synchronization (`jax.block_until_ready`), timing loops, XProf trace
    extraction, per-chip metric derivation, and Roofline analysis.
-   **Dual Timing Domains (`wall_clock_*` & `xprof_*`)**: Captures end-to-end
    Python wall-clock latency alongside pure on-device kernel duration parsed
    from XLA / XProf hardware traces (`.xplane.pb`).
-   **Single-Benchmark YAML & 3-Stage Precedence**: Reproducible YAML configs
    supporting a strict 3-stage parameter evaluation order (`params:` ->
    `cases:` / `cases_from_csv:` -> `sweep:`).
-   **Hardware-Aware Roofline Analysis**: Automatic computation of arithmetic
    intensity, device-level compute/memory ceilings, and achieved Roofline
    efficiency (`RooflineMode.COMPUTE`, `RooflineMode.MEMORY_HBM`,
    `RooflineMode.NONE`), supporting both analytical formulas and trace-based
    `jax.experimental.roofline` profiling.
-   **Runtime Hardware Auto-Detection**: Automatically queries active TPU
    generation, chip topology, and per-chip device ratios at runtime
    (`core/platform.py` and `core/system.py`).

## 3. Architecture & Design

### Directory Structure

```text
accelerator_microbenchmarks/
├── configs/                    # YAML configurations and shape CSVs
│   ├── sample_configs/         # Introductory sweeps and validation configs
│   ├── shapes/                 # Predefined matrix shape sweeps (CSV)
│   └── tpu7x/, v6e/            # Hardware-saturating topology configurations
├── docs/                       # Architecture and developer documentation
│   ├── DESIGN.md
│   ├── DEVELOPERS.md
│   └── RATIONALE.md
├── pyproject.toml              # Package build config and `tpums` entry point
├── results/                    # Default destination for CSV and JSON reports
├── tests/                      # Unit, config validation, and hardware tests
└── src/
    └── accelerator_microbenchmarks/
        ├── benchmarks/         # Benchmark implementations (gemm, hbm, etc.)
        ├── core/               # Framework core (base, config, runner, report)
        ├── cli.py              # Resource-action CLI entry point (`tpums`)
        └── op_flags.yaml       # Per-benchmark XLA / LIBTPU_INIT_ARGS mappings
```

### Core Components

-   **`core/base.py` (`BaseBenchmark[TConfig]`)**: Generic abstract base class
    defining the benchmark lifecycle:
    -   `Config`: Typed `BaseBenchmarkParams` subclass declaring configurable
        fields, defaults, and validation metadata.
    -   `setup(self)`: One-time initialization and `@jax.jit` kernel compilation
        using `self.config`.
    -   `generate_inputs(self)`: Device-resident input tensor allocation.
    -   `run_op(self, *args, **kwargs)`: Core JAX kernel invoked during warmup
        and measurement loops.
    -   `get_run_identifier(self)`: Deterministic string identifier for the
        active parameter set.
    -   `get_total_bytes(self)` & `get_arithmetic_intensity(self)`: Required
        abstract methods returning analytical byte traffic and FLOP/byte ratio
        for Roofline modeling.
    -   `get_workload_metadata(self)` & `calculate_metrics(self, times_ms)`:
        Combines static workload metadata, IQR-filtered latency statistics
        (`calculate_latency_stats`), and domain throughput metrics.
    -   `calculate_throughput_metrics(self, latency_ms, prefix)`: Required
        abstract method computing domain-prefixed throughput (`wall_clock_*` and
        `xprof_*`) in `TFLOPS` or `GB/s`.
    -   `derive_chip_metrics(self, metrics)`: Scales per-device throughput to
        per-chip throughput via `hardware_spec.devices_per_chip`.
    -   `apply_roofline_analysis(self, metrics)`: Computes Roofline efficiency
        percentages against hardware specifications.
    -   `run(self)`: Orchestrates mesh setup, warmup iterations, XProf profiler
        context management, measurement timing loops, and result packaging.
-   **`core/config.py`**: Validates single-benchmark YAML files (`benchmark:`
    mapping) and expands parameter combinations across the 3-stage precedence
    pipeline (`params` -> `cases` / `cases_from_csv` -> `sweep`).
-   **`core/runner.py`**: Initializes distributed JAX environments, loads
    benchmark-specific `LIBTPU_INIT_ARGS` from `op_flags.yaml`, executes all
    expanded cases, and writes `summary.csv` and `detailed.json` via
    `core/report.py`.
-   **`cli.py`**: Implements the `tpums` CLI (`tpums platform describe`,
    `tpums benchmark list`, `tpums benchmark run`, and
    `tpums benchmark run-config`).

### Configuration Evaluation Flow

1.  **Load & Validate YAML (`core/config.py`)**: Verify the single top-level
    `benchmark:` block (`name`, `xprof_timing`, `params`, `cases` or
    `cases_from_csv`, `sweep`) and check that `sweep:` keys are disjoint from
    `params:` and case keys.
2.  **Stage 1 — Baseline (`params:`)**: Load shared baseline parameters (and
    optional `model_preset` defaults).
3.  **Stage 2 — Case Overrides (`cases:` or `cases_from_csv:`)**: Overlay
    explicit per-case dictionaries or rows loaded from external CSV files via
    `core/csv_loader.py`.
4.  **Stage 3 — Cartesian Sweep (`sweep:`)**: Expand each case across discrete
    lists or geometric/arithmetic ranges (`start`, `end`, `multiplier` /
    `increase_by`) to produce typed `Config` instances.

## 4. Usage

(See `README.md` at the repository root for comprehensive CLI and config
examples).

```bash
# Install package in editable mode
pip install -e .

# Inspect hardware topology and software versions
tpums platform describe

# Run a single benchmark interactively via CLI flags
tpums benchmark run hbm --xprof_timing --op_type copy --size 134217728 \
    --dtype bfloat16

# Run a multi-case YAML configuration or parameter sweep
tpums benchmark run-config configs/sample_configs/parameter_sweep.yaml \
    --xprof_dir /tmp/tensorboard \
    --output_dir results/
```

## 5. Extensibility

To add a new benchmark or custom operation, follow the step-by-step developer
guide in [DEVELOPERS.md](DEVELOPERS.md).
