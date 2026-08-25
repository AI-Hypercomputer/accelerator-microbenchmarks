# JAX Benchmarks: Design Document

## 1. Introduction & Motivation

This document details the design and architecture of the
`accelerator-microbenchmarks` repository, a framework for high-fidelity,
modular, and scalable JAX microbenchmarks. The primary motivation is to address
limitations of ad-hoc benchmarking scripts, providing a standardized,
extensible, and insight-rich platform for performance analysis on TPUs and
GPUs.

(See [RATIONALE.md](RATIONALE.md) for a detailed breakdown of motivations
and gaps addressed).

## 2. Core Features

-   **Modular Architecture:** Registry-based system (`core/registry.py`) for
    easy discovery and addition of new benchmarks.
-   **Standardized Benchmark Lifecycle:** `BaseBenchmark` class (`core/base.py`)
    enforces consistent setup, input generation, execution, and metric
    calculation.
-   **Hierarchical Configuration:** YAML-based configs with global parameters,
    hardware specifications, and per-benchmark settings.
-   **Bulk Configuration:** Support for large-scale studies via:
    -   YAML Sweeps: Cartesian product generation of parameters.
    -   CSV/Google Sheets Ingestion: Loading shapes directly from spreadsheets.
-   **Integrated Roofline Analysis:** Automatic calculation of arithmetic
    intensity, roofline ceilings, and efficiency, supporting both:
    -   **Analytical Models:** Theory-based Flops/Bytes counts.
    -   **Trace-Based Models:** Using `jax.experimental.roofline` for Jaxpr
        profiling.
-   **Power-Aware Runs:** Capability to run benchmarks for a minimum duration
    (`min_duration_s`) to capture thermal/power effects.
-   **Model Presets:** Pre-defined configurations for common LLM sizes (e.g.,
    `LLM-200B`) in `core/model_configs.py`.
-   **Installable Package:** Packaged using `pyproject.toml` with a `tpums`
    CLI entry point.

## 3. Architecture & Design

### Directory Structure

```text
accelerator_microbenchmarks/
├── configs/            # YAML configuration files (e.g., sample.yaml, hbm_sweep.yaml)
├── docs/               # Documentation (README, DEVELOPERS, DESIGN, RATIONALE)
│   ├── DESIGN.md
│   ├── DEVELOPERS.md
│   └── RATIONALE.md
├── pyproject.toml
├── results/            # Output directory for benchmark metrics (JSON, CSV)
├── src/
│   └── accelerator_microbenchmarks/
│       ├── benchmarks/ # Concrete benchmark implementations (collectives, matmul, etc.)
│       ├── core/       # Framework core (BaseBenchmark, registry, config parsing)
│       └── cli.py      # Entry point for running benchmarks (tpums)
```

### Core Components

-   **`core/base.py:BaseBenchmark`:** Abstract base class defining the benchmark
    interface:
    -   `setup()`: One-time setup (e.g., JIT compilation).
    -   `generate_inputs()`: Data generation for `run_op`.
    -   `run_op()`: The core JAX function to be benchmarked.
    -   `get_run_identifier()`: Run differentiator according to parameters.
    -   `calculate_metrics()`: Computes performance metrics (TFLOPS, GB/s).
    -   `get_total_bytes()`, `get_arithmetic_intensity()`: For roofline
        analysis.
    -   `apply_roofline_analysis()`: Calculates roofline efficiency.
    -   `run()`: Orchestrates the benchmark execution, including warm-up, timing
        loops, and power-aware runs.
-   **`core/registry.py`:** A simple decorator-based registry
    (`@registry.register`) to make benchmark classes discoverable by name in
    YAML configs.
-   **`core/config.py`:** Handles loading YAML files, expanding sweeps, loading
    shapes from CSVs (`core/csv_loader.py`), and merging with model presets
    (`core/model_configs.py`).
-   **`accelerator_microbenchmarks/cli.py`:** The main entry point for the
    `tpums` CLI, parses arguments, loads configs, and runs the selected
    benchmarks.

### Configuration System Flow

1.  Load base YAML file.
2.  Merge global parameters.
3.  For each benchmark entry: a. Resolve model presets. b. Expand CSV shapes if
    specified. c. Generate parameter combinations from `sweep` definitions.
4.  Return a list of fully resolved benchmark run configurations.

## 4. Usage

(See [README.md](README.md) for detailed usage instructions and examples).

-   Installation: `pip install -e .`
-   Running: `tpums benchmark run-config configs/sample.yaml`

## 5. Extensibility

The framework is designed to be easily extensible. To add a new benchmark,
please refer to the step-by-step guide in [DEVELOPERS.md](DEVELOPERS.md).

## 6. Future Directions

-   Automated testing suite (`tests/`).
-   More sophisticated hardware-aware roofline models (e.g., considering cache
    hierarchies).
-   Integration with profiling tools (e.g., Xprof).
-   Enhanced result visualization capabilities.

This design provides a solid foundation for collaborative development and
comprehensive JAX performance analysis.
