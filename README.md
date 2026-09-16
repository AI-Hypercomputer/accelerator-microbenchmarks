# TPUMS: TPU Microbenchmark Suite

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.10+-red.svg)](https://github.com/google/jax)
[![Hardware](https://img.shields.io/badge/Hardware-tpu7x%20(ironwood)%20%7C%20v6e%20(trillium)-orange.svg)](https://cloud.google.com/tpu)

**TPUMS (TPU Microbenchmark Suite)** is a modular, high-fidelity benchmarking and profiling framework designed to evaluate the compute, memory, and interconnect performance of Cloud TPUs (currently targeting **tpu7x (ironwood)** and **v6e (trillium)**) using JAX.

---

## Table of Contents

- [Overview](#overview)
  - [Core Capabilities Matrix](#core-capabilities-matrix)
- [Quickstart](#quickstart)
  - [Prerequisites](#prerequisites)
  - [1. Installation](#1-installation)
  - [2. Inspect Accelerator Hardware & Environment](#2-inspect-accelerator-hardware--environment)
  - [3. Run Your First Benchmark](#3-run-your-first-benchmark)
  - [4. Console Output Preview](#4-console-output-preview)
- [CLI Reference](#cli-reference)
  - [1. Discovery & Environment Inspection](#1-discovery--environment-inspection)
  - [2. Interactive Single-Benchmark Runs (`tpums benchmark run`)](#2-interactive-single-benchmark-runs-tpums-benchmark-run)
  - [3. Config-Driven Suite & Sweep Runs (`tpums benchmark run-config`)](#3-config-driven-suite--sweep-runs-tpums-benchmark-run-config)
  - [4. Common Execution Flags](#4-common-execution-flags)
- [Configuration & Parameter Sweeps](#configuration-and-sweeps)
  - [1. Structure of a Benchmark Config](#1-structure-of-a-benchmark-config)
  - [2. Parameter Sweeps (`sweep:`)](#2-parameter-sweeps-sweep)
  - [3. Explicit Non-Uniform Cases (`cases:`)](#3-explicit-non-uniform-cases-cases)
  - [4. Bulk Case Ingestion via CSV (`cases_from_csv:`)](#4-bulk-case-ingestion-via-csv-cases_from_csv)
- [Benchmark Catalog](#benchmark-catalog)
  - [1. Matrix Multiplication (`gemm`)](#1-matrix-multiplication-gemm)
  - [2. HBM Memory Bandwidth (`hbm`)](#2-hbm-memory-bandwidth-hbm)
  - [3. Host I/O Bandwidth (`host_to_device`, `device_to_host`)](#3-host-io-bandwidth-host_to_device-device_to_host)
  - [4. Inter-Chip Interconnect (`device_to_device`)](#4-inter-chip-interconnect-device-to-device)
  - [5. Collective Communications (`all_gather`, `all_reduce`, `all_to_all`)](#5-collective-communications-all_gather-all_reduce-all_to_all)
- [Results, Profiling & Output Formats](#results-profiling-and-output)
  - [Wall Clock vs. XProf Timing Metrics](#wall-clock-vs-xprof-timing-metrics)
  - [Primary Metrics](#primary-metrics)
  - [Profiling Traces (TensorBoard / XProf)](#profiling-traces-tensorboard--xprof)
  - [Output Files](#output-files)
- [Platform Automation (GKE)](#platform-automation)
- [Repository Structure](#repository-structure)
- [Extending TPUMS (Adding New Benchmarks)](#extending-tpums)

---

<a id="overview"></a>
## Overview

**TPUMS** provides a standardized, end-to-end benchmarking framework to measure, validate, and track the hardware performance of Cloud TPUs from single chips to multi-host slices:

- **Consistent, Reproducible Measurement**: Eliminates measurement noise by automatically handling compilation warmup, device synchronization, and repeatable timing loops.
- **Hardware-Accurate Roofline Insights**: Captures both host wall-clock and on-device XProf hardware metrics, comparing achieved **TFLOPS** and **GB/s** directly against theoretical hardware limits (**%**).
- **Flexible Sweeps & Structured Reporting**: Runs single benchmarks or large YAML/CSV parameter sweeps and exports structured CSV and JSON reports for dashboards and regression tracking.

<a id="core-capabilities-matrix"></a>
### Core Capabilities Matrix

| Subsystem | Primary Benchmarks | Operations & Scope | Key Metrics Reported |
| :--- | :--- | :--- | :--- |
| **Compute** | `gemm` | Dense matrix multiplication ($$C = \alpha (A \times B) + \beta C$$) across configurable data types | • Compute Throughput (**TFLOPS**)<br>• Compute Roofline Efficiency (**%**)<br>• Compute Latency (**ms**) |
| **Memory (HBM)** | `hbm` | High Bandwidth Memory STREAM operations (`copy`, `scale`, `add`, `triad`) | • Memory Bandwidth (**GB/s**)<br>• Memory Roofline Efficiency (**%**)<br>• Memory Access Latency (**ms**) |
| **Host I/O (PCIe)** | `host_to_device`<br>`device_to_host` | Host CPU memory to/from accelerator HBM data transfers | • PCIe Transfer Bandwidth (**GB/s**)<br>• Transfer Latency (**ms**) |
| **Interconnect (ICI)** | `device_to_device` | Point-to-point inter-chip data transfers across ICI links | • ICI Link Bandwidth (**GB/s**)<br>• Pairwise N × N Device Bandwidth Matrix<br>• Transfer Latency (**ms**) |
| **Collectives** | `all_gather`<br>`all_reduce`<br>`all_to_all` | Distributed collective communication across multi-chip topologies | • Collective Bus Bandwidth (**GB/s**)<br>• Collective Step Latency (**ms**) |

---

<a id="quickstart"></a>
## ⚡ Quickstart

Get up and running on your accelerator environment in seconds.

<a id="prerequisites"></a>
### Prerequisites

- **Hardware**: A Cloud TPU VM or GKE TPU container (`tpu7x` or `v6e`).
- **Python**: Python `3.12+`.

<a id="1-installation"></a>
### 1. Installation

Clone the repository and install in editable mode within your Python environment:

```bash
git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git
cd accelerator-microbenchmarks
pip install -e .
```

<a id="2-inspect-accelerator-hardware--environment"></a>
### 2. Inspect Accelerator Hardware & Environment

Verify your TPU environment and detect active topology, chip count, and runtime libraries:

```bash
tpums platform describe
```

*Example output on Cloud TPU `tpu7x`:*

```json
{
  "tpu_type": "tpu7x",
  "topology": "2x2x1",
  "total_devices": 8,
  "local_devices": 8,
  "process_count": 1,
  "process_index": 0,
  "python_version": "3.12.14",
  "jax_version": "0.10.1",
  "jaxlib_version": "0.10.1",
  "libtpu_version": "0.0.41"
}
```

<a id="3-run-your-first-benchmark"></a>
### 3. Run Your First Benchmark

Execute an HBM memory bandwidth benchmark directly from the command line without writing any configuration files:

```bash
tpums benchmark run hbm --xprof_timing --op_type copy --size 134217728 --dtype bfloat16 --device_id 0
```

<a id="4-console-output-preview"></a>
### 4. Console Output Preview

TPUMS formats results into a clean, aligned summary banner:

```text
=====================================================================================================================================================
Benchmark Results (HBMBandwidthBenchmark)
=====================================================================================================================================================
   dtype op_type device_id      size total_bytes_mib wall_clock_p50_ms wall_clock_bandwidth_per_chip_gb_s xprof_p50_ms xprof_bandwidth_per_chip_gb_s
bfloat16    copy         0 134217728          512.00            0.3085                            3242.45       0.1655                       6037.16
=====================================================================================================================================================
```

---

<a id="cli-reference"></a>
## 🛠️ CLI Reference

The `tpums` executable provides a structured resource-action CLI organized into two functional categories:

- **Discovery & Inspection Utilities**:
  - `tpums platform describe`
  - `tpums benchmark list`
  - `tpums benchmark run <benchmark_name> --help`
- **Benchmark Execution Modes**:
  - `tpums benchmark run` (interactive single-benchmark run)
  - `tpums benchmark run-config` (config-driven suite/sweep run)

```text
tpums
├── platform
│   └── describe                        # Query hardware topology, device count, and versions
└── benchmark
    ├── list                            # List all registered, production-ready benchmarks
    ├── run <benchmark_name> [options]  # Mode 1: Run a single benchmark interactively via CLI flags
    └── run-config <path.yaml>          # Mode 2: Run a suite or parameter sweep defined in YAML
```

<a id="1-discovery--environment-inspection"></a>
### 1. Discovery & Environment Inspection

Query hardware topology, list available benchmarks, or inspect benchmark-specific CLI parameters before executing a run:

```bash
# 1. Query TPU hardware topology, chip count, and JAX/libtpu versions
tpums platform describe

# 2. List all registered, production-ready benchmarks
tpums benchmark list

# 3. Inspect typed CLI flags and default values for a specific benchmark
# tpums benchmark run <benchmark_name> --help
tpums benchmark run gemm --help
```

<a id="2-interactive-single-benchmark-runs-tpums-benchmark-run"></a>
### 2. Interactive Single-Benchmark Runs (`tpums benchmark run`)

Execute benchmarks with typed arguments directly passed to the command line:

```bash
# 1. HBM Memory Bandwidth on Device 0 (STREAM copy kernel)
tpums benchmark run hbm --xprof_timing --op_type copy --size 134217728 --dtype bfloat16 --device_id 0

# 2. Matrix Multiplication (GEMM 4096 x 4096 x 4096)
tpums benchmark run gemm --xprof_timing --m 4096 --k 4096 --n 4096 --in_dtype bfloat16

# 3. Host-to-Device (PCIe) Transfer Latency & Bandwidth
tpums benchmark run host_to_device --xprof_timing --data_size_mib 256 --dtype bfloat16

# 4. Device-to-Host (PCIe) Transfer Latency & Bandwidth
tpums benchmark run device_to_host --xprof_timing --data_size_mib 256 --dtype bfloat16

# 5. Device-to-Device (ICI) point-to-point transfer across all pairs
tpums benchmark run device_to_device --xprof_timing --data_size_mib 1024 --direction UNI --dtype bfloat16

# 6. Multi-Device All-Reduce Collective across a 2x2x2 mesh with 2x2x1 sharding
tpums benchmark run all_reduce --xprof_timing --mesh_shape 2x2x2 --sharding_strategy 2x2x1 --matrix_dim 8192 --dtype bfloat16 --reduce_op sum
```

<a id="3-config-driven-suite--sweep-runs-tpums-benchmark-run-config"></a>
### 3. Config-Driven Suite & Sweep Runs (`tpums benchmark run-config`)

Execute test suites, parameter sweeps, and profiling sessions defined in YAML:

```bash
tpums benchmark run-config configs/sample_configs/parameter_sweep.yaml \
    --xprof_dir /tmp/tensorboard \
    --output_dir results/
```

<a id="4-common-execution-flags"></a>
### 4. Common Execution Flags (`run` & `run-config`)

- `--xprof_timing`: Enable hardware-level kernel profiling via XProf trace capture:
  - **Interactive CLI (`tpums benchmark run`)**: Pass `--xprof_timing` directly on the command line.
  - **YAML Config (`tpums benchmark run-config`)**: Configure `xprof_timing: true` inside the YAML `benchmark:` block.
- `--xprof_dir <path>`: Directory to record TensorBoard / XProf trace files (`.xplane.pb`). *(Only active and exported when `xprof_timing` is enabled).*
- `--output_dir <path>`: Directory to persist `summary.csv` and `detailed.json` (defaults to `results/`).
- `--xla_flags_file_path <path>`: Optional YAML file containing customized XLA/compiler runtime flags.

---

<a id="configuration-and-sweeps"></a>
## ⚙️ Configuration & Parameter Sweeps

YAML configuration files allow defining reproducible benchmark suites and automated parameter sweeps across matrix dimensions, data types, and mesh topologies.

<a id="1-structure-of-a-benchmark-config"></a>
### 1. Structure of a Benchmark Config

Configuration files define a top-level `benchmark:` mapping containing:

- **`name:`** — Target benchmark identifier (e.g., `gemm`, `hbm`, `all_reduce`).
- **`xprof_timing:`** *(optional)* — Boolean (`true` / `false`) to enable XProf hardware trace collection and device timing analysis.
- **`params:`** *(optional)* — Baseline execution parameters shared across all generated runs.
- **`cases:` / `cases_from_csv:` / `sweep:`** *(optional)* — Case override and parameter sweep generators.

```yaml
# configs/sample_configs/gemm_sweep.yaml
benchmark:
  name: gemm
  xprof_timing: true            # Enable hardware trace timing and XProf capture

  # Baseline parameters shared across all generated executions
  params:
    warmup_tries: 2
    num_runs: 10
    dtype: bfloat16
    k: 4096
    n: 4096

  # Parameter variation generator: Cartesian product across axes
  sweep:
    m: [1024, 2048, 4096, 8192]
```

**Parameter Precedence & Evaluation Order:**

1. **`params:`** — Defines baseline parameters shared across all runs in the suite.
2. **`cases:` or `cases_from_csv:`** — Applies per-case parameter overrides on top of `params:`.
3. **`sweep:`** — Expands each case across the Cartesian product of all specified sweep axes *(sweep keys must be disjoint from keys defined in `params:` and `cases:` / `cases_from_csv:`)*.

Run the configuration with:

```bash
tpums benchmark run-config configs/sample_configs/gemm_sweep.yaml
```

<a id="2-parameter-sweeps-sweep"></a>
### 2. Parameter Sweeps (`sweep:`)

The `sweep:` block generates the Cartesian product of all specified parameter lists or geometric ranges:

- **Discrete Value Sweep:** Test specific matrix dimensions, sharding strategies, or operations:

  ```yaml
  benchmark:
    name: all_reduce
    params:
      warmup_tries: 2
      num_runs: 5
      dtype: bfloat16
      mesh_shape: 2x2x2
    sweep:
      sharding_strategy: ["2x2x1", "2x2x2"]
      matrix_dim: [1024, 2048, 4096, 8192]
  ```

- **Geometric Multiplier Sweep:** Automatically scale values across a geometric range:

  ```yaml
  benchmark:
    name: hbm
    params:
      warmup_tries: 5
      num_runs: 20
      dtype: bfloat16
    sweep:
      op_type: ["copy", "scale", "add", "triad"]
      size:
        start: 134217728    # 128 MiB (elements)
        end: 1073741824     # 1 GiB
        multiplier: 2
  ```

<a id="3-explicit-non-uniform-cases-cases"></a>
### 3. Explicit Non-Uniform Cases (`cases:`)

To test an explicit list of non-uniform parameter configurations without generating a combinatorial Cartesian product, use `cases:`:

```yaml
benchmark:
  name: gemm
  params:
    warmup_tries: 2
    num_runs: 5
    dtype: bfloat16
  cases:
    - m: 1024
      k: 1024
      n: 1024
    - m: 2048
      k: 4096
      n: 8192
```

<a id="4-bulk-case-ingestion-via-csv-cases_from_csv"></a>
### 4. Bulk Case Ingestion via CSV (`cases_from_csv:`)

To benchmark large sets of parameter combinations from external tables or workloads, TPUMS can ingest test cases directly from a CSV file via `cases_from_csv:`. Each CSV column header maps to a benchmark parameter and each row defines a case override (for example, sweeping diverse GEMM `m`, `k`, `n` matrix shapes):

```yaml
benchmark:
  name: gemm
  params:
    warmup_tries: 2
    num_runs: 5
    in_dtype: bfloat16
    out_dtype: bfloat16
  cases_from_csv: configs/shapes/matrix_shapes.csv
```

**Example CSV (`configs/shapes/matrix_shapes.csv`):**

```csv
m,k,n
1,8192,1024
1024,4096,4096
2048,4096,8192
4096,8192,8192
```

Each row in the CSV is treated as an individual benchmark case, inheriting shared baseline options from `params:` while evaluating the specific `m`, `k`, `n` dimensions.

---

<a id="benchmark-catalog"></a>
## 📊 Benchmark Catalog

TPUMS includes battle-tested microbenchmarks targeting every critical accelerator subsystem.

To explore all registered benchmarks or inspect parameter definitions, supported data types, and default values for a specific benchmark:

```bash
# List all registered benchmarks
tpums benchmark list

# View full parameter definitions and defaults for a specific benchmark:
# tpums benchmark run <benchmark_name> --help
tpums benchmark run gemm --help
```

<a id="1-matrix-multiplication-gemm"></a>
### 1. Matrix Multiplication (`gemm`)
Profiles dense matrix multiplication kernels ($$C = \alpha (A \times B) + \beta C$$).

- **Key Parameters:** `m`, `k`, `n`, `in_dtype`, `out_dtype`, `transpose_a`, `transpose_b`, `alpha`, `beta`.
- **Metrics Reported:** `xprof_tflops_per_chip`, `wall_clock_tflops_per_chip`, `total_flops`, `xprof_p50_ms`, `wall_clock_p50_ms` *(plus per-device `xprof_tflops_per_device` and `wall_clock_tflops_per_device` in `detailed.json`)*.

<a id="2-hbm-memory-bandwidth-hbm"></a>
### 2. HBM Memory Bandwidth (`hbm`)
Measures raw High Bandwidth Memory throughput by executing standard STREAM kernels across accelerator cores.

- **Operations Supported:** `copy` (array-to-array assignment), `scale` (scalar multiplication), `add` (vector addition), `triad` (fused scale-add).
- **Key Parameters:** `size` (elements), `dtype`, `op_type`, `device_id`.
- **Metrics Reported:** `xprof_bandwidth_per_chip_gb_s`, `wall_clock_bandwidth_per_chip_gb_s`, `total_bytes_mib`, `xprof_p50_ms`, `wall_clock_p50_ms` *(plus per-device `xprof_bandwidth_per_device_gb_s` and `wall_clock_bandwidth_per_device_gb_s` in `detailed.json`)*.

<a id="3-host-io-bandwidth-host_to_device-device_to_host"></a>
### 3. Host I/O Bandwidth (`host_to_device`, `device_to_host`)
Measures PCIe data transfer bandwidth and latency between host CPU system memory and accelerator device HBM.

- **Key Parameters:** `data_size_mib`, `dtype`.
- **Metrics Reported:** `xprof_bandwidth_per_device_gb_s`, `wall_clock_bandwidth_per_device_gb_s`, `xprof_p50_ms`, `wall_clock_p50_ms`.

<a id="4-inter-chip-interconnect-device_to_device"></a>
### 4. Inter-Chip Interconnect (`device_to_device`)
Measures point-to-point bandwidth across physical ICI (Inter-Chip Interconnect) links. Automatically sweeps all `(src, dst)` device pairs to evaluate mesh link performance.

- **Key Parameters:** `data_size_mib`, `direction` (`UNI` or `BI`), `dtype`.
- **Metrics Reported:** Point-to-point `xprof_bandwidth_per_device_gb_s`, `wall_clock_bandwidth_per_device_gb_s`, full N × N pairwise device bandwidth matrix.

<a id="5-collective-communications-all_gather-all_reduce-all_to_all"></a>
### 5. Collective Communications (`all_gather`, `all_reduce`, `all_to_all`)
Profiles distributed collective communication primitives across 2D and 3D torus/mesh network topologies.

- **Key Parameters:** `mesh_shape` (e.g. `"2x2x2"` or `"2x4"`), `sharding_strategy` (e.g. `"2x2x1"` or `"2x2x2"`), `matrix_dim`, `dtype`, `reduce_op` (for `all_reduce`: `sum`, `mean`, `max`, `min`).
- **Metrics Reported:** Bus `xprof_bandwidth_per_chip_gb_s`, `wall_clock_bandwidth_per_chip_gb_s`, `shard_size_mib`, `xprof_p50_ms`, `wall_clock_p50_ms`.

> [!NOTE]
> Additional experimental benchmarks (such as `reduce_scatter`, FlashAttention, MoE Transformer layers, and custom fusion ops) are under active development — use with caution.

---

<a id="results-profiling-and-output"></a>
## 📈 Results, Profiling & Output Formats

<a id="wall-clock-vs-xprof-timing-metrics"></a>
### Wall Clock vs. XProf Timing Metrics

TPUMS captures timing data across two distinct domains to provide full visibility into end-to-end framework execution versus raw on-device accelerator performance:

- **Wall Clock Metrics (`wall_clock_*`)**: Measures end-to-end execution time in the Python runtime (`wall_clock_p50_ms`), as well as derived throughput (`wall_clock_tflops_per_chip`, `wall_clock_bandwidth_per_chip_gb_s`, `wall_clock_bandwidth_per_device_gb_s`). Because wall-clock measurements include host dispatch overhead, Python runtime latency, and device synchronization barriers, they are less accurate for assessing true kernel hardware performance.
- **Hardware XProf Metrics (`xprof_*`)**: Extracted directly from accelerator hardware traces via XLA trace events when XProf timing is enabled (`xprof_p50_ms`, `xprof_tflops_per_chip`, `xprof_bandwidth_per_chip_gb_s`, `xprof_bandwidth_per_device_gb_s`). These metrics isolate pure on-device kernel execution duration, free from host dispatch and synchronization overhead.

> [!TIP]
> **Measurement Recommendation:** Because wall-clock metrics include host dispatch and synchronization overhead, always enable `--xprof_timing` (or `xprof_timing: true` in YAML) and evaluate **`xprof_*`** metrics (`xprof_p50_ms`, `xprof_tflops_per_chip`, `xprof_bandwidth_per_chip_gb_s`) to obtain the most accurate hardware performance figures.

<a id="primary-metrics"></a>
### Primary Metrics Reference

Output reports (`summary.csv` and `detailed.json`) provide standardized, non-overlapping columns distinguishing host wall-clock execution from device kernel execution:

| Metric Column | Domain | Description | Profiling Requirement |
| :--- | :--- | :--- | :--- |
| **`wall_clock_p50_ms`** | Wall Clock | Median end-to-end wall-clock latency across iterations (includes host dispatch and device sync). | Always recorded |
| **`wall_clock_tflops_per_chip`** / **`wall_clock_tflops_per_device`** | Wall Clock | Compute throughput per chip or per device calculated from wall-clock duration ($$10^{12}$$ FLOPS). | Always recorded |
| **`wall_clock_bandwidth_per_chip_gb_s`** / **`wall_clock_bandwidth_per_device_gb_s`** | Wall Clock | Memory, PCIe, ICI, or collective bus bandwidth per chip or per device calculated from wall-clock duration ($$10^9$$ bytes/sec). | Always recorded |
| **`wall_clock_compute_roofline_efficiency_pct`** / **`wall_clock_memory_roofline_efficiency_pct`** | Wall Clock | Roofline utilization percentage relative to peak compute or memory ceiling calculated from wall-clock metrics. | Always recorded |
| **`xprof_p50_ms`** | Device (XProf) | Pure accelerator kernel execution duration parsed directly from XLA trace events. | Requires `xprof_timing` |
| **`xprof_tflops_per_chip`** / **`xprof_tflops_per_device`** | Device (XProf) | True on-device arithmetic compute throughput per chip or per device calculated from XProf kernel duration. | Requires `xprof_timing` |
| **`xprof_bandwidth_per_chip_gb_s`** / **`xprof_bandwidth_per_device_gb_s`** | Device (XProf) | True on-device memory, PCIe, ICI, or collective bus bandwidth per chip or per device calculated from XProf kernel duration. | Requires `xprof_timing` |
| **`xprof_compute_roofline_efficiency_pct`** / **`xprof_memory_roofline_efficiency_pct`** | Device (XProf) | True on-device roofline utilization percentage relative to peak compute or memory ceiling. | Requires `xprof_timing` |

<a id="profiling-traces-tensorboard--xprof"></a>
### Profiling Traces (TensorBoard / XProf)
Enabling XProf timing (via `--xprof_timing` on CLI or `xprof_timing: true` in YAML, with optional `--xprof_dir`) records JAX profiler trace files. The resulting `.xplane.pb` traces can be inspected directly in TensorBoard or the Google Cloud Vertex AI / XProf viewer:

```bash
tpums benchmark run-config configs/sample_configs/parameter_sweep.yaml \
    --xprof_dir /tmp/traces \
    --output_dir results/
```

<a id="output-files"></a>
### Output Files
After every run, TPUMS automatically persists artifacts in the specified `--output_dir`:
1. **`summary.csv`**: A compact, tabular CSV containing the primary metrics for every configuration tested. Ideal for loading into Pandas, Google Sheets, or dashboarding pipelines.
2. **`detailed.json`**: A comprehensive record containing raw per-iteration timings, full configuration parameters, platform metadata, and execution timestamps.

---

<a id="platform-automation"></a>
## ☁️ Platform Automation (GKE)

TPUMS supports automated execution and fleet orchestration on **Google Kubernetes Engine (GKE)**:

- **Containerized GKE Jobs**: Single-host and multi-host TPU slice benchmarks can be packaged into container images and deployed as GKE Jobs / JobSets across TPU node pools.
- **Automated Suite Execution & Artifact Collection**: Orchestrates end-to-end YAML benchmark suites on GKE clusters and exports structured CSV, JSON, and XProf trace artifacts to Cloud Storage.

*(Stay tuned: The GKE automation command and sample GKE Job deployment YAML manifests will be added in an upcoming update).*

---

<a id="repository-structure"></a>
## 📁 Repository Structure

```text
accelerator_microbenchmarks/
├── configs/                    # Ready-to-use YAML configs and parameter sweeps
│   ├── sample_configs/         # Introductory sweeps and validation configs
│   ├── shapes/                 # Predefined matrix shape sweeps (CSV)
│   └── tpu7x/, v6e/            # Hardware-specific topology configurations
├── docs/                       # Architecture and developer guides
│   ├── DESIGN.md               # Framework design document
│   ├── DEVELOPERS.md           # Guide to adding custom benchmarks
│   └── RATIONALE.md            # Architectural motivation and design principles
├── pyproject.toml              # Build system, dependencies, and CLI entry point
├── results/                    # Default destination directory for CSV and JSON reports
├── tests/                      # Comprehensive unit and integration test suites
└── src/
    └── accelerator_microbenchmarks/
        ├── benchmarks/         # Concrete benchmark implementations (gemm, hbm, collectives, etc.)
        ├── core/               # Core framework (base class, config, runner, reporting)
        ├── cli.py              # Canonical CLI entry point (tpums)
        └── op_flags.yaml       # Hardware-specific compiler & XLA flag mappings
```

---

<a id="extending-tpums"></a>
## 🔌 Extending TPUMS (Adding New Benchmarks)

TPUMS is designed to be easily extensible. To implement custom microbenchmarks or integrate new accelerator operations, refer directly to our step-by-step developer guide in [DEVELOPERS.md](docs/DEVELOPERS.md).