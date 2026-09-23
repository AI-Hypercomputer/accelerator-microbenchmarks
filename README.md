
# TPUMS: TPU Microbenchmark Suite

[![Status](https://img.shields.io/badge/Status-Pre--Release-yellow.svg)](#overview)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.10+-red.svg)](https://github.com/google/jax)
[![Hardware](https://img.shields.io/badge/Hardware-tpu7x%20(ironwood)%20%7C%20v6e%20(trillium)-orange.svg)](https://cloud.google.com/tpu)

**TPUMS (TPU Microbenchmark Suite)** is a modular, high-fidelity benchmarking and profiling framework designed to evaluate the compute, memory, and interconnect performance of Cloud TPUs (currently targeting **tpu7x (ironwood)** and **v6e (trillium)**) using JAX.

> [!IMPORTANT]
> **Project Status — Active Development (Pre-Release):** TPUMS is currently under active development and has not yet reached an official stable release. CLI flags, YAML configuration schemas, and reported metric names may evolve prior to the first tagged release.

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
  - [2. Interactive Benchmark & Sweep Runs (`tpums benchmark run`)](#2-interactive-single-benchmark-runs-tpums-benchmark-run)
  - [3. Config-Driven Benchmark & Sweep Runs (`tpums benchmark run-config`)](#3-config-driven-benchmark--sweep-runs-tpums-benchmark-run-config)
  - [4. Common Execution Flags](#4-common-execution-flags)
- [Configuration & Parameter Sweeps](#configuration-and-sweeps)
  - [1. Structure of a Benchmark Config](#1-structure-of-a-benchmark-config)
  - [2. Explicit Test Cases (`cases:`)](#2-explicit-test-cases-cases)
  - [3. Bulk Case Ingestion via CSV (`cases_from_csv:`)](#3-bulk-case-ingestion-via-csv-cases_from_csv)
  - [4. Parameter Sweeps (`sweep:`)](#4-parameter-sweeps-sweep)
- [Benchmark Catalog](#benchmark-catalog)
- [Results, Profiling & Output Formats](#results-profiling-and-output)
  - [Wall Clock vs. XProf Timing Metrics](#wall-clock-vs-xprof-timing-metrics)
  - [Primary Metrics Reference](#primary-metrics)
  - [Output Files](#output-files)
  - [Profiling Traces (TensorBoard / XProf)](#profiling-traces-tensorboard--xprof)
- [Running on GKE & Platform Automation](#platform-automation)
- [Repository Structure](#repository-structure)

---

<a id="overview"></a>
## Overview

**TPUMS** provides a standardized, end-to-end benchmarking framework to measure, validate, and track the hardware performance of Cloud TPUs from single chips to multi-host topologies:

- **Consistent, Reproducible Measurement**: Eliminates measurement noise by automatically handling compilation warmup, device synchronization, and repeatable timing loops.
- **Hardware-Accurate Roofline Insights**: Captures both host wall-clock and on-device XProf hardware metrics, comparing achieved **TFLOPS** and **GB/s** directly against theoretical hardware limits (**%**).
- **Flexible Sweeps & Structured Reporting**: Runs single benchmarks,
  interactive CLI parameter sweeps, or large YAML/CSV parameter sweeps and
  exports structured CSV and JSON reports for dashboards and regression
  tracking.

<a id="core-capabilities-matrix"></a>
### Core Capabilities Matrix

| Subsystem | Benchmark Suite | Operations & Scope | Key Metrics Reported |
| :--- | :--- | :--- | :--- |
| **Compute** | **Matrix Multiplication (GEMM)** | Dense matrix multiplication ($$C = \alpha (A \times B) + \beta C$$) across configurable data types | • Compute Throughput (**TFLOPS**)<br>• Compute Roofline Efficiency (**%**)<br>• Compute Latency (**ms**) |
| **Memory (HBM)** | **HBM Memory Bandwidth** | High Bandwidth Memory STREAM operations (Copy, Scale, Add, Triad, Read-Only, Write-Only) | • Memory Bandwidth (**GB/s**)<br>• Memory Roofline Efficiency (**%**)<br>• Memory Access Latency (**ms**) |
| **Host I/O (PCIe)** | **Host-to-Device (H2D)**<br>**Device-to-Host (D2H)** | Host CPU memory to/from accelerator HBM data transfers | • PCIe Transfer Bandwidth (**GB/s**)<br>• Transfer Latency (**ms**) |
| **Interconnect (ICI)** | **Device-to-Device (D2D)** | Point-to-point inter-chip data transfers across ICI links | • ICI Link Bandwidth (**GB/s**)<br>• Pairwise N × N Device Bandwidth Matrix<br>• Transfer Latency (**ms**) |
| **Collectives** | **All-Gather**<br>**All-Reduce**<br>**All-to-All** | Distributed collective communication across multi-chip topologies | • Collective Bus Bandwidth (**GB/s**)<br>• Collective Step Latency (**ms**) |

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

Execute an HBM memory bandwidth sweep across multiple array sizes (`256 MiB` to `2048 MiB` total traffic) directly from the command line without writing any configuration files:

```bash
tpums benchmark run hbm --xprof_timing --op_type copy --size 67108864 134217728 268435456 536870912 --dtype bfloat16 --device_id 0
```

<a id="4-console-output-preview"></a>
### 4. Console Output Preview

TPUMS formats results into a clean, aligned summary banner:

```text
========================================================================================================================================================
Benchmark Results (HBMBandwidthBenchmark)
========================================================================================================================================================
   dtype op_type device_id      size total_bytes_mib wall_clock_p50_ms wall_clock_bandwidth_per_device_gb_s xprof_p50_ms xprof_bandwidth_per_device_gb_s
bfloat16    copy         0  67108864          256.00            0.2788                               962.86       0.0835                         3214.91
bfloat16    copy         0 134217728          512.00            0.3596                              1492.96       0.1669                         3216.35
bfloat16    copy         0 268435456         1024.00            0.4889                              2196.43       0.3316                         3238.06
bfloat16    copy         0 536870912         2048.00            0.8582                              2502.18       0.6642                         3233.08
========================================================================================================================================================
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
  - `tpums benchmark run` (interactive single-benchmark or parameter sweep run)
  - `tpums benchmark run-config` (config-driven multi-case or parameter sweep run)

```text
tpums
├── platform
│   └── describe                        # Query hardware topology, device count, and versions
└── benchmark
    ├── list                            # List all registered, production-ready benchmarks
    ├── run <benchmark_name> [options]  # Mode 1: Run a single benchmark or interactive parameter sweep via CLI flags
    └── run-config <path.yaml>          # Mode 2: Run multi-case tests or parameter sweeps defined in YAML
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
### 2. Interactive Benchmark & Sweep Runs (`tpums benchmark run`)

Execute benchmarks with typed arguments directly passed to the command line:

```bash
# 1. HBM Memory Bandwidth on Device 0 (STREAM copy kernel)
tpums benchmark run hbm --xprof_timing --op_type copy --size 134217728 --dtype bfloat16 --device_id 0

# 2. Matrix Multiplication (GEMM 4096 x 4096 x 4096)
tpums benchmark run gemm --xprof_timing --m 4096 --k 4096 --n 4096 --in_dtype bfloat16 --out_dtype bfloat16

# 3. Host-to-Device (PCIe) Transfer Latency & Bandwidth
tpums benchmark run host_to_device --xprof_timing --data_size_mib 256 --dtype bfloat16

# 4. Device-to-Host (PCIe) Transfer Latency & Bandwidth
tpums benchmark run device_to_host --xprof_timing --data_size_mib 256 --dtype bfloat16

# 5. Device-to-Device (ICI) point-to-point transfer across all pairs
tpums benchmark run device_to_device --xprof_timing --data_size_mib 1024 --direction uni --dtype bfloat16

# 6. Multi-Device All-Reduce Collective
#    - tpu7x (2x2x1, 8 logical devices): --mesh_shape 2x2x2 --sharding_strategy 2x2x1
#    - v6e   (2x2,   4 logical devices): --mesh_shape 2x2   --sharding_strategy 2x2
tpums benchmark run all_reduce --xprof_timing --mesh_shape 2x2x2 --sharding_strategy 2x2x1 --matrix_dim 8192 --dtype bfloat16 --reduce_op sum
```

#### Interactive CLI Parameter Sweeps

Pass multiple space-separated values to any flag in `tpums benchmark run` to expand and run their **Cartesian product** in a single invocation:

```bash
# 1. Numeric + String DType Sweep (HBM: 2 sizes × 2 dtypes = 4 runs)
tpums benchmark run hbm --size 134217728 268435456 --dtype bfloat16 float32

# 2. Enum + Integer Dimension Sweep (All-Reduce: 2 ops × 2 dims = 4 runs)
tpums benchmark run all_reduce --reduce_op sum max --matrix_dim 1024 2048

# 3. Boolean + Dimension Sweep (GEMM: 2 m × 2 n × 2 transpose_a = 8 runs)
tpums benchmark run gemm -m 1024 2048 -n 512 1024 --transpose_a true false
```

> [!NOTE]
> CLI parameter sweeps are capped at **1,000 combinations** per invocation.

<a id="3-config-driven-benchmark--sweep-runs-tpums-benchmark-run-config"></a>
### 3. Config-Driven Benchmark & Sweep Runs (`tpums benchmark run-config`)

Execute multi-case test lists, parameter sweeps, and profiling sessions defined in YAML:

```bash
tpums benchmark run-config configs/sample_configs/parameter_sweep.yaml \
    --xprof_dir /tmp/tensorboard \
    --output_dir results/
```

<a id="4-common-execution-flags"></a>
### 4. Common Execution Flags (`run` & `run-config`)

- **XProf Hardware Timing & Trace Capture**:
    1. **Enable XProf Timing (`--xprof_timing` / `xprof_timing: true`)**:
        - **Interactive CLI (`tpums benchmark run`)**: Pass `--xprof_timing` directly on the command line.
        - **YAML Config (`tpums benchmark run-config`)**: Set `xprof_timing: true` inside the YAML `benchmark:` block.
    2. **Set Trace Output Directory (`--xprof_dir <path>`, optional)**: Directory to save TensorBoard / XProf hardware trace files (`.xplane.pb`). Defaults to `/tmp/tensorboard`; **only active when XProf timing is enabled**.
- **Report Output & Compiler Flags**:
    - **`--output_dir <path>`**: Directory to save `summary.csv` and `detailed.json` (defaults to `results/`).
    - **`--xla_flags_file_path <path>`**: Optional path to a custom YAML file overriding default per-benchmark XLA / compiler runtime flags.

---

<a id="configuration-and-sweeps"></a>
## ⚙️ Configuration & Parameter Sweeps

While `tpums benchmark run` supports quick interactive CLI sweeps across space-separated flag values, YAML configuration files (`tpums benchmark run-config`) allow defining reproducible benchmark configurations, multi-case test lists, geometric/arithmetic parameter sweeps, and CSV shape tables.

<a id="1-structure-of-a-benchmark-config"></a>
### 1. Structure of a Benchmark Config

Configuration files define a top-level `benchmark:` mapping containing:

- **`name:` (`<benchmark_name>`)** — Target benchmark name (e.g., `gemm`, `hbm`, `all_reduce`; see [Benchmark Catalog](#benchmark-catalog)).
- **`xprof_timing:`** *(optional)* — Boolean (`true` / `false`) to enable XProf hardware trace collection and device timing analysis.
- **`params:`** *(optional)* — Baseline execution parameters shared across all generated runs.
- **`cases:` / `cases_from_csv:` / `sweep:`** *(optional)* — Case override and parameter sweep generators.

```yaml
benchmark:
  name: gemm                      # Target <benchmark_name>
  xprof_timing: true              # Enable hardware trace timing and XProf capture

  # 1. Baseline parameters shared across all generated executions
  params:
    warmup_tries: 2
    num_runs: 10
    out_dtype: bfloat16

  # 2. Per-case parameter overrides (or load from CSV via `cases_from_csv: configs/shapes/matrix_shapes.csv`)
  cases:
    - m: 1024
      k: 4096
      n: 4096
    - m: 2048
      k: 4096
      n: 8192

  # 3. Cartesian product sweep applied across every case above (2 cases × 2 dtypes = 4 runs)
  sweep:
    in_dtype: [bfloat16, float8_e4m3fn]
```

**Parameter Precedence & Evaluation Order:**

1. **`params:`** — Defines baseline parameters shared across all generated runs.
2. **`cases:` or `cases_from_csv:`** — Applies per-case parameter overrides on top of `params:`.
3. **`sweep:`** — Expands each case across the Cartesian product of all specified sweep axes *(sweep keys must be disjoint from keys defined in `params:` and `cases:` / `cases_from_csv:`)*.

Run the configuration with:

```bash
tpums benchmark run-config <path_to_config.yaml>
```

<a id="2-explicit-test-cases-cases"></a>
### 2. Explicit Test Cases (`cases:`)

Use `cases:` to define an explicit list of specific parameter configurations to benchmark. Each entry in `cases:` inherits all shared baseline options from `params:` and overrides only the keys specified in that entry (entries can override the same keys or different subsets of keys):

```yaml
benchmark:
  name: gemm
  params:
    warmup_tries: 2
    num_runs: 5
    in_dtype: bfloat16
    out_dtype: bfloat16
  cases:
    - m: 1024
      k: 1024
      n: 1024
    - m: 2048
      k: 4096
      n: 8192
```

<a id="3-bulk-case-ingestion-via-csv-cases_from_csv"></a>
### 3. Bulk Case Ingestion via CSV (`cases_from_csv:`)

To benchmark large sets of parameter combinations from external tables or workloads, TPUMS can ingest test cases directly from a CSV file via `cases_from_csv:`. Just like `cases:`, each row in the CSV is treated as an individual benchmark case that inherits shared baseline options from `params:` while overriding the columns specified in the CSV header (for example, `m`, `k`, `n` matrix dimensions):

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

<a id="4-parameter-sweeps-sweep"></a>
### 4. Parameter Sweeps (`sweep:`)

Unlike `cases:` (which runs an explicit list of individual configurations), the `sweep:` block automatically generates the **Cartesian product** across all specified parameter lists or geometric ranges:

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
        start: 134217728    # 128M elements (256 MiB per array in bfloat16)
        end: 1073741824     # 1G elements (2 GiB per array in bfloat16)
        multiplier: 2
  ```

---

<a id="benchmark-catalog"></a>
## 📊 Benchmark Catalog

TPUMS provides microbenchmarks across core accelerator subsystems. Select any benchmark name (`<benchmark_name>`) below to view its detailed parameter specifications, default values, and metric formulas in **[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)**:

| Subsystem | Benchmark Suite | Benchmark Name (`<benchmark_name>`) | Topology Support |
| :--- | :--- | :--- | :--- |
| **Compute** | Matrix Multiplication (GEMM) | [`gemm`](docs/BENCHMARKS.md#21-matrix-multiplication-gemm) | **Single-Host Only** |
| **Memory (HBM)** | HBM Memory Bandwidth | [`hbm`](docs/BENCHMARKS.md#22-hbm-memory-bandwidth-hbm) | **Single-Host Only** |
| **Host I/O (PCIe)** | Host-to-Device (H2D)<br>Device-to-Host (D2H) | [`host_to_device`](docs/BENCHMARKS.md#23-host-io-bandwidth-host_to_device-device_to_host)<br>[`device_to_host`](docs/BENCHMARKS.md#23-host-io-bandwidth-host_to_device-device_to_host) | **Single-Host Only** |
| **Interconnect (ICI)** | Device-to-Device (D2D) | [`device_to_device`](docs/BENCHMARKS.md#24-inter-chip-interconnect-device_to_device) | **Single-Host & Multi-Host** |
| **Collectives** | All-Gather<br>All-Reduce<br>All-to-All | [`all_gather`](docs/BENCHMARKS.md#25-collective-communications-all_reduce-all_gather-all_to_all)<br>[`all_reduce`](docs/BENCHMARKS.md#25-collective-communications-all_reduce-all_gather-all_to_all)<br>[`all_to_all`](docs/BENCHMARKS.md#25-collective-communications-all_reduce-all_gather-all_to_all) | **Single-Host & Multi-Host** |

To list all registered benchmarks or inspect CLI flags for a specific `<benchmark_name>`:

```bash
# List all registered benchmarks
tpums benchmark list

# View full parameter definitions and defaults for a specific benchmark:
# tpums benchmark run <benchmark_name> --help
tpums benchmark run gemm --help
```

> [!TIP]
> **Single-Host vs. Multi-Host Platform Selection & Example Benchmark Configs (`configs/<tpu_type>/<topology>/`)**:
>
> Ready-to-run YAML configurations are organized under `configs/<tpu_type>/<topology>/<benchmark_name>.yaml` and are executed according to the target topology:
>
> - **Single-Host Topologies (e.g., `tpu7x` `2x2x1`, `v6e` `2x2` or `2x4` on `ct6e-standard-8t`)**: Run directly on a **Cloud TPU VM** (`tpums benchmark run-config configs/v6e/2x2/gemm.yaml` or `configs/tpu7x/2x2x1/all_reduce.yaml`) or on **GKE**.
> - **Multi-Host Topologies (e.g., `tpu7x` `2x2x2`+, `v6e` `2x4` on `ct6e-standard-4t` / `4x4`+)**: Require multi-node coordination across all hosts in the topology (such as on **GKE**—see **[Running on GKE & Platform Automation](#platform-automation)** below).

---

<a id="results-profiling-and-output"></a>
## 📈 Results, Profiling & Output Formats

<a id="wall-clock-vs-xprof-timing-metrics"></a>
### Wall Clock vs. XProf Timing Metrics

TPUMS captures timing data across two distinct domains to provide full visibility into end-to-end framework execution versus raw on-device accelerator performance:

- **Wall Clock Metrics (`wall_clock_*`)**: Measures end-to-end execution time in the Python runtime. Because wall-clock measurements include host dispatch overhead, Python runtime latency, and device synchronization barriers, they are less accurate for assessing true kernel hardware performance.
- **Hardware XProf Metrics (`xprof_*`)**: Extracted directly from accelerator hardware traces via XLA trace events when XProf timing is enabled. These metrics isolate pure on-device kernel execution duration, free from host dispatch and synchronization overhead.

> [!TIP]
> **Measurement Recommendation:** Because wall-clock metrics include host dispatch and synchronization overhead, always enable `--xprof_timing` (or `xprof_timing: true` in YAML) and evaluate **`xprof_*`** metrics (`xprof_p50_ms`, `xprof_tflops_per_chip`, `xprof_bandwidth_per_chip_gb_s`) to obtain the most accurate hardware performance figures.

<a id="primary-metrics"></a>
### Primary Metrics Reference

TPUMS records primary performance metrics across both timing domains:

| Category | Wall Clock Metrics (Always Recorded) | Device XProf Metrics (Requires `xprof_timing`) | Description |
| :--- | :--- | :--- | :--- |
| **Latency (`ms`)** | `wall_clock_p50_ms`<br>*(+ `p90`, `avg`, `std`)* | `xprof_p50_ms`<br>*(+ `p90`, `avg`, `std`)* | Execution duration across timed iterations (`p50` median, `p90`, mean, and standard deviation). |
| **Compute Throughput (`TFLOPS`)** | `wall_clock_tflops_per_chip`<br>`wall_clock_tflops_per_device` | `xprof_tflops_per_chip`<br>`xprof_tflops_per_device` | Achieved compute throughput per physical TPU chip or per logical device ($$10^{12}$$ FLOPS). |
| **Bandwidth (`GB/s`)** | `wall_clock_bandwidth_per_chip_gb_s`<br>`wall_clock_bandwidth_per_device_gb_s` | `xprof_bandwidth_per_chip_gb_s`<br>`xprof_bandwidth_per_device_gb_s` | Achieved HBM, PCIe, ICI link, or collective bus bandwidth per chip or per device ($$10^9$$ bytes/sec). |
| **Roofline Efficiency (`%`)** | `wall_clock_compute_roofline_efficiency_pct`<br>`wall_clock_memory_roofline_efficiency_pct` | `xprof_compute_roofline_efficiency_pct`<br>`xprof_memory_roofline_efficiency_pct` | Achieved compute or memory performance as a percentage of the hardware's theoretical peak (`0–100%`). |

See **[Throughput & Bandwidth Metric Conventions in `docs/BENCHMARKS.md`](docs/BENCHMARKS.md#per-device-vs-per-chip-scaling)** for how `_per_chip` vs. `_per_device` scales across dual-device (`tpu7x`) and single-device (`v6e`) TPU chips and which metrics each benchmark reports.

<a id="output-files"></a>
### Output Files

After every run, TPUMS automatically saves two files in the specified `--output_dir`, both exporting the **exact same complete set of metrics, benchmark parameters, and platform metadata**:

1. **`summary.csv`**: A flat, tabular CSV containing all reported metrics, benchmark parameters, and platform metadata for every configuration tested. Ideal for loading into Pandas, Google Sheets, or dashboarding pipelines.
2. **`detailed.json`**: A structured JSON record containing the complete benchmark metrics, configuration parameters, platform metadata, and execution timestamps for programmatic analysis and automation pipelines.

<a id="profiling-traces-tensorboard--xprof"></a>
### Profiling Traces (TensorBoard / XProf)

When XProf timing is enabled (see [Common Execution Flags](#4-common-execution-flags)), TPUMS records `.xplane.pb` hardware trace files to `--xprof_dir` (default: `/tmp/tensorboard`) that can be inspected directly in TensorBoard or the Google Cloud Vertex AI / XProf viewer:

```bash
tpums benchmark run-config configs/sample_configs/parameter_sweep.yaml \
    --xprof_dir /tmp/traces \
    --output_dir results/
```

---

<a id="platform-automation"></a>
## ☁️ Running on GKE & Platform Automation

Beyond interactive runs on a single Cloud TPU VM, TPUMS supports running both single-host and multi-host TPU workloads on **Google Kubernetes Engine (GKE)**:

- **[Single-Benchmark GKE Deployment (`docs/GKE.md`)](docs/GKE.md)**: Run any individual benchmark configuration across single-host or multi-host TPU topologies on GKE, with optional result and XProf trace export to Google Cloud Storage (GCS).
- **[Multi-Topology Platform Automation (`docs/AUTOMATION.md`)](docs/AUTOMATION.md)**: Automatically orchestrate multiple benchmarks across multiple TPU topologies on GKE and aggregate results from GCS into consolidated cross-topology performance reports.

---

<a id="repository-structure"></a>
## 📁 Repository Structure

```text
accelerator_microbenchmarks/
├── automation/                 # Automated multi-topology GKE orchestration and report generation
├── configs/                    # Ready-to-use YAML configs and parameter sweeps
│   ├── sample_configs/         # Introductory sweeps and validation configs
│   ├── shapes/                 # Predefined matrix shape sweeps (CSV)
│   └── tpu7x/, v6e/            # Hardware-specific single-host & multi-host topology configs
├── docs/                       # Reference, deployment, and automation guides
│   ├── AUTOMATION.md           # Automated multi-topology benchmark pipeline and reporting runbook
│   ├── BENCHMARKS.md           # Per-benchmark parameters, metrics, and collective sharding concepts
│   └── GKE.md                  # Single-benchmark GKE deployment and topology sizing guide
├── pyproject.toml              # Build system, dependencies, and CLI entry point
├── results/                    # Default destination directory for CSV and JSON reports
├── templates/                  # Standalone deployment templates (e.g., GKE Job manifest)
│   └── gke_job_template.yaml   # Single-host and multi-host GKE Job template
├── tests/                      # Comprehensive unit and integration test suites
└── src/
    └── accelerator_microbenchmarks/
        ├── benchmarks/         # Concrete benchmark implementations (gemm, hbm, collectives, etc.)
        ├── core/               # Core framework (base class, config, runner, reporting)
        ├── cli.py              # Canonical CLI entry point (tpums)
        └── op_flags.yaml       # Hardware-specific compiler & XLA flag mappings
```