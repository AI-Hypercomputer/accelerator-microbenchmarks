
# TPUMS Benchmark Reference & Concepts Guide

This guide provides the reference for **TPUMS (TPU Microbenchmark Suite)** parameters, default values, metric reporting conventions, roofline/bandwidth formulas, and collective communication concepts.

> [!TIP]
> For installation, CLI usage, and YAML configuration instructions, see the main **[`README.md`](../README.md)**.

---

## Table of Contents

- [1. Shared Execution Parameters & Metric Conventions](#1-shared-base-parameters--chip-scaling-conventions)
  - [1.1 Shared Execution Parameters](#shared-execution-parameters)
  - [1.2 Throughput & Bandwidth Metric Conventions (`<timing_domain>_<metric_type>_<hardware_scope>`)](#per-device-vs-per-chip-scaling)
- [2. Per-Benchmark Parameter & Metric Specifications](#2-per-benchmark-parameter--metric-specifications)
  - [2.1 Matrix Multiplication (`gemm`)](#21-matrix-multiplication-gemm)
    - [2.1.1 GEMM Throttling (`gemm_throttling`) [experimental]](#211-gemm-throttling-gemm_throttling-experimental)
    - [2.1.2 Chained Matrix Multiplication & Determinism Check (`gemm_chained`) [experimental]](#212-chained-matrix-multiplication--determinism-check-gemm_chained-experimental)
    - [2.1.3 GEMM Pulse (`gemm_pulse`) [experimental]](#213-gemm-pulse-gemm_pulse-experimental)
    - [2.1.4 Quantized Matrix Multiplication (`gemm_quantized`) [experimental]](#214-quantized-matrix-multiplication-gemm_quantized-experimental)
  - [2.2 HBM Memory Bandwidth (`hbm`)](#22-hbm-memory-bandwidth-hbm)
  - [2.3 Host I/O Bandwidth (`host_to_device`, `device_to_host`)](#23-host-io-bandwidth-host_to_device-device_to_host)
  - [2.4 Inter-Chip Interconnect (`device_to_device`)](#24-inter-chip-interconnect-device_to_device)
  - [2.5 Collective Communications (`all_reduce`, `all_gather`, `all_to_all`)](#25-collective-communications-all_reduce-all_gather-all_to_all)
    - [Collective Concepts: `mesh_shape`, `sharding_strategy` & Parallel vs. Non-Parallel Replica Groups](#collective-concepts)

---

<a id="1-shared-base-parameters--chip-scaling-conventions"></a>
## 1. Shared Execution Parameters & Metric Conventions

<a id="shared-execution-parameters"></a>
### 1.1 Shared Execution Parameters

Every benchmark supports the following baseline execution and profiling parameters:

- **Interactive CLI (`tpums benchmark run`)**: Pass any parameter directly as a command-line flag (`bool` parameters take explicit `true` or `false` tokens):
    - **Single Value**: `tpums benchmark run <benchmark_name> --<param> <val>` *(e.g., `--m 4096 --transpose_a true`)*
    - **Multi-Value Sweep**: `tpums benchmark run <benchmark_name> --<param> <val1> <val2> ...` *(e.g., `--num_elements 134217728 268435456 --transpose_a true false`)*
- **YAML Config (`tpums benchmark run-config`)**: Configure inside `params:`, `sweep:`, `cases:`, or `cases_from_csv:` blocks.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `warmup_tries` | `int` | `10` | Number of untimed warmup iterations executed prior to measurement to complete XLA JIT compilation and stabilize device clocks. |
| `num_runs` | `int` | `10` | Minimum number of timed measurement iterations executed inside the timing loop. |
| `min_duration_s` | `float` | `0.0` | Minimum total measurement duration in seconds. When `> 0.0`, warmup also runs for at least `min(1.0, min_duration_s / 5)` seconds, and the measurement loop continues until both `num_runs` and `min_duration_s` are satisfied. Must be `0.0` for [`gemm_chained`](#212-chained-matrix-multiplication--determinism-check-gemm_chained-experimental). |

> [!NOTE]
> Dtype parameters (e.g. `dtype`, `in_dtype`, `out_dtype`) accept values from
> the curated `DType` enum in
> [`core/dtypes.py`](../src/accelerator_microbenchmarks/core/dtypes.py) (JAX
> dtypes outside it, such as `uint16` or `complex128`, are rejected) and are
> validated when the config is built: 64-bit dtypes (`float64`, `int64`) are
> rejected unless `jax_enable_x64` is enabled (e.g. `JAX_ENABLE_X64=1`; JAX
> would otherwise silently truncate them to 32 bits), and benchmarks restrict
> each field further via `DTYPE_CONSTRAINTS` (shown as `{...}` choices in
> `tpums benchmark run <benchmark_name> --help` and listed in
> `Supported dtypes for <field>: ...` error messages). `hbm`, the collectives,
> `device_to_device`, `host_to_device`, and `device_to_host` also reject
> sub-byte dtypes (`int4`, `float4_e2m1fn`), so their choices and lists omit
> them: TPUs store them packed below one byte per element, while these
> benchmarks count 1 byte per element.

<a id="per-device-vs-per-chip-scaling"></a>
### 1.2 Throughput & Bandwidth Metric Conventions (`<timing_domain>_<metric_type>_<hardware_scope>`)

Every throughput and bandwidth metric reported by TPUMS follows a standardized 3-part naming structure (for latency `_ms` and roofline `_pct` metrics, which share the same `<timing_domain>_` prefix, see [Primary Metrics Reference in `README.md`](../README.md#primary-metrics)):

```text
<timing_domain>_<metric_type>_<hardware_scope>[_<unit>]
# Examples:
#   xprof_tflops_per_chip
#   wall_clock_bandwidth_per_device_gb_s
```

1. **`<timing_domain>` (`wall_clock` vs. `xprof`)** — Timing measurement domain (see [Wall Clock vs. XProf Timing Metrics in `README.md`](../README.md#wall-clock-vs-xprof-timing-metrics)):
    - **`wall_clock_*`** *(always recorded)*: End-to-end execution time measured on the host Python runtime.
    - **`xprof_*`** *(recorded when `xprof_timing` is enabled)*: Pure on-device hardware kernel time extracted from XLA / XProf traces.
2. **`<metric_type>` & `[_<unit>]` (`tflops` vs. `bandwidth_..._gb_s`)** — Performance rate category:
    - **`tflops_<hardware_scope>`**: Floating-point compute throughput in teraFLOPs per second ($$10^{12}$$ FLOPS).
    - **`bandwidth_<hardware_scope>_gb_s`**: Data transfer bandwidth in decimal gigabytes per second ($$10^9$$ bytes/sec).
3. **`<hardware_scope>` (`per_device`, `per_chip`, `per_host`)** — Hardware
   aggregation boundary:
    - **Physical TPU Chips vs. Logical JAX Devices**:
        - **Dual-Device TPU Chips (e.g., `tpu7x`)**: Each physical TPU chip exposes **2 logical JAX devices** (`TPU:0` and `TPU:1`), so per-chip performance equals **`2 × per_device`**.
        - **Single-Device TPU Chips (e.g., `v6e`)**: Each physical TPU chip exposes **1 logical JAX device** (`TPU:0`), so per-chip performance equals **`1 × per_device`**.
    - **Per-Device, Per-Chip, and Per-Host Metrics**:
        - **`*_per_device` (`*_tflops_per_device`,
          `*_bandwidth_per_device_gb_s`)**: Reports the throughput or bandwidth
          achieved by a single logical JAX device (`scope="device"`).
        - **`*_per_chip` (`*_tflops_per_chip`,
          `*_bandwidth_per_chip_gb_s`)**: Reports the aggregate performance of a
          physical TPU chip (`scope="chip"`).
        - **`*_per_host` (`*_bandwidth_per_host_gb_s`)**: Reports the aggregate
          bandwidth across all local devices on a host (`scope="host"`).

The table below summarizes which `<metric_type>` and `<hardware_scope>` each benchmark reports:

| Benchmark | Primary Metric Type | Reported Throughput / Bandwidth Scope | Metric Meaning & Scope |
| :--- | :--- | :--- | :--- |
| **`gemm`** | **Compute Throughput (`TFLOPS`)** & Roofline (`%`) | **Both Per-Chip & Per-Device**<br>(`*_tflops_per_chip`, `*_tflops_per_device`) | **Measures** per-device and aggregate per-chip matrix compute throughput (`TFLOPS`), and **compares** against peak chip compute rooflines (`%`). |
| **`hbm`** | **Memory Bandwidth (`GB/s`)** & Roofline (`%`) | **All Three Scopes (`Per-Device`, `Per-Chip`, `Per-Host`; active measured, smaller derived, larger `None`)**<br>(`*_bandwidth_per_{device,chip,host}_gb_s`) | **Measures** HBM memory bandwidth (`GB/s`) across the selected hardware `scope` (`device` default, `chip`, or `host`), **derives** smaller scopes as the average share of the concurrent aggregate measurement, leaves larger scopes as `None` (no upward extrapolation), and **compares** against peak per-device HBM bandwidth rooflines (`%`). |
| **`host_to_device` / `device_to_host`** | **Host I/O Bandwidth (`GB/s`)** | **All Three Scopes (`Per-Device`, `Per-Chip`, `Per-Host`; active measured, smaller derived, larger `None`)**<br>(`*_bandwidth_per_{device,chip,host}_gb_s`) | **Measures** PCIe transfer bandwidth (`GB/s`) between the host CPU and the selected hardware `scope`: all local devices concurrently (`scope: "host"`, default), one physical chip (`scope: "chip"`), or one local device (`scope: "device"`), **deriving** smaller scopes as the average share while leaving larger scopes as `None` (since shared PCIe links do not scale linearly across cores). |
| **`device_to_device`** | **ICI Link Bandwidth (`GB/s`)** | **Per-Device**<br>(`*_bandwidth_per_device_gb_s`) | **Measures** point-to-point ICI link bandwidth (`GB/s`) between each logical device pair `(src_device_index, dst_device_index)`. |
| **`all_reduce` / `all_gather` / `all_to_all`** | **Collective Bus Bandwidth (`GB/s`)** | **Per-Chip**<br>(`*_bandwidth_per_chip_gb_s`) | **Measures** aggregate external ICI collective bus bandwidth (`GB/s`) per physical TPU chip across the active device mesh. |

---

<a id="2-per-benchmark-parameter--metric-specifications"></a>
## 2. Per-Benchmark Parameter & Metric Specifications

<a id="21-matrix-multiplication-gemm"></a>
### 2.1 Matrix Multiplication (`gemm`)

- **Operation**: Evaluates dense matrix multiplication independently across each accelerator device to measure full-chip compute throughput (**TFLOPS**) and compute roofline efficiency (**%**).
- **Kernel Formula**: $$C_{\text{out}} = (\alpha \cdot \text{op}(A) \times \text{op}(B) + \beta \cdot C) \odot (SF_0 \times SF_1)$$
- **FLOPs Formula (`total_flops`)**: By default (`alpha=1.0`, `beta=0.0`, `use_scaling_factors=False`), per-device FLOPs is `2 × M × N × K`. When optional scaling or accumulation flags are enabled, element-wise operations on the `M × N` output matrix are added:

  | Stage | Condition | Per-Device FLOPs |
  | :--- | :--- | :--- |
  | **Base Matrix Multiplication** ($$\text{op}(A) \times \text{op}(B)$$) | Always | `2 × M × N × K` |
  | **Scalar Multiplier** ($$\alpha \cdot \text{op}(A)\text{op}(B)$$) | `alpha != 1.0` | `+ M × N` |
  | **Unscaled Accumulator Addition** ($$+ C$$) | `beta == 1.0` | `+ M × N` |
  | **Scaled Accumulator Addition** ($$+ \beta \cdot C$$) | `beta != 0.0` and `beta != 1.0` | `+ 2 × M × N` |
  | **Quantized Output Rescaling** ($$\odot (SF_0 \times SF_1)$$) | `use_scaling_factors=True` | `+ 2 × M × N` |

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `m` | `int` | `1024` | Matrix dimension `M` (rows of `op(A)` and output matrix `C_out`). |
| `k` | `int` | `1024` | Contracting dimension `K` (columns of `op(A)` and rows of `op(B)`). |
| `n` | `int` | `1024` | Matrix dimension `N` (columns of `op(B)` and output matrix `C_out`). |
| `in_dtype` | `str` | `"bfloat16"` | Input operand data type (e.g., `bfloat16`, `float8_e4m3fn`, `float32`, `int8`). Selects the hardware peak TFLOPS ceiling for roofline efficiency. Dtypes absent from the hardware spec fall back to the `bfloat16` peak; dtypes the spec marks unsupported (`None`) omit compute roofline metrics. |
| `out_dtype` | `str` | `"bfloat16"` | Output accumulation and result tensor data type (e.g., `bfloat16`, `float32`). |
| `transpose_a` | `bool` | `False` | Transpose operand matrix `A` prior to multiplication (`shape = (k, m)` when `True`, `(m, k)` when `False`). |
| `transpose_b` | `bool` | `False` | Transpose operand matrix `B` prior to multiplication (`shape = (n, k)` when `True`, `(k, n)` when `False`). |
| `alpha` | `float` | `1.0` | Scalar multiplier applied to the matrix product ($$\alpha \cdot A \times B$$). |
| `beta` | `float` | `0.0` | Scalar multiplier applied to accumulator matrix `C` ($$\beta \cdot C$$). Allocates and adds an `M × N` accumulator matrix `C` when `beta != 0.0`. |
| `use_scaling_factors` | `bool` | `False` | Allocate and apply row-wise (`M × 1`) and column-wise (`1 × N`) `float32` scaling factor vectors for quantized/FP8 rescaling. |
| `seed` | `int` | `0` | PRNG seed for initializing input tensors in HBM. |

#### Reported Metrics

- **Compute Throughput & Roofline**:
  - `{wall_clock|xprof}_tflops_per_chip` — Aggregate compute throughput per physical TPU chip (`TFLOPS`).
  - `{wall_clock|xprof}_tflops_per_device` — Compute throughput per logical JAX device (`TFLOPS`).
  - `{wall_clock|xprof}_compute_roofline_efficiency_pct` — Achieved compute throughput as a percentage of hardware peak (`%`).
- **Latency (`ms`)**:
  - `{wall_clock|xprof}_{p50|p90|avg|std}_ms` — Iteration execution latency across median (`p50`), 90th percentile (`p90`), mean (`avg`), and standard deviation (`std`) in milliseconds (`ms`).
- **Benchmark Attributes**:
  - `total_flops` — Total floating-point operations executed per iteration.
  - `intensity` — Operational arithmetic intensity (`FLOPs / Byte`).
  - `roofline_tflops_limit_per_device` — Theoretical compute roofline ceiling per logical device (`TFLOPS`).
  - `peak_hbm_bw_per_device_gb_s` — Hardware asymptotic peak HBM bandwidth per logical device (`GB/s`).

<a id="211-gemm-throttling-gemm_throttling-experimental"></a>
#### 2.1.1 GEMM Throttling (`gemm_throttling`) [experimental]

Evaluates thermal steady-state performance and power throttling by soaking the
accelerator (`min_duration_s`) and measuring the degradation between the
`pre_soaking` and `post_soaking` windows, alongside device firmware telemetry.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `min_duration_s` | `float` | *(required)* | Duration of the untraced continuous thermal soak (Phase 2) in seconds. |
| `samples_per_run` | `int` | `25` | Number of iterations per execution window. |

| New Metric | Description |
| :--- | :--- |
| `*_pre_soaking_p50_ms`, `*_post_soaking_p50_ms`, `*_slowdown_ratio` | Steady-state degradation (latency and ratio) between `pre_soaking_window` and `post_soaking_window` across both XProf and Wall-Clock timing domains. |
| `*_pre_soaking_tflops_per_device`, `*_post_soaking_tflops_per_device` | Steady-state throughput degradation. |
| `hbm_peak_temp_c`, `peak_temp_c`, `peak_temp_source` | Post-soak peak temperature (°C) and sensor source. |
| `hbm_throttle_time_pct`, `vdd_core_throttle_time_pct` | Percentage of post-soak window time where throttling was active. |
| `hbm_throttle_mean_pct`, `vdd_core_throttle_mean_pct` | Time-weighted mean throttle percentage during the post-soak window. |
| `vdd_core_power_mean_w`, `hbm_power_mean_w`, `total_power_mean_w` | Time-weighted mean power level 1 meter readings (W). |
| `pstate_min`, `pstate_max`, `pstate_changes`, `pstate_requested` | Observed and requested DVFS P-state statistics. |

<a id="212-chained-matrix-multiplication--determinism-check-gemm_chained-experimental"></a>
#### 2.1.2 Chained Matrix Multiplication & Determinism Check (`gemm_chained`) [experimental]

Runs the same deterministic chain of `chain_length` matmuls
($$X_{t+1} = X_t \times B$$, square `B`, no collectives) on every device, then
compares a bit-exact, position-weighted checksum of each device's final `X`
within each host and across the slice to find miscomputing chips. Each device
generates its own `X_0` and `B` from `seed` (no copies between devices), so a
fault during input generation only affects that device. Only the chain is timed:
each timed run is checksummed after its timer stops, and the checksums of every
run are compared after the measurement loop.
`total_flops = chain_length × 2 × M × K²`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `m` | `int` | `1024` | Matrix dimension `M` (rows of `X`). |
| `k` | `int` | `1024` | Matrix dimension `K` (columns of `X`; `B` is `K × K`). |
| `in_dtype` | `str` | `"bfloat16"` | Input operand data type of each matmul. |
| `out_dtype` | `str` | `"bfloat16"` | Data type of `X` after each matmul. |
| `seed` | `int` | `0` | PRNG seed for initializing `X_0` and `B`. |
| `chain_length` | `int` | `1` | Number of chained matmuls per run (minimum `1`). |

`in_dtype` and `out_dtype` must be `bfloat16`, `float16`, `float32`,
`float8_e4m3fn` or `float8_e5m2`: the floats whose bit width (8, 16 or 32) the
checksum supports. Unlike `gemm`, there is no `n`, `transpose_a`,
`transpose_b`, `alpha`, `beta`, or `use_scaling_factors`.
`min_duration_s` is not supported and must remain `0.0` so that every host runs
the same number of timed iterations (`num_runs`) before the cross-host checksum
gather.

| New Metric | Description |
| :--- | :--- |
| `global_checksum_match` | **The check passed iff `True`.** `False` if any two checksums differ, across devices or timed runs; `None` if it cannot be decided (fewer than 2 devices, a missing checksum, every checksum degenerate, or a failed checksum gather). |
| `intra_host_checksum_match` | Same rule over this host's devices. |
| `checksum` | Non-degenerate device checksum (see `device_checksums`) held by more than half of the devices, or `None`. |
| `diverging_devices` | `[]` unless two checksums differ; then the devices with a timed run whose checksum differs from `checksum`, or every device if `checksum` is `None`. Each lists those runs (every run if `checksum` is `None`) in `failed_runs`, as `{"run", "checksum", "degenerate"}`, where `degenerate` marks an `inf`/`NaN` or all-zero result in that run. |
| `all_devices_valid_checksum` | Every device has a checksum for every timed run and none is degenerate (`inf`/`NaN` or all zeros). |
| `device_checksums` | This host's device checksums. A device's checksum is the one of more than half of its timed runs, or `None`. |

A failure in only some runs suggests a transient or intermittent fault, and a
failure in every run a persistent one. A failure in every run from some run on
suggests a corrupted `X_0` or `B`, which are generated once and reused by every
run. The same wrong checksum in every failed run suggests a deterministic fault,
such as a stuck bit.

**Limits**:

- Every timed run is checksummed (warmup is not), so increasing `num_runs` adds coverage.
- Not an HBM scan (`X` and `B` stay at fixed addresses); one shape and dtype covers a subset of MXU datapaths, so sweep for broader coverage.
- Choose `chain_length` so values stay finite; `inf`/`NaN` or all-zero results never pass.
- Golden-reference comparison and collective/H2D/D2H integrity are out of scope.

<a id="213-gemm-pulse-gemm_pulse-experimental"></a>
#### 2.1.3 GEMM Pulse (`gemm_pulse`) [experimental]

Combines multi-cycle idle-to-load pulsing (`PulseRunExecutor`) with
per-window deterministic chained matrix multiplication (`X <- X @ B`,
`chain_length` steps) and bit-exact cross-device/cross-host 64-bit checksum
verification (`GemmChainedBenchmark`).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_cycles` | `int` | `10` | Minimum number of active load cycles (`>= 2`, including `pre_soaking_window` and `post_soaking_window`) separated by `idle_s` host pauses. |
| `idle_s` | `float` | `0.05` | Host-side idle pause in seconds (`>= 0.0`) between active load cycles, bracketed by multi-host `sync_global_devices` barriers. |
| `chain_length` | `int` | `1` | Number of sequential matmuls (`X <- X @ B`) per `run_op` call. |
| `min_duration_s` | `float` | `0.0` | Minimum total duration in seconds (`>= 0.0`) of the untraced middle soaking phase between `pre_soaking_window` and `post_soaking_window`, synchronized across hosts via `process_allgather`. |
| `enable_checksum` | `bool` | `True` | Compute and verify per-device 64-bit checksums after each window. |

Cycle `0` starts with traced `pre_soaking_window`, intermediate cycles run
untraced `(idle_s -> window)` pulse cycles until at least `num_cycles - 2`
middle cycles and `min_duration_s` of middle soaking time have elapsed, and the
final cycle ends with an `idle_s` pause followed by traced
`post_soaking_window`. Emits `num_cycles`, `idle_s`, `enable_checksum`, all
`gemm_throttling` soaking and thermal metrics, plus all `gemm_chained` checksum
metrics when `enable_checksum=True`.

<a id="214-quantized-matrix-multiplication-gemm_quantized-experimental"></a>
#### 2.1.4 Quantized Matrix Multiplication (`gemm_quantized`) [experimental]

Evaluates end-to-end quantized matrix multiplication (`bfloat16` or `float32`
inputs `A[M, K]` and `B[K, N]` quantized row-wise/column-wise to
`float8_e4m3fn`, `float8_e5m2`, or `int8` inside the timed step before
`lax.dot_general`, with dynamic or static scaling and optional output
dequantization).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `m` | `int` | `1024` | Matrix dimension `M` (rows of `A` and output matrix `C`). |
| `k` | `int` | `1024` | Contracting dimension `K` (columns of `A` and rows of `B`). |
| `n` | `int` | `1024` | Matrix dimension `N` (columns of `B` and output matrix `C`). |
| `in_dtype` | `str` | `"bfloat16"` | Unquantized input operand dtype (e.g., `bfloat16`, `float32`). |
| `quant_dtype` | `str` | `"float8_e4m3fn"` | Quantized matmul operand dtype (e.g., `float8_e4m3fn`, `float8_e5m2`, `int8`). Selects the hardware peak TFLOPS ceiling for roofline efficiency. |
| `out_dtype` | `str` | `"bfloat16"` | Output tensor dtype (e.g., `bfloat16`, `float32`). |
| `scaling` | `str` | `"dynamic"` | Scale calibration: `"dynamic"` (per-row/column `amax` computed on the fly) or `"static"` (fixed clip bound `0.5 * Q_MAX`, i.e., compile-time scale `0.5`). |
| `dequantize` | `bool` | `False` | When `True`, multiply the matmul accumulator by the row and column dequantization scales before casting to `out_dtype`. |
| `seed` | `int` | `0` | PRNG seed for initializing input tensors in HBM. |

---

<a id="22-hbm-memory-bandwidth-hbm"></a>
### 2.2 HBM Memory Bandwidth (`hbm`)

- **Operation**: Executes 1D STREAM memory operations across the local devices
  selected by `scope` (`device` default, `chip`, or `host`) with zero
  cross-device ICI collectives to measure High-Bandwidth Memory (HBM)
  throughput (**GB/s**) and memory roofline efficiency (**%**).
- **Default `scope` & Multi-Device Scaling**: Defaults to `scope="device"`
  (unlike `host_to_device` and `device_to_host`, which default to
  `scope="host"` to saturate shared chip PCIe links and host memory channels)
  because HBM stacks are isolated per die: on-device `xprof_*` bandwidth scales
  linearly across `chip` and `host` with zero per-die contention, whereas
  sub-millisecond HBM kernels incur higher `wall_clock_*` latency at larger
  scopes from multi-device host dispatch and `block_until_ready()`
  synchronization skew (and `N×` buffer allocation).
- **Per-Device HBM Traffic Calculation (`data_size_mib`)**:
  `Array Transfer Count (Reads + Writes) × Single-Array Size (MiB)`
  - **Single-Array Size (`MiB`)**:
    `(num_elements * dtype_bytes) / (1024 * 1024)` — `num_elements` is
    specified **per device**. By default, `num_elements = 134,217,728`
    (`128 * 1024 * 1024` elements) and `dtype = bfloat16` (`2 bytes`),
    allocating **`256 MiB` per array per device**.
  - **Array Transfer Count (Reads + Writes)**: Total full-array HBM reads + writes executed by the kernel (`1` for `read_only`/`write_only`, `2` for `copy`/`scale`, `3` for `add`/`triad`).
  - **Per-Device HBM Traffic (`data_size_mib`)**:
    `Array Transfer Count (Reads + Writes) × Single-Array Size (MiB)` —
    Per-device HBM traffic moved per iteration (e.g., `2 × 256 MiB = 512 MiB`
    for `copy`, `3 × 256 MiB = 768 MiB` for `triad`). Aggregate bytes across
    the scope equal `num_devices × per_device_bytes`.

#### Supported `op_type` STREAM Kernels

| `op_type` | Kernel Expression | Input Arrays | Array Transfer Count (Reads + Writes) | FLOPs / Element |
| :--- | :--- | :--- | :--- | :--- |
| `copy` | `Y = X + 1.0` | `1` (`X`) | `2` (1 Read + 1 Write) | `1.0` |
| `scale` | `Y = c * X` | `1` (`X`) | `2` (1 Read + 1 Write) | `1.0` |
| `add` | `Z = X + Y` | `2` (`X, Y`) | `3` (2 Reads + 1 Write) | `1.0` |
| `triad` | `Z = X + c * Y` | `2` (`X, Y`) | `3` (2 Reads + 1 Write) | `2.0` |
| `read_only` (`read`) | `jnp.any(X)` | `1` (`X`) | `1` (1 Read) | `1.0` |
| `write_only` (`write`) | `jnp.full(shape, c)` | `0` | `1` (1 Write) | `0.0` |

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `op_type` | `str` | `"copy"` | Memory kernel operation (`copy`, `scale`, `add`, `triad`, `read_only`, `write_only`). |
| `num_elements` | `int` | `134217728` | Number of elements per 1D array per device (`Single-Array Size (MiB) = (num_elements * dtype_bytes) / (1024 * 1024)`). Default `134,217,728` (`128 * 1024 * 1024` elements) × `2 bytes` (`bfloat16`) = `256 MiB` per array per device. |
| `dtype` | `str` | `"bfloat16"` | Element data type (e.g., `bfloat16`, `float32`, `int8`). |
| `scope` | `str` | `"device"` | Hardware measurement scope: `"device"` (1 local device, default), `"chip"` (1 physical chip = `devices_per_chip` local devices), or `"host"` (all local devices). |
| `local_index` | `int` | `0` | Host-local index of the target unit within `scope`: the N-th local device (`scope="device"`) or N-th local chip (`scope="chip"`). Must be `0` for `scope="host"`. |

#### Reported Metrics

- **Memory Bandwidth & Roofline**:
  - `{wall_clock|xprof}_bandwidth_per_{device|chip|host}_gb_s` — Aggregate HBM
    memory bandwidth (`GB/s`). All three scope columns are always emitted: the
    column matching `scope` holds the **measured** bandwidth, smaller scopes are
    **derived** as the average share of the concurrent measurement (`chip` sets
    `per_device = per_chip / devices_per_chip`; `host` sets
    `per_device = per_host / num_devices` and
    `per_chip = per_device * devices_per_chip`), and larger scopes remain `None`
    (`-` in console tables, empty in `summary.csv`, `null` in `detailed.json`),
    with no upward extrapolation.
  - `{wall_clock|xprof}_memory_roofline_efficiency_pct` — Achieved HBM
    bandwidth as a percentage of hardware peak (`%`).
- **Latency (`ms`)**:
  - `{wall_clock|xprof}_{p50|p90|avg|std}_ms` — Iteration execution latency
    across median (`p50`), 90th percentile (`p90`), mean (`avg`), and standard
    deviation (`std`) in milliseconds (`ms`). For multi-device scopes
    (`scope="chip"` or `"host"` with multiple local devices), `xprof_*` latency
    defaults to `local_slowest_device` across the devices in the scope so
    `xprof_bandwidth_per_{chip,host}_gb_s` reflects the slowest device in the
    scope.
- **Benchmark Attributes**:
  - `scope` — Selected hardware scope (`"device"`, `"chip"`, or `"host"`).
  - `local_index` — Host-local unit index within `scope` (`0` for `"host"`).
  - `num_devices` — Number of local devices participating in the measurement
    (`1`, `devices_per_chip`, or the host's local device count).
  - `data_size_mib` — Per-device HBM traffic
    (`Array Transfer Count (Reads + Writes) × Single-Array Size (MiB)`) moved
    per iteration (`MiB`).
  - `intensity` — Operational arithmetic intensity (`FLOPs / Byte`).
  - `peak_hbm_bw_per_device_gb_s` — Hardware asymptotic peak HBM bandwidth per
    logical device (`GB/s`).

---

<a id="23-host-io-bandwidth-host_to_device-device_to_host"></a>
### 2.3 Host I/O Bandwidth (`host_to_device`, `device_to_host`)

- **Operations**:
  - **`host_to_device` (H2D)**: Transfers a payload buffer from host CPU RAM to local accelerator HBM over PCIe, synchronizing on full DMA transfer completion.
  - **`device_to_host` (D2H)**: Transfers a resident buffer from local accelerator HBM back to host CPU RAM over PCIe.
- **Transfer Backends** (`transfer_backend`): the two backends measure
  different things, and `tpu_sync` typically reports much higher bandwidth than
  `jax` for the same payload.
  - **`jax`**: Standard `jax.device_put` (H2D) / `jax.device_get` (D2H) of a
    `(rows, 128)` buffer sharded across the selected devices. Measures
    general-purpose framework transfer throughput on arbitrary pageable NumPy
    arrays, so the timed window includes per-call XLA host bounce-buffer
    allocation, CPU-side linearization/copying, and unpinned-memory overhead.
  - **`tpu_sync`**: [`tpu-sync-jax`](https://pypi.org/project/tpu-sync-jax/)
    `KVCacheManager` DMA between NUMA-local pinned host staging buffers and
    device HBM. A specialized zero-copy DMA path: host staging buffers are
    pre-allocated NUMA-local, then pinned and DMA-mapped once in `setup()`,
    and per-NUMA C++ worker threads dispatch to multiple devices concurrently.
    Bypassing per-iteration host allocation and CPU staging, it measures the
    hardware PCIe DMA ceiling.
    The buffer is shaped `(1, num_devices * rows_per_device, 128)`
    and sharded along axis 1, so each device shard `(1, rows_per_device, 128)`
    is a single `KVCacheManager` block and every iteration moves the full
    per-device payload with one `h2d([0], [0], [1])` / `d2h([0], [0], [1])`
    call. The `KVCacheManager` and its pinned host memory (`data_size_mib` per
    device) are allocated once in `setup()` and released in `teardown()` after
    each sweep point.
- **Timing**: Both directions record the `MARKER` annotation on the `/host:CPU`
  XProf plane, spanning dispatch through DMA completion.

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_size_mib` | `int` | `64` | Payload size transferred per device per iteration in Mebibytes (`1 MiB = 1024 * 1024` bytes). |
| `dtype` | `str` | `"bfloat16"` | Tensor element data type (e.g., `bfloat16`, `float32`, `int8`). |
| `transfer_backend` | `str` | `"tpu_sync"` | Transfer implementation: `"tpu_sync"` (default, used by canonical configs) or `"jax"`. |
| `scope` | `str` | `"host"` | Hardware measurement scope: `"device"` (1 local device), `"chip"` (1 physical chip = `devices_per_chip` local devices), or `"host"` (all local devices concurrently, default). |
| `local_index` | `int` | `0` | Host-local index of the target unit within `scope`: the N-th local device (`scope="device"`) or N-th local chip (`scope="chip"`). Must be `0` for `scope="host"`. |

#### Reported Metrics

- **PCIe Transfer Bandwidth**:
  - `{wall_clock|xprof}_bandwidth_per_{device|chip|host}_gb_s` — Aggregate
    host-to-device or device-to-host PCIe transfer bandwidth
    (`num_devices × per_device_bytes / latency_s`, `GB/s`). All three scope
    columns are always emitted: the column matching `scope` holds the
    **measured** bandwidth, smaller scopes are **derived** as the average share
    of that concurrent transfer (`chip` sets
    `per_device = per_chip / devices_per_chip`; `host` sets
    `per_device = per_host / num_devices` and
    `per_chip = per_device * devices_per_chip`), and larger scopes remain `None`
    (`-` in console tables, empty in `summary.csv`, `null` in `detailed.json`)
    because shared PCIe links (such as a dual-die TPU chip sharing one PCIe
    endpoint) cannot be extrapolated upward from a smaller scope.
- **Latency (`ms`)**:
  - `{wall_clock|xprof}_{p50|p90|avg|std}_ms` — Iteration execution latency across median (`p50`), 90th percentile (`p90`), mean (`avg`), and standard deviation (`std`) in milliseconds (`ms`).
- **Benchmark Attributes**:
  - `data_size_mib` — Payload volume transferred per device per iteration (`MiB`).
  - `transfer_backend` — Active transfer implementation (`"jax"` or `"tpu_sync"`).
  - `scope` — Selected hardware scope (`"device"`, `"chip"`, or `"host"`).
  - `local_index` — Host-local unit index within `scope` (`0` for `"host"`).
  - `num_devices` — Number of local devices transferring concurrently.

---

<a id="24-inter-chip-interconnect-device_to_device"></a>
### 2.4 Inter-Chip Interconnect (`device_to_device`)

- **Operation**: Measures point-to-point Inter-Chip Interconnect (ICI) link bandwidth (**GB/s**) and latency (**ms**) across all **`N × (N - 1)` directional device pairs** `(src_device_index, dst_device_index)` in the active topology (`src != dst`), supporting both single-host and multi-host topologies.

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_size_mib` | `int` | `1024` | Payload size transferred per directional link in Mebibytes (`MiB`). |
| `direction` | `str` | `"uni"` | Transfer direction mode: `"uni"` (unidirectional send `src → dst`) or `"bi"` (simultaneous bidirectional exchange `src ↔ dst`, doubling total bytes transferred across the pair). |
| `dtype` | `str` | `"bfloat16"` | Tensor element data type (e.g., `bfloat16`, `float32`). |
| `seed` | `int` | `0` | PRNG seed for initializing device buffers. |

#### Reported Metrics

- **ICI Link Bandwidth**:
  - `{wall_clock|xprof}_bandwidth_per_device_gb_s` — Point-to-point link bandwidth between `(src_device_index, dst_device_index)` (`GB/s`), also formatted in `stdout` as an **`N × N` pairwise device bandwidth matrix** (`src` rows × `dst` columns).
- **Latency (`ms`)**:
  - `{wall_clock|xprof}_{p50|p90|avg|std}_ms` — Iteration execution latency across median (`p50`), 90th percentile (`p90`), mean (`avg`), and standard deviation (`std`) in milliseconds (`ms`).
- **Benchmark Attributes**:
  - `src_device_index` / `dst_device_index` — Source and destination logical device indices for the measured pair.
  - `direction` — Active transfer direction (`"uni"` or `"bi"`).
  - `data_size_mib` — Payload size per directional stream (`MiB`).

---

<a id="25-collective-communications-all_reduce-all_gather-all_to_all"></a>
### 2.5 Collective Communications (`all_reduce`, `all_gather`, `all_to_all`)

- **Operation**: Evaluates distributed collective communication primitives (`all_reduce`, `all_gather`, `all_to_all`) across a configurable 2D or 3D device mesh (`mesh_shape`) and active collective axis pattern (`sharding_strategy`) on single-host or multi-host topologies.

<a id="collective-concepts"></a>
#### Collective Concepts: `mesh_shape`, `sharding_strategy` & Parallel vs. Non-Parallel Replica Groups

Understanding how TPUMS configures collective meshes and computes per-chip collective bus bandwidth requires four key concepts:

**1. `mesh_shape` vs. `sharding_strategy` (Logical Mesh vs. Active Collective Axes):**

- **`mesh_shape`** (e.g., `"2x4x4"` or `"2x2x1"`) arranges the available accelerator devices into an N-dimensional logical mesh. The product of the dimensions in `mesh_shape` must match the number of devices in the target mesh.
- **`sharding_strategy`** (e.g., `"2x2x1"` or `"2x4x4"`) has the **same number of dimensions** as `mesh_shape` and specifies which mesh dimensions actively participate in the collective operation:
  - **Active Collective Dimensions (`> 1`)**: Any dimension with a value `> 1` participates in the collective exchange.
  - **Replicated Dimensions (`== 1`)**: Any dimension with a value of `1` is excluded from the collective exchange, partitioning the mesh into independent replica subgroups along that dimension.
- **Concrete Example**: With `mesh_shape: "2x2x2"` (8 devices total across 3 dimensions) and `sharding_strategy: "2x2x1"`, only the first two dimensions have size `> 1`. The collective executes across 4-device subgroups (`2 × 2`), while the two planes along the third dimension run as independent parallel replica groups simultaneously.

**2. `parallel` vs. `non-parallel` Replica Groups (`replica_group_type`):**

On TPU architectures with two logical devices per physical chip—such as **`tpu7x`**, where logical devices `(0, 1)` reside on Chip 0, `(2, 3)` on Chip 1, etc.—how replica groups map to logical device IDs determines how much traffic traverses external Inter-Chip Interconnect (ICI) links:

- **`parallel` Replica Group** (e.g., `[0, 2, 4, 6]` and `[1, 3, 5, 7]`): Logical Device 0 of each chip forms one inter-chip collective ring (`[0, 2, 4, 6]`), while Logical Device 1 of each chip simultaneously forms a second disjoint inter-chip collective ring (`[1, 3, 5, 7]`). Because **both logical devices on every physical chip transmit across external ICI links simultaneously**, each physical chip drives **`C = 2` active rings in parallel**, with **`P = R - 1`** external inter-chip peers per ring of size `R`.
- **`non-parallel` Replica Group** (e.g., `[0, 1, 2, 3]`): Both logical devices (`0` and `1`) of the **same physical chip** participate in the **same** collective replica group (`C = 1` ring per chip). Because intra-chip communication between Logical Device 0 and Logical Device 1 stays internal to the physical chip rather than traversing external ICI links, each ring of size `R` has **`P = max(R - 2, 1)`** external inter-chip peers.
- **Reported Metadata**: TPUMS automatically classifies the first replica group
  from the JAX mesh by the devices' `core_on_chip` (`"parallel"` iff all
  devices share one core) and records `replica_group_type` (`"parallel"` or
  `"non-parallel"`) and `replica_group_rank` (`R`) in the output reports.

**3. Tensor Shape & Per-Device Shard Payload (`matrix_dim` → `shard_size_mib`):**

- For `all_reduce`, `all_gather`, and `all_to_all`, `matrix_dim` defines a 3D per-device tensor of shape **`(matrix_dim, 8, 128)`**, which aligns with native TPU matrix register tiles (`8 * 128 = 1,024` elements per unit of `matrix_dim`).
- **Per-Device Shard Size Calculation (`shard_size_mib`)**: `(matrix_dim * 8 * 128 * dtype_bytes) / (1024 * 1024)`
    - **1. Per-Device Shard Payload (`S`, in `Bytes`)**: `S = matrix_dim * 8 * 128 * dtype_bytes`
    - **2. Reported Shard Size (`shard_size_mib`, in `MiB`)**: `S / (1024 * 1024)` — With `dtype: bfloat16` (`2` bytes per element), `matrix_dim: 1024` produces **`2.00 MiB`** per device; `matrix_dim: 8192` produces **`16.00 MiB`** per device.

**4. Mesh Device Placement (`device_placement`):**

`device_placement` controls which physical device lands at each coordinate of
the `mesh_shape` logical mesh. It only takes effect when `mesh_shape` is set.
Because replica groups are derived from mesh coordinates, the placement changes
which devices exchange data in each collective and therefore the communication
paths the traffic takes.

- **`mesh_utils`** *(default)*: Uses
  `jax.experimental.mesh_utils.create_device_mesh(mesh_shape, devices=jax.devices())`,
  which assigns devices based on their physical topology coordinates.
- **`id_sorted`**: Sorts `jax.devices()` by ascending `device.id` and reshapes
  them into `mesh_shape`
  (`np.asarray(sorted(jax.devices(), key=lambda d: d.id)).reshape(mesh_shape)`).
- **When to use `id_sorted`**: Because the resulting replica-group device order
  differs from `mesh_utils`, `id_sorted` can yield higher collective bandwidth
  on some full-chip mesh shapes (e.g., `all_gather` on TPU v7x `2x2x2`). If
  bandwidth on a given topology looks lower than expected, benchmark both
  placements and compare.
- **Caveat**: `id_sorted` ignores physical topology and may map logical axes
  across physical ones on non-standard or sub-mesh shapes. Prefer the default
  `mesh_utils` for those cases.

#### Parameters

| Parameter | Type | Default | Applies To | Description |
| :--- | :--- | :--- | :--- | :--- |
| `mesh_shape` | `Optional[str]` | `None` | All Collectives | Logical TPU mesh dimensions formatted as `'DxDyDz'` or `'DxDy'` (e.g., `"2x2x1"`, `"2x4x4"`, `"4x4"`). Defaults to a 1D mesh over all devices if `None`. |
| `device_placement` | `str` | `"mesh_utils"` | All Collectives | Mesh device placement: `"mesh_utils"` (topology-aware `create_device_mesh`) or `"id_sorted"` (sorted ascending by `device.id`). Ignored when `mesh_shape` is `None`. See [Mesh Device Placement](#collective-concepts). |
| `sharding_strategy` | `Optional[str]` | `None` | All Collectives | Axis participation pattern matching the dimensionality of `mesh_shape` (e.g., `"2x2x1"`). Dimensions with value `> 1` participate in the collective; defaults to all mesh dimensions if `None`. |
| `matrix_dim` | `int` | `1024` | All Collectives | Leading dimension of the per-device tensor `(matrix_dim, 8, 128)`. |
| `dtype` | `str` | `"bfloat16"` | All Collectives | Tensor element data type (e.g., `bfloat16`, `float32`). |
| `reduce_op` | `str` | `"sum"` | `all_reduce` only | Reduction operator applied across active collective dimensions (`"sum"`, `"mean"`, `"max"`, `"min"`). |
| `seed` | `int` | `0` | All Collectives | PRNG seed for tensor initialization. |

#### Collective Bus Bandwidth Formulas (`transfer_size_per_chip_mib`)

Given per-device shard payload $$S$$ (bytes), replica group rank $$R$$
(`replica_group_rank`, devices per ring), external inter-chip peers $$P$$
($$R - 1$$ for single-device or `parallel` dual-device; $$R - 2$$ for
`non-parallel` dual-device), active rings per physical chip $$C$$ ($$2$$ for
`parallel` dual-device; $$1$$ otherwise), and devices per physical chip $$D$$
(`devices_per_chip`, e.g., $$2$$ for dual-device chips, $$1$$ for
single-device chips):

Each collective's per-device transfer size is topology-neutral; the ICI
benchmarks scale it to the chip by the number of devices of one chip that send
their own data ($$C$$, or $$D$$ for `all_to_all`).

| Benchmark | JAX Primitive | Per-Device Transfer Size | Per-Chip Scale | Per-Chip Transfer Size (`transfer_size_per_chip_mib`) |
| :--- | :--- | :--- | :--- | :--- |
| **`all_reduce`** | `jax.lax.psum` / `pmean` / `pmax` / `pmin` | $$2 \times S \times \frac{P}{R}$$ | $$C$$ | $$2 \times S \times \left(\frac{P}{R}\right) \times C$$ |
| **`all_gather`** | `jax.lax.all_gather` | $$S \times P$$ | $$C$$ | $$S \times P \times C$$ |
| **`all_to_all`** | `jax.lax.all_to_all` | $$S \times \frac{P}{R}$$ | $$D$$ | $$S \times \left(\frac{P}{R}\right) \times D$$ |

#### Reported Metrics

- **Collective Bus Bandwidth**:
  - `{wall_clock|xprof}_bandwidth_per_chip_gb_s` — Effective external ICI collective bus bandwidth per physical TPU chip (`GB/s`).
- **ICI Roofline (`all_to_all` only)**:
  - `peak_ici_bandwidth_per_chip_gb_s` — Theoretical peak all-to-all bus bandwidth per chip (`GB/s`).
  - `{wall_clock|xprof}_ici_roofline_efficiency_pct` — Achieved bus bandwidth as a percentage of `peak_ici_bandwidth_per_chip_gb_s` (`%`).
- **Latency (`ms`)**:
  - `{wall_clock|xprof}_{p50|p90|avg|std}_ms` — Iteration execution latency across median (`p50`), 90th percentile (`p90`), mean (`avg`), and standard deviation (`std`) in milliseconds (`ms`).
- **Benchmark Attributes**:
  - `shard_size_mib` — Per-device tensor shard payload (`S` expressed in `MiB`).
  - `replica_group_rank` — Number of participating devices (`R`) per collective replica group (product of dimensions `> 1` in `sharding_strategy`).
  - `replica_group_type` — Classified replica ring mode (`"parallel"` vs. `"non-parallel"`).
  - `transfer_size_per_chip_mib` — Effective external ICI transfer size per physical chip per collective step (`MiB`); the bytes behind `*_bandwidth_per_chip_gb_s`.
  - `transfer_size_per_device_mib` — Multi-slice DCN collectives (`dcn_*`) only: DCN transfer size per device per collective step (`MiB`); `*_bandwidth_per_host_gbps` scales it by the local device count.

---

> [!NOTE]
> Additional experimental benchmarks (such as ReduceScatter, FlashAttention, MoE Transformer layers, and custom fusion ops) are under active development — use with caution.
