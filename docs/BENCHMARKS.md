
# Benchmark Reference

Parameters, default values, metric conventions, and roofline/bandwidth formulas
for every TPUMS benchmark.

> [!TIP]
> For installation, CLI usage, and YAML configuration instructions, see the main **[`README.md`](../README.md)**.

---

<a id="1-shared-base-parameters--chip-scaling-conventions"></a>
## 1. Shared Execution Parameters & Metric Conventions

<a id="shared-execution-parameters"></a>
### 1.1 Shared Execution Parameters

Every benchmark supports the following baseline execution parameters:

- **Interactive CLI (`tpums benchmark run <name> --<param> <val> [val2 ...]`)**:
    Pass single values or space-separated sweeps (`bool` flags take explicit
    `true` or `false`, e.g., `--transpose_a true false`).
- **YAML Config (`tpums benchmark run-config`)**: Set under `params:`, `sweep:`,
    `cases:`, or `cases_from_csv:`.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `warmup_tries` | `int` | `10` | Untimed iterations before measurement (XLA JIT compile and clock settle). |
| `num_runs` | `int` | `10` | Minimum timed measurement iterations executed in the timing loop. |
| `min_duration_s` | `float` | `0.0` | Minimum measurement duration in seconds. When `> 0.0`, warmup runs at least `min(1.0, min_duration_s / 5)` s and measurement continues until both `num_runs` and `min_duration_s` are met. Must be `0.0` for [`gemm_chained`](#212-chained-matrix-multiplication--determinism-check-gemm_chained-experimental) and `> 0.0` for [`gemm_throttling`](#211-gemm-throttling-gemm_throttling-experimental); [`gemm_pulse`](#213-gemm-pulse-gemm_pulse-experimental) uses it as the untraced middle-phase duration. |

> [!NOTE]
> Dtype fields accept
> [`core/dtypes.py`](../src/accelerator_microbenchmarks/core/dtypes.py) `DType`
> values, narrowed per field by `DTYPE_CONSTRAINTS` (see `--help` choices;
> 64-bit `float64`/`int64` require `JAX_ENABLE_X64=1`). Benchmarks that count
> 1 byte/element (`hbm`, collectives, `device_to_device`, `host_to_device`,
> `device_to_host`, 2D activation/norm ops, and `rope`) reject sub-byte types
> (`int4`, `float4_e2m1fn`).

<a id="per-device-vs-per-chip-scaling"></a>
### 1.2 Throughput & Bandwidth Metric Conventions (`<timing_domain>_<metric_type>_<hardware_scope>`)

Throughput and bandwidth metrics follow
`<timing_domain>_<metric_type>_<hardware_scope>[_<unit>]` (e.g.,
`xprof_tflops_per_chip`, `wall_clock_bandwidth_per_device_gb_s`; see
[Primary Metrics in `README.md`](../README.md#primary-metrics)):

| Component | Values | Meaning |
| :--- | :--- | :--- |
| `<timing_domain>` | `wall_clock`, `xprof` | `wall_clock_*` (always recorded): host Python runtime latency. `xprof_*` (when `xprof_timing: true`): on-device kernel time from XLA/XProf traces (see [Wall Clock vs. XProf](../README.md#wall-clock-vs-xprof-timing-metrics)). |
| `<metric_type>[_<unit>]` | `tflops`, `bandwidth_..._gb_s`, `bandwidth_..._gbps` | Compute throughput in TFLOPS ($$10^{12}$$ FLOPS) or transfer rate in decimal GB/s ($$10^9$$ bytes/s). Exception: multi-slice DCN collectives (`dcn_*`) emit `_gbps` in decimal Gbit/s (`×8` bytes/s). |
| `<hardware_scope>` | `per_device`, `per_chip`, `per_host` | Aggregation boundary: one logical JAX device (`per_device`), one physical TPU chip (`per_chip = devices_per_chip × per_device`; `2×` on `tpu7x`, `1×` on `v6e`), or all local devices on a host (`per_host`). |

- **Latency (`ms`)**: Every benchmark reports
    `{wall_clock|xprof}_{p50,p90,avg,std}_ms` (median, p90, mean, std in ms).
- **Multi-Scope Rule (`hbm`, `host_to_device`, `device_to_host`)**: Always emits
    all three `*_bandwidth_per_{device,chip,host}_gb_s` columns: `scope` holds
    the **measured** rate, smaller scopes are **derived** (`per_device =
    per_chip / devices_per_chip` or `per_host / num_devices`; `per_chip =
    per_device * devices_per_chip`), and larger scopes are `None` (`-` in
    console, empty in CSV, `null` in JSON) with no upward extrapolation.

| Benchmark | Primary Metric Type | Reported Scope Columns |
| :--- | :--- | :--- |
| **`gemm`** | Compute Throughput (`TFLOPS`) & Roofline (`%`) | `*_tflops_per_{chip,device}` |
| **`hbm`** | Memory Bandwidth (`GB/s`) & Roofline (`%`) | `*_bandwidth_per_{device,chip,host}_gb_s` (measured/derived/`None` per `scope`) |
| **`host_to_device` / `device_to_host`** | Host I/O Bandwidth (`GB/s`) & Roofline (`%`) | `*_bandwidth_per_{device,chip,host}_gb_s` (measured/derived/`None` per `scope`) |
| **`device_to_device`** | Link Bandwidth (`GB/s`) | `*_bandwidth_per_device_gb_s` |
| **`all_reduce` / `all_gather` / `all_to_all`** | ICI Collective Bus Bandwidth (`GB/s`) | `*_bandwidth_per_chip_gb_s` (+ ICI roofline `%` for `all_to_all`) |
| **`dcn_all_reduce` / `dcn_all_to_all`** | Multi-Slice DCN Bus Bandwidth (`Gbit/s`) | `wall_clock_bandwidth_per_host_gbps` (`xprof_*` not populated) |

---

<a id="2-per-benchmark-parameter--metric-specifications"></a>
## 2. Per-Benchmark Parameter & Metric Specifications

<a id="21-matrix-multiplication-gemm"></a>
### 2.1 Matrix Multiplication (`gemm`)

- **Operation**: Evaluates dense matrix multiplication independently on each
    device to measure compute throughput (**TFLOPS**) and compute roofline
    efficiency (**%**).
- **Kernel Formula**: $$C_{\text{out}} = (\alpha \cdot \text{op}(A) \times \text{op}(B) + \beta \cdot C) \odot (SF_0 \times SF_1)$$
- **FLOPs Formula (`total_flops`)**: Default (`alpha=1.0`, `beta=0.0`,
    `use_scaling_factors=False`) is `2 × M × N × K` per device; optional scaling
    or accumulation flags add elementwise `M × N` terms:

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
| `in_dtype` | `str` | `"bfloat16"` | Input operand dtype (e.g., `bfloat16`, `float8_e4m3fn`, `float32`, `int8`). Selects `peak_tflops` in `roofline_tflops_limit_per_device` (dtypes absent from the spec fall back to `bfloat16`; dtypes marked `None` omit compute roofline metrics). |
| `out_dtype` | `str` | `"bfloat16"` | Output accumulation and result tensor data type (e.g., `bfloat16`, `float32`). |
| `transpose_a` | `bool` | `False` | Transpose operand matrix `A` prior to multiplication (`shape = (k, m)` when `True`, `(m, k)` when `False`). |
| `transpose_b` | `bool` | `False` | Transpose operand matrix `B` prior to multiplication (`shape = (n, k)` when `True`, `(k, n)` when `False`). |
| `alpha` | `float` | `1.0` | Scalar multiplier applied to the matrix product ($$\alpha \cdot A \times B$$). |
| `beta` | `float` | `0.0` | Scalar multiplier applied to accumulator matrix `C` ($$\beta \cdot C$$). Allocates and adds an `M × N` accumulator matrix `C` when `beta != 0.0`. |
| `use_scaling_factors` | `bool` | `False` | Allocate and apply row-wise (`M × 1`) and column-wise (`1 × N`) `float32` scaling factor vectors for quantized/FP8 rescaling. |
| `seed` | `int` | `0` | PRNG seed for initializing input tensors in HBM. |

> [!NOTE]
> **`float32` shares the `bfloat16` peak.** MXU multiplies take `bfloat16`
> inputs and accumulate in `float32` (see
> [TPU architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)),
> and the GEMM benchmarks pass no `precision` to the matmul, so JAX's default
> precision applies and a `float32` matmul runs as one `bfloat16` pass. JAX's
> [precision levels](https://docs.jax.dev/en/latest/201/precision.html#precision-the-classic-three-levels)
> affect only `float32` computations. On TPU:
>
> - `Precision.DEFAULT` (what TPUMS runs): fastest, least accurate; computes in
>   `bfloat16`.
> - `Precision.HIGH`: slower, more accurate; three `bfloat16` passes.
> - `Precision.HIGHEST`: slowest, most accurate; six `bfloat16` passes.

#### Reported Metrics

- **Compute Throughput & Roofline**:
    `{wall_clock|xprof}_tflops_per_{chip,device}` (`TFLOPS`) and
    `{wall_clock|xprof}_compute_roofline_efficiency_pct` (`%` of
    `roofline_tflops_limit_per_device`).
- **Benchmark Attributes**: `total_flops`, `intensity` (`FLOPs / Byte`),
    `roofline_tflops_limit_per_device = min(peak_tflops, intensity ×
    peak_hbm_bw_per_device_gb_s / 1000)` (`TFLOPS`, comparing memory-bound
    shapes against the HBM slope rather than MXU peak), and
    `peak_hbm_bw_per_device_gb_s` (`GB/s`).

<details class="experimental" markdown="1">
<summary><strong>2.1.1 GEMM Throttling (<code>gemm_throttling</code>)</strong> <code>[EXPERIMENTAL]</code></summary>
<a id="211-gemm-throttling-gemm_throttling-experimental"></a>

Evaluates thermal steady-state performance and power throttling by soaking the
accelerator (`min_duration_s`) between traced `pre_soaking` and `post_soaking`
windows alongside firmware telemetry. Inherits all matrix shape and dtype
parameters from `gemm`. Requires `xprof_timing: true`; `num_runs` is ignored,
and headline `xprof_*` latency/throughput stats come from the post-soak window
only.

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `min_duration_s` | `float` | *(required)* | Duration of the untraced continuous thermal soak (Phase 2) in seconds (`> 0.0`). |
| `samples_per_run` | `int` | `25` | Number of iterations per execution window. |

**Reported Metrics:**

| New Metric | Description |
| :--- | :--- |
| `*_pre_soaking_p50_ms`, `*_post_soaking_p50_ms`, `*_slowdown_ratio` | Pre- vs. post-soak latency and slowdown ratio across XProf and Wall-Clock timing domains. |
| `*_pre_soaking_tflops_per_device`, `*_post_soaking_tflops_per_device` | Pre- vs. post-soak per-device throughput (`TFLOPS`). |
| `hbm_peak_temp_c`, `peak_temp_c`, `hbm_peak_temp_c_pre_soaking`, `peak_temp_c_pre_soaking`, `peak_temp_source` | Post-soak and pre-soak peak temperatures (°C) and sensor source. |
| `hbm_throttle_time_pct`, `vdd_core_throttle_time_pct` | Percentage of the post-soak window where the firmware throttle counter is above zero (active duration, not throttling magnitude; host-level max across devices). |
| `hbm_throttle_mean_pct`, `vdd_core_throttle_mean_pct` | Time-weighted mean throttle percentage during the post-soak window (throttling magnitude; host-level max across devices). |
| `vdd_core_power_mean_w`, `hbm_power_mean_w`, `total_power_mean_w` | Time-weighted mean power level 1 meter readings (W). |
| `pstate_min`, `pstate_max`, `pstate_changes`, `pstate_requested` | Observed and requested DVFS P-state statistics. |

**Interpreting Results:**

- **Focus on performance and thermal metrics first**: Use
    `*_post_soaking_tflops_per_device` and `*_slowdown_ratio` to evaluate actual
    compute impact, alongside `hbm_throttle_*` and peak temperatures
    (`hbm_peak_temp_c`, `peak_temp_c`) for thermal anomalies.
- **Do not treat `vdd_core_throttle_time_pct > 0` alone as performance
    throttling**: Because `*_throttle_time_pct` measures only how much of the
    window the firmware counter stayed above zero, light chip-level power
    management under sustained load can produce a high `time_pct` while
    `vdd_core_throttle_mean_pct` remains low and throughput matches healthy
    peers.
- **Cluster health checks and pass/fail thresholds**: TPUMS reports raw
    microbenchmark measurements without pass/fail verdicts. For automated
    cluster health checks with curated pass/fail thresholds, use
    [Cluster Health Scanner (CHS)](https://github.com/GoogleCloudPlatform/cluster-health-scanner).

**Example Config (`tpums benchmark run-config`):**
[`configs/sample_configs/gemm_throttling.yaml`](../configs/sample_configs/gemm_throttling.yaml)
</details>

<details class="experimental" markdown="1">
<summary><strong>2.1.2 Chained Matrix Multiplication &amp; Determinism Check (<code>gemm_chained</code>)</strong> <code>[EXPERIMENTAL]</code></summary>
<a id="212-chained-matrix-multiplication--determinism-check-gemm_chained-experimental"></a>

Runs the same deterministic chain of `chain_length` matmuls
($$X_{t+1} = X_t \times B$$, square `B`, no collectives) on every device, then
compares a bit-exact, position-weighted checksum of each device's final `X`
within each host and across the slice to find miscomputing chips. Each device
generates its own `X_0` and `B` from `seed` (no copies between devices), so a
fault during input generation only affects that device. Only the chain is timed:
each timed run is checksummed after its timer stops, and the checksums of every
run are compared after the measurement loop.
`total_flops = chain_length × 2 × M × K²`.

**Parameters:**

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

**Reported Metrics:**

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

**Limits:**

- Every timed run is checksummed (warmup is not), so increasing `num_runs` adds
    coverage.
- Not an HBM scan (`X` and `B` stay at fixed addresses); one shape and dtype
    covers a subset of MXU datapaths, so sweep for broader coverage.
- Choose `chain_length` so values stay finite; `inf`/`NaN` or all-zero results
    never pass.
- Golden-reference comparison and collective/H2D/D2H integrity are out of scope.

**Example Config (`tpums benchmark run-config`):**
[`configs/sample_configs/gemm_chained.yaml`](../configs/sample_configs/gemm_chained.yaml)
</details>

<details class="experimental" markdown="1">
<summary><strong>2.1.3 GEMM Pulse (<code>gemm_pulse</code>)</strong> <code>[EXPERIMENTAL]</code></summary>
<a id="213-gemm-pulse-gemm_pulse-experimental"></a>

Combines multi-cycle idle-to-load pulsing with deterministic chained matrix
multiplication (`X <- X @ B`, `chain_length` steps) and cross-device/cross-host
64-bit checksum verification. Inherits all `gemm_chained` parameters (`m`, `k`,
`in_dtype`, `out_dtype`, `seed`, `chain_length`). Each pulse window runs one
chain of `chain_length` matmuls (no `samples_per_run`). Requires
`xprof_timing: true`; `num_runs` is ignored, and headline `xprof_*`
latency/throughput stats come from the post-soak window only.

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_cycles` | `int` | `10` | Minimum active load cycles (`>= 2`, including `pre_soaking_window` and `post_soaking_window`) separated by `idle_s` host pauses. |
| `idle_s` | `float` | `0.05` | Host-side idle pause in seconds (`>= 0.0`) between active load cycles, bracketed by multi-host barriers. |
| `min_duration_s` | `float` | `0.0` | Minimum total duration in seconds (`>= 0.0`) of the untraced middle soaking phase between `pre_soaking_window` and `post_soaking_window`. |
| `enable_checksum` | `bool` | `True` | Compute and verify per-device 64-bit checksums after each window. |

**Reported Metrics:** Cycle `0` runs traced `pre_soaking_window`, middle cycles
run untraced `(idle_s -> window)` pulses until `>= num_cycles - 2` cycles and
`min_duration_s` elapse, and the final cycle runs `idle_s` then traced
`post_soaking_window`. Emits `num_cycles`, `idle_s`, `enable_checksum`, all
`gemm_throttling` soaking/thermal metrics, and (when `enable_checksum=True`) all
`gemm_chained` checksum metrics.

**Example Config (`tpums benchmark run-config`):**
[`configs/sample_configs/gemm_pulse.yaml`](../configs/sample_configs/gemm_pulse.yaml)
</details>

<details class="experimental" markdown="1">
<summary><strong>2.1.4 Quantized Matrix Multiplication (<code>gemm_quantized</code>)</strong> <code>[EXPERIMENTAL]</code></summary>
<a id="214-quantized-matrix-multiplication-gemm_quantized-experimental"></a>

Per-channel quantized matrix multiplication replicated on each device:

- **Timed Scope**: Quantizes `A[M, K]` rowwise to `quant_dtype` and (when
    `prequantize_rhs=False`) `B[K, N]` columnwise to `rhs_quant_dtype` in
    `float32` (including mixed-precision `FP8 × INT4` when
    `rhs_quant_dtype="int4"`), or reads offline pre-quantized `B_q` when
    `prequantize_rhs=True`; runs `jax.lax.dot_general` (`int32` accumulator for
    `int8`, `float32` when `A` is FP8), and optionally dequantizes in `float32`
    (`acc * (sf_a * sf_b)`) before casting to `out_dtype`.
- **FLOPs & Traffic**: `total_flops = 2 × M × N × K` (in-scope quant/dequant
    cost is reflected in latency only); 1-pass HBM bytes =
    `M × K × in_bytes + K × N × rhs_bytes + M × N × out_bytes` (`rhs_bytes` is
    `rhs_quant_bytes` when `prequantize_rhs=True`, else `in_bytes`).
- **Platform Note**: On `tpu7x`, FP8 uses the native FP8 MXU while `int8` runs
    at the `bfloat16` rate; on `v6e`, `int8` is the native fast path while FP8
    has no native MXU path and runs below `bfloat16` peak (with
    `float8_e4m3fn` slower than `float8_e5m2`, expected rather than a TPUMS
    bug).

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `m` | `int` | `1024` | Matrix dimension `M` (`>= 1`; rows of `A` and output matrix `C`). |
| `k` | `int` | `1024` | Contracting dimension `K` (`>= 1`; columns of `A` and rows of `B`). |
| `n` | `int` | `1024` | Matrix dimension `N` (`>= 1`; columns of `B` and output matrix `C`). |
| `in_dtype` | `str` | `"bfloat16"` | Unquantized input operand dtype (`"bfloat16"` or `"float32"`). |
| `quant_dtype` | `str` | `"float8_e4m3fn"` | Quantized LHS (and default RHS) operand dtype (`"float8_e4m3fn"`, `"float8_e5m2"`, or `"int8"`). Selects `peak_tflops` in `roofline_tflops_limit_per_device`. |
| `rhs_quant_dtype` | `str` | `` `quant_dtype` `` | Quantized RHS (`B`) operand dtype. Defaults to `quant_dtype`; also supports `"int4"` when `quant_dtype` is `float8_e4m3fn` or `float8_e5m2` (`W4A8`). |
| `out_dtype` | `str` | `"bfloat16"` | Output tensor dtype (`"bfloat16"` or `"float32"`). |
| `prequantize_rhs` | `bool` | `False` | When `True` (weight-offline), pre-quantizes `B` to `rhs_quant_dtype` once before the timed loop and quantizes only `A` on the fly; when `False`, quantizes both `A` and `B` inside the timed step. |
| `scaling` | `str` | `"dynamic"` | `"dynamic"` (per-row/column `amax` on the fly, `scale = max(amax, 1e-7) / Q_MAX`) or `"static"` (compile-time clip `0.5 * Q_MAX`, `scale = 0.5`). |
| `dequantize` | `bool` | `False` | Multiply accumulator in `float32` by `sf_a * sf_b` before casting to `out_dtype`. |
| `seed` | `int` | `0` | PRNG seed (`>= 0`) for initializing input tensors in HBM. |

**Reported Metrics:** Same compute metric set as [`gemm`](#21-matrix-multiplication-gemm)
(`*_compute_roofline_efficiency_pct` compares against
`roofline_tflops_limit_per_device = min(peak_tflops, intensity × peak_hbm_bw /
1000)` with `peak_tflops` keyed by `quant_dtype`), plus `quant_dtype`,
`rhs_quant_dtype`, `prequantize_rhs`, `scaling`, and `dequantize`.

**Example Config (`tpums benchmark run-config`):**
[`configs/sample_configs/gemm_quantized.yaml`](../configs/sample_configs/gemm_quantized.yaml)
</details>

---

<a id="22-hbm-memory-bandwidth-hbm"></a>
### 2.2 HBM Memory Bandwidth (`hbm`)

- **Operation**: Executes 1D STREAM memory operations across the local devices
    selected by `scope` (`device` default, `chip`, or `host`) with no ICI
    collectives to measure HBM bandwidth (**GB/s**) and memory roofline
    efficiency (**%**).
- **Default `scope`**: Defaults to `scope="device"` because HBM stacks are
    isolated per die (`xprof_*` scales linearly across `chip`/`host`), whereas
    larger scopes add host-dispatch and barrier skew to `wall_clock_*`.
- **Per-Device HBM Traffic (`data_size_mib`)**: Each 1D array has size
    `single_array_mib = (num_elements * dtype_bytes) / 2^20` per device
    (default `134,217,728 × 2 B = 256 MiB` for `bfloat16`). Per-device traffic
    per iteration is `transfer_count × single_array_mib` (`512 MiB` for `copy`,
    `768 MiB` for `triad`); aggregate bytes across `scope` equal
    `num_devices × per_device_bytes`.

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
| `num_elements` | `int` | `134217728` | Elements per 1D array per device (`256 MiB` per array at default `bfloat16`; see formula above). |
| `dtype` | `str` | `"bfloat16"` | Element data type (integer or floating; `bool`, `complex`, and sub-byte dtypes rejected — e.g., `bfloat16`, `float32`, `int8`). |
| `scope` | `str` | `"device"` | Hardware measurement scope: `"device"` (1 local device, default), `"chip"` (1 physical chip = `devices_per_chip` local devices), or `"host"` (all local devices). |
| `local_index` | `int` | `0` | Host-local index of the target unit within `scope`: the N-th local device (`scope="device"`) or N-th local chip (`scope="chip"`). Must be `0` for `scope="host"`. |

#### Reported Metrics

- **Memory Bandwidth & Roofline**:
    - `{wall_clock|xprof}_bandwidth_per_{device|chip|host}_gb_s` — HBM bandwidth
        (`GB/s`); measured/derived/`None` scope columns follow
        [§1.2](#per-device-vs-per-chip-scaling). For multi-device scopes
        (`chip` or `host` with `> 1` device), `xprof_*` latency defaults to
        `local_slowest_device`.
    - `{wall_clock|xprof}_memory_roofline_efficiency_pct` — Achieved HBM
        bandwidth as a percentage of hardware peak (`%`).
- **Benchmark Attributes**: `scope`, `local_index`, `num_devices` (`1`,
    `devices_per_chip`, or host device count), `data_size_mib` (per-device HBM
    bytes moved per iteration, see above), `intensity` (`FLOPs / Byte`), and
    `peak_hbm_bw_per_device_gb_s` (`GB/s`).

---

<a id="23-host-io-bandwidth-host_to_device-device_to_host"></a>
### 2.3 Host I/O Bandwidth (`host_to_device`, `device_to_host`)

- **Operations**:
    - **`host_to_device` (H2D)**: Transfers a buffer from host RAM to local HBM
        over PCIe, synchronizing on DMA completion.
    - **`device_to_host` (D2H)**: Transfers a resident buffer from local HBM
        back to host RAM over PCIe.
- **Transfer Backends** (`transfer_backend`):
    - **`jax`**: `jax.device_put` (H2D) / `jax.device_get` (D2H) of a
        `(rows, 128)` buffer sharded across the selected devices; includes
        per-call XLA host bounce-buffer allocation, CPU staging, and
        unpinned-memory overhead.
    - **`tpu_sync`**: [`tpu-sync-jax`](https://pypi.org/project/tpu-sync-jax/)
        `KVCacheManager` DMA between pinned host staging buffers and HBM, shaped
        `(1, num_devices * rows_per_device, 128)` so one `h2d`/`d2h` call moves
        one `(1, rows_per_device, 128)` block (`data_size_mib`) per device.
        Setup allocation is untimed, approximating the PCIe ceiling.
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
    `{wall_clock|xprof}_bandwidth_per_{device|chip|host}_gb_s`
    (`num_devices × per_device_bytes / latency_s`, `GB/s`;
    measured/derived/`None` scope columns follow
    [§1.2](#per-device-vs-per-chip-scaling) since shared chip PCIe links cannot
    be extrapolated upward).
- **Host Link Roofline** (`summary.csv` / `detailed.json` only):
    `roofline_bound_by` (`"pcie"`),
    `peak_link_bandwidth_per_{device|chip|host}_gb_s` (`HardwareSpec.pcie ×
    ceil(num_devices / devices_per_chip)`, e.g., `32 GB/s` on `v6e`, `64 GB/s`
    on `tpu7x`; a single device at `scope="device"` shares the full chip link;
    omitted if unset), and `{wall_clock|xprof}_roofline_efficiency_pct`
    (`tpu_sync` approaches the practical PCIe ceiling after encoding/TLP
    overhead; lower `jax` values reflect host staging; host
    root-complex/NUMA/DRAM limits are not modeled).
- **Benchmark Attributes**: `data_size_mib`, `transfer_backend`, `scope`,
    `local_index`, and `num_devices`.

---

<a id="24-inter-chip-interconnect-device_to_device"></a>
### 2.4 Device-to-Device Transfers (`device_to_device`)

- **Operation**: Measures transfer bandwidth (**GB/s**) and latency (**ms**)
    across all **`N × (N - 1)` directional device pairs**
    `(src_device_index, dst_device_index)` (`src != dst`) on single-host or
    multi-host topologies, over ICI across chips or the on-package die-to-die
    link within a dual-device chip (e.g., `tpu7x`).

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_size` | `str` | `"1GiB"` | Payload size transferred per directional link: a whole number with an IEC unit (`B`, `KiB`, `MiB`, `GiB`, `TiB`; `1 KiB = 1024 B`), e.g. `"4096B"`, `"4KiB"`, `"256MiB"`. Kept as written. Must be a positive whole number of `dtype` elements. Bare numbers and ambiguous units (`4K`, `4KB`) are rejected. To sweep sizes, list them (e.g. `[4KiB, 1MiB, 256MiB]`). |
| `direction` | `str` | `"uni"` | Transfer direction mode: `"uni"` (unidirectional send `src → dst`) or `"bi"` (simultaneous bidirectional exchange `src ↔ dst`, doubling total bytes transferred across the pair). |
| `dtype` | `str` | `"bfloat16"` | Tensor element data type (e.g., `bfloat16`, `float32`). |
| `seed` | `int` | `0` | PRNG seed for initializing device buffers. |

#### Reported Metrics

- **Link Bandwidth**:
    - `{wall_clock|xprof}_bandwidth_per_device_gb_s` — Pairwise transfer
        bandwidth (`GB/s`, also printed in `stdout` as an `N × N` `src × dst`
        matrix; `"bi"` counts bytes in both directions).
- **Benchmark Attributes**:
    - `src_device_index` / `dst_device_index` — Positions in the
        topology-ordered 1D mesh (`mesh_utils.create_device_mesh((N,))`, printed
        at setup as `[D2D Mesh Mapping]`), not `device.id`.
    - `direction` — Active transfer direction (`"uni"` or `"bi"`).
    - `data_size` — Payload size per directional stream, as written in the
        config (e.g., `4KiB`).
    - `data_size_bytes` — Exact payload size in bytes (for sorting or grouping).

> [!NOTE]
> At small sizes (KiB range), the measured time is dominated by fixed
> per-transfer overhead, so `*_bandwidth_per_device_gb_s` reflects that overhead
> rather than link bandwidth.

---

<a id="25-collective-communications-all_reduce-all_gather-all_to_all"></a>
### 2.5 ICI Collective Communications (`all_reduce`, `all_gather`, `all_to_all`)

- **Operation**: Evaluates intra-slice ICI collectives (`all_reduce`,
    `all_gather`, `all_to_all`) across a configurable 2D/3D device mesh
    (`mesh_shape`) and active collective axis pattern (`sharding_strategy`) on
    single-host or multi-host topologies.

<a id="collective-concepts"></a>
#### Collective Concepts: `mesh_shape`, `sharding_strategy` & Parallel vs. Non-Parallel Replica Groups

**1. `mesh_shape` vs. `sharding_strategy` (Logical Mesh vs. Collective Axes):**

- **`mesh_shape`** (e.g., `"2x4x4"`, `"2x2x1"`) reshapes the slice's devices
    into an N-D logical mesh (if the product does not match the device count or
    cannot be built, TPUMS logs a warning and falls back to the 1D mesh over all
    devices).
- **`sharding_strategy`** (e.g., `"2x2x1"`) matches `mesh_shape`'s
    dimensionality: axes `> 1` participate in the collective, while axes `== 1`
    partition the mesh into independent replica groups (e.g., `mesh_shape:
    "2x2x2"` + `sharding_strategy: "2x2x1"` runs two concurrent 4-device
    `2 × 2` groups). Ignored when `mesh_shape` is `None`.

**2. `parallel` vs. `non-parallel` Replica Groups (`replica_group_type`):**

On dual-device TPU chips (e.g., **`tpu7x`**, where logical devices `(0, 1)`
reside on Chip 0, `(2, 3)` on Chip 1, etc.), replica-group device mapping
determines external ICI link traffic:

- **`parallel` Replica Group** (e.g., `[0, 2, 4, 6]` and `[1, 3, 5, 7]`): Core 0
    and Core 1 of each chip form two disjoint inter-chip rings, driving
    **`C = 2` active rings per chip** with **`P = R - 1`** external peers per
    ring of size `R`.
- **`non-parallel` Replica Group** (e.g., `[0, 1, 2, 3]`): Both cores of a chip
    share one ring (`C = 1`), keeping intra-chip traffic on-package so each ring
    has **`P = max(R - D, 1)`** external peers (`D = devices_per_chip`).
- **Classification Rule**: `"parallel"` iff `devices_per_chip > 1`, `R > 1`, and
    every device in the first replica group shares the same `core_on_chip`.
    Single-device chips (e.g., **`v6e`**, `D = 1`) always report
    `"non-parallel"` (`C = 1`, `P = max(R - 1, 1)`).

**3. Tensor Shape & Per-Device Shard (`matrix_dim` → `shard_size_mib`):**

`matrix_dim` defines a 3D per-device tensor `(matrix_dim, 8, 128)` with byte
size `S = matrix_dim × 8 × 128 × dtype_bytes` (`shard_size_mib = S / 2^20`; for
`bfloat16`, `matrix_dim: 1024` → `2.00 MiB`, `8192` → `16.00 MiB`).

**4. Mesh Device Placement (`device_placement`, when `mesh_shape` is set):**

- **`mesh_utils`** *(default)*: Topology-aware
    `mesh_utils.create_device_mesh(mesh_shape, devices=jax.devices())`.
- **`id_sorted`**: Reshapes `sorted(jax.devices(), key=lambda d: d.id)` into
    `mesh_shape`. Can improve bandwidth on some full-chip meshes (e.g.,
    `all_gather` on `tpu7x` `2x2x2`), though it ignores physical topology on
    sub-meshes.

#### Parameters

| Parameter | Type | Default | Applies To | Description |
| :--- | :--- | :--- | :--- | :--- |
| `mesh_shape` | `Optional[str]` | `None` | All ICI Collectives | Logical mesh `'DxDyDz'` or `'DxDy'` (e.g., `"2x4x4"`). Defaults to a 1D mesh over all devices if `None`; an invalid shape logs a warning and falls back to the 1D mesh. |
| `device_placement` | `str` | `"mesh_utils"` | All ICI Collectives | `"mesh_utils"` (topology-aware) or `"id_sorted"` (sorted by `device.id`). Ignored when `mesh_shape` is `None`. |
| `sharding_strategy` | `Optional[str]` | `None` | All ICI Collectives | Active collective axis pattern matching `mesh_shape` dimensionality (axes `> 1` participate; defaults to all mesh axes if `None`). Ignored when `mesh_shape` is `None`. |
| `matrix_dim` | `int` | `1024` | All ICI Collectives | Leading dimension of `(matrix_dim, 8, 128)`. `all_to_all` floors `matrix_dim` to a multiple of `R` (raises `ValueError` if `< R`) and reports the input as `requested_matrix_dim`. |
| `dtype` | `str` | `"bfloat16"` | All ICI Collectives | Element dtype (e.g., `bfloat16`, `float32`; `all_reduce` requires integer or floating). |
| `reduce_op` | `str` | `"sum"` | `all_reduce` only | Reduction operator (`"sum"`, `"mean"`, `"max"`, `"min"`). |
| `seed` | `int` | `0` | All ICI Collectives | PRNG seed for tensor initialization. |

#### Collective Bus Bandwidth Formulas (`transfer_size_per_chip_mib`)

Symbols: $$S$$ per-device shard bytes, $$R$$ `replica_group_rank`, $$D$$
`devices_per_chip`, $$P$$ external inter-chip peers ($$R - 1$$ for single-device
or `parallel` dual-device; $$\max(R - D, 1)$$ for `non-parallel` dual-device),
and $$C$$ sending rings per chip ($$2$$ for `parallel` dual-device, else $$1$$;
`all_to_all` scales by $$D$$ because every core sends distinct chunks):

| Benchmark | JAX Primitive | Per-Device Transfer Size | Per-Chip Scale | Per-Chip Transfer Size (`transfer_size_per_chip_mib`) |
| :--- | :--- | :--- | :--- | :--- |
| **`all_reduce`** | `jax.lax.psum` / `pmean` / `pmax` / `pmin` | $$2 \times S \times \frac{P}{R}$$ | $$C$$ | $$2 \times S \times \left(\frac{P}{R}\right) \times C$$ |
| **`all_gather`** | `jax.lax.all_gather` | $$S \times P$$ | $$C$$ | $$S \times P \times C$$ |
| **`all_to_all`** | `jax.lax.all_to_all` | $$S \times \frac{P}{R}$$ | $$D$$ | $$S \times \left(\frac{P}{R}\right) \times D$$ |

#### Reported Metrics

- **Collective Bus Bandwidth & Roofline**:
    - `{wall_clock|xprof}_bandwidth_per_chip_gb_s` — Effective external ICI bus
        bandwidth per physical TPU chip (`GB/s`).
    - `peak_ici_bandwidth_per_chip_gb_s` and
        `{wall_clock|xprof}_ici_roofline_efficiency_pct` (`all_to_all` only).
- **Benchmark Attributes**:
    - `shard_size_mib` (`S / 2^20`) and `transfer_size_per_chip_mib` (`MiB`).
    - `replica_group_rank` — Devices per replica group (`R`): product of the
        `mesh_shape` sizes on axes where `sharding_strategy > 1` (or all devices
        when `mesh_shape` is `None`).
    - `replica_group_type` (`"parallel"` or `"non-parallel"`) and
        `requested_matrix_dim` (`all_to_all` only).

<details class="experimental" markdown="1">
<summary><strong>2.5.1 Reduce-Scatter (<code>reduce_scatter</code>)</strong> <code>[EXPERIMENTAL]</code></summary>
<a id="251-reduce-scatter-reduce_scatter-experimental"></a>

Benchmarks distributed sum reduction + scatter
(`jax.lax.psum_scatter(..., tiled=True)`) across active ICI mesh axes
(`sharding_strategy`). Each participating device holds a per-device input shard
of shape `(R, matrix_dim, 256)` (where $$R$$ is `replica_group_rank`) and
receives one reduced `(1, matrix_dim, 256)` output chunk of byte size
$$S_{\text{chunk}} = \text{matrix\_dim} \times 256 \times \text{dtype\_bytes}$$
— giving full per-device input shard $$S = R \times S_{\text{chunk}}$$
(consistent with $$S$$ in the §2.5 symbol table above).

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mesh_shape` | `Optional[str]` | `None` | Logical TPU mesh dimensions `'DxDyDz'` or `'DxDy'` (defaults to 1D mesh over all devices if `None`). |
| `device_placement` | `str` | `"mesh_utils"` | `"mesh_utils"` or `"id_sorted"` (ignored when `mesh_shape` is `None`). |
| `sharding_strategy` | `Optional[str]` | `None` | Active collective axis pattern matching `mesh_shape` dimensionality (axes `> 1` participate; ignored when `mesh_shape` is `None`). |
| `matrix_dim` | `int` | `1024` | Middle dimension of per-device input `(R, matrix_dim, 256)` ($$S = R \times S_{\text{chunk}}$$) and reduced output chunk `(1, matrix_dim, 256)` ($$S_{\text{chunk}} = \text{matrix\_dim} \times 256 \times \text{dtype\_bytes}$$). |
| `dtype` | `str` | `"bfloat16"` | Element dtype (integer or floating non-sub-byte dtypes, e.g., `bfloat16`, `float32`, `int32`; always sum reduction, no `reduce_op`). |
| `seed` | `int` | `0` | PRNG seed for tensor initialization. |

**Bus Bandwidth Formula & Reported Metrics:**

- **Per-Device / Per-Chip Transfer Size**:
    $$S \times \frac{P}{R} = S_{\text{chunk}} \times P$$ per device;
    $$S \times (P / R) \times C = S_{\text{chunk}} \times P \times C$$ per chip
    (`transfer_size_per_chip_mib`), where $$C$$ is sending rings per chip from
    the §2.5 table (`2` for `parallel` dual-device, else `1`). Earlier runs that
    set $$S = S_{\text{chunk}}$$ under-reported `shard_size_mib`,
    `transfer_size_per_chip_mib`, and bandwidth by $$R$$.
- **Reported Metrics**: `{wall_clock|xprof}_bandwidth_per_chip_gb_s`,
    `shard_size_mib` ($$S / 2^{20}$$, full per-device input),
    `transfer_size_per_chip_mib`, `replica_group_rank`, and
    `replica_group_type`.

**Example Config (`tpums benchmark run-config`):**
[`configs/tpu7x/2x2x1/reduce_scatter.yaml`](../configs/tpu7x/2x2x1/reduce_scatter.yaml)
</details>

---

<a id="26-multi-slice-dcn-collectives-dcn_all_reduce-dcn_all_to_all"></a>
### 2.6 Multi-Slice DCN Collectives (`dcn_all_reduce`, `dcn_all_to_all`)

- **Operation**: Evaluates cross-slice Data Center Network (DCN) collectives
    over a 2D `("dcn", "ici")` mesh across `R = num_slices >= 2` equal-size TPU
    slices (each device exchanges `(matrix_dim, 8, 128)` shards with the device
    at the same ICI coordinate in every other slice). `xprof_*` metrics are not
    populated on DCN (no `MARKER` scope; canonical configs set
    `xprof_timing: false`).
- **Parameters**: `matrix_dim` (`int`, default `1024`; `dcn_all_to_all` floors
    it to a multiple of `R`, raises if `< R`, and reports
    `requested_matrix_dim`), `dtype` (`str`, default `"bfloat16"`, sub-byte
    rejected; integer/floating only for `dcn_all_reduce`), `reduce_op`
    (`"sum"` default, `"mean"`, `"max"`, `"min"` — `dcn_all_reduce` only), and
    `seed` (`int`, default `0`).
- **Formulas & Reported Metrics**: Per-device DCN transfer bytes
    (`transfer_size_per_device_mib` in `MiB`) are
    $$2 \times S \times \frac{R - 1}{R}$$ (`dcn_all_reduce`) and
    $$S \times \frac{R - 1}{R}$$ (`dcn_all_to_all`). Emits
    `wall_clock_bandwidth_per_host_gbps` in **Gbit/s** — an exception to the
    `_gb_s` convention, computed as
    `(per_device_bytes × 8 / (latency_s × 1e9)) × local_device_count` — plus
    `shard_size_mib`, `transfer_size_per_device_mib`, `replica_group_rank`
    (`num_slices`), and `requested_matrix_dim` (`dcn_all_to_all`).
- **Canonical Configs**:
    [`configs/tpu7x/2x2x1/dcn_all_reduce.yaml`](../configs/tpu7x/2x2x1/dcn_all_reduce.yaml),
    [`configs/tpu7x/2x2x1/dcn_all_to_all.yaml`](../configs/tpu7x/2x2x1/dcn_all_to_all.yaml), and
    [`configs/v6e/2x2/dcn_all_reduce.yaml`](../configs/v6e/2x2/dcn_all_reduce.yaml).

---

<a id="3-experimental-benchmarks"></a>
## 3. Experimental Benchmarks

> [!NOTE]
> Experimental benchmarks (`is_experimental=True`) may change parameters or
> metrics across releases. They are omitted from `tpums benchmark list` unless
> `--include_experimental` is passed, and run via `tpums benchmark run` or
> `run-config` as usual. Experimental GEMM
> ([`gemm_throttling`](#211-gemm-throttling-gemm_throttling-experimental),
> [`gemm_chained`](#212-chained-matrix-multiplication--determinism-check-gemm_chained-experimental),
> [`gemm_pulse`](#213-gemm-pulse-gemm_pulse-experimental),
> [`gemm_quantized`](#214-quantized-matrix-multiplication-gemm_quantized-experimental))
> and [`reduce_scatter`](#251-reduce-scatter-reduce_scatter-experimental) live
> under [§2.1](#21-matrix-multiplication-gemm) and
> [§2.5](#25-collective-communications-all_reduce-all_gather-all_to_all); the
> remaining 11 are grouped below.

**Shared Memory-Bound Metric Set (Cards 3.1–3.3):** Every memory-bound op in
3.1–3.3 reports `{wall_clock|xprof}_bandwidth_per_{device,chip}_gb_s`,
`{wall_clock|xprof}_memory_roofline_efficiency_pct`, `total_bytes_mib`,
`intensity`, and `peak_hbm_bw_per_device_gb_s` (plus [§1.2](#per-device-vs-per-chip-scaling)
latency stats).

<details class="experimental" markdown="1">
<summary><strong>3.1 2D Activation &amp; Normalization Ops (<code>swiglu</code>, <code>sigmoid</code>, <code>rmsnorm</code>, <code>layernorm</code>, <code>softmax</code>)</strong> <code>[EXPERIMENTAL]</code></summary>
<a id="31-2d-activation--normalization-ops-experimental"></a>

Memory-bound 2D activation and normalization kernels replicated on every
device, supporting forward (`"fwd"`) and backward VJP (`"bwd"`) passes.

**Parameters (shared across all 5 benchmarks):**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `batch` | `int` | `1024` | Leading batch/token dimension (`>= 1`). |
| `dim` | `int` | `4096` | Hidden feature dimension (`>= 1`). For `swiglu`, input `X` is `(batch, 2 * dim)` and output `Y` is `(batch, dim)`; for all other ops, both `X` and `Y` are `(batch, dim)`. |
| `dtype` | `str` | `"bfloat16"` | Activation dtype (floating non-sub-byte dtypes, e.g., `bfloat16`, `float32`). |
| `mode` | `str` | `"fwd"` | `"fwd"` (forward) or `"bwd"` (backward VJP w.r.t. activations and learnable norm weights). |

**Per-Benchmark Formulas & 1-Pass Algorithmic HBM Traffic (Per Device):**

| Benchmark | Forward Formula | `fwd` HBM Bytes ($$\times \text{dtype\_bytes}$$) | `bwd` HBM Bytes ($$\times \text{dtype\_bytes}$$) |
| :--- | :--- | :--- | :--- |
| `swiglu` | $$Y = (A \odot \sigma(A)) \odot B$$, $$[A, B] = \text{split}(X, 2)$$ | $$3 \times \text{batch} \times \text{dim}$$ | $$5 \times \text{batch} \times \text{dim}$$ |
| `sigmoid` | $$Y = \sigma(X)$$ | $$2 \times \text{batch} \times \text{dim}$$ | $$3 \times \text{batch} \times \text{dim}$$ |
| `rmsnorm` | $$Y = \frac{X}{\sqrt{\text{mean}(X^2) + 10^{-6}}} \odot W$$ | $$2 \times \text{batch} \times \text{dim} + \text{dim}$$ | $$3 \times \text{batch} \times \text{dim} + 2 \times \text{dim}$$ |
| `layernorm` | $$Y = \frac{X - \mu}{\sqrt{\sigma^2 + 10^{-6}}} \odot \gamma + \beta$$ (`float32` stats) | $$2 \times \text{batch} \times \text{dim} + 2 \times \text{dim}$$ | $$3 \times \text{batch} \times \text{dim} + 4 \times \text{dim}$$ |
| `softmax` | $$Y = \text{softmax}(X)$$ over last axis (`float32` compute) | $$2 \times \text{batch} \times \text{dim}$$ | $$3 \times \text{batch} \times \text{dim}$$ |

**Reported Metrics:** Shared §3 memory-bound metric set.

**Example Config (`tpums benchmark run-config`):** Sample configs in
[`swiglu.yaml`](../configs/sample_configs/cop/swiglu.yaml),
[`sigmoid.yaml`](../configs/sample_configs/cop/sigmoid.yaml), and
[`rmsnorm.yaml`](../configs/sample_configs/cop/rmsnorm.yaml); `layernorm` and
`softmax` take the exact same parameters (just change `name`).
</details>

<details class="experimental" markdown="1">
<summary><strong>3.2 Rotary Positional Embedding (<code>rope</code>)</strong> <code>[EXPERIMENTAL]</code></summary>
<a id="32-rotary-positional-embedding-rope-experimental"></a>

Half-split (`rotate_half`) Rotary Positional Embedding replicated on every
device. Activations `X` are a 2D `(batch * seq_len, heads * head_dim)` matrix
in `dtype` rotated in `float32` against a precomputed `float32`
`(seq_len, head_dim)` cos/sin table before casting back to `dtype`. At the
sample config sizes, `rope` is latency-bound on both `tpu7x` and `v6e` (~12–34%
of peak HBM bandwidth).

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `batch` | `int` | `32` | Batch size dimension (`>= 1`). |
| `seq_len` | `int` | `1024` | Sequence length dimension (`>= 1`). |
| `heads` | `int` | `32` | Number of attention heads (`>= 1`). |
| `head_dim` | `int` | `128` | Per-head dimension (`>= 2`, must be even). |
| `dtype` | `str` | `"bfloat16"` | Activation input/output dtype (floating non-sub-byte dtypes, e.g., `bfloat16`, `float32`). |
| `mode` | `str` | `"fwd"` | `"fwd"` or `"bwd"` (VJP w.r.t. `X`; the cos/sin table is constant, so both passes move `2 × batch × seq_len × heads × head_dim × dtype_bytes + seq_len × head_dim × 4` bytes per device). |

**Reported Metrics:** Shared §3 memory-bound metric set.

**Example Config (`tpums benchmark run-config`):**
[`configs/sample_configs/rope.yaml`](../configs/sample_configs/rope.yaml)
</details>

<details class="experimental" markdown="1">
<summary><strong>3.3 Quantization Kernels (<code>quantization</code>, <code>quantize_transpose</code>)</strong> <code>[EXPERIMENTAL]</code></summary>
<a id="33-quantization-kernels-quantization-quantize_transpose-experimental"></a>

Dynamic rowwise quantization (`quantization`) and fused Pallas transposed
quantization (`quantize_transpose`) replicated on every device:

- **`quantization`**: Computes rowwise scales
    $$SF_i = Q_{\max} / \max_j |X_{i,j}|$$ (`m` scales) and writes
    $$\text{OUT}_{M \times N} = \text{cast\_quant}(X \odot SF)$$. Per-device
    1-pass HBM bytes:
    $$M \cdot N \cdot \text{in\_bytes} + M \cdot N \cdot \text{quant\_bytes} + M \cdot \text{scale\_bytes}$$.
- **`quantize_transpose`**: Fused Pallas kernel producing
    $$\text{OUT}_{N \times M} = \text{cast\_quant}(X^\top \odot SF)$$ with `n`
    columnwise scales of `X` without writing an unquantized $$X^\top$$ to HBM
    (1-pass HBM bytes replace $$M \cdot \text{scale\_bytes}$$ with
    $$N \cdot \text{scale\_bytes}$$). Derives Pallas tile dimensions
    automatically and records `block_m` and `block_n` in output metadata.

**Parameters (shared by both benchmarks):**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `m` | `int` | `4096` | Input matrix rows `M` (`>= 1`). |
| `n` | `int` | `4096` | Input matrix columns `N` (`>= 1`). |
| `dtype` | `str` | `"bfloat16"` | Input activation dtype (`bfloat16`, `float16`, `float32`, or `float64` with `JAX_ENABLE_X64=1`). |
| `quant_dtype` | `str` | `"float8_e4m3fn"` | Target quantized output dtype (floating or integer dtypes, including sub-byte `float4_e2m1fn` and `int4`). |
| `scale_dtype` | `str` | `"bfloat16"` | Scaling factor dtype (`bfloat16`, `float16`, `float32`, or `float64` with `JAX_ENABLE_X64=1`). |

**Reported Metrics:** Shared §3 memory-bound metric set, plus `quant_dtype`,
`scale_dtype`, and (`quantize_transpose` only) `block_m` and `block_n`.

**Example Config (`tpums benchmark run-config`):**
[`configs/sample_configs/verify_study/quantization.yaml`](../configs/sample_configs/verify_study/quantization.yaml).
Same parameters for `quantize_transpose`; just change `name` (`block_m` and
`block_n` are derived automatically, not passed in `params:`).
</details>

<details class="experimental" markdown="1">
<summary><strong>3.4 FlashAttention, MoE Layer &amp; Smoke-Test Ops (<code>attention_flashed</code>, <code>transformer_layer_moe</code>, <code>simple_add</code>)</strong> <code>[EXPERIMENTAL]</code></summary>
<a id="34-flashattention-moe-layer--smoke-test-ops-experimental"></a>

**1. FlashAttention (`attention_flashed`):**

Executes `jax.nn.dot_product_attention` across MHA/GQA shapes, sharding `Q` and
`K/V` across the 1D device mesh along the head dimension whenever `num_q_heads`
and `num_kv_heads` are divisible by the device count (otherwise replicating).
Global forward FLOPs is $$2 \cdot B \cdot S^2 \cdot H_q \cdot D$$
(`causal=True`) or $$4 \cdot B \cdot S^2 \cdot H_q \cdot D$$ (`causal=False`);
`mode="bwd"` times `jax.vjp` + `vjp_fn` (`3.5×` forward FLOPs), divided by the
active query-head shard degree.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dtype` | `str` | `"bfloat16"` | Tensor dtype (`"bfloat16"` or `"float32"`). |
| `mode` | `str` | `"fwd"` | Execution pass: `"fwd"` or `"bwd"`. |
| `causal` | `bool` | `True` | Apply causal attention masking. |
| `batch` | `int` | `1` | Batch size `B` (`>= 1`). |
| `seq_len` | `int` | `8192` | Sequence length `S` (`>= 1`). |
| `num_q_heads` | `int` | `56` | Number of query attention heads `H_q` (`>= 1`). |
| `num_kv_heads` | `int` | `56` | Number of key/value heads `H_kv` (`>= 1`). |
| `head_dim` | `int` | `128` | Per-head dimension `D` (`>= 1`). |

- **Reported Metrics (`attention_flashed`)**: Same compute metric set as
    [`gemm`](#21-matrix-multiplication-gemm)
    (`{wall_clock|xprof}_tflops_per_{device,chip}`,
    `{wall_clock|xprof}_compute_roofline_efficiency_pct` against
    `roofline_tflops_limit_per_device`, `total_flops`, `intensity`, and
    `peak_hbm_bw_per_device_gb_s`).
- **Example Config (`tpums benchmark run-config`)**:

    ```yaml
    benchmark:
      name: attention_flashed
      xprof_timing: true
      params:
        warmup_tries: 2
        num_runs: 10
        dtype: bfloat16
        mode: fwd
        causal: true
        batch: 1
        seq_len: 8192
        num_q_heads: 32
        num_kv_heads: 8
        head_dim: 128
    ```

**2. MoE Transformer Layer Proxy (`transformer_layer_moe`):**

Forward-pass layer proxy (`batch=1`: standardization norm + QKV/output
attention matmuls `×3` + a dense 2-matmul FFN proxy `up ×2`, `down` standing in
for MoE experts with **no routing**, plus residual adds =
$$20 \times \text{mslen} \times \text{model\_dim}^2$$ global FLOPs),
partitioning `mslen` and `model_dim` across the mesh when divisible by the
device count.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_dim` | `int` | `7168` | Transformer hidden dimension (`>= 1`). |
| `mslen` | `int` | `1024` | Micro-sequence length (`>= 1`). |
| `dtype` | `str` | `"bfloat16"` | Layer floating-point dtype (e.g., `bfloat16`, `float32`). |

- **Reported Metrics (`transformer_layer_moe`)**: Same compute metric set as
    [`gemm`](#21-matrix-multiplication-gemm) (with `*_compute_roofline_efficiency_pct`
    compared against `roofline_tflops_limit_per_device`).
- **Example Config (`tpums benchmark run-config`)**:

    ```yaml
    benchmark:
      name: transformer_layer_moe
      xprof_timing: true
      params:
        warmup_tries: 2
        num_runs: 10
        dtype: bfloat16
        model_dim: 7168
        mslen: 1024
    ```

**3. Elementwise Add Smoke Check (`simple_add`):**

1D elementwise vector addition (`Z = X + Y`) on fixed `bfloat16` arrays of
shape `(size,)` replicated across devices for harness and profiler smoke
testing.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `size` | `int` | `1048576` | Number of 1D elements per vector (`>= 1`, `1024 * 1024` default). |
| `dtype` | `str` | `"bfloat16"` | Inherited parameter field (`simple_add` allocates fixed `bfloat16` inputs). |

- **Reported Metrics (`simple_add`)**:
    `{wall_clock|xprof}_{p50|p90|avg|std}_ms` latency statistics only (no
    throughput or roofline columns).

```yaml
benchmark:
  name: simple_add
  params:
    warmup_tries: 2
    num_runs: 5
    size: 1048576
```
</details>
