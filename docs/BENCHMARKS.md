
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
| `min_duration_s` | `float` | `0.0` | Minimum total measurement duration in seconds. When `> 0.0`, warmup also runs for at least `min(1.0, min_duration_s / 5)` seconds, and the measurement loop continues until both `num_runs` and `min_duration_s` are satisfied. |

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
3. **`<hardware_scope>` (`per_device` vs. `per_chip`)** — Hardware aggregation boundary:
    - **Physical TPU Chips vs. Logical JAX Devices**:
        - **Dual-Device TPU Chips (e.g., `tpu7x`)**: Each physical TPU chip exposes **2 logical JAX devices** (`TPU:0` and `TPU:1`), so per-chip performance equals **`2 × per_device`**.
        - **Single-Device TPU Chips (e.g., `v6e`)**: Each physical TPU chip exposes **1 logical JAX device** (`TPU:0`), so per-chip performance equals **`1 × per_device`**.
    - **Per-Device vs. Per-Chip Metrics**:
        - **`*_per_device` (`*_tflops_per_device`, `*_bandwidth_per_device_gb_s`)**: Reports the throughput or bandwidth achieved by a single logical JAX device.
        - **`*_per_chip` (`*_tflops_per_chip`, `*_bandwidth_per_chip_gb_s`)**: Reports the aggregate performance of a physical TPU chip.

The table below summarizes which `<metric_type>` and `<hardware_scope>` each benchmark reports:

| Benchmark | Primary Metric Type | Reported Throughput / Bandwidth Scope | Metric Meaning & Scope |
| :--- | :--- | :--- | :--- |
| **`gemm`** | **Compute Throughput (`TFLOPS`)** & Roofline (`%`) | **Both Per-Chip & Per-Device**<br>(`*_tflops_per_chip`, `*_tflops_per_device`) | **Measures** per-device and aggregate per-chip matrix compute throughput (`TFLOPS`), and **compares** against peak chip compute rooflines (`%`). |
| **`hbm`** | **Memory Bandwidth (`GB/s`)** & Roofline (`%`) | **Per-Device**<br>(`*_bandwidth_per_device_gb_s`) | **Measures** HBM memory bandwidth (`GB/s`) on a single local logical device (`device_id`), and **compares** against peak per-device HBM bandwidth rooflines (`%`). |
| **`host_to_device` / `device_to_host`** | **Host I/O Bandwidth (`GB/s`)** | **Per-Device**<br>(`*_bandwidth_per_device_gb_s`) | **Measures** PCIe transfer bandwidth (`GB/s`) between the host CPU and a single local logical device (`device 0`). |
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
| `in_dtype` | `str` | `"bfloat16"` | Input operand data type (e.g., `bfloat16`, `float8_e4m3fn`, `float32`, `int8`). Selects the hardware peak TFLOPS ceiling for roofline efficiency. |
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

---

<a id="22-hbm-memory-bandwidth-hbm"></a>
### 2.2 HBM Memory Bandwidth (`hbm`)

- **Operation**: Executes 1D STREAM memory operations on a target local device (`device_id`) to measure High-Bandwidth Memory (HBM) throughput (**GB/s**) and memory roofline efficiency (**%**).
- **HBM Traffic Calculation (`total_bytes_mib`)**: `Array Transfer Count (Reads + Writes) × Single-Array Size (MiB)`
  - **Single-Array Size (`MiB`)**: `(num_elements * dtype_bytes) / (1024 * 1024)` — By default, `num_elements = 134,217,728` (`128 * 1024 * 1024` elements) and `dtype = bfloat16` (`2 bytes`), allocating **`256 MiB` per array**.
  - **Array Transfer Count (Reads + Writes)**: Total full-array HBM reads + writes executed by the kernel (`1` for `read_only`/`write_only`, `2` for `copy`/`scale`, `3` for `add`/`triad`).
  - **Total HBM Traffic (`total_bytes_mib`)**: `Array Transfer Count (Reads + Writes) × Single-Array Size (MiB)` — Total HBM traffic moved per iteration (e.g., `2 × 256 MiB = 512 MiB` for `copy`, `3 × 256 MiB = 768 MiB` for `triad`).

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
| `num_elements` | `int` | `134217728` | Number of elements per 1D array (`Single-Array Size (MiB) = (num_elements * dtype_bytes) / (1024 * 1024)`). Default `134,217,728` (`128 * 1024 * 1024` elements) × `2 bytes` (`bfloat16`) = `256 MiB` per array. |
| `dtype` | `str` | `"bfloat16"` | Element data type (e.g., `bfloat16`, `float32`, `int8`). |
| `device_id` | `int` | `0` | Target local accelerator device index (`0` by default). |

#### Reported Metrics

- **Memory Bandwidth & Roofline**:
  - `{wall_clock|xprof}_bandwidth_per_device_gb_s` — HBM memory bandwidth on the target logical JAX device (`GB/s`).
  - `{wall_clock|xprof}_memory_roofline_efficiency_pct` — Achieved HBM bandwidth as a percentage of hardware peak (`%`).
- **Latency (`ms`)**:
  - `{wall_clock|xprof}_{p50|p90|avg|std}_ms` — Iteration execution latency across median (`p50`), 90th percentile (`p90`), mean (`avg`), and standard deviation (`std`) in milliseconds (`ms`).
- **Benchmark Attributes**:
  - `total_bytes_mib` — Total HBM traffic (`Array Transfer Count (Reads + Writes) × Single-Array Size (MiB)`) moved per iteration (`MiB`).
  - `intensity` — Operational arithmetic intensity (`FLOPs / Byte`).
  - `peak_hbm_bw_per_device_gb_s` — Hardware asymptotic peak HBM bandwidth per logical device (`GB/s`).

---

<a id="23-host-io-bandwidth-host_to_device-device_to_host"></a>
### 2.3 Host I/O Bandwidth (`host_to_device`, `device_to_host`)

- **Operations**:
  - **`host_to_device` (H2D)**: Transfers a payload buffer from host CPU RAM to local accelerator HBM over PCIe, synchronizing on full DMA transfer completion.
  - **`device_to_host` (D2H)**: Transfers a resident buffer from local accelerator HBM back to host CPU RAM over PCIe.

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data_size_mib` | `int` | `64` | Payload size transferred per iteration in Mebibytes (`1 MiB = 1024 * 1024` bytes). |
| `dtype` | `str` | `"bfloat16"` | Nominal benchmark data type tag recorded in output reports. |

#### Reported Metrics

- **PCIe Transfer Bandwidth**:
  - `{wall_clock|xprof}_bandwidth_per_device_gb_s` — Host-to-device or device-to-host PCIe transfer bandwidth (`GB/s`).
- **Latency (`ms`)**:
  - `{wall_clock|xprof}_{p50|p90|avg|std}_ms` — Iteration execution latency across median (`p50`), 90th percentile (`p90`), mean (`avg`), and standard deviation (`std`) in milliseconds (`ms`).
- **Benchmark Attributes**:
  - `data_size_mib` / `total_bytes_mib` — Payload volume transferred per iteration (`MiB`).

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
  - `data_size_mib` / `total_bytes_mib` — Payload size per directional stream (`data_size_mib`, `MiB`) and total bytes transferred across the pair (`total_bytes_mib`, equal to `2 * data_size_mib` when `direction == "bi"`).

---

<a id="25-collective-communications-all_reduce-all_gather-all_to_all"></a>
### 2.5 Collective Communications (`all_reduce`, `all_gather`, `all_to_all`)

- **Operation**: Evaluates distributed collective communication primitives (`all_reduce`, `all_gather`, `all_to_all`) across a configurable 2D or 3D device mesh (`mesh_shape`) and active collective axis pattern (`sharding_strategy`) on single-host or multi-host topologies.

<a id="collective-concepts"></a>
#### Collective Concepts: `mesh_shape`, `sharding_strategy` & Parallel vs. Non-Parallel Replica Groups

Understanding how TPUMS configures collective meshes and computes per-chip collective bus bandwidth requires three key concepts:

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
- **Reported Metadata**: TPUMS automatically classifies the compiled HLO replica groups and records `replica_group_type` (`"parallel"` or `"non-parallel"`) and `replica_group_rank` (`R`) in the output reports.

**3. Tensor Shape & Per-Device Shard Payload (`matrix_dim` → `shard_size_mib`):**

- For `all_reduce`, `all_gather`, and `all_to_all`, `matrix_dim` defines a 3D per-device tensor of shape **`(matrix_dim, 8, 128)`**, which aligns with native TPU matrix register tiles (`8 * 128 = 1,024` elements per unit of `matrix_dim`).
- **Per-Device Shard Size Calculation (`shard_size_mib`)**: `(matrix_dim * 8 * 128 * dtype_bytes) / (1024 * 1024)`
    - **1. Per-Device Shard Payload (`S`, in `Bytes`)**: `S = matrix_dim * 8 * 128 * dtype_bytes`
    - **2. Reported Shard Size (`shard_size_mib`, in `MiB`)**: `S / (1024 * 1024)` — With `dtype: bfloat16` (`2` bytes per element), `matrix_dim: 1024` produces **`2.00 MiB`** per device; `matrix_dim: 8192` produces **`16.00 MiB`** per device.

#### Parameters

| Parameter | Type | Default | Applies To | Description |
| :--- | :--- | :--- | :--- | :--- |
| `mesh_shape` | `Optional[str]` | `None` | All Collectives | Logical TPU mesh dimensions formatted as `'DxDyDz'` or `'DxDy'` (e.g., `"2x2x1"`, `"2x4x4"`, `"4x4"`). Defaults to a 1D mesh over all devices if `None`. |
| `sharding_strategy` | `Optional[str]` | `None` | All Collectives | Axis participation pattern matching the dimensionality of `mesh_shape` (e.g., `"2x2x1"`). Dimensions with value `> 1` participate in the collective; defaults to all mesh dimensions if `None`. |
| `matrix_dim` | `int` | `1024` | All Collectives | Leading dimension of the per-device tensor `(matrix_dim, 8, 128)`. |
| `dtype` | `str` | `"bfloat16"` | All Collectives | Tensor element data type (e.g., `bfloat16`, `float32`). |
| `reduce_op` | `str` | `"sum"` | `all_reduce` only | Reduction operator applied across active collective dimensions (`"sum"`, `"mean"`, `"max"`, `"min"`). |
| `seed` | `int` | `0` | All Collectives | PRNG seed for tensor initialization. |
| `xla_dump_dir` | `Optional[str]` | `None` | All Collectives | Optional directory containing XLA HLO dump files for inspecting compiler `replica_groups`. |

#### Collective Bus Bandwidth Formulas (`data_transferred_bytes`)

Given per-device shard payload $$S$$ (bytes), replica group rank $$R$$ (`replica_group_rank`, devices per ring), external inter-chip peers $$P$$ ($$R - 1$$ for single-device or `parallel` dual-device; $$R - 2$$ for `non-parallel` dual-device), and active rings per physical chip $$C$$ ($$2$$ for `parallel` dual-device; $$1$$ otherwise):

| Benchmark | JAX Primitive | Per-Chip Bytes Transferred (`data_transferred_bytes`) |
| :--- | :--- | :--- |
| **`all_reduce`** | `jax.lax.psum` / `pmean` / `pmax` / `pmin` | $$2 \times S \times \left(\frac{P}{R}\right) \times C$$ |
| **`all_gather`** | `jax.lax.all_gather` | $$S \times P \times C$$ |
| **`all_to_all`** | `jax.lax.all_to_all` | $$S \times \left(\frac{P}{R}\right) \times C$$ |

#### Reported Metrics

- **Collective Bus Bandwidth**:
  - `{wall_clock|xprof}_bandwidth_per_chip_gb_s` — Effective external ICI collective bus bandwidth per physical TPU chip (`GB/s`).
- **Latency (`ms`)**:
  - `{wall_clock|xprof}_{p50|p90|avg|std}_ms` — Iteration execution latency across median (`p50`), 90th percentile (`p90`), mean (`avg`), and standard deviation (`std`) in milliseconds (`ms`).
- **Benchmark Attributes**:
  - `shard_size_mib` — Per-device tensor shard payload (`S` expressed in `MiB`).
  - `replica_group_rank` — Number of participating devices (`R`) per collective replica group (product of dimensions `> 1` in `sharding_strategy`).
  - `replica_group_type` — Classified replica ring mode (`"parallel"` vs. `"non-parallel"`).
  - `data_transferred_bytes` — Effective external ICI bytes transferred per physical chip per collective step.

---

> [!NOTE]
> Additional experimental benchmarks (such as ReduceScatter, FlashAttention, MoE Transformer layers, and custom fusion ops) are under active development — use with caution.
