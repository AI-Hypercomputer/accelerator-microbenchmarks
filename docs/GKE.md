
# Running TPUMS on Google Kubernetes Engine (GKE)

This guide walks through how to run **TPUMS (TPU Microbenchmark Suite)** benchmarks across single-host and multi-host TPU topologies (`tpu7x` and `v6e`) on **Google Kubernetes Engine (GKE)**, with optional result and XProf trace export to Google Cloud Storage (GCS).

> [!TIP]
> For running automated multi-benchmark suites across multiple topologies and generating consolidated Markdown/HTML reports, see **[`docs/AUTOMATION.md`](AUTOMATION.md)**.

---

## Table of Contents

- [1. TPU Topology & Node Pool Sizing (`tpu7x` & `v6e`)](#1-tpu-topology--node-pool-sizing-tpu7x--v6e)
  - [`tpu7x` (Ironwood) Topologies](#tpu7x-ironwood-topologies)
  - [`v6e` (Trillium) Topologies](#v6e-trillium-topologies)
- [2. Deploying a Benchmark on GKE](#2-deploying-a-benchmark-on-gke)
  - [Step 1: Prerequisites & Cluster Authentication](#step-1-prerequisites--cluster-authentication)
  - [Step 2: Configure `templates/gke_job_template.yaml`](#step-2-configure-templatesgke_job_templateyaml)
  - [Step 3: Launch a Benchmark Job](#step-3-launch-a-benchmark-job)
  - [Step 4: Monitor Logs, Inspect GCS Results & Clean Up](#step-4-monitor-logs-inspect-gcs-results--clean-up)

---

<a id="1-tpu-topology--node-pool-sizing-tpu7x--v6e"></a>
## 1. TPU Topology & Node Pool Sizing (`tpu7x` & `v6e`)

The same [`templates/gke_job_template.yaml`](../templates/gke_job_template.yaml) manifest supports both single-host and multi-host TPU runs on GKE. Match the following four hardware sizing values to your target GKE TPU node pool:

- **`ACCELERATOR_TYPE`**: GKE accelerator node label (`cloud.google.com/gke-tpu-accelerator` — `tpu7x` for Ironwood or `tpu-v6e-slice` for Trillium).
- **`TOPOLOGY`**: Physical TPU topology (`cloud.google.com/gke-tpu-topology` — e.g., `2x2x1`, `4x4x4`, `2x2`, `2x4`).
- **`NUM_HOSTS`**: Number of TPU VM hosts (pods) in the topology.
- **`CHIPS_PER_NODE`**: Number of TPU chips requested per host (`google.com/tpu` resource request/limit).

<a id="tpu7x-ironwood-topologies"></a>
### `tpu7x` (Ironwood) Topologies

Each `tpu7x` chip contains **2 logical TensorCores (JAX devices)** (4 physical chips / 8 logical JAX devices per host). For hardware details, see the official [Cloud TPU `tpu7x` Configurations](https://docs.cloud.google.com/tpu/docs/tpu7x#configurations).

| `ACCELERATOR_TYPE` | `TOPOLOGY` | Mode | Physical Chips | Logical JAX Devices | Hosts (`NUM_HOSTS`) | `CHIPS_PER_NODE` | Pre-Configured Config Directory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tpu7x` | **`2x2x1`** | Single-Host | `4` | `8` | **`1`** | `4` | [`configs/tpu7x/2x2x1/`](../configs/tpu7x/2x2x1/) |
| `tpu7x` | **`2x2x2`** | Multi-Host | `8` | `16` | **`2`** | `4` | [`configs/tpu7x/2x2x2/`](../configs/tpu7x/2x2x2/) |
| `tpu7x` | **`2x2x4`** | Multi-Host | `16` | `32` | **`4`** | `4` | [`configs/tpu7x/2x2x4/`](../configs/tpu7x/2x2x4/) |
| `tpu7x` | **`2x4x4`** | Multi-Host | `32` | `64` | **`8`** | `4` | [`configs/tpu7x/2x4x4/`](../configs/tpu7x/2x4x4/) |
| `tpu7x` | **`4x4x4`** | Multi-Host | `64` | `128` | **`16`** | `4` | [`configs/tpu7x/4x4x4/`](../configs/tpu7x/4x4x4/) |

<a id="v6e-trillium-topologies"></a>
### `v6e` (Trillium) Topologies

Each `v6e` chip contains **1 logical TensorCore (JAX device)** (4 physical chips / 4 logical JAX devices per `ct6e-standard-4t` host, or 8 physical chips / 8 logical JAX devices per `ct6e-standard-8t` host). For hardware details, see the official [Cloud TPU `v6e` Configurations](https://docs.cloud.google.com/tpu/docs/v6e#configurations).

> [!IMPORTANT]
> **`v6e` `2x4` (8-Chip) Machine-Type Sizing**
>
> Check your node pool's allocatable `CHIPS` (`google.com/tpu`) count (`4` vs. `8`) before launching a `2x4` job:
>
> - **Multi-Host (`ct6e-standard-4t`)**: 2 hosts × 4 chips/host (`NUM_HOSTS=2`, `CHIPS_PER_NODE=4`).
> - **Single-Host (`ct6e-standard-8t`)**: 1 host × 8 chips/host (`NUM_HOSTS=1`, `CHIPS_PER_NODE=8`).

| `ACCELERATOR_TYPE` | `TOPOLOGY` | Mode (Machine Type) | Physical Chips | Logical JAX Devices | Hosts (`NUM_HOSTS`) | `CHIPS_PER_NODE` | Pre-Configured Config Directory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tpu-v6e-slice` | **`2x2`** | Single-Host (`ct6e-standard-4t`) | `4` | `4` | **`1`** | `4` | [`configs/v6e/2x2/`](../configs/v6e/2x2/) |
| `tpu-v6e-slice` | **`2x4`** | Multi-Host (`ct6e-standard-4t`) | `8` | `8` | **`2`** | `4` | [`configs/v6e/2x4/`](../configs/v6e/2x4/) |
| `tpu-v6e-slice` | **`2x4`** | Single-Host (`ct6e-standard-8t`) | `8` | `8` | **`1`** | `8` | [`configs/v6e/2x4/`](../configs/v6e/2x4/) |
| `tpu-v6e-slice` | **`4x4`** | Multi-Host (`ct6e-standard-4t`) | `16` | `16` | **`4`** | `4` | [`configs/v6e/4x4/`](../configs/v6e/4x4/) |
| `tpu-v6e-slice` | **`4x8`** | Multi-Host (`ct6e-standard-4t`) | `32` | `32` | **`8`** | `4` | [`configs/v6e/4x8/`](../configs/v6e/4x8/) |
| `tpu-v6e-slice` | **`8x8`** | Multi-Host (`ct6e-standard-4t`) | `64` | `64` | **`16`** | `4` | [`configs/v6e/8x8/`](../configs/v6e/8x8/) |
| `tpu-v6e-slice` | **`8x16`** | Multi-Host (`ct6e-standard-4t`) | `128` | `128` | **`32`** | `4` | [`configs/v6e/8x16/`](../configs/v6e/8x16/) |
| `tpu-v6e-slice` | **`16x16`** | Multi-Host (`ct6e-standard-4t`) | `256` | `256` | **`64`** | `4` | [`configs/v6e/16x16/`](../configs/v6e/16x16/) |

---

<a id="2-deploying-a-benchmark-on-gke"></a>
## 2. Deploying a Benchmark on GKE

<a id="step-1-prerequisites--cluster-authentication"></a>
### Step 1: Prerequisites & Cluster Authentication

Ensure you have the following before launching a job:

1. **GKE Cluster & TPU Node Pool**: Access to a GKE cluster with a `tpu7x` or `v6e` node pool.
2. **Local CLI Utilities**: `gcloud`, `kubectl`, and `envsubst` (`gettext`) installed locally.

Authenticate `kubectl` against your target GKE cluster and inspect your TPU node pool labels:

```bash
gcloud container clusters get-credentials <CLUSTER_NAME> \
    --project=<PROJECT_ID> \
    --region=<REGION_OR_ZONE>

# Inspect TPU node pool accelerator label, topology, and chips per node:
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator \
    -o 'custom-columns=NAME:.metadata.name,NODEPOOL:.metadata.labels.cloud\.google\.com/gke-nodepool,ACCEL:.metadata.labels.cloud\.google\.com/gke-tpu-accelerator,TOPOLOGY:.metadata.labels.cloud\.google\.com/gke-tpu-topology,CHIPS:.status.allocatable.google\.com/tpu'
```

<a id="step-2-configure-templatesgke_job_templateyaml"></a>
### Step 2: Configure `templates/gke_job_template.yaml`

The repository provides [`templates/gke_job_template.yaml`](../templates/gke_job_template.yaml) to run single-host and multi-host TPUMS benchmarks on GKE. On each scheduled TPU host, the job automatically:

- Coordinates multi-host network discovery across all TPU workers in the topology.
- Clones this repository and installs required package dependencies.
- Executes `tpums benchmark run-config` for your selected YAML configuration.
- Optionally uploads benchmark reports and XProf hardware traces to Google Cloud Storage (GCS).

Set the following environment variables before rendering [`templates/gke_job_template.yaml`](../templates/gke_job_template.yaml) with `envsubst`:

| Environment Variable | Required? | Purpose & What to Assign |
| :--- | :--- | :--- |
| **`WORKLOAD_NAME`** | **Required** | Unique name for the Kubernetes Job and Service (lowercase RFC 1123 format, e.g., `tpums-tpu7x-gemm-2x2x1`). |
| **`ACCELERATOR_TYPE`** | **Required** | GKE node pool accelerator label (`cloud.google.com/gke-tpu-accelerator`) — set to **`tpu7x`** for Ironwood or **`tpu-v6e-slice`** for Trillium (`v6e`). |
| **`TOPOLOGY`** | **Required** | Physical TPU topology label (`cloud.google.com/gke-tpu-topology`, e.g., `2x2x1`, `4x4x4`, `2x2`, `2x4`) from [Section 1](#1-tpu-topology--node-pool-sizing-tpu7x--v6e). |
| **`NUM_HOSTS`** | **Required** | Number of TPU VM hosts (pods) in the topology (`1` for single-host; `2`, `4`, `8`, `16`, etc. for multi-host topologies). |
| **`CHIPS_PER_NODE`** | **Required** | Number of TPU chips requested per host (`google.com/tpu` — `4` on standard nodes, or `8` on `ct6e-standard-8t` single-host VMs). |
| **`CONFIG_PATH`** | **Required** | Relative repository path to the benchmark YAML config to execute (e.g., `configs/tpu7x/2x2x1/gemm.yaml`). |
| **`GCS_OUTPUT_DIR`** | *Optional* | GCS destination URI (`gs://<bucket>/...`) for `summary.csv`, `detailed.json`, and `execution.log`. **Set to `""` to skip uploading** (results are still printed to `kubectl logs`). |
| **`GCS_XPROF_DIR`** | *Optional* | GCS destination URI (`gs://<bucket>/...`) for `.xplane.pb` / `.trace.json.gz` XProf trace files. **Set to `""` to skip uploading** (on-device `xprof_*` metrics are still computed locally). |

> [!NOTE]
> **GCS Bucket Permissions (When `GCS_OUTPUT_DIR` or `GCS_XPROF_DIR` is Set)**
>
> Ensure your GKE node pool service account (or GKE Workload Identity principal `<project>.svc.id.goog`) has `roles/storage.objectUser` (`storage.objects.create`) permission on the target GCS bucket. Uploads are non-fatal, so missing bucket permissions will not fail the benchmark `Job`.

<a id="step-3-launch-a-benchmark-job"></a>
### Step 3: Launch a Benchmark Job

Run the following commands from the repository root to render [`templates/gke_job_template.yaml`](../templates/gke_job_template.yaml) and submit the workload to your cluster:

#### Example A: Single-Host Compute (`gemm` on `tpu7x` `2x2x1` — 1 Host / 4 Chips / 8 Devices)

```bash
export WORKLOAD_NAME="tpums-tpu7x-gemm-2x2x1"
export ACCELERATOR_TYPE="tpu7x"
export TOPOLOGY="2x2x1"
export NUM_HOSTS=1
export CHIPS_PER_NODE=4
export CONFIG_PATH="configs/tpu7x/2x2x1/gemm.yaml"
export GCS_OUTPUT_DIR="gs://<your-gcs-bucket>/tpums/${WORKLOAD_NAME}/results"  # Set to "" to skip result upload
export GCS_XPROF_DIR="gs://<your-gcs-bucket>/tpums/${WORKLOAD_NAME}/xprof"     # Set to "" to skip XProf trace upload

envsubst '${WORKLOAD_NAME} ${ACCELERATOR_TYPE} ${TOPOLOGY} ${NUM_HOSTS} ${CHIPS_PER_NODE} ${CONFIG_PATH} ${GCS_OUTPUT_DIR} ${GCS_XPROF_DIR}' \
  < templates/gke_job_template.yaml | kubectl apply -f -
```

*Verified console output (`kubectl logs job/tpums-tpu7x-gemm-2x2x1`):*

```text
=================================================================================================================================================================================================
Benchmark Results (GeneralizedGemmBenchmark)
=================================================================================================================================================================================================
     in_dtype out_dtype     m     k     n transpose_a transpose_b alpha beta use_scaling_factors      total_flops wall_clock_p50_ms wall_clock_tflops_per_chip xprof_p50_ms xprof_tflops_per_chip
     bfloat16  bfloat16  8192  8192  8192       False       False   1.0  0.0               False 1099511627776.00            1.4393                    1527.80       1.0028               2192.93
     bfloat16  bfloat16 16384 16384 16384       False       False   1.0  0.0               False 8796093022208.00            9.3947                    1872.56       8.8521               1987.34
float8_e4m3fn  bfloat16  8192  8192  8192       False       False   1.0  0.0               False 1099511627776.00            0.9282                    2369.25       0.5138               4280.24
float8_e4m3fn  bfloat16 16384 16384 16384       False       False   1.0  0.0               False 8796093022208.00            4.4757                    3930.59       4.0372               4357.50
=================================================================================================================================================================================================
```

#### Example B: Multi-Host Collective (`all_gather` on `tpu7x` `4x4x4` — 16 Hosts / 64 Chips / 128 Devices)

Only `WORKLOAD_NAME`, `TOPOLOGY`, `NUM_HOSTS`, and `CONFIG_PATH` change:

```bash
export WORKLOAD_NAME="tpums-tpu7x-allgather-4x4x4"
export ACCELERATOR_TYPE="tpu7x"
export TOPOLOGY="4x4x4"
export NUM_HOSTS=16
export CHIPS_PER_NODE=4
export CONFIG_PATH="configs/tpu7x/4x4x4/all_gather.yaml"
export GCS_OUTPUT_DIR="gs://<your-gcs-bucket>/tpums/${WORKLOAD_NAME}/results"  # Set to "" to skip result upload
export GCS_XPROF_DIR="gs://<your-gcs-bucket>/tpums/${WORKLOAD_NAME}/xprof"     # Set to "" to skip XProf trace upload

envsubst '${WORKLOAD_NAME} ${ACCELERATOR_TYPE} ${TOPOLOGY} ${NUM_HOSTS} ${CHIPS_PER_NODE} ${CONFIG_PATH} ${GCS_OUTPUT_DIR} ${GCS_XPROF_DIR}' \
  < templates/gke_job_template.yaml | kubectl apply -f -
```

*Verified console output (`kubectl logs job/tpums-tpu7x-allgather-4x4x4`):*

```text
===============================================================================================================================================================
Benchmark Results (AllGatherBenchmark)
===============================================================================================================================================================
   dtype mesh_shape sharding_strategy matrix_dim shard_size_mib wall_clock_p50_ms wall_clock_bandwidth_per_chip_gb_s xprof_p50_ms xprof_bandwidth_per_chip_gb_s
bfloat16     16x4x2            16x4x1       8192          16.00            5.8752                             359.80       3.8297                        551.98
bfloat16     16x4x2            16x4x1      16384          32.00           12.0646                             350.43       7.6263                        554.38
bfloat16     16x4x2            16x4x1      32768          64.00           21.2463                             397.99      15.2165                        555.69
bfloat16     16x4x2            16x4x2       8192          16.00            7.9645                             265.42       4.2250                        500.34
bfloat16     16x4x2            16x4x2      16384          32.00           13.7713                             307.01       8.4095                        502.75
bfloat16     16x4x2            16x4x2      32768          64.00           24.8586                             340.15      17.2008                        491.59
===============================================================================================================================================================
```

#### Example C: Single-Host Compute (`gemm` on `v6e` `2x2` `ct6e-standard-4t` — 1 Host / 4 Chips / 4 Devices)

For `v6e` on GKE, set `ACCELERATOR_TYPE="tpu-v6e-slice"`:

```bash
export WORKLOAD_NAME="tpums-v6e-2x2-gemm"
export ACCELERATOR_TYPE="tpu-v6e-slice"
export TOPOLOGY="2x2"
export NUM_HOSTS=1
export CHIPS_PER_NODE=4
export CONFIG_PATH="configs/v6e/2x2/gemm.yaml"
export GCS_OUTPUT_DIR="gs://<your-gcs-bucket>/tpums/${WORKLOAD_NAME}/results"  # Set to "" to skip result upload
export GCS_XPROF_DIR="gs://<your-gcs-bucket>/tpums/${WORKLOAD_NAME}/xprof"     # Set to "" to skip XProf trace upload

envsubst '${WORKLOAD_NAME} ${ACCELERATOR_TYPE} ${TOPOLOGY} ${NUM_HOSTS} ${CHIPS_PER_NODE} ${CONFIG_PATH} ${GCS_OUTPUT_DIR} ${GCS_XPROF_DIR}' \
  < templates/gke_job_template.yaml | kubectl apply -f -
```

*Verified console output (`kubectl logs job/tpums-v6e-2x2-gemm`):*

```text
=========================================================================================================================================================================================
Benchmark Results (GeneralizedGemmBenchmark)
=========================================================================================================================================================================================
in_dtype out_dtype    m    k    n transpose_a transpose_b alpha beta use_scaling_factors      total_flops wall_clock_p50_ms wall_clock_tflops_per_chip xprof_p50_ms xprof_tflops_per_chip
bfloat16  bfloat16 4096 4096 4096       False       False   1.0  0.0               False  137438953472.00            0.5076                     270.78       0.1628                844.26
bfloat16  bfloat16 8192 8192 8192       False       False   1.0  0.0               False 1099511627776.00            1.7234                     638.00       1.3196                833.23
=========================================================================================================================================================================================
```

#### Example D: Multi-Host Collective (`all_gather` on `v6e` `2x4` `ct6e-standard-4t` — 2 Hosts / 8 Chips / 8 Devices)

For a multi-host `v6e` `2x4` topology (`2` hosts of `ct6e-standard-4t`), set `NUM_HOSTS=2` and `CHIPS_PER_NODE=4`:

```bash
export WORKLOAD_NAME="tpums-v6e-allgather-2x4"
export ACCELERATOR_TYPE="tpu-v6e-slice"
export TOPOLOGY="2x4"
export NUM_HOSTS=2
export CHIPS_PER_NODE=4
export CONFIG_PATH="configs/v6e/2x4/all_gather.yaml"
export GCS_OUTPUT_DIR="gs://<your-gcs-bucket>/tpums/${WORKLOAD_NAME}/results"  # Set to "" to skip result upload
export GCS_XPROF_DIR="gs://<your-gcs-bucket>/tpums/${WORKLOAD_NAME}/xprof"     # Set to "" to skip XProf trace upload

envsubst '${WORKLOAD_NAME} ${ACCELERATOR_TYPE} ${TOPOLOGY} ${NUM_HOSTS} ${CHIPS_PER_NODE} ${CONFIG_PATH} ${GCS_OUTPUT_DIR} ${GCS_XPROF_DIR}' \
  < templates/gke_job_template.yaml | kubectl apply -f -
```

*Verified console output (`kubectl logs job/tpums-v6e-allgather-2x4`):*

```text
==============================================================================================================================================================
Benchmark Results (AllGatherBenchmark)
==============================================================================================================================================================
   dtype mesh_shape sharding_strategy matrix_dim shard_size_mib wall_clock_p50_ms wall_clock_bandwidth_per_chip_gb_s xprof_p50_ms xprof_bandwidth_per_chip_gb_s
bfloat16        2x4               2x4       8192          16.00            1.9204                              61.16       1.3613                         86.27
bfloat16        2x4               2x4      16384          32.00            3.4875                              67.35       2.7226                         86.27
bfloat16        2x4               2x4      32768          64.00            6.5876                              71.31       5.4447                         86.28
==============================================================================================================================================================
```

<a id="step-4-monitor-logs-inspect-gcs-results--clean-up"></a>
### Step 4: Monitor Logs, Inspect GCS Results & Clean Up

Monitor the running `Job`, inspect exported GCS artifacts, and delete the Kubernetes resources when finished:

```bash
# 1. Check pod scheduling status and stream logs from Process 0
kubectl get pods -l job-name="${WORKLOAD_NAME}" -o wide
kubectl logs -f job/"${WORKLOAD_NAME}"

# 2. List per-process exported CSV, JSON, and XProf artifacts in GCS (if enabled)
gcloud storage ls "${GCS_OUTPUT_DIR}/**"
gcloud storage ls "${GCS_XPROF_DIR}/**"

# 3. Clean up the Job and Service when finished
kubectl delete job/"${WORKLOAD_NAME}" svc/"${WORKLOAD_NAME}-svc"
```

#### Exported GCS Directory Layout

When `GCS_OUTPUT_DIR` and/or `GCS_XPROF_DIR` are configured, each host (pod) in the topology exports its outputs to a dedicated `process_<index>/` subdirectory (`process_0/` for single-host runs; `process_0/`, `process_1/`, ..., `process_<NUM_HOSTS-1>/` for multi-host runs):

```text
gs://<your-gcs-bucket>/tpums/<WORKLOAD_NAME>/
├── results/                          # Uploaded when GCS_OUTPUT_DIR is set
│   ├── process_0/                    # Host 0 (always created)
│   │   ├── summary.csv
│   │   ├── detailed.json
│   │   └── execution.log
│   └── process_1/                    # Host 1..N-1 (created on multi-host runs)
│       ├── summary.csv
│       ├── detailed.json
│       └── execution.log
└── xprof/                            # Uploaded when GCS_XPROF_DIR is set
    ├── process_0/
    │   └── <benchmark_case>/plugins/profile/<run>/<pod>.xplane.pb
    └── process_1/
        └── <benchmark_case>/plugins/profile/<run>/<pod>.xplane.pb
```

- **`summary.csv`**: Tabular summary of all executed configurations, platform metadata (`process_index`), and `xprof_*` / `wall_clock_*` metrics.
- **`detailed.json`**: Full per-iteration timings, hardware metadata, and configuration parameters.
- **`execution.log`**: Complete stdout/stderr console log from that host.
- **`<pod>.xplane.pb` & `<pod>.trace.json.gz`**: Exported XProf hardware trace files for TensorBoard / XProf inspection.
