# TPUMS Automated Benchmark Pipeline Runbook

## 1. Prerequisites

Before running the benchmark pipeline, ensure you have:

*   A GKE cluster with [Kueue](https://kueue.sigs.k8s.io/),
    [JobSet](https://jobset.sigs.k8s.io/), and
    [GKE TPU Node Auto-Provisioning (NAP)](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus#autoprovisioning)
    enabled.
*   Google Cloud SDK (`gcloud`) and `kubectl` installed and authenticated:

    ```bash
    gcloud container clusters get-credentials <CLUSTER> \
      --project=<PROJECT> \
      --region=<REGION>
    ```

*   A Google Cloud Storage (GCS) bucket for storing benchmark logs and generated
    reports.

*   Appropriate IAM permissions for creating JobSets and reading/writing to your
    GCS bucket.

### Placement Policy Prerequisites

For multi-node topologies (e.g. `tpu7x` multi-node topologies such as `2x2x2`,
`4x4x4` or `v6e` multi-node topologies such as `2x4`, `4x4`, `16x16`),
`utils.py` (`compute_topology_info`) populates the
`cloud.google.com/placement-policy-name` nodeSelector using the naming
convention `{accelerator}-{total_cores}-{topology}-placement-policy`.

Pre-create the matching workload placement policies in your cluster region using
`gcloud`:

```bash
gcloud compute resource-policies create workload-policy <POLICY_NAME> \
  --region=<REGION> \
  --project=<PROJECT> \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=2x2x2
```

For example:

*   For TPU `tpu7x` `2x2x2` (8 chips, 16 cores): `tpu7x-16-2x2x2-placement-policy`
*   For TPU `tpu7x` `4x4x4` (64 chips, 128 cores):
    `tpu7x-128-4x4x4-placement-policy`
*   For TPU `v6e` `4x4` (16 chips, 16 cores): `v6e-16-4x4-placement-policy`
*   For TPU `v6e` `16x16` (256 chips, 256 cores): `v6e-256-16x16-placement-policy`

Automation always requests 4 chips per host (fewer for slices smaller than 4
chips), so `v6e` `2x4` runs as 2 hosts on `ct6e-standard-4t`; the single-host
`ct6e-standard-8t` layout in [GKE.md](GKE.md) is not supported here.

Each JobSet clones the GitHub `main` branch (or `--git-branch`) and runs the
config from that checkout, so config and source changes only take effect in
nightly runs after they are exported to GitHub.

--------------------------------------------------------------------------------

## 2. Quickstart & Dry-Run Preview

To validate your configurations and preview generated Kubernetes `JobSet`
manifests without submitting workloads or consuming TPU quota, run
[`nightly_runner.py`](../automation/nightly_runner.py) with `--dry-run`:

```bash
python3 -m automation.nightly_runner --dry-run \
  --suite-file=automation/canonical_suite_v7x.yaml
```

This renders all targeted JobSet manifests to `stdout` and generates preview
Markdown and HTML summary reports.

Run these commands from the repository root. The automation scripts must run as
modules (`python3 -m automation.<module>`); `python3 automation/<module>.py`
fails with `ModuleNotFoundError`.

--------------------------------------------------------------------------------

## 3. Running the Benchmark Pipeline

[`nightly_runner.py`](../automation/nightly_runner.py) is the single pipeline
orchestrator for TPUMS automation. It handles CLI argument and environment
variable parsing (`CLUSTER`, `PROJECT`, `REGION`, `ZONE`, `NAMESPACE`, `QUEUE`,
`PRIORITY`, `DOCKER_IMAGE`, `GCS_BUCKET`, `RESERVATION_NAME`), GKE cluster
credential acquisition (`gcloud container clusters get-credentials`), JobSet
submission, execution monitoring, metrics aggregation, and report generation.

Every run selects its configs with exactly one of three modes (see
[Section 4](#4-workload-selection--filtering)). The nightly job runs the
accelerator's canonical suite (`canonical_suite_v7x.yaml` for tpu7x,
`canonical_suite_v6e.yaml` for v6e). The accelerator is inferred from each
config's `configs/<accelerator>/<topology>/` path, and drives the GKE node
accelerator label (`tpu7x` vs `tpu-v6e-slice`) and placement policies. A
selection whose configs span both accelerators, or whose topology or accelerator
cannot be inferred, is rejected rather than guessed.

```bash
python3 -m automation.nightly_runner \
  --suite-file="automation/canonical_suite_<v7x|v6e>.yaml" \
  --cluster="<your-gke-cluster>" \
  --project="<your-gcp-project>" \
  --region="<region>" \
  --zone="<zone>" \
  --namespace="default" \
  --queue="multislice-queue" \
  --docker-image="us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest" \
  --gcs-bucket="gs://<your-gcs-bucket>/tpums"
```

--------------------------------------------------------------------------------

## 4. Workload Selection & Filtering

You can select specific subsets of benchmarks instead of running the entire
suite by passing selection flags to
[`nightly_runner.py`](../automation/nightly_runner.py) (or
[`workload_submitter.py`](../automation/workload_submitter.py)). Exactly one
selection mode is required; mixing flags from different modes is an error.

| Mode | Flags | Selects |
| --- | --- | --- |
| 1. Suite | `--suite-file <yaml>` | The configs listed in the suite. |
| 2. Explicit configs | `--configs <yaml> [<yaml> ...]` | Exactly these files (directories are rejected; duplicates are dropped). |
| 3. Accelerator scan | `--accelerator <tpu7x\|v6e>` plus optional `--config-dir`, `--topologies`, `--categories` | Every topology config under `--config-dir` (default `configs/<accelerator>`), narrowed by the optional filters. |

*   `--config-dir`, `--topologies` and `--categories` are only valid with
    `--accelerator`. When both filters are given, a config must match both
    (AND).
*   `--topologies` must match the accelerator (3D such as `2x2x1` for tpu7x,
    2D such as `4x4` for v6e). `--categories` matches substrings of config file
    names (e.g. `all_gather`, `gemm`), not directory names.
*   All configs selected by any mode must target a single accelerator, and each
    must live in a `configs/<accelerator>/<topology>/` directory. In mode 3,
    configs without a topology directory (e.g.
    `configs/tpu7x/attention_jax_flash.yaml`) are skipped with a warning; modes
    1 and 2 reject them.
*   Relative `--configs`, `--suite-file` and `--config-dir` paths resolve
    against the repository root first, then the current directory.

> **Note on Report Generation:** No subsequent commands are required after
> running any filtered or custom suite command below. `nightly_runner.py`
> executes the complete end-to-end pipeline automatically: it submits the
> selected JobSets, polls until completion, aggregates metrics from GCS, and
> publishes the final Markdown and HTML reports to both local disk and
> `gs://<bucket>/<date>/consolidated_report.{md,html}`.

### Run Specific Benchmark Categories

Select benchmarks by file-name substring (e.g. `all_gather` and `gemm`) across
all tpu7x topologies:

```bash
python3 -m automation.nightly_runner \
  --accelerator=tpu7x \
  --categories all_gather gemm \
  --cluster="<cluster>" \
  --project="<project>" \
  --gcs-bucket="gs://<bucket>"
```

### Run Specific Topologies

Target specific TPU topologies (e.g. single-host `2x2x1` or multi-node `4x4x4`):

```bash
python3 -m automation.nightly_runner \
  --accelerator=tpu7x \
  --topologies 2x2x1 4x4x4 \
  --cluster="<cluster>" \
  --project="<project>" \
  --gcs-bucket="gs://<bucket>"
```

Combine `--topologies` with `--categories` to keep only configs matching both.
For example, this runs only `configs/v6e/2x2/all_gather.yaml` and
`configs/v6e/4x4/all_gather.yaml`:

```bash
python3 -m automation.nightly_runner \
  --accelerator=v6e \
  --topologies 2x2 4x4 \
  --categories all_gather \
  --cluster="<cluster>" \
  --project="<project>" \
  --gcs-bucket="gs://<bucket>"
```

### Scan a Specific Config Directory

`--config-dir` replaces the default `configs/<accelerator>` scan root. It can be
a topology subdirectory or any directory whose configs sit under a
`<topology>/` directory (e.g. `my_configs/v6e/4x4/gemm.yaml`); every config
found must match `--accelerator`. For example, run every v6e `4x4` config:

```bash
python3 -m automation.nightly_runner \
  --accelerator=v6e \
  --config-dir=configs/v6e/4x4 \
  --cluster="<cluster>" \
  --project="<project>" \
  --gcs-bucket="gs://<bucket>"
```

`--topologies` and `--categories` still apply within `--config-dir`. JobSets run
configs from the GitHub checkout of `--git-branch`, so a custom directory must
live inside the repository.

### Run Explicit Config Files

```bash
python3 -m automation.nightly_runner \
  --configs configs/v6e/4x4/all_gather.yaml configs/v6e/2x2/gemm.yaml \
  --cluster="<cluster>" \
  --project="<project>" \
  --gcs-bucket="gs://<bucket>"
```

### Run a Custom Suite File

Define a custom YAML file listing target configs using the `suite_name`,
`description`, and `configs:` schema:

```yaml
suite_name: custom_tpu7x_subset
description: Custom benchmark subset for targeted regression testing

configs:
  - configs/tpu7x/4x4x4/all_gather.yaml
  - configs/tpu7x/2x2x1/gemm.yaml
  - configs/tpu7x/2x2x1/device_to_device.yaml
```

Suite entries must be paths relative to the repository root, because each
JobSet runs its config from a GitHub checkout of `--git-branch`. A bare YAML list
without the `configs:` key is rejected, as is a suite whose configs span more
than one accelerator.

Then pass the suite file path (absolute, or relative to the repository root or
the current directory) via `--suite-file`:

```bash
python3 -m automation.nightly_runner \
  --suite-file="path/to/custom_suite.yaml" \
  --cluster="<cluster>" \
  --project="<project>" \
  --gcs-bucket="gs://<bucket>"
```

--------------------------------------------------------------------------------

## 5. Reports & Metrics Output

After execution, reports and metrics are published to both local disk and GCS:

*   **HTML Summary Dashboard:** `gs://<bucket>/<date>/consolidated_report.html`
    (and local preview `/tmp/tpums_daily_report_<date>.html`)
*   **Markdown Summary:** `gs://<bucket>/<date>/consolidated_report.md` (and
    local `/tmp/tpums_daily_report_<date>.md`)
*   **Aggregated Category CSVs:**
    `gs://<bucket>/<date>/aggregated_results/<category>.csv`
*   **Raw Workload Artifacts:** `gs://<bucket>/<date>/<workload_name>/`
    (`summary.csv`, `detailed.json`, `git_commit.txt`, and
    `tpums_results.tar.gz` come from worker 0; every worker uploads
    `execution_<index>.log`.)

### Standalone Report Regeneration

If you want to re-generate the consolidated Markdown and HTML reports from
existing GCS artifacts for a given date without re-running TPU workloads, run
[`report_generator.py`](../automation/report_generator.py) directly:

```bash
python3 -m automation.report_generator \
  --gcs-bucket="gs://<bucket>" \
  --date="YYYY-MM-DD" \
  --cluster="<cluster>" \
  --project="<project>"
```

--------------------------------------------------------------------------------

## 6. Benchmark Configurations

TPUMS automation provides dedicated canonical benchmark suites for both TPU
`tpu7x` and TPU `v6e`:

*   **TPU tpu7x:** Defined in
    [`canonical_suite_v7x.yaml`](../automation/canonical_suite_v7x.yaml). Hardware-tuned
    configurations are maintained under [`configs/tpu7x/`](../configs/tpu7x/).
*   **TPU v6e:** Defined in
    [`canonical_suite_v6e.yaml`](../automation/canonical_suite_v6e.yaml).
    Hardware-tuned configurations are maintained under
    [`configs/v6e/`](../configs/v6e/).
