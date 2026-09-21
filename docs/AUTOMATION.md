# TPUMS Automated Benchmark Pipeline Runbook

## 1. Prerequisites

Before running the benchmark pipeline, ensure you have:

* A GKE cluster with [Kueue](https://kueue.sigs.k8s.io/),
  [JobSet](https://jobset.sigs.k8s.io/), and
  [GKE TPU Node Auto-Provisioning (NAP)](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus#autoprovisioning)
  enabled.
* Google Cloud SDK (`gcloud`) and `kubectl` installed and authenticated:

  ```bash
  gcloud container clusters get-credentials <CLUSTER> \
    --project=<PROJECT> \
    --region=<REGION>
  ```

* A Google Cloud Storage (GCS) bucket for storing benchmark logs and generated
  reports.
* Appropriate IAM permissions for creating JobSets and reading/writing to your
  GCS bucket.

---

## 2. Quickstart & Dry-Run Preview

To validate your configurations and preview generated Kubernetes `JobSet`
manifests without submitting workloads or consuming TPU quota, run
[`nightly_runner.py`](../automation/nightly_runner.py) with `--dry-run`:

```bash
python3 automation/nightly_runner.py --dry-run
```

This renders all targeted JobSet manifests to `stdout` and generates preview
Markdown and HTML summary reports.

---

## 3. Running the Benchmark Pipeline

[`nightly_runner.py`](../automation/nightly_runner.py) is the single pipeline
orchestrator for TPUMS automation. It handles CLI argument and environment
variable parsing (`CLUSTER`, `PROJECT`, `REGION`, `ZONE`, `NAMESPACE`, `QUEUE`,
`PRIORITY`, `DOCKER_IMAGE`, `GCS_BUCKET`, `RESERVATION_NAME`), GKE cluster
credential acquisition (`gcloud container clusters get-credentials`), JobSet
submission, execution monitoring, metrics aggregation, and report generation.

To launch the full canonical benchmark suite:

```bash
python3 automation/nightly_runner.py \
  --cluster="<your-gke-cluster>" \
  --project="<your-gcp-project>" \
  --region="us-central1" \
  --zone="us-central1-c" \
  --namespace="default" \
  --queue="multislice-queue" \
  --docker-image="us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:latest" \
  --gcs-bucket="gs://<your-gcs-bucket>/tpums"
```

---

## 4. Workload Selection & Filtering

You can select specific subsets of benchmarks instead of running the entire
suite by passing selection flags to
[`nightly_runner.py`](../automation/nightly_runner.py).

> **Note on Report Generation:** No subsequent commands are required after
> running any filtered or custom suite command below. `nightly_runner.py`
> executes the complete end-to-end pipeline automatically: it submits the
> selected JobSets, polls until completion, aggregates metrics from GCS, and
> publishes the final Markdown and HTML reports to both local disk and
> `gs://<bucket>/<date>/consolidated_report.{md,html}`.

### Run Specific Benchmark Names

Filter by benchmark substring (e.g. `all_gather` and `gemm`):

```bash
python3 automation/nightly_runner.py \
  --filter all_gather gemm \
  --cluster="<cluster>" \
  --project="<project>" \
  --gcs-bucket="gs://<bucket>"
```

### Run Specific Topologies

Target specific TPU topologies (e.g. single-host `2x2x1` or multi-node `4x4x4`):

```bash
python3 automation/nightly_runner.py \
  --topologies 2x2x1 4x4x4 \
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

Then pass the suite file path via `--suite-file`:

```bash
python3 automation/nightly_runner.py \
  --suite-file="automation/canonical_suite.yaml" \
  --cluster="<cluster>" \
  --project="<project>" \
  --gcs-bucket="gs://<bucket>"
```

---

## 5. Reports & Metrics Output

After execution, reports and metrics are published to both local disk and GCS:

* **HTML Summary Dashboard:** `gs://<bucket>/<date>/consolidated_report.html`
  (and local preview `/tmp/tpums_daily_report_<date>.html`)
* **Markdown Summary:** `gs://<bucket>/<date>/consolidated_report.md` (and local
  `/tmp/tpums_daily_report_<date>.md`)
* **Aggregated Category CSVs:**
  `gs://<bucket>/<date>/aggregated_results/<category>.csv`
* **Raw Workload Artifacts:** `gs://<bucket>/<date>/<workload_name>/`

### Standalone Report Regeneration

If you want to re-generate the consolidated Markdown and HTML reports from
existing GCS artifacts for a given date without re-running TPU workloads, run
[`report_generator.py`](../automation/report_generator.py) directly:

```bash
python3 automation/report_generator.py \
  --gcs-bucket="gs://<bucket>" \
  --date="YYYY-MM-DD" \
  --cluster="<cluster>" \
  --project="<project>"
```

---

## 6. Benchmark Configurations

By default, the automated pipeline runs the canonical benchmark suite defined in
[`canonical_suite.yaml`](../automation/canonical_suite.yaml). All
hardware-tuned benchmark configurations referenced by the suite are maintained
under [`configs/tpu7x/`](../configs/tpu7x).
