# Microbenchmarks
Microbenchmarks that assess the performance of individual operations and components on accelerators with JAX.

## Setup

Setup the cloud TPU environment. For more information about how to set up a TPU environment, refer to one of the following references:

* GCE: [Manage TPU resources | Google Cloud](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus)
* GKE: [Deploy TPU workloads in GKE Standard | Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus)

### Quick Start

The following command sets up a V4 TPU VM:

```bash
gcloud compute tpus tpu-vm create $TPU_NAME /
        --zone=$ZONE /
        --accelerator-type=v4-8  /
        --version=tpu-ubuntu2204-base
```

You may ssh into the VM for subsequent testing:
```bash
gcloud compute ssh $TPU_NAME --zone=$ZONE
```

## Running the microbenchmarks on VM

Now that you have the VM environment set up, `git clone` the accelerator-microbenchmarks on the VM and install the dependencies:
```bash
git clone https://github.com/qinyiyan/accelerator-microbenchmarks.git
pip install -r requirements.txt
```

You can run the benchmarks with a config file:

```bash
cd accelerator-microbenchmarks
python src/run_benchmark.py --config=configs/sample_benchmark_matmul.yaml
```