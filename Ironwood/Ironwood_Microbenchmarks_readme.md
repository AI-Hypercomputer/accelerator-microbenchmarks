# Microbenchmarks
Microbenchmarks that assess the performance of individual operations and components on accelerators with JAX.

## Setup

Setup the cloud TPU environment. For more information about how to set up a TPU environment, refer to one of the following references:

* GCE: [Manage TPU resources | Google Cloud](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus)
* GKE: [Deploy TPU workloads in GKE Standard | Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus)

### Quick Start

The following command sets up a Ironwood TPU VM:

```bash
gcloud compute tpus tpu-vm create $TPU_NAME /
        --zone=${ZONE} /
        --accelerator-type=tpu7x-8  /
        --version=v2-alpha-tpu7-ubuntu2404 /
        --project=${PROJECT_ID}
```

You may ssh into the VM for subsequent testing:
```bash
gcloud compute ssh $TPU_NAME --zone=$ZONE
```

## Create the Venv
```bash
sudo apt install python3-venv
python3 -m venv microbenchmarks
source microbenchmarks/bin/activate
```


## Running the microbenchmarks on VM

Now that you have the VM environment set up, `git clone` the accelerator-microbenchmarks on the VM and install the dependencies:
```bash
git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git
cd accelerator-microbenchmarks/
pip install -r requirements.txt
```

You can run the benchmarks with a config file:

```bash
python Ironwood/src/run_benchmark.py --config=Ironwood/configs/gemm_demo.yaml
```