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
git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git
pip install -r requirements.txt
```

You can run the benchmarks with a config file:

```bash
cd accelerator-microbenchmarks
python src/run_benchmark.py --config=configs/sample_benchmark_matmul.yaml
```

Create your own config.yaml file to customize the benchmarks and parameters you want to run. You may refer to the src/benchmark_*.py for the benchmarks and tunable parameters, or you may refer to the sample YAML files in configs/ directory. 


## Examine the outputs

The benchmarks will print metrics to the terminal. If you wish to dump formatted metrics in a file, you may set this parameter in your YAML file:
* `csv_path`: Dumps the benchmark metrics in a CSV.
Examples can be found in the YAML files under config/ directory.

If you wish to generate the xprof profile, set this parameter in the YAML file:
* `trace_dir`: Dumps the xprof profile to either a local location or GCS bucket.
Examples can be found in the YAML files under config/ directory.
