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
python3 -m venv micro_venv
source micro_venv/bin/activate
```


## Running the microbenchmarks on VM

Now that you have the VM environment set up, `git clone` the accelerator-microbenchmarks on the VM and install the dependencies:
```bash
git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git
cd accelerator-microbenchmarks/
pip install -r requirements.txt
pip install libtpu==0.0.26.dev20251022+nightly -f'https://storage.googleapis.com/jax-releases/libtpu_releases.html'
```

You can run the benchmarks with a config file:

```bash
python Ironwood/src/run_benchmark.py --config=Ironwood/configs/training/compute_microbenchmark_demo.yaml
```

## Microbenchmark scripts

Run the training compute microbenchmark, including gemm, add, quantization, and transpose quantization:

```bash
sh ./Ironwood/scripts/run_training_compute_microbenchmark.sh
```

| Operation | Function Description | Formula / Logic |
| :--- | :--- | :--- |
| **`gemm_multiple_run`** | **Configurable GEMM.** Benchmarks matmul with configurable input types (supports `float8_e4m3fn`, `bfloat16`). Accumulation is FP32, output cast to BF16. Rerun multiple times and record all profilers. | $O_{bf16} = (A \times B)$ |
| **`gemm_simple`** | **Basic FP8 GEMM.** Benchmarks pure FP8 matmul with FP32 accumulation. | $O_{bf16} = (A_{fp8} \times B_{fp8})$ |
| **`gemm`** | **FP8 GEMM + Scaling.** Performs FP8 matmul and applies scaling factors (dequantization) to the result. | $O_{bf16} = (A_{fp8} \times B_{fp8}) \times (S_A \times S_B^T)$ |
| **`gemm_accum`** | **FP8 GEMM + Accumulation.** Performs FP8 matmul with scaling factors and accumulates the result into an existing FP32 output buffer. | $O_{fp32} \ += (A_{fp8} \times B_{fp8}) \times (S_A \times S_B^T)$ |
| **`gemm_fp8_rowwise`** | **FP8 Row-wise GEMM.** Quantizes inputs dynamically (row-wise/channel-wise) using absmax calibration before performing the matrix multiplication. | $O_{bf16} = (Quant(A) \times Quant(B))$ |
| **`add`** | **Element-wise Addition.** Adds two BF16 tensors. | $Z = X + Y$ |
| **`quantization`** | **Dynamic Quantization.** Quantizes a BF16 input tensor to FP8 using dynamic scaling (absmax calibration). Returns quantized values and scale factors. | $S = \frac{Max}{absmax(X)}$, $O = Cast(\frac{X}{S})$ |
| **`transpose_quantization`** | **Transpose + Quantization.** Transposes a BF16 input tensor first, then applies dynamic quantization. | $S = \frac{Max}{absmax(X^T)}$, $O = Cast(\frac{X^T}{S})$ |

## Examine the outputs

The benchmarks will print metrics to the terminal. If you wish to dump formatted metrics in a file, you may set this parameter in your YAML file:
* `csv_path`: Dumps the benchmark metrics in a CSV.
Examples can be found in the YAML files under config/ directory.

If you wish to generate the xprof profile, set this parameter in the YAML file:
* `trace_dir`: Dumps the xprof profile to either a local location or GCS bucket.
Examples can be found in the YAML files under config/ directory.