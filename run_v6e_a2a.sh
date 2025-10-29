#!/bin/bash

export PROJECT_ID=tpu-prod-env-one-vm
export ZONE=us-east5-a
export TPU_NAME=chishuen-v6e-16
export BRANCH_NAME="jax-0p80-v6e-a2a"
export CONIFG="configs/v6e_4x4_a2a.yaml"

cmd="sudo apt update && \
sudo apt install -y python3.12 python3.12-venv && \
python3.12 -m venv venv && \
source venv/bin/activate && \
rm -rf accelerator-microbenchmarks && \
git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git && \
mkdir -p /tmp/microbenchmarks/outputs && \
rm -rf /tmp/microbenchmarks/outputs/* && \
cd accelerator-microbenchmarks && \
git checkout ${BRANCH_NAME} && \
pip install -r requirements.txt && \
pip show jax && \
export LIBTPU_INIT_ARGS='--xla_jf_all_to_all_shard_kib=16' && \
python3.12 src/run_benchmark.py --config=${CONIFG}"

gcloud compute tpus tpu-vm ssh --zone $ZONE $TPU_NAME --project $PROJECT_ID --worker=all --command="$cmd"

gcloud compute tpus tpu-vm scp --zone $ZONE $TPU_NAME:/tmp/microbenchmarks/outputs . --project $PROJECT_ID --worker=0 --recurse
