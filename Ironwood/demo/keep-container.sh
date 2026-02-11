#!/usr/bin/env bash

xpk workload create --cluster "${CLUSTER}" --zone "${ZONE}" --device-type "tpu7x-8" \
  --command 'git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git && cd accelerator-microbenchmarks && git checkout tpu7x-demo-0211 && pip install -r requirements.txt && export GCS_BUCKET_DIR=${GCS_PATH} && sleep infinity' \
  --num-slices "1" --docker-image=docker.io/library/python:3.12-bookworm --workload "test-container-env" --project "${PROJECT}"