xpk workload create --workload=${RUN_NAME} \
--cluster=${TPU_CLUSTER_NAME} \
--num-slices=1 \
--tpu-type=${TPU_TYPE} \
--docker-image="${GCR_IMAGE_PATH}" \
--project=${PROJECT_ID} \
--zone=${TPU_ZONE} \
--priority=very-high \
--command="export HOME=/tmp && cd /tmp && \
git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git && \
cd accelerator-microbenchmarks && \
git checkout accelerator_microbenchmarks_test && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
export PATH=\"/tmp/.local/bin:\$PATH\" && \
uv venv .venv && \
source .venv/bin/activate && \
uv pip install -r requirements.txt && \
which python; which python3;pip install -U setuptools;env; \
export JAX_PLATFORMS=tpu,cpu && \
export ENABLE_PJRT_COMPATIBILITY=true && \
export TF_CPP_MIN_LOG_LEVEL='1' && \
echo ${RUN_SCRIPT} && \
${RUN_SCRIPT} | tee xpk_output.txt && \
if [ -f results/summary.csv ] || [ -d results ] || [ -d ../microbenchmarks ]; then tar czf /tmp/microbenchmarks.tar.gz results ../microbenchmarks xpk_output.txt 2>/dev/null || tar czf /tmp/microbenchmarks.tar.gz results xpk_output.txt && gsutil -o 'GSUtil:parallel_composite_upload_threshold=150M' cp /tmp/microbenchmarks.tar.gz ${BASE_OUTPUT_DIR}; fi"
echo ${BASE_OUTPUT_DIR}

# uv pip show libtpu && \
# uv pip install -U libtpu==0.0.36.dev20260205+nightly -f && \


# #Version for tar exported from within codebase
# xpk workload create --workload=${RUN_NAME} \
# --cluster=${TPU_CLUSTER_NAME} \
# --num-slices=1 \
# --tpu-type=${TPU_TYPE} \
# --docker-image="${GCR_IMAGE_PATH}" \
# --project=${PROJECT_ID} \
# --zone=${TPU_ZONE} \
# --priority=very-high \
# --command="export HOME=/tmp && cd /tmp && \
# git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git && \
# cd accelerator-microbenchmarks && \
# git checkout vvashishth-first-3drs-run && \
# curl -LsSf https://astral.sh/uv/install.sh | sh && \
# export PATH=\"/tmp/.local/bin:\$PATH\" && \
# uv venv .venv && \
# source .venv/bin/activate && \
# uv pip install -r requirements.txt && \
# which python; which python3; pip install -U setuptools; env; \
# export JAX_PLATFORMS=tpu,cpu && \
# export ENABLE_PJRT_COMPATIBILITY=true && \
# export TF_CPP_MIN_LOG_LEVEL='1' && \
# ${RUN_SCRIPT} | tee xpk_output.txt"
# echo ${BASE_OUTPUT_DIR}