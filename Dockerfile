# Use a base image with Python and Git
FROM python:3.12-slim

# Install Git
RUN apt-get update && apt-get install -y git curl gnupg apt-transport-https ca-certificates

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-bookworm main" | tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk gcsfuse



# Set the working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git
WORKDIR /app/accelerator-microbenchmarks

RUN git checkout local_wip
# Navigate to the repository directory


# Self-install UV.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create venv
# Don't create venv in tmp directory as it is mounted as noexec.
RUN PY_VERSION="$(python3 -c "import platform; print(platform.python_version())")" && \
    echo "Python Version: ${PY_VERSION}" && \
    uv venv --python="${PY_VERSION}" --seed ./.uv_venv

ENV VIRTUAL_ENV="/app/accelerator-microbenchmarks/.uv_venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
RUN uv pip install --upgrade pip
RUN uv pip install -r requirements.txt
# RUN uv pip install -U libtpu==0.0.36.dev20260205+nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# # RUN uv pip install -U setuptools
# RUN uv pip install -U libtpu==0.0.29

# Set environment variables
ENV JAX_PLATFORMS=tpu,cpu \
    ENABLE_PJRT_COMPATIBILITY=true \
    TF_CPP_MIN_LOG_LEVEL='1'

