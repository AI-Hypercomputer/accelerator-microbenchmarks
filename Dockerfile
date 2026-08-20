# Use a base image with Python and Git
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1

ENV PATH="/usr/local/google-cloud-sdk/bin:/usr/local/bin/python3.12:${PATH}"

# Set the working directory
WORKDIR /app/accelerator-microbenchmarks
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --no-cache-dir tpu-info

# Verify that the benchmark script can be run
RUN python Ironwood/src/run_benchmark.py --help

# Set environment variables
ENV JAX_PLATFORMS=tpu,cpu \
    ENABLE_PJRT_COMPATIBILITY=true
