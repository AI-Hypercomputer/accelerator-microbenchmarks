# Use a base image with Python and Git
FROM python:3.10-slim

# Install Git
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

# Set environment variables for Google Cloud SDK and Python 3.10
ENV PATH="/usr/local/google-cloud-sdk/bin:/usr/local/bin/python3.10:${PATH}"

# Set the working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git

# Navigate to the repository directory
WORKDIR /app/accelerator-microbenchmarks

# Install dependencies
RUN pip install --upgrade pip && \
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --upgrade clu tensorflow tensorflow-datasets && \
    pip install jsonlines && \
    pip install ray[default]

# Set environment variables
ENV JAX_PLATFORMS=tpu,cpu \
    ENABLE_PJRT_COMPATIBILITY=true

