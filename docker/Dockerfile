# Base image with CUDA and cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables for Python and CUDA
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils python3-pip wget git unzip && \
    rm -rf /var/lib/apt/lists/*

# Set Python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10

# Install TensorFlow and other Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow==2.12.0

# Copy and install your application
COPY code.tar.gz /workspace/code.tar.gz
RUN mkdir -p /workspace/code && tar -xzf /workspace/code.tar.gz -C /workspace/code && \
    pip install /workspace/code && \
    mv /workspace/code/RML2016.10a_dict.pkl /workspace/code/src/ml_wireless_classification/


# Verify CUDA and cuDNN
RUN python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
