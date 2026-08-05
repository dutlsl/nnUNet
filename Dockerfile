# Dockerfile for Vivim + SAGD (nnUNet v2 base)
# Tested for single GPU execution (e.g. RTX 4090 / 5090 / A100)

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install Mamba & Causal Conv1d dependencies
RUN pip install causal-conv1d==1.4.0
RUN pip install mamba-ssm==2.2.2 --no-build-isolation

# Install nnUNetv2 & project requirements
WORKDIR /workspace/nnUNet

COPY requirements.txt /workspace/nnUNet/requirements.txt
RUN pip install -r requirements.txt

# Copy source code
COPY . /workspace/nnUNet
RUN pip install -e .

# Default Environment Variables for nnUNet
ENV nnUNet_raw="/workspace/nnUNet/nnUNet_raw"
ENV nnUNet_preprocessed="/workspace/nnUNet/nnUNet_preprocessed"
ENV nnUNet_results="/workspace/nnUNet/nnUNet_results"

CMD ["/bin/bash"]
