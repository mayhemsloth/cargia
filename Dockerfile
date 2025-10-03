# Use CUDA 12.2.2 with Ubuntu 22.04 as base image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python (pip is already available)
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements-docker.txt .

# Upgrade PyTorch to the required version
RUN pip install torch==2.6.0 torchvision torchaudio

# Install Python dependencies (excluding PyTorch which is already in base image)
RUN pip install -r requirements-docker.txt

# Install additional dependencies from pyproject.toml that aren't in requirements.txt
RUN pip install \
    transformers>=4.38.0 \
    accelerate>=0.27.0 \
    datasets>=2.17.0 \
    tensorboard>=2.15.0 \
    scikit-image>=0.22.0 \
    sentencepiece \
    huggingface-hub

# Copy the entire codebase
COPY . .

# Install the cargia package in development mode
RUN pip install -e .

# Set the working directory to where the main code lives
WORKDIR /app/cargia

# Default command - keep container alive with sleep infinity
CMD ["bash", "-lc", "sleep infinity"] 