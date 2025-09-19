FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    git \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.6 as default
RUN ln -s /usr/bin/python3.6 /usr/bin/python

# Upgrade pip and setuptools
RUN python -m pip install --upgrade pip setuptools

# Copy requirements.txt (You can grab this from the GitHub repo or the current container)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
WORKDIR /app
RUN pip install -r requirements.txt && \
    pip install tensorflow-gpu==1.14 keras==2.2.4

# Clone ScanNet repo
RUN git clone https://github.com/jertubiana/ScanNet.git /ScanNet
WORKDIR /ScanNet
