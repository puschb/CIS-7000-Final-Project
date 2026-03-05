FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    microsoft-aurora \
    google-cloud-storage \
    xarray \
    netCDF4 \
    ruff

WORKDIR /app
