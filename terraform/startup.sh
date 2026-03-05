#!/bin/bash
set -euo pipefail

LOG_FILE="/var/log/startup-script.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Startup script begin: $(date) ==="

# Wait for NVIDIA drivers (Deep Learning VM installs them on first boot)
echo "Waiting for NVIDIA drivers..."
for i in $(seq 1 30); do
    if nvidia-smi &>/dev/null; then
        echo "NVIDIA drivers ready."
        nvidia-smi
        break
    fi
    echo "  Attempt $i/30 — drivers not ready yet, waiting 10s..."
    sleep 10
done

if ! nvidia-smi &>/dev/null; then
    echo "ERROR: NVIDIA drivers did not become available after 5 minutes."
    exit 1
fi

# Install Docker if not present
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi

# Install NVIDIA Container Toolkit
if ! command -v nvidia-ctk &>/dev/null; then
    echo "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update
    apt-get install -y nvidia-container-toolkit

    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# Verify Docker can see the GPU
echo "Verifying Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

echo "=== Startup script complete: $(date) ==="
