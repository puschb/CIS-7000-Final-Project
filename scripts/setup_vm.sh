#!/bin/bash
set -euo pipefail

# Run this after SSHing into the VM to set up the project.
# Usage: bash scripts/setup_vm.sh

REPO_URL="https://github.com/puschb/CIS-7000-Final-Project.git"
PROJECT_DIR="$HOME/CIS-7000-Final-Project"
GCS_BUCKET="${GCS_BUCKET:-cis7000-aurora-data}"

echo "=== VM Setup ==="

# Clone or pull the repo
if [ -d "$PROJECT_DIR" ]; then
    echo "Repo exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull
else
    echo "Cloning repo..."
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# Build Docker image
echo "Building Docker image..."
docker build -t aurora:latest -f docker/Dockerfile .

# Sync data from GCS (if any exists)
echo "Syncing data from GCS..."
mkdir -p data checkpoints
gsutil -m rsync -r "gs://$GCS_BUCKET/data/" ./data/ 2>/dev/null || echo "No data in GCS yet."
gsutil -m rsync -r "gs://$GCS_BUCKET/checkpoints/" ./checkpoints/ 2>/dev/null || echo "No checkpoints in GCS yet."

echo ""
echo "=== Setup complete ==="
echo "Run inference:    bash scripts/run_inference.sh"
echo "Run fine-tuning:  bash scripts/run_finetune.sh"
