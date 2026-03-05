#!/bin/bash
set -euo pipefail

# Run Aurora fine-tuning inside Docker.
# All extra arguments are forwarded to finetune.py.
#
# Examples:
#   bash scripts/run_finetune.sh                   # default: 5 steps, small grid
#   bash scripts/run_finetune.sh --steps 100       # 100 training steps
#   bash scripts/run_finetune.sh --lr 1e-4         # custom learning rate

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GCS_BUCKET="${GCS_BUCKET:-cis7000-aurora-data}"

docker run --rm -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
    -v "$SCRIPT_DIR":/app \
    aurora:latest \
    python -m src.finetune "$@"

echo ""
echo "Fine-tuning complete. To save checkpoints to GCS:"
echo "  gsutil -m rsync -r ./checkpoints gs://$GCS_BUCKET/checkpoints"
