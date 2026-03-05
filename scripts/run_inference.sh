#!/bin/bash
set -euo pipefail

# Run Aurora inference inside Docker.
# All extra arguments are forwarded to inference.py.
#
# Examples:
#   bash scripts/run_inference.sh                        # default: small grid, single step
#   bash scripts/run_inference.sh --small                # use small model
#   bash scripts/run_inference.sh --rollout-steps 5      # 5-step autoregressive rollout
#   bash scripts/run_inference.sh --device cpu --small    # CPU test with small model

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$SCRIPT_DIR":/app \
    aurora:latest \
    python -m src.inference "$@"
