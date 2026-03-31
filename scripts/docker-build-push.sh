#!/bin/bash
set -euo pipefail

# Build and push the Aurora Docker image to Docker Hub.
# Run this whenever you change dependencies in pyproject.toml.
#
# Usage:
#   bash scripts/docker-build-push.sh

IMAGE="puschb/aurora-dev:latest"

echo "Building image: $IMAGE"
docker build -t "$IMAGE" -f docker/Dockerfile .

echo ""
echo "Pushing image: $IMAGE"
docker push "$IMAGE"

echo ""
echo "Done. Image available at: docker.io/$IMAGE"
