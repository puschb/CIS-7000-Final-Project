#!/bin/bash
set -euo pipefail

# Submit CPU-only test job (small model, no GPU needed)
# Logs are streamed to terminal. Cleans up on exit.

JOB_NAME="aurora-test-cpu"
YAML="k8s/aurora-test-cpu-job.yaml"

echo "Submitting job: $JOB_NAME"
kubectl create -f "$YAML"

echo "Waiting for pod to start..."
kubectl wait --for=condition=Ready pod -l job-name="$JOB_NAME" --timeout=300s 2>/dev/null || true

POD=$(kubectl get pods -l job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD"
echo "--- Logs ---"
kubectl logs -f "$POD"

echo ""
echo "Cleaning up..."
kubectl delete job "$JOB_NAME"
echo "Done."
