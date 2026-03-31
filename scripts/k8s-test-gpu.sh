#!/bin/bash
set -euo pipefail

# Submit GPU test job (small model on A100 80GB)
# Logs are streamed to terminal. Cleans up on exit.

JOB_NAME="aurora-test"
YAML="k8s/aurora-test-job.yaml"

echo "Submitting job: $JOB_NAME"
kubectl create -f "$YAML"

echo "Waiting for pod to be scheduled (may take a while for GPU)..."
kubectl wait --for=condition=Ready pod -l job-name="$JOB_NAME" --timeout=600s 2>/dev/null || true

POD=$(kubectl get pods -l job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD"
echo "--- Logs ---"
kubectl logs -f "$POD"

echo ""
echo "Cleaning up..."
kubectl delete job "$JOB_NAME"
echo "Done."
