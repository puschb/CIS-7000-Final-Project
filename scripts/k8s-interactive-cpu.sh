#!/bin/bash
set -euo pipefail

# Launch a CPU-only interactive pod and exec into it.
# Repo is cloned and aurora is pre-installed. Working dir: /opt/repo
# Cleans up the pod when you exit the shell.

POD_NAME="aurora-dev-cpu"
YAML="k8s/aurora-interactive-cpu-pod.yaml"

kubectl delete pod "$POD_NAME" --ignore-not-found=true 2>/dev/null

echo "Creating pod: $POD_NAME"
kubectl create -f "$YAML"

echo "Waiting for pod to be ready (cloning repo)..."
kubectl wait --for=condition=Ready pod "$POD_NAME" --timeout=600s 2>/dev/null || {
    echo "Pod not ready yet. Checking status..."
    kubectl describe pod "$POD_NAME" | tail -20
    echo ""
    echo "Connect manually when ready:"
    echo "  kubectl exec -it $POD_NAME -- /bin/bash -c 'cd /opt/repo && bash'"
    exit 1
}

echo "Connecting to pod (working dir: /opt/repo)..."
echo "(Type 'exit' to leave. Pod will be deleted after.)"
echo ""
kubectl exec -it "$POD_NAME" -- /bin/bash -c 'cd /opt/repo && exec bash'

echo ""
echo "Cleaning up pod..."
kubectl delete pod "$POD_NAME"
echo "Done."
