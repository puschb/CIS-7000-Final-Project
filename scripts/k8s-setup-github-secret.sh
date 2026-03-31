#!/bin/bash
set -euo pipefail

# Creates a Kubernetes secret for cloning the private GitHub repo.
# Run this ONCE before using any k8s jobs or pods.
#
# Usage:
#   bash scripts/k8s-setup-github-secret.sh
#
# You need a GitHub Personal Access Token with 'repo' scope.
# Create one at: https://github.com/settings/tokens

echo "=== GitHub Secret Setup for Nautilus ==="
echo ""
echo "This creates a Kubernetes secret so pods can clone your private repo."
echo "You need a GitHub Personal Access Token (PAT) with 'repo' scope."
echo "Create one at: https://github.com/settings/tokens"
echo ""

read -p "GitHub username: " GH_USER

if [ -z "$GH_USER" ]; then
    echo "Error: username cannot be empty."
    exit 1
fi

read -sp "GitHub Personal Access Token: " GH_TOKEN
echo ""

if [ -z "$GH_TOKEN" ]; then
    echo "Error: token cannot be empty."
    exit 1
fi

# Delete existing secret if present
kubectl delete secret github-secret --ignore-not-found=true 2>/dev/null

kubectl create secret generic github-secret \
    --from-literal=user="$GH_USER" \
    --from-literal=password="$GH_TOKEN"

echo ""
echo "Secret 'github-secret' created in namespace $(kubectl config view --minify -o jsonpath='{.contexts[0].context.namespace}')."
echo "All k8s jobs and pods can now clone the private repo."
