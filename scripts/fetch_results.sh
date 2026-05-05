#!/usr/bin/env bash
# Copies metrics + logs from all stage1 runs on the era5-rechunked PVC
# into results/<run_name>/ in the local repo, then commits and pushes.
#
# Usage:
#   bash scripts/fetch_results.sh
#
# Requirements: kubectl configured with upenn-dyer-lab namespace access,
#               an active pod mounting the PVC (aurora-interactive).

set -euo pipefail

NAMESPACE="upenn-dyer-lab"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"

# Find the running pod that mounts the PVC
POD=$(kubectl -n "$NAMESPACE" get pods -l job-name=aurora-interactive \
      --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

if [[ -z "$POD" ]]; then
  echo "ERROR: No running aurora-interactive pod found in namespace $NAMESPACE."
  echo "       Start one with: kubectl -n $NAMESPACE create -f k8s/aurora-interactive-pod.yaml"
  exit 1
fi

echo "Using pod: $POD"

# List all run directories on the PVC
RUNS=$(kubectl -n "$NAMESPACE" exec "$POD" -- sh -c "ls /mnt/data/runs/ | grep '^stage1_'" 2>/dev/null)

if [[ -z "$RUNS" ]]; then
  echo "No stage1_* runs found on PVC."
  exit 0
fi

for RUN in $RUNS; do
  PVC_RUN="/mnt/data/runs/$RUN"

  # Skip runs with no metrics
  LINE_COUNT=$(kubectl -n "$NAMESPACE" exec "$POD" -- \
    sh -c "wc -l < $PVC_RUN/metrics/metrics.jsonl 2>/dev/null || echo 0")
  if [[ "$LINE_COUNT" -eq 0 ]]; then
    echo "Skipping $RUN (empty metrics)"
    continue
  fi

  echo "Fetching $RUN ($LINE_COUNT metric lines)..."
  LOCAL_DIR="$RESULTS_DIR/$RUN"
  mkdir -p "$LOCAL_DIR"

  kubectl -n "$NAMESPACE" cp "$POD:$PVC_RUN/metrics/metrics.jsonl" "$LOCAL_DIR/metrics.jsonl" 2>/dev/null || true
  kubectl -n "$NAMESPACE" cp "$POD:$PVC_RUN/metrics/summary.json"  "$LOCAL_DIR/summary.json"  2>/dev/null || true
  kubectl -n "$NAMESPACE" cp "$POD:$PVC_RUN/logs/train.log"        "$LOCAL_DIR/train.log"      2>/dev/null || true
done

echo ""
echo "=== Committing results ==="
cd "$REPO_ROOT"
git add results/
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -m "Update Stage 1 training results from Nautilus PVC"
  git push
  echo "Pushed to remote."
fi
