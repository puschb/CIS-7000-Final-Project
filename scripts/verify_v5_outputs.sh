#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/mnt/dst}"

EXPECTED=(
  era5/2025/2025-06-d13-15-atmospheric.nc
  era5/2025/2025-06-d22-24-atmospheric.nc
  era5/2025/2025-06-d28-30-atmospheric.nc
  era5/2025/2025-07-d01-03-atmospheric.nc
  era5/2025/2025-07-d04-06-atmospheric.nc
  era5/2025/2025-07-d07-09-atmospheric.nc
  era5/2025/2025-07-d10-12-atmospheric.nc
  era5/2025/2025-07-d13-15-atmospheric.nc
  era5/2025/2025-07-d16-18-atmospheric.nc
  era5/2025/2025-07-d19-21-atmospheric.nc
  era5/2025/2025-07-d22-24-atmospheric.nc
  era5/2025/2025-07-d25-27-atmospheric.nc
  era5/2025/2025-07-d28-30-atmospheric.nc
  era5/2025/2025-07-d31-31-atmospheric.nc
)

echo "=== Expected NetCDF files (full atmospheric ~6.5–7.5 GiB) ==="
bad=0
for f in "${EXPECTED[@]}"; do
  p="$ROOT/$f"
  if [[ ! -f "$p" ]]; then
    echo "MISSING: $f"
    bad=1
    continue
  fi
  sz=$(stat -c%s "$p")
  mb=$((sz / 1048576))
  if (( mb < 5500 )); then
    echo "TOO_SMALL: ${mb} MB  $f"
    bad=1
  else
    echo "${mb} MB  $f"
  fi
done

echo ""
echo "=== v5 log file count ==="
shopt -s nullglob
logs=( "$ROOT/logs/v5-"*.log )
echo "${#logs[@]} log(s)"

echo ""
echo "=== v5 logs missing DONE ==="
for log in "${logs[@]}"; do
  if ! grep -q "=== DONE ===" "$log"; then
    echo "INCOMPLETE: $(basename "$log")"
    bad=1
  fi
done
if (( bad == 0 )); then
  echo "(all present logs end with DONE)"
fi

echo ""
echo "=== df ==="
df -h "$ROOT"

exit "$bad"
