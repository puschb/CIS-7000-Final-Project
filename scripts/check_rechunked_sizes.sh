#!/usr/bin/env bash
# Run inside a pod with /mnt/dst mounted to era5-rechunked PVC
set -e
DST="${1:-/mnt/dst}"

echo "=== Suspected partial (v4 Step 3 interrupted) — check exists + size ==="
for f in \
  era5/2025/2025-06-d13-15-atmospheric.nc \
  era5/2025/2025-06-d22-24-atmospheric.nc \
  era5/2025/2025-06-d28-30-atmospheric.nc \
  era5/2025/2025-07-d01-03-atmospheric.nc \
  era5/2025/2025-07-d04-06-atmospheric.nc \
  era5/2025/2025-07-d07-09-atmospheric.nc \
  era5/2025/2025-07-d10-12-atmospheric.nc \
  era5/2025/2025-07-d13-15-atmospheric.nc \
  era5/2025/2025-07-d16-18-atmospheric.nc \
  era5/2025/2025-07-d19-21-atmospheric.nc
do
  p="$DST/$f"
  if [[ -f "$p" ]]; then
    sz=$(stat -c%s "$p")
    echo "$((sz / 1048576)) MB  $f"
  else
    echo "MISSING  $f"
  fi
done

echo
echo "=== Atmospheric files smaller than 5500 MiB (likely bad / partial) ==="
find "$DST/era5" -name '*atmospheric*.nc' -type f -print0 | while IFS= read -r -d '' p; do
  sz=$(stat -c%s "$p")
  if (( sz < 5500 * 1024 * 1024 )); then
    echo "$((sz / 1048576)) MB  ${p#$DST/}"
  fi
done

echo
echo "=== Surface files smaller than 4500 MiB (likely bad / partial) ==="
find "$DST/era5" -name '*surface*.nc' -type f -print0 | while IFS= read -r -d '' p; do
  sz=$(stat -c%s "$p")
  if (( sz < 4500 * 1024 * 1024 )); then
    echo "$((sz / 1048576)) MB  ${p#$DST/}"
  fi
done
