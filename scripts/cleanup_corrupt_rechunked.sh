#!/usr/bin/env bash
# Remove truncated 2025 atmospheric outputs and stale v4 logs from era5-rechunked PVC.
# Mount PVC at /mnt/dst (read-write).
set -euo pipefail
DST="${1:-/mnt/dst}"

FILES=(
  "era5/2025/2025-06-d13-15-atmospheric.nc"
  "era5/2025/2025-06-d22-24-atmospheric.nc"
  "era5/2025/2025-06-d28-30-atmospheric.nc"
  "era5/2025/2025-07-d01-03-atmospheric.nc"
  "era5/2025/2025-07-d04-06-atmospheric.nc"
  "era5/2025/2025-07-d07-09-atmospheric.nc"
  "era5/2025/2025-07-d10-12-atmospheric.nc"
  "era5/2025/2025-07-d13-15-atmospheric.nc"
  "era5/2025/2025-07-d16-18-atmospheric.nc"
  "era5/2025/2025-07-d19-21-atmospheric.nc"
  "era5/2025/2025-07-d22-24-atmospheric.nc"
  "era5/2025/2025-07-d25-27-atmospheric.nc"
  "era5/2025/2025-07-d28-30-atmospheric.nc"
  "era5/2025/2025-07-d31-31-atmospheric.nc"
)

echo "Removing bad / to-be-redone NetCDF outputs:"
for f in "${FILES[@]}"; do
  p="$DST/$f"
  if [[ -f "$p" ]]; then
    sz=$(stat -c%s "$p")
    echo "  rm $f ($((sz / 1048576)) MB)"
    rm -f "$p"
  else
    echo "  (missing, skip) $f"
  fi
done

echo "Removing stale rechunk logs (names match job FILE with tr / -, not era5/ prefix):"
for f in "${FILES[@]}"; do
  rel="${f#era5/}"
  base=$(echo "$rel" | tr '/' '-')
  for log in "$DST/logs/v4-$base.log" "$DST/logs/v5-$base.log"; do
    if [[ -f "$log" ]]; then
      echo "  rm logs/$(basename "$log")"
      rm -f "$log"
    fi
  done
done

echo "Done."
