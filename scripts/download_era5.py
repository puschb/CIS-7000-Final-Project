"""Download ERA5 data for a range of dates to a target directory.

Usage:
    python -u download_era5.py --start 2025-01-01 --days 365 --out /mnt/data/era5
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import cdsapi

SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
STATIC_VARS = ["geopotential", "land_sea_mask", "soil_type"]
ATMOS_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
]
PRESSURE_LEVELS = [
    "50", "100", "150", "200", "250", "300",
    "400", "500", "600", "700", "850", "925", "1000",
]
TIMES = ["00:00", "06:00", "12:00", "18:00"]


def download_day(c: cdsapi.Client, date: datetime, out_dir: Path) -> bool:
    """Download one day. Returns True if any new files were downloaded."""
    y = str(date.year)
    m = f"{date.month:02d}"
    d = f"{date.day:02d}"
    date_str = f"{y}-{m}-{d}"
    downloaded = False

    static_path = out_dir / "static.nc"
    if not static_path.exists():
        print(f"  Downloading static variables...", flush=True)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {"product_type": "reanalysis", "variable": STATIC_VARS,
             "year": y, "month": m, "day": d, "time": "00:00", "format": "netcdf"},
            str(static_path),
        )
        downloaded = True

    surf_path = out_dir / f"{date_str}-surface-level.nc"
    if not surf_path.exists():
        print(f"  Downloading surface-level variables...", flush=True)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {"product_type": "reanalysis", "variable": SURFACE_VARS,
             "year": y, "month": m, "day": d, "time": TIMES, "format": "netcdf"},
            str(surf_path),
        )
        downloaded = True
    else:
        print(f"  {surf_path.name} exists, skipping", flush=True)

    atmos_path = out_dir / f"{date_str}-atmospheric.nc"
    if not atmos_path.exists():
        print(f"  Downloading atmospheric variables...", flush=True)
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {"product_type": "reanalysis", "variable": ATMOS_VARS,
             "pressure_level": PRESSURE_LEVELS,
             "year": y, "month": m, "day": d, "time": TIMES, "format": "netcdf"},
            str(atmos_path),
        )
        downloaded = True
    else:
        print(f"  {atmos_path.name} exists, skipping", flush=True)

    return downloaded


def count_done(out_dir: Path) -> int:
    return len(list(out_dir.glob("*-surface-level.nc")))


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 data")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, required=True, help="Number of days")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    already_done = count_done(out_dir)
    print(f"Target: {args.days} days starting {args.start}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print(f"Already downloaded: {already_done} days\n", flush=True)

    t_start = time.time()

    for i in range(args.days):
        date = start + timedelta(days=i)
        t0 = time.time()
        print(f"[{i + 1:3d}/{args.days}] {date.strftime('%Y-%m-%d')}", flush=True)
        download_day(c, date, out_dir)

        done = count_done(out_dir)
        elapsed_day = time.time() - t0
        elapsed_total = time.time() - t_start
        remaining = args.days - done
        avg = elapsed_total / max(done - already_done, 1)
        eta_h = (remaining * avg) / 3600

        print(f"         {done}/{args.days} done | "
              f"this day: {elapsed_day:.0f}s | "
              f"total: {elapsed_total / 3600:.1f}h | "
              f"ETA: {eta_h:.1f}h\n", flush=True)

    print("=== Download complete ===", flush=True)
    total = sum(f.stat().st_size for f in out_dir.glob("*.nc"))
    print(f"Files: {len(list(out_dir.glob('*.nc')))}  "
          f"Total: {total / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
