"""Download ERA5 data in bulk requests.

Surface variables are downloaded per month (12 API calls).
Atmospheric variables are downloaded in 3-day chunks to keep file sizes
manageable (~5-7 GB per chunk instead of ~50-70 GB per month).
Static variables are downloaded once.

All 24 hourly time steps are included to maximise training pair diversity.

Usage:
    python -u download_era5.py --year 2025 --out /mnt/data/era5
    python -u download_era5.py --year 2025 --months 1 2 3 --out /mnt/data/era5
"""

import argparse
import calendar
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi

SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "volumetric_soil_water_layer_1",
    "soil_temperature_level_1",
    "snow_depth",
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

ALL_TIMES = [f"{h:02d}:00" for h in range(24)]

ATMOS_CHUNK_DAYS = 3


def download_static(c: cdsapi.Client, year: str, out_dir: Path):
    """Download static variables once."""
    path = out_dir / "static.nc"
    if path.exists():
        print(f"  {path.name} exists, skipping", flush=True)
        return
    print("  Downloading static variables...", flush=True)
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": STATIC_VARS,
            "year": year,
            "month": "01",
            "day": "01",
            "time": "00:00",
            "data_format": "netcdf",
        },
        str(path),
    )


def download_surface_month(c: cdsapi.Client, year: str, month: int, out_dir: Path):
    """Download all surface variables for one month."""
    m = f"{month:02d}"
    path = out_dir / f"{year}-{m}-surface.nc"
    if path.exists():
        print(f"  {path.name} exists, skipping", flush=True)
        return

    n_days = calendar.monthrange(int(year), month)[1]
    days = [f"{d:02d}" for d in range(1, n_days + 1)]

    print(f"  Downloading surface variables ({n_days} days × 24h)...", flush=True)
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": SURFACE_VARS,
            "year": year,
            "month": m,
            "day": days,
            "time": ALL_TIMES,
            "data_format": "netcdf",
        },
        str(path),
    )


def download_atmos_chunk(
    c: cdsapi.Client, year: str, month: int, day_start: int, day_end: int, out_dir: Path
):
    """Download atmospheric variables for a chunk of days within a month."""
    m = f"{month:02d}"
    path = out_dir / f"{year}-{m}-d{day_start:02d}-{day_end:02d}-atmospheric.nc"
    if path.exists():
        print(f"    {path.name} exists, skipping", flush=True)
        return

    days = [f"{d:02d}" for d in range(day_start, day_end + 1)]
    n_days = len(days)

    print(f"    days {day_start:02d}–{day_end:02d} ({n_days} days × 24h)...", flush=True)
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": ATMOS_VARS,
            "pressure_level": PRESSURE_LEVELS,
            "year": year,
            "month": m,
            "day": days,
            "time": ALL_TIMES,
            "data_format": "netcdf",
        },
        str(path),
    )


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 data (bulk)")
    parser.add_argument("--year", required=True, help="Year to download (e.g. 2025)")
    parser.add_argument(
        "--months", type=int, nargs="+", default=list(range(1, 13)),
        help="Months to download (default: all 12)",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=1,
        help="Number of parallel CDS downloads (default: 1)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_months = len(args.months)
    total_days = sum(calendar.monthrange(int(args.year), m)[1] for m in args.months)
    atmos_calls = sum(
        math.ceil(calendar.monthrange(int(args.year), m)[1] / ATMOS_CHUNK_DAYS)
        for m in args.months
    )
    total_calls = total_months + atmos_calls + 1

    print(f"Year: {args.year}", flush=True)
    print(f"Months: {args.months}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Surface vars: {len(SURFACE_VARS)}", flush=True)
    print(f"Atmospheric vars: {len(ATMOS_VARS)} × {len(PRESSURE_LEVELS)} levels", flush=True)
    print(f"Atmospheric chunk size: {ATMOS_CHUNK_DAYS} days", flush=True)
    print(f"API calls: {total_months} surface + {atmos_calls} atmos + 1 static = {total_calls}\n", flush=True)

    t_start = time.time()
    year = args.year

    # Each task creates its own cdsapi.Client for thread safety.
    tasks: list[tuple[str, callable]] = []

    tasks.append(("static", lambda: download_static(cdsapi.Client(), year, out_dir)))

    for month in args.months:
        m = month
        tasks.append((
            f"surface {year}-{m:02d}",
            lambda m=m: download_surface_month(cdsapi.Client(), year, m, out_dir),
        ))
        n_days = calendar.monthrange(int(year), m)[1]
        d = 1
        while d <= n_days:
            end = min(d + ATMOS_CHUNK_DAYS - 1, n_days)
            tasks.append((
                f"atmos {year}-{m:02d} d{d:02d}-{end:02d}",
                lambda m=m, ds=d, de=end: download_atmos_chunk(
                    cdsapi.Client(), year, m, ds, de, out_dir
                ),
            ))
            d = end + 1

    def _run_task(label, fn):
        try:
            print(f"[{label}]...", flush=True)
            fn()
            return label, None
        except Exception as e:
            print(f"  FAILED ({label}), re-queuing: {e}", flush=True)
            return label, fn

    while tasks:
        batch = tasks[:args.workers]
        tasks = tasks[len(batch):]

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_task, label, fn) for label, fn in batch]
            for future in as_completed(futures):
                label, retry_fn = future.result()
                if retry_fn is not None:
                    tasks.append((label, retry_fn))

    elapsed = time.time() - t_start
    nc_files = list(out_dir.glob("*.nc"))
    total_bytes = sum(f.stat().st_size for f in nc_files)
    print("\n=== Download complete ===", flush=True)
    print(
        f"Files: {len(nc_files)}  "
        f"Total: {total_bytes / 1e9:.1f} GB  "
        f"Time: {elapsed / 3600:.1f}h",
        flush=True,
    )


if __name__ == "__main__":
    main()
