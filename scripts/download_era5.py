"""Download ERA5 data for a range of dates to a target directory.

Usage:
    python download_era5.py --start 2023-01-01 --days 10 --out /mnt/data/era5
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cdsapi


def download_day(c: cdsapi.Client, date: datetime, out_dir: Path) -> None:
    y = str(date.year)
    m = f"{date.month:02d}"
    d = f"{date.day:02d}"
    date_str = f"{y}-{m}-{d}"

    static_path = out_dir / "static.nc"
    if not static_path.exists():
        print(f"  Downloading static variables...")
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["geopotential", "land_sea_mask", "soil_type"],
                "year": y, "month": m, "day": d, "time": "00:00",
                "format": "netcdf",
            },
            str(static_path),
        )

    surf_path = out_dir / f"{date_str}-surface-level.nc"
    if not surf_path.exists():
        print(f"  Downloading surface-level variables...")
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "mean_sea_level_pressure",
                ],
                "year": y, "month": m, "day": d,
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "format": "netcdf",
            },
            str(surf_path),
        )
    else:
        print(f"  {surf_path.name} already exists, skipping")

    atmos_path = out_dir / f"{date_str}-atmospheric.nc"
    if not atmos_path.exists():
        print(f"  Downloading atmospheric variables...")
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "specific_humidity",
                    "geopotential",
                ],
                "pressure_level": [
                    "50", "100", "150", "200", "250", "300",
                    "400", "500", "600", "700", "850", "925", "1000",
                ],
                "year": y, "month": m, "day": d,
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "format": "netcdf",
            },
            str(atmos_path),
        )
    else:
        print(f"  {atmos_path.name} already exists, skipping")


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

    for i in range(args.days):
        date = start + timedelta(days=i)
        print(f"[{i + 1}/{args.days}] {date.strftime('%Y-%m-%d')}")
        download_day(c, date, out_dir)

    print(f"\nDone. Files in {out_dir}:")
    total = 0
    for f in sorted(out_dir.glob("*.nc")):
        size = f.stat().st_size
        total += size
        print(f"  {f.name:40s} {size / 1e6:8.1f} MB")
    print(f"  {'TOTAL':40s} {total / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
