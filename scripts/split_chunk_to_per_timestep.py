"""Split one multi-day atmospheric NetCDF + monthly surface into per-hour files.

Writes pairs ``YYYY-MM-DDTHH-surface.nc`` and ``YYYY-MM-DDTHH-atmospheric.nc``
with ``valid_time`` length 1 each (same chunking as ``scripts/rechunk_era5.py``:
zlib level 1, full spatial chunks). Copies ``static.nc`` into the output dir.

Used to benchmark whether many small files reduce CephFS + HDF5 open overhead
vs multi-GB monthly / 3-day files.

Usage (paths on PVC or local):

    python scripts/split_chunk_to_per_timestep.py \\
      --atmos-src /mnt/data/era5/2024/2024-06-d16-18-atmospheric.nc \\
      --surface-src /mnt/data/era5/2024/2024-06-surface.nc \\
      --static-src /mnt/data/era5/2024/static.nc \\
      --out-dir /mnt/data/era5/per-step-bench-2024-06-d16-18
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr


def _to_naive_utc(val) -> datetime:
    """Decode xarray time coordinate value to naive UTC ``datetime``."""
    if isinstance(val, datetime):
        if val.tzinfo is not None:
            return val.astimezone(timezone.utc).replace(tzinfo=None)
        return val
    try:
        v = np.datetime64(val, "s")
        sec = int(v.astype("datetime64[s]").astype("int64"))
        return datetime.utcfromtimestamp(sec)
    except (TypeError, ValueError, OverflowError):
        pass
    return datetime(
        int(val.year),
        int(val.month),
        int(val.day),
        int(getattr(val, "hour", 0)),
        int(getattr(val, "minute", 0)),
        int(getattr(val, "second", 0)),
    )


def _encoding_for(ds: xr.Dataset) -> dict[str, dict]:
    """Match ``scripts/rechunk_era5.py``: one timestep per chunk, zlib 1."""
    enc: dict[str, dict] = {}
    for var in ds.data_vars:
        chunks = tuple(int(ds[var].sizes[d]) for d in ds[var].dims)
        enc[var] = {"chunksizes": chunks, "zlib": True, "complevel": 1}
    return enc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atmos-src", type=Path, required=True)
    p.add_argument("--surface-src", type=Path, required=True)
    p.add_argument("--static-src", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Atmospheric source: {args.atmos_src}")
    print(f"Surface source:     {args.surface_src}")
    print(f"Static source:      {args.static_src}")
    print(f"Output directory:   {out}")
    print()

    t0 = time.perf_counter()
    atmos = xr.open_dataset(args.atmos_src)
    surf = xr.open_dataset(args.surface_src)
    print(f"Opened sources in {time.perf_counter() - t0:.1f}s")

    nt = int(atmos.sizes["valid_time"])
    print(f"Timesteps in atmospheric chunk: {nt}")
    print()

    static_dst = out / "static.nc"
    shutil.copy2(args.static_src, static_dst)
    print(f"Copied static -> {static_dst} ({static_dst.stat().st_size / 1e6:.1f} MB)")
    print()

    written = 0
    t_loop = time.perf_counter()
    for i in range(nt):
        atmos_one = atmos.isel(valid_time=[i])
        t_val = atmos_one["valid_time"].values.item()
        dt = _to_naive_utc(t_val)
        stem = dt.strftime("%Y-%m-%dT%H")

        surf_one = surf.sel(valid_time=atmos_one["valid_time"], drop=True)
        if int(surf_one.sizes["valid_time"]) != 1:
            raise RuntimeError(
                f"Surface slice for {stem} has valid_time size "
                f"{surf_one.sizes['valid_time']}, expected 1"
            )

        surf_path = out / f"{stem}-surface.nc"
        atmos_path = out / f"{stem}-atmospheric.nc"

        surf_one.to_netcdf(surf_path, encoding=_encoding_for(surf_one))
        atmos_one.to_netcdf(atmos_path, encoding=_encoding_for(atmos_one))
        written += 1

        if (i + 1) % 12 == 0 or i + 1 == nt:
            elapsed = time.perf_counter() - t_loop
            print(f"  {i + 1}/{nt}  ({written} pairs)  elapsed {elapsed:.1f}s")

    atmos.close()
    surf.close()
    print()
    print(f"Done. Wrote {written} surface + {written} atmospheric files in {out}")
    print(f"Total wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
