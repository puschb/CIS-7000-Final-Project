"""Compute normalisation statistics (mean, std) for new ERA5 surface variables.

Reads the per-timestep layout produced by scripts/split_chunk_to_per_timestep.py:
files named ``YYYY-MM-DDTHH-surface.nc`` under the data directories.

Only processes timestamps within the TRAINING split (Jun+Jul 2024, Jun+Jul 2025)
to avoid data leakage from val/test.

Uses Welford's online algorithm so the full dataset never needs to fit in memory.
Each file contains exactly one timestep so we open it, read the arrays, and close.

Usage:
    python -u scripts/compute_norm_stats.py \
        --data-dir /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025
    python -u scripts/compute_norm_stats.py \
        --data-dir /mnt/data/era5/per-step/2024 --vars swvl1 stl1 sd
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_VARS = ["swvl1", "stl1", "sd"]

# Training date ranges -- must match src/data.DEFAULT_TRAIN_RANGES
TRAIN_RANGES: list[tuple[datetime, datetime]] = [
    (datetime(2024, 6, 1), datetime(2024, 8, 1)),
    (datetime(2025, 6, 1), datetime(2025, 8, 1)),
]

# Per-timestep file pattern: 2024-06-16T12-surface.nc
_PER_STEP_PAT = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2})-surface\.nc$")


def _parse_per_step_dt(fname: str) -> datetime | None:
    m = _PER_STEP_PAT.match(fname)
    if not m:
        return None
    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def _timestamp_in_train(dt: datetime) -> bool:
    for rng_start, rng_end in TRAIN_RANGES:
        if rng_start <= dt < rng_end:
            return True
    return False


def _welford_update(
    count: dict[str, int],
    mean: dict[str, float],
    m2: dict[str, float],
    vmin: dict[str, float],
    vmax: dict[str, float],
    var: str,
    arr: np.ndarray,
) -> None:
    """Batch Welford update for one variable's spatial field."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return
    vmin[var] = min(vmin[var], float(valid.min()))
    vmax[var] = max(vmax[var], float(valid.max()))
    n_new = len(valid)
    new_mean = valid.mean()
    new_m2 = valid.var() * n_new
    n_old = count[var]
    n_total = n_old + n_new
    delta = new_mean - mean[var]
    mean[var] = (n_old * mean[var] + n_new * new_mean) / n_total
    m2[var] += new_m2 + delta**2 * n_old * n_new / n_total
    count[var] = n_total


def compute_stats(
    data_dirs: list[Path], var_names: list[str]
) -> dict[str, dict[str, float]]:
    """Compute global mean and std using per-timestep surface files in data_dirs."""
    surface_files: list[tuple[datetime, Path]] = []
    for d in data_dirs:
        for f in sorted(d.glob("*-surface.nc")):
            dt = _parse_per_step_dt(f.name)
            if dt is not None and _timestamp_in_train(dt):
                surface_files.append((dt, f))

    surface_files.sort(key=lambda x: x[0])

    if not surface_files:
        raise FileNotFoundError(
            f"No training-split YYYY-MM-DDTHH-surface.nc files found in {data_dirs}.\n"
            f"Expected layout: per-timestep files from split_chunk_to_per_timestep.py\n"
            f"Training ranges: {TRAIN_RANGES}"
        )

    print(f"Found {len(surface_files)} per-timestep surface files in training split")
    print(f"Training ranges: {TRAIN_RANGES}")
    print(f"Variables: {var_names}\n")

    count: dict[str, int] = {v: 0 for v in var_names}
    mean: dict[str, float] = {v: 0.0 for v in var_names}
    m2: dict[str, float] = {v: 0.0 for v in var_names}
    vmin: dict[str, float] = {v: float("inf") for v in var_names}
    vmax: dict[str, float] = {v: float("-inf") for v in var_names}

    log_every = max(1, len(surface_files) // 20)
    for i, (dt, fpath) in enumerate(surface_files):
        ds = xr.open_dataset(fpath, engine="netcdf4")
        for var in var_names:
            if var not in ds:
                continue
            arr = ds[var].isel(valid_time=0).values.astype(np.float64).ravel()
            _welford_update(count, mean, m2, vmin, vmax, var, arr)
        ds.close()

        if (i + 1) % log_every == 0 or i + 1 == len(surface_files):
            print(
                f"  {i + 1}/{len(surface_files)}  {fpath.name}  "
                f"(swvl1 n={count.get('swvl1', 0):,})",
                flush=True,
            )

    results: dict[str, dict[str, float]] = {}
    for var in var_names:
        if count[var] < 2:
            print(f"\n  WARNING: {var} has {count[var]} valid samples, cannot compute std")
            continue
        std = np.sqrt(m2[var] / count[var])
        results[var] = {
            "mean": mean[var],
            "std": std,
            "min": vmin[var],
            "max": vmax[var],
            "count": count[var],
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute ERA5 normalisation statistics")
    parser.add_argument(
        "--data-dir",
        required=True,
        nargs="+",
        help="One or more directories containing ERA5 surface NetCDF files",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        default=DEFAULT_VARS,
        help=f"Variables to compute stats for (default: {DEFAULT_VARS})",
    )
    args = parser.parse_args()

    data_dirs = [Path(d) for d in args.data_dir]
    results = compute_stats(data_dirs, args.vars)

    print("\n" + "=" * 60)
    print("Normalisation statistics  (paste into src/finetune.py)")
    print("=" * 60)
    print()
    print("_NEW_VAR_NORM: dict[str, tuple[float, float]] = {")
    for var, info in results.items():
        print(f'    "{var}": ({info["mean"]:.6e}, {info["std"]:.6e}),')
    print("}")
    print()

    print("# Or set them directly:")
    for var, info in results.items():
        print(f'locations["{var}"] = {info["mean"]:.6e}')
        print(f'scales["{var}"] = {info["std"]:.6e}')

    print()
    print("Summary:")
    for var, info in results.items():
        print(
            f"  {var}: mean={info['mean']:.6e}, std={info['std']:.6e}, "
            f"range=[{info['min']:.4f}, {info['max']:.4f}], "
            f"n={info['count']:,}"
        )


if __name__ == "__main__":
    main()
