"""Compute normalisation statistics (mean, std) for new ERA5 surface variables.

Scans the original monthly surface files (``YYYY-MM-surface.nc``) which are
much faster to iterate because you only open ~4 files total (Jun+Jul × 2 years)
instead of ~2,880 individual per-timestep files.  The monthly files are still
present on the PVC at /mnt/data/era5/2024/ and /mnt/data/era5/2025/.

Only processes timestamps within the TRAINING split (Jun+Jul 2024, Jun+Jul 2025)
to avoid data leakage from val/test.

Uses Welford's online algorithm so the full dataset never needs to fit in memory.

**Where the metrics live**

The script does **not** write a file. It prints mean/std to stdout (e.g. ``kubectl logs
job/compute-norm-stats``). You copy the printed ``_NEW_VAR_NORM`` block into
``src/finetune.py``. That dict is the persistent source of truth in git.

**How they are used**

Before constructing the model, ``src/finetune._register_new_var_normalisation()``
writes each ``(mean, std)`` into ``aurora.normalisation.locations`` and
``aurora.normalisation.scales`` (module-level dicts shipped with ``microsoft-aurora``).
Aurora's forward pass reads those globals to normalize inputs and unnormalize
outputs for each surface variable. Density channels (``*_density``) are registered
separately in code (mean 0, std 1) and are not produced by this script.

Usage:
    python -u scripts/compute_norm_stats.py \
        --data-dir /mnt/data/era5/2024 /mnt/data/era5/2025
    python -u scripts/compute_norm_stats.py \
        --data-dir /mnt/data/era5/2024 --vars swvl1 stl1 sd
"""

import argparse
import re
from calendar import monthrange
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

# Monthly surface file pattern: 2024-06-surface.nc
_MONTHLY_PAT = re.compile(r"^(\d{4})-(\d{2})-surface\.nc$")


def _month_in_train(year: int, month: int) -> bool:
    """True if any day of this (year, month) falls within a training range."""
    month_start = datetime(year, month, 1)
    month_end = datetime(year, month, monthrange(year, month)[1], 23, 59)
    for rng_start, rng_end in TRAIN_RANGES:
        if month_start < rng_end and month_end >= rng_start:
            return True
    return False


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
    """Compute global mean and std for each variable using monthly surface files."""
    surface_files: list[Path] = []
    for d in data_dirs:
        for f in sorted(d.glob("*-surface.nc")):
            m = _MONTHLY_PAT.match(f.name)
            if m and _month_in_train(int(m.group(1)), int(m.group(2))):
                surface_files.append(f)

    if not surface_files:
        raise FileNotFoundError(
            f"No training-split YYYY-MM-surface.nc files found in {data_dirs}.\n"
            f"Training ranges: {TRAIN_RANGES}"
        )

    print(f"Found {len(surface_files)} monthly surface files in training split")
    print(f"Training ranges: {TRAIN_RANGES}")
    print(f"Variables: {var_names}\n")

    count: dict[str, int] = {v: 0 for v in var_names}
    mean: dict[str, float] = {v: 0.0 for v in var_names}
    m2: dict[str, float] = {v: 0.0 for v in var_names}
    vmin: dict[str, float] = {v: float("inf") for v in var_names}
    vmax: dict[str, float] = {v: float("-inf") for v in var_names}

    for fpath in surface_files:
        print(f"  Opening {fpath.name} ...", end="", flush=True)
        ds = xr.open_dataset(fpath, engine="netcdf4")
        times = ds["valid_time"].values
        n_included = 0
        for t_idx in range(len(times)):
            dt = np.datetime64(times[t_idx], "us").astype("datetime64[s]").item()
            if not _timestamp_in_train(dt):
                continue
            n_included += 1
            for var in var_names:
                if var not in ds:
                    continue
                arr = ds[var].isel(valid_time=t_idx).values.astype(np.float64).ravel()
                _welford_update(count, mean, m2, vmin, vmax, var, arr)
        ds.close()
        print(f" ({n_included} timesteps in training range)", flush=True)

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
