"""Utilities for constructing Aurora batches from various data sources."""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from aurora import Batch, Metadata

SURF_VAR_NAMES = ("2t", "10u", "10v", "msl")
STATIC_VAR_NAMES = ("lsm", "z", "slt")
ATMOS_VAR_NAMES = ("z", "u", "v", "t", "q")
PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

FULL_RES_LAT = 721   # 0.25 degree: 90 to -90
FULL_RES_LON = 1440  # 0.25 degree: 0 to 359.75

# ERA5 NetCDF short names -> Aurora names
SURF_ERA5_TO_AURORA = {
    "t2m": "2t",
    "u10": "10u",
    "v10": "10v",
    "msl": "msl",
}

# Extra surface vars we downloaded that aren't in the standard Aurora model.
# These get added when extending Aurora with new variables.
EXTRA_SURF_ERA5_TO_AURORA = {
    "swvl1": "swvl1",
    "stl1": "stl1",
    "sd": "sd",
}

STATIC_ERA5_TO_AURORA = {
    "z": "z",
    "lsm": "lsm",
    "slt": "slt",
}

ATMOS_ERA5_TO_AURORA = {
    "t": "t",
    "u": "u",
    "v": "v",
    "q": "q",
    "z": "z",
}


# ---------------------------------------------------------------------------
# File index helpers
# ---------------------------------------------------------------------------

def _parse_surface_files(data_dir: Path) -> dict[tuple[int, int], Path]:
    """Map (year, month) -> surface NetCDF path."""
    pattern = re.compile(r"(\d{4})-(\d{2})-surface\.nc$")
    result = {}
    for f in sorted(data_dir.glob("*-surface.nc")):
        m = pattern.search(f.name)
        if m:
            result[(int(m.group(1)), int(m.group(2)))] = f
    return result


def _parse_atmos_files(data_dir: Path) -> list[tuple[datetime, datetime, Path]]:
    """Return sorted list of (start_dt, end_dt, path) for atmospheric chunks."""
    pattern = re.compile(r"(\d{4})-(\d{2})-d(\d{2})-(\d{2})-atmospheric\.nc$")
    chunks = []
    for f in sorted(data_dir.glob("*-atmospheric.nc")):
        m = pattern.search(f.name)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
            day_start, day_end = int(m.group(3)), int(m.group(4))
            dt_start = datetime(year, month, day_start, 0, 0)
            dt_end = datetime(year, month, day_end, 23, 0)
            chunks.append((dt_start, dt_end, f))
    return chunks


def _find_atmos_file(
    dt: datetime, atmos_chunks: list[tuple[datetime, datetime, Path]]
) -> tuple[Path, int] | None:
    """Find the atmospheric chunk file containing *dt* and the time index within it."""
    for chunk_start, chunk_end, path in atmos_chunks:
        if chunk_start <= dt <= chunk_end:
            hours_offset = int((dt - chunk_start).total_seconds() // 3600)
            return path, hours_offset
    return None


def _surface_time_index(dt: datetime) -> int:
    """Time index within a monthly surface file for a given datetime."""
    return (dt.day - 1) * 24 + dt.hour


# ---------------------------------------------------------------------------
# ERA5Dataset
# ---------------------------------------------------------------------------

class ERA5Dataset(Dataset):
    """PyTorch Dataset that serves Aurora-compatible (input, target) Batch pairs
    from ERA5 NetCDF files produced by scripts/download_era5.py.

    Each sample is a triplet of timestamps (t-6h, t, t+6h):
      - input batch: history dim with t-6h and t
      - target batch: ground truth at t+6h

    Args:
        data_dirs: One or more directories containing downloaded ERA5 data.
            Each should have static.nc, monthly surface files, and atmospheric
            chunk files. Typically one dir per year.
        start_date: Inclusive start of the date range (filters triplets).
        end_date: Exclusive end of the date range (filters triplets).
        step_hours: Time gap between consecutive steps (default 6h for Aurora).
        include_extra_surf: If True, include soil moisture / soil temp / snow
            depth in surf_vars (for extending Aurora with new variables).
    """

    def __init__(
        self,
        data_dirs: str | Path | list[str | Path],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        step_hours: int = 6,
        include_extra_surf: bool = True,
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.data_dirs = [Path(d) for d in data_dirs]
        self.step_hours = step_hours
        self.include_extra_surf = include_extra_surf

        self.surf_map = SURF_ERA5_TO_AURORA.copy()
        if include_extra_surf:
            self.surf_map.update(EXTRA_SURF_ERA5_TO_AURORA)

        # Build file indices across all data directories
        self.surface_files: dict[tuple[int, int], Path] = {}
        self.atmos_chunks: list[tuple[datetime, datetime, Path]] = []
        static_path = None

        for d in self.data_dirs:
            self.surface_files.update(_parse_surface_files(d))
            self.atmos_chunks.extend(_parse_atmos_files(d))
            candidate = d / "static.nc"
            if candidate.exists() and static_path is None:
                static_path = candidate

        self.atmos_chunks.sort(key=lambda x: x[0])

        if static_path is None:
            raise FileNotFoundError("No static.nc found in any data directory")

        # Load static vars once (small, reused every sample)
        static_ds = xr.open_dataset(static_path, engine="netcdf4")
        self.static_vars = {
            STATIC_ERA5_TO_AURORA[k]: torch.from_numpy(static_ds[k].values[0]).float()
            for k in STATIC_ERA5_TO_AURORA
        }
        self.lat = torch.from_numpy(static_ds.latitude.values).float()
        self.lon = torch.from_numpy(static_ds.longitude.values).float()
        static_ds.close()

        # Cache for open xarray Dataset handles to avoid re-opening multi-GB
        # files on every __getitem__ call.  Keys are file paths.
        self._ds_cache: dict[Path, xr.Dataset] = {}

        # Build list of valid triplets
        self.triplets = self._build_triplets(start_date, end_date)

    def _build_triplets(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[tuple[datetime, datetime, datetime]]:
        """Enumerate all valid (t-step, t, t+step) triplets within the date range."""
        step = timedelta(hours=self.step_hours)
        triplets = []

        for (year, month), surf_path in sorted(self.surface_files.items()):
            n_days = calendar.monthrange(year, month)[1]
            month_start = datetime(year, month, 1, 0, 0)
            month_end = datetime(year, month, n_days, 23, 0)

            # Iterate every hour in this month as the "current" time t1
            t1 = month_start + step  # earliest t1 so that t0 = t1-step is in range
            while t1 + step <= month_end:
                t0 = t1 - step
                t2 = t1 + step

                # Apply date range filter (on t1, the "current" time)
                if start_date and t1 < start_date:
                    t1 += timedelta(hours=1)
                    continue
                if end_date and t1 >= end_date:
                    break

                # Check that all 3 timestamps are in the same month (surface file)
                if t0.month != month or t2.month != month:
                    t1 += timedelta(hours=1)
                    continue

                # Check atmospheric file coverage for all 3 timestamps
                if (
                    _find_atmos_file(t0, self.atmos_chunks) is not None
                    and _find_atmos_file(t1, self.atmos_chunks) is not None
                    and _find_atmos_file(t2, self.atmos_chunks) is not None
                ):
                    triplets.append((t0, t1, t2))

                t1 += timedelta(hours=1)

        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def _open_ds(self, path: Path) -> xr.Dataset:
        """Return a cached xarray Dataset handle, opening the file only once."""
        if path not in self._ds_cache:
            self._ds_cache[path] = xr.open_dataset(path, engine="netcdf4")
        return self._ds_cache[path]

    def _load_surface(self, dt: datetime) -> dict[str, torch.Tensor]:
        """Load surface variables for a single timestamp."""
        key = (dt.year, dt.month)
        path = self.surface_files[key]
        ds = self._open_ds(path)
        idx = _surface_time_index(dt)
        sliced = ds.isel(valid_time=idx)
        result = {}
        for era5_name, aurora_name in self.surf_map.items():
            result[aurora_name] = torch.from_numpy(
                sliced[era5_name].values
            ).float()
        return result

    def _load_atmos(self, dt: datetime) -> dict[str, torch.Tensor]:
        """Load atmospheric variables for a single timestamp."""
        info = _find_atmos_file(dt, self.atmos_chunks)
        assert info is not None
        path, time_idx = info
        ds = self._open_ds(path)
        sliced = ds.isel(valid_time=time_idx)
        result = {}
        for era5_name, aurora_name in ATMOS_ERA5_TO_AURORA.items():
            result[aurora_name] = torch.from_numpy(
                sliced[era5_name].values
            ).float()
        return result

    def close(self):
        """Close all cached file handles."""
        for ds in self._ds_cache.values():
            ds.close()
        self._ds_cache.clear()

    def __getitem__(self, idx: int) -> tuple[Batch, Batch]:
        t0, t1, t2 = self.triplets[idx]

        # Load 3 timestamps
        surf0, surf1, surf2 = (
            self._load_surface(t0),
            self._load_surface(t1),
            self._load_surface(t2),
        )
        atmos0, atmos1, atmos2 = (
            self._load_atmos(t0),
            self._load_atmos(t1),
            self._load_atmos(t2),
        )

        # Stack into history dimension: (2, H, W) for surf, (2, C, H, W) for atmos
        # Then add batch dim: (1, 2, H, W) and (1, 2, C, H, W)
        input_surf = {
            k: torch.stack([surf0[k], surf1[k]])[None] for k in surf0
        }
        input_atmos = {
            k: torch.stack([atmos0[k], atmos1[k]])[None] for k in atmos0
        }

        # Target: single timestep with batch dim: (1, 1, H, W) and (1, 1, C, H, W)
        target_surf = {k: surf2[k][None, None] for k in surf2}
        target_atmos = {k: atmos2[k][None, None] for k in atmos2}

        atmos_levels = PRESSURE_LEVELS

        input_batch = Batch(
            surf_vars=input_surf,
            static_vars=self.static_vars,
            atmos_vars=input_atmos,
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(t1,),
                atmos_levels=atmos_levels,
            ),
        )

        target_batch = Batch(
            surf_vars=target_surf,
            static_vars=self.static_vars,
            atmos_vars=target_atmos,
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(t2,),
                atmos_levels=atmos_levels,
            ),
        )

        return input_batch, target_batch


# ---------------------------------------------------------------------------
# Train / val / test split helper
# ---------------------------------------------------------------------------

# Default split for the soil-moisture fine-tuning task.
#
# Data: Jun-Aug 2024 and Jun-Aug 2025 (summer only, two years).
#   Train : Jun 1 – Aug 1    both years  (~122 days, 66%)
#   Val   : Aug 1 – Aug 16   both years  (~30 days,  17%)
#   Test  : Aug 16 – Sep 1   both years  (~32 days,  17%)
#
# Val and test draw from both years so any performance difference reflects
# generalization quality, not inter-annual weather variability.
DEFAULT_TRAIN_RANGES: list[tuple[datetime, datetime]] = [
    (datetime(2024, 6, 1), datetime(2024, 8, 1)),   # Jun+Jul 2024
    (datetime(2025, 6, 1), datetime(2025, 8, 1)),   # Jun+Jul 2025
]
DEFAULT_VAL_RANGES: list[tuple[datetime, datetime]] = [
    (datetime(2024, 8, 1), datetime(2024, 8, 16)),  # Aug 1-15 2024
    (datetime(2025, 8, 1), datetime(2025, 8, 16)),  # Aug 1-15 2025
]
DEFAULT_TEST_RANGES: list[tuple[datetime, datetime]] = [
    (datetime(2024, 8, 16), datetime(2024, 9, 1)),  # Aug 16-31 2024
    (datetime(2025, 8, 16), datetime(2025, 9, 1)),  # Aug 16-31 2025
]


class MultiRangeERA5Dataset(Dataset):
    """Thin wrapper that concatenates multiple date-range slices of ERA5Dataset.

    Useful when train/val/test splits are non-contiguous (e.g. summer months
    across multiple years).
    """

    def __init__(
        self,
        data_dirs: str | Path | list[str | Path],
        date_ranges: list[tuple[datetime, datetime]],
        step_hours: int = 6,
        include_extra_surf: bool = True,
    ):
        self.datasets: list[ERA5Dataset] = []
        self._lengths: list[int] = []

        for start, end in date_ranges:
            ds = ERA5Dataset(
                data_dirs=data_dirs,
                start_date=start,
                end_date=end,
                step_hours=step_hours,
                include_extra_surf=include_extra_surf,
            )
            self.datasets.append(ds)
            self._lengths.append(len(ds))

        self._cumulative = []
        total = 0
        for length in self._lengths:
            total += length
            self._cumulative.append(total)

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, idx: int) -> tuple[Batch, Batch]:
        for ds_idx, cum_len in enumerate(self._cumulative):
            if idx < cum_len:
                offset = cum_len - self._lengths[ds_idx]
                return self.datasets[ds_idx][idx - offset]
        raise IndexError(f"index {idx} out of range for dataset of length {len(self)}")

    @property
    def triplets(self) -> list:
        """All triplets across sub-datasets (for inspection / debugging)."""
        out = []
        for ds in self.datasets:
            out.extend(ds.triplets)
        return out

    def close(self):
        for ds in self.datasets:
            ds.close()


def make_era5_splits(
    data_dirs: str | Path | list[str | Path],
    train_ranges: list[tuple[datetime, datetime]] | None = None,
    val_ranges: list[tuple[datetime, datetime]] | None = None,
    test_ranges: list[tuple[datetime, datetime]] | None = None,
    step_hours: int = 6,
    include_extra_surf: bool = True,
) -> tuple[MultiRangeERA5Dataset, MultiRangeERA5Dataset, MultiRangeERA5Dataset]:
    """Create train / val / test splits from (possibly non-contiguous) date ranges.

    Defaults to the soil-moisture summer split:
        Train : Jun+Jul 2024, Jun+Jul 2025
        Val   : Aug 1-15 2024, Aug 1-15 2025
        Test  : Aug 16-31 2024, Aug 16-31 2025
    """
    if train_ranges is None:
        train_ranges = DEFAULT_TRAIN_RANGES
    if val_ranges is None:
        val_ranges = DEFAULT_VAL_RANGES
    if test_ranges is None:
        test_ranges = DEFAULT_TEST_RANGES

    kwargs = dict(
        data_dirs=data_dirs,
        step_hours=step_hours,
        include_extra_surf=include_extra_surf,
    )
    train_ds = MultiRangeERA5Dataset(date_ranges=train_ranges, **kwargs)
    val_ds = MultiRangeERA5Dataset(date_ranges=val_ranges, **kwargs)
    test_ds = MultiRangeERA5Dataset(date_ranges=test_ranges, **kwargs)
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Random batch helpers (for testing without real data)
# ---------------------------------------------------------------------------

def make_random_batch(
    n_lat: int = 17,
    n_lon: int = 32,
    n_levels: int = 4,
    batch_size: int = 1,
    include_extra_surf: bool = True,
) -> Batch:
    """Create a random batch for testing.

    Uses small spatial dimensions by default for fast CPU testing.
    Set n_lat=721, n_lon=1440, n_levels=13 for full 0.25-degree resolution.
    """
    levels = PRESSURE_LEVELS[:n_levels]
    surf_keys = SURF_VAR_NAMES
    if include_extra_surf:
        surf_keys = surf_keys + tuple(EXTRA_SURF_ERA5_TO_AURORA.values())

    return Batch(
        surf_vars={k: torch.randn(batch_size, 2, n_lat, n_lon) for k in surf_keys},
        static_vars={k: torch.randn(n_lat, n_lon) for k in STATIC_VAR_NAMES},
        atmos_vars={
            k: torch.randn(batch_size, 2, n_levels, n_lat, n_lon) for k in ATMOS_VAR_NAMES
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, n_lat),
            lon=torch.linspace(0, 360, n_lon + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=levels,
        ),
    )


def make_random_batch_sequence(
    steps: int,
    n_lat: int = 17,
    n_lon: int = 32,
    n_levels: int = 4,
    batch_size: int = 1,
    start_time: datetime | None = None,
    step_hours: int = 6,
    include_extra_surf: bool = True,
) -> list[Batch]:
    """Create a sequence of random batches for testing multi-step fine-tuning.

    Each batch is offset by `step_hours` hours from the previous one.
    Returns `steps` batches, each usable as input to get the next prediction.
    """
    if start_time is None:
        start_time = datetime(2020, 6, 1, 0, 0)

    levels = PRESSURE_LEVELS[:n_levels]
    surf_keys = SURF_VAR_NAMES
    if include_extra_surf:
        surf_keys = surf_keys + tuple(EXTRA_SURF_ERA5_TO_AURORA.values())
    batches = []

    for i in range(steps):
        t = start_time + timedelta(hours=step_hours * i)
        batch = Batch(
            surf_vars={
                k: torch.randn(batch_size, 2, n_lat, n_lon) for k in surf_keys
            },
            static_vars={k: torch.randn(n_lat, n_lon) for k in STATIC_VAR_NAMES},
            atmos_vars={
                k: torch.randn(batch_size, 2, n_levels, n_lat, n_lon)
                for k in ATMOS_VAR_NAMES
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, n_lat),
                lon=torch.linspace(0, 360, n_lon + 1)[:-1],
                time=(t,),
                atmos_levels=levels,
            ),
        )
        batches.append(batch)

    return batches
