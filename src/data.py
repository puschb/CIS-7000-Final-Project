"""Utilities for constructing Aurora batches from various data sources."""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from aurora import Batch, Metadata

BASE_SURF_VAR_NAMES = ("2t", "10u", "10v", "msl")
EXTRA_SURF_VAR_NAMES = ("swvl1", "stl1", "sd")
SURF_VAR_NAMES = BASE_SURF_VAR_NAMES + EXTRA_SURF_VAR_NAMES
STATIC_VAR_NAMES = ("lsm", "z", "slt")
ATMOS_VAR_NAMES = ("z", "u", "v", "t", "q")
PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

FULL_RES_LAT = 721   # 0.25 degree: 90 to -90
FULL_RES_LON = 1440  # 0.25 degree: 0 to 359.75

DENSITY_VARS = ("swvl1", "stl1", "sd")

SOIL_SURF_VARS = (
    "2t", "10u", "10v", "msl",
    "swvl1", "swvl1_density",
    "stl1", "stl1_density",
    "sd", "sd_density",
)

# ERA5 NetCDF short names -> Aurora names
SURF_ERA5_TO_AURORA = {
    "t2m": "2t",
    "u10": "10u",
    "v10": "10v",
    "msl": "msl",
}

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
# Density channel helper
# ---------------------------------------------------------------------------

def _add_density_channels(surf_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create density channels for soil variables and replace NaN with 0.

    For each variable in DENSITY_VARS that exists in surf_dict, adds a
    companion ``{var}_density`` tensor (1 where data present, 0 where NaN)
    and replaces NaN values in the data with 0.
    """
    for var in DENSITY_VARS:
        if var in surf_dict:
            data = surf_dict[var]
            surf_dict[f"{var}_density"] = (~torch.isnan(data)).float()
            surf_dict[var] = data.nan_to_num(0.0)
    return surf_dict


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


# One NetCDF per valid_time (ISO-like stem), e.g. ``2024-06-16T09-surface.nc``.
_PER_TIMESTEP_SURF = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2})-surface\.nc$"
)
_PER_TIMESTEP_ATMOS = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2})-atmospheric\.nc$"
)


def _parse_per_timestep_surface_files(data_dir: Path) -> dict[datetime, Path]:
    """Map UTC hour ``datetime`` -> single-timestep surface NetCDF path."""
    out: dict[datetime, Path] = {}
    for f in sorted(data_dir.glob("*-surface.nc")):
        m = _PER_TIMESTEP_SURF.match(f.name)
        if not m:
            continue
        dt = datetime(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
            0,
            0,
        )
        out[dt] = f
    return out


def _parse_per_timestep_atmos_files(data_dir: Path) -> dict[datetime, Path]:
    """Map UTC hour ``datetime`` -> single-timestep atmospheric NetCDF path."""
    out: dict[datetime, Path] = {}
    for f in sorted(data_dir.glob("*-atmospheric.nc")):
        m = _PER_TIMESTEP_ATMOS.match(f.name)
        if not m:
            continue
        dt = datetime(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
            0,
            0,
        )
        out[dt] = f
    return out


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
# Multi-worker safety
# ---------------------------------------------------------------------------

def collate_era5_batch(
    samples: list[tuple[Batch, list[Batch]]],
) -> tuple[Batch, list[Batch]]:
    """Collate a list of (input_batch, targets) samples into a single batched pair.

    PyTorch's default collator doesn't know how to handle Aurora's Batch dataclass
    (it contains dicts of tensors and non-tensor metadata like datetime tuples).
    This function handles the collation manually:

    - surf_vars / atmos_vars: concatenated along the batch dim (dim 0)
    - static_vars: taken from the first sample (identical across samples)
    - metadata.time: tuples are concatenated → one datetime per sample
    - metadata.lat / lon / atmos_levels / rollout_step: taken from first sample
    """
    inputs = [s[0] for s in samples]
    target_lists = [s[1] for s in samples]

    def cat_batch(batches: list[Batch]) -> Batch:
        return Batch(
            surf_vars={
                k: torch.cat([b.surf_vars[k] for b in batches], dim=0)
                for k in batches[0].surf_vars
            },
            static_vars=batches[0].static_vars,
            atmos_vars={
                k: torch.cat([b.atmos_vars[k] for b in batches], dim=0)
                for k in batches[0].atmos_vars
            },
            metadata=Metadata(
                lat=batches[0].metadata.lat,
                lon=batches[0].metadata.lon,
                time=sum((b.metadata.time for b in batches), ()),
                atmos_levels=batches[0].metadata.atmos_levels,
                rollout_step=batches[0].metadata.rollout_step,
            ),
        )

    collated_input = cat_batch(inputs)

    n_targets = len(target_lists[0])
    collated_targets = [
        cat_batch([target_lists[i][step] for i in range(len(samples))])
        for step in range(n_targets)
    ]

    return collated_input, collated_targets


def era5_worker_init_fn(worker_id: int) -> None:
    """Clear xarray file handle caches after fork.

    Pass this to ``DataLoader(worker_init_fn=...)`` when using
    ``num_workers > 0``. Each forked worker needs its own file handles.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    targets = dataset.datasets if hasattr(dataset, "datasets") else [dataset]
    for ds in targets:
        ds._ds_cache.clear()


# ---------------------------------------------------------------------------
# ERA5Dataset
# ---------------------------------------------------------------------------

class ERA5Dataset(Dataset):
    """PyTorch Dataset that serves Aurora-compatible samples from ERA5 NetCDF
    files produced by ``scripts/download_era5.py``.

    Each sample is a sequence of timestamps whose length depends on
    ``rollout_steps``:

    - ``rollout_steps=1``: triplet ``(t-6h, t, t+6h)`` — 1 target
    - ``rollout_steps=2``: quadruplet ``(t-6h, t, t+6h, t+12h)`` — 2 targets

    ``__getitem__`` returns ``(input_batch, targets)`` where *targets* is a
    **list** of Aurora ``Batch`` objects (length == ``rollout_steps``).

    Args:
        data_dirs: One or more directories containing downloaded ERA5 data.
        start_date: Inclusive start of the date range (filters on ``t``).
        end_date: Exclusive end of the date range (filters on ``t``).
        step_hours: Time gap between consecutive steps (default 6h).
        include_extra_surf: Include soil/snow variables and density channels.
        rollout_steps: Number of target timesteps (1 or 2).
        file_layout: ``"chunked"`` — monthly surface + multi-day atmospheric
            NetCDFs (default). ``"per_timestep"`` — one small NetCDF per UTC
            hour, names ``YYYY-MM-DDTHH-{surface,atmospheric}.nc`` (see
            ``scripts/split_chunk_to_per_timestep.py``).
    """

    def __init__(
        self,
        data_dirs: str | Path | list[str | Path],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        step_hours: int = 6,
        include_extra_surf: bool = True,
        rollout_steps: int = 1,
        file_layout: Literal["chunked", "per_timestep"] = "chunked",
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.data_dirs = [Path(d) for d in data_dirs]
        self.step_hours = step_hours
        self.include_extra_surf = include_extra_surf
        self.rollout_steps = rollout_steps
        self.file_layout = file_layout

        self.surf_map = SURF_ERA5_TO_AURORA.copy()
        if include_extra_surf:
            self.surf_map.update(EXTRA_SURF_ERA5_TO_AURORA)

        self.surface_files: dict[tuple[int, int], Path] = {}
        self.atmos_chunks: list[tuple[datetime, datetime, Path]] = []
        self.surf_paths_by_time: dict[datetime, Path] = {}
        self.atmos_paths_by_time: dict[datetime, Path] = {}
        static_path = None

        for d in self.data_dirs:
            candidate = d / "static.nc"
            if candidate.exists() and static_path is None:
                static_path = candidate

            if file_layout == "chunked":
                self.surface_files.update(_parse_surface_files(d))
                self.atmos_chunks.extend(_parse_atmos_files(d))
            elif file_layout == "per_timestep":
                self.surf_paths_by_time.update(_parse_per_timestep_surface_files(d))
                self.atmos_paths_by_time.update(_parse_per_timestep_atmos_files(d))
            else:
                raise ValueError(f"Unknown file_layout: {file_layout!r}")

        self.atmos_chunks.sort(key=lambda x: x[0])

        if static_path is None:
            raise FileNotFoundError("No static.nc found in any data directory")

        if file_layout == "per_timestep":
            if not self.surf_paths_by_time or not self.atmos_paths_by_time:
                raise FileNotFoundError(
                    "per_timestep layout requires per-hour "
                    "*-surface.nc and *-atmospheric.nc files in data_dirs"
                )

        # Load static vars once (small, reused every sample)
        static_ds = xr.open_dataset(static_path, engine="netcdf4")
        self.static_vars = {
            STATIC_ERA5_TO_AURORA[k]: torch.from_numpy(static_ds[k].values[0]).float()
            for k in STATIC_ERA5_TO_AURORA
        }
        self.lat = torch.from_numpy(static_ds.latitude.values).float()
        self.lon = torch.from_numpy(static_ds.longitude.values).float()
        static_ds.close()

        self._ds_cache: dict[Path, xr.Dataset] = {}

        self.sequences = self._build_sequences(start_date, end_date)

    # Keep backward-compat alias
    @property
    def triplets(self) -> list[tuple[datetime, ...]]:
        return self.sequences

    def _build_sequences(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[tuple[datetime, ...]]:
        """Build valid timestamp sequences of length ``2 + rollout_steps``.

        ``rollout_steps=1`` → ``(t-6h, t, t+6h)``
        ``rollout_steps=2`` → ``(t-6h, t, t+6h, t+12h)``
        """
        if self.file_layout == "per_timestep":
            return self._build_sequences_per_timestep(start_date, end_date)

        step = timedelta(hours=self.step_hours)
        sequences: list[tuple[datetime, ...]] = []

        for (year, month), _surf_path in sorted(self.surface_files.items()):
            n_days = calendar.monthrange(year, month)[1]
            month_start = datetime(year, month, 1, 0, 0)
            month_end = datetime(year, month, n_days, 23, 0)

            t1 = month_start + step
            while True:
                t0 = t1 - step
                targets = [t1 + step * i for i in range(1, self.rollout_steps + 1)]
                last_ts = targets[-1]

                if last_ts > month_end:
                    break

                if start_date and t1 < start_date:
                    t1 += timedelta(hours=1)
                    continue
                if end_date and t1 >= end_date:
                    break

                timestamps = [t0, t1] + targets
                if any(ts.month != month for ts in timestamps):
                    t1 += timedelta(hours=1)
                    continue

                if all(
                    _find_atmos_file(ts, self.atmos_chunks) is not None
                    for ts in timestamps
                ):
                    sequences.append(tuple(timestamps))

                t1 += timedelta(hours=1)

        return sequences

    def _build_sequences_per_timestep(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[tuple[datetime, ...]]:
        """Sequence builder when each hour lives in its own NetCDF pair."""
        step = timedelta(hours=self.step_hours)
        avail = set(self.surf_paths_by_time) & set(self.atmos_paths_by_time)
        sequences: list[tuple[datetime, ...]] = []

        for t1 in sorted(avail):
            if start_date and t1 < start_date:
                continue
            if end_date and t1 >= end_date:
                continue

            t0 = t1 - step
            targets = [t1 + step * i for i in range(1, self.rollout_steps + 1)]
            timestamps = [t0, t1] + targets

            if not all(ts in avail for ts in timestamps):
                continue
            if any(ts.month != t1.month or ts.year != t1.year for ts in timestamps):
                continue

            sequences.append(tuple(timestamps))

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def _open_ds(self, path: Path) -> xr.Dataset:
        """Return a cached xarray Dataset handle, opening the file only once."""
        if path not in self._ds_cache:
            self._ds_cache[path] = xr.open_dataset(path, engine="netcdf4")
        return self._ds_cache[path]

    def _load_surface_raw(self, dt: datetime) -> dict[str, torch.Tensor]:
        """Load raw surface variables for a single timestamp (no density channels)."""
        if self.file_layout == "per_timestep":
            path = self.surf_paths_by_time[dt]
            ds = self._open_ds(path)
            sliced = ds.isel(valid_time=0)
        else:
            key = (dt.year, dt.month)
            path = self.surface_files[key]
            ds = self._open_ds(path)
            idx = _surface_time_index(dt)
            sliced = ds.isel(valid_time=idx)
        result = {}
        for era5_name, aurora_name in self.surf_map.items():
            result[aurora_name] = torch.from_numpy(sliced[era5_name].values).float()
        return result

    def _load_surface(self, dt: datetime) -> dict[str, torch.Tensor]:
        """Load surface variables with density channels applied."""
        surf = self._load_surface_raw(dt)
        if self.include_extra_surf:
            surf = _add_density_channels(surf)
        return surf

    def _load_atmos(self, dt: datetime) -> dict[str, torch.Tensor]:
        """Load atmospheric variables for a single timestamp."""
        if self.file_layout == "per_timestep":
            path = self.atmos_paths_by_time[dt]
            ds = self._open_ds(path)
            sliced = ds.isel(valid_time=0)
        else:
            info = _find_atmos_file(dt, self.atmos_chunks)
            assert info is not None, f"No atmospheric file covers {dt}"
            path, time_idx = info
            ds = self._open_ds(path)
            sliced = ds.isel(valid_time=time_idx)
        result = {}
        for era5_name, aurora_name in ATMOS_ERA5_TO_AURORA.items():
            result[aurora_name] = torch.from_numpy(sliced[era5_name].values).float()
        return result

    def load_timestep(
        self, dt: datetime
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Load all variables for a single timestamp.

        Returns ``(surf_dict, atmos_dict)`` with density channels applied.
        Useful for Stage 2 replay buffer ground-truth lookups.
        """
        return self._load_surface(dt), self._load_atmos(dt)

    def close(self):
        """Close all cached file handles."""
        for ds in self._ds_cache.values():
            ds.close()
        self._ds_cache.clear()

    def _make_target_batch(
        self,
        surf: dict[str, torch.Tensor],
        atmos: dict[str, torch.Tensor],
        time: datetime,
    ) -> Batch:
        return Batch(
            surf_vars={k: v[None, None] for k, v in surf.items()},
            static_vars=self.static_vars,
            atmos_vars={k: v[None, None] for k, v in atmos.items()},
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(time,),
                atmos_levels=PRESSURE_LEVELS,
            ),
        )

    def __getitem__(self, idx: int) -> tuple[Batch, list[Batch]]:
        timestamps = self.sequences[idx]
        t0, t1 = timestamps[0], timestamps[1]
        target_times = timestamps[2:]

        surf0 = self._load_surface(t0)
        surf1 = self._load_surface(t1)
        atmos0 = self._load_atmos(t0)
        atmos1 = self._load_atmos(t1)

        input_surf = {k: torch.stack([surf0[k], surf1[k]])[None] for k in surf0}
        input_atmos = {k: torch.stack([atmos0[k], atmos1[k]])[None] for k in atmos0}

        input_batch = Batch(
            surf_vars=input_surf,
            static_vars=self.static_vars,
            atmos_vars=input_atmos,
            metadata=Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(t1,),
                atmos_levels=PRESSURE_LEVELS,
            ),
        )

        targets: list[Batch] = []
        for tt in target_times:
            surf_t = self._load_surface(tt)
            atmos_t = self._load_atmos(tt)
            targets.append(self._make_target_batch(surf_t, atmos_t, tt))

        return input_batch, targets


# ---------------------------------------------------------------------------
# Train / val / test split helper
# ---------------------------------------------------------------------------

# Default split: train on June, val/test on July.
DEFAULT_TRAIN_RANGES: list[tuple[datetime, datetime]] = [
    (datetime(2024, 6, 1), datetime(2024, 7, 1)),
    (datetime(2025, 6, 1), datetime(2025, 7, 1)),
]
DEFAULT_VAL_RANGES: list[tuple[datetime, datetime]] = [
    (datetime(2024, 7, 1), datetime(2024, 7, 16)),
    (datetime(2025, 7, 1), datetime(2025, 7, 16)),
]
DEFAULT_TEST_RANGES: list[tuple[datetime, datetime]] = [
    (datetime(2024, 7, 16), datetime(2024, 8, 1)),
    (datetime(2025, 7, 16), datetime(2025, 8, 1)),
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
        rollout_steps: int = 1,
        file_layout: Literal["chunked", "per_timestep"] = "chunked",
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
                rollout_steps=rollout_steps,
                file_layout=file_layout,
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

    def __getitem__(self, idx: int) -> tuple[Batch, list[Batch]]:
        for ds_idx, cum_len in enumerate(self._cumulative):
            if idx < cum_len:
                offset = cum_len - self._lengths[ds_idx]
                return self.datasets[ds_idx][idx - offset]
        raise IndexError(f"index {idx} out of range for dataset of length {len(self)}")

    @property
    def sequences(self) -> list[tuple[datetime, ...]]:
        """All timestamp sequences across sub-datasets."""
        out: list[tuple[datetime, ...]] = []
        for ds in self.datasets:
            out.extend(ds.sequences)
        return out

    @property
    def triplets(self) -> list[tuple[datetime, ...]]:
        """Backward-compat alias for sequences."""
        return self.sequences

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
    rollout_steps: int = 1,
    file_layout: Literal["chunked", "per_timestep"] = "chunked",
) -> tuple[MultiRangeERA5Dataset, MultiRangeERA5Dataset, MultiRangeERA5Dataset]:
    """Create train / val / test splits from (possibly non-contiguous) date ranges.

    Defaults to the soil-moisture summer split:
        Train : June 2024, June 2025
        Val   : Jul 1-15 2024, Jul 1-15 2025
        Test  : Jul 16-31 2024, Jul 16-31 2025
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
        rollout_steps=rollout_steps,
        file_layout=file_layout,
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
    surf_keys: tuple[str, ...] = BASE_SURF_VAR_NAMES
    if include_extra_surf:
        surf_keys = SOIL_SURF_VARS

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

    Each batch is offset by ``step_hours`` hours from the previous one.
    Returns ``steps`` batches, each usable as input to get the next prediction.
    """
    if start_time is None:
        start_time = datetime(2020, 6, 1, 0, 0)

    levels = PRESSURE_LEVELS[:n_levels]
    surf_keys: tuple[str, ...] = BASE_SURF_VAR_NAMES
    if include_extra_surf:
        surf_keys = SOIL_SURF_VARS
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
