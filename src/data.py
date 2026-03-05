"""Utilities for constructing Aurora batches from various data sources."""

from datetime import datetime, timedelta

import torch

from aurora import Batch, Metadata

SURF_VAR_NAMES = ("2t", "10u", "10v", "msl")
STATIC_VAR_NAMES = ("lsm", "z", "slt")
ATMOS_VAR_NAMES = ("z", "u", "v", "t", "q")
PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

FULL_RES_LAT = 721   # 0.25 degree: 90 to -90
FULL_RES_LON = 1440  # 0.25 degree: 0 to 359.75


def make_random_batch(
    n_lat: int = 17,
    n_lon: int = 32,
    n_levels: int = 4,
    batch_size: int = 1,
) -> Batch:
    """Create a random batch for testing.

    Uses small spatial dimensions by default for fast CPU testing.
    Set n_lat=721, n_lon=1440, n_levels=13 for full 0.25-degree resolution.
    """
    levels = PRESSURE_LEVELS[:n_levels]

    return Batch(
        surf_vars={k: torch.randn(batch_size, 2, n_lat, n_lon) for k in SURF_VAR_NAMES},
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
) -> list[Batch]:
    """Create a sequence of random batches for testing multi-step fine-tuning.

    Each batch is offset by `step_hours` hours from the previous one.
    Returns `steps` batches, each usable as input to get the next prediction.
    """
    if start_time is None:
        start_time = datetime(2020, 6, 1, 0, 0)

    levels = PRESSURE_LEVELS[:n_levels]
    batches = []

    for i in range(steps):
        t = start_time + timedelta(hours=step_hours * i)
        batch = Batch(
            surf_vars={
                k: torch.randn(batch_size, 2, n_lat, n_lon) for k in SURF_VAR_NAMES
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
