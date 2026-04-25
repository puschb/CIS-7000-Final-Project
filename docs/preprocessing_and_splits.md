# ERA5 Pre-Processing & Train/Val/Test Splits

## Data Pre-Processing for New Variables

### What Aurora does for ERA5 (from the paper)

Aurora applies **one** pre-processing step to ERA5 data: per-variable normalisation (Supplementary B.5).

```
X_normalized = (X - centre) / scale
```

- **centre** = empirical mean over the ERA5 training data
- **scale** = empirical standard deviation over the ERA5 training data
- These are computed once from ERA5 and reused for all datasets.
- Normalisation happens **inside the model** — batches are fed with raw (unnormalised) values.

No other transforms are applied to standard meteorological ERA5 variables. The log-linear transforms, differencing, and density channels described in the paper are only for air pollution (CAMS) and wave (HRES-WAM) fine-tuning tasks.

### What we need to do for the new variables

Our three new surface variables (`swvl1`, `stl1`, `sd`) are standard physical quantities, not concentrations or angles, so they only need the standard normalisation:

| Variable | Pre-processing needed | Notes |
|---|---|---|
| `swvl1` (soil moisture) | Mean + std from training data | Bounded ~[0, 0.5] m³/m³. Standard normalisation is fine. |
| `stl1` (soil temperature) | Mean + std from training data | Behaves like 2m temperature. Straightforward. |
| `sd` (snow depth) | Mean + std from training data | Skewed (zero in warm regions), but no log transform needed. |

**Not needed:**
- Log or log-linear transform (only for concentration variables)
- sin/cos angle transform (only for wave directions)
- Density channels (ERA5 fills land-surface vars over ocean with constants, not NaN)
- Extra static variables or differencing (only for air pollution diurnal cycles)

### How to compute normalisation statistics

Stats must be computed **only on the training split** to avoid data leakage.

```bash
# On the Nautilus cluster (push code first):
kubectl apply -f k8s/compute-norm-stats-job.yaml
kubectl logs -f job/compute-norm-stats
```

This runs `scripts/compute_norm_stats.py`, which:
1. Scans all `*-surface.nc` files in the data directories
2. Filters to only timestamps within the training date ranges
3. Computes global mean and std using Welford's online algorithm (memory-efficient)
4. Prints output ready to paste into `src/finetune.py`

After the job completes, copy the printed values into the `_NEW_VAR_NORM` dict in `src/finetune.py`:

```python
_NEW_VAR_NORM: dict[str, tuple[float, float]] = {
    "swvl1": (<mean>, <std>),  # paste real values here
    "stl1": (<mean>, <std>),
    "sd": (<mean>, <std>),
}
```

---

## Train / Val / Test Split

### Data available

Summer ERA5 data (Jun–Aug) across two years, downloaded onto the `era5-summer-soil-moisture` PVC:

| Year | Months | PVC path |
|---|---|---|
| 2024 | Jun, Jul, Aug | `/mnt/data/era5/2024/` |
| 2025 | Jun, Jul, Aug | `/mnt/data/era5/2025/` |

Each month contains:
- `YYYY-MM-surface.nc` — all 7 surface variables (4 standard + 3 new) × 24 hourly timesteps × all days
- `YYYY-MM-dDD-DD-atmospheric.nc` — 5 atmospheric variables × 13 pressure levels × 24h × 3-day chunks
- `static.nc` — geopotential, land-sea mask, soil type (downloaded once per year directory)

### Split definition

| Split | Date range | Days | % |
|---|---|---|---|
| **Train** | Jun 1 – Aug 1, 2024 + Jun 1 – Aug 1, 2025 | ~122 | 66% |
| **Val** | Aug 1 – Aug 16, 2024 + Aug 1 – Aug 16, 2025 | ~30 | 17% |
| **Test** | Aug 16 – Sep 1, 2024 + Aug 16 – Sep 1, 2025 | ~32 | 17% |

All ranges are `[start, end)` — the end date is exclusive.

### Rationale

- **Training on both years** exposes the model to inter-annual variability in soil moisture patterns (e.g., different drought/wet conditions in 2024 vs 2025).
- **Val and test draw from both years**, so any performance difference reflects generalization quality, not inter-annual weather variability.
- **Val and test are both August** (same seasonal regime as late training), so differences reflect temporal generalization, not seasonal mismatch.
- The download covers Jun–Aug only (no September data exists), so `datetime(2024, 9, 1)` as the exclusive end captures all of August.

### Usage in code

The split is defined as the default in `src/data.py`:

```python
from src.data import make_era5_splits

train_ds, val_ds, test_ds = make_era5_splits(
    data_dirs=["/mnt/data/era5/2024", "/mnt/data/era5/2025"],
)

print(f"Train: {len(train_ds)} samples")
print(f"Val:   {len(val_ds)} samples")
print(f"Test:  {len(test_ds)} samples")
```

Custom splits can be passed explicitly:

```python
from datetime import datetime
from src.data import make_era5_splits

train_ds, val_ds, test_ds = make_era5_splits(
    data_dirs=["/mnt/data/era5/2024", "/mnt/data/era5/2025"],
    train_ranges=[
        (datetime(2024, 6, 1), datetime(2024, 8, 1)),
        (datetime(2025, 6, 1), datetime(2025, 8, 1)),
    ],
    val_ranges=[
        (datetime(2024, 8, 1), datetime(2024, 8, 16)),
        (datetime(2025, 8, 1), datetime(2025, 8, 16)),
    ],
    test_ranges=[
        (datetime(2024, 8, 16), datetime(2024, 9, 1)),
        (datetime(2025, 8, 16), datetime(2025, 9, 1)),
    ],
)
```

### Sample counts (estimated)

With 6-hour step size and all 24 hourly timesteps per day:

| Split | Days | Hourly timestamps | Valid 6h triplets (approx) |
|---|---|---|---|
| Train | ~122 (61 × 2 yr) | ~2,928 | ~2,800 |
| Val | ~30 (15 × 2 yr) | ~720 | ~690 |
| Test | ~32 (16 × 2 yr) | ~768 | ~740 |

---

## Model Configuration for New Variables

When fine-tuning, Aurora must be told about the new variables and given their normalisation stats:

```python
from aurora import AuroraPretrained
from aurora.normalisation import locations, scales

# 1. Register normalisation stats BEFORE creating the model
locations["swvl1"] = <mean>
scales["swvl1"] = <std>
locations["stl1"] = <mean>
scales["stl1"] = <std>
locations["sd"] = <mean>
scales["sd"] = <std>

# 2. Create model with extended surface variables
model = AuroraPretrained(
    autocast=True,
    surf_vars=("2t", "10u", "10v", "msl", "swvl1", "stl1", "sd"),
)

# 3. Load pretrained weights (strict=False because new vars have no pretrained weights)
model.load_checkpoint(strict=False)
```

Per the Aurora docs, use a **higher learning rate for new patch embeddings** (e.g., `1e-3`) and a lower rate for the rest of the network (e.g., `3e-4`). This is already configured in `src/finetune.py`.

---

## Checklist

- [ ] Download job completes for all 6 months (Jun–Aug 2024 + Jun–Aug 2025)
- [ ] Run `compute-norm-stats` job on the cluster
- [ ] Paste normalisation values into `src/finetune.py`
- [ ] Run fine-tuning
