# Data Pipeline: Loading, Caching, and Validation

## Data Split

**Available data:** ERA5 0.25°, June–July 2024 and June–July 2025 (rechunked to 1×721×1440).

| Split | Date Range | Days | Approx. samples |
|---|---|---|---|
| Train | Jun 1 – Jul 1 2024 + Jun 1 – Jul 1 2025 | 60 | ~1,400 |
| Val | Jul 1 – Jul 16 2024 + Jul 1 – Jul 16 2025 | 30 | ~700 |
| Test | Jul 16 – Aug 1 2024 + Jul 16 – Aug 1 2025 | 30 | ~700 |

All ranges are `[start, end)`. A sample is a valid sequence of timestamps where all
timestamps fall within the same month and have atmospheric data coverage. The sequence
length depends on `rollout_steps` (see below).

---

## Configurable Rollout Steps

`ERA5Dataset` accepts a `rollout_steps` parameter that controls how many forward passes
the training loop will make per sample. This determines how many target timesteps are
loaded:

| `rollout_steps` | Timestamps loaded | Sequence | Targets | Use case |
|---|---|---|---|---|
| 1 | t−6h, t, t+6h | triplet | 1 (t+6h) | Single-step Stage 1 training |
| 2 | t−6h, t, t+6h, t+12h | quadruplet | 2 (t+6h, t+12h) | Two-step Stage 1 training |

**`__getitem__` returns `(input_batch, targets)`** where `targets` is a list of Aurora
`Batch` objects. `len(targets) == rollout_steps`.

With `rollout_steps=2`, the training loop:
1. Predicts t+6h from (t−6h, t), computes loss against targets[0]
2. Assembles new input (t, pred_t+6h), predicts t+12h, computes loss against targets[1]
3. Total loss = average of both MAEs, backprop through both forward passes

The sample count drops slightly with `rollout_steps=2` because the last 6 hours of each
month lose one valid sample (t+12h would fall outside the month). In practice the
difference is negligible (~1,390 vs ~1,400).

---

## CephFS Read Performance

Measured on the Nautilus `rook-cephfs` PVC with rechunked (1, 721, 1440) NetCDF data:

| Operation | Time | Notes |
|---|---|---|
| Load 1 surface variable, 1 timestep | ~1–3s | ~4 MB per variable |
| Load 1 atmospheric variable, 1 timestep | ~3s | ~54 MB (13 levels) |
| Load ALL variables, 1 timestep | ~24–36s | ~300 MB total; varies with cache hits |
| Load 1 sample, `rollout_steps=1` (3 timesteps) | **~72–108s** | ~900 MB |
| Load 1 sample, `rollout_steps=2` (4 timesteps) | **~96–144s** | ~1.2 GB |

**Key constraint:** CephFS throughput is ~8 MB/s effective per sequential read stream.
Parallel reads from separate processes achieve much better aggregate throughput because
each worker opens independent connections to the storage backend.

---

## Stage 1: Short-Lead-Time Fine-Tuning

### Data flow: `rollout_steps=1`

```
┌──────────────────────────────────────────────────────────────────┐
│ DataLoader (num_workers=8, prefetch_factor=1, batch_size=1)      │
│                                                                  │
│  Worker 0: [──── load sample A (~100s) ────][──── load E ────]…  │
│  Worker 1: [──── load sample B (~100s) ────][──── load F ────]…  │
│  …workers 2–7 staggered…                                        │
│                                                                  │
│  Shared memory (/dev/shm): up to 8 × ~900 MB ≈ 7 GB            │
└──────────────────────┬───────────────────────────────────────────┘
                       │ one sample ready every ~13s
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ GPU (L40 48 GB)                                                  │
│                                                                  │
│  [forward ~3s][backward ~5s][optim] ← ~8s per step               │
│                                                                  │
│  GPU utilization: ~100% (workers produce faster than GPU needs)  │
└──────────────────────────────────────────────────────────────────┘
```

### Data flow: `rollout_steps=2`

```
┌──────────────────────────────────────────────────────────────────┐
│ DataLoader (num_workers=10, prefetch_factor=1, batch_size=1)     │
│                                                                  │
│  Worker 0: [──── load sample A (~120s) ────][──── load E ────]…  │
│  Worker 1: [──── load sample B (~120s) ────][──── load F ────]…  │
│  …workers 2–9 staggered…                                        │
│                                                                  │
│  Shared memory (/dev/shm): up to 10 × ~1.2 GB ≈ 12 GB          │
└──────────────────────┬───────────────────────────────────────────┘
                       │ one sample ready every ~12s
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ GPU (L40 48 GB)                                                  │
│                                                                  │
│  [fwd1 ~3s][fwd2 ~3s][backward ~8s][optim] ← ~15s per step      │
│                                                                  │
│  GPU utilization: ~100% (workers produce faster than GPU needs)  │
└──────────────────────────────────────────────────────────────────┘
```

### Worker count rationale

| Config | Load time | GPU step time | Workers needed | Workers used |
|---|---|---|---|---|
| `rollout_steps=1` | ~100s (3 timesteps) | ~8s (1 fwd + bwd) | ceil(100/8) = 13 | 8 (GPU is bottleneck, workers have surplus) |
| `rollout_steps=2` | ~120s (4 timesteps) | ~15s (2 fwd + bwd) | ceil(120/15) = 8 | 10 (safety margin) |

For `rollout_steps=1`, 8 workers produce a sample every ~12.5s while the GPU only
consumes one every ~8s — workers are comfortably ahead. No need for more.

For `rollout_steps=2`, the math is tighter (120s / 10 workers = 12s per sample, GPU needs
one every 15s), so 10 workers provides a comfortable margin.

### What each worker returns

`ERA5Dataset.__getitem__(idx)` returns `(input_batch, targets)`:

- `input_batch`: Aurora `Batch` with surf_vars containing the (t−6h, t) history for
  all surface variables (including density channels), atmos_vars with the same history,
  static_vars, and metadata.
- `targets`: list of Aurora `Batch` objects, one per rollout step.
  - `rollout_steps=1`: `[batch_at_t6h]` — ~900 MB total
  - `rollout_steps=2`: `[batch_at_t6h, batch_at_t12h]` — ~1.2 GB total

### Density channels

Created on the fly in `__getitem__` for the three soil variables (swvl1, stl1, sd):

```python
for var in ("swvl1", "stl1", "sd"):
    data = surf_vars[var]
    density = (~torch.isnan(data)).float()
    surf_vars[f"{var}_density"] = density
    surf_vars[var] = data.nan_to_num(0.0)
```

This happens in the worker process before the sample enters shared memory.

### Multi-worker safety: xarray file handles

**Problem:** The current `ERA5Dataset` caches xarray Dataset handles in `self._ds_cache`.
When `num_workers > 0`, PyTorch forks the Dataset object into each worker process. File
handles from the parent process become invalid in the child.

**Fix:** Use a `worker_init_fn` that clears the cache when each worker starts:

```python
def era5_worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    dataset = info.dataset
    targets = dataset.datasets if hasattr(dataset, 'datasets') else [dataset]
    for ds in targets:
        ds._ds_cache.clear()
```

Each worker then lazily opens its own file handles on first access.

### DataLoader configuration

```python
from torch.utils.data import DataLoader

# rollout_steps=1
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=8,
    prefetch_factor=1,
    persistent_workers=True,
    worker_init_fn=era5_worker_init_fn,
    pin_memory=True,
)

# rollout_steps=2
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=10,
    prefetch_factor=1,
    persistent_workers=True,
    worker_init_fn=era5_worker_init_fn,
    pin_memory=True,
)
```

- `persistent_workers=True`: Workers stay alive between epochs. Avoids re-forking (slow)
  and keeps xarray file handles warm in each worker's cache.
- `pin_memory=True`: Copies tensors to pinned (page-locked) CPU memory, speeding up the
  CPU→GPU transfer.
- `prefetch_factor=1`: Each worker loads 1 sample ahead.

---

## Stage 1: Validation

### Strategy

Pre-load a fixed subset of 30 validation samples into CPU RAM at startup. This avoids
CephFS reads during validation entirely.

The val samples must match the training `rollout_steps` so we can compute loss the same
way:
- `rollout_steps=1`: 30 × ~900 MB = **~27 GB**
- `rollout_steps=2`: 30 × ~1.2 GB = **~36 GB**

**Validation time:** 30 samples × ~5s (1-step) or ~10s (2-step) = **2.5–5 minutes**

### Implementation

```python
def preload_val_cache(val_dataset, n_samples=30, seed=42):
    """Load n_samples from val set into a list of (input, targets) in CPU RAM."""
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(val_dataset), generator=rng)[:n_samples].tolist()
    cache = []
    for i, idx in enumerate(indices):
        input_batch, targets = val_dataset[idx]
        cache.append((input_batch, targets))
        print(f"  Pre-loaded val sample {i+1}/{n_samples}", flush=True)
    return cache
```

Pre-loading 30 samples takes ~18 min (1-step) or ~24 min (2-step). This happens once at
training startup. Use the same fixed seed every run for reproducibility.

### Validation loop

Run every 500 training steps. With 3,000 total steps, that's 6 validation passes.

```python
def validate_stage1(model, val_cache, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_batch, targets in val_cache:
            current = input_batch.to(device)
            step_losses = []
            for target in targets:
                pred = model(current)
                target_gpu = target.to(device)
                step_losses.append(weighted_mae(pred, target_gpu))
                if len(targets) > 1:
                    current = assemble_input(current, pred)
            total_loss += torch.stack(step_losses).mean().item()
    model.train()
    return total_loss / len(val_cache)
```

This handles both 1-step and 2-step: when `targets` has 1 element it's a simple
single-step eval; with 2 elements it runs the 2-step rollout.

---

## Stage 2: Rollout Fine-Tuning

### Data flow

Stage 2 does not use a standard DataLoader pipeline. Instead it has:

1. **Replay buffer** in CPU RAM — 30 entries, each containing an input pair and its
   next-step ground truth
2. **Dataset** for fetching fresh initial conditions and ground truth — accessed
   infrequently (every 10 steps)

```
┌───────────────────────────────────────────────────────────┐
│ Replay Buffer (CPU RAM, 30 entries)                       │
│                                                           │
│  Each entry:                                              │
│    input:        (t−6h, t) pair         ~600 MB           │
│    ground_truth: ERA5 at t+6h           ~300 MB           │
│    lead_time:    int (0 = real data, 12 = 3 days out)     │
│    timestamp:    datetime of current "t"                  │
│                                                           │
│  Total: 30 × 900 MB = ~27 GB                             │
└────────────────────┬──────────────────────────────────────┘
                     │ sample one entry
                     ▼
┌───────────────────────────────────────────────────────────┐
│ GPU (L40 48 GB)                                           │
│                                                           │
│  1. Move input to GPU                                     │
│  2. Forward pass (with gradients, LoRA only)              │
│  3. Compute MAE loss vs ground truth (also moved to GPU)  │
│  4. Backward + optimizer step                             │
│  5. Detach prediction, assemble new input pair            │
│  6. Move back to CPU, store in buffer                     │
│                                                           │
│  ~5–8s per step (LoRA backward is cheaper than full)      │
└───────────────────────────────────────────────────────────┘
                     │
                     ▼ after step, update buffer entry
┌───────────────────────────────────────────────────────────┐
│ Background thread: fetch next ground truth from disk      │
│                                                           │
│  The entry advanced from timestamp T to T+6h.             │
│  Background thread reads ERA5 at T+12h (~36s) so it's    │
│  ready for the next time this entry is sampled.           │
│                                                           │
│  Since training steps take ~5–8s and buffer has 30        │
│  entries, each entry is sampled roughly every 30 steps    │
│  = ~150–240s. The 36s background fetch easily fits.       │
└───────────────────────────────────────────────────────────┘
```

### Buffer lifecycle

**Initialization (before training):**

1. Sample 30 triplets from the training dataset (using `rollout_steps=1` mode)
2. For each, load the (t−6h, t) input pair and the t+6h ground truth
3. Store in buffer with `lead_time = 0`
4. Loading time: 30 × 100s ≈ **50 minutes** (sequential, one-time cost)

Could parallelize initialization with a temporary DataLoader (8 workers → ~7 min).

**Each training step:**

1. Sample a random entry from the buffer
2. Move input + ground truth to GPU
3. Forward pass → prediction at t+6h
4. Loss = weighted MAE(prediction, ground truth)
5. Backward + optimizer step (only LoRA params get gradients)
6. Detach prediction, assemble new input: `(old_t, detached_pred)`
7. `lead_time += 1`
8. If `lead_time > max_lead_time` (12 for 3-day max): discard entry, mark for refresh
9. Move new input back to CPU buffer
10. **Background thread**: read the *next* ground truth (ERA5 at the new predicted
    timestamp + 6h) from disk for this entry

**Every 10 steps (`dataset_sampling_period`):**

Replace one buffer entry with a fresh initial condition from the training dataset.
This keeps the buffer diverse and prevents it from being entirely high-lead-time entries.

### Lead time schedule

Simplified from the paper since our max is 3 days:

| Training steps | Max lead time | Rationale |
|---|---|---|
| 0 – 1,000 | 1.5 days (6 steps) | Learn short-range stability first |
| 1,000 – 3,000 | 3 days (12 steps) | Extend to full target range |

### Ground truth in the buffer

Each buffer entry stores its next-step ground truth alongside the input pair. When the
entry advances (after a training step), the new ground truth needs to be fetched from
disk. This is done in a background thread to avoid blocking the training loop.

**Why not fetch ground truth on demand?** A CephFS read takes ~36s. The training step
takes ~5–8s. If we read ground truth synchronously, the GPU would idle for ~36s every
step — worse than Stage 1 without workers.

**Why not cache ALL ground truth?** The training set has ~1,400 unique timesteps × 300 MB
= ~420 GB. Far too much for RAM.

**The background fetch works because:**
- Each buffer entry is sampled roughly every 30 steps (30 entries, uniform random)
- 30 steps × 5–8s = 150–240s between samplings of the same entry
- The background fetch takes ~36s — comfortably finishes before the entry is needed again
- Edge case: if an entry is sampled twice in quick succession before the fetch completes,
  fall back to a synchronous read (rare, acceptable)

### What if ground truth doesn't exist?

At high lead times, the predicted timestamp may fall outside the training date range
(e.g., an entry initialized at Jun 29 that has been rolled forward past Jul 1). In
this case, discard the entry and replace it with a fresh initial condition.

---

## Stage 2: Validation (Rollout)

### Strategy

Run 3 complete autoregressive rollouts from fixed initial conditions in the validation
set, each out to 3 days (12 steps). Pre-cache all required ground truth.

**Ground truth to cache:**
- 3 initial conditions × 12 rollout steps × ~300 MB = **~10.8 GB CPU RAM**
- Initial conditions themselves: 3 × 600 MB = ~1.8 GB
- **Total val cache: ~12.6 GB**

**Validation time:** 3 rollouts × 12 forward passes × ~5s = **~3 minutes** (pure GPU)

### Implementation

```python
def preload_rollout_val_cache(dataset, n_rollouts=3, max_steps=12, seed=42):
    """Pre-load initial conditions and ground truth for rollout validation.

    Returns a list of (initial_input, [gt_step1, gt_step2, ..., gt_step12]).
    """
    rng = torch.Generator().manual_seed(seed)
    # Pick initial conditions from early in val set (need 3 days of headroom)
    valid_starts = [
        i for i, (t0, t1, t2) in enumerate(dataset.triplets)
        if t1 + timedelta(hours=6 * max_steps) <= val_end_date
    ]
    chosen = torch.randperm(len(valid_starts), generator=rng)[:n_rollouts]

    cache = []
    for idx in chosen:
        t0, t1, t2 = dataset.triplets[valid_starts[idx]]
        input_batch, _ = dataset[valid_starts[idx]]

        ground_truths = []
        for step in range(1, max_steps + 1):
            target_time = t1 + timedelta(hours=6 * step)
            gt = dataset.load_timestep(target_time)
            ground_truths.append(gt)

        cache.append((input_batch, ground_truths))
    return cache
```

### Rollout validation loop

Run every 500 training steps. Report MAE per variable at each lead time.

```python
def validate_rollout(model, val_cache, device):
    model.eval()
    # results[lead_time][var_name] = list of MAE values
    results = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for initial_input, ground_truths in val_cache:
            current = initial_input.to(device)
            for step, gt in enumerate(ground_truths):
                pred = model(current)
                gt_gpu = gt.to(device)
                for var in pred.surf_vars:
                    mae = compute_mae(pred.surf_vars[var], gt_gpu.surf_vars[var])
                    results[step + 1][var].append(mae.item())
                current = assemble_input(current, pred)

    model.train()
    return {
        step: {var: np.mean(vals) for var, vals in var_dict.items()}
        for step, var_dict in results.items()
    }
```

---

## Memory Budget Summary

### Stage 1 with `rollout_steps=1`

| Component | Memory | Type |
|---|---|---|
| DataLoader workers (8 × ~900 MB) | ~7 GB | Shared memory (`/dev/shm`) |
| Worker process overhead | ~2 GB | CPU RAM |
| Val cache (30 × ~900 MB) | ~27 GB | CPU RAM |
| Python + PyTorch runtime | ~3 GB | CPU RAM |
| **Total CPU** | **~32 GB RAM + 8 GB shm** | |
| Model + optimizer + activations (1 fwd) | ~10–15 GB | GPU |
| **Total GPU** | **~10–15 GB of 48 GB** | |

### Stage 1 with `rollout_steps=2`

| Component | Memory | Type |
|---|---|---|
| DataLoader workers (10 × ~1.2 GB) | ~12 GB | Shared memory (`/dev/shm`) |
| Worker process overhead | ~2 GB | CPU RAM |
| Val cache (30 × ~1.2 GB) | ~36 GB | CPU RAM |
| Python + PyTorch runtime | ~3 GB | CPU RAM |
| **Total CPU** | **~41 GB RAM + 12 GB shm** | |
| Model + optimizer + activations (2 fwd) | ~18–33 GB | GPU |
| **Total GPU** | **~18–33 GB of 48 GB** | |

### Stage 2

| Component | Memory | Type |
|---|---|---|
| Replay buffer (30 entries × 900 MB) | ~27 GB | CPU RAM |
| Rollout val cache | ~12.6 GB | CPU RAM |
| Python + PyTorch runtime | ~3 GB | CPU RAM |
| Background fetch thread overhead | negligible | CPU RAM |
| **Total CPU** | **~43 GB RAM + 2 GB shm** | |
| Model + LoRA optimizer + activations | ~10–15 GB | GPU |
| **Total GPU** | **~10–15 GB of 48 GB** | |

---

## Changes Needed in `src/data.py`

### 1. Update default date ranges

Current ranges point to Jun–Aug with Aug splits. Update to Jun–Jul:

```python
DEFAULT_TRAIN_RANGES = [
    (datetime(2024, 6, 1), datetime(2024, 7, 1)),
    (datetime(2025, 6, 1), datetime(2025, 7, 1)),
]
DEFAULT_VAL_RANGES = [
    (datetime(2024, 7, 1), datetime(2024, 7, 16)),
    (datetime(2025, 7, 1), datetime(2025, 7, 16)),
]
DEFAULT_TEST_RANGES = [
    (datetime(2024, 7, 16), datetime(2024, 8, 1)),
    (datetime(2025, 7, 16), datetime(2025, 8, 1)),
]
```

### 2. Add `rollout_steps` parameter to `ERA5Dataset`

Replace `_build_triplets` with a generalized `_build_sequences`:

```python
class ERA5Dataset(Dataset):
    def __init__(self, ..., rollout_steps: int = 1):
        self.rollout_steps = rollout_steps
        ...
        self.sequences = self._build_sequences(start_date, end_date)

    def _build_sequences(self, start_date, end_date):
        """Build valid timestamp sequences of length (2 + rollout_steps).

        rollout_steps=1 → (t-6h, t, t+6h)           — 3 timestamps
        rollout_steps=2 → (t-6h, t, t+6h, t+12h)    — 4 timestamps
        """
        step = timedelta(hours=self.step_hours)
        n_timestamps = 2 + self.rollout_steps  # input pair + targets
        sequences = []

        for (year, month), surf_path in sorted(self.surface_files.items()):
            n_days = calendar.monthrange(year, month)[1]
            month_start = datetime(year, month, 1, 0, 0)
            month_end = datetime(year, month, n_days, 23, 0)

            t1 = month_start + step
            while True:
                t0 = t1 - step
                timestamps = [t0, t1] + [t1 + step * i for i in range(1, self.rollout_steps + 1)]
                last_ts = timestamps[-1]

                if last_ts > month_end:
                    break

                if start_date and t1 < start_date:
                    t1 += timedelta(hours=1)
                    continue
                if end_date and t1 >= end_date:
                    break

                # All timestamps must be in the same month
                if any(ts.month != month for ts in timestamps):
                    t1 += timedelta(hours=1)
                    continue

                # All timestamps must have atmospheric coverage
                if all(_find_atmos_file(ts, self.atmos_chunks) is not None for ts in timestamps):
                    sequences.append(tuple(timestamps))

                t1 += timedelta(hours=1)

        return sequences
```

### 3. Update `__getitem__` to return `(input, targets)`

```python
def __getitem__(self, idx):
    timestamps = self.sequences[idx]
    t0, t1 = timestamps[0], timestamps[1]
    target_times = timestamps[2:]  # 1 or 2 targets depending on rollout_steps

    surf0 = _add_density_channels(self._load_surface(t0))
    surf1 = _add_density_channels(self._load_surface(t1))
    atmos0, atmos1 = self._load_atmos(t0), self._load_atmos(t1)

    input_surf = {k: torch.stack([surf0[k], surf1[k]])[None] for k in surf0}
    input_atmos = {k: torch.stack([atmos0[k], atmos1[k]])[None] for k in atmos0}

    input_batch = Batch(
        surf_vars=input_surf,
        static_vars=self.static_vars,
        atmos_vars=input_atmos,
        metadata=Metadata(lat=self.lat, lon=self.lon, time=(t1,), atmos_levels=PRESSURE_LEVELS),
    )

    targets = []
    for tt in target_times:
        surf_t = _add_density_channels(self._load_surface(tt))
        atmos_t = self._load_atmos(tt)
        target_batch = Batch(
            surf_vars={k: v[None, None] for k, v in surf_t.items()},
            static_vars=self.static_vars,
            atmos_vars={k: v[None, None] for k, v in atmos_t.items()},
            metadata=Metadata(lat=self.lat, lon=self.lon, time=(tt,), atmos_levels=PRESSURE_LEVELS),
        )
        targets.append(target_batch)

    return input_batch, targets
```

### 4. Add density channel helper

```python
DENSITY_VARS = ("swvl1", "stl1", "sd")

def _add_density_channels(surf_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for var in DENSITY_VARS:
        if var in surf_dict:
            data = surf_dict[var]
            surf_dict[f"{var}_density"] = (~torch.isnan(data)).float()
            surf_dict[var] = data.nan_to_num(0.0)
    return surf_dict
```

### 5. Add single-timestep loader

For Stage 2's replay buffer, we need to load ground truth for arbitrary timestamps
without constructing a full sequence:

```python
def load_timestep(self, dt: datetime) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load all variables for a single timestamp.

    Returns (surf_dict, atmos_dict) with density channels applied.
    """
    surf = _add_density_channels(self._load_surface(dt))
    atmos = self._load_atmos(dt)
    return surf, atmos
```

### 6. Fix multi-worker file handle safety

Add a `worker_init_fn` and clear the xarray cache on fork:

```python
def era5_worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    dataset = info.dataset
    targets = dataset.datasets if hasattr(dataset, 'datasets') else [dataset]
    for ds in targets:
        ds._ds_cache.clear()
```

### 7. Add `SOIL_SURF_VARS` constant

For the model configuration to reference:

```python
SOIL_SURF_VARS = (
    "2t", "10u", "10v", "msl",
    "swvl1", "swvl1_density",
    "stl1", "stl1_density",
    "sd", "sd_density",
)
```

### 8. Propagate `rollout_steps` through `MultiRangeERA5Dataset` and `make_era5_splits`

Both wrappers need to accept and forward the new parameter.

---

## K8s Resource Requests

### Stage 1 Job (`rollout_steps=1`)

```yaml
resources:
  requests:
    memory: 48Gi
    cpu: "10"
    nvidia.com/gpu: "1"
  limits:
    memory: 48Gi
    cpu: "10"
    nvidia.com/gpu: "1"
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 8Gi
```

### Stage 1 Job (`rollout_steps=2`)

```yaml
resources:
  requests:
    memory: 64Gi
    cpu: "12"
    nvidia.com/gpu: "1"
  limits:
    memory: 64Gi
    cpu: "12"
    nvidia.com/gpu: "1"
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 12Gi
```

### Stage 2 Job

```yaml
resources:
  requests:
    memory: 48Gi
    cpu: "6"
    nvidia.com/gpu: "1"
  limits:
    memory: 48Gi
    cpu: "6"
    nvidia.com/gpu: "1"
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 2Gi
```

All jobs target NVIDIA L40 (48 GB VRAM) via node affinity.

---

## Timing Estimates

### Stage 1 with `rollout_steps=1` (3,000 steps)

| Phase | Time |
|---|---|
| Pre-load 30 val samples | ~18 min |
| Training (3,000 steps × ~8s) | ~6.7 hours |
| Validation (6 passes × 2.5 min) | ~15 min |
| **Total** | **~7.5 hours** |

### Stage 1 with `rollout_steps=2` (3,000 steps)

| Phase | Time |
|---|---|
| Pre-load 30 val samples | ~24 min |
| Training (3,000 steps × ~15s) | ~12.5 hours |
| Validation (6 passes × 5 min) | ~30 min |
| **Total** | **~13.5 hours** |

### Stage 2 (3,000 steps)

| Phase | Time |
|---|---|
| Initialize buffer (30 samples) | ~50 min (sequential) or ~7 min (8 workers) |
| Pre-load rollout val cache | ~15 min |
| Training (3,000 steps × ~8s) | ~6.7 hours |
| Validation (6 passes × 3 min) | ~18 min |
| **Total** | **~8 hours** |

### Combined

| Configuration | Stage 1 | Stage 2 | Total |
|---|---|---|---|
| `rollout_steps=1` | ~7.5 hours | ~8 hours | **~15.5 hours** |
| `rollout_steps=2` | ~13.5 hours | ~8 hours | **~21.5 hours** |

Budget ~2 days of GPU time to account for restarts and potential tuning reruns.
