# Aurora Fine-Tuning: What We Actually Ran

This document captures the two-stage fine-tuning runs that produced our usable
checkpoints. For the architectural / hyperparameter rationale, see
[`finetuning_plan.md`](finetuning_plan.md). This page is a reference for *which
artifacts exist on the PVC, and what config produced them*.

The base model in both stages is `AuroraSmallPretrained` (~113 M params,
`embed_dim=256`). The fine-tuned task is forecasting three new ERA5 surface
variables that are not in Aurora's pretraining set:

| ERA5 short name | Full name | Description |
|---|---|---|
| `swvl1` | Volumetric Soil Water Layer 1 | Top 7 cm soil moisture, m³/m³ |
| `stl1`  | Soil Temperature Level 1 | Top soil-layer temperature, K |
| `sd`    | Snow Depth | Snow water equivalent, m |

All artifacts live on the `era5-rechunked` PVC under `/mnt/data/runs/`. Code
lives at `src/finetune_stage1.py` and `src/finetune_stage2.py`.

---

## Stage 1 — short-lead-time fine-tuning

Trains the **whole model** (no LoRA) on single-step (6 h) predictions, using
the weighted MAE loss from Aurora paper §D.1. New-variable patch embeddings
start at zero (paper §B.8); existing-variable weights start from the published
Aurora pretrained checkpoint.

### Successful run

```
Run name:    stage1_20260505_053645
Run dir:     /mnt/data/runs/stage1_20260505_053645/
Wall time:   ~7.5 h on 1 × NVIDIA L40 (46 GB)
Final state: 6 epochs × 1,416 train samples = 8,496 steps
Best ckpt:   checkpoints/best.pt    (val loss 148.38, saved at step 8000)
```

### Artifacts

```
/mnt/data/runs/stage1_20260505_053645/
├── checkpoints/
│   ├── best.pt          1.35 GB   ← used as input to Stage 2
│   ├── final.pt         1.35 GB
│   └── step_NNNNN.pt    1.35 GB   (every 500 steps, step_00500 … step_08000)
├── metrics/
│   ├── metrics.jsonl    25 MB     (one JSON record per train step + per val pass)
│   └── summary.json     15 MB     (columnar mirror of metrics.jsonl, refreshed each val)
└── logs/
    └── train.log        163 KB    (mirrors stdout)
```

### Config used

Launched as a Kubernetes Job from `k8s/aurora-finetune-stage1-job.yaml`. The
exact CLI was:

```bash
python -u -m src.finetune_stage1 \
    --data-dir /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025 \
    --run-dir  /mnt/data/runs \
    --run-name "stage1_20260505_053645" \
    --epochs           6 \
    --rollout-steps    1 \
    --warmup-steps     500 \
    --lr-base          5e-5 \
    --lr-new-embed     1e-3 \
    --weight-decay     5e-6 \
    --grad-clip        1.0 \
    --val-every        300 \
    --n-val-samples    50 \
    --save-every       500 \
    --num-workers      8 \
    --prefetch-factor  2
```

Per-variable surface weights (paper §D.1 + plan doc):

```
2t=3.0  10u=0.77  10v=0.66  msl=1.5
swvl1=1.5  swvl1_density=1.5
stl1=2.0   stl1_density=2.0
sd=1.0     sd_density=1.0
```

Atmospheric weights (per pressure level): `z=2.8 t=1.7 u=0.87 v=0.6 q=0.78`.
Loss scaling: `α=0.25` (surf), `β=1.0` (atmos), `γ=2.0` (ERA5).

### Resources
- Pod: 1 × L40 GPU, 64 GiB RAM, 24 GiB `/dev/shm`, 8 CPUs.

---

## Stage 2 — rollout fine-tuning

Loads the Stage 1 checkpoint, **freezes everything except LoRA adapters**
injected into the Swin3D backbone's self-attention `qkv` and `proj` projections
(paper §D.4). Trains on multi-day rollouts using a replay buffer + pushforward
trick. Predicted density channels are thresholded at 0.5 when fed back as
input to avoid input-distribution mismatch (paper §B.8).

### Successful run

```
Run name:    stage2_20260509_090700
Run dir:     /mnt/data/runs/stage2_20260509_090700/
Wall time:   ~14.9 h on 1 × NVIDIA L40 (46 GB)
Final state: 8000 / 8000 steps (run completed cleanly)
Best ckpt:   checkpoints/best.pt    (mean lat-weighted RMSE 119.89,
                                      saved at step 8000)
20 val passes completed; 8 periodic ckpts + best + final saved.
```

This was launched **resuming from Stage 1's `best.pt`** (LoRA params
zero-initialized via `load_state_dict(strict=False)`).

### Artifacts

```
/mnt/data/runs/stage2_20260509_090700/
├── checkpoints/
│   ├── best.pt          458 MB    ← lowest mean rmse_lat across all (lead, var)
│   ├── final.pt         458 MB
│   └── step_NNNNN.pt    458 MB    (every 1000 steps, step_01000 … step_08000)
├── metrics/
│   ├── metrics.jsonl    27 MB     (train rows: 8000; val_rollout rows: 18000 = 20 passes × 900 (lead, var))
│   └── summary.json     15 MB     (columnar mirror)
└── logs/
    └── train.log        156 KB
```

Stage 2 checkpoints are smaller than Stage 1's because the optimizer state
only contains the ~540 K LoRA trainables (vs. all 113 M params in Stage 1).

### Config used

Launched directly via `kubectl exec` into the running interactive pod. The
exact CLI was:

```bash
PYTHONPATH=/opt/repo PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
python -u -m src.finetune_stage2 \
    --data-dir       /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025 \
    --stage1-ckpt    /mnt/data/runs/stage1_20260505_053645/checkpoints/best.pt \
    --run-dir        /mnt/data/runs \
    --run-name       "stage2_20260509_090700" \
    --swvl1-weight   4.0 \
    --stl1-weight    3.0 \
    --sd-weight      2.0 \
    --total-steps    8000 \
    --buffer-size    30 \
    --dataset-sampling-period 10 \
    --lead-warmup-steps 1000 \
    --max-lead-early 6 \
    --max-lead-late  12 \
    --lr             5e-5 \
    --weight-decay   0.0 \
    --grad-clip      1.0 \
    --val-every      400 \
    --n-val-samples  20 \
    --save-every     1000 \
    --num-workers    6 \
    --num-val-workers 6 \
    --prefetch-factor 2 \
    --gt-prefetch-workers 4
```

This run used **boosted surface weights for the new soil/snow variables**
(`--swvl1-weight 4.0 --stl1-weight 3.0 --sd-weight 2.0`). The full effective
`SURF_WEIGHTS` dict the loss saw:

```
2t=3.0  10u=0.77  10v=0.66  msl=1.5
swvl1=4.0  swvl1_density=4.0    ← boosted from 1.5
stl1=3.0   stl1_density=3.0     ← boosted from 2.0
sd=2.0     sd_density=2.0       ← boosted from 1.0
```

Atmospheric weights and `α / β / γ` are unchanged from Stage 1.

### Lead-time curriculum (replay buffer)

| Window         | `max_lead` cap | Forecast horizon |
|---|---|---|
| Steps 0 – 1000 | 6  | up to 36 h |
| Steps 1000 – 8000 | 12 | up to 72 h |

The buffer holds 30 entries. Each step samples one entry, runs one forward
pass with gradients (pushforward trick — only the last step has gradients),
then either extends the chain (push back at `lead+1` if under cap) or
replaces with a fresh dataset sample. A fresh sample is also injected every
10 steps for diversity.

### Validation protocol

Every 400 train steps, 20 random validation samples are autoregressively
rolled out for 12 steps (= 72 h). Each `val_rollout` JSONL row is one
`(step, lead_h, var)` tuple with latitude-weighted RMSE (paper eq. F15) and
unweighted MAE. Soil variables are masked to land via the target density
channel before computing RMSE. The "best" checkpoint is selected by the mean
of `rmse_lat` over **all** records in a pass (900 records per pass: 12 leads ×
75 vars/levels), so it is dominated by atmospheric variables.

### Resources
- Pod: 1 × L40 GPU, **96 GiB RAM**, 24 GiB `/dev/shm`, 8 CPUs.
  - The 96 GiB bump from Stage 1's 64 GiB was required because the replay
    buffer holds 30 cloned `Batch` objects (~1 GB each) in main-process
    pageable RAM. See [`finetuning_plan.md`](finetuning_plan.md#stage-2-rollout-fine-tuning)
    for the memory budget.

---

## How to load these checkpoints

Either checkpoint loads via PyTorch's standard pattern:

```python
import torch
from aurora import AuroraSmallPretrained
from src.data import SOIL_SURF_VARS
from src.finetune_stage1 import register_norm_stats

register_norm_stats()  # MUST come before model construction

# Stage 1 model:
model = AuroraSmallPretrained(autocast=True, surf_vars=SOIL_SURF_VARS, use_lora=False)
state = torch.load("/mnt/data/runs/stage1_20260505_053645/checkpoints/best.pt", map_location="cpu")
model.load_state_dict(state["model"], strict=False)

# Or Stage 2 model (LoRA-enabled):
model = AuroraSmallPretrained(autocast=True, surf_vars=SOIL_SURF_VARS, use_lora=True)
state = torch.load("/mnt/data/runs/stage2_20260509_090700/checkpoints/best.pt", map_location="cpu")
model.load_state_dict(state["model"], strict=False)
```

`strict=False` is required when loading the Stage 1 checkpoint into a
`use_lora=True` model — the 80 LoRA params don't exist in Stage 1's state
dict and will be initialised to zero, which is the correct identity-LoRA
starting point.

## Reproducing

Both runs read ERA5 from the per-timestep layout at
`/mnt/data/era5/per-step/{2024,2025}/`. Train / val / test split is the
default summer split (June train, July 1-15 val, July 16-31 test for both
years) — see [`preprocessing_and_splits.md`](preprocessing_and_splits.md).

The Stage 1 launch yaml is `k8s/aurora-finetune-stage1-job.yaml`. There is no
dedicated yaml for Stage 2; it was launched into the long-lived
`aurora-interactive` pod (see `k8s/aurora-interactive-pod.yaml`, which now
requests 96 GiB to fit Stage 2's replay buffer).
