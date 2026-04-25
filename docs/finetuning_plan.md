# Aurora Fine-Tuning Plan: Soil Moisture Variables

## Overview

We are fine-tuning **AuroraSmallPretrained** (~113M parameters, `embed_dim=256`) to forecast
three new ERA5 surface variables that were not part of Aurora's original pretraining:

| Variable | ERA5 short name | Description |
|---|---|---|
| Volumetric soil water layer 1 | `swvl1` | Soil moisture in the top 7 cm |
| Soil temperature level 1 | `stl1` | Temperature of the top soil layer |
| Snow depth | `sd` | Depth of snow water equivalent |

We use the small model because we have ~48 GB GPU memory (not 80 GB A100s), and limited
compute budget. The small model at 0.25° resolution fits comfortably in 48 GB even with
multi-step rollouts.

**Training data:** ERA5 0.25° resolution, June–August 2024 and 2025 (summer only).

**Train/val/test split:**

| Split | Date Range | Days | % |
|---|---|---|---|
| Train | Jun 1 – Aug 1 2024 + Jun 1 – Aug 1 2025 | ~122 | 66% |
| Val | Aug 1 – Aug 16 2024 + Aug 1 – Aug 16 2025 | ~30 | 17% |
| Test | Aug 16 – Sep 1 2024 + Aug 16 – Sep 1 2025 | ~32 | 17% |

---

## Closest Paper Analog: HRES-WAM 0.25°

The paper describes four fine-tuning tasks. Here's how they compare to our scenario:

| Task | New vars? | Resolution | Data source | LR (new) | LR (rest) | Warmup | Schedule | Variable weight changes |
|---|---|---|---|---|---|---|---|---|
| **HRES 0.25° T0** | No | 0.25° | IFS T0 | N/A | 5e-5 | 1k steps | Constant | IFS-specific adjustments |
| **HRES 0.1° analysis** | No | 0.1° | IFS analysis | N/A | 2e-4 | 1k steps | Constant | IFS-specific adjustments |
| **CAMS 0.4°** | Yes (pollution) | 0.4° | CAMS | 1e-3 | 1e-4 | 100 steps | None | Persistence-MAE for new vars only |
| **HRES-WAM 0.25°** | Yes (waves) | 0.25° | WAM+ERA5 | 1e-3 | 3e-4 | 500 steps | Cosine→1e-5 | New var weights only |
| **Ours** | Yes (soil) | 0.25° | ERA5 | 1e-3 | 5e-5 | 500 steps | Cosine→1e-5 | New var weights only |

**We follow the HRES-WAM pattern** because:
- We are adding new surface variables (like waves were added)
- We are at 0.25° resolution (same as WAM)
- Our data source is ERA5 (not IFS, so IFS-specific weight adjustments don't apply)

**One difference**: our base LR is 5e-5 (from HRES 0.25° T0) rather than 3e-4 (from WAM).
The WAM task uses a higher base LR because it fine-tunes with a cosine schedule over 30k
steps on a much larger dataset (6 years of data). With our small dataset (~4 months),
we need a lower base LR to avoid overshooting.

---

## Handling Missing Data: Density Channels

### The problem

`swvl1`, `stl1`, and `sd` are undefined over ocean — they're NaN in ERA5. This is ~70%
of the globe. We can't feed NaN to the model (training breaks), but we also can't just
set them to zero, because:

1. After normalization, zero is a specific numerical value, not "no data."
2. The encoder mixes all surface variables together via shared attention. Injecting a
   constant zero signal across 70% of the globe creates a spurious pattern that can
   perturb the pretrained representations of the other variables (`2t`, `10u`, etc.).
3. The model would have to learn indirectly (via the `lsm` static variable) that these
   zeros are meaningless — a harder learning problem, especially early in fine-tuning
   when the new embeddings are still near zero and gradients are noisy.

### The solution: density channels (following AuroraWave)

The Aurora library implements density channels in the `AuroraWave` class (Section B.8 of
the paper). For each variable that can be missing, a companion **density channel** is
added as a separate surface variable:

- **density = 1** where data is present (land for soil variables)
- **density = 0** where data is absent (ocean)

This gives the model an explicit, unambiguous signal at the patch embedding level: "this
variable has no data here." The model can learn very quickly to ignore the companion
data channel wherever density = 0, without relying on indirect signals.

### How it works in code

The `AuroraWave` class implements this in three hooks:

**Before the encoder** (`_pre_encoder_hook`):
```python
# For each soil variable:
density = (~torch.isnan(x)).float()           # 1 on land, 0 on ocean
batch.surf_vars[f"{name}_density"] = density
batch.surf_vars[name] = x.nan_to_num(0)       # Replace NaN with 0
```

**After the decoder** (`_post_decoder_hook`):
```python
# The model predicts raw density logits; apply sigmoid to bound [0, 1]
density = torch.sigmoid(pred.surf_vars[f"{name}_density"])
data = pred.surf_vars[name]
data[density < 0.5] = float("nan")  # Model predicts "no data here"
```

**In the loss**: only compute MAE for soil variables where the target density = 1
(i.e., land points only). The density channels themselves also contribute to the loss
with the same weight as their parent variable.

### Our configuration

We add 3 density channels, one for each new variable:

| Data variable | Density channel | Meaning |
|---|---|---|
| `swvl1` | `swvl1_density` | 1 on land, 0 on ocean |
| `stl1` | `stl1_density` | 1 on land, 0 on ocean |
| `sd` | `sd_density` | 1 on land, 0 on ocean |

This brings the total new surface variables from 3 to 6 (3 data + 3 density), meaning
6 new patch embeddings to learn.

For normalization of density channels: following the wave pattern, density channels have
**mean = 0, std = 1** (no normalization applied). They are inherently in [0, 1].

### Why not just use the land-sea mask?

The `lsm` static variable is already available, but:

- It's a **static** variable, processed through a different pathway than surface variables.
  The encoder concatenates static vars to surface vars, but they don't share the same
  patch embedding mechanism.
- Density channels are per-variable and sit right next to their data variable in the
  surface variable dict, giving the model a direct per-variable "data present" signal
  at the embedding level.
- This matches the proven pattern from the paper. The wave model uses density channels
  and works well. We follow the same approach for consistency and safety.

---

## Two-Stage Fine-Tuning Pipeline

Following the paper (Sections D.3 and D.4), fine-tuning happens in two sequential stages:

1. **Short-lead-time fine-tuning** — train the entire model on 1–2 step predictions
2. **Rollout fine-tuning** — freeze the base model, train only LoRA layers using a replay
   buffer for multi-day forecasts

Stage 2 starts from the checkpoint produced by Stage 1.

---

## Stage 1: Short-Lead-Time Fine-Tuning

### What it does

Fine-tunes **all model parameters** through 1–2 autoregressive steps (6–12 h forecasts),
backpropagating through all steps. This teaches the model the new surface variables at
short forecast horizons.

### Model configuration

```python
from aurora import AuroraSmallPretrained

# Include density channels alongside each new variable
SOIL_SURF_VARS = (
    "2t", "10u", "10v", "msl",
    "swvl1", "swvl1_density",
    "stl1",  "stl1_density",
    "sd",    "sd_density",
)

model = AuroraSmallPretrained(
    autocast=True,       # bf16 mixed precision
    surf_vars=SOIL_SURF_VARS,
    use_lora=False,      # full fine-tuning, no LoRA yet
)
model.load_checkpoint(strict=False)  # zeros for new variable embeddings
model.configure_activation_checkpointing()
```

- `strict=False` because the pretrained checkpoint doesn't contain weights for the new
  variables or their density channels. All 6 new patch embeddings are initialized to
  zero, matching the paper's approach for new wave variables (Section B.8).
- Normalization statistics for the new data variables must be registered before training
  (see "Prerequisites" section below). Density channels use mean=0, std=1 (no
  normalization).

### Optimizer setup: split learning rates

Following the CAMS and wave fine-tuning patterns (Sections D.3), new variable patch
embeddings get a higher learning rate than the pretrained backbone:

```python
optimizer = torch.optim.AdamW([
    {"params": base_params,      "lr": 5e-5},   # pretrained weights
    {"params": new_embed_params, "lr": 1e-3},   # new variable embeddings
], weight_decay=5e-6)
```

The new embeddings start from zero and need to learn from scratch, so they need a much
higher learning rate. The pretrained backbone already has good representations and only
needs gentle adaptation.

### Training loop: 1-step vs 2-step rollout

**1-step (recommended to start):**
```
input: (t-6h, t) → model → prediction at t+6h → MAE loss vs ground truth at t+6h
```

**2-step (if memory allows):**
```
input: (t-6h, t) → model → pred at t+6h → assemble new input → model → pred at t+12h
loss = average(MAE at t+6h, MAE at t+12h)
```

With AuroraSmall on a 48 GB GPU, 2-step rollout should fit. The full 1.3B model required
8 GPUs with gradient sharding to fit 2 steps on A100-80GB, but our model is ~10x smaller.

### Hyperparameters

| Parameter | Value | What it controls | Why this value |
|---|---|---|---|
| **Model** | `AuroraSmallPretrained` | Architecture size (~113M params, embed_dim=256) | Fits in 48 GB GPU; we lack compute for the 1.3B model |
| **Rollout steps** | 2 (try 1 if OOM) | Number of autoregressive forward passes during training, backprop through all | Paper uses 2 for HRES 0.25° (D.3). Small model should fit. |
| **Batch size** | 1 | Samples per GPU per step | Always 1 for Aurora at 0.25° resolution — the full grid is huge |
| **Total training steps** | 3,000 | How long to train | We have ~488 triplets in training set (122 days × 4 per day at 6h). 3k steps ≈ 6 epochs. Paper uses 8k–14k but with orders of magnitude more data. |
| **LR (pretrained params)** | 5e-5 | Learning rate for all existing pretrained weights | Matches HRES 0.25° fine-tuning (D.3). Low because these weights are already good. |
| **LR (new embeddings)** | 1e-3 | Learning rate for swvl1/stl1/sd and their density channel patch embeddings | Matches CAMS/wave pattern (D.3). High because these start from zero. |
| **LR warmup** | 500 steps | Linearly ramp LR from 0 to target over this many steps | Matches wave fine-tuning (D.3). Prevents early instability when loss landscape is unexplored. |
| **LR schedule** | Cosine decay to 1e-5 | Anneal LR after warmup | Matches wave fine-tuning (D.3). Helps convergence in later stages. |
| **Optimizer** | AdamW | Adaptive optimizer with decoupled weight decay | Used throughout the paper. Standard for transformer training. |
| **Weight decay** | 5e-6 | L2 regularization strength (decoupled from LR in AdamW) | Same as pretraining (D.2). Small value — Aurora relies on other regularization. |
| **Drop path** | 0.0 (disabled) | Stochastic depth probability — randomly drops entire transformer blocks during training | All fine-tuning in the paper disables this (D.3). Only used during pretraining. |
| **Gradient clipping** | 1.0 | Max gradient norm before clipping | Standard for transformers. Prevents exploding gradients during early training with new variables. |
| **Activation checkpointing** | All layers | Recompute activations during backward instead of storing them | Trades compute for memory. Required to fit model + 2-step rollout in 48 GB. |
| **Mixed precision** | bf16 (`autocast=True`) | Use bfloat16 for forward/backward, float32 for optimizer state | Standard for Aurora. Halves activation memory, ~1.5x speedup. |

### The weighted MAE loss

The paper's loss function (Section D.1) is:

$$
\mathcal{L} = \frac{\gamma}{V_S + V_A}
\left[
\alpha \sum_{k=1}^{V_S} w_k^S \cdot \text{MAE}_k^{\text{surf}}
+ \beta \sum_{k=1}^{V_A} w_k^A \cdot \text{MAE}_k^{\text{atmos}}
\right]
$$

Where each variable's MAE is averaged over all grid points. For soil variables, the MAE
is computed **only over land points** (where the target density channel = 1). The density
channels themselves also contribute to the loss with the same weight as their parent
variable — this trains the model to correctly predict where data is present vs absent.

#### Loss weighting hyperparameters

| Parameter | Value | What it controls | Why this value |
|---|---|---|---|
| **α (surface weight)** | 0.25 | Global weight on surface loss component | Paper value (D.1). Downweights surface relative to atmospheric because there are fewer surface variables. |
| **β (atmospheric weight)** | 1.0 | Global weight on atmospheric loss component | Paper value (D.1). |
| **γ (dataset weight)** | 2.0 | Overall multiplier for ERA5 data | Paper value (D.1). ERA5 is higher fidelity than other datasets Aurora trains on. |

#### Which variable weights to use?

The paper defines several sets of per-variable weights for different tasks (D.1):

1. **Pretraining weights** — the base set, used across all datasets
2. **IFS T0 / analysis fine-tuning weights** — slight adjustments for IFS-specific tasks
3. **CAMS weights** — persistence-MAE trick for sparse pollution variables
4. **Wave weights** — manually tuned for wave-specific variables

**Our task is most analogous to HRES-WAM 0.25° (wave fine-tuning)**: we are adding
new surface variables to a pretrained model at the same 0.25° resolution, trained on ERA5.

For HRES-WAM and CAMS fine-tuning, the paper does **not** adjust the existing
meteorological variable weights — it only adds weights for the new variables. The "IFS
fine-tuning weights" (w_2t=3.5, w_msl=1.6, w_z=3.5, etc.) were specifically tuned for
improving forecasts evaluated against IFS analysis data, which is a different distribution
from ERA5.

**We use the pretraining weights for existing variables** and add weights for our
new variables.

#### Per-variable loss weights

**Surface variables** (existing — pretraining weights from D.1):

| Variable | Weight | Rationale |
|---|---|---|
| `2t` (2m temperature) | 3.0 | Highest surface weight. Temperature is the most closely watched surface variable. |
| `msl` (mean sea level pressure) | 1.5 | Key synoptic-scale variable driving large-scale flow. |
| `10u` (10m u-wind) | 0.77 | Wind components are less individually critical than temperature/pressure. |
| `10v` (10m v-wind) | 0.66 | Slightly less weight than u-wind (balances loss contributions). |

**Surface variables** (new — starting points, tune based on loss curves):

| Variable | Weight | Rationale |
|---|---|---|
| `swvl1` (soil moisture) | 1.5 | Primary variable of interest for this project. Start at a moderate weight. |
| `swvl1_density` | 1.5 | Density channel inherits parent weight (following wave pattern). |
| `stl1` (soil temperature) | 2.0 | Analogous to 2t but for soil. Important for land-atmosphere coupling. |
| `stl1_density` | 2.0 | Density channel inherits parent weight. |
| `sd` (snow depth) | 1.0 | Least important of our new variables; very sparse in summer data. |
| `sd_density` | 1.0 | Density channel inherits parent weight. |

**Atmospheric variables** (pretraining weights from D.1, same for all pressure levels):

| Variable | Weight | Rationale |
|---|---|---|
| `z` (geopotential) | 2.8 | Most important upper-air variable (encodes height of pressure surfaces). |
| `t` (temperature) | 1.7 | Second most important atmospheric variable. |
| `u` (u-wind) | 0.87 | Wind is well-constrained by geopotential via geostrophic balance. |
| `q` (specific humidity) | 0.78 | Moisture field; noisier, lower weight. |
| `v` (v-wind) | 0.6 | Lowest weight (similar reasoning to u-wind). |

---

## Stage 2: Rollout Fine-Tuning

### What it does

Takes the checkpoint from Stage 1 and trains **only LoRA layers** on long autoregressive
sequences (multi-day forecasts) using a replay buffer. This teaches the model stable
dynamics over many steps without the model diverging or accumulating errors.

### Why LoRA + pushforward + replay buffer

Three techniques work together to make long rollout training feasible:

1. **LoRA (Low-Rank Adaptation):** Only trains small low-rank matrices injected into the
   self-attention layers. Dramatically fewer trainable parameters → less memory for
   optimizer states and gradients.

2. **Pushforward trick:** Only backpropagates through the **last** rollout step. Prior
   steps run with `torch.no_grad()`. This means GPU memory is constant regardless of
   rollout length — the same as single-step training.

3. **Replay buffer:** Stores model states at various lead times in CPU memory. Each
   training step samples one entry, advances it one step, and puts the result back.
   Periodically refreshes with fresh data from the dataset. This avoids having to
   generate long rollouts from scratch at each training step.

### Model configuration

```python
model = AuroraSmallPretrained(
    autocast=True,
    surf_vars=SOIL_SURF_VARS,  # same tuple as Stage 1, includes density channels
    use_lora=True,             # enable LoRA on all self-attention linear layers
)
# Load the Stage 1 checkpoint (not the original pretrained one)
model.load_checkpoint_local("checkpoints/short_leadtime_best.pt", strict=False)
model.configure_activation_checkpointing()

# Freeze everything except LoRA parameters
for name, param in model.named_parameters():
    if "lora" not in name.lower():
        param.requires_grad = False
```

### The replay buffer algorithm

```
1. Initialize buffer with N samples from the training dataset (initial conditions)
   Each entry stores: (input_batch, ground_truth_timestamp, lead_time_steps=0)

2. For each training step:
   a. Sample one entry from the buffer
   b. Forward the model ONE step (with gradients)
   c. Compute MAE loss against ground truth at the predicted timestamp
   d. Backprop + optimizer step
   e. Detach the prediction and add it back to the buffer with lead_time += 1
   f. If lead_time exceeds max_lead_time, discard the entry
   g. Every `dataset_sampling_period` steps, replace some entries with fresh
      initial conditions from the dataset

3. Lead time schedule:
   - First 2k steps: only keep entries with lead time ≤ 4 days (16 steps)
   - After 2k steps: allow lead times up to 10 days (40 steps)
```

The detaching is the pushforward trick — by detaching the prediction before storing it
in the buffer, when we later sample that entry and forward through it, gradients only
flow through that single forward pass.

### Hyperparameters

| Parameter | Value | What it controls | Why this value |
|---|---|---|---|
| **LoRA** | `use_lora=True` | Injects low-rank adaptation matrices into all self-attention linear layers in the Swin3D backbone | Paper's approach for rollout fine-tuning (D.4). Enables parameter-efficient adaptation. |
| **LoRA rank** | Library default | Rank of the low-rank matrices A, B. Lower = fewer params, less capacity. | Use library default; can increase if underfitting. |
| **Frozen params** | Everything except LoRA | Which parameters receive gradients | Only LoRA layers train during rollout (D.4). Base model is fixed. |
| **Buffer size (per GPU)** | 100 | Number of (batch, lead_time) entries in CPU memory per GPU | Paper uses 100–200 per GPU (D.4). Each entry ≈ 300 MB at 0.25° (all vars × 721 × 1440 × float32). 100 entries ≈ 30 GB CPU RAM. |
| **Dataset sampling period** | 10 | Every N steps, replace some buffer entries with fresh initial conditions from the dataset | Paper value (D.4). Balances exploration of new initial conditions vs. extending existing rollouts. |
| **Max lead time (early)** | 4 days (first 2k steps) | Maximum forecast horizon in the buffer during early training | Paper uses this curriculum for HRES 0.25° (D.4). Learn short-range dynamics before attempting long-range. |
| **Max lead time (late)** | 10 days (after 2k steps) | Maximum forecast horizon in the buffer during later training | Paper's HRES 0.25° protocol (D.4). 10 days is the typical medium-range forecast horizon. |
| **Total steps** | 4,000 | Training duration | Paper uses 6k–13k with far more data. Scale down proportionally for our ~122 training days. |
| **LR** | 5e-5, constant | Learning rate for LoRA parameters | Paper uses this for all rollout tasks (D.4). No schedule needed — LoRA layers are small and start from zero. |
| **Optimizer** | AdamW | Same as Stage 1 | Consistency with paper. |
| **Weight decay** | 0 | No regularization on LoRA weights | LoRA matrices are small; weight decay can hurt their expressivity. |
| **Batch size** | 1 | Samples per GPU | Same as always at 0.25°. |
| **Loss** | Same weighted MAE as Stage 1 | Training objective | Same loss throughout all training stages (D.1). |

---

## Prerequisites (before training)

### 1. Compute normalization statistics

The new variables need per-variable mean and standard deviation computed from the
**training split only** (to prevent data leakage).

```bash
# On the cluster (via k8s/compute-norm-stats-job.yaml):
python -u scripts/compute_norm_stats.py \
    --data-dir /mnt/data/era5/2024 /mnt/data/era5/2025
```

This outputs values to paste into `src/finetune.py`'s `_NEW_VAR_NORM` dictionary.

### 2. Register normalization before model creation

```python
from aurora.normalisation import locations, scales

# Data variables: replace with real computed values
locations["swvl1"] = <computed_mean>
scales["swvl1"]    = <computed_std>
locations["stl1"]  = <computed_mean>
scales["stl1"]     = <computed_std>
locations["sd"]    = <computed_mean>
scales["sd"]       = <computed_std>

# Density channels: no normalization (already in [0, 1])
locations["swvl1_density"] = 0.0
scales["swvl1_density"]    = 1.0
locations["stl1_density"]  = 0.0
scales["stl1_density"]     = 1.0
locations["sd_density"]    = 0.0
scales["sd_density"]       = 1.0
```

These must be set **before** `model.forward()` is called. Aurora's normalisation module
uses these global dictionaries to normalize inputs and unnormalize outputs.

---

## Things to Watch Out For

### Memory

- **48 GB GPU** with AuroraSmall at 0.25° should be comfortable for single-step, and
  likely fine for 2-step rollout with activation checkpointing. If you hit OOM on
  2-step, fall back to 1-step.
- **CPU memory** for the replay buffer: 100 entries × ~300 MB = ~30 GB. Make sure your
  k8s job requests enough CPU memory (at least 48 GB to be safe).
- **Activation checkpointing** is critical. Without it, even single-step training can OOM.

### Loss dynamics

- The new variable losses will start very high (embeddings are zero → predictions are
  garbage). This is expected.
- The pretrained variable losses should stay roughly stable or dip slightly. If they
  spike, the base LR is too high.
- If new variable losses plateau early, the new embedding LR may be too low, or the loss
  weights are drowning them out.
- Monitor per-variable losses separately, not just the total.

### Checkpoint management

- Save checkpoints every N steps and keep the best based on validation loss.
- Stage 2 must load the Stage 1 checkpoint, not the original pretrained one.
- When loading with `use_lora=True`, the LoRA parameters won't exist in the Stage 1
  checkpoint. Use `strict=False` — they'll be initialized to zero (which is correct,
  since LoRA starts as the identity transformation).

### Numerical stability

- bf16 mixed precision (`autocast=True`) is standard for Aurora. The model was pretrained
  this way.
- Gradient clipping at 1.0 prevents occasional large gradients from destabilizing training.
- If training is unstable, try reducing the new embedding LR before anything else.

### Data loading and density channels

- The dataloader must create density channels on the fly: for each soil variable,
  add a `{name}_density` tensor that is 1 where not NaN, 0 where NaN, then replace
  NaN with 0 in the data variable. This happens in the dataloader, before the batch
  is passed to the model (the model's normalisation will handle the rest).
- The dataloader currently returns triplets (t-6h, t, t+6h). This works for Stage 1.
- For Stage 2's replay buffer, you also need to look up arbitrary ground-truth timestamps
  to compute loss at any lead time. The current dataset can be queried for this, but the
  replay buffer itself manages the autoregressive chain.
- During autoregressive rollout, the model's own density predictions feed back as input.
  Following the wave pattern: threshold predicted density at 0.5 — set density to 1
  where > 0.5, and set both density and data to 0 where < 0.5. This prevents
  distribution mismatch between training inputs (binary 0/1) and model predictions
  (continuous sigmoid output).
- With `num_workers > 0` in the DataLoader, data loading should not be a bottleneck
  even with the current NetCDF chunking.

### Multi-GPU (if available)

- The paper uses 8–32 GPUs with gradient sharding (FSDP/DDP) for all fine-tuning.
- If you can request multiple GPUs on Nautilus, use `torchrun` with PyTorch DDP/FSDP.
  This linearly scales effective batch size and allows larger replay buffers.
- With a single GPU, everything still works — just slower and with smaller buffer.

---

## File Structure

```
src/
  finetune_short.py     # Stage 1: short-lead-time fine-tuning
  finetune_rollout.py   # Stage 2: rollout fine-tuning with LoRA + replay buffer
  data.py               # Dataset and data loading
  finetune.py           # Current prototype (to be replaced by the above)
k8s/
  aurora-finetune-job.yaml  # K8s job for running training
scripts/
  compute_norm_stats.py     # Compute normalization statistics
docs/
  finetuning_plan.md        # This document
  preprocessing_and_splits.md  # Data preprocessing and split details
```

---

## Summary of All Hyperparameters at a Glance

### Stage 1 (Short-Lead-Time)

| Hyperparameter | Value |
|---|---|
| Model | AuroraSmallPretrained (~113M params) |
| Rollout steps | 2 (1 if OOM) |
| Batch size | 1 |
| Training steps | 3,000 |
| LR (pretrained) | 5e-5 |
| LR (new embeddings) | 1e-3 |
| LR warmup | 500 steps |
| LR schedule | Cosine → 1e-5 |
| Optimizer | AdamW |
| Weight decay | 5e-6 |
| Drop path | 0.0 |
| Grad clip | 1.0 |
| Activation ckpt | All layers |
| Precision | bf16 |
| α (surface) | 0.25 |
| β (atmospheric) | 1.0 |
| γ (ERA5) | 2.0 |

### Stage 2 (Rollout)

| Hyperparameter | Value |
|---|---|
| Model | AuroraSmallPretrained + LoRA |
| LoRA | All self-attention linears |
| Trainable params | LoRA only |
| Buffer size / GPU | 100 |
| Dataset sampling period | 10 |
| Max lead time (early) | 4 days (steps 0–2k) |
| Max lead time (late) | 10 days (steps 2k+) |
| Training steps | 4,000 |
| LR | 5e-5 (constant) |
| Optimizer | AdamW |
| Weight decay | 0 |
| Batch size | 1 |
| Grad clip | 1.0 |
| Activation ckpt | All layers |
| Precision | bf16 |
