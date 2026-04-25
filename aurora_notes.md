# Aurora: A Foundation Model for the Earth System — Detailed Notes

## Overview

Aurora is a **1.3 billion parameter foundation model** for the Earth system, developed by Microsoft Research AI for Science. It is trained on **over one million hours** of diverse Earth system data (weather, climate simulations, atmospheric chemistry, ocean waves). The key idea is a **pretrain → fine-tune** paradigm: a single large model is pretrained on heterogeneous atmospheric data, then cheaply fine-tuned for specific downstream tasks, outperforming dedicated operational systems at orders of magnitude less computational cost.

### Key Results

| Domain | Resolution | Comparison | Outperforms on |
|---|---|---|---|
| Air quality (CAMS) | 0.4° | CAMS operational forecasts | 74% of targets |
| Ocean waves (HRES-WAM) | 0.25° | HRES-WAM operational | 86% of targets |
| Tropical cyclone tracks | 0.25° | 7 operational agencies | 100% of targets |
| Weather forecasting | 0.1° | IFS HRES (state-of-art NWP) | 92% of targets |

---

## Section 8: Methods

### 8.0 Problem Statement

- The atmosphere state at time `t` is represented as a tensor `X_t ∈ R^{V × H × W}`, where:
  - `V` = total number of variables
  - `H` = number of latitude coordinates
  - `W` = number of longitude coordinates
- This state is split into **surface** (`S_t ∈ R^{V_S × H × W}`) and **atmospheric** (`A_t ∈ R^{V_A × C × H × W}`) components, where `C` = number of pressure levels.
- Aurora learns a **simulator** `Φ: (X_{t-1}, X_t) → X̂_{t+1}` that maps two consecutive observed states to a predicted next state.
- For longer lead times, predictions are generated via **autoregressive rollout**: feed predictions back as inputs repeatedly.

### 8.1 The Aurora Model Architecture

Aurora follows an **encoder → processor → decoder** design:

```
Input State → [3D Perceiver Encoder] → [3D Swin Transformer U-Net Backbone] → [3D Perceiver Decoder] → Predicted State
```

#### 3D Perceiver Encoder

- Treats all variables as `H × W` images on a lat-lon grid.
- Static variables (orography/geopotential, land-sea mask, soil type) are appended as extra surface variables.
- Images are split into **P × P patches** (P=4 for pretraining at 0.25°, P=10 for 0.1° fine-tuning).
- Patches are mapped to **D-dimensional embeddings** using **variable-specific linear transformations** (weights `W_v` per variable `v`).
- For each pressure level and the surface, variable embeddings are **summed** and tagged with an **additive level encoding** (sine/cosine encoding of pressure, or a learned vector for the surface).
- A **Perceiver module** (cross-attention + residual MLP) reduces the variable number of physical pressure levels `C` down to a fixed **`L = 3` latent pressure levels**. The surface level is passed through a separate residual MLP and concatenated, yielding `(L+1) = 4` latent levels.
- Result: a 3D tensor of shape `(L+1) × H/P × W/P` of D-dimensional embeddings.
- These are tagged with additive **Fourier encodings** for:
  - **Position** (lat/lon of patch center; λ_min=0.01, λ_max=720)
  - **Scale / patch area** (km² per patch, enabling multi-resolution operation; λ_max = Earth's surface area)
  - **Absolute time** (hours since Jan 1 1970; λ_min=1 hour, λ_max=hours in a year — captures diurnal/seasonal cycles)
  - **Pressure level** (λ_min=0.01, λ_max=10000)

#### Multi-Scale 3D Swin Transformer U-Net Backbone

- The "neural simulator" that evolves the latent representation in time.
- Architecture: **3D Swin Transformer U-Net** — a symmetric encoder-decoder with skip connections.
- **48 layers** across 3 stages (vs. 16 layers and 2 stages in Pangu-Weather):
  - Encoder stages: **(6, 10, 8)** layers
  - Decoder stages: **(8, 10, 6)** layers
- Each encoder stage **halves** spatial resolution via patch merging (and doubles embedding dim). Each decoder stage **doubles** resolution via patch splitting (and halves embedding dim). This enables **multi-scale physics simulation**.
- Each layer performs **local 3D self-attention within windows** (window size `(2, 12, 6)` in depth, width, height).
- Every other layer, windows **shift** by half the window size `(1, 6, 3)` — this allows information flow between neighboring regions while maintaining linear complexity.
- Spherical topology: left and right edges of images communicate directly during shifted windows.
- Uses **res-post-norm** layer normalization (from Swin v2) for training stability.
- Standard dot-product attention (no cosine attention).
- **No positional bias** in attention — positional information is handled by the encoder's Fourier encodings, making the backbone resolution-agnostic.
- Embedding dimension at first stage: **512**, doubling at each subsequent stage.
- Attention head dimension: **64** throughout. Number of heads scales with embedding dimension.

#### 3D Perceiver Decoder

- Mirrors the encoder in reverse.
- Deaggregates the 3 latent atmospheric levels back to `C` physical pressure levels using a **Perceiver layer** (sine/cosine pressure embeddings as queries).
- Decodes D-dimensional embeddings into `P × P` patches via **variable-specific linear layers** (weights selected dynamically per variable).
- Can output predictions at **arbitrary pressure levels** and for an **arbitrary set of variables**.

### 8.2 Training Methods

#### Training Objective

- **Mean Absolute Error (MAE)** throughout pretraining and fine-tuning.
- Loss is a weighted sum of per-variable surface MAE and per-variable-per-level atmospheric MAE.
- Key weights: surface weight `α = 1/4`, atmospheric weight `β = 1`, plus variable-specific weights `w_k` and dataset-specific weight `γ`.
- Dataset upweighting: `γ_{ERA5} = 2.0`, `γ_{GFS-T0} = 1.5`, rest = 1.

#### Pretraining

- **150k steps** on **32 A100 GPUs**, batch size 1 per GPU (~2.5 weeks, ~4.8M frames seen).
- Half-cosine decay LR schedule with 1k step linear warmup. Base LR = 5e-4, reduced 10× at end.
- Optimizer: **AdamW** (weight decay 5e-6).
- Regularization: **Drop path / stochastic depth** (probability 0.2).
- Memory: activation checkpointing for backbone layers, gradient sharding across GPUs.
- Precision: **bf16 mixed precision**.

#### Short-Lead-Time Fine-Tuning

- Fine-tune the **entire architecture** through 1–2 rollout steps (task-dependent).
- Typical: 8–12.5k steps on 8 GPUs, constant LR after warmup.
- Drop path disabled; activation checkpointing and gradient/weight sharding used.

#### Rollout (Long-Lead-Time) Fine-Tuning

- Uses **LoRA (Low-Rank Adaptation)** on all linear layers in the backbone's self-attention.
- Uses the **"pushforward trick"**: backprop only through the last rollout step (not the full chain).
- Uses an **in-memory replay buffer** (inspired by deep RL):
  1. Buffer starts populated with dataset samples.
  2. Each step: sample from buffer → predict next step → add prediction back to buffer.
  3. Every K steps: refresh buffer with new samples from dataset.
  - This enables training at arbitrary rollout lengths without extra memory/speed cost.

### 8.3 Datasets

Aurora is pretrained on **10 datasets** spanning ~1.26M hours of data (~1260 TB total):

| Dataset | Resolution | Timeframe | Type |
|---|---|---|---|
| ERA5 | 0.25° | 1979–2020 | Reanalysis |
| HRES forecasts | 0.25° | 2016–2020 | Operational forecasts |
| IFS ENS | 0.25° | 2018–2020 | Ensemble forecasts (50 members) |
| GFS forecasts | 0.25° | 2015–2020 | Operational forecasts |
| GFS T0 | 0.25° | 2015–2020 | Operational analysis |
| GEFS reforecasts | 0.25° | 2000–2019 | Ensemble reforecasts (5 members) |
| CMCC-CM2-VHR4 | 0.25° | 1950–2014 | CMIP6 climate simulation |
| ECMWF-IFS-HR | 0.45° | 1950–2014 | CMIP6 climate simulation |
| MERRA-2 | varies | varies | NASA atmospheric reanalysis |
| IFS ENS mean | 0.25° | 2018–2020 | Ensemble mean |

### 8.4 Task-Specific Adaptations

- **Ocean waves**: wave data has spatially varying absence (land, sea ice). Each wave variable gets an additional **density channel** (1 = present, 0 = absent) so the model can handle and predict missing data.

### 8.5 Data Infrastructure

- Individual datapoints are ~2 GB at 0.1°; training is **bottlenecked by data loading**, not compute.
- Azure blob storage with co-located data/compute.
- Advanced multi-source pipeline: YAML-configured `BatchGenerator` objects per dataset → combined, shuffled, sharded across GPUs.
- Different batch sizes for different datasets to balance workloads.

### 8.6 Verification Metrics

- **RMSE** (Root Mean Square Error) with latitude weighting.
- **ACC** (Anomaly Correlation Coefficient) against daily climatology.
- **Thresholded RMSE** for extreme event evaluation: only grid points exceeding an intensity threshold (based on ERA5 mean ± std) are included.

---

## Supplement B: The Aurora Model (Full Architecture Details)

### B.1 3D Perceiver Encoder (Detailed)

**Inputs:**
- All variables treated as `H × W` images on a regular lat-lon grid.
- For each variable, both current time `t` and previous time `t-1` are included → `T = 2` time dimension.
- Atmospheric state: `V_A × C × T × H × W` tensor.
- Surface state: `V_S × T × H × W` tensor.

**Static Variables:**
- Geopotential at surface (Z) — local orography
- Land-sea mask (LSM)
- Soil type (SLT)
- Internally appended to surface-level variables.

**Patch Embedding:**
- Each `H × W` image is split into `P × P` patches.
- Patches at each level mapped to `R^D` by a **variable-specific** linear layer:
  - Atmospheric: `C × V_A × T × P × P → C × D`
  - Surface: `V_S × T × P × P → 1 × D`
- The linear transform is **constructed dynamically** per variable `v` using weights `W_v`.

**Level Encodings:**
- Each level embedding gets an additive encoding:
  - Atmospheric levels: sine/cosine encoding of the pressure value (e.g., 500 hPa).
  - Surface level: **fully-learned** D-dimensional vector.

**Level Aggregation (Perceiver):**
- Reduces `C` physical pressure levels → `C_L = 3` latent pressure levels.
- Uses a Perceiver module: 1 cross-attention layer + residual MLP.
  - Queries: `C_L = 3` latent vectors
  - Keys/Values: `C` vectors computed from level embeddings via linear transform.
  - Output: `C_L × D` tensor (latent atmospheric state).
- Surface embedding goes through a separate residual MLP.
- Latent surface is concatenated with latent atmosphere → **(C_L + 1) × D** representation per patch location.

**Positional/Scale/Time Encodings:**
- Applied to the `(C_L + 1) × H/P × W/P` 3D tensor before it enters the backbone.
- All use Fourier expansion: `Emb(x) = [cos(2πx/λ_i), sin(2πx/λ_i)]` with log-spaced wavelengths.

### B.2 Multi-Scale 3D Swin Transformer U-Net Backbone (Detailed)

**Structure:**
```
Input (H/P × W/P × L)
    ↓ Enc Stage 1: 6 layers
    ↓ Patch Merging (halve spatial, double embed dim)
    ↓ Enc Stage 2: 10 layers
    ↓ Patch Merging
    ↓ Enc Stage 3: 8 layers
    ↓ Dec Stage 1: 8 layers
    ↓ Patch Splitting (double spatial, halve embed dim) + Skip Connection
    ↓ Dec Stage 2: 10 layers
    ↓ Patch Splitting + Skip Connection
    ↓ Dec Stage 3: 6 layers
Output (H/P × W/P × L)
```

**Key Design Choices:**
- 3D Swin Transformer layers with **local self-attention within windows** — analogous to local computations in numerical integration.
- Window size: **(2, 12, 6)** along (depth, width, height).
- Window shifting: every other layer shifts by **(1, 6, 3)** to enable cross-window information flow.
- Earth spherical topology: left/right edges of images wrap around during shifted attention.
- **res-post-norm** (from Swin v2) for stability.
- **No positional bias** in attention (unlike original Swin/Pangu) — replaced by encoder's Fourier encodings, making backbone resolution-independent.
- 48 layers total (3× deeper than Pangu-Weather's backbone), enabled by efficient encoding to just 3 latent levels.

### B.3 3D Perceiver Decoder (Detailed)

- Mirrors the encoder.
- 3 latent atmospheric levels → `C` output atmospheric levels via Perceiver cross-attention (uses sine/cosine pressure embeddings of output levels as queries).
- D-dimensional embeddings → `P × P` output patches via variable-specific linear layers (dynamically constructed per variable).
- Can output to **any set of pressure levels** and **any set of variables**.

### B.4 Fourier Encodings

All encodings use the form:
```
Emb(x) = [cos(2πx/λ_i), sin(2πx/λ_i)]  for i = 0, ..., D/2 - 1
```
where λ_i are log-spaced between λ_min and λ_max.

| Encoding | Input value `x` | λ_min | λ_max |
|---|---|---|---|
| Position (lat) | Average latitude of patch (first D/2 dims) | 0.01 | 720 |
| Position (lon) | Average longitude of patch (next D/2 dims) | 0.01 | 720 |
| Scale | Patch area in km² via `A = R²(sinφ₂ - sinφ₁)(θ₂ - θ₁)` | very small | Earth surface area |
| Pressure level | Pressure in hPa (e.g., 500) | 0.01 | 10,000 |
| Absolute time | Hours since Jan 1, 1970 | 1 hour | hours in a year |

### B.5 Data Normalization

- Every surface variable and every (variable, pressure level) pair is normalized independently:
  ```
  X_normalized = (X - center) / scale
  ```
- Centers = empirical means over ERA5 training data.
- Scales = empirical standard deviations over ERA5 training data.
- Same normalization statistics used for **all datasets** (not just ERA5).

### B.6 Extensions for 0.1° Weather Forecasting

Three modifications to fit the larger resolution in GPU memory:

1. **Patch size increase**: 4 → 10 (2.5× increase matches 0.25° → 0.1° resolution change). Backbone spatial resolution stays the same. Weights transferred via bilinear interpolation with magnitude scaling.
2. **Remove backbone layers**: middle stages go from (6,10,8)/(8,10,6) to (6,8,8)/(8,8,6) — 2 layers removed from each middle stage. Model is robust to this due to stochastic depth in pretraining.

### B.7 Extensions for Air Pollution Forecasting

Significant adaptations for the heterogeneous, sparse, skewed nature of air pollutant data:

1. **12-hour model**: Pretrained with Δt=12h (instead of 6h) so input spans t and t-24h, capturing diurnal cycles. Pretrained for 80.5k steps.
2. **Differencing**: For variables with clear diurnal cycles (NO, NO₂, SO₂, PMs), predict difference w.r.t. t-24h. For others (CO, O₃), predict difference w.r.t. t-12h. A learnable modulation factor allows the model to interpolate between direct and difference prediction:
   ```
   X̂_t = [Φ_mod(X_{t-1}, X_{t-2}) + 1] · X_a + Φ_pred(X_{t-1}, X_{t-2})
   ```
3. **Custom normalization**: center=0, scale = ½ × (spatial max averaged over time). Yields normalized values typically in [0, 2].
4. **Log-linear transformation** for concentration variables (large dynamic range):
   ```
   x_transformed = c₁·min(x, 2.5) + c₂·[log(max(x, 1e-4)) - log(1e-4)] / log(1e-4)
   ```
   Balances sensitivity to normal-magnitude and low-magnitude values.
5. **14 additional static variables**: time of day, day of week, day of year (as sin/cos pairs), plus monthly average anthropogenic emissions for NH₃, CO, NOx, SO₂ (in linear and log forms).
6. **Pressure-level-specific patch embeddings** for pollution variables (behavior varies dramatically by altitude).
7. **Separate 3D Perceiver decoder** for pollution variables (distinct from meteorological decoder).
8. **32-bit precision** for encoder/decoder computations (backbone stays bf16).
9. **Smaller patch size** (3 instead of 4) due to lower CAMS resolution (0.4°).

### B.8 Extensions for Wave Forecasting

1. **Angle-valued variables** (MWD, MDWW, MDTS, MWD1, MWD2): transformed to (sin, cos) pairs before encoding, with separate patch embeddings. Converted back via atan2 after decoding.
2. **Density channels for missing data**: each wave variable gets a companion channel (1=present, 0=absent). Missing values set to 0 after normalization. Model predicts density channels too (with sigmoid bounding). During autoregressive rollout, density > 0.5 → set to 1; density < 0.5 → set variable and density to 0.
3. **Additional static variables**: bathymetry and a HRES-WAM coverage mask.
4. **Additional layer normalization** on keys and queries of first level aggregation attention block for training stability.

### B.9 Model Hyperparameters Summary

**1.3B parameter model (main):**
- Embedding dim: 512 (doubles per stage)
- Attention head dim: 64 throughout
- Perceiver cross-attention heads: 16
- Backbone encoder layers: (6, 10, 8)
- Backbone decoder layers: (8, 10, 6)
- Decoder embedding dim: 1024 (due to skip connection concatenation)
- Latent pressure levels: 3

**Scaling experiment configurations:**

| Model Size | Enc Layers | Dec Layers | Embed Dim | Attn Head Dim | Perceiver Heads |
|---|---|---|---|---|---|
| 117M | (2, 6, 2) | (2, 6, 2) | 256 | 64 | 8 |
| 290M | (4, 8, 4) | (4, 8, 4) | 320 | 64 | 8 |
| 660M | (6, 8, 8) | (8, 8, 6) | 384 | 64 | 12 |
| 1.3B | (6, 10, 8) | (8, 10, 6) | 512 | 64 | 16 |

Validation performance improves by ~6% for every 10× increase in model size.

---

## Key Architectural Innovations Summary

1. **Perceiver-based encoding/decoding**: Compresses arbitrary numbers of pressure levels into a fixed small set of latent levels (C_L=3), enabling much deeper backbones within memory constraints and handling heterogeneous inputs.
2. **Variable-specific linear layers**: Dynamically constructed per-variable patch embeddings allow the same model to ingest/output different variable sets across datasets.
3. **Fourier position/scale/time encodings**: Replace fixed positional biases, making the model resolution-agnostic and able to handle multi-resolution data.
4. **3D Swin Transformer with spherical topology**: Local windowed attention with shifting provides efficient multi-scale simulation of physical dynamics on the sphere.
5. **LoRA + replay buffer for rollout fine-tuning**: Enables long autoregressive fine-tuning of a 1.3B model without memory explosion.
6. **Density channels**: Principled handling of spatially varying missing data (e.g., waves over land).
7. **Foundation model paradigm**: Pretrain once on diverse data → cheap fine-tuning for many downstream tasks (4-8 weeks per task vs. years for traditional systems).
