# Linear Probe Baseline: Hydrology Decodability from Aurora's Surface Latent

## Question we are answering

> Without changing Aurora's encoder or backbone, how much of `{swvl1, stl1, sd}`
> at `t+6h` is **linearly decodable** from the surface latent Aurora produces
> from atmospheric inputs alone?

This is the strict, defensible baseline that the LoRA-with-hydrology run will be
compared against. The phrasing matters: this is a *linear probe*, not a
"learn-anything-you-can" head. We are measuring properties of Aurora's frozen
representation, not training an auxiliary model.

## What I validated against the codebase and aurora package

1. **Surface vs atmospheric branches in the decoder are separate.** The Aurora
   paper (Supplementary B.3) states: "The latent surface level is decoded
   directly. ... the linear layer creating the output patches is constructed
   dynamically by selecting the weights associated with each variable." That is,
   the surface side of the decoder is a single `Linear(D → P·P)` per surface
   variable, applied to a `D`-dim per-patch surface latent — exactly the head
   form you sketched.
2. **Surface latent shape.** The backbone output is a 3D latent grid of shape
   `(B, latent_levels · Hp · Wp, D)`. `latent_levels` is `4` for the small model
   (confirmed in `src/baseline.py:115` via `getattr(self.aurora.encoder,
   "latent_levels", 4)` and in `docs/aurora_regional_embeddings.md`). One of
   those `latent_levels` slabs is the surface latent — i.e. a `(B, Hp, Wp, D)`
   tensor after we slice the surface slab and reshape. With `H=721, W=1440,
   P=4`: `Hp=180, Wp=360`. `D=256` for small, `D=512` for the 1.3 B model.
3. **Existing surface heads are linear in `D`.** The pretrained model's `2t /
   10u / 10v / msl` heads each consume the same surface latent and apply a
   `Linear(D, P·P)` to produce a 4×4 patch. This is what the new heads have to
   mirror.
4. **Dataset already supports the right input/target split.** `ERA5Dataset`
   accepts separate `input_surf_vars` and `target_surf_vars` tuples
   (`src/data.py:131-156`). Set `input_surf_vars=BASE_SURF_VAR_NAMES` (no
   hydrology) and `target_surf_vars=("swvl1","stl1","sd")` and the rest is in
   place.
5. **`src/baseline.py` already wires up the freezing, hooking, and metrics
   loop**, but currently hooks the encoder/bottleneck and uses a conv readout +
   bilinear upsample. The plan below replaces those two choices with the
   stricter linear-probe form.
6. **Memory budget on 1×A100 80 GB is fine.** Frozen Aurora forward at full
   0.25° resolution is ~40 GB VRAM (per the README); no gradients flow back into
   Aurora, so the only extra memory is the new heads (a few MB).
7. **Caching is feasible only with a bigger PVC.** The current PVC is 20 GiB
   (`k8s/aurora-data-pvc.yaml:11`). One sample of cached surface latent is
   `Hp·Wp·D` floats: small + bf16 = 33 MB; full + bf16 = 66 MB. So 20 GiB holds
   ~600 small or ~310 full samples — only enough for a tiny pilot. A serious
   train/val/test cache wants 50–100 GiB. The plan below makes feature caching
   optional and supports both modes.
8. **Aurora's documented extension API exists and would also work.** Microsoft's
   fine-tuning docs describe constructing `AuroraPretrained(surf_vars=("2t",
   "10u","10v","msl","swvl1","stl1","sd"))` plus `load_checkpoint(strict=False)`
   to make Aurora itself instantiate the new decoder heads. To stay a *strict*
   linear probe under that API, you would also have to (a) zero-initialise the
   new entries in `model.encoder.surf_token_embeds.weights` and freeze them so
   no hydrology can leak through the input side, and (b) freeze every other
   parameter so only the new entries in the decoder's surf-head dictionary are
   trainable. This is mathematically equivalent to the external-head approach
   below, but it depends on knowing Aurora's exact internal attribute paths and
   on remembering to zero-and-freeze the encoder additions. The external-head
   path is preferred for this baseline because it is provably non-leaky by
   construction (no new encoder parameters exist at all) and does not depend on
   Aurora's internal naming conventions.

## The architecture, end to end

```
ERA5Dataset                            (already implemented)
  input_surf_vars = ("2t","10u","10v","msl")        ← no hydrology
  target_surf_vars = ("swvl1","stl1","sd")
       │
       ▼
   Batch(t-6h, t)                       Batch(t+6h)  (target only)
       │
       ▼
┌──────────────────────────────────────────────────┐
│ Aurora                       FROZEN              │
│   encoder    requires_grad = False               │
│   backbone   requires_grad = False               │
│   decoder    requires_grad = False               │
│   wrapped in torch.inference_mode() during fwd   │
└──────────────────┬───────────────────────────────┘
                   │
   tap: surface latent right before                 (B, Hp, Wp, D)
   Aurora's existing surf_heads
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│ NewSurfaceHeads  (the only trainable thing)      │
│   nn.ModuleDict({                                │
│     "swvl1": nn.Linear(D, P*P),                  │
│     "stl1":  nn.Linear(D, P*P),                  │
│     "sd":    nn.Linear(D, P*P),                  │
│   })                                             │
│   forward: (B, Hp, Wp, D) → (B, Hp, Wp, P*P)     │
│   then pixel_shuffle / reshape → (B, 1, H, W)    │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
   predicted swvl1, stl1, sd at t+6h on full grid
                   │
                   ▼
   masked, per-variable normalised MAE loss ─→ optimiser updates only NewSurfaceHeads
```

Three new linear projections, three new bias vectors, **nothing else** is
trainable. That is what makes this a linear probe and what makes the resulting
number a clean answer to the question.

## Where exactly to tap (and the safety net)

The "right" tap is the input to one of Aurora's existing surface heads — the
`(B, Hp, Wp, D)` surface latent. Two ways to get it, in order of cleanliness:

**Tap A (preferred, mirrors what real surf heads see).** Register a forward
*pre*-hook (or forward hook) on Aurora's existing surface variable head — the
`Linear(D, P·P)` for one of the pretrained surface vars (e.g. `2t`). The
positional input to that linear layer **is** the surface latent. We capture it
verbatim and route it into the new heads. Implementation note: the exact
attribute path inside `aurora.model.decoder` (e.g.
`model.decoder.surf_heads['2t']` vs `model.decoder.surf_head['2t']` vs a
`ParameterDict`) needs to be confirmed by `print(model.decoder)` once on a
running pod. Pick whichever one is the `Linear(D, P·P)` for `2t` and hook its
input.

**Tap B (fallback if Tap A's exact module path is awkward).** Register a hook
on `model.backbone` and capture the full latent grid. Reshape to
`(B, latent_levels, Hp, Wp, D)`, slice the surface slab. We need to confirm
which index is the surface slab — in the encoder it's concatenated last per the
paper ("This latent state of the surface is then concatenated with the latent
state of the atmosphere across the vertical dimension"), so `[:, -1]` is the
likely choice, but verify with a quick equivalence check: feed a batch through,
extract latent via Tap B, run it through one of the existing surface heads
(e.g. `model.decoder.surf_heads['2t']`), and confirm the output equals
`pred.surf_vars['2t']` exactly. If not, try other slab indices or transpose the
reshape.

Tap A is preferred because it requires no assumption about latent ordering.
Tap B is a guaranteed-correct fallback as long as we run the equivalence check.

Either tap requires no Aurora source modification — pure forward hooks.

**Mirror the existing surface head exactly.** When you do the equivalence
check above, also inspect the existing `2t` head's full structure: in
particular, whether it has `bias=True` or `bias=False`, and whether there is
any `LayerNorm` / activation between the surface latent and the linear
projection. Construct the new `nn.Linear(D, P*P)` heads with the same `bias`
setting and apply the same pre-norm/post-activation if present. The point of
"mirror exactly" is that any structural mismatch turns the result from "what
Aurora's representation linearly contains" into "what Aurora's representation
contains given a different head form", which weakens the comparison to the
LoRA run.

## Two-stage pipeline: cache then probe

The user's optimisation insight is correct and important. The new heads are
independent of Aurora at training time; we only need `(latent, target)` pairs.
So we run Aurora once and reuse the latents for every probe-training epoch.

### Stage 1: feature extraction (Aurora forward, write latents to disk)

```
for each (input_batch, target_batch) in ERA5Dataset:
    with torch.inference_mode():
        aurora(input_batch)               # populates the hooked tensor
    latent = captured_surface_latent      # (1, Hp, Wp, D), bf16
    targets = stack(swvl1, stl1, sd at t+6h on full grid, fp16)  # (3, H, W)
    write({"latent": latent, "targets": targets, "t": t},
          path=f"{out_dir}/{t.isoformat()}.pt")
```

Run as a one-off batch job. GPU only needed in this stage. Output is a
directory of small `.pt` files (one per timestamp), simpler than a single huge
tensor and easy to shard across train/val/test dates.

Disk costs (one sample):
- latent (bf16): `Hp·Wp·D·2` bytes — 33 MB (small `D=256`), 66 MB (full `D=512`)
- targets (fp16): `3·H·W·2` bytes — 6.2 MB
- total per sample: ~40 MB (small) or ~73 MB (full)

For a 1 000 train + 200 val + 200 test budget:
- small model: ~56 GB → need a ~75 GiB PVC
- full model: ~102 GB → need a ~150 GiB PVC

Action item: bump `aurora-data-pvc.yaml` to `100Gi` (small) or `150Gi` (full)
*for this experiment*, in a new dedicated PVC named `hydrology-features` so the
existing `aurora-data` PVC is untouched. If quota is the issue, fall back to a
shorter time range or run the no-cache mode below.

### Stage 1.5: no-cache fallback

If the bigger PVC isn't available, we can also run Stage 1 + Stage 2 inline in
a single GPU job: stream samples through Aurora and immediately update the
linear heads with the freshly produced latent. This is what `src/baseline.py`
currently does, modulo replacing its head with the linear probe. Slower per
epoch, no extra storage. Recommended for the initial pilot run.

### Stage 2: linear-probe training (CPU- or GPU-light, no Aurora)

```
heads = nn.ModuleDict({
    name: nn.Linear(D, P*P) for name in ("swvl1", "stl1", "sd")
})
opt = AdamW(heads.parameters(), lr=3e-4, weight_decay=1e-4)

for epoch in range(num_epochs):
    for sample in cached_features:
        z = sample["latent"]                     # (1, Hp, Wp, D)
        y_true = sample["targets"]               # (3, H, W)
        y_pred = []
        for name in ("swvl1","stl1","sd"):
            y_var = heads[name](z)               # (1, Hp, Wp, P*P)
            y_var = pixel_shuffle(y_var, P)      # (1, 1, H, W)
            y_pred.append(y_var.squeeze(1))
        y_pred = torch.cat(y_pred, dim=0)        # (3, H, W)
        y_pred_n = (y_pred - mu) / sigma         # per-variable normalisation
        y_true_n = (y_true - mu) / sigma
        loss = (mask * (y_pred_n - y_true_n).abs()).sum() / mask.sum()
        loss.backward(); opt.step(); opt.zero_grad()
```

This is the entire probe-training loop. It runs in seconds per epoch on a
laptop CPU and converges in tens of epochs. A notebook is fine; a small CLI
script is better for reproducibility.

### Closed-form solve (recommended for the headline number)

The head is a single linear layer, the same `Linear(D, P·P)` applied at every
spatial position. With `M = N · Hp · Wp` rows of `D`-dim latent features and
matching `P·P`-dim target patches, the ridge-regression solution
`W = (XᵀX + λI)⁻¹ XᵀY` of shape `(D, P·P)` is the *provably optimal* linear
probe. Both `XᵀX` (256×256 ≈ 0.5 MB) and `XᵀY` (256×16 ≈ 4 KB per variable)
are streaming-accumulable: walk the dataset once, accumulate the two
matrices, solve once.

Why this is worth doing instead of (or alongside) SGD:

- It removes any "did the optimizer converge?" question from the report.
  The number is the optimum, full stop.
- It's faster than SGD for this problem size (one pass over the data + one
  256×256 solve, vs. tens of epochs).
- It separates the *measurement* (a closed-form ridge regression) from the
  *training framework*, which makes the apples-to-apples comparison with the
  LoRA run rest only on the dataset, normalisation, and mask — not on the
  optimiser.

Implementation sketch:

```python
# Per variable v in {swvl1, stl1, sd}:
XtX = torch.zeros(D, D)
XtY = torch.zeros(D, P*P)
n   = 0

for sample in cached_features:
    z = sample["latent"].reshape(-1, D)              # (Hp*Wp, D)
    y = sample["targets"][v]                         # (H, W) → (Hp*Wp, P*P)
    y = patchify(y, P)                               # invert pixel_shuffle
    m = sample["land_mask"].reshape(-1)              # (Hp*Wp,)
    z = z[m]; y = y[m]
    XtX += z.T @ z
    XtY += z.T @ y
    n   += z.shape[0]

W = torch.linalg.solve(XtX + lam * torch.eye(D), XtY)   # (D, P*P)
```

Use SGD only as a sanity check that closed-form gives the same loss. They
must match to numerical precision; if they don't, there is a bug.

## Loss / metrics / fairness checklist

- **Per-variable target normalisation.** Compute `mu`, `sigma` for `swvl1`,
  `stl1`, `sd` once on the training timestamps and freeze. Train and report
  loss in normalised space; report MAE/RMSE in native units.
- **Land mask.** All three target variables are only physically defined over
  land. Use `static_vars["lsm"] > 0.5` and apply to both prediction and target
  in the loss and in every reported metric.
- **Per-variable metrics.** RMSE and MAE per variable, in native units.
  Aggregated metrics are misleading because the variable scales differ.
- **Reference baselines reported alongside.** Persistence (`x(t+6h) := x(t)`)
  and climatology (long-term mean per `(lat, lon, hour-of-year)`) on the same
  test timestamps. These are dirt cheap and bound the result from below; if
  the linear probe doesn't beat persistence, that itself is a finding.
- **Asserts to keep the experiment honest.**
  - `assert "swvl1" not in input_batch.surf_vars` (and similarly for `stl1`,
    `sd`) at the top of the training loop, so a future code change can't
    accidentally leak hydrology into the input.
  - `assert all(p.requires_grad is False for p in aurora.parameters())`.
  - `assert sum(p.numel() for p in heads.parameters() if p.requires_grad) <
    1e6` — guards against the head silently growing into a full network.

## Deliverables (what gets written, where)

| Artefact | Where |
|---|---|
| Tap-point spike notebook | `notebooks/probe_tap_point.ipynb` (new) |
| Stage 1 script (latent extraction) | `scripts/extract_surface_latents.py` (new) |
| Stage 2 script (closed-form ridge solve + metrics) | `src/linear_probe.py` (new) |
| Reference-baselines script (persistence, climatology) | `scripts/reference_baselines.py` (new) |
| Configs (date splits, normalisation stats, ridge λ, mask) | `configs/baseline.yaml` (new) |
| All four rows of the comparison | `results/linear_probe/metrics.json` |
| Trained head weights | `results/linear_probe/heads.pt` |
| Larger PVC (if caching) | `k8s/hydrology-features-pvc.yaml` (new, 100 Gi) |
| Stage-1 batch job | `k8s/extract-latents-job.yaml` (new) |
| Stage-2 batch job (or run locally) | `k8s/linear-probe-job.yaml` (new, CPU is fine) |

## Step-by-step build order (translates directly to code)

1. **Tap-point spike (≤ 1 hour, AuroraSmall on CPU or a small interactive
   pod).** In a notebook:
   - `model = AuroraSmallPretrained(); model.load_checkpoint(); model.eval()`
   - `print(model.decoder)` — find the surface head module for `2t`. Note
     whether it's keyed under `surf_heads`, `surf_head`, etc.; whether it's
     `nn.ModuleDict` of `Linear` or `nn.ParameterDict` of weight tensors;
     whether `bias=True/False`; whether there is a `LayerNorm` immediately
     before it.
   - Register a forward pre-hook on that module to capture its input. Run one
     forward pass.
   - Run the captured tensor back through the same module and pixel-shuffle
     it; assert the result matches `pred.surf_vars["2t"]` to fp32 precision.
   - Now you know exactly the (i) tap-point module path, (ii) latent shape,
     and (iii) head structure to mirror. **Do not write `src/linear_probe.py`
     until this notebook check passes.**
2. **Build the linear-probe head as a thin standalone module.** A single
   class `SurfacePatchHead(nn.Linear(D, P*P))` plus a `patches_to_image`
   helper that mirrors whatever pixel-shuffle / reshape Aurora's existing
   head does. Construct one per target variable.
3. **Inline pilot on AuroraSmall, no cache.** Edit `src/baseline.py`:
   - Drop `SurfaceReadoutHead`; use the new `SurfacePatchHead`.
   - Replace `_tokens_to_grid` with the surface-slab slicing that matches
     Tap A (or, if Tap A's path is awkward, Tap B with the equivalence
     check).
   - Add per-variable normalisation, land mask, per-variable metrics.
   - Add the three asserts (no hydrology in input, Aurora frozen, head
     parameter count under 1 M).
   - Run `python -m src.baseline --small --train-limit 8 --val-limit 4
     --epochs 1` end-to-end on real ERA5 in an interactive GPU pod.
   - Sanity check: loss decreases; held-out MAE for at least one of the three
     variables beats persistence.
4. **Build the cache pipeline once the pilot is green.**
   - `scripts/extract_surface_latents.py`: same hook logic as the pilot,
     writes `{"latent": (Hp,Wp,D) bf16, "targets": dict, "land_mask": ...,
     "t": ...}` per timestamp. One `.pt` file per sample.
   - `k8s/hydrology-features-pvc.yaml` (new, 100 GiB) and
     `k8s/extract-latents-job.yaml` (new). Submit one batch job per split.
5. **Stage-2 closed-form solver.** `src/linear_probe.py`:
   - `CachedFeatureDataset` yields `(latent, targets, land_mask)` from disk.
   - Per variable, accumulate `XᵀX` and `XᵀY` over the training split (with
     land mask), solve `(XᵀX + λI)⁻¹ XᵀY`, save `W` and `b` to
     `results/linear_probe/heads.pt`.
   - Sweep ridge `λ` on the val split, pick the winner, evaluate on the
     test split.
   - As a sanity check, also run the SGD loop from Stage 2 in the doc above
     and assert its final test loss equals the closed-form loss to within a
     small tolerance. They should match because the problem is convex and
     the closed-form is the optimum.
6. **Reference baselines on the same test split.** A small script that, for
   each test timestamp, computes (a) persistence prediction `x(t)` and
   (b) climatology prediction (precomputed long-term mean per `(lat, lon,
   hour-of-year)` from training data). Same masked, per-variable RMSE/MAE
   path; same JSON output schema. Numbers go into the same
   `results/linear_probe/metrics.json` so all four rows of the comparison
   table live in one file.
7. **Headline run on AuroraFull.** Same scripts, larger `D=512`. Re-extract
   latents (Stage 1), re-solve closed-form (Stage 2). Single A100 80GB job
   for Stage 1; CPU job for Stage 2.
8. **Report**: a four-row table — persistence, climatology, linear probe on
   AuroraSmall latent, linear probe on AuroraFull latent — with per-variable
   MAE and RMSE in native units on the test split. This is the deliverable
   the LoRA run will be compared against.

## Confirming this matches your specification

- ✅ Aurora encoder + backbone + decoder all frozen.
- ✅ Only new heads are trainable, registered for the new surface variables.
- ✅ New heads are `nn.Linear(D, P·P)` — exactly mirrors the form of the
  existing pretrained surface heads.
- ✅ New heads consume the surface latent (per-patch `D`-dim), not any
  atmospheric-level latent.
- ✅ Input batches contain only Aurora's pretrained variables; hydrology is
  excluded from input and asserted absent.
- ✅ Targets at `t+6h` cover all 7 surface vars in the dataset, but the loss is
  computed only over `{swvl1, stl1, sd}`, as you specified.
- ✅ Two-stage pipeline supported: cache once, probe-train cheaply afterwards.
  Inline single-stage mode also supported as a fallback.
- ✅ Feasibility on Nautilus: A100 80 GB is enough for the frozen forward;
  PVC needs to be enlarged only if we want to cache features at scale.

## What still has to be confirmed at the keyboard (and what doesn't)

Verified against the codebase and the Aurora paper while writing this plan:

- The decoder's surface side is a per-variable `Linear(D → P·P)` over a
  `D`-dim per-patch surface latent (Aurora paper, Supplementary B.3).
- Aurora exposes new-variable extension via the `surf_vars` constructor
  argument plus `model.load_checkpoint(strict=False)`, and the encoder-side
  patch embeddings live at `model.encoder.surf_token_embeds.weights` (Aurora
  fine-tuning docs). This is the "alternative" path described above; we are
  not using it for the probe but it confirms the decoder structure is
  variable-keyed in exactly the way our heads need to mirror.
- `ERA5Dataset` in `src/data.py` accepts `input_surf_vars` and
  `target_surf_vars` independently and already exposes `static_vars["lsm"]`
  for masking; no dataset changes are required for the linear probe.
- `src/baseline.py` already implements the freeze-Aurora + hook + readout
  pattern; the plan reuses its structure and replaces the head + the
  feature-extraction shape.

Cannot be confirmed in this sandbox (no aurora package installed, no
external network); to be checked in Step 1 of the build order:

- The exact attribute path of the surface heads inside
  `aurora.model.decoder` (most likely `model.decoder.surf_heads['2t']` —
  but verify with `print(model.decoder)` before writing code that depends on
  the name).
- Whether `bias=True` on those linear heads, and whether there is a
  `LayerNorm` immediately upstream of them. Mirror exactly what is there.
- Which slab index of the latent grid is the surface latent (verify by the
  equivalence check described in Tap B).

## Sources

- [Aurora fine-tuning guide (microsoft.github.io)](https://microsoft.github.io/aurora/finetuning.html)
- [Aurora API reference (microsoft.github.io)](https://microsoft.github.io/aurora/api.html)
- [Aurora module index (microsoft.github.io)](https://microsoft.github.io/aurora/_modules/aurora/model/aurora.html)
- [microsoft/aurora repo (github.com)](https://github.com/microsoft/aurora)
- Aurora paper, Supplementary B.1–B.3 (`aurora.pdf` in this repo, pages ~14–16)
- This repo: `src/baseline.py`, `src/data.py`, `docs/aurora_regional_embeddings.md`, `k8s/aurora-data-pvc.yaml`
