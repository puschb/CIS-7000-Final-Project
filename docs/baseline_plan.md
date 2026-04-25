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
script is better for reproducibility. **Closed form is also acceptable** —
since the head is a single linear layer, the optimal weights have an
analytical solution `(XᵀX + λI)⁻¹ Xᵀy` per variable per output-pixel position,
and accumulators for `XᵀX` and `Xᵀy` fit easily in memory (`D=256` →
`XᵀX` is 256×256 ≈ 0.5 MB). If you want to *prove* you've found the optimal
linear probe — useful for the report — solve it in closed form and verify
against the SGD result.

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
| Stage 1 script | `scripts/extract_surface_latents.py` (new) |
| Stage 2 script | `src/linear_probe.py` (new) — replaces the head/training portion of `src/baseline.py` |
| Configs (date splits, normalisation stats, mask) | `configs/baseline.yaml` (new) |
| Per-variable metrics + reference baselines | `results/linear_probe/metrics.json` |
| Trained head weights | `results/linear_probe/heads.pt` |
| Larger PVC (if caching) | `k8s/hydrology-features-pvc.yaml` (new, 100 Gi) |
| Stage-1 batch job | `k8s/extract-latents-job.yaml` (new) |
| Stage-2 batch job (or run locally) | `k8s/linear-probe-job.yaml` (new, CPU is fine) |

## Step-by-step build order (translates directly to code)

1. **Confirm tap point on a small model.** In an interactive pod or notebook,
   load `AuroraSmallPretrained`, `print(model.decoder)`, find the surface head
   linear for `2t`. Register a forward pre-hook, run a forward, capture the
   input. Reshape to `(B, Hp, Wp, D)`. Run `model.decoder.surf_heads['2t'](z)`
   on the captured tensor (or whatever the actual call is) and verify that the
   pixel-shuffled output equals `pred.surf_vars['2t']` to numerical precision.
   This single check pins down Tap A.
2. **Replace `src/baseline.py`'s head with the linear probe.** Drop the
   `SurfaceReadoutHead` (Conv→GELU→Conv→bilinear); replace with
   `nn.ModuleDict({name: nn.Linear(D, P*P) for name in target_vars})` plus a
   pixel-shuffle. Replace `_tokens_to_grid` with the surface-slab slicing that
   matches Tap A (or B).
3. **Add normalisation, masking, per-variable metrics** as described above.
   These are mechanical edits to `run_epoch` in `src/baseline.py`.
4. **Add asserts and tests.** Two assert lines at the top of the loop; one
   integration test that runs `baseline.py --small --train-limit 4
   --val-limit 2 --epochs 1` end-to-end.
5. **Run a single-pod, no-cache pilot** on AuroraSmall with 4–6 weeks of data
   to verify the loss curve is sensible and persistence is being beaten on at
   least one variable. ~1–2 hours.
6. **(Optional) Build the cache pipeline.** Implement
   `scripts/extract_surface_latents.py` as a thin wrapper around the same hook
   logic, writing one `.pt` per sample. Bump the PVC. Run as a Stage-1 job.
7. **(Optional) Replace the inline Aurora forward in Stage 2 with a
   `CachedFeatureDataset`.** Trivial — just yields `(latent, target)`
   directly from disk. Same head and loss code. Now training is CPU-only and
   fast.
8. **Final run on AuroraFull** for the headline number. Same code, larger
   `D=512`, possibly fewer cached samples or no cache.
9. **Report**: per-variable MAE/RMSE on the test split, alongside persistence
   and climatology, in `results/linear_probe/metrics.json` and as a small
   table in the report.

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

## Sources used

- [aurora.model.aurora — Aurora docs (module index)](https://microsoft.github.io/aurora/_modules/aurora/model/aurora.html)
- [Aurora API reference](https://microsoft.github.io/aurora/api.html)
- [Aurora fine-tuning guide](https://microsoft.github.io/aurora/finetuning.html)
- [aurora/aurora/model/aurora.py on GitHub](https://github.com/microsoft/aurora/blob/main/aurora/model/aurora.py)
- Aurora paper, Supplementary B.1–B.3 (`aurora.pdf` in this repo, pages ~14–16)
- This repo: `src/baseline.py`, `src/data.py`, `docs/aurora_regional_embeddings.md`, `k8s/aurora-data-pvc.yaml`
