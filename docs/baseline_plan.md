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
5. **The implemented baseline is `src/linear_probe.py`.** It freezes Aurora,
   taps the existing surface-head latent, streams the train split once, and
   solves the linear heads in closed form. The older conv/bilinear pilot was
   removed to avoid confusing it with the strict linear-probe result.
6. **Memory budget on 1×A100 80 GB is fine.** Frozen Aurora forward at full
   0.25° resolution is ~40 GB VRAM (per the README); no gradients flow back into
   Aurora, so the only extra memory is the new heads (a few MB).
7. **Do not cache all latents for the final run.** We tried the cached-latent
   design and it was storage-heavy: the full train/val/test cache would exceed
   100 GB once targets and metadata are included. The final implementation
   avoids this by streaming latents through the ridge accumulator and writing
   only compact heads, metrics, spatial aggregates, and scatter samples.
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

## Final pipeline: stream latents, solve closed-form ridge

The original plan was a two-stage cache: first write every `(latent, target)`
pair to disk, then train heads from the cache. We abandoned that route because
it created tens to hundreds of GB of feature files and made the PVC the bottleneck.

The implemented baseline is a streaming closed-form linear probe:

```
for each train sample:
    with torch.inference_mode():
        aurora(input_batch)                    # populates the surface-head hook
    z = captured_surface_latent               # (1, Hp, Wp, D)
    y = patchified hydrology target at t+6h   # per variable, (Hp*Wp, P*P)
    apply land mask
    accumulate XtX += z.T @ z
    accumulate XtY += z.T @ y

for each variable:
    W = solve(XtX + lambda * I, XtY)
    save W, bias, metadata to heads.pt
```

Why this is the right final form:

- It gives the exact optimum for the linear head, so the report does not depend
  on SGD convergence or epoch count.
- It uses one pass over the train split and does not materialize the latent
  cache on disk.
- It keeps the experiment strict: Aurora is frozen, hydrology is absent from
  the input surface variables, and the only learned mapping is `Linear(D, P*P)`.
- It writes compact outputs: `heads.pt`, aggregate metrics, optional
  per-sample metrics, spatial aggregates, and scatter samples.

Current commands:

```bash
python -u -m src.linear_probe \
  --data-dir /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025 \
  --small \
  --device cuda \
  --output-dir /mnt/data/runs/linear_probe_streaming

python -u -m scripts.eval_linear_probe \
  --heads /mnt/data/runs/linear_probe_streaming/heads.pt \
  --data-dir /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025 \
  --output-dir /mnt/data/runs/linear_probe_streaming_eval \
  --splits val test \
  --small \
  --device cuda

python -u -m scripts.compute_reference_baselines \
  --data-dir /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025 \
  --output-base /mnt/data/runs/reference_baselines \
  --baselines persistence climatology \
  --splits val test
```

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
| Streaming linear-probe fit | `src/linear_probe.py` |
| Linear-probe evaluation pass | `scripts/eval_linear_probe.py` |
| Reference baselines | `scripts/compute_reference_baselines.py` |
| Linear-probe aggregate metrics | `results/linear_probe/streaming_fit/metrics.json` and `results/linear_probe/eval_20260506/eval_summary.json` |
| Per-sample, spatial, and scatter outputs | `results/linear_probe/eval_20260506/` |
| Reference-baseline outputs | `results/reference_baselines/{persistence,climatology}/` |
| Analysis notebook and figures | `notebooks/linear_probe_analysis.ipynb`, `results/linear_probe/figures/` |

## Implemented build order

1. **Find and verify the surface-head tap point.** We hook the input to the
   existing `2t` surface head, which is the same surface latent consumed by
   Aurora's pretrained surface heads.
2. **Fit the streaming probe.** `src.linear_probe` runs frozen Aurora over the
   Stage 1 train split, accumulates `X^T X` and `X^T Y`, solves the ridge system,
   and evaluates val/test aggregate metrics.
3. **Run the expanded evaluation pass.** `scripts.eval_linear_probe` writes
   per-sample metrics, spatial aggregate maps, and reservoir-subsampled scatter
   tensors for report figures.
4. **Run reference baselines.** `scripts.compute_reference_baselines` computes
   persistence and hour-of-day climatology using surface files only, avoiding
   unnecessary atmospheric IO.
5. **Analyze locally.** `notebooks.linear_probe_analysis.ipynb` produces the
   headline metric charts, spatial maps, temporal plots, STL1 hour-of-day
   baseline check, and predicted-vs-truth scatter plots.

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
- ✅ Streaming closed-form pipeline avoids the abandoned latent-cache storage blow-up.
- ✅ Feasibility on Nautilus: the A100/L40-class GPU is enough for the frozen forward;
  the PVC only stores compact results rather than full latent tensors.

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
- `src/linear_probe.py` implements the freeze-Aurora + surface-head hook +
  closed-form ridge solve used for the final reported baseline.

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
- This repo: `src/linear_probe.py`, `src/data.py`, `docs/aurora_regional_embeddings.md`, `k8s/era5-rechunked-pvc.yaml`
