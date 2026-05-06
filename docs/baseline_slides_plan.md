# Slide Plan — Domain Shift Quantification Baseline

Section sits between "Adaptation Strategy overview" and "Evaluation".
Audience is experienced and has already seen the Aurora architecture and
variable-selection story. Keep slides sparse.

## Slide 1 — Why a baseline at all?

**Title idea:** "The number our fine-tune has to beat"

**Bullets:**
- We extended Aurora to predict `{swvl1, stl1, sd}` — but did LoRA actually
  teach new physics, or was the answer already latent in the pretrained
  representation?
- Honest comparison isn't "from scratch" or persistence; it's:
  *with no parameter updates and no hydrology inputs, how much is there?*
- **Question we answer:** *"How much of `{swvl1, stl1, sd}` at `t+6h` is
  linearly decodable from Aurora's surface latent, given only atmospheric
  inputs?"*
- Whatever the fine-tune gains over this number = marginal value of
  (a) past hydrology in input and (b) backbone adaptation.

**Visual:** 3-bar conceptual chart with no numbers — Persistence <
Frozen-Aurora linear probe (this section) < LoRA + hydrology fine-tune.

**Speaker notes:** without this baseline, any improvement reported from
LoRA is uninterpretable. Foundation models capture dynamics in pretraining;
we want to quantify how much.

---

## Slide 2 — Architecture: freeze everything, attach three linear heads

**Title:** "Linear probe on Aurora's surface latent"

**Visual (centerpiece):** reuse the Aurora encoder → backbone → decoder
diagram from earlier slides with overlays:
- 🔒 lock icons on encoder, backbone, and the rest of the decoder
  (`requires_grad=False`).
- highlighted arrow from the decoder's `(B, Hp, Wp, D)` surface-latent
  stage labeled *"same tensor `2t`/`10u`/`10v`/`msl` heads consume."*
- three green boxes: `nn.Linear(D, P·P)` for `swvl1`, `stl1`, `sd` —
  labeled *"the only trainable parameters."*
- input batch box annotated in red: **"no hydrology in `surf_vars`"**.

**Bullets around the diagram:**
- New heads are exactly `Linear(D → P·P)` — same form as Aurora's
  pretrained surface heads. No MLP, no convs.
- Tap point captured via a forward pre-hook on the existing `2t` head;
  verified by `surf_head_2t(captured) == pred.surf_vars['2t']`.
- Trainable parameters: `3 × (D × P² + P²)` ≈ a few thousand. The probe
  cannot "cheat."

**Speaker notes:** the linear-only commitment is what makes the result
clean — we're measuring linear decodability, a well-defined property of
the representation, not "decodability with auxiliary capacity."

---

## Slide 3 — Pipeline, loss, reference floors

**Title:** "Two-stage pipeline: cache once, probe cheaply"

**Visual:** simple two-box flow:

```
Stage 1 (GPU, A100 80GB)              Stage 2 (CPU)
ERA5 (atmos only) → frozen Aurora  →  cached (latent, target) pairs
                       │                       │
                  (forward hook)               ▼
                       │                nn.Linear heads
                       ▼                + masked MAE loss
                .pt files on PVC
```

**Bullets:**
- **Stage 1 (one-time):** frozen Aurora forward over train/val/test;
  dump surface latent (bf16) + hydrology target (fp16) to PVC.
- **Stage 2 (cheap):** train three linear heads on cached features. CPU
  is enough; converges in minutes; iterate on normalization/mask freely.
- **Loss:** per-variable normalized MAE, masked to land via
  `static_vars["lsm"] > 0.5`. Report MAE/RMSE per variable in native units.
- **Reference floors:** persistence + seasonal climatology on the same
  test split. Probe must beat both.
- **Splits:** summer 2024 + 2025; Jun 1–Aug 1 train / Aug 1–16 val /
  Aug 16–Sep 1 test. ~2.8k training pairs at 6-hourly cadence.

**Speaker notes:** Stage 2 being so cheap is what makes this a baseline
and not its own research project — we are measuring a property of
Aurora's representation, not building a small model.

---

## Slide 4 — Mirror ablation: hiding hydrology *after* fine-tuning

**Title:** "What does the fine-tuned model actually rely on?"

**Setup:** once LoRA fine-tuning is complete, run inference with
`swvl1`, `stl1`, `sd` masked (density-channel = 0 everywhere) at every
input timestep, and measure prediction quality on the same test set.

**Bullets:**
- **Baseline (slides 2–3):** frozen Aurora, no hydro in, linear probe →
  *what's already in the pretrained latent.*
- **Full fine-tune (Ben's section):** LoRA + hydro in → upper bound.
- **This ablation:** fine-tuned model, hydro masked at inference →
  did the backbone *encode* hydrology dynamics into the latent, or did
  it just learn a passthrough from the input?
- Decomposition table:

  | Source of skill | Measured by |
  |---|---|
  | Already in pretraining | linear probe baseline |
  | LoRA taught backbone (no input lean) | fine-tune w/ hydro hidden |
  | Input access alone | gap between full and ablation |

- Interpretation: sharp degradation → fine-tune leans on input
  passthrough → fragile at long rollouts. Stays close to full →
  backbone genuinely internalized hydrology → robust.

**Visual:** same architecture diagram as Slide 2, with hydrology input
box crossed out / greyed at inference; question mark on the latent
("encoded? or shortcutted?").

**Speaker notes:** symmetric companion to the linear probe — that
measures decodability *without* fine-tuning; this measures whether
fine-tuning changed how the latent encodes hydrology *independently*
of input access.

---

## Compression option (3 slides, if time-pressed)

Drop Slide 1's framing into the intro of Slide 2; collapse Slides 2 + 3
visuals into one. Final order: (architecture + pipeline) → (metrics +
floors) → (mirror ablation). The mirror-ablation slide should not be
dropped — it's the most distinctive part of the section.
