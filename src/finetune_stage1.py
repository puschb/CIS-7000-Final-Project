"""Stage 1 short-lead-time fine-tuning: AuroraSmallPretrained + soil moisture variables.

Trains the full model (no LoRA) on 1–2 autoregressive steps with a weighted MAE loss
following the HRES-WAM fine-tuning pattern from the Aurora paper (Sections D.1, D.3).

All artefacts land under --run-dir/<run-name>/:
  checkpoints/  best.pt  +  step_NNNNN.pt every --save-every steps
  metrics/      metrics.jsonl  (one JSON record per logged step, append-only)
                summary.json   (columnar: {field: [val1, val2, ...]}, refreshed on each val)
  logs/         train.log      (mirrors stdout line-for-line)

The metrics.jsonl format uses one record per training step (phase=train) and one per
validation run (phase=val).  The summary.json has the same fields but as arrays, making
it easy to load into pandas for plotting:

    import pandas as pd, json
    with open("summary.json") as f: d = json.load(f)
    df_train = pd.DataFrame({k: v for k, v in d.items() if ...})

Usage:
    PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \\
    python -m src.finetune_stage1 \\
        --data-dir /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025 \\
        --run-dir /mnt/data/runs \\
        --epochs 6 --rollout-steps 1
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aurora import AuroraSmallPretrained, Batch, Metadata
from aurora.normalisation import locations, scales

from src.data import (
    DENSITY_VARS,
    PRESSURE_LEVELS,
    SOIL_SURF_VARS,
    collate_era5_batch,
    era5_worker_init_fn,
    make_era5_splits,
)

# ---------------------------------------------------------------------------
# Normalisation statistics (computed by scripts/compute_norm_stats.py on
# training split only — do not modify without re-running that job)
# ---------------------------------------------------------------------------
_NEW_VAR_NORM: dict[str, tuple[float, float]] = {
    "swvl1": (8.707704e-02, 1.428390e-01),
    "stl1":  (2.823009e+02, 2.123255e+01),
    "sd":    (1.151846e+00, 3.172587e+00),
}
_DENSITY_NORM: dict[str, tuple[float, float]] = {
    "swvl1_density": (0.0, 1.0),
    "stl1_density":  (0.0, 1.0),
    "sd_density":    (0.0, 1.0),
}

# ---------------------------------------------------------------------------
# Per-variable loss weights  (Aurora paper D.1 + HRES-WAM analogy)
# ---------------------------------------------------------------------------
SURF_WEIGHTS: dict[str, float] = {
    "2t":             3.0,
    "10u":            0.77,
    "10v":            0.66,
    "msl":            1.5,
    "swvl1":          1.5,
    "swvl1_density":  1.5,
    "stl1":           2.0,
    "stl1_density":   2.0,
    "sd":             1.0,
    "sd_density":     1.0,
}
ATMOS_WEIGHTS: dict[str, float] = {
    "z": 2.8,
    "t": 1.7,
    "u": 0.87,
    "v": 0.6,
    "q": 0.78,
}
ALPHA = 0.25   # surface component scale
BETA  = 1.0    # atmospheric component scale
GAMMA = 2.0    # ERA5 dataset scale


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def register_norm_stats() -> None:
    """Write new-variable norm stats into Aurora's global location/scale dicts."""
    for name, (loc, sc) in {**_NEW_VAR_NORM, **_DENSITY_NORM}.items():
        locations[name] = loc
        scales[name] = sc


def setup_run_dir(run_name: str, base_dir: str | Path) -> Path:
    run_dir = Path(base_dir) / run_name
    for sub in ("checkpoints", "metrics", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> logging.Logger:
    log = logging.getLogger("stage1")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "logs" / "train.log")
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log


# ---------------------------------------------------------------------------
# Metrics writer
# ---------------------------------------------------------------------------

class MetricsWriter:
    """Append-only JSONL log + periodic columnar summary.json.

    metrics.jsonl — one JSON object per line; never rewritten, safe through crashes.
    summary.json  — columnar {field: [v1, v2, ...]}, flushed on every val pass.
    """

    def __init__(self, metrics_dir: Path) -> None:
        self._jsonl = open(metrics_dir / "metrics.jsonl", "a", buffering=1)
        self._summary_path = metrics_dir / "summary.json"
        self._cols: dict[str, list] = {}

    def log(self, record: dict) -> None:
        record = {**record, "ts": datetime.utcnow().isoformat()}
        self._jsonl.write(json.dumps(record, default=float) + "\n")
        for k, v in record.items():
            if k != "ts":
                self._cols.setdefault(k, []).append(v)

    def flush_summary(self) -> None:
        with open(self._summary_path, "w") as f:
            json.dump(self._cols, f, default=float)

    def close(self) -> None:
        self.flush_summary()
        self._jsonl.close()


# ---------------------------------------------------------------------------
# Loss and per-variable metrics
# ---------------------------------------------------------------------------

def _trim_to_pred(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Trim target spatial dims to match prediction.

    Aurora internally crops latitude from 721 → 720 (patch size 4 requires
    dimensions divisible by 4). Trim the last two dims of t to p's shape so
    the loss comparison is valid.
    """
    return t[..., : p.shape[-2], : p.shape[-1]]


def weighted_mae_loss(
    pred: Batch,
    target: Batch,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted MAE following Aurora paper D.1.

    Returns
    -------
    loss : scalar Tensor
    per_var : dict mapping metric names to float values, including:
        mae_<surf_var>          — unweighted MAE per surface variable
                                  (soil vars: land points only)
        mae_<atmos_var>         — unweighted MAE averaged over all levels
        mae_<atmos_var>_<hPa>   — unweighted MAE at each pressure level
        loss_surf, loss_atmos, loss_total
    """
    per_var: dict[str, float] = {}
    surf_loss = torch.tensor(0.0, device=device)

    for var, p in pred.surf_vars.items():
        t = target.surf_vars[var]
        t = t[:, -1:] if t.shape[1] > 1 else t
        t = _trim_to_pred(p, t)

        if var in DENSITY_VARS:
            density_key = f"{var}_density"
            if density_key in target.surf_vars:
                d = target.surf_vars[density_key]
                d = d[:, -1:] if d.shape[1] > 1 else d
                d = _trim_to_pred(p, d)
                mask = (d >= 0.5).expand_as(p)
                mae = F.l1_loss(p[mask], t[mask]) if mask.any() else torch.tensor(0.0, device=device)
            else:
                mae = F.l1_loss(p, t)
        else:
            mae = F.l1_loss(p, t)

        per_var[f"mae_{var}"] = mae.item()
        surf_loss = surf_loss + SURF_WEIGHTS.get(var, 1.0) * mae

    atmos_loss = torch.tensor(0.0, device=device)

    for var, p in pred.atmos_vars.items():
        t = target.atmos_vars[var]
        t = t[:, -1:] if t.shape[1] > 1 else t
        t = _trim_to_pred(p, t)

        # Per pressure-level MAE (use only levels present in the prediction)
        n_levels = p.shape[2]
        for i, lvl in enumerate(PRESSURE_LEVELS[:n_levels]):
            per_var[f"mae_{var}_{lvl}"] = F.l1_loss(p[:, :, i], t[:, :, i]).item()

        # Average over all levels (used in the weighted loss)
        mae = F.l1_loss(p, t)
        per_var[f"mae_{var}"] = mae.item()
        atmos_loss = atmos_loss + ATMOS_WEIGHTS.get(var, 1.0) * mae

    V_S = len(pred.surf_vars)
    V_A = len(pred.atmos_vars)
    loss = GAMMA / (V_S + V_A) * (ALPHA * surf_loss + BETA * atmos_loss)

    per_var["loss_surf"]  = surf_loss.item()
    per_var["loss_atmos"] = atmos_loss.item()
    per_var["loss_total"] = loss.item()
    return loss, per_var


def _avg_per_var(records: list[dict[str, float]]) -> dict[str, float]:
    """Element-wise average across a list of per_var dicts."""
    return {k: sum(r[k] for r in records) / len(records) for k in records[0]}


# ---------------------------------------------------------------------------
# Rollout helper
# ---------------------------------------------------------------------------

def assemble_next_input(
    prev_input: Batch,
    pred: Batch,
    step_hours: int = 6,
) -> Batch:
    """Build (t, t+6h) input from (t-6h, t) and the prediction for t+6h.

    Concatenates the second history slot from prev_input with pred along dim 1.
    """
    surf = {
        k: torch.cat([prev_input.surf_vars[k][:, 1:2], pred.surf_vars[k]], dim=1)
        for k in pred.surf_vars
    }
    atmos = {
        k: torch.cat([prev_input.atmos_vars[k][:, 1:2], pred.atmos_vars[k]], dim=1)
        for k in pred.atmos_vars
    }
    new_time = tuple(t + timedelta(hours=step_hours) for t in prev_input.metadata.time)
    return Batch(
        surf_vars=surf,
        static_vars=prev_input.static_vars,
        atmos_vars=atmos,
        metadata=Metadata(
            lat=prev_input.metadata.lat,
            lon=prev_input.metadata.lon,
            time=new_time,
            atmos_levels=prev_input.metadata.atmos_levels,
        ),
    )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    tag: str | None = None,
) -> Path:
    state = {
        "step": step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    name = f"{tag}.pt" if tag else f"step_{step:05d}.pt"
    path = run_dir / "checkpoints" / name
    torch.save(state, path)
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    n_samples: int,
    device: torch.device,
) -> dict[str, float]:
    """Run validation on n_samples from val_loader.

    val_loader must use num_workers=0 so it doesn't compete with the
    train loader for /dev/shm.
    """
    model.eval()
    accum: list[dict[str, float]] = []

    for input_batch, targets in itertools.islice(val_loader, n_samples):
        input_batch = input_batch.to(device)
        current = input_batch
        step_records: list[dict] = []

        for target in targets:
            target_gpu = target.to(device)
            pred = model(current)
            _, per_var = weighted_mae_loss(pred, target_gpu, device)
            step_records.append(per_var)
            if len(targets) > 1:
                current = assemble_next_input(current, pred)

        accum.append(_avg_per_var(step_records))

    model.train()

    if not accum:
        return {}
    return _avg_per_var(accum)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aurora Stage 1 fine-tuning (soil moisture)")
    parser.add_argument("--data-dir",        nargs="+", required=True,        help="One or more dirs containing YYYY-MM-DDTHH-*.nc per-timestep files")
    parser.add_argument("--run-dir",         default="/mnt/data/runs",         help="Base directory on PVC for run artefacts")
    parser.add_argument("--run-name",        default=None,                     help="Sub-directory name (default: stage1_<timestamp>)")
    parser.add_argument("--epochs",          type=int,   default=6,            help="Number of full passes over the training set")
    parser.add_argument("--rollout-steps",   type=int,   default=1,  choices=[1, 2])
    parser.add_argument("--warmup-steps",    type=int,   default=500)
    parser.add_argument("--lr-base",         type=float, default=5e-5,  help="LR for pretrained weights")
    parser.add_argument("--lr-new-embed",    type=float, default=1e-3,  help="LR for new variable patch embeddings")
    parser.add_argument("--weight-decay",    type=float, default=5e-6)
    parser.add_argument("--grad-clip",       type=float, default=1.0)
    parser.add_argument("--val-every",       type=int,   default=300,  help="Validate every N training steps")
    parser.add_argument("--n-val-samples",   type=int,   default=50,   help="Number of val samples to evaluate per validation pass")
    parser.add_argument("--save-every",      type=int,   default=500,  help="Save periodic checkpoint every N steps")
    parser.add_argument("--num-workers",     type=int,   default=8)
    parser.add_argument("--prefetch-factor", type=int,   default=2)
    args = parser.parse_args()

    run_name = args.run_name or f"stage1_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir  = setup_run_dir(run_name, args.run_dir)
    log      = setup_logging(run_dir)
    writer   = MetricsWriter(run_dir / "metrics")

    log.info(f"Run: {run_name}")
    log.info(f"Run dir: {run_dir}")
    log.info(f"Config: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        import subprocess
        log.info(subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total",
                                          "--format=csv,noheader"]).decode().strip())

    # ── Datasets & loaders ──────────────────────────────────────────────────
    log.info("Building datasets (per_timestep layout)...")
    train_ds, val_ds, _ = make_era5_splits(
        data_dirs=args.data_dir,
        rollout_steps=args.rollout_steps,
        file_layout="per_timestep",
    )
    n_train = len(train_ds)
    total_steps = args.epochs * n_train
    log.info(f"Train: {n_train:,} samples | Val: {len(val_ds):,} samples")
    log.info(f"Epochs: {args.epochs} × {n_train:,} samples = {total_steps:,} total steps")

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        worker_init_fn=era5_worker_init_fn,
        collate_fn=collate_era5_batch,
        pin_memory=(device.type == "cuda"),
    )

    # Validation loader: num_workers=0 so it runs in the main process with zero
    # /dev/shm usage and never competes with the train loader's 8 workers.
    # shuffle=True gives a different random slice of val_ds each validation pass.
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_era5_batch,
    )
    log.info(f"Val loader ready (num_workers=0, {len(val_ds):,} samples, using {args.n_val_samples} per pass)")

    # ── Model ───────────────────────────────────────────────────────────────
    log.info("Registering normalisation stats and building model...")
    register_norm_stats()

    model = AuroraSmallPretrained(
        autocast=True,
        surf_vars=SOIL_SURF_VARS,
        use_lora=False,
    )
    model.load_checkpoint(strict=False)
    # Zero-init new-variable patch embeddings in the encoder (and decoder if present).
    # By default Aurora initialises them randomly, which perturbs existing-variable
    # predictions.  The paper uses zero-init for new wave variables (Section B.8).
    _new_surf_vars = [v for v in SOIL_SURF_VARS if v not in {"2t", "10u", "10v", "msl"}]
    for var in _new_surf_vars:
        if var in model.encoder.surf_token_embeds.weights:
            model.encoder.surf_token_embeds.weights[var].data.zero_()
            log.info(f"  Zero-initialised encoder.surf_token_embeds.weights[{var!r}]")
        if hasattr(model, "decoder") and hasattr(model.decoder, "surf_token_embeds"):
            if var in model.decoder.surf_token_embeds.weights:
                model.decoder.surf_token_embeds.weights[var].data.zero_()
                log.info(f"  Zero-initialised decoder.surf_token_embeds.weights[{var!r}]")
    model.configure_activation_checkpointing()
    model = model.to(device)
    model.train()

    n_total  = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Params total: {n_total:,}  trainable: {n_train:,}")

    # Split param groups: new soil variable embeddings get a higher LR
    _soil_embed_names = set(SOIL_SURF_VARS) - {"2t", "10u", "10v", "msl"}
    new_params, base_params = [], []
    for name, param in model.named_parameters():
        if any(v in name for v in _soil_embed_names):
            new_params.append(param)
        else:
            base_params.append(param)
    log.info(f"Base param tensors: {len(base_params)} | New-embed param tensors: {len(new_params)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr_base},
            {"params": new_params,  "lr": args.lr_new_embed},
        ],
        weight_decay=args.weight_decay,
    )

    # Linear warmup → cosine decay to 1e-5 (same shape for both param groups)
    min_ratio = 1e-5 / args.lr_base

    def _lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[_lr_lambda, _lr_lambda]
    )

    # ── Training loop ────────────────────────────────────────────────────────
    log.info(
        f"\nStarting training: {args.epochs} epochs = {total_steps:,} steps | "
        f"rollout_steps={args.rollout_steps} | warmup={args.warmup_steps} | "
        f"val_every={args.val_every} | train_samples={n_train:,}"
    )

    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    step = 0
    epoch = 0
    samples_since_epoch_start = 0
    # Rolling window for smoothed throughput (last 20 steps)
    _recent_step_times: list[float] = []

    while step < total_steps:
        # ── fetch batch (DataLoader / I/O time) ──────────────────────────────
        t_io_start = time.time()
        try:
            input_batch, targets = next(train_iter)
            samples_since_epoch_start += 1
        except StopIteration:
            epoch += 1
            log.info(f"── Epoch {epoch} complete (step {step}) ──────────────────────────────")
            train_iter = iter(train_loader)
            input_batch, targets = next(train_iter)
            samples_since_epoch_start = 1
        t_io = time.time() - t_io_start

        input_batch = input_batch.to(device)

        # ── forward / backward / optimizer (compute time) ────────────────────
        t_compute_start = time.time()
        optimizer.zero_grad()
        current = input_batch
        total_loss = torch.tensor(0.0, device=device)
        rollout_records: list[dict] = []

        for target in targets:
            target_gpu = target.to(device)
            pred = model(current)
            loss, per_var = weighted_mae_loss(pred, target_gpu, device)
            total_loss = total_loss + loss / len(targets)
            rollout_records.append(per_var)
            if len(targets) > 1:
                current = assemble_next_input(current, pred)

        # Single backward through all rollout steps so gradients flow through
        # every forward pass (true multi-step backprop, not the pushforward trick).
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()   # ensure compute time excludes async queuing
        t_compute = time.time() - t_compute_start

        step += 1
        step_time = t_io + t_compute

        # Smoothed throughput over last 20 steps
        _recent_step_times.append(step_time)
        if len(_recent_step_times) > 20:
            _recent_step_times.pop(0)
        samples_per_sec = 1.0 / (sum(_recent_step_times) / len(_recent_step_times))

        lrs = [pg["lr"] for pg in optimizer.param_groups]
        avg = _avg_per_var(rollout_records)

        writer.log({
            "phase":            "train",
            "step":             step,
            "epoch":            epoch,
            **avg,
            "grad_norm":        grad_norm.item(),
            "lr_base":          lrs[0],
            "lr_new_embed":     lrs[1],
            "step_time_s":      step_time,
            "io_time_s":        t_io,
            "compute_time_s":   t_compute,
            "samples_per_sec":  samples_per_sec,
        })

        if step % 10 == 0:
            pct = 100.0 * step / total_steps
            eta_s = (total_steps - step) * (sum(_recent_step_times) / len(_recent_step_times))
            eta_min = eta_s / 60
            log.info(
                f"[{pct:5.1f}%] step {step:>5}/{total_steps} ep{epoch} | "
                f"loss={avg['loss_total']:.4f} | "
                f"swvl1={avg.get('mae_swvl1', float('nan')):.4f} "
                f"stl1={avg.get('mae_stl1', float('nan')):.4f} "
                f"sd={avg.get('mae_sd', float('nan')):.4f} | "
                f"grad={grad_norm:.3f} lr={lrs[0]:.1e} | "
                f"io={t_io:.1f}s gpu={t_compute:.1f}s | "
                f"{samples_per_sec:.2f}samp/s ETA={eta_min:.0f}min"
            )

        # ── Validation ───────────────────────────────────────────────────────
        if step % args.val_every == 0:
            log.info(f"Running validation ({args.n_val_samples} samples, num_workers=0)...")
            val_metrics = validate(model, val_loader, args.n_val_samples, device)
            writer.log({
                "phase":     "val",
                "step":      step,
                "n_samples": args.n_val_samples,
                **val_metrics,
            })
            writer.flush_summary()

            val_loss = val_metrics.get("loss_total", float("inf"))
            log.info(
                f"  val loss={val_loss:.4f} | "
                f"mae_swvl1={val_metrics.get('mae_swvl1', float('nan')):.4f} | "
                f"mae_stl1={val_metrics.get('mae_stl1', float('nan')):.4f} | "
                f"mae_sd={val_metrics.get('mae_sd', float('nan')):.4f} | "
                f"mae_2t={val_metrics.get('mae_2t', float('nan')):.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt = save_checkpoint(run_dir, model, optimizer, scheduler, step, tag="best")
                log.info(f"  ✓ New best val loss {best_val_loss:.4f} → {ckpt}")

        # ── Periodic checkpoint ──────────────────────────────────────────────
        if step % args.save_every == 0:
            ckpt = save_checkpoint(run_dir, model, optimizer, scheduler, step)
            log.info(f"Saved periodic checkpoint: {ckpt}")

    # ── End of training ──────────────────────────────────────────────────────
    save_checkpoint(run_dir, model, optimizer, scheduler, step, tag="final")
    writer.close()
    log.info(f"\nTraining complete — {args.epochs} epochs ({total_steps:,} steps) | best val loss: {best_val_loss:.4f}")
    log.info(f"All artefacts at: {run_dir}")


if __name__ == "__main__":
    main()
