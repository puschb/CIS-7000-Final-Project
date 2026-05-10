"""Stage 2 rollout fine-tuning: LoRA + replay buffer + pushforward trick.

Loads the Stage 1 checkpoint, freezes all non-LoRA parameters, and trains the
LoRA weights (qkv + output projection of every Swin3D self-attention block)
on multi-day rollouts via the paper's protocol (Sections D.4 + B.8):

  - Each buffer entry holds (input_pair, target, lead_step, source_t1, current_t1).
  - One forward pass per training step (with grad). Predictions are detached
    and re-pushed to extend the rollout — gradient flows only through that
    single step (pushforward trick).
  - Curriculum: lead_step capped at --max-lead-early for the first
    --lead-warmup-steps, then opens to --max-lead-late.
  - Buffer is refreshed with fresh dataset samples every
    --dataset-sampling-period steps.
  - Predicted density channels are thresholded at 0.5 when fed back as input
    (Aurora wave-model protocol, Section B.8).

Validation runs full autoregressive rollouts (no buffer) on a small set of
val samples and reports latitude-weighted RMSE per (lead_h, var) — long
format, loadable in pandas as
``df.pivot_table(index="lead_h", columns="var", values="rmse_lat")``.

All artefacts land under --run-dir/<run-name>/:
  checkpoints/  best.pt + step_NNNNN.pt every --save-every steps
  metrics/      metrics.jsonl + summary.json
  logs/         train.log

Usage:
    PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \\
    python -m src.finetune_stage2 \\
        --data-dir /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025 \\
        --stage1-ckpt /mnt/data/runs/stage1_<run>/checkpoints/best.pt \\
        --run-dir /mnt/data/runs \\
        --total-steps 8000
"""

from __future__ import annotations

import argparse
import math
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

# Pre-built HDF5/netCDF4 wheels are NOT thread-safe — concurrent reads from
# threads in one process corrupt internal state and segfault inside
# ``H5O_attr_iterate_real``.  All thread-driven HDF5 reads must serialise on
# this lock.  DataLoader workers are forked processes with independent HDF5
# state, so they don't need the lock.
_HDF5_LOCK = threading.Lock()

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aurora import AuroraSmallPretrained, Batch, Metadata

from src.data import (
    DENSITY_VARS,
    PRESSURE_LEVELS,
    SOIL_SURF_VARS,
    ERA5Dataset,
    MultiRangeERA5Dataset,
    collate_era5_batch,
    era5_worker_init_fn,
    make_era5_splits,
)
from src import finetune_stage1 as _stage1
from src.finetune_stage1 import (
    MetricsWriter,
    _trim_to_pred,
    register_norm_stats,
    save_checkpoint,
    setup_logging,
    setup_run_dir,
    weighted_mae_loss,
)


# ---------------------------------------------------------------------------
# Latitude-weighted RMSE (Aurora paper §F.1, eqs. F14-F15)
# ---------------------------------------------------------------------------

def latitude_weights(lat: torch.Tensor) -> torch.Tensor:
    """Latitude-area weights normalised to mean 1 (paper eq. F14)."""
    cos_lat = torch.cos(torch.deg2rad(lat))
    return cos_lat / cos_lat.mean()


def lat_weighted_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_weights: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    """Latitude-weighted RMSE (paper eq. F15).

    pred / target: shape (..., H, W). lat_weights: shape (H,) normalised to
    mean 1. mask: optional bool tensor broadcastable to ``pred`` — points
    where ``mask`` is False are excluded and the weights are renormalised
    over the included region.
    """
    sq = (pred - target) ** 2
    w = lat_weights.view(*([1] * (sq.ndim - 2)), -1, 1).expand_as(sq)
    if mask is not None:
        w = w * mask.to(w.dtype)
    den = w.sum()
    if den.item() == 0:
        return float("nan")
    return torch.sqrt((sq * w).sum() / den).item()


# ---------------------------------------------------------------------------
# Density-aware autoregressive feedback (Aurora paper §B.8)
# ---------------------------------------------------------------------------

def assemble_next_input_with_density_threshold(
    prev_input: Batch,
    pred: Batch,
    step_hours: int = 6,
) -> Batch:
    """Build (t, t+6h) from (t-6h, t) and pred, thresholding density channels.

    The model predicts continuous density values; training inputs are binary
    0/1.  When feeding predictions back during autoregression, threshold
    predicted density at 0.5 and zero the parent variable wherever density
    falls below the threshold (paper §B.8). Without this the rollout drifts
    onto a distribution the model never saw.

    Aurora trims latitude 721 → 720 internally for patch-size divisibility,
    so ``pred`` is one row shorter than fresh DataLoader inputs.  Trim
    ``prev_input`` to match before concatenating along the time dim.  This
    is idempotent: extended entries are already at pred's spatial shape, so
    re-trimming is a no-op.
    """
    pred_h = next(iter(pred.surf_vars.values())).shape[-2]
    pred_w = next(iter(pred.surf_vars.values())).shape[-1]

    surf: dict[str, torch.Tensor] = {}
    for name, p in pred.surf_vars.items():
        prev_slot = prev_input.surf_vars[name][:, 1:2, :pred_h, :pred_w]
        pred_slot = p

        base = name[: -len("_density")] if name.endswith("_density") else name
        if base in DENSITY_VARS and f"{base}_density" in pred.surf_vars:
            mask = (pred.surf_vars[f"{base}_density"] >= 0.5).to(pred_slot.dtype)
            if name.endswith("_density"):
                pred_slot = mask
            else:
                pred_slot = pred_slot * mask

        surf[name] = torch.cat([prev_slot, pred_slot], dim=1)

    atmos = {
        k: torch.cat(
            [prev_input.atmos_vars[k][:, 1:2, :, :pred_h, :pred_w], pred.atmos_vars[k]],
            dim=1,
        )
        for k in pred.atmos_vars
    }
    static = {k: v[:pred_h, :pred_w] for k, v in prev_input.static_vars.items()}
    new_time = tuple(t + timedelta(hours=step_hours) for t in prev_input.metadata.time)
    return Batch(
        surf_vars=surf,
        static_vars=static,
        atmos_vars=atmos,
        metadata=Metadata(
            lat=prev_input.metadata.lat[:pred_h],
            lon=prev_input.metadata.lon[:pred_w],
            time=new_time,
            atmos_levels=prev_input.metadata.atmos_levels,
        ),
    )


# ---------------------------------------------------------------------------
# Batch utilities — escape /dev/shm, move to CPU pageable RAM
# ---------------------------------------------------------------------------

def _clone_dict(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in d.items()}


def clone_batch_to_cpu(batch: Batch) -> Batch:
    """Clone a Batch into pageable CPU RAM.

    DataLoader workers serialise tensors via /dev/shm; storing those tensors
    directly in the replay buffer would pin shm allocations and exhaust the
    24 GB cap.  ``clone()`` after ``.cpu()`` copies into pageable RAM, after
    which the worker's shm slot can be reclaimed.
    """
    return Batch(
        surf_vars=_clone_dict(batch.surf_vars),
        static_vars=_clone_dict(batch.static_vars),
        atmos_vars=_clone_dict(batch.atmos_vars),
        metadata=Metadata(
            lat=batch.metadata.lat.detach().cpu().clone(),
            lon=batch.metadata.lon.detach().cpu().clone(),
            time=batch.metadata.time,
            atmos_levels=batch.metadata.atmos_levels,
        ),
    )


def find_subdataset_for_time(
    ds: MultiRangeERA5Dataset, dt: datetime
) -> ERA5Dataset | None:
    """Return the inner per_timestep ERA5Dataset that holds files for *dt*."""
    for sub in ds.datasets:
        if sub.file_layout != "per_timestep":
            continue
        if dt in sub.surf_paths_by_time and dt in sub.atmos_paths_by_time:
            return sub
    return None


def load_timestep_as_target_batch(
    ds: MultiRangeERA5Dataset,
    dt: datetime,
) -> Batch:
    """Load a single timestamp as a target Batch (B=1, T=1, H, W).

    The actual NetCDF/HDF5 reads are serialised on ``_HDF5_LOCK`` because
    pre-built HDF5 wheels are not thread-safe; concurrent reads segfault
    inside attribute iteration.  Threads still parallelise non-HDF5 work
    (tensor allocation, slicing, Batch construction).
    """
    sub = find_subdataset_for_time(ds, dt)
    if sub is None:
        raise FileNotFoundError(f"No per-timestep files cover {dt}")
    with _HDF5_LOCK:
        surf, atmos = sub.load_timestep(dt)
    return Batch(
        surf_vars={k: v[None, None] for k, v in surf.items()},
        static_vars=sub.static_vars,
        atmos_vars={k: v[None, None] for k, v in atmos.items()},
        metadata=Metadata(
            lat=sub.lat,
            lon=sub.lon,
            time=(dt,),
            atmos_levels=PRESSURE_LEVELS,
        ),
    )


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class BufferEntry:
    """One rollout-chain state held in CPU pageable RAM.

    ``target`` may be a Future — the ground-truth NetCDF read is dispatched
    to a thread pool when the entry is pushed and resolved when it is
    sampled.  This hides ~5 s CephFS reads behind GPU compute.
    """
    input: Batch
    target: Union[Batch, Future]
    lead_step: int
    source_t1: datetime
    current_t1: datetime

    def resolve_target(self) -> Batch:
        if isinstance(self.target, Future):
            self.target = self.target.result()
        return self.target


class ReplayBuffer:
    """Random-access buffer with FIFO eviction at capacity."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries: list[BufferEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def push(self, entry: BufferEntry) -> None:
        if len(self.entries) >= self.capacity:
            self.entries.pop(0)
        self.entries.append(entry)

    def sample_pop(self) -> BufferEntry:
        idx = random.randrange(len(self.entries))
        return self.entries.pop(idx)


# ---------------------------------------------------------------------------
# Per-(lead, var) metrics
# ---------------------------------------------------------------------------

def compute_per_var_lead_metrics(
    pred: Batch,
    target: Batch,
    lat_weights: torch.Tensor,
    lead_h: int,
) -> list[dict]:
    """Return long-format metric records for one rollout step.

    One record per surface variable (with land-only masking for density-bearing
    vars) and one per (atmospheric variable, pressure level).  Each record:
    ``{lead_h, var, rmse_lat, mae}``.
    """
    out: list[dict] = []

    for name, p in pred.surf_vars.items():
        t = target.surf_vars[name]
        t = t[:, -1:] if t.shape[1] > 1 else t
        t = _trim_to_pred(p, t)

        mask = None
        if name in DENSITY_VARS:
            density_key = f"{name}_density"
            if density_key in target.surf_vars:
                d = target.surf_vars[density_key]
                d = d[:, -1:] if d.shape[1] > 1 else d
                d = _trim_to_pred(p, d)
                mask = d >= 0.5

        rmse = lat_weighted_rmse(p, t, lat_weights, mask=mask)
        if mask is not None:
            mae = F.l1_loss(p[mask], t[mask]).item() if mask.any() else float("nan")
        else:
            mae = F.l1_loss(p, t).item()

        out.append({"lead_h": lead_h, "var": name, "rmse_lat": rmse, "mae": mae})

    for name, p in pred.atmos_vars.items():
        t = target.atmos_vars[name]
        t = t[:, -1:] if t.shape[1] > 1 else t
        t = _trim_to_pred(p, t)
        n_levels = p.shape[2]
        for i, lvl in enumerate(PRESSURE_LEVELS[:n_levels]):
            rmse = lat_weighted_rmse(p[:, :, i], t[:, :, i], lat_weights)
            mae = F.l1_loss(p[:, :, i], t[:, :, i]).item()
            out.append({
                "lead_h": lead_h,
                "var": f"{name}_{lvl}",
                "rmse_lat": rmse,
                "mae": mae,
            })

    return out


def aggregate_records(per_sample: list[list[dict]]) -> list[dict]:
    """Average per-sample (lead, var) records across samples."""
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for sample_rec in per_sample:
        for r in sample_rec:
            grouped[(r["lead_h"], r["var"])].append(r)

    def _avg(vals: list[float]) -> float:
        clean = [v for v in vals if not math.isnan(v)]
        return sum(clean) / len(clean) if clean else float("nan")

    out: list[dict] = []
    for (lead_h, var), recs in sorted(grouped.items()):
        out.append({
            "lead_h": lead_h,
            "var": var,
            "rmse_lat": _avg([r["rmse_lat"] for r in recs]),
            "mae": _avg([r["mae"] for r in recs]),
            "n_samples": len(recs),
        })
    return out


# ---------------------------------------------------------------------------
# Validation — full autoregressive rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_rollout(
    model: torch.nn.Module,
    val_ds: MultiRangeERA5Dataset,
    n_samples: int,
    max_lead: int,
    device: torch.device,
    lat_weights: torch.Tensor,
    num_workers: int,
    prefetch_factor: int,
) -> list[dict]:
    """Run no_grad rollouts of length ``max_lead`` for ``n_samples`` initial
    conditions sampled from ``val_ds``.  Returns long-format records, one per
    (lead_h, var) tuple, averaged across samples.

    The val DataLoader provides (input, [gt_+6h]) per sample; remaining
    ground-truths (lead 2..max_lead) are loaded in parallel via a thread pool
    while the GPU rolls out the previous sample.
    """
    model.eval()
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        worker_init_fn=era5_worker_init_fn,
        collate_fn=collate_era5_batch,
    )
    val_iter = iter(val_loader)
    per_sample_records: list[list[dict]] = []

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(n_samples):
                try:
                    inp, [gt1] = next(val_iter)
                except StopIteration:
                    break

                t1 = inp.metadata.time[0]
                # Submit parallel GT loads for leads 2..max_lead.
                gt_futures: list[Future] = []
                for k in range(2, max_lead + 1):
                    tk = t1 + timedelta(hours=6 * k)
                    gt_futures.append(executor.submit(
                        load_timestep_as_target_batch, val_ds, tk
                    ))

                current = inp.to(device)
                sample_rec: list[dict] = []
                for k in range(1, max_lead + 1):
                    gt_k = gt1 if k == 1 else gt_futures[k - 2].result()
                    gt_gpu = gt_k.to(device)
                    pred = model(current)
                    sample_rec.extend(
                        compute_per_var_lead_metrics(pred, gt_gpu, lat_weights, lead_h=6 * k)
                    )
                    if k < max_lead:
                        current = assemble_next_input_with_density_threshold(current, pred)
                    del gt_gpu

                per_sample_records.append(sample_rec)
                del current
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        del val_iter
        del val_loader

    model.train()
    return aggregate_records(per_sample_records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_fresh(train_iter, train_loader):
    """Pull one fresh sample from the DataLoader, restarting if exhausted."""
    try:
        return next(train_iter), train_iter
    except StopIteration:
        train_iter = iter(train_loader)
        return next(train_iter), train_iter


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurora Stage 2 rollout fine-tuning")
    parser.add_argument("--data-dir", nargs="+", required=True)
    parser.add_argument("--stage1-ckpt", required=True, help="Path to Stage 1 .pt")
    parser.add_argument("--run-dir", default="/mnt/data/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--total-steps",            type=int,   default=8000)
    parser.add_argument("--buffer-size",            type=int,   default=30)
    parser.add_argument("--dataset-sampling-period", type=int,  default=10)
    parser.add_argument("--lead-warmup-steps",      type=int,   default=1000)
    parser.add_argument("--max-lead-early",         type=int,   default=6,
                        help="Max rollout depth (in 6h steps) for first --lead-warmup-steps")
    parser.add_argument("--max-lead-late",          type=int,   default=12,
                        help="Max rollout depth (in 6h steps) after --lead-warmup-steps; "
                             "also the validation rollout length")
    parser.add_argument("--lr",                     type=float, default=5e-5)
    parser.add_argument("--weight-decay",           type=float, default=0.0)
    parser.add_argument("--grad-clip",              type=float, default=1.0)
    parser.add_argument("--val-every",              type=int,   default=400)
    parser.add_argument("--n-val-samples",          type=int,   default=20)
    parser.add_argument("--save-every",             type=int,   default=1000)
    parser.add_argument("--num-workers",            type=int,   default=6)
    parser.add_argument("--num-val-workers",        type=int,   default=6)
    parser.add_argument("--prefetch-factor",        type=int,   default=2)
    parser.add_argument("--gt-prefetch-workers",    type=int,   default=4)
    parser.add_argument("--swvl1-weight", type=float, default=None,
                        help="Surface weight for swvl1 (and swvl1_density). "
                             "Default: keep stage1 value (1.5).")
    parser.add_argument("--stl1-weight",  type=float, default=None,
                        help="Surface weight for stl1 (and stl1_density). "
                             "Default: keep stage1 value (2.0).")
    parser.add_argument("--sd-weight",    type=float, default=None,
                        help="Surface weight for sd (and sd_density). "
                             "Default: keep stage1 value (1.0).")
    args = parser.parse_args()

    # Override per-variable surface weights for the new soil/snow variables.
    # `weighted_mae_loss` reads SURF_WEIGHTS from `_stage1`'s module globals at
    # call time, so mutating the dict here applies to every loss evaluation
    # without touching stage 1's source. Density channels track their parent
    # variable's weight (per the plan doc).
    overrides = []
    if args.swvl1_weight is not None:
        _stage1.SURF_WEIGHTS["swvl1"] = args.swvl1_weight
        _stage1.SURF_WEIGHTS["swvl1_density"] = args.swvl1_weight
        overrides.append(f"swvl1={args.swvl1_weight}")
    if args.stl1_weight is not None:
        _stage1.SURF_WEIGHTS["stl1"] = args.stl1_weight
        _stage1.SURF_WEIGHTS["stl1_density"] = args.stl1_weight
        overrides.append(f"stl1={args.stl1_weight}")
    if args.sd_weight is not None:
        _stage1.SURF_WEIGHTS["sd"] = args.sd_weight
        _stage1.SURF_WEIGHTS["sd_density"] = args.sd_weight
        overrides.append(f"sd={args.sd_weight}")

    run_name = args.run_name or f"stage2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = setup_run_dir(run_name, args.run_dir)
    log = setup_logging(run_dir)
    writer = MetricsWriter(run_dir / "metrics")

    log.info(f"Run: {run_name}")
    log.info(f"Run dir: {run_dir}")
    log.info(f"Config: {vars(args)}")
    if overrides:
        log.info(f"Surface weight overrides: {', '.join(overrides)}")
    log.info(f"Effective SURF_WEIGHTS: {_stage1.SURF_WEIGHTS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        import subprocess
        log.info(subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total",
            "--format=csv,noheader",
        ]).decode().strip())

    # ── Datasets ───────────────────────────────────────────────────────────
    log.info("Building datasets (per_timestep, rollout_steps=1)...")
    train_ds, val_ds, _ = make_era5_splits(
        data_dirs=args.data_dir,
        rollout_steps=1,
        file_layout="per_timestep",
    )
    log.info(f"Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")

    # Aurora trims latitude 721 → 720 for patch divisibility; build weights
    # against the trimmed grid so they line up with prediction tensors.
    lat_full = train_ds.datasets[0].lat
    lat_weights_trimmed = latitude_weights(lat_full[:720]).to(device)

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
    train_iter = iter(train_loader)

    # ── Model ──────────────────────────────────────────────────────────────
    log.info("Registering normalisation stats and building model (use_lora=True)...")
    register_norm_stats()
    model = AuroraSmallPretrained(
        autocast=True,
        surf_vars=SOIL_SURF_VARS,
        use_lora=True,
    )

    log.info(f"Loading Stage 1 checkpoint: {args.stage1_ckpt}")
    state = torch.load(args.stage1_ckpt, map_location="cpu")
    state_dict = state.get("model", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    n_lora_missing  = sum(1 for k in missing if "lora" in k.lower())
    n_other_missing = len(missing) - n_lora_missing
    log.info(
        f"Stage 1 ckpt: {len(state_dict):,} keys | "
        f"missing: {len(missing)} ({n_lora_missing} LoRA, {n_other_missing} other) | "
        f"unexpected: {len(unexpected)}"
    )
    if n_other_missing:
        non_lora = [k for k in missing if "lora" not in k.lower()][:10]
        log.warning(f"Non-LoRA missing keys (first 10): {non_lora}")
    if unexpected:
        log.warning(f"Unexpected keys (first 10): {unexpected[:10]}")

    model.configure_activation_checkpointing()
    model = model.to(device)

    # Freeze everything except LoRA.  The library injects LoRA only at
    # WindowAttention.{lora_qkv, lora_proj} inside Swin3D backbone blocks
    # — matching paper §D.4 ("all linear layers in self-attention of the
    # backbone").  No MLP / encoder / decoder / patch-embedding LoRA.
    n_total = n_trainable = 0
    for name, p in model.named_parameters():
        is_lora = "lora" in name.lower()
        p.requires_grad = is_lora
        n_total += p.numel()
        if is_lora:
            n_trainable += p.numel()
    log.info(f"Params total: {n_total:,} | trainable (LoRA only): {n_trainable:,}")
    if n_trainable == 0:
        raise RuntimeError("No LoRA parameters found — refusing to run with frozen model.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # No-op scheduler so checkpoint format matches Stage 1.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    model.train()

    # ── Replay buffer warm-up ─────────────────────────────────────────────
    log.info(f"Warming up replay buffer to {args.buffer_size} fresh samples...")
    buffer = ReplayBuffer(capacity=args.buffer_size)
    for _ in range(args.buffer_size):
        (inp, [tgt]), train_iter = _load_fresh(train_iter, train_loader)
        t1 = inp.metadata.time[0]
        buffer.push(BufferEntry(
            input=clone_batch_to_cpu(inp),
            target=clone_batch_to_cpu(tgt),
            lead_step=0,
            source_t1=t1,
            current_t1=t1,
        ))
    log.info(f"Replay buffer ready ({len(buffer)} entries).")

    gt_executor = ThreadPoolExecutor(max_workers=args.gt_prefetch_workers)

    # ── Training loop ─────────────────────────────────────────────────────
    log.info(
        f"\nStarting Stage 2 training: {args.total_steps:,} steps | "
        f"buffer={args.buffer_size} | sampling_period={args.dataset_sampling_period} | "
        f"lead_curriculum={args.max_lead_early} → {args.max_lead_late} after step "
        f"{args.lead_warmup_steps} | val every {args.val_every} ({args.n_val_samples} samples × {args.max_lead_late} steps)"
    )

    best_val_rmse = float("inf")
    _recent_step_times: list[float] = []
    step = 0

    while step < args.total_steps:
        max_lead = (
            args.max_lead_early if step < args.lead_warmup_steps else args.max_lead_late
        )
        t_start = time.time()

        # 1. Sample a buffer entry (blocking on its GT future if not ready).
        entry = buffer.sample_pop()
        target_batch = entry.resolve_target()

        # 2. Forward + backward (gradients only through this step).
        input_gpu = entry.input.to(device)
        target_gpu = target_batch.to(device)

        optimizer.zero_grad()
        pred = model(input_gpu)
        loss, per_var = weighted_mae_loss(pred, target_gpu, device)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip
        )
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        # 3. Extend chain (no_grad, density thresholded), OR replace with a
        #    fresh dataset sample when extending isn't possible. Either way
        #    the buffer stays at capacity by construction. A naive drop-on-cap
        #    policy drains the buffer at rate 1/max_lead per step and empties
        #    it; replacing on drop keeps it stable.
        new_lead = entry.lead_step + 1
        new_lead_pushed = -1
        extend_ok = False
        if new_lead < max_lead:
            new_current_t1 = entry.current_t1 + timedelta(hours=6)
            new_target_time = new_current_t1 + timedelta(hours=6)
            if find_subdataset_for_time(train_ds, new_target_time) is not None:
                with torch.no_grad():
                    next_input = assemble_next_input_with_density_threshold(input_gpu, pred)
                gt_future = gt_executor.submit(
                    load_timestep_as_target_batch, train_ds, new_target_time,
                )
                buffer.push(BufferEntry(
                    input=clone_batch_to_cpu(next_input),
                    target=gt_future,
                    lead_step=new_lead,
                    source_t1=entry.source_t1,
                    current_t1=new_current_t1,
                ))
                new_lead_pushed = new_lead
                extend_ok = True
                del next_input

        if not extend_ok:
            (fresh_inp, [fresh_tgt]), train_iter = _load_fresh(train_iter, train_loader)
            ft1 = fresh_inp.metadata.time[0]
            buffer.push(BufferEntry(
                input=clone_batch_to_cpu(fresh_inp),
                target=clone_batch_to_cpu(fresh_tgt),
                lead_step=0,
                source_t1=ft1,
                current_t1=ft1,
            ))

        del input_gpu, target_gpu, pred

        # 4. Periodic refresh with a fresh dataset sample.
        if step > 0 and step % args.dataset_sampling_period == 0:
            (fresh_inp, [fresh_tgt]), train_iter = _load_fresh(train_iter, train_loader)
            ft1 = fresh_inp.metadata.time[0]
            buffer.push(BufferEntry(
                input=clone_batch_to_cpu(fresh_inp),
                target=clone_batch_to_cpu(fresh_tgt),
                lead_step=0,
                source_t1=ft1,
                current_t1=ft1,
            ))

        step += 1
        step_time = time.time() - t_start
        _recent_step_times.append(step_time)
        if len(_recent_step_times) > 20:
            _recent_step_times.pop(0)
        samples_per_sec = 1.0 / (sum(_recent_step_times) / len(_recent_step_times))

        writer.log({
            "phase": "train",
            "step": step,
            **per_var,
            "grad_norm": grad_norm.item(),
            "lr": optimizer.param_groups[0]["lr"],
            "step_time_s": step_time,
            "samples_per_sec": samples_per_sec,
            "buffer_size": len(buffer),
            "entry_lead_step": entry.lead_step,
            "new_lead_pushed": new_lead_pushed,
            "max_lead_curr": max_lead,
        })

        if step % 10 == 0:
            pct = 100.0 * step / args.total_steps
            eta_min = (
                (args.total_steps - step)
                * (sum(_recent_step_times) / len(_recent_step_times))
                / 60
            )
            log.info(
                f"[{pct:5.1f}%] step {step:>5}/{args.total_steps} | "
                f"loss={per_var['loss_total']:.4f} | "
                f"swvl1={per_var.get('mae_swvl1', float('nan')):.4f} "
                f"stl1={per_var.get('mae_stl1', float('nan')):.4f} "
                f"sd={per_var.get('mae_sd', float('nan')):.4f} | "
                f"buf={len(buffer):>2} lead={entry.lead_step}→{new_lead_pushed} "
                f"(cap={max_lead}) | grad={grad_norm:.3f} | "
                f"{samples_per_sec:.2f}samp/s ETA={eta_min:.0f}min"
            )

        # ── Validation ──────────────────────────────────────────────────
        if step % args.val_every == 0:
            log.info(
                f"Running validation rollout: {args.n_val_samples} samples × "
                f"{args.max_lead_late} steps ({args.num_val_workers} workers)..."
            )
            t_val = time.time()

            # Shut down the buffer GT executor before val. validate_rollout
            # forks a fresh val DataLoader with num_val_workers; if any
            # gt_executor thread is alive at fork time, the val workers
            # inherit Python/HDF5 lock state (the threads themselves don't
            # transfer) and deadlock on first read. Drain pending futures
            # and tear down so fork happens with only the main thread alive.
            log.info("  Pausing buffer GT executor for clean val fork...")
            gt_executor.shutdown(wait=True)

            val_records = validate_rollout(
                model=model,
                val_ds=val_ds,
                n_samples=args.n_val_samples,
                max_lead=args.max_lead_late,
                device=device,
                lat_weights=lat_weights_trimmed,
                num_workers=args.num_val_workers,
                prefetch_factor=args.prefetch_factor,
            )

            # Recreate the executor for subsequent training steps.
            gt_executor = ThreadPoolExecutor(max_workers=args.gt_prefetch_workers)
            log.info("  Buffer GT executor restarted.")

            val_dt = time.time() - t_val

            # Long-format JSONL: one row per (lead_h, var).
            for rec in val_records:
                writer.log({"phase": "val_rollout", "step": step, **rec})
            writer.flush_summary()

            # Summary metric for best-checkpoint selection: mean lat-weighted
            # RMSE across all (lead, var) records — a single scalar that
            # rewards models with low error at every lead and variable.
            rmses = [r["rmse_lat"] for r in val_records if not math.isnan(r["rmse_lat"])]
            mean_rmse = sum(rmses) / len(rmses) if rmses else float("inf")

            log.info(f"  val_rollout: {len(val_records)} (lead, var) records "
                     f"| mean rmse_lat={mean_rmse:.4f} | took {val_dt:.0f}s")
            for lead_h in (6, 24, 48, 72):
                line_parts = [f"lead={lead_h:>3}h:"]
                for var in ("swvl1", "stl1", "sd", "2t"):
                    r = next(
                        (x for x in val_records
                         if x["lead_h"] == lead_h and x["var"] == var),
                        None,
                    )
                    if r is not None:
                        line_parts.append(f"rmse_{var}={r['rmse_lat']:.4f}")
                log.info("  " + " ".join(line_parts))

            if mean_rmse < best_val_rmse:
                best_val_rmse = mean_rmse
                ckpt = save_checkpoint(run_dir, model, optimizer, scheduler, step, tag="best")
                log.info(f"  ✓ New best mean rmse_lat {best_val_rmse:.4f} → {ckpt}")

        # ── Periodic checkpoint ────────────────────────────────────────
        if step % args.save_every == 0:
            ckpt = save_checkpoint(run_dir, model, optimizer, scheduler, step)
            log.info(f"Saved checkpoint: {ckpt}")

    # ── End of training ────────────────────────────────────────────────
    save_checkpoint(run_dir, model, optimizer, scheduler, step, tag="final")
    writer.close()
    gt_executor.shutdown(wait=False)
    log.info(
        f"\nStage 2 complete — {step:,} steps | best mean rmse_lat: {best_val_rmse:.4f}"
    )
    log.info(f"All artefacts at: {run_dir}")


if __name__ == "__main__":
    main()
