"""Second-pass evaluation for the streaming linear-probe baseline.

Loads `heads.pt` produced by `src/linear_probe.py`, walks the val + test
splits once, and writes the five outputs needed to populate the report's
quantitative tables, spatial maps, scatter plots, and qualitative figures
without ever storing per-sample full-grid tensors.

Outputs (under --output-dir, ~325 MB total at default settings):

    per_sample_metrics.csv           (~150 KB)
        One row per (split, sample_idx, target_var). Columns:
        split, sample_idx, source_time, target_time, target_var,
        n_land_pixels, mae, rmse, mean_signed_error.

    spatial_aggregates/
        <split>_<var>_mean_abs_err.npy     (H, W) float32
        <split>_<var>_mean_signed_err.npy  (H, W) float32
        <split>_<var>_rmse.npy             (H, W) float32
        <split>_count.npy                  (H, W) int32

    snapshots/
        <split>_<idx>_<ts>_<var>.pt
            { "truth": (H,W) fp16, "pred": (H,W) fp16, "abs_err": (H,W) fp16 }
        Saved at --num-snapshots evenly-spaced indices per split.

    scatter_samples.pt
        { split: { var: { "pred": (N,) fp32, "truth": (N,) fp32 } } }
        N ≈ scatter_target across the split, drawn uniformly from land pixels
        (one fixed-seed subsample per sample, K_per_sample = ceil(target / n_samples)).

    eval_summary.json
        Aggregate per-variable, per-split MAE / RMSE in native units, plus
        the heads file path, splits evaluated, and dataset metadata.

Usage on Nautilus:
    python -m scripts.eval_linear_probe \\
        --heads /mnt/data/runs/linear_probe_streaming/heads.pt \\
        --data-dir /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025 \\
        --output-dir /mnt/data/runs/linear_probe_streaming/eval \\
        --splits val test --small
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aurora import AuroraPretrained, AuroraSmallPretrained

from src.data import (
    BASE_SURF_VAR_NAMES,
    collate_era5_batch,
    era5_worker_init_fn,
    make_era5_splits,
)
from src.finetune_stage1 import _NEW_VAR_NORM
from src.linear_probe import (
    SurfaceLatentTap,
    predict_from_latent,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval pass over a trained linear probe")
    parser.add_argument("--heads", type=Path, required=True,
                        help="Path to heads.pt from src.linear_probe")
    parser.add_argument("--data-dir", nargs="+", required=True,
                        help="ERA5 per-step year directories (must match the fit run)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--small", action="store_true",
                        help="Use AuroraSmallPretrained (must match the fit run)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--file-layout", choices=["per_timestep", "chunked"],
                        default="per_timestep")
    parser.add_argument("--static-path", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--surface-head-var", default="2t")
    parser.add_argument("--num-snapshots", type=int, default=6,
                        help="Snapshots per split for qualitative figures")
    parser.add_argument("--scatter-target", type=int, default=50_000,
                        help="Target number of (pred, truth) pairs per (split, var) for scatter plots")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per split (for quick test runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty --output-dir. By default, the script refuses to overwrite prior eval artefacts.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Online aggregators
# ---------------------------------------------------------------------------

class SpatialAggregator:
    """Per-pixel running sums of |err|, err, err^2, and count over a split."""

    def __init__(self, target_vars: tuple[str, ...]):
        self.sum_abs:  dict[str, torch.Tensor | None] = {v: None for v in target_vars}
        self.sum_sq:   dict[str, torch.Tensor | None] = {v: None for v in target_vars}
        self.sum_sign: dict[str, torch.Tensor | None] = {v: None for v in target_vars}
        self.count:    torch.Tensor | None = None  # shared across vars (mask is the same)

    def _ensure(self, hw: tuple[int, int]) -> None:
        if self.count is None:
            self.count = torch.zeros(hw, dtype=torch.int32)
        for v in self.sum_abs:
            if self.sum_abs[v] is None:
                self.sum_abs[v]  = torch.zeros(hw, dtype=torch.float64)
                self.sum_sq[v]   = torch.zeros(hw, dtype=torch.float64)
                self.sum_sign[v] = torch.zeros(hw, dtype=torch.float64)

    def update(self, var: str, err: torch.Tensor, mask: torch.Tensor) -> None:
        """err: (H, W) fp32 on any device. mask: (H, W) bool on same device."""
        self._ensure(err.shape)
        err_cpu  = err.detach().to("cpu", dtype=torch.float64)
        mask_cpu = mask.detach().to("cpu", dtype=torch.bool)
        masked_abs = torch.where(mask_cpu, err_cpu.abs(), torch.zeros_like(err_cpu))
        masked_sq  = torch.where(mask_cpu, err_cpu * err_cpu, torch.zeros_like(err_cpu))
        masked_sgn = torch.where(mask_cpu, err_cpu, torch.zeros_like(err_cpu))
        self.sum_abs[var]  += masked_abs
        self.sum_sq[var]   += masked_sq
        self.sum_sign[var] += masked_sgn
        # count is shared per split (lsm doesn't change), but update once per call
        if var == next(iter(self.sum_abs)):
            self.count += mask_cpu.to(torch.int32)

    def finalize(self) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor | None]:
        """Return ({var: {metric_name: (H,W) fp32}}, count: (H,W) int32)."""
        if self.count is None:
            return {}, None
        denom = self.count.clamp(min=1).to(torch.float64)
        out: dict[str, dict[str, torch.Tensor]] = {}
        never_seen = self.count == 0
        for v in self.sum_abs:
            mean_abs = (self.sum_abs[v]  / denom).to(torch.float32)
            mean_sgn = (self.sum_sign[v] / denom).to(torch.float32)
            rmse     = (self.sum_sq[v]   / denom).sqrt().to(torch.float32)
            mean_abs[never_seen] = float("nan")
            mean_sgn[never_seen] = float("nan")
            rmse[never_seen]     = float("nan")
            out[v] = {
                "mean_abs_err": mean_abs,
                "mean_signed_err": mean_sgn,
                "rmse": rmse,
            }
        return out, self.count.clone()


class ScatterReservoir:
    """Per-sample stratified subsampling: K random land pixels per (split, var, sample).

    Cheaper and unbiased compared to a streaming reservoir over all pixels:
    for total ~M samples and target N pairs, K_per_sample = ceil(N / M)
    gives an exactly-uniform random subsample of land pixels per sample.
    """

    def __init__(self, target_vars: tuple[str, ...], total_samples: int, target_total: int,
                 rng: np.random.Generator):
        per_sample = max(1, int(np.ceil(target_total / max(total_samples, 1))))
        self.k = per_sample
        self.rng = rng
        self.preds:  dict[str, list[np.ndarray]] = {v: [] for v in target_vars}
        self.truths: dict[str, list[np.ndarray]] = {v: [] for v in target_vars}

    def update(self, var: str, pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> None:
        """1-D arrays of land pixels for this sample/var."""
        flat_pred  = pred.detach().to("cpu", dtype=torch.float32).flatten()
        flat_truth = truth.detach().to("cpu", dtype=torch.float32).flatten()
        flat_mask  = mask.detach().to("cpu", dtype=torch.bool).flatten()
        idx = torch.nonzero(flat_mask, as_tuple=False).flatten().numpy()
        if idx.size == 0:
            return
        if idx.size <= self.k:
            chosen = idx
        else:
            chosen = self.rng.choice(idx, size=self.k, replace=False)
        self.preds[var].append(flat_pred.numpy()[chosen])
        self.truths[var].append(flat_truth.numpy()[chosen])

    def finalize(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            v: {
                "pred":  np.concatenate(self.preds[v])  if self.preds[v]  else np.empty(0, dtype=np.float32),
                "truth": np.concatenate(self.truths[v]) if self.truths[v] else np.empty(0, dtype=np.float32),
            }
            for v in self.preds
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loader(args: argparse.Namespace, dataset) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        # This is a one-pass evaluation loader, so keep worker lifetime short.
        # Using persistent workers here has triggered teardown instability on
        # Nautilus after a split finishes.
        persistent_workers=False,
        worker_init_fn=era5_worker_init_fn,
        collate_fn=collate_era5_batch,
        pin_memory=(args.device == "cuda"),
    )
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def load_heads(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"target_vars", "patch_size", "weights"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"heads.pt is missing keys: {missing}")
    return payload


def heads_to_solved(payload: dict) -> dict[str, dict[str, torch.Tensor]]:
    return {
        var: {"weight": payload["weights"][var]["weight"], "bias": payload["weights"][var]["bias"]}
        for var in payload["target_vars"]
    }


def evenly_spaced_indices(n: int, k: int) -> list[int]:
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def safe_ts(ts: str) -> str:
    return ts.replace(":", "").replace("-", "").replace("T", "_")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    """Protect existing evaluation outputs unless overwrite is explicit."""
    if path.exists():
        existing = list(path.iterdir())
        if existing and not overwrite:
            preview = ", ".join(sorted(p.name for p in existing[:10]))
            raise FileExistsError(
                f"Output directory {path} already exists and is not empty. "
                f"Refusing to overwrite existing evaluation artefacts. "
                f"Pass --overwrite to allow it. Existing entries: {preview}"
            )
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-split eval loop
# ---------------------------------------------------------------------------

def eval_split(
    args: argparse.Namespace,
    split_name: str,
    model,
    tap: SurfaceLatentTap,
    dataset,
    target_vars: tuple[str, ...],
    norm_stats: dict[str, tuple[float, float]],
    solved: dict[str, dict[str, torch.Tensor]],
    land_mask: torch.Tensor,
    snapshot_indices: set[int],
    output_dir: Path,
    csv_writer: csv.writer,
    rng: np.random.Generator,
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    torch.Tensor | None,
    dict[str, dict[str, np.ndarray]],
    dict[str, float],
]:
    spatial = SpatialAggregator(target_vars)
    n_total = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    scatter = ScatterReservoir(
        target_vars=target_vars,
        total_samples=n_total,
        target_total=args.scatter_target,
        rng=rng,
    )

    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    accum = {v: {"abs": 0.0, "sq": 0.0, "count": 0} for v in target_vars}
    n_seen = 0
    t0 = time.time()
    loader = make_loader(args, dataset)
    loader_iter = iter(loader)

    try:
        while True:
            if args.limit is not None and n_seen >= args.limit:
                break
            try:
                input_batch, targets = next(loader_iter)
            except StopIteration:
                break
            target_batch = targets[0]

            for forbidden in target_vars:
                assert forbidden not in input_batch.surf_vars, f"{forbidden} leaked into inputs"

            with torch.inference_mode():
                model(input_batch.to(args.device))
            latent = tap.get_surface_latent()

            preds = predict_from_latent(
                latent=latent,
                solved=solved,
                target_vars=target_vars,
                norm_stats=norm_stats,
                patch_size=args.heads_patch_size,
            )

            valid_h = latent.shape[1] * args.heads_patch_size
            valid_w = latent.shape[2] * args.heads_patch_size
            mask_hw = land_mask[:valid_h, :valid_w].to(device=latent.device, dtype=torch.bool)

            source_time = input_batch.metadata.time[0].isoformat()
            target_time = target_batch.metadata.time[0].isoformat()
            do_snapshot = n_seen in snapshot_indices

            for var in target_vars:
                truth = target_batch.surf_vars[var][:, -1, :valid_h, :valid_w].to(
                    device=latent.device, dtype=torch.float32
                )
                pred = preds[var]
                # batch dim is always 1 in this script
                truth_2d = truth[0]
                pred_2d  = pred[0]
                err_2d   = pred_2d - truth_2d

                # Per-sample scalar metrics (land-only).
                masked_err = err_2d[mask_hw]
                n_pix = int(masked_err.numel())
                if n_pix == 0:
                    mae = rmse = signed = float("nan")
                else:
                    abs_sum = float(masked_err.abs().sum().item())
                    sq_sum  = float(masked_err.square().sum().item())
                    sgn_sum = float(masked_err.sum().item())
                    mae    = abs_sum / n_pix
                    rmse   = (sq_sum / n_pix) ** 0.5
                    signed = sgn_sum / n_pix
                    accum[var]["abs"]   += abs_sum
                    accum[var]["sq"]    += sq_sum
                    accum[var]["count"] += n_pix

                csv_writer.writerow([
                    split_name, n_seen, source_time, target_time, var,
                    n_pix, f"{mae:.6e}", f"{rmse:.6e}", f"{signed:.6e}",
                ])

                spatial.update(var, err_2d, mask_hw)
                scatter.update(var, pred_2d, truth_2d, mask_hw)

                if do_snapshot:
                    snap_path = snapshot_dir / f"{split_name}_{n_seen:04d}_{safe_ts(target_time)}_{var}.pt"
                    torch.save(
                        {
                            "truth":   truth_2d.detach().to("cpu", dtype=torch.float16),
                            "pred":    pred_2d.detach().to("cpu",  dtype=torch.float16),
                            "abs_err": err_2d.detach().abs().to("cpu", dtype=torch.float16),
                            "source_time": source_time,
                            "target_time": target_time,
                            "var": var,
                            "split": split_name,
                            "valid_h": valid_h,
                            "valid_w": valid_w,
                        },
                        snap_path,
                    )

            n_seen += 1
            if n_seen == 1:
                print(f"  {split_name}: first sample | source={source_time} target={target_time} "
                      f"| latent_shape={tuple(latent.shape)}", flush=True)
            if n_seen % 25 == 0:
                rate = n_seen / max(time.time() - t0, 1e-6)
                print(f"  {split_name}: evaluated {n_seen} / {n_total} samples "
                      f"({rate:.2f} samples/s)", flush=True)
    finally:
        del loader_iter
        del loader
        dataset.close()

    summary = {f"{split_name}_samples": n_seen}
    for v in target_vars:
        c = max(accum[v]["count"], 1)
        summary[f"{split_name}_{v}_mae"]  = accum[v]["abs"] / c
        summary[f"{split_name}_{v}_rmse"] = (accum[v]["sq"] / c) ** 0.5
    spatial_maps, count_map = spatial.finalize()
    return spatial_maps, count_map, scatter.finalize(), summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    payload = load_heads(args.heads)
    target_vars: tuple[str, ...] = tuple(payload["target_vars"])
    args.heads_patch_size = int(payload["patch_size"])
    surface_head_module = payload.get("surface_head_module")
    norm_stats = {v: _NEW_VAR_NORM[v] for v in target_vars}
    solved = heads_to_solved(payload)

    print(f"Device: {args.device}")
    print(f"Heads file: {args.heads}")
    print(f"Target vars: {target_vars}")
    print(f"Surface head module from heads.pt: {surface_head_module}")
    print(f"Patch size: {args.heads_patch_size}")
    print(f"Splits: {args.splits}")
    print(f"Snapshots per split: {args.num_snapshots}")
    print(f"Scatter target / split / var: {args.scatter_target}")

    train_ds, val_ds, test_ds = make_era5_splits(
        data_dirs=[Path(p) for p in args.data_dir],
        step_hours=args.step_hours,
        include_extra_surf=False,
        rollout_steps=1,
        file_layout=args.file_layout,
        input_surf_vars=BASE_SURF_VAR_NAMES,
        target_surf_vars=target_vars,
        static_path=args.static_path,
    )
    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    print(f"Dataset sizes | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
          flush=True)

    aurora_cls = AuroraSmallPretrained if args.small else AuroraPretrained
    model = aurora_cls()
    print("Loading Aurora checkpoint...", flush=True)
    t0 = time.time()
    model.load_checkpoint()
    print(f"Checkpoint loaded in {time.time() - t0:.1f}s", flush=True)
    model.eval()
    model = model.to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)

    # Pull lsm + grid sizing from the first sample of any split.
    probe_split = next(s for s in args.splits if len(splits[s]) > 0)
    probe_sample_in, probe_sample_targets = splits[probe_split][0]
    probe_target = probe_sample_targets[0]
    first_var = target_vars[0]
    target_hw = (
        probe_target.surf_vars[first_var].shape[-2],
        probe_target.surf_vars[first_var].shape[-1],
    )
    land_mask = (probe_target.static_vars["lsm"] > 0.5).cpu()

    tap = SurfaceLatentTap(
        model=model,
        target_hw=target_hw,
        patch_size=args.heads_patch_size,
        surface_head_var=args.surface_head_var,
        module_name=surface_head_module,
    )
    print(f"Tap module: {tap.module_name}", flush=True)

    aggregates_dir = args.output_dir / "spatial_aggregates"
    aggregates_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_sample_metrics.csv"
    scatter_path = args.output_dir / "scatter_samples.pt"
    summary_path = args.output_dir / "eval_summary.json"

    rng = np.random.default_rng(args.seed)
    summary: dict[str, float] = {}
    scatter_all: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "sample_idx", "source_time", "target_time", "target_var",
            "n_land_pixels", "mae", "rmse", "mean_signed_error",
        ])

        try:
            for split_name in args.splits:
                ds = splits[split_name]
                if len(ds) == 0:
                    print(f"Skipping empty split: {split_name}")
                    continue
                n = len(ds) if args.limit is None else min(len(ds), args.limit)
                snap_idx = set(evenly_spaced_indices(n, args.num_snapshots))
                print(f"\n=== Split: {split_name} (n={n}, snapshot indices={sorted(snap_idx)}) ===",
                      flush=True)

                spatial_maps, count_map, scatter, split_summary = eval_split(
                    args=args,
                    split_name=split_name,
                    model=model,
                    tap=tap,
                    dataset=ds,
                    target_vars=target_vars,
                    norm_stats=norm_stats,
                    solved=solved,
                    land_mask=land_mask,
                    snapshot_indices=snap_idx,
                    output_dir=args.output_dir,
                    csv_writer=writer,
                    rng=rng,
                )

                for var, maps in spatial_maps.items():
                    for metric_name, tensor in maps.items():
                        np.save(
                            aggregates_dir / f"{split_name}_{var}_{metric_name}.npy",
                            tensor.numpy(),
                        )
                if count_map is not None:
                    np.save(
                        aggregates_dir / f"{split_name}_count.npy",
                        count_map.numpy().astype(np.int32),
                    )

                scatter_all[split_name] = scatter
                summary.update(split_summary)
                print(json.dumps(split_summary, indent=2), flush=True)
        finally:
            tap.close()

    # Save scatter samples (small).
    torch.save(scatter_all, scatter_path)

    # Native units for each target var, so plotting code does not need to
    # cross-reference physics/ERA5 documentation.
    var_units = {
        "swvl1": "m^3/m^3",
        "stl1":  "K",
        "sd":    "m",
    }

    # Final summary JSON.
    summary_payload = {
        "heads_path": str(args.heads),
        "data_dirs": args.data_dir,
        "splits": args.splits,
        "small": args.small,
        "patch_size": args.heads_patch_size,
        "surface_head_module": tap.module_name,
        "target_vars": list(target_vars),
        "norm_stats": {v: {"mean": float(norm_stats[v][0]), "std": float(norm_stats[v][1])}
                       for v in target_vars},
        "var_units": {v: var_units.get(v, "") for v in target_vars},
        "num_snapshots": args.num_snapshots,
        "scatter_target": args.scatter_target,
        "seed": args.seed,
        "metrics": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    print("\nDone.")
    print(f"Per-sample CSV   : {csv_path}")
    print(f"Spatial maps     : {aggregates_dir}")
    print(f"Snapshots        : {args.output_dir / 'snapshots'}")
    print(f"Scatter samples  : {scatter_path}")
    print(f"Eval summary     : {summary_path}")


if __name__ == "__main__":
    main()
