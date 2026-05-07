"""Reference baselines for the strict hydrology-decoding question.

Computes two reference baselines on the same val/test splits, with the same
land mask and per-sample / spatial / scatter outputs as
``scripts/eval_linear_probe.py``, so the analysis notebook can plot all three
side by side without any glue code.

Baselines
---------
persistence
    pred(t+6h) := y(t).  Uses the hydrology value at the source timestep as
    the prediction for the target timestep.  This is a very strong baseline
    on short horizons for slowly-varying fields (especially soil moisture).

climatology  (hour_of_day strategy)
    pred(t+6h) := mean over training samples of y(target_hour, lat, lon),
    bucketed per UTC hour-of-day.  Captures the diurnal cycle without any
    information about the current synoptic state.

Both baselines use exactly the same dataset splits, land mask, target
variables, MAE/RMSE definitions and per-pixel aggregation as
``scripts/eval_linear_probe.py``.

Output layout (under --output-base)
-----------------------------------
{output_base}/{baseline}/
    eval_summary.json                            aggregate per-(split, var) MAE/RMSE
    per_sample_metrics.csv                       one row per (split, sample, var)
    spatial_aggregates/
        {split}_count.npy                        land-pixel coverage map (int32)
        {split}_{var}_mean_abs_err.npy           per-pixel time-MAE       (float32)
        {split}_{var}_mean_signed_err.npy        per-pixel time-mean err  (float32)
        {split}_{var}_rmse.npy                   per-pixel time-RMSE      (float32)
    scatter_samples_{split}.pt                   {var: {pred, truth}} numpy arrays
    climatology_table.pt                         (climatology only) the (24, V, H, W) lookup

Usage on Nautilus
-----------------
    python -m scripts.compute_reference_baselines \\
        --data-dir /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025 \\
        --output-base /mnt/data/runs/reference_baselines \\
        --splits val test \\
        --baselines persistence climatology

Runtime: dominated by ERA5 IO.  No GPU is required.
The implementation intentionally reads only per-step surface files; it does not
instantiate the Aurora ERA5Dataset and does not load atmospheric NetCDFs.
"""

from __future__ import annotations

# UCX/atexit guards — same as eval_linear_probe.py.  Even with num_workers=0
# this script touches torch + nccl import paths in the same docker image.
import os
os.environ.setdefault("UCX_HANDLE_ERRORS", "none")
os.environ.setdefault("UCX_ERROR_SIGNALS", "")

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from src.data import (
    DEFAULT_TEST_RANGES,
    DEFAULT_TRAIN_RANGES,
    DEFAULT_VAL_RANGES,
    DENSITY_VARS,
    EXTRA_SURF_VAR_NAMES,
    EXTRA_SURF_ERA5_TO_AURORA,
    STATIC_ERA5_TO_AURORA,
    SURF_ERA5_TO_AURORA,
    _base_surface_var,
    _parse_per_timestep_surface_files,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistence + climatology hydrology baselines")
    parser.add_argument("--data-dir", nargs="+", required=True,
                        help="ERA5 per-step year directories (must match the LP fit)")
    parser.add_argument("--output-base", type=Path, required=True,
                        help="Output directory; subdirectories per baseline are created under here")
    parser.add_argument("--baselines", nargs="+", default=["persistence", "climatology"],
                        choices=["persistence", "climatology"])
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--target-vars", nargs="+", default=list(EXTRA_SURF_VAR_NAMES),
                        help="Default: swvl1 stl1 sd")
    parser.add_argument("--mask", choices=["lsm", "density"], default="lsm",
                        help="lsm (matches the linear-probe eval) or per-variable density")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--file-layout", choices=["per_timestep", "chunked"], default="per_timestep")
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--static-path", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Ignored. Kept for compatibility; this script uses a surface-only "
                             "iterator and no DataLoader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--num-snapshots", type=int, default=0,
                        help="Per-split full-grid (truth, pred, abs_err) snapshots saved as .pt; 0 disables")
    parser.add_argument("--scatter-target", type=int, default=50_000,
                        help="Total (pred, truth) pairs per (split, var) for scatter plots")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per split (for quick test runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Wipe any existing output subdirectories")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Online aggregators (mirror scripts/eval_linear_probe.py exactly so the
# notebook can load the outputs interchangeably)
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
        if var == next(iter(self.sum_abs)):
            self.count += mask_cpu.to(torch.int32)

    def finalize(self) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor | None]:
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
    """Per-sample stratified subsampling of (pred, truth) pairs over land."""

    def __init__(self, target_vars: tuple[str, ...], total_samples: int,
                 target_total: int, rng: np.random.Generator):
        per_sample = max(1, int(np.ceil(target_total / max(total_samples, 1))))
        self.k = per_sample
        self.rng = rng
        self.preds:  dict[str, list[np.ndarray]] = {v: [] for v in target_vars}
        self.truths: dict[str, list[np.ndarray]] = {v: [] for v in target_vars}

    def update(self, var: str, pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> None:
        flat_pred  = pred.detach().to("cpu", dtype=torch.float32).flatten()
        flat_truth = truth.detach().to("cpu", dtype=torch.float32).flatten()
        flat_mask  = mask.detach().to("cpu", dtype=torch.bool).flatten()
        idx = torch.nonzero(flat_mask, as_tuple=False).flatten().numpy()
        if idx.size == 0:
            return
        chosen = idx if idx.size <= self.k else self.rng.choice(idx, size=self.k, replace=False)
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

@dataclass
class SurfaceOnlySample:
    source_time: datetime
    target_time: datetime
    source: dict[str, torch.Tensor]
    target: dict[str, torch.Tensor]
    densities: dict[str, torch.Tensor]


class SurfaceOnlySplit:
    """Surface-only replacement for ERA5Dataset for reference baselines.

    The Aurora dataset loads atmospheric files because model inference needs them.
    Persistence and climatology never use atmosphere, so this class indexes the
    same timestamp sequences but opens only ``*-surface.nc`` files.
    """

    def __init__(
        self,
        data_dirs: list[Path],
        date_ranges: list[tuple[datetime, datetime]],
        target_vars: tuple[str, ...],
        step_hours: int,
        static_path: Path | None,
        mask_kind: str,
    ):
        self.data_dirs = data_dirs
        self.target_vars = target_vars
        self.step = timedelta(hours=step_hours)
        self.mask_kind = mask_kind
        self.surf_paths_by_time: dict[datetime, Path] = {}
        self.era5_by_var = {
            aurora_name: era5_name
            for era5_name, aurora_name in (SURF_ERA5_TO_AURORA | EXTRA_SURF_ERA5_TO_AURORA).items()
        }
        self.needed_era5_vars = sorted({
            self.era5_by_var[_base_surface_var(name)] for name in target_vars
        })

        discovered_static_path = static_path
        for data_dir in data_dirs:
            candidate = data_dir / "static.nc"
            if candidate.exists() and discovered_static_path is None:
                discovered_static_path = candidate
            self.surf_paths_by_time.update(_parse_per_timestep_surface_files(data_dir))

        if not self.surf_paths_by_time:
            raise FileNotFoundError("No per-timestep *-surface.nc files found in data dirs")
        if discovered_static_path is None:
            raise FileNotFoundError("No static.nc found in any data directory")

        with xr.open_dataset(discovered_static_path, engine="netcdf4") as static_ds:
            self.static_vars = {
                STATIC_ERA5_TO_AURORA[k]: torch.from_numpy(static_ds[k].values[0]).float()
                for k in STATIC_ERA5_TO_AURORA
            }

        self.sequences = self._build_sequences(date_ranges)
        if not self.sequences:
            raise ValueError(f"No valid surface-only samples found for ranges: {date_ranges}")

        first_target = self._load_surface(self.sequences[0][2], include_density=False)
        first_var = target_vars[0]
        self.shape = tuple(first_target[first_var].shape)

    def _build_sequences(
        self,
        date_ranges: list[tuple[datetime, datetime]],
    ) -> list[tuple[datetime, datetime, datetime]]:
        avail = set(self.surf_paths_by_time)
        sequences: list[tuple[datetime, datetime, datetime]] = []

        for start_date, end_date in date_ranges:
            for t1 in sorted(avail):
                if t1 < start_date or t1 >= end_date:
                    continue
                t0 = t1 - self.step
                t2 = t1 + self.step
                timestamps = (t0, t1, t2)
                if not all(ts in avail for ts in timestamps):
                    continue
                if any(ts.month != t1.month or ts.year != t1.year for ts in timestamps):
                    continue
                sequences.append(timestamps)
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def _load_surface(self, dt: datetime, include_density: bool) -> dict[str, torch.Tensor]:
        path = self.surf_paths_by_time[dt]
        with xr.open_dataset(path, engine="netcdf4") as ds:
            sliced = ds[self.needed_era5_vars].load().isel(valid_time=0)

        out: dict[str, torch.Tensor] = {}
        for name in self.target_vars:
            base_name = _base_surface_var(name)
            raw = torch.from_numpy(sliced[self.era5_by_var[base_name]].values).float()
            out[base_name] = raw.nan_to_num(0.0) if base_name in DENSITY_VARS else raw
            if include_density and base_name in DENSITY_VARS:
                out[f"{base_name}_density"] = (~torch.isnan(raw)).float()
        return out

    def __getitem__(self, idx: int) -> SurfaceOnlySample:
        _, source_time, target_time = self.sequences[idx]
        source = self._load_surface(source_time, include_density=False)
        target = self._load_surface(target_time, include_density=self.mask_kind == "density")
        densities = {f"{v}_density": target[f"{v}_density"] for v in self.target_vars
                     if f"{v}_density" in target}
        return SurfaceOnlySample(
            source_time=source_time,
            target_time=target_time,
            source=source,
            target=target,
            densities=densities,
        )

    def close(self) -> None:
        return None


def evenly_spaced_indices(n: int, k: int) -> list[int]:
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def safe_ts(ts: str) -> str:
    return ts.replace(":", "").replace("-", "").replace("T", "_")


def build_land_mask(
    dataset: SurfaceOnlySplit,
    sample: SurfaceOnlySample,
    target_vars: tuple[str, ...],
    mask_kind: str,
    threshold: float,
    valid_h: int,
    valid_w: int,
) -> dict[str, torch.Tensor]:
    """Build per-variable boolean masks, matching the linear-probe eval."""
    if mask_kind == "lsm":
        m = (dataset.static_vars["lsm"][:valid_h, :valid_w] > threshold).cpu()
        return {v: m for v in target_vars}
    if mask_kind == "density":
        fallback = (dataset.static_vars["lsm"][:valid_h, :valid_w] > threshold).cpu()
        masks: dict[str, torch.Tensor] = {}
        for v in target_vars:
            density = sample.densities.get(f"{v}_density")
            masks[v] = (density[:valid_h, :valid_w] >= threshold).cpu() if density is not None else fallback
        return masks
    raise ValueError(f"unknown mask kind: {mask_kind}")


# ---------------------------------------------------------------------------
# Climatology: hour-of-day per-pixel mean from training samples
# ---------------------------------------------------------------------------

def build_hourly_climatology(
    args: argparse.Namespace,
    train_ds: SurfaceOnlySplit,
    target_vars: tuple[str, ...],
    valid_h: int,
    valid_w: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Iterate through train_ds once, accumulating per-(hour, var, pixel) sums.

    Returns
    -------
    clim : (24, V, H, W) float32 tensor of per-hour, per-pixel means
    counts : (24,) int64 tensor of training samples per UTC hour
    var_order : list of variable names matching the V axis
    """
    var_order = list(target_vars)
    n_vars = len(var_order)
    sums = torch.zeros((24, n_vars, valid_h, valid_w), dtype=torch.float64)
    counts = torch.zeros(24, dtype=torch.int64)

    n_seen = 0
    t0 = time.time()
    print(f"  climatology fit: train_ds has {len(train_ds)} samples", flush=True)

    n_total = len(train_ds) if args.limit is None else min(len(train_ds), args.limit)
    for idx in range(n_total):
        if args.limit is not None and n_seen >= args.limit:
            break
        sample = train_ds[idx]
        target_time = sample.target_time
        h = int(target_time.hour)
        for vi, v in enumerate(var_order):
            y = sample.target[v][:valid_h, :valid_w].double()
            sums[h, vi] += y
        counts[h] += 1
        n_seen += 1
        if n_seen == 1:
            print(f"  climatology fit: first sample t+6h = {target_time.isoformat()}",
                  flush=True)
        if n_seen % 100 == 0:
            rate = n_seen / max(time.time() - t0, 1e-6)
            print(f"  climatology fit: accumulated {n_seen} train samples ({rate:.2f} samples/s)",
                  flush=True)

    train_ds.close()

    # Per-pixel mean = sum / count  (broadcast over H, W; clamp count to avoid /0)
    counts_safe = counts.clamp(min=1).double()  # (24,)
    clim = (sums / counts_safe.view(24, 1, 1, 1)).to(torch.float32)

    print(f"  climatology fit: done. Per-hour sample counts: {counts.tolist()}", flush=True)
    return clim, counts, var_order


# ---------------------------------------------------------------------------
# Per-split eval
# ---------------------------------------------------------------------------

def eval_split(
    args: argparse.Namespace,
    baseline_name: str,
    split_name: str,
    dataset: SurfaceOnlySplit,
    target_vars: tuple[str, ...],
    output_dir: Path,
    csv_writer,
    csv_file,
    rng: np.random.Generator,
    valid_h: int,
    valid_w: int,
    *,
    climatology: torch.Tensor | None = None,
    var_order: list[str] | None = None,
):
    """One pass through `dataset`, writing per-sample CSV rows + accumulating
    spatial aggregates and scatter samples.  Saves outputs to disk before any
    teardown so a worker-shutdown segfault can never lose data."""
    spatial = SpatialAggregator(target_vars)
    n_total = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    scatter = ScatterReservoir(
        target_vars=target_vars,
        total_samples=n_total,
        target_total=args.scatter_target,
        rng=rng,
    )
    snapshot_dir = output_dir / "snapshots"
    if args.num_snapshots > 0:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_indices = set(evenly_spaced_indices(n_total, args.num_snapshots))

    accum = {v: {"abs": 0.0, "sq": 0.0, "count": 0} for v in target_vars}
    n_seen = 0
    t0 = time.time()

    try:
        for idx in range(n_total):
            sample = dataset[idx]
            source_time = sample.source_time.isoformat()
            target_time = sample.target_time
            target_time_str = target_time.isoformat()
            do_snapshot = n_seen in snapshot_indices

            masks = build_land_mask(
                dataset=dataset,
                sample=sample,
                target_vars=target_vars,
                mask_kind=args.mask,
                threshold=args.mask_threshold,
                valid_h=valid_h,
                valid_w=valid_w,
            )

            for var in target_vars:
                truth_2d = sample.target[var][:valid_h, :valid_w]

                if baseline_name == "persistence":
                    pred_2d = sample.source[var][:valid_h, :valid_w]
                elif baseline_name == "climatology":
                    assert climatology is not None and var_order is not None
                    vi = var_order.index(var)
                    pred_2d = climatology[int(target_time.hour), vi].clone()
                else:
                    raise ValueError(baseline_name)

                err_2d = pred_2d - truth_2d
                mask_hw = masks[var][:valid_h, :valid_w]

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
                    baseline_name, split_name, n_seen, source_time, target_time_str, var,
                    n_pix, f"{mae:.6e}", f"{rmse:.6e}", f"{signed:.6e}",
                ])

                spatial.update(var, err_2d, mask_hw)
                scatter.update(var, pred_2d, truth_2d, mask_hw)

                if do_snapshot and args.num_snapshots > 0:
                    snap_path = (snapshot_dir
                                 / f"{split_name}_{n_seen:04d}_{safe_ts(target_time_str)}_{var}.pt")
                    torch.save(
                        {
                            "truth":   truth_2d.detach().to("cpu", dtype=torch.float16),
                            "pred":    pred_2d.detach().to("cpu", dtype=torch.float16),
                            "abs_err": err_2d.detach().abs().to("cpu", dtype=torch.float16),
                            "source_time": source_time,
                            "target_time": target_time_str,
                            "var": var,
                            "split": split_name,
                            "valid_h": valid_h,
                            "valid_w": valid_w,
                            "baseline": baseline_name,
                        },
                        snap_path,
                    )

            csv_file.flush()
            n_seen += 1
            if n_seen == 1:
                print(f"  {split_name}: first sample | source={source_time} target={target_time_str}",
                      flush=True)
            if n_seen % 50 == 0:
                rate = n_seen / max(time.time() - t0, 1e-6)
                print(f"  {split_name}: evaluated {n_seen} / {n_total} samples "
                      f"({rate:.2f} samples/s)", flush=True)

        # Persist split outputs immediately (before any potentially-segfaulty
        # dataloader teardown).
        summary = {f"{split_name}_samples": n_seen}
        for v in target_vars:
            c = max(accum[v]["count"], 1)
            summary[f"{split_name}_{v}_mae"]  = accum[v]["abs"] / c
            summary[f"{split_name}_{v}_rmse"] = (accum[v]["sq"] / c) ** 0.5
        spatial_maps, count_map = spatial.finalize()
        scatter_data = scatter.finalize()

        aggregates_dir = output_dir / "spatial_aggregates"
        aggregates_dir.mkdir(parents=True, exist_ok=True)
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
        torch.save(scatter_data, output_dir / f"scatter_samples_{split_name}.pt")

        partial_path = output_dir / "eval_summary_partial.json"
        if partial_path.exists():
            try:
                partial_payload = json.loads(partial_path.read_text())
            except Exception:
                partial_payload = {"metrics": {}}
        else:
            partial_payload = {"metrics": {}}
        partial_payload["metrics"].update(summary)
        partial_payload["last_completed_split"] = split_name
        partial_path.write_text(json.dumps(partial_payload, indent=2))
        print(f"  {split_name}: per-split outputs saved to disk", flush=True)
    finally:
        try:
            dataset.close()
        except BaseException as exc:  # noqa: BLE001
            print(f"  {split_name}: warning — dataset.close() raised {exc!r}", flush=True)

    return spatial_maps, count_map, scatter_data, summary


# ---------------------------------------------------------------------------
# Per-baseline driver
# ---------------------------------------------------------------------------

def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            return
        for child in sorted(path.glob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_file():
                        sub.unlink()
                    else:
                        sub.rmdir()
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def make_split_datasets(args: argparse.Namespace, target_vars: tuple[str, ...]):
    """Build train/val/test splits using only per-timestep surface files."""
    if args.file_layout != "per_timestep":
        raise ValueError("surface-only reference baselines currently require --file-layout per_timestep")
    data_dirs = [Path(p) for p in args.data_dir]
    kwargs = dict(
        data_dirs=data_dirs,
        target_vars=target_vars,
        step_hours=args.step_hours,
        static_path=args.static_path,
        mask_kind=args.mask,
    )
    return (
        SurfaceOnlySplit(date_ranges=DEFAULT_TRAIN_RANGES, **kwargs),
        SurfaceOnlySplit(date_ranges=DEFAULT_VAL_RANGES, **kwargs),
        SurfaceOnlySplit(date_ranges=DEFAULT_TEST_RANGES, **kwargs),
    )


def run_baseline(
    args: argparse.Namespace,
    baseline_name: str,
    target_vars: tuple[str, ...],
):
    output_dir = args.output_base / baseline_name
    prepare_output_dir(output_dir, overwrite=args.overwrite)

    print(f"\n=========== Baseline: {baseline_name} ===========", flush=True)
    print(f"  output dir: {output_dir}", flush=True)

    train_ds, val_ds, test_ds = make_split_datasets(args, target_vars)
    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    print(f"  Dataset sizes | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
          flush=True)

    # Determine the valid (H, W) crop.  All samples share the same lat/lon grid.
    probe_split = next(s for s in args.splits if len(splits[s]) > 0)
    H, W = splits[probe_split].shape
    print(f"  Grid: {H} x {W}", flush=True)

    # For climatology, build the per-(hour, var, pixel) lookup from training.
    climatology = None
    var_order: list[str] | None = None
    if baseline_name == "climatology":
        # Rebuild train_ds because we'll close it inside build_hourly_climatology.
        train_ds_for_clim, _, _ = make_split_datasets(args, target_vars)
        climatology, hour_counts, var_order = build_hourly_climatology(
            args=args,
            train_ds=train_ds_for_clim,
            target_vars=target_vars,
            valid_h=H,
            valid_w=W,
        )
        torch.save(
            {
                "climatology": climatology,
                "hour_counts": hour_counts,
                "var_order": var_order,
                "H": H, "W": W,
                "data_dirs": args.data_dir,
            },
            output_dir / "climatology_table.pt",
        )

    csv_path = output_dir / "per_sample_metrics.csv"
    summary_path = output_dir / "eval_summary.json"
    rng = np.random.default_rng(args.seed)
    summary: dict[str, float] = {}

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "baseline", "split", "sample_idx", "source_time", "target_time", "target_var",
            "n_land_pixels", "mae", "rmse", "mean_signed_error",
        ])

        for split_name in args.splits:
            ds = splits[split_name]
            if len(ds) == 0:
                print(f"  Skipping empty split: {split_name}")
                continue
            print(f"\n  === {baseline_name} | split: {split_name} (n={len(ds)}) ===", flush=True)
            try:
                _, _, _, split_summary = eval_split(
                    args=args,
                    baseline_name=baseline_name,
                    split_name=split_name,
                    dataset=ds,
                    target_vars=target_vars,
                    output_dir=output_dir,
                    csv_writer=writer,
                    csv_file=f,
                    rng=rng,
                    valid_h=H,
                    valid_w=W,
                    climatology=climatology,
                    var_order=var_order,
                )
            except BaseException as exc:  # noqa: BLE001
                print(f"  {split_name}: ERROR — {exc!r}.  Continuing to next split.", flush=True)
                continue
            summary.update(split_summary)
            print(f"  {split_name}: " + json.dumps(split_summary, indent=2), flush=True)

    var_units = {"swvl1": "m^3/m^3", "stl1": "K", "sd": "m"}
    summary_payload = {
        "baseline": baseline_name,
        "data_dirs": args.data_dir,
        "splits": args.splits,
        "target_vars": list(target_vars),
        "var_units": {v: var_units.get(v, "") for v in target_vars},
        "mask": args.mask,
        "mask_threshold": args.mask_threshold,
        "step_hours": args.step_hours,
        "file_layout": args.file_layout,
        "num_snapshots": args.num_snapshots,
        "scatter_target": args.scatter_target,
        "seed": args.seed,
        "metrics": summary,
    }
    if baseline_name == "climatology":
        summary_payload["climatology_strategy"] = "hour_of_day"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"\n  Wrote {summary_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_base.mkdir(parents=True, exist_ok=True)
    target_vars = tuple(args.target_vars)

    print(f"Data dirs: {args.data_dir}")
    print(f"Output base: {args.output_base}")
    print(f"Baselines: {args.baselines}")
    print(f"Splits: {args.splits}")
    print(f"Target vars: {target_vars}")
    print(f"Mask: {args.mask} > {args.mask_threshold}")
    print(f"File layout: {args.file_layout}")
    if args.num_workers:
        print(
            f"Note: --num-workers={args.num_workers} is ignored; using surface-only direct IO.",
            flush=True,
        )

    for baseline_name in args.baselines:
        run_baseline(args, baseline_name, target_vars)

    print("\nDone.")
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
    # Skip Python's normal interpreter shutdown — same atexit/UCX hazard as in
    # eval_linear_probe.py.  Everything we care about is on disk by now.
    os._exit(0)
