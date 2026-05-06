"""Streaming linear-probe baseline for hydrology decodability from Aurora.

This script answers the strict baseline question:

    Without changing Aurora's encoder or backbone, how much of
    {swvl1, stl1, sd} at t+6h is linearly decodable from Aurora's frozen
    surface latent, using atmospheric inputs alone?

Implementation choices:
  - Aurora is fully frozen.
  - Inputs use only Aurora's pretrained surface vars: 2t, 10u, 10v, msl.
  - We tap the surface latent right before an existing surface head.
  - We fit a linear probe in closed form, streaming over the train split and
    accumulating normal equations instead of caching every latent to disk.
  - Metrics are reported on land only, in native units, on the same Stage 1
    train / val / test splits and per-timestep ERA5 layout.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from aurora import AuroraPretrained, AuroraSmallPretrained

from src.data import (
    BASE_SURF_VAR_NAMES,
    MultiRangeERA5Dataset,
    collate_era5_batch,
    era5_worker_init_fn,
    make_era5_splits,
)
from src.finetune_stage1 import _NEW_VAR_NORM

DEFAULT_TARGET_VARS = ("swvl1", "stl1", "sd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming Aurora linear-probe baseline")
    parser.add_argument(
        "--data-dir",
        nargs="+",
        required=True,
        help="ERA5 per-step year directories, e.g. /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025",
    )
    parser.add_argument(
        "--target-vars",
        nargs="+",
        default=list(DEFAULT_TARGET_VARS),
        help="Hydrology vars to decode. Default: swvl1 stl1 sd",
    )
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument(
        "--file-layout",
        choices=["per_timestep", "chunked"],
        default="per_timestep",
    )
    parser.add_argument("--static-path", type=Path, default=None)
    parser.add_argument("--small", action="store_true", help="Use AuroraSmallPretrained")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--surface-head-var", default="2t")
    parser.add_argument("--surface-head-module", default=None)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument(
        "--train-metrics-limit",
        type=int,
        default=0,
        help="Optional number of train samples to score after fitting. 0 skips train metrics.",
    )
    parser.add_argument("--print-decoder", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/linear_probe_streaming"))
    return parser.parse_args()


class SurfaceLatentTap:
    """Capture the input to an Aurora surface head."""

    def __init__(
        self,
        model: AuroraPretrained | AuroraSmallPretrained,
        target_hw: tuple[int, int],
        patch_size: int,
        surface_head_var: str,
        module_name: str | None = None,
    ):
        self.model = model
        self.target_hw = target_hw
        self.patch_size = patch_size
        self.surface_head_var = surface_head_var
        self.module_name = module_name or self._autodiscover_module_name()
        self.module = self._get_named_module(self.module_name)
        self.captured: torch.Tensor | None = None
        self.handle = self.module.register_forward_pre_hook(self._capture_input)

    def close(self) -> None:
        self.handle.remove()

    def _capture_input(self, module, inputs):
        if not inputs:
            raise RuntimeError("Surface head hook received no positional inputs.")
        self.captured = inputs[0].detach()

    def _get_named_module(self, name: str) -> nn.Module:
        for module_name, module in self.model.decoder.named_modules():
            if module_name == name:
                return module
        raise ValueError(f"Decoder module '{name}' not found.")

    def _autodiscover_module_name(self) -> str:
        candidates: list[str] = []
        preferred: list[str] = []
        for name, module in self.model.decoder.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == self.patch_size * self.patch_size:
                candidates.append(name)
                if self.surface_head_var in name:
                    preferred.append(name)
        if len(preferred) == 1:
            return preferred[0]
        if not preferred and len(candidates) == 1:
            return candidates[0]
        raise RuntimeError(
            "Could not uniquely auto-discover the Aurora surface head. "
            f"preferred={preferred} candidates={candidates}"
        )

    def get_surface_latent(self) -> torch.Tensor:
        if self.captured is None:
            raise RuntimeError("No latent captured. Run a forward pass first.")
        return self._canonicalize(self.captured)

    def _canonicalize(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            batch_size, n_tokens, n_levels, channels = latent.shape
            if n_levels != 1:
                raise ValueError(f"Expected singleton surface-level dim, found {n_levels}.")
            latent = latent.squeeze(2)
            return self._canonicalize(latent)
        if latent.ndim != 3:
            raise ValueError(f"Unexpected latent rank {latent.ndim}; expected 3 or 4.")

        batch_size, n_tokens, channels = latent.shape
        target_h, target_w = self.target_hw
        patch_w = target_w // self.patch_size
        patch_h = n_tokens // patch_w
        return latent.reshape(batch_size, patch_h, patch_w, channels).contiguous()


def make_loader(args: argparse.Namespace, dataset, shuffle: bool) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=era5_worker_init_fn,
        collate_fn=collate_era5_batch,
        pin_memory=(args.device == "cuda"),
    )
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def describe_batch(input_batch, target_batch) -> str:
    source_time = input_batch.metadata.time[0].isoformat()
    target_time = target_batch.metadata.time[0].isoformat()
    return f"source={source_time} target={target_time}"


def patchify_grid(grid: torch.Tensor, patch_h: int, patch_w: int, patch_size: int) -> torch.Tensor:
    """(B, H, W) -> (B * patch_h * patch_w, patch_size * patch_size)."""
    valid_h = patch_h * patch_size
    valid_w = patch_w * patch_size
    cropped = grid[:, :valid_h, :valid_w]
    return (
        cropped.reshape(grid.shape[0], patch_h, patch_size, patch_w, patch_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, patch_size * patch_size)
        .contiguous()
    )


def depatchify_grid(
    patches: torch.Tensor,
    batch_size: int,
    patch_h: int,
    patch_w: int,
    patch_size: int,
) -> torch.Tensor:
    """(B * patch_h * patch_w, patch_size * patch_size) -> (B, H, W)."""
    return (
        patches.reshape(batch_size, patch_h, patch_w, patch_size, patch_size)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size, patch_h * patch_size, patch_w * patch_size)
        .contiguous()
    )


def build_split_datasets(args: argparse.Namespace, target_vars: tuple[str, ...]):
    common = dict(
        data_dirs=[Path(p) for p in args.data_dir],
        step_hours=args.step_hours,
        include_extra_surf=False,
        rollout_steps=1,
        file_layout=args.file_layout,
        input_surf_vars=BASE_SURF_VAR_NAMES,
        target_surf_vars=target_vars,
        static_path=args.static_path,
    )
    return make_era5_splits(**common)


class StreamingProbeFitter:
    def __init__(
        self,
        target_vars: tuple[str, ...],
        norm_stats: dict[str, tuple[float, float]],
        ridge_lambda: float,
        patch_size: int,
    ):
        self.target_vars = target_vars
        self.norm_stats = norm_stats
        self.ridge_lambda = ridge_lambda
        self.patch_size = patch_size
        self.design_dim: int | None = None
        self.xxt: torch.Tensor | None = None
        self.xty: dict[str, torch.Tensor] = {}
        self.valid_patches = 0
        self.samples = 0

    def update(self, latent: torch.Tensor, target_batch, land_mask: torch.Tensor) -> None:
        batch_size, patch_h, patch_w, channels = latent.shape
        design_dim = channels + 1
        if self.design_dim is None:
            self.design_dim = design_dim
            self.xxt = torch.zeros(design_dim, design_dim, dtype=torch.float64)
            self.xty = {
                var: torch.zeros(design_dim, self.patch_size * self.patch_size, dtype=torch.float64)
                for var in self.target_vars
            }
        assert self.xxt is not None

        x = latent.reshape(-1, channels).float()
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        x_aug = torch.cat([x, ones], dim=1)

        patch_mask = patchify_grid(
            land_mask.to(device=latent.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1),
            patch_h=patch_h,
            patch_w=patch_w,
            patch_size=self.patch_size,
        )
        full_land = patch_mask.bool().all(dim=1)
        if not full_land.any():
            self.samples += batch_size
            return

        x_valid = x_aug[full_land]
        self.xxt += (x_valid.T @ x_valid).double().cpu()

        for var in self.target_vars:
            y = target_batch.surf_vars[var][:, -1].to(device=latent.device, dtype=torch.float32)
            y_patch = patchify_grid(y, patch_h=patch_h, patch_w=patch_w, patch_size=self.patch_size)
            mean, std = self.norm_stats[var]
            y_norm = (y_patch - mean) / std
            self.xty[var] += (x_valid.T @ y_norm[full_land]).double().cpu()

        self.valid_patches += int(full_land.sum().item())
        self.samples += batch_size

    def solve(self) -> dict[str, dict[str, torch.Tensor]]:
        if self.design_dim is None or self.xxt is None:
            raise RuntimeError("No training statistics accumulated; cannot solve probe.")

        reg = torch.eye(self.design_dim, dtype=torch.float64) * self.ridge_lambda
        reg[-1, -1] = 0.0  # leave the bias unregularised
        system = self.xxt + reg

        solved: dict[str, dict[str, torch.Tensor]] = {}
        for var in self.target_vars:
            coef = torch.linalg.solve(system, self.xty[var])
            solved[var] = {
                "weight": coef[:-1].float().contiguous(),
                "bias": coef[-1].float().contiguous(),
            }
        return solved


def predict_from_latent(
    latent: torch.Tensor,
    solved: dict[str, dict[str, torch.Tensor]],
    target_vars: tuple[str, ...],
    norm_stats: dict[str, tuple[float, float]],
    patch_size: int,
) -> dict[str, torch.Tensor]:
    batch_size, patch_h, patch_w, channels = latent.shape
    x = latent.reshape(-1, channels).float()
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    x_aug = torch.cat([x, ones], dim=1)

    preds: dict[str, torch.Tensor] = {}
    for var in target_vars:
        weight = solved[var]["weight"].to(device=latent.device, dtype=torch.float32)
        bias = solved[var]["bias"].to(device=latent.device, dtype=torch.float32)
        coef = torch.cat([weight, bias.unsqueeze(0)], dim=0)
        pred_norm = x_aug @ coef
        mean, std = norm_stats[var]
        pred = pred_norm * std + mean
        preds[var] = depatchify_grid(
            pred,
            batch_size=batch_size,
            patch_h=patch_h,
            patch_w=patch_w,
            patch_size=patch_size,
        )
    return preds


def fit_probe(
    args: argparse.Namespace,
    model,
    tap: SurfaceLatentTap,
    train_ds,
    target_vars: tuple[str, ...],
    land_mask: torch.Tensor,
    norm_stats: dict[str, tuple[float, float]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, float]]:
    loader = make_loader(args, train_ds, shuffle=False)
    fitter = StreamingProbeFitter(
        target_vars=target_vars,
        norm_stats=norm_stats,
        ridge_lambda=args.ridge_lambda,
        patch_size=args.patch_size,
    )
    n_seen = 0
    t0 = time.time()
    try:
        for input_batch, targets in loader:
            if args.train_limit is not None and n_seen >= args.train_limit:
                break
            target_batch = targets[0]
            for forbidden in target_vars:
                assert forbidden not in input_batch.surf_vars, f"{forbidden} leaked into inputs"
            with torch.inference_mode():
                model(input_batch.to(args.device))
            latent = tap.get_surface_latent()
            fitter.update(latent, target_batch, land_mask)
            n_seen += 1
            if n_seen == 1:
                print(
                    f"  fit: first sample | {describe_batch(input_batch, target_batch)} "
                    f"| latent_shape={tuple(latent.shape)}",
                    flush=True,
                )
            if n_seen % 25 == 0:
                elapsed = time.time() - t0
                rate = n_seen / max(elapsed, 1e-6)
                print(
                    f"  fit: processed {n_seen} train samples "
                    f"({rate:.2f} samples/s, valid patches={fitter.valid_patches}) "
                    f"| latest {describe_batch(input_batch, target_batch)}",
                    flush=True,
                )
    finally:
        train_ds.close()
        del loader

    solved = fitter.solve()
    fit_stats = {
        "train_samples_seen": n_seen,
        "valid_land_patches": fitter.valid_patches,
    }
    return solved, fit_stats


def evaluate_split(
    args: argparse.Namespace,
    split_name: str,
    model,
    tap: SurfaceLatentTap,
    dataset,
    target_vars: tuple[str, ...],
    land_mask: torch.Tensor,
    norm_stats: dict[str, tuple[float, float]],
    solved: dict[str, dict[str, torch.Tensor]],
    limit: int | None,
) -> dict[str, float]:
    loader = make_loader(args, dataset, shuffle=False)
    accum = {
        var: {"abs": 0.0, "sq": 0.0, "count": 0}
        for var in target_vars
    }
    n_seen = 0
    t0 = time.time()
    try:
        for input_batch, targets in loader:
            if limit is not None and n_seen >= limit:
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
                patch_size=args.patch_size,
            )

            valid_h = latent.shape[1] * args.patch_size
            valid_w = latent.shape[2] * args.patch_size
            eval_mask = land_mask[:valid_h, :valid_w].to(device=latent.device, dtype=torch.bool)

            for var in target_vars:
                target = target_batch.surf_vars[var][:, -1, :valid_h, :valid_w].to(
                    device=latent.device,
                    dtype=torch.float32,
                )
                diff = preds[var] - target
                masked = diff[:, eval_mask]
                accum[var]["abs"] += float(masked.abs().sum().item())
                accum[var]["sq"] += float(masked.square().sum().item())
                accum[var]["count"] += int(masked.numel())

            n_seen += 1
            if n_seen == 1:
                print(
                    f"  {split_name}: first sample | {describe_batch(input_batch, target_batch)} "
                    f"| latent_shape={tuple(latent.shape)}",
                    flush=True,
                )
            if n_seen % 25 == 0:
                elapsed = time.time() - t0
                print(
                    f"  {split_name}: evaluated {n_seen} samples "
                    f"({n_seen / max(elapsed, 1e-6):.2f} samples/s) "
                    f"| latest {describe_batch(input_batch, target_batch)}",
                    flush=True,
                )
    finally:
        dataset.close()
        del loader

    metrics: dict[str, float] = {f"{split_name}_samples": n_seen}
    for var in target_vars:
        count = max(accum[var]["count"], 1)
        metrics[f"{split_name}_{var}_mae"] = accum[var]["abs"] / count
        metrics[f"{split_name}_{var}_rmse"] = (accum[var]["sq"] / count) ** 0.5
    return metrics


def save_outputs(
    output_dir: Path,
    solved: dict[str, dict[str, torch.Tensor]],
    metrics: dict[str, float],
    fit_stats: dict[str, float],
    args: argparse.Namespace,
    surface_head_module: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    heads_payload = {
        "target_vars": list(solved.keys()),
        "patch_size": args.patch_size,
        "surface_head_module": surface_head_module,
        "ridge_lambda": args.ridge_lambda,
        "weights": {
            var: {"weight": solved[var]["weight"].cpu(), "bias": solved[var]["bias"].cpu()}
            for var in solved
        },
    }
    torch.save(heads_payload, output_dir / "heads.pt")

    metrics_payload = {
        "data_dirs": args.data_dir,
        "file_layout": args.file_layout,
        "target_vars": list(solved.keys()),
        "small": args.small,
        "surface_head_var": args.surface_head_var,
        "surface_head_module": surface_head_module,
        "ridge_lambda": args.ridge_lambda,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "fit_stats": fit_stats,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    target_vars = tuple(args.target_vars)
    norm_stats = {var: _NEW_VAR_NORM[var] for var in target_vars}

    print(f"Device: {args.device}")
    print(f"Data dirs: {args.data_dir}")
    print(f"File layout: {args.file_layout}")
    print(f"Target vars: {target_vars}")
    print(f"Loader workers: {args.num_workers}  prefetch_factor: {args.prefetch_factor}")
    print(f"Ridge lambda: {args.ridge_lambda}")

    train_ds, val_ds, test_ds = build_split_datasets(args, target_vars)
    print(
        f"Dataset sizes | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        flush=True,
    )

    aurora_cls = AuroraSmallPretrained if args.small else AuroraPretrained
    model = aurora_cls()
    print("Loading Aurora checkpoint...", flush=True)
    t0 = time.time()
    model.load_checkpoint()
    print(f"Checkpoint loaded in {time.time() - t0:.1f}s", flush=True)
    model.eval()
    model = model.to(args.device)
    for param in model.parameters():
        param.requires_grad_(False)
    assert all(not p.requires_grad for p in model.parameters())

    if args.print_decoder:
        print(model.decoder)

    first_input, first_targets = train_ds[0]
    first_target = first_targets[0]
    first_hw = (
        first_target.surf_vars[target_vars[0]].shape[-2],
        first_target.surf_vars[target_vars[0]].shape[-1],
    )
    land_mask = (first_target.static_vars["lsm"] > 0.5).cpu()
    tap = SurfaceLatentTap(
        model=model,
        target_hw=first_hw,
        patch_size=args.patch_size,
        surface_head_var=args.surface_head_var,
        module_name=args.surface_head_module,
    )
    print(f"Using surface head module: {tap.module_name}", flush=True)

    try:
        print("\nFitting streaming linear probe on train split...", flush=True)
        solved, fit_stats = fit_probe(
            args=args,
            model=model,
            tap=tap,
            train_ds=train_ds,
            target_vars=target_vars,
            land_mask=land_mask,
            norm_stats=norm_stats,
        )

        train_metrics: dict[str, float] = {}
        if args.train_metrics_limit != 0:
            print("\nEvaluating train split...", flush=True)
            train_metrics = evaluate_split(
                args=args,
                split_name="train",
                model=model,
                tap=tap,
                dataset=build_split_datasets(args, target_vars)[0],
                target_vars=target_vars,
                land_mask=land_mask,
                norm_stats=norm_stats,
                solved=solved,
                limit=None if args.train_metrics_limit is None else args.train_metrics_limit,
            )

        print("\nEvaluating val split...", flush=True)
        val_metrics = evaluate_split(
            args=args,
            split_name="val",
            model=model,
            tap=tap,
            dataset=val_ds,
            target_vars=target_vars,
            land_mask=land_mask,
            norm_stats=norm_stats,
            solved=solved,
            limit=args.val_limit,
        )

        print("\nEvaluating test split...", flush=True)
        test_metrics = evaluate_split(
            args=args,
            split_name="test",
            model=model,
            tap=tap,
            dataset=test_ds,
            target_vars=target_vars,
            land_mask=land_mask,
            norm_stats=norm_stats,
            solved=solved,
            limit=args.test_limit,
        )
    finally:
        tap.close()

    metrics = {}
    metrics.update(train_metrics)
    metrics.update(val_metrics)
    metrics.update(test_metrics)
    save_outputs(
        output_dir=output_dir,
        solved=solved,
        metrics=metrics,
        fit_stats=fit_stats,
        args=args,
        surface_head_module=tap.module_name,
    )

    print("\nDone.")
    print(json.dumps(metrics, indent=2))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
