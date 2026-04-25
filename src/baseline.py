"""Baseline hydrology prediction from frozen Aurora features.

This baseline answers:
    How much future hydrology can Aurora already predict from its existing
    atmospheric state, without hydrology variables in the input?

Approach:
    1. Feed Aurora only the variables it was pretrained on.
    2. Freeze the Aurora model.
    3. Hook either the encoder or bottleneck features.
    4. Train a small readout head to predict future hydrology fields.

This gives a fair comparison point for later experiments that add hydrology
inputs and/or LoRA adapters.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from aurora import AuroraPretrained, AuroraSmallPretrained

from src.data import BASE_SURF_VAR_NAMES, ERA5Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-Aurora hydrology baseline")
    parser.add_argument(
        "--data-dir",
        action="append",
        required=True,
        help="ERA5 directory. Pass multiple times for multiple yearly folders.",
    )
    parser.add_argument("--train-start", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--train-end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--val-end", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--target-vars",
        nargs="+",
        default=["swvl1"],
        help="Future surface vars to predict. Example: swvl1 stl1 sd",
    )
    parser.add_argument(
        "--feature-source",
        choices=["encoder", "bottleneck"],
        default="encoder",
        help="Which frozen Aurora representation to read out from.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument("--small", action="store_true", help="Use AuroraSmallPretrained")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
    return parser.parse_args()


def parse_date(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def extract_targets(batch, target_vars: list[str], device: str) -> torch.Tensor:
    tensors = [batch.surf_vars[name][:, -1].to(device) for name in target_vars]
    return torch.stack(tensors, dim=1)


class SurfaceReadoutHead(nn.Module):
    """Project a frozen Aurora feature grid to surface hydrology variables."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, feature_grid: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        x = self.net(feature_grid)
        return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)


class FrozenAuroraHydrologyBaseline(nn.Module):
    def __init__(
        self,
        aurora_model: AuroraPretrained | AuroraSmallPretrained,
        feature_source: str,
        out_channels: int,
        hidden_channels: int,
        patch_size: int = 4,
    ):
        super().__init__()
        self.aurora = aurora_model
        self.feature_source = feature_source
        self.patch_size = patch_size
        self.latent_levels = int(getattr(self.aurora.encoder, "latent_levels", 4))
        self._captured_feature: torch.Tensor | None = None

        for param in self.aurora.parameters():
            param.requires_grad_(False)

        if feature_source == "encoder":
            handle = self.aurora.encoder.register_forward_hook(self._feature_hook)
            in_channels = int(getattr(self.aurora.encoder, "embed_dim", 256))
            self._downsample_factor = self.patch_size
        else:
            handle = self.aurora.backbone.encoder_layers[-1].register_forward_hook(
                self._feature_hook
            )
            encoder_dim = int(getattr(self.aurora.encoder, "embed_dim", 256))
            in_channels = encoder_dim * 4
            self._downsample_factor = self.patch_size * 4

        self._handle = handle
        self.readout = SurfaceReadoutHead(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )

    def close(self):
        self._handle.remove()

    def _feature_hook(self, module, inputs, output):
        feature = output[0] if isinstance(output, tuple) else output
        self._captured_feature = feature

    def _tokens_to_grid(
        self,
        tokens: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        batch_size, n_tokens, channels = tokens.shape
        target_h, target_w = target_hw
        patch_w = target_w // self._downsample_factor
        if patch_w <= 0:
            raise ValueError(f"Invalid target width {target_w} for factor {self._downsample_factor}")

        patch_h = n_tokens // (self.latent_levels * patch_w)
        expected_tokens = self.latent_levels * patch_h * patch_w
        if expected_tokens != n_tokens:
            raise ValueError(
                "Cannot reshape Aurora tokens to spatial grid. "
                f"tokens={n_tokens} levels={self.latent_levels} patch_h={patch_h} patch_w={patch_w}"
            )

        x = tokens.reshape(batch_size, self.latent_levels, patch_h, patch_w, channels)
        x = x.mean(dim=1)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, batch, target_hw: tuple[int, int]) -> torch.Tensor:
        self._captured_feature = None
        with torch.no_grad():
            self.aurora(batch)

        if self._captured_feature is None:
            raise RuntimeError("Aurora feature hook did not capture any activations.")

        feature_grid = self._tokens_to_grid(self._captured_feature, target_hw)
        return self.readout(feature_grid, target_hw)


def make_dataset(
    data_dirs: list[str],
    start_date: datetime | None,
    end_date: datetime | None,
    step_hours: int,
    target_vars: list[str],
) -> ERA5Dataset:
    return ERA5Dataset(
        data_dirs=data_dirs,
        start_date=start_date,
        end_date=end_date,
        step_hours=step_hours,
        include_extra_surf=True,
        input_surf_vars=BASE_SURF_VAR_NAMES,
        target_surf_vars=tuple(target_vars),
    )


def iter_indices(length: int, limit: int | None, shuffle: bool) -> list[int]:
    indices = list(range(length))
    if shuffle:
        random.shuffle(indices)
    if limit is not None:
        indices = indices[:limit]
    return indices


def run_epoch(
    model: FrozenAuroraHydrologyBaseline,
    dataset: ERA5Dataset,
    optimizer: torch.optim.Optimizer | None,
    target_vars: list[str],
    device: str,
    limit: int | None,
    shuffle: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    indices = iter_indices(len(dataset), limit=limit, shuffle=shuffle)
    if not indices:
        raise ValueError("Dataset split is empty. Adjust your date ranges.")

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    n_steps = 0

    for idx in indices:
        input_batch, target_batch = dataset[idx]
        target = extract_targets(target_batch, target_vars, device=device)
        target_hw = (target.shape[-2], target.shape[-1])

        pred = model(input_batch, target_hw=target_hw)
        loss = F.mse_loss(pred, target)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mae = F.l1_loss(pred, target)
        rmse = torch.sqrt(loss.detach())

        total_loss += float(loss.detach().cpu())
        total_mae += float(mae.detach().cpu())
        total_rmse += float(rmse.detach().cpu())
        n_steps += 1

    return {
        "loss": total_loss / n_steps,
        "mae": total_mae / n_steps,
        "rmse": total_rmse / n_steps,
        "samples": n_steps,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_start = parse_date(args.train_start)
    train_end = parse_date(args.train_end)
    val_end = parse_date(args.val_end)
    if train_end is None or val_end is None:
        raise ValueError("--train-end and --val-end are required.")

    print(f"Device: {args.device}")
    print(f"Feature source: {args.feature_source}")
    print(f"Target vars: {args.target_vars}")
    print(f"Data dirs: {args.data_dir}")

    train_ds = make_dataset(args.data_dir, train_start, train_end, args.step_hours, args.target_vars)
    val_ds = make_dataset(args.data_dir, train_end, val_end, args.step_hours, args.target_vars)
    test_ds = make_dataset(args.data_dir, val_end, None, args.step_hours, args.target_vars)

    print(
        f"Dataset sizes | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
        flush=True,
    )

    aurora_cls = AuroraSmallPretrained if args.small else AuroraPretrained
    aurora = aurora_cls()

    print("Loading Aurora checkpoint...", flush=True)
    t0 = time.time()
    aurora.load_checkpoint()
    print(f"Checkpoint loaded in {time.time() - t0:.1f}s", flush=True)

    aurora.eval()
    aurora = aurora.to(args.device)

    model = FrozenAuroraHydrologyBaseline(
        aurora_model=aurora,
        feature_source=args.feature_source,
        out_channels=len(args.target_vars),
        hidden_channels=args.hidden_channels,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.readout.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = run_epoch(
            model=model,
            dataset=train_ds,
            optimizer=optimizer,
            target_vars=args.target_vars,
            device=args.device,
            limit=args.train_limit,
            shuffle=True,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                dataset=val_ds,
                optimizer=None,
                target_vars=args.target_vars,
                device=args.device,
                limit=args.val_limit,
                shuffle=False,
            )

        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2), flush=True)

    with torch.no_grad():
        test_metrics = run_epoch(
            model=model,
            dataset=test_ds,
            optimizer=None,
            target_vars=args.target_vars,
            device=args.device,
            limit=args.test_limit,
            shuffle=False,
        )

    metrics = {
        "config": {
            "data_dir": args.data_dir,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_end": args.val_end,
            "target_vars": args.target_vars,
            "feature_source": args.feature_source,
            "small": args.small,
            "step_hours": args.step_hours,
        },
        "history": history,
        "test": test_metrics,
    }

    metrics_path = args.output_dir / "metrics.json"
    head_path = args.output_dir / "hydrology_head.pt"
    torch.save(model.readout.state_dict(), head_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved head to {head_path}")
    print(f"Saved metrics to {metrics_path}")

    model.close()
    train_ds.close()
    val_ds.close()
    test_ds.close()


if __name__ == "__main__":
    main()
