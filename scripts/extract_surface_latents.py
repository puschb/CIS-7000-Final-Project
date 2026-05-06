"""Extract frozen Aurora surface latents for the hydrology linear probe.

This follows the same data path as Stage 1 fine-tuning:
  - per-timestep ERA5 files under ``/mnt/data/era5/per-step/{2024,2025}``
  - default train / val / test splits from ``src.data.make_era5_splits``
  - multi-worker DataLoader with the same worker-init / collate helpers

Aurora itself still receives only its pretrained surface inputs
(`2t`, `10u`, `10v`, `msl`). The future hydrology fields
(`swvl1`, `stl1`, `sd`) are saved only as targets for the linear probe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from aurora import AuroraPretrained, AuroraSmallPretrained

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    BASE_SURF_VAR_NAMES,
    MultiRangeERA5Dataset,
    collate_era5_batch,
    era5_worker_init_fn,
    make_era5_splits,
)

DEFAULT_TARGET_VARS = ("swvl1", "stl1", "sd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Aurora surface latents for linear probing")
    parser.add_argument(
        "--data-dir",
        nargs="+",
        required=True,
        help="One or more ERA5 year directories, e.g. /mnt/data/era5/per-step/2024 /mnt/data/era5/per-step/2025",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Which default splits to extract.",
    )
    parser.add_argument(
        "--date-range",
        action="append",
        default=[],
        help=(
            "Custom split range in the form split:YYYY-MM-DD:YYYY-MM-DD. "
            "Can be passed multiple times. If provided, these ranges are used "
            "instead of the default summer splits."
        ),
    )
    parser.add_argument(
        "--target-vars",
        nargs="+",
        default=list(DEFAULT_TARGET_VARS),
        help="Future surface variables to save as targets.",
    )
    parser.add_argument("--step-hours", type=int, default=6)
    parser.add_argument(
        "--file-layout",
        choices=["per_timestep", "chunked"],
        default="per_timestep",
        help="Use the same per-step layout as Stage 1 by default.",
    )
    parser.add_argument(
        "--static-path",
        type=Path,
        default=None,
        help="Optional explicit path to static.nc if it is not inside each year directory.",
    )
    parser.add_argument("--small", action="store_true", help="Use AuroraSmallPretrained")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--surface-head-var",
        default="2t",
        help="Existing Aurora surface head to hook. Usually 2t.",
    )
    parser.add_argument(
        "--surface-head-module",
        default=None,
        help="Explicit named_modules path for the hooked surface head. If omitted, autodiscover.",
    )
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--latent-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Storage dtype for cached latents.",
    )
    parser.add_argument(
        "--target-dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Storage dtype for cached targets.",
    )
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample files.")
    parser.add_argument("--print-decoder", action="store_true", help="Print model.decoder for debugging.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/linear_probe_features"))
    return parser.parse_args()


def storage_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _parse_custom_ranges(specs_raw: list[str]) -> dict[str, list[tuple[datetime, datetime]]]:
    split_ranges: dict[str, list[tuple[datetime, datetime]]] = {"train": [], "val": [], "test": []}
    for raw in specs_raw:
        try:
            split, start_raw, end_raw = raw.split(":")
        except ValueError as exc:
            raise ValueError(
                f"Invalid --date-range '{raw}'. Expected split:YYYY-MM-DD:YYYY-MM-DD."
            ) from exc
        if split not in split_ranges:
            raise ValueError(f"Invalid split '{split}' in --date-range '{raw}'.")
        start = datetime.fromisoformat(start_raw)
        end = datetime.fromisoformat(end_raw)
        if start >= end:
            raise ValueError(f"Invalid --date-range '{raw}': start must be before end.")
        split_ranges[split].append((start, end))
    return split_ranges


def build_split_datasets(args: argparse.Namespace, data_dirs: list[Path], target_vars: tuple[str, ...]):
    common_kwargs = dict(
        data_dirs=data_dirs,
        step_hours=args.step_hours,
        include_extra_surf=False,
        rollout_steps=1,
        file_layout=args.file_layout,
        input_surf_vars=BASE_SURF_VAR_NAMES,
        target_surf_vars=target_vars,
        static_path=args.static_path,
    )
    if args.date_range:
        split_ranges = _parse_custom_ranges(args.date_range)
        datasets = {
            split: MultiRangeERA5Dataset(date_ranges=ranges, **common_kwargs)
            for split, ranges in split_ranges.items()
            if split in args.split and ranges
        }
        manifest_ranges = {
            split: [
                {"start": start.isoformat(), "end": end.isoformat()}
                for start, end in ranges
            ]
            for split, ranges in split_ranges.items()
            if split in args.split and ranges
        }
        return datasets, manifest_ranges, False

    train_ds, val_ds, test_ds = make_era5_splits(**common_kwargs)
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    return (
        {split: datasets[split] for split in args.split},
        {},
        True,
    )


class SurfaceLatentTap:
    """Capture the input to an existing Aurora surface head."""

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

    def close(self):
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

        details = {
            "surface_head_var": self.surface_head_var,
            "preferred_candidates": preferred,
            "all_patch_candidates": candidates,
        }
        raise RuntimeError(
            "Could not uniquely auto-discover the Aurora surface head. "
            "Pass --surface-head-module explicitly. "
            + json.dumps(details, indent=2)
        )

    def get_surface_latent(self) -> torch.Tensor:
        if self.captured is None:
            raise RuntimeError("No latent captured. Run the Aurora forward pass first.")
        return self._canonicalize(self.captured)

    def _canonicalize(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            batch_size, n_tokens, n_levels, channels = latent.shape
            if n_levels != 1:
                raise ValueError(
                    f"Expected singleton surface-level dim in tapped latent, found {n_levels}."
                )
            latent = latent.squeeze(2)
            return self._canonicalize(latent)

        if latent.ndim != 3:
            raise ValueError(f"Unexpected surface latent rank {latent.ndim}; expected 3 or 4.")

        batch_size, n_tokens, channels = latent.shape
        target_h, target_w = self.target_hw
        patch_w = target_w // self.patch_size
        if patch_w <= 0 or n_tokens % patch_w != 0:
            raise ValueError(
                f"Cannot reshape latent: tokens={n_tokens}, patch_w={patch_w}, target_hw={self.target_hw}"
            )
        patch_h = n_tokens // patch_w

        if target_h not in {patch_h * self.patch_size, patch_h * self.patch_size + 1}:
            raise ValueError(
                "Unexpected target height for patchified latent. "
                f"target_h={target_h}, patch_h={patch_h}, patch_size={self.patch_size}"
            )

        return latent.reshape(batch_size, patch_h, patch_w, channels).contiguous()


def ensure_shared_metadata(
    output_dir: Path,
    target_batch,
    target_vars: tuple[str, ...],
    patch_size: int,
    surface_head_module: str,
):
    metadata_path = output_dir / "metadata.pt"
    if metadata_path.exists():
        return

    metadata = {
        "target_vars": list(target_vars),
        "patch_size": patch_size,
        "surface_head_module": surface_head_module,
        "lat": target_batch.metadata.lat.cpu(),
        "lon": target_batch.metadata.lon.cpu(),
        "land_mask": (target_batch.static_vars["lsm"] > 0.5).cpu(),
        "atmos_levels": tuple(int(x) for x in target_batch.metadata.atmos_levels),
    }
    torch.save(metadata, metadata_path)


def save_sample(
    path: Path,
    latent: torch.Tensor,
    source_time: datetime,
    target_batch,
    target_vars: tuple[str, ...],
    split: str,
    year: int,
    latent_dtype: torch.dtype,
    target_dtype: torch.dtype,
):
    targets = torch.stack([target_batch.surf_vars[name][:, -1] for name in target_vars], dim=1)
    payload = {
        "latent": latent.to(dtype=latent_dtype).cpu(),
        "targets": targets.to(dtype=target_dtype).cpu(),
        "split": split,
        "year": year,
        "source_time": source_time.isoformat(),
        "target_time": target_batch.metadata.time[0].isoformat(),
    }
    torch.save(payload, path)


def make_loader(args: argparse.Namespace, dataset) -> DataLoader:
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=era5_worker_init_fn,
        collate_fn=collate_era5_batch,
        pin_memory=(args.device == "cuda"),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**loader_kwargs)


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dirs = [Path(p) for p in args.data_dir]
    target_vars = tuple(args.target_vars)
    latent_dtype = storage_dtype(args.latent_dtype)
    target_dtype = storage_dtype(args.target_dtype)

    print(f"Device: {args.device}")
    print(f"Data dirs: {[str(p) for p in data_dirs]}")
    print(f"Splits: {args.split}")
    print(f"Target vars: {target_vars}")
    print(f"File layout: {args.file_layout}")
    print(f"Loader workers: {args.num_workers}  prefetch_factor: {args.prefetch_factor}")
    if args.static_path is not None:
        print(f"Static path override: {args.static_path}")
    split_datasets, manifest_ranges, used_default_splits = build_split_datasets(args, data_dirs, target_vars)
    for split, dataset in split_datasets.items():
        print(f"  {split:<5} samples={len(dataset)}")

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

    metadata_written = False
    surface_head_module_name: str | None = None
    split_counts = {split: 0 for split in args.split}
    skipped_existing = {split: 0 for split in args.split}

    for split, dataset in split_datasets.items():
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        loader = make_loader(args, dataset)
        print(
            f"\nExtracting {split}: {len(dataset)} samples",
            flush=True,
        )

        tap: SurfaceLatentTap | None = None
        try:
            for input_batch, targets in loader:
                if args.max_samples_per_split is not None and split_counts[split] >= args.max_samples_per_split:
                    break

                target_batch = targets[0]
                for forbidden in target_vars:
                    assert forbidden not in input_batch.surf_vars, f"{forbidden} leaked into inputs"

                target_time = target_batch.metadata.time[0]
                sample_name = f"{target_time.isoformat().replace(':', '-')}.pt"
                sample_path = split_dir / sample_name
                if sample_path.exists() and not args.overwrite:
                    skipped_existing[split] += 1
                    continue

                if tap is None:
                    first_target = target_batch.surf_vars[target_vars[0]][:, -1]
                    tap = SurfaceLatentTap(
                        model=model,
                        target_hw=(first_target.shape[-2], first_target.shape[-1]),
                        patch_size=args.patch_size,
                        surface_head_var=args.surface_head_var,
                        module_name=args.surface_head_module,
                    )
                    surface_head_module_name = tap.module_name
                    print(f"Using surface head module: {surface_head_module_name}", flush=True)

                with torch.inference_mode():
                    model(input_batch.to(args.device))

                latent = tap.get_surface_latent()

                if not metadata_written:
                    ensure_shared_metadata(
                        output_dir=output_dir,
                        target_batch=target_batch,
                        target_vars=target_vars,
                        patch_size=args.patch_size,
                        surface_head_module=surface_head_module_name or "unknown",
                    )
                    metadata_written = True

                save_sample(
                    path=sample_path,
                    latent=latent,
                    source_time=input_batch.metadata.time[0],
                    target_batch=target_batch,
                    target_vars=target_vars,
                    split=split,
                    year=target_time.year,
                    latent_dtype=latent_dtype,
                    target_dtype=target_dtype,
                )
                split_counts[split] += 1

                if split_counts[split] % 25 == 0:
                    print(
                        f"  {split}: wrote {split_counts[split]} samples "
                        f"(latest {sample_name})",
                        flush=True,
                    )
        finally:
            dataset.close()
            del loader
            if tap is not None:
                tap.close()

    manifest = {
        "data_dirs": [str(p) for p in data_dirs],
        "splits": args.split,
        "target_vars": list(target_vars),
        "step_hours": args.step_hours,
        "small": args.small,
        "file_layout": args.file_layout,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "surface_head_var": args.surface_head_var,
        "surface_head_module": surface_head_module_name,
        "latent_dtype": args.latent_dtype,
        "target_dtype": args.target_dtype,
        "counts": split_counts,
        "skipped_existing": skipped_existing,
        "used_default_stage1_splits": used_default_splits,
        "ranges": manifest_ranges,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nExtraction complete.")
    print(json.dumps(split_counts, indent=2))
    print(f"Output written to {output_dir}")


if __name__ == "__main__":
    main()
