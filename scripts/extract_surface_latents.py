"""Stage 1 feature extraction for the linear-probe hydrology baseline.

This script runs frozen Aurora on atmospheric inputs only, taps the surface
latent right before an existing surface head (preferably the `2t` head), and
writes one `(latent, targets)` sample per timestamp to disk.

Default split logic follows `docs/preprocessing_and_splits.md`:
  - train: Jun 1 - Aug 1 for 2024 and 2025
  - val:   Aug 1 - Aug 16 for 2024 and 2025
  - test:  Aug 16 - Sep 1 for 2024 and 2025

Each saved sample contains:
  - `latent`:   `(1, Hp, Wp, D)` surface latent in bf16 by default
  - `targets`:  `(3, H, W)` future `{swvl1, stl1, sd}` at `t+6h`
  - `source_time`: current Aurora input time `t`
  - `target_time`: future hydrology time `t+6h`
  - `year`: data directory year inferred from the path
  - `split`: train / val / test

Shared metadata is saved once to `<output-dir>/metadata.pt`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from aurora import AuroraPretrained, AuroraSmallPretrained

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import BASE_SURF_VAR_NAMES, ERA5Dataset

DEFAULT_TARGET_VARS = ("swvl1", "stl1", "sd")
DEFAULT_SPLIT_RANGES = {
    "train": ((6, 1, 8, 1),),
    "val": ((8, 1, 8, 16),),
    "test": ((8, 16, 9, 1),),
}


@dataclass(frozen=True)
class RangeSpec:
    year: int
    start: datetime
    end: datetime
    split: str
    data_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Aurora surface latents for linear probing")
    parser.add_argument(
        "--data-dir",
        action="append",
        required=True,
        help="ERA5 year directory. Pass once per year, e.g. /mnt/data/era5/2024",
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


def infer_year(data_dir: Path) -> int:
    try:
        return int(data_dir.name)
    except ValueError as exc:
        raise ValueError(f"Could not infer year from data dir {data_dir}") from exc


def build_default_ranges(data_dirs: list[Path], splits: list[str]) -> list[RangeSpec]:
    specs: list[RangeSpec] = []
    for data_dir in data_dirs:
        year = infer_year(data_dir)
        for split in splits:
            for start_month, start_day, end_month, end_day in DEFAULT_SPLIT_RANGES[split]:
                specs.append(
                    RangeSpec(
                        year=year,
                        start=datetime(year, start_month, start_day),
                        end=datetime(year, end_month, end_day),
                        split=split,
                        data_dir=data_dir,
                    )
                )
    return specs


def build_custom_ranges(data_dirs: list[Path], specs_raw: list[str]) -> list[RangeSpec]:
    dir_by_year = {infer_year(path): path for path in data_dirs}
    specs: list[RangeSpec] = []

    for raw in specs_raw:
        try:
            split, start_raw, end_raw = raw.split(":")
        except ValueError as exc:
            raise ValueError(
                f"Invalid --date-range '{raw}'. Expected split:YYYY-MM-DD:YYYY-MM-DD."
            ) from exc

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split '{split}' in --date-range '{raw}'.")

        start = datetime.fromisoformat(start_raw)
        end = datetime.fromisoformat(end_raw)
        if start >= end:
            raise ValueError(f"Invalid --date-range '{raw}': start must be before end.")
        if start.year != end.year:
            raise ValueError(
                f"Invalid --date-range '{raw}': start and end must stay within one year directory."
            )
        if start.year not in dir_by_year:
            raise ValueError(
                f"Invalid --date-range '{raw}': no data-dir provided for year {start.year}."
            )

        specs.append(
            RangeSpec(
                year=start.year,
                start=start,
                end=end,
                split=split,
                data_dir=dir_by_year[start.year],
            )
        )

    return specs


def make_dataset(spec: RangeSpec, step_hours: int, target_vars: tuple[str, ...]) -> ERA5Dataset:
    return ERA5Dataset(
        data_dirs=[spec.data_dir],
        start_date=spec.start,
        end_date=spec.end,
        step_hours=step_hours,
        include_extra_surf=True,
        input_surf_vars=BASE_SURF_VAR_NAMES,
        target_surf_vars=target_vars,
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

    if args.date_range:
        ranges = build_custom_ranges(data_dirs, args.date_range)
    else:
        ranges = build_default_ranges(data_dirs, args.split)
    for spec in ranges:
        print(
            f"  {spec.split:<5} year={spec.year} "
            f"range=[{spec.start.date()} -> {spec.end.date()}) dir={spec.data_dir}"
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

    metadata_written = False
    surface_head_module_name: str | None = None
    split_counts = {split: 0 for split in args.split}

    for spec in ranges:
        split_dir = output_dir / spec.split
        split_dir.mkdir(parents=True, exist_ok=True)

        dataset = make_dataset(spec, step_hours=args.step_hours, target_vars=target_vars)
        print(
            f"\nExtracting {spec.split} {spec.year}: "
            f"{len(dataset)} samples from [{spec.start} -> {spec.end})",
            flush=True,
        )

        tap: SurfaceLatentTap | None = None
        try:
            for idx in range(len(dataset)):
                if args.max_samples_per_split is not None and split_counts[spec.split] >= args.max_samples_per_split:
                    break

                input_batch, targets = dataset[idx]
                target_batch = targets[0]
                for forbidden in target_vars:
                    assert forbidden not in input_batch.surf_vars, f"{forbidden} leaked into inputs"

                target_time = target_batch.metadata.time[0]
                sample_name = f"{target_time.isoformat().replace(':', '-')}.pt"
                sample_path = split_dir / sample_name
                if sample_path.exists() and not args.overwrite:
                    split_counts[spec.split] += 1
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
                    split=spec.split,
                    year=spec.year,
                    latent_dtype=latent_dtype,
                    target_dtype=target_dtype,
                )
                split_counts[spec.split] += 1

                if split_counts[spec.split] % 25 == 0:
                    print(
                        f"  {spec.split}: wrote {split_counts[spec.split]} samples "
                        f"(latest {sample_name})",
                        flush=True,
                    )
        finally:
            dataset.close()
            if tap is not None:
                tap.close()

    manifest = {
        "data_dirs": [str(p) for p in data_dirs],
        "splits": args.split,
        "target_vars": list(target_vars),
        "step_hours": args.step_hours,
        "small": args.small,
        "surface_head_var": args.surface_head_var,
        "surface_head_module": surface_head_module_name,
        "latent_dtype": args.latent_dtype,
        "target_dtype": args.target_dtype,
        "counts": split_counts,
        "ranges": [
            {
                "split": spec.split,
                "year": spec.year,
                "data_dir": str(spec.data_dir),
                "start": spec.start.isoformat(),
                "end": spec.end.isoformat(),
            }
            for spec in ranges
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nExtraction complete.")
    print(json.dumps(split_counts, indent=2))
    print(f"Output written to {output_dir}")


if __name__ == "__main__":
    main()
