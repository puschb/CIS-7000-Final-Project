"""Extract Aurora embeddings and visualise distribution shift with t-SNE.

For each ERA5 data pair (two consecutive time steps), run Aurora and extract
the embedding at the Perceiver encoder output and the U-Net bottleneck.
Spatial global average pool each embedding to a single D-dimensional vector.
Plot all samples in a t-SNE, coloured by global vs Rhine Valley crop.

Saves raw embeddings as .pt files and plots as .png to --output-dir.

Usage:
    python scripts/extract_embeddings_tsne.py \
        --data-dir /mnt/data/era5 --output-dir /mnt/results/embeddings
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from sklearn.manifold import TSNE

from aurora import AuroraSmallPretrained, Batch, Metadata

CROP_LAT = (46.0, 52.0)  # Rhine River Valley
CROP_LON = (6.0, 10.0)


# ---------------------------------------------------------------------------
# Hook-based embedding extractor
# ---------------------------------------------------------------------------

class EmbeddingExtractor:
    def __init__(self, model: AuroraSmallPretrained):
        self.model = model
        self.embeddings: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def _encoder_hook(self, module, input, output):
        self.embeddings["encoder"] = output.detach().cpu()

    def _bottleneck_hook(self, module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        self.embeddings["bottleneck"] = x.detach().cpu()

    def register(self):
        self._handles = [
            self.model.encoder.register_forward_hook(self._encoder_hook),
            self.model.backbone.encoder_layers[-1].register_forward_hook(self._bottleneck_hook),
        ]

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self.embeddings.clear()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_day_files(data_dir: Path) -> list[str]:
    """Return sorted list of date strings for which both surface and atmos files exist."""
    surf_files = sorted(data_dir.glob("*-surface-level.nc"))
    dates = []
    for f in surf_files:
        date_str = f.name.replace("-surface-level.nc", "")
        if (data_dir / f"{date_str}-atmospheric.nc").exists():
            dates.append(date_str)
    return dates


def load_batch(data_dir: Path, date_str: str, t0: int, t1: int) -> Batch:
    """Load one ERA5 sample (a pair of consecutive time steps)."""
    static_ds = xr.open_dataset(data_dir / "static.nc", engine="netcdf4")
    surf_ds = xr.open_dataset(data_dir / f"{date_str}-surface-level.nc", engine="netcdf4")
    atmos_ds = xr.open_dataset(data_dir / f"{date_str}-atmospheric.nc", engine="netcdf4")

    return Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_ds["t2m"].values[t0 : t1 + 1][None]),
            "10u": torch.from_numpy(surf_ds["u10"].values[t0 : t1 + 1][None]),
            "10v": torch.from_numpy(surf_ds["v10"].values[t0 : t1 + 1][None]),
            "msl": torch.from_numpy(surf_ds["msl"].values[t0 : t1 + 1][None]),
        },
        static_vars={
            "z": torch.from_numpy(static_ds["z"].values[0]),
            "slt": torch.from_numpy(static_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_ds["t"].values[t0 : t1 + 1][None]),
            "u": torch.from_numpy(atmos_ds["u"].values[t0 : t1 + 1][None]),
            "v": torch.from_numpy(atmos_ds["v"].values[t0 : t1 + 1][None]),
            "q": torch.from_numpy(atmos_ds["q"].values[t0 : t1 + 1][None]),
            "z": torch.from_numpy(atmos_ds["z"].values[t0 : t1 + 1][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_ds.latitude.values),
            lon=torch.from_numpy(surf_ds.longitude.values),
            time=(surf_ds.valid_time.values.astype("datetime64[s]").tolist()[t1],),
            atmos_levels=tuple(int(lev) for lev in atmos_ds.pressure_level.values),
        ),
    )


def crop_batch(batch: Batch) -> Batch:
    """Crop a batch to the Rhine Valley bounding box."""
    lat = batch.metadata.lat.numpy()
    lon = batch.metadata.lon.numpy()

    lat_mask = (lat >= CROP_LAT[0]) & (lat <= CROP_LAT[1])
    lon_mask = (lon >= CROP_LON[0]) & (lon <= CROP_LON[1])
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    li, lj = lat_idx[0], lat_idx[-1] + 1
    oi, oj = lon_idx[0], lon_idx[-1] + 1

    return Batch(
        surf_vars={k: v[:, :, li:lj, oi:oj] for k, v in batch.surf_vars.items()},
        static_vars={k: v[li:lj, oi:oj] for k, v in batch.static_vars.items()},
        atmos_vars={k: v[:, :, :, li:lj, oi:oj] for k, v in batch.atmos_vars.items()},
        metadata=Metadata(
            lat=torch.from_numpy(lat[lat_idx]),
            lon=torch.from_numpy(lon[lon_idx]),
            time=batch.metadata.time,
            atmos_levels=batch.metadata.atmos_levels,
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aurora embedding t-SNE")
    parser.add_argument("--data-dir", type=Path, default=Path("data/era5"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/embeddings"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dates = find_day_files(args.data_dir)
    if args.max_days:
        dates = dates[: args.max_days]
    total_days = len(dates)
    total_samples = total_days * 3
    print(f"Found {total_days} days -> {total_samples} samples", flush=True)
    print(f"Device: {args.device}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)

    # Load model
    print("Loading AuroraSmallPretrained...", flush=True)
    model = AuroraSmallPretrained()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
    model.eval()
    model = model.to(args.device)
    print("Model loaded.\n", flush=True)

    extractor = EmbeddingExtractor(model)
    extractor.register()

    time_pairs = [(0, 1), (1, 2), (2, 3)]

    global_enc_vecs = []
    global_bot_vecs = []
    crop_enc_vecs = []
    crop_bot_vecs = []

    t_start = time.time()
    sample_idx = 0

    for day_idx, date_str in enumerate(dates):
        t_day = time.time()

        for t0, t1 in time_pairs:
            batch = load_batch(args.data_dir, date_str, t0, t1)
            cropped = crop_batch(batch)

            extractor.clear()
            with torch.inference_mode():
                model(batch)
            global_enc_vecs.append(extractor.embeddings["encoder"].mean(dim=1))
            global_bot_vecs.append(extractor.embeddings["bottleneck"].mean(dim=1))

            extractor.clear()
            with torch.inference_mode():
                model(cropped)
            crop_enc_vecs.append(extractor.embeddings["encoder"].mean(dim=1))
            crop_bot_vecs.append(extractor.embeddings["bottleneck"].mean(dim=1))

            sample_idx += 1

        elapsed = time.time() - t_start
        day_time = time.time() - t_day
        done_days = day_idx + 1
        avg_per_day = elapsed / done_days
        eta = avg_per_day * (total_days - done_days)
        print(
            f"[{done_days:4d}/{total_days}] {date_str}  "
            f"{day_time:.1f}s/day  "
            f"{sample_idx}/{total_samples} samples  "
            f"elapsed {elapsed / 60:.1f}m  "
            f"ETA {eta / 60:.1f}m",
            flush=True,
        )

    extractor.remove()

    # Stack and save raw embeddings
    results = {
        "global_encoder": torch.cat(global_enc_vecs).float(),
        "global_bottleneck": torch.cat(global_bot_vecs).float(),
        "crop_encoder": torch.cat(crop_enc_vecs).float(),
        "crop_bottleneck": torch.cat(crop_bot_vecs).float(),
        "dates": dates,
        "crop_lat": CROP_LAT,
        "crop_lon": CROP_LON,
    }

    embeddings_path = args.output_dir / "embeddings.pt"
    torch.save(results, embeddings_path)
    print(f"\nRaw embeddings saved to {embeddings_path}", flush=True)
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}", flush=True)

    # t-SNE plot
    global_enc = results["global_encoder"].numpy()
    global_bot = results["global_bottleneck"].numpy()
    crop_enc = results["crop_encoder"].numpy()
    crop_bot = results["crop_bottleneck"].numpy()
    n_global = global_enc.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, g, c, title in [
        (axes[0], global_enc, crop_enc, "Perceiver Encoder"),
        (axes[1], global_bot, crop_bot, "U-Net Bottleneck"),
    ]:
        combined = np.concatenate([g, c], axis=0)
        perplexity = min(30.0, max(1.0, (combined.shape[0] - 1) / 3.0))

        coords = TSNE(
            n_components=2, perplexity=perplexity, random_state=42, init="pca",
        ).fit_transform(combined)

        ax.scatter(coords[:n_global, 0], coords[:n_global, 1],
                   c="steelblue", label="Global", alpha=0.7, s=40,
                   edgecolors="white", linewidths=0.3)
        ax.scatter(coords[n_global:, 0], coords[n_global:, 1],
                   c="coral", label="Rhine Valley crop", alpha=0.7, s=40,
                   edgecolors="white", linewidths=0.3)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Aurora Embedding Distribution Shift: Global vs Rhine Valley Crop  ({n_global} samples)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    plot_path = args.output_dir / "tsne.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}", flush=True)
    print(f"Total time: {(time.time() - t_start) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
