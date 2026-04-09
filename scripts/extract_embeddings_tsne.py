"""Extract Aurora encoder & bottleneck embeddings, then visualise with t-SNE.

Compares spatially-pooled embeddings from:
  - Full-globe ERA5 inputs
  - Regional crops of ERA5

Two embedding extraction points:
  1. After the Perceiver encoder  (shape: B, L', embed_dim)
  2. At the U-Net bottleneck       (shape: B, L_bottleneck, embed_dim * 4)

Embeddings are treated like feature maps: we pool over the spatial dimensions
and retain the latent (channel) dimensions for t-SNE. Two pooling granularities:
  - Per-sample: pool over all spatial+level tokens -> (B, D)
  - Per-level:  pool over spatial tokens per latent level -> (B*C, D)

Usage:
    python scripts/extract_embeddings_tsne.py \
        --data-dir data/era5 \
        --crop-lat 25 50 --crop-lon -130 -60   # North America crop example
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from sklearn.manifold import TSNE

from aurora import AuroraSmallPretrained, Batch, Metadata


# ---------------------------------------------------------------------------
# Hook-based embedding extractor
# ---------------------------------------------------------------------------

class EmbeddingExtractor:
    """Registers forward hooks on Aurora to capture intermediate representations."""

    def __init__(self, model: AuroraSmallPretrained):
        self.model = model
        self.embeddings: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def _encoder_hook(self, module, input, output):
        self.embeddings["encoder"] = output.detach().cpu()

    def _bottleneck_hook(self, module, input, output):
        # BasicLayer3D.forward returns (x_scaled, x_unscaled).
        # Last encoder layer has no downsample, so output = (x, None).
        x_bottleneck = output[0] if isinstance(output, tuple) else output
        self.embeddings["bottleneck"] = x_bottleneck.detach().cpu()

    def register(self):
        h1 = self.model.encoder.register_forward_hook(self._encoder_hook)
        h2 = self.model.backbone.encoder_layers[-1].register_forward_hook(
            self._bottleneck_hook
        )
        self._handles = [h1, h2]

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self.embeddings.clear()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_era5_batch(
    data_dir: Path,
    time_indices: tuple[int, int] = (0, 1),
) -> Batch:
    """Load a full-globe ERA5 batch from the downloaded NetCDF files."""
    static_ds = xr.open_dataset(data_dir / "static.nc", engine="netcdf4")
    surf_ds = xr.open_dataset(data_dir / "2023-01-01-surface-level.nc", engine="netcdf4")
    atmos_ds = xr.open_dataset(data_dir / "2023-01-01-atmospheric.nc", engine="netcdf4")

    t0, t1 = time_indices
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


def crop_batch(
    batch: Batch,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> Batch:
    """Crop an ERA5 batch to a lat/lon bounding box.

    ERA5 latitudes go 90 -> -90 (descending), longitudes 0 -> 359.75 (ascending).
    lat_range: (south, north), e.g. (25, 50)
    lon_range: (west, east), e.g. (-130, -60)  -- negative values are converted to 0-360
    """
    lat = batch.metadata.lat.numpy()
    lon = batch.metadata.lon.numpy()

    south, north = sorted(lat_range)
    west, east = lon_range
    if west < 0:
        west += 360
    if east < 0:
        east += 360

    lat_mask = (lat >= south) & (lat <= north)
    if west <= east:
        lon_mask = (lon >= west) & (lon <= east)
    else:
        lon_mask = (lon >= west) | (lon <= east)

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(
            f"Empty crop: lat_range={lat_range}, lon_range={lon_range}. "
            f"Got {len(lat_idx)} lat and {len(lon_idx)} lon points."
        )

    new_surf = {
        k: v[:, :, lat_idx[0] : lat_idx[-1] + 1, lon_idx[0] : lon_idx[-1] + 1]
        for k, v in batch.surf_vars.items()
    }
    new_static = {
        k: v[lat_idx[0] : lat_idx[-1] + 1, lon_idx[0] : lon_idx[-1] + 1]
        for k, v in batch.static_vars.items()
    }
    new_atmos = {
        k: v[:, :, :, lat_idx[0] : lat_idx[-1] + 1, lon_idx[0] : lon_idx[-1] + 1]
        for k, v in batch.atmos_vars.items()
    }

    return Batch(
        surf_vars=new_surf,
        static_vars=new_static,
        atmos_vars=new_atmos,
        metadata=Metadata(
            lat=torch.from_numpy(lat[lat_idx]),
            lon=torch.from_numpy(lon[lon_idx]),
            time=batch.metadata.time,
            atmos_levels=batch.metadata.atmos_levels,
        ),
    )


# ---------------------------------------------------------------------------
# Spatial pooling — treat embeddings like feature maps
# ---------------------------------------------------------------------------

def spatial_pool(emb: torch.Tensor, num_levels: int) -> np.ndarray:
    """Global average pool over spatial dims, returning one D-dim vector per sample.

    Input:  (B, L, D)  where L = num_levels * H_p * W_p
    Output: (B, D)     — spatial content removed, only latent features remain.
    """
    return emb.mean(dim=1).float().numpy()


def spatial_pool_per_level(emb: torch.Tensor, num_levels: int) -> np.ndarray:
    """Pool over spatial dims *per latent level*, giving one D-dim vector per level per sample.

    Input:  (B, L, D)  where L = num_levels * H_p * W_p
    Output: (B * num_levels, D)

    This gives more data points for t-SNE when the number of samples is small.
    """
    B, L, D = emb.shape
    spatial_per_level = L // num_levels
    # (B, num_levels, H_p*W_p, D)
    reshaped = emb.view(B, num_levels, spatial_per_level, D)
    # Pool over spatial tokens per level -> (B, num_levels, D)
    pooled = reshaped.mean(dim=2)
    return pooled.reshape(B * num_levels, D).float().numpy()


# ---------------------------------------------------------------------------
# Collect embeddings across multiple samples/batches
# ---------------------------------------------------------------------------

def collect_embeddings(
    model: AuroraSmallPretrained,
    extractor: EmbeddingExtractor,
    batches: list[Batch],
) -> dict[str, list[torch.Tensor]]:
    """Run each batch through the model and collect encoder + bottleneck embeddings."""
    all_embs: dict[str, list[torch.Tensor]] = {"encoder": [], "bottleneck": []}

    for batch in batches:
        extractor.clear()
        with torch.inference_mode():
            model(batch)

        for key in all_embs:
            all_embs[key].append(extractor.embeddings[key])

    return all_embs


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_tsne(
    embs_global: np.ndarray,
    embs_crop: np.ndarray,
    title: str,
    ax: plt.Axes,
    perplexity: float = 5.0,
):
    """Run t-SNE on combined spatially-pooled embeddings and scatter-plot them."""
    combined = np.concatenate([embs_global, embs_crop], axis=0)
    n_global = embs_global.shape[0]
    n_total = combined.shape[0]

    effective_perplexity = min(perplexity, max(1.0, (n_total - 1) / 3.0))

    tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42, init="pca")
    coords = tsne.fit_transform(combined)

    ax.scatter(
        coords[:n_global, 0],
        coords[:n_global, 1],
        c="steelblue",
        label="Global",
        alpha=0.7,
        s=60,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.scatter(
        coords[n_global:, 0],
        coords[n_global:, 1],
        c="coral",
        label="Regional crop",
        alpha=0.7,
        s=60,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_tsne_per_level(
    embs_global: np.ndarray,
    embs_crop: np.ndarray,
    num_levels: int,
    num_samples: int,
    title: str,
    ax: plt.Axes,
    perplexity: float = 5.0,
):
    """t-SNE on per-level pooled embeddings, coloured by source and shaped by level."""
    combined = np.concatenate([embs_global, embs_crop], axis=0)
    n_global = embs_global.shape[0]
    n_total = combined.shape[0]

    effective_perplexity = min(perplexity, max(1.0, (n_total - 1) / 3.0))

    tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42, init="pca")
    coords = tsne.fit_transform(combined)

    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    for lev_idx in range(num_levels):
        # Global points for this level
        g_indices = [s * num_levels + lev_idx for s in range(num_samples)]
        c_indices = [n_global + s * num_levels + lev_idx for s in range(num_samples)]

        label_g = f"Global L{lev_idx}" if lev_idx == 0 else f"_Global L{lev_idx}"
        label_c = f"Crop L{lev_idx}" if lev_idx == 0 else f"_Crop L{lev_idx}"
        marker = markers[lev_idx % len(markers)]

        ax.scatter(
            coords[g_indices, 0], coords[g_indices, 1],
            c="steelblue", marker=marker, s=80, alpha=0.8,
            edgecolors="white", linewidths=0.5, label=label_g,
        )
        ax.scatter(
            coords[c_indices, 0], coords[c_indices, 1],
            c="coral", marker=marker, s=80, alpha=0.8,
            edgecolors="white", linewidths=0.5, label=label_c,
        )

    # Legend: just show global vs crop colour distinction + level markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=8, label="Global"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="coral",
               markersize=8, label="Regional crop"),
    ]
    for lev_idx in range(num_levels):
        legend_elements.append(
            Line2D([0], [0], marker=markers[lev_idx % len(markers)], color="w",
                   markerfacecolor="gray", markersize=8, label=f"Level {lev_idx}")
        )
    ax.legend(handles=legend_elements, fontsize=9, loc="best")
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aurora embedding t-SNE comparison")
    parser.add_argument("--data-dir", type=Path, default=Path("data/era5"))
    parser.add_argument(
        "--crop-lat", type=float, nargs=2, default=[25.0, 50.0],
        help="Latitude range (south, north)",
    )
    parser.add_argument(
        "--crop-lon", type=float, nargs=2, default=[-130.0, -60.0],
        help="Longitude range (west, east); negatives converted to 0-360",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None, help="Save figure to this path")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Crop: lat={args.crop_lat}, lon={args.crop_lon}")

    # --- Load model ---
    print("Loading AuroraSmallPretrained...")
    model = AuroraSmallPretrained()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
    model.eval()
    model = model.to(args.device)

    num_levels = model.encoder.latent_levels

    extractor = EmbeddingExtractor(model)
    extractor.register()

    # --- Build sample batches ---
    # With 4 time steps (00, 06, 12, 18) we get pairs: (0,1), (1,2), (2,3).
    # This is just an initial test; scale up with more days for real analysis.
    time_pairs = [(0, 1), (1, 2), (2, 3)]

    print("Loading ERA5 data and building batches...")
    global_batches: list[Batch] = []
    crop_batches: list[Batch] = []

    for t0, t1 in time_pairs:
        full_batch = load_era5_batch(args.data_dir, time_indices=(t0, t1))
        global_batches.append(full_batch)
        crop_batches.append(crop_batch(full_batch, tuple(args.crop_lat), tuple(args.crop_lon)))

    # --- Extract embeddings ---
    print("Extracting global embeddings...")
    global_embs = collect_embeddings(model, extractor, global_batches)

    print("Extracting crop embeddings...")
    crop_embs = collect_embeddings(model, extractor, crop_batches)

    extractor.remove()

    # Stack: (N_samples, L, D)
    global_enc = torch.cat(global_embs["encoder"], dim=0)
    crop_enc = torch.cat(crop_embs["encoder"], dim=0)
    global_bot = torch.cat(global_embs["bottleneck"], dim=0)
    crop_bot = torch.cat(crop_embs["bottleneck"], dim=0)

    n_samples = len(time_pairs)

    print(f"Encoder embeddings  — global: {global_enc.shape}, crop: {crop_enc.shape}")
    print(f"Bottleneck embeddings — global: {global_bot.shape}, crop: {crop_bot.shape}")
    print(f"Latent levels: {num_levels}")

    # --- Spatial pooling ---
    # Per-sample: global avg pool over all tokens -> (N, D)
    global_enc_pooled = spatial_pool(global_enc, num_levels)
    crop_enc_pooled = spatial_pool(crop_enc, num_levels)
    global_bot_pooled = spatial_pool(global_bot, num_levels)
    crop_bot_pooled = spatial_pool(crop_bot, num_levels)

    # Per-level: pool spatial dims per latent level -> (N * num_levels, D)
    global_enc_perlev = spatial_pool_per_level(global_enc, num_levels)
    crop_enc_perlev = spatial_pool_per_level(crop_enc, num_levels)
    global_bot_perlev = spatial_pool_per_level(global_bot, num_levels)
    crop_bot_perlev = spatial_pool_per_level(crop_bot, num_levels)

    print(f"Per-sample pooled — encoder: {global_enc_pooled.shape}, bottleneck: {global_bot_pooled.shape}")
    print(f"Per-level pooled  — encoder: {global_enc_perlev.shape}, bottleneck: {global_bot_perlev.shape}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    print("Running t-SNE (per-sample, encoder)...")
    plot_tsne(
        global_enc_pooled, crop_enc_pooled,
        "Perceiver Encoder — Spatial Avg Pool (per sample)", axes[0, 0],
    )

    print("Running t-SNE (per-sample, bottleneck)...")
    plot_tsne(
        global_bot_pooled, crop_bot_pooled,
        "U-Net Bottleneck — Spatial Avg Pool (per sample)", axes[0, 1],
    )

    print("Running t-SNE (per-level, encoder)...")
    plot_tsne_per_level(
        global_enc_perlev, crop_enc_perlev,
        num_levels, n_samples,
        "Perceiver Encoder — Spatial Avg Pool (per level)", axes[1, 0],
    )

    print("Running t-SNE (per-level, bottleneck)...")
    plot_tsne_per_level(
        global_bot_perlev, crop_bot_perlev,
        num_levels, n_samples,
        "U-Net Bottleneck — Spatial Avg Pool (per level)", axes[1, 1],
    )

    fig.suptitle(
        "Aurora Embedding Distribution Shift: Global vs Regional Crop\n"
        f"Crop: lat {args.crop_lat}, lon {args.crop_lon}  |  "
        f"{n_samples} samples, {num_levels} latent levels",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
