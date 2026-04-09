# Regional Cropping & Embedding Comparison in Aurora

## Architecture Summary

Aurora is an encoder → backbone → decoder model:

| Component | Role | Output shape (0.25°, AuroraSmall) |
|---|---|---|
| **Perceiver Encoder** | Patches input variables, aggregates pressure levels into `L` latent levels | `(B, L·Hp·Wp, D)` = `(1, 4·180·360, 256)` |
| **Swin3D U-Net Backbone** | 3-stage encoder (downsample 4×) → bottleneck → 3-stage decoder (upsample) | bottleneck: `(1, 4·45·90, 1024)` |
| **Perceiver Decoder** | De-aggregates latent levels back to physical variables | `Batch` |

Key dimensions for `AuroraSmallPretrained`: `embed_dim=256`, `latent_levels=4`, `patch_size=4`, backbone stages `(2,6,2)+(2,6,2)`.

## 1. Regional Cropping

### Why it works

- **Geometry-aware positional encoding**: Fourier expansion of actual lat/lon, not grid indices.
- **Scale encoding**: Each patch knows its physical area (km²).
- **No fixed positional biases** in backbone attention — no assumption about input size.
- `Batch` takes arbitrary `lat`/`lon` tensors in `Metadata`.

### How to crop

Slice the ERA5 arrays and build a `Batch` with the sub-region's lat/lon. The only constraints:
- **Width** (lon points) must be divisible by `patch_size` (4).
- **Height** (lat points) must satisfy `h % 4 ∈ {0, 1}` (`Batch.crop()` handles the `==1` case).

See `notebooks/era5_download_crop.ipynb` — it auto-trims to valid dimensions.

### Caveats

1. **Lost global context**: model was pretrained on full-globe data. Jet streams, teleconnections, etc. won't be captured in a regional crop. Expect degraded predictions at longer rollouts.
2. **Swin window boundaries**: backbone uses `(2,12,6)` windows. Small crops may have very few windows.
3. **Edge effects**: global model wraps longitude; a crop breaks this wrapping.
4. **Normalization**: uses global ERA5 statistics — fine for most mid-latitude regions.

## 2. Extracting Embeddings

The forward pass (`aurora/model/aurora.py`) is cleanly separated:

```python
x = self.encoder(batch, lead_time=...)          # encoder output
x = self.backbone(x, lead_time=..., patch_res=...)  # backbone processes
pred = self.decoder(x, batch, ...)              # back to weather vars
```

### Using forward hooks (recommended)

```python
captured = {}

def hook(name):
    def fn(mod, inp, out):
        captured[name] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
    return fn

model.encoder.register_forward_hook(hook("encoder"))
model.backbone.encoder_layers[-1].register_forward_hook(hook("bottleneck"))

with torch.inference_mode():
    model(batch)

encoder_emb    = captured["encoder"]     # (1, L*Hp*Wp, D)
bottleneck_emb = captured["bottleneck"]  # (1, L*Hp/4*Wp/4, 4D)
```

### Reshaping to spatial grids

```python
L = model.encoder.latent_levels  # 4
encoder_grid    = encoder_emb[0].reshape(L, Hp, Wp, D)
bottleneck_grid = bottleneck_emb[0].reshape(L, Hp//4, Wp//4, 4*D)
```

Token ordering is `(level, lat_patch, lon_patch)` — level varies slowest.

## 3. Comparing Global vs Cropped Embeddings

### Approach

1. Run **global** ERA5 through the model → extract embeddings → reshape to spatial grid → slice the sub-region.
2. Run **cropped** ERA5 through the model → extract embeddings.
3. Compare the two sets for the same geographic region.

### Spatial alignment

The crop region starts at some pixel offset `(lat_px, lon_px)` in the global grid. Divide by `patch_size` (and by 4 again for the bottleneck) to get patch-level offsets:

```python
enc_offset  = (lat_px // 4,  lon_px // 4)
bot_offset  = (lat_px // 16, lon_px // 16)
```

For clean bottleneck alignment, ensure the crop boundaries fall on 4°-aligned boundaries (multiples of 16 pixels at 0.25°). The default region in the notebook (26°N–50°N, 260°E–300°E) satisfies this.

### Comparison metrics

| Metric | What it captures |
|---|---|
| **Cosine similarity** | Direction alignment per patch |
| **MSE** | Magnitude differences |
| **t-SNE** | Whether global/cropped form separate clusters (structural difference) |
| **Spatial heatmap** | Where in the region embeddings diverge most (likely edges) |

### Expected results

- **Encoder embeddings** operate per-patch with limited spatial mixing → should be **very similar** (high cosine similarity).
- **Bottleneck embeddings** propagate information across many patches via 48 Swin attention layers → should show **more divergence**, especially near region edges where the global model has context the crop does not.

## 4. Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/era5_download_crop.ipynb` | Download ERA5, build global + cropped `Batch`, save as NetCDF |
| `notebooks/embedding_comparison.ipynb` | Load batches, extract encoder/bottleneck embeddings via hooks, t-SNE + cosine similarity + spatial heatmaps |

Run them in order. Both are configurable — edit the first code cell to change the date or region.
