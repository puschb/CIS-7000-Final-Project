# %% [markdown]
# # Predictions for ERA5 using Aurora Small
# 
# Based on the [Aurora ERA5 example](https://microsoft.github.io/aurora/example_era5.html).
# 
# Downloads one day of ERA5 data (1 Jan 2023) at 0.25° resolution and runs
# `AuroraSmallPretrained` to produce a 2-step rollout (predicting hours 12:00 and 18:00
# from inputs at 00:00 and 06:00).
# 
# **Prerequisites:**
# - Register at the [Climate Data Store](https://cds.climate.copernicus.eu/)
# - Create `$HOME/.cdsapirc` with your API key:
#   ```
#   url: https://cds.climate.copernicus.eu/api
#   key: <your API key>
#   ```
# - Accept the [ERA5 terms of use](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)

# %% [markdown]
# ## 1. Download ERA5 Data

# %%
from pathlib import Path

import cdsapi

download_path = Path(__file__).resolve().parent.parent / "data" / "era5" if "__file__" in dir() else Path.cwd().parent / "data" / "era5"
download_path.mkdir(parents=True, exist_ok=True)
print(f"Download path: {download_path}")

c = cdsapi.Client()

# %%
# Static variables (geopotential, land-sea mask, soil type)
if not (download_path / "static.nc").exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["geopotential", "land_sea_mask", "soil_type"],
            "year": "2023",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
        },
        str(download_path / "static.nc"),
    )
print("Static variables downloaded!")

# %%
# Surface-level variables (4 time steps on 1 Jan 2023)
if not (download_path / "2023-01-01-surface-level.nc").exists():
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
            ],
            "year": "2023",
            "month": "01",
            "day": "01",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(download_path / "2023-01-01-surface-level.nc"),
    )
print("Surface-level variables downloaded!")

# %%
# Atmospheric variables (13 pressure levels, 4 time steps)
if not (download_path / "2023-01-01-atmospheric.nc").exists():
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "specific_humidity",
                "geopotential",
            ],
            "pressure_level": [
                "50", "100", "150", "200", "250", "300",
                "400", "500", "600", "700", "850", "925", "1000",
            ],
            "year": "2023",
            "month": "01",
            "day": "01",
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(download_path / "2023-01-01-atmospheric.nc"),
    )
print("Atmospheric variables downloaded!")

# %% [markdown]
# ## 2. Prepare a Batch
# 
# Convert the downloaded NetCDF files into an `aurora.Batch`.
# We use hours 00:00 and 06:00 as the two input time steps.

# %%
import torch
import xarray as xr

from aurora import Batch, Metadata

static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2023-01-01-atmospheric.nc", engine="netcdf4")

batch = Batch(
    surf_vars={
        "2t": torch.from_numpy(surf_vars_ds["t2m"].values[:2][None]),
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[:2][None]),
        "10v": torch.from_numpy(surf_vars_ds["v10"].values[:2][None]),
        "msl": torch.from_numpy(surf_vars_ds["msl"].values[:2][None]),
    },
    static_vars={
        "z": torch.from_numpy(static_vars_ds["z"].values[0]),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_vars_ds["t"].values[:2][None]),
        "u": torch.from_numpy(atmos_vars_ds["u"].values[:2][None]),
        "v": torch.from_numpy(atmos_vars_ds["v"].values[:2][None]),
        "q": torch.from_numpy(atmos_vars_ds["q"].values[:2][None]),
        "z": torch.from_numpy(atmos_vars_ds["z"].values[:2][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_vars_ds.latitude.values),
        lon=torch.from_numpy(surf_vars_ds.longitude.values),
        time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)

print(f"Surface vars:     {list(batch.surf_vars.keys())}")
print(f"Static vars:      {list(batch.static_vars.keys())}")
print(f"Atmospheric vars: {list(batch.atmos_vars.keys())}")
print(f"Grid shape:       {batch.surf_vars['2t'].shape}")
print(f"Pressure levels:  {batch.metadata.atmos_levels}")
print(f"Time:             {batch.metadata.time}")

# %% [markdown]
# ## 3. Load and Run the Model
# 
# We use `AuroraSmallPretrained` (smaller, faster, works on CPU or any GPU).
# The full `AuroraPretrained` model would give better predictions but requires
# an A100 80GB GPU for 0.25° global data.

# %%
import os
import time

import psutil

from aurora import AuroraSmallPretrained, rollout


def print_memory():
    proc = psutil.Process(os.getpid())
    ram_gb = proc.memory_info().rss / 1e9
    print(f"  System RAM:  {ram_gb:.2f} GB")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU memory:  {alloc:.2f} / {total:.1f} GB (peak: {peak:.2f} GB)")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

print("Before loading model:")
print_memory()

t0 = time.time()
model = AuroraSmallPretrained()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
model.eval()
model = model.to(device)
print(f"\nModel loaded in {time.time() - t0:.1f}s")
print_memory()

# %%
if torch.cuda.is_available():
    print("using cuda")
    torch.cuda.reset_peak_memory_stats()
else:
  print("using cpu")

print("HERE")

t0 = time.time()
with torch.inference_mode():
    preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]
print(f"Rollout completed in {time.time() - t0:.1f}s")
print(f"Predictions: {len(preds)} steps")
for i, p in enumerate(preds):
    print(f"  Step {i + 1}: time={p.metadata.time[0]}")

print("\nAfter rollout:")
print_memory()

# %% [markdown]
# ## 4. Visualize Predictions vs ERA5
# 
# Compare Aurora's predicted 2m temperature against the actual ERA5 data
# for hours 12:00 and 18:00.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

for i in range(2):
    pred = preds[i]

    ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
    ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
    if i == 0:
        ax[i, 0].set_title("Aurora Small Prediction")
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    ax[i, 1].imshow(surf_vars_ds["t2m"][2 + i].values - 273.15, vmin=-50, vmax=50)
    if i == 0:
        ax[i, 1].set_title("ERA5 Ground Truth")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Compute Error Metrics

# %%
for i, pred in enumerate(preds):
    pred_t2m = pred.surf_vars["2t"][0, 0].numpy()
    true_t2m = surf_vars_ds["t2m"][2 + i].values

    # Aurora may output a slightly different grid size; crop to match
    min_lat = min(pred_t2m.shape[0], true_t2m.shape[0])
    min_lon = min(pred_t2m.shape[1], true_t2m.shape[1])
    pred_t2m = pred_t2m[:min_lat, :min_lon]
    true_t2m = true_t2m[:min_lat, :min_lon]

    mae = abs(pred_t2m - true_t2m).mean()
    rmse = ((pred_t2m - true_t2m) ** 2).mean() ** 0.5
    bias = (pred_t2m - true_t2m).mean()

    print(f"Step {i + 1} ({pred.metadata.time[0]}):")
    print(f"  MAE:  {mae:.2f} K")
    print(f"  RMSE: {rmse:.2f} K")
    print(f"  Bias: {bias:+.2f} K")


