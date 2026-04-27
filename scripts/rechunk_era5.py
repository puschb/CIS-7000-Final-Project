"""
Rechunk a single ERA5 NetCDF file using xarray.

Reads SRC, loads the entire dataset into memory, then writes it out with
per-timestep chunking (1 × full_spatial) and zlib compression level 1.

Environment variables:
    SRC   path to input NetCDF file (typically on local scratch)
    DST   path to output NetCDF file (typically on local scratch)

Chunk scheme:
    Surface files     (valid_time, latitude, longitude):
        → (1, 721, 1440) per variable
    Atmospheric files (valid_time, pressure_level, latitude, longitude):
        → (1, 13, 721, 1440) per variable

Usage:
    SRC=/scratch/input.nc DST=/scratch/output.nc python3 rechunk_era5.py
"""

import os
import sys
import time

import xarray as xr

SRC = os.environ.get("SRC", "/scratch/input.nc")
DST = os.environ.get("DST", "/scratch/output.nc")

print(f"SRC: {SRC}")
print(f"DST: {DST}")
print()

t_total = time.time()

# --- Open ---
t0 = time.time()
ds = xr.open_dataset(SRC)
print(f"Opened in {time.time() - t0:.2f}s")
print(f"Variables: {list(ds.data_vars)}")
print(f"Dims:      {dict(ds.sizes)}")

is_atmospheric = "pressure_level" in ds.dims
target_chunks: dict[str, int]
if is_atmospheric:
    target_chunks = {
        "valid_time": 1,
        "pressure_level": ds.sizes["pressure_level"],
        "latitude": ds.sizes["latitude"],
        "longitude": ds.sizes["longitude"],
    }
else:
    target_chunks = {
        "valid_time": 1,
        "latitude": ds.sizes["latitude"],
        "longitude": ds.sizes["longitude"],
    }
print(f"Target chunks: {target_chunks}")
print()

# --- Load into RAM ---
t0 = time.time()
ds_loaded = ds.load()
t_load = time.time() - t0
mem_gb = sum(ds_loaded[v].nbytes for v in ds_loaded.data_vars) / 1e9
print(f"Loaded in {t_load:.1f}s  ({mem_gb:.1f} GB of array data)")
print()

# --- Write with new chunking ---
encoding: dict[str, dict] = {}
for var in ds_loaded.data_vars:
    dims = ds_loaded[var].dims
    chunks = tuple(target_chunks.get(d, ds_loaded.sizes[d]) for d in dims)
    encoding[var] = {
        "chunksizes": chunks,
        "zlib": True,
        "complevel": 1,
    }

t0 = time.time()
ds_loaded.to_netcdf(DST, encoding=encoding)
t_write = time.time() - t0

dst_size = os.path.getsize(DST)
print(f"Written in {t_write:.1f}s  ({dst_size / 1e6:.0f} MB)")
print()

ds.close()
ds_loaded.close()

t_elapsed = time.time() - t_total
print(f"Total: {t_elapsed:.1f}s  (load={t_load:.1f}s, write={t_write:.1f}s)")
