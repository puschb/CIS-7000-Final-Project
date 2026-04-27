import time
import xarray as xr
import numpy as np

DATA = "/mnt/data/era5"

print("=" * 60)
print("TEST: Load one timestep of ALL variables into memory")
print("=" * 60)
print()

# --- Surface ---
print("--- Surface variables ---")
t0 = time.time()
ds_surf = xr.open_dataset(f"{DATA}/2024/2024-06-surface.nc")
t_open = time.time() - t0
print(f"  Open surface file: {t_open:.2f}s")

t0 = time.time()
surf_data = {}
for var in ds_surf.data_vars:
    v0 = time.time()
    surf_data[var] = ds_surf[var].isel(valid_time=0).values
    dt = time.time() - v0
    print(f"  Read {var}: shape={surf_data[var].shape}, "
          f"size={surf_data[var].nbytes/1e6:.1f} MB, time={dt:.2f}s")
t_surf = time.time() - t0
print(f"  Total surface read: {t_surf:.2f}s")
ds_surf.close()
print()

# --- Atmospheric ---
print("--- Atmospheric variables ---")
t0 = time.time()
ds_atm = xr.open_dataset(f"{DATA}/2024/2024-06-d01-03-atmospheric.nc")
t_open2 = time.time() - t0
print(f"  Open atmospheric file: {t_open2:.2f}s")

t0 = time.time()
atm_data = {}
for var in ds_atm.data_vars:
    v0 = time.time()
    atm_data[var] = ds_atm[var].isel(valid_time=0).values
    dt = time.time() - v0
    print(f"  Read {var}: shape={atm_data[var].shape}, "
          f"size={atm_data[var].nbytes/1e6:.1f} MB, time={dt:.2f}s")
t_atm = time.time() - t0
print(f"  Total atmospheric read: {t_atm:.2f}s")
ds_atm.close()
print()

# --- Static ---
print("--- Static variables ---")
t0 = time.time()
ds_static = xr.open_dataset(f"{DATA}/2024/static.nc")
for var in ds_static.data_vars:
    static_val = ds_static[var].values
    print(f"  Read {var}: shape={static_val.shape}, "
          f"size={static_val.nbytes/1e6:.1f} MB")
t_static = time.time() - t0
print(f"  Total static read: {t_static:.2f}s")
ds_static.close()
print()

total_mb = (sum(v.nbytes for v in surf_data.values())
            + sum(v.nbytes for v in atm_data.values()))
total_time = t_open + t_surf + t_open2 + t_atm + t_static
print("=" * 60)
print(f"TOTAL: {total_time:.2f}s for {total_mb/1e6:.0f} MB")
print(f"Effective throughput: {total_mb/1e6/total_time:.1f} MB/s")
print("=" * 60)
