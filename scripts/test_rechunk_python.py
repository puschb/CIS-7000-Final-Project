import time
import os
import xarray as xr

SRC = "/scratch/input.nc"
DST = "/scratch/output.nc"

print("=" * 60)
print("TEST: Python xarray rechunk (atmospheric file)")
print("=" * 60)

print(f"\n--- Step 1: Open source file ---")
t0 = time.time()
ds = xr.open_dataset(SRC)
print(f"  Opened in {time.time() - t0:.2f}s")
print(f"  Variables: {list(ds.data_vars)}")
for var in ds.data_vars:
    print(f"    {var}: shape={ds[var].shape}, "
          f"chunks={ds[var].encoding.get('chunksizes', 'none')}, "
          f"compression={ds[var].encoding.get('zlib', False)}")

is_atmospheric = "pressure_level" in ds.dims
if is_atmospheric:
    target_chunks = {"valid_time": 1, "pressure_level": 13, "latitude": 721, "longitude": 1440}
else:
    target_chunks = {"valid_time": 1, "latitude": 721, "longitude": 1440}

print(f"\n  Target chunks: {target_chunks}")

print(f"\n--- Step 2: Load entire dataset into memory ---")
t0 = time.time()
ds_loaded = ds.load()
t_load = time.time() - t0
mem_bytes = sum(ds_loaded[v].nbytes for v in ds_loaded.data_vars)
print(f"  Loaded in {t_load:.2f}s ({mem_bytes / 1e9:.1f} GB)")

print(f"\n--- Step 3: Write with new chunking (compressed) ---")
encoding = {}
for var in ds_loaded.data_vars:
    dims = ds_loaded[var].dims
    chunks = tuple(target_chunks.get(d, ds_loaded.sizes[d]) for d in dims)
    encoding[var] = {
        "chunksizes": chunks,
        "zlib": True,
        "complevel": 1,
    }
    print(f"  {var}: encoding chunks={chunks}")

t0 = time.time()
ds_loaded.to_netcdf(DST, encoding=encoding)
t_write = time.time() - t0

dst_size = os.path.getsize(DST)
print(f"  Written in {t_write:.2f}s")
print(f"  Output size: {dst_size / 1e6:.0f} MB")

ds.close()
ds_loaded.close()

print(f"\n{'=' * 60}")
print(f"Load: {t_load:.1f}s | Write: {t_write:.1f}s | Total: {t_load + t_write:.1f}s")
print(f"{'=' * 60}")
