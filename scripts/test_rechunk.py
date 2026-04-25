"""Test read speed from the rechunked file."""
import time
import xarray as xr

PATH = "/mnt/data/era5/2024/_rechunk_test.nc"


def main():
    ds = xr.open_dataset(PATH, engine="netcdf4")
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Variables:  {list(ds.data_vars)}")
    for v in ds.data_vars:
        enc = ds[v].encoding
        print(f"  {v}: shape={ds[v].shape}, chunksizes={enc.get('chunksizes')}")

    print()

    # Read single timestamp (all vars)
    for idx in [0, 100, 200, 300, 400]:
        t0 = time.time()
        sliced = ds.isel(valid_time=idx)
        for v in ds.data_vars:
            _ = sliced[v].values
        elapsed = time.time() - t0
        print(f"  Read timestamp {idx}: {elapsed:.3f}s")

    # Read a triplet (3 timestamps)
    print()
    t0 = time.time()
    for idx in [100, 106, 112]:
        sliced = ds.isel(valid_time=idx)
        for v in ds.data_vars:
            _ = sliced[v].values
    elapsed = time.time() - t0
    print(f"  Read triplet [100, 106, 112]: {elapsed:.3f}s")

    ds.close()


if __name__ == "__main__":
    main()
