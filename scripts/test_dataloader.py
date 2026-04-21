"""Test ERA5Dataset against real data on the cluster.

Usage:
    python -u scripts/test_dataloader.py --data-dir /mnt/data/era5
"""

import argparse

import torch

from src.data import ERA5Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to ERA5 data directory")
    args = parser.parse_args()

    print("Creating dataset from", args.data_dir)
    ds = ERA5Dataset(data_dirs=args.data_dir)
    print(f"Total triplets: {len(ds)}")
    print(f"First triplet: {ds.triplets[0]}")
    print(f"Last triplet:  {ds.triplets[-1]}")
    print()

    print("Loading sample 0 ...")
    inp, tgt = ds[0]
    print(f"Input surf_vars keys: {list(inp.surf_vars.keys())}")
    print(f"Input atmos_vars keys: {list(inp.atmos_vars.keys())}")
    print(f"Static vars keys: {list(inp.static_vars.keys())}")
    for k, v in inp.surf_vars.items():
        print(f"  surf[{k}] shape={v.shape} dtype={v.dtype}")
    for k, v in inp.atmos_vars.items():
        print(f"  atmos[{k}] shape={v.shape} dtype={v.dtype}")
    print(f"Input metadata time: {inp.metadata.time}")
    print(f"Input metadata atmos_levels: {inp.metadata.atmos_levels}")
    print(f"Lat shape: {inp.metadata.lat.shape}, range: [{inp.metadata.lat[-1]:.2f}, {inp.metadata.lat[0]:.2f}]")
    print(f"Lon shape: {inp.metadata.lon.shape}, range: [{inp.metadata.lon[0]:.2f}, {inp.metadata.lon[-1]:.2f}]")
    print()

    print(f"Target 2t shape: {tgt.surf_vars['2t'].shape}")
    print(f"Target metadata time: {tgt.metadata.time}")
    print()

    # NaN check
    nan_found = False
    for k, v in inp.surf_vars.items():
        if torch.isnan(v).any():
            print(f"WARNING: NaN in input surf_vars[{k}]")
            nan_found = True
    for k, v in inp.atmos_vars.items():
        if torch.isnan(v).any():
            print(f"WARNING: NaN in input atmos_vars[{k}]")
            nan_found = True
    for k, v in tgt.surf_vars.items():
        if torch.isnan(v).any():
            print(f"WARNING: NaN in target surf_vars[{k}]")
            nan_found = True
    if not nan_found:
        print("No NaNs detected in sample 0")
    print()

    # Value sanity checks
    t2m = inp.surf_vars["2t"]
    print(f"2t (temperature) range: [{t2m.min():.1f}, {t2m.max():.1f}] K")
    msl = inp.surf_vars["msl"]
    print(f"msl (pressure) range: [{msl.min():.0f}, {msl.max():.0f}] Pa")
    if "swvl1" in inp.surf_vars:
        swvl = inp.surf_vars["swvl1"]
        print(f"swvl1 (soil moisture) range: [{swvl.min():.4f}, {swvl.max():.4f}] m3/m3")
    print()

    # Load a sample from the middle
    mid = len(ds) // 2
    print(f"Loading sample {mid} (middle) ...")
    inp2, tgt2 = ds[mid]
    print(f"Sample {mid} input time: {inp2.metadata.time}, target time: {tgt2.metadata.time}")
    print(f"2t shape: {inp2.surf_vars['2t'].shape}")
    print()

    print("=== Dataloader test PASSED ===")


if __name__ == "__main__":
    main()
