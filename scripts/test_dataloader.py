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
    inp, targets = ds[0]
    tgt = targets[0]
    print(f"Rollout steps: {len(targets)} target(s)")
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

    for ti, tgt in enumerate(targets):
        print(f"Target {ti} 2t shape: {tgt.surf_vars['2t'].shape}")
        print(f"Target {ti} metadata time: {tgt.metadata.time}")
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
    for ti, tgt in enumerate(targets):
        for k, v in tgt.surf_vars.items():
            if torch.isnan(v).any():
                print(f"WARNING: NaN in target[{ti}] surf_vars[{k}]")
                nan_found = True
    if not nan_found:
        print("No NaNs detected in sample 0")
    print()

    # Density channel checks
    if "swvl1_density" in inp.surf_vars:
        dens = inp.surf_vars["swvl1_density"]
        print(f"swvl1_density unique values: {torch.unique(dens).tolist()}")
        land_frac = dens.mean().item()
        print(f"swvl1_density land fraction: {land_frac:.3f}")
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
    inp2, targets2 = ds[mid]
    tgt2 = targets2[0]
    print(f"Sample {mid} input time: {inp2.metadata.time}, target time: {tgt2.metadata.time}")
    print(f"2t shape: {inp2.surf_vars['2t'].shape}")
    print()

    print("=== Dataloader test PASSED ===")


if __name__ == "__main__":
    main()
