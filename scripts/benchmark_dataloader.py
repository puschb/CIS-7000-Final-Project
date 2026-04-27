"""Benchmark ERA5Dataset loading speed and memory for different batch sizes.

Tests:
  1. Single-sample sequential load: timing and memory breakdown
  2. DataLoader throughput at batch_size = 1, 2, 4, 8 (8 workers each)

Usage:
    python -u scripts/benchmark_dataloader.py --data-dir /mnt/data/era5/2024 /mnt/data/era5/2025
    python -u scripts/benchmark_dataloader.py --data-dir /mnt/data/era5/2024 --rollout-steps 2
    python -u scripts/benchmark_dataloader.py --data-dir /mnt/data/era5/per-step-bench \\
        --file-layout per_timestep
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import psutil
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import ERA5Dataset, collate_era5_batch, era5_worker_init_fn


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def batch_bytes(batch) -> int:
    total = 0
    for v in batch.surf_vars.values():
        total += v.nelement() * v.element_size()
    for v in batch.atmos_vars.values():
        total += v.nelement() * v.element_size()
    for v in batch.static_vars.values():
        total += v.nelement() * v.element_size()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", nargs="+", required=True)
    parser.add_argument("--rollout-steps", type=int, default=1, choices=[1, 2])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--n-batches", type=int, default=6,
                        help="Batches to pull per batch_size test")
    parser.add_argument(
        "--file-layout",
        choices=("chunked", "per_timestep"),
        default="chunked",
        help="chunked: monthly surface + dXX-YY-atmospheric; "
        "per_timestep: YYYY-MM-DDTHH-*.nc (see split_chunk_to_per_timestep.py)",
    )
    args = parser.parse_args()

    print(f"Data dirs:     {args.data_dir}")
    print(f"File layout:   {args.file_layout}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"Workers:       {args.workers}")
    print(f"Batch sizes:   {args.batch_sizes}")
    print(f"CPU count:     {os.cpu_count()}")
    print(f"RAM:           {bytes_to_mb(psutil.virtual_memory().total):.0f} MB")
    if os.path.exists("/dev/shm"):
        st = os.statvfs("/dev/shm")
        print(f"Shared memory: {bytes_to_mb(st.f_bavail * st.f_frsize):.0f} / "
              f"{bytes_to_mb(st.f_blocks * st.f_frsize):.0f} MB free")
    print()

    ds = ERA5Dataset(
        data_dirs=args.data_dir,
        rollout_steps=args.rollout_steps,
        file_layout=args.file_layout,
    )
    n_samples = len(ds)
    if n_samples == 0:
        raise SystemExit("Dataset has zero samples (check date range and files).")

    batch_sizes = [b for b in args.batch_sizes if b <= n_samples]
    if not batch_sizes:
        batch_sizes = [1]
    if batch_sizes != args.batch_sizes:
        print(
            f"Note: batch_sizes filtered to <= dataset len ({n_samples}): {batch_sizes}"
        )
        print()

    print(f"Dataset: {n_samples} samples")
    print(f"First sequence: {ds.sequences[0]}")
    print()

    # -------------------------------------------------------------------------
    # 1. Single sample: timing and memory breakdown
    # -------------------------------------------------------------------------
    print("=== Single sample load (no workers) ===")
    process = psutil.Process()
    gc.collect()
    rss_before = process.memory_info().rss

    t0 = time.perf_counter()
    input_batch, targets = ds[0]
    elapsed = time.perf_counter() - t0

    gc.collect()
    rss_after = process.memory_info().rss

    input_mb = bytes_to_mb(batch_bytes(input_batch))
    target_mb = sum(bytes_to_mb(batch_bytes(t)) for t in targets)
    total_mb = input_mb + target_mb

    print(f"  Load time:        {elapsed:.1f}s")
    print(f"  Input batch:      {input_mb:.0f} MB")
    print(f"  Targets ({len(targets)}×):      {target_mb:.0f} MB")
    print(f"  Total tensors:    {total_mb:.0f} MB")
    print(f"  RSS delta:        {bytes_to_mb(rss_after - rss_before):.0f} MB "
          f"(includes xarray/Python overhead)")
    print()
    print("  Surf vars (input):")
    for k, v in input_batch.surf_vars.items():
        print(f"    {k:22s} {str(list(v.shape)):28s} {bytes_to_mb(v.nelement()*v.element_size()):5.1f} MB")
    print("  Atmos vars (input):")
    for k, v in input_batch.atmos_vars.items():
        print(f"    {k:22s} {str(list(v.shape)):28s} {bytes_to_mb(v.nelement()*v.element_size()):5.1f} MB")
    print()

    # NaN check
    nan_vars = [k for k, v in input_batch.surf_vars.items() if torch.isnan(v).any()]
    if nan_vars:
        print(f"  WARNING: NaN in input surf_vars: {nan_vars}")
    else:
        print("  No NaNs in input (density channels applied correctly)")
    if "swvl1_density" in input_batch.surf_vars:
        d = input_batch.surf_vars["swvl1_density"]
        print(f"  swvl1_density unique values: {torch.unique(d).tolist()}  "
              f"land fraction: {d.mean().item():.3f}")
    print()

    # -------------------------------------------------------------------------
    # 2. DataLoader throughput at different batch sizes
    # -------------------------------------------------------------------------
    print(f"=== DataLoader throughput ({args.workers} workers) ===")
    print(f"{'batch_size':>12}  {'warmup':>8}  {'avg/batch':>10}  "
          f"{'samples/s':>10}  {'batch_MB':>10}  {'shm_used':>10}")
    print("-" * 68)

    for bs in batch_sizes:
        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=True,
            num_workers=args.workers,
            prefetch_factor=1,
            persistent_workers=True,
            worker_init_fn=era5_worker_init_fn,
            collate_fn=collate_era5_batch,
            pin_memory=False,
        )

        loader_iter = iter(loader)

        # Warmup: workers fork + first reads from cold cache
        t0 = time.perf_counter()
        inp, tgts = next(loader_iter)
        warmup = time.perf_counter() - t0

        # Check batch shape is correct
        actual_bs = next(iter(inp.surf_vars.values())).shape[0]
        assert actual_bs == bs, f"Expected batch_size={bs}, got {actual_bs}"
        batch_mb = bytes_to_mb(batch_bytes(inp) + sum(batch_bytes(t) for t in tgts))

        # Steady-state timing
        times = []
        for _ in range(args.n_batches):
            t0 = time.perf_counter()
            try:
                next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                next(loader_iter)
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        samples_per_s = bs / avg

        shm_str = "n/a"
        if os.path.exists("/dev/shm"):
            st = os.statvfs("/dev/shm")
            used = (st.f_blocks - st.f_bavail) * st.f_frsize
            shm_str = f"{bytes_to_mb(used):.0f} MB"

        print(f"{bs:>12}  {warmup:>7.1f}s  {avg:>9.1f}s  "
              f"{samples_per_s:>9.2f}/s  {batch_mb:>8.0f} MB  {shm_str:>10}")

        del loader, loader_iter
        gc.collect()

    print()
    print("=== Done ===")


if __name__ == "__main__":
    main()
