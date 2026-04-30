#!/usr/bin/env python3
"""
Verify rechunked ERA5 NetCDF files against the original source tree.

This is the strongest practical check short of a full byte-level diff:
  - Same variables, dimensions, coordinate lengths
  - Optional: on-disk chunk layout on the rechunked file (per-timestep chunks)
  - Data arrays match source exactly after decode (lossless zlib → bitwise match)

Usage (on a pod with both PVCs mounted):
  python3 verify_rechunked_era5.py \\
    --src-root /mnt/src/era5 \\
    --dst-root /mnt/dst/era5 \\
    --check-chunks

  Quick spot-check (first + last timestep only, much faster / less RAM):
  python3 verify_rechunked_era5.py --src-root ... --dst-root ... --quick

  Limit to one relative path:
  python3 verify_rechunked_era5.py ... --only 2025/2025-07-d31-31-atmospheric.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def expected_chunks_for_var(ds: xr.Dataset, var: str) -> tuple[int, ...]:
    """Chunksizes in dimension order for our rechunk pipeline (valid_time → 1)."""
    dims = ds[var].dims
    return tuple(1 if d == "valid_time" else int(ds.sizes[d]) for d in dims)


def encoding_chunks(ds: xr.Dataset, var: str) -> tuple[int, ...] | None:
    enc = ds[var].encoding
    ch = enc.get("chunksizes") or enc.get("chunks")
    if ch is None:
        return None
    return tuple(int(x) for x in ch)


def expected_chunks_for_ds(ds: xr.Dataset) -> dict[str, tuple[int, ...]]:
    return {name: expected_chunks_for_var(ds, name) for name in ds.data_vars}


def compare_arrays(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if np.issubdtype(a.dtype, np.floating):
        if not np.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True):
            d = np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))
            raise ValueError(f"float data differ, max abs diff={d}")
    else:
        if not np.array_equal(a, b):
            raise ValueError("non-float data differ")


def verify_pair(
    src_path: Path,
    dst_path: Path,
    *,
    check_chunks: bool,
    quick: bool,
) -> None:
    with xr.open_dataset(src_path) as src_raw, xr.open_dataset(dst_path) as dst_raw:
        if set(src_raw.data_vars) != set(dst_raw.data_vars):
            raise ValueError(
                f"data_vars differ: {set(src_raw.data_vars) ^ set(dst_raw.data_vars)}",
            )
        if src_raw.sizes != dst_raw.sizes:
            raise ValueError(
                f"dimension sizes differ:\n  src={dict(src_raw.sizes)}\n  dst={dict(dst_raw.sizes)}",
            )

        for cname in src_raw.coords:
            if cname not in dst_raw.coords:
                raise ValueError(f"coord {cname} missing in dst")
            compare_arrays(
                np.asarray(src_raw[cname].values),
                np.asarray(dst_raw[cname].values),
            )

        exp = expected_chunks_for_ds(dst_raw)
        for name in sorted(src_raw.data_vars):
            if check_chunks:
                got = encoding_chunks(dst_raw, name)
                want = exp[name]
                if got != want:
                    raise ValueError(
                        f"{name}: chunksizes got {got}, want {want} (rechunk layout)",
                    )

            if quick and "valid_time" in src_raw[name].dims:
                nt = int(src_raw.sizes["valid_time"])
                for i in (0, nt - 1):
                    sv = np.asarray(src_raw[name].isel(valid_time=i).values)
                    dv = np.asarray(dst_raw[name].isel(valid_time=i).values)
                    compare_arrays(sv, dv)
            else:
                sv = np.asarray(src_raw[name].values)
                dv = np.asarray(dst_raw[name].values)
                compare_arrays(sv, dv)


def iter_dst_files(dst_root: Path) -> list[Path]:
    return sorted(dst_root.rglob("*.nc"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--dst-root", type=Path, required=True)
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Relative path under roots (e.g. 2025/2025-07-d01-03-atmospheric.nc)",
    )
    p.add_argument(
        "--check-chunks",
        action="store_true",
        help="Assert on-disk chunk sizes match per-timestep layout",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Only compare first and last valid_time slice per variable",
    )
    args = p.parse_args()

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()

    if args.only:
        rel = args.only.lstrip("/")
        verify_pair(
            src_root / rel,
            dst_root / rel,
            check_chunks=args.check_chunks,
            quick=args.quick,
        )
        print(f"OK: {rel}")
        return

    failures: list[tuple[str, str]] = []
    ok = 0
    for dst_path in iter_dst_files(dst_root):
        rel = dst_path.relative_to(dst_root).as_posix()
        src_path = src_root / rel
        if not src_path.is_file():
            failures.append((rel, f"no source file at {src_path}"))
            continue
        try:
            verify_pair(
                src_path,
                dst_path,
                check_chunks=args.check_chunks,
                quick=args.quick,
            )
            ok += 1
            print(f"OK: {rel}")
        except Exception as e:
            failures.append((rel, str(e)))
            print(f"FAIL: {rel}: {e}", file=sys.stderr)

    print(f"\nSummary: {ok} ok, {len(failures)} failed")
    if failures:
        for rel, err in failures:
            print(f"  {rel}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
