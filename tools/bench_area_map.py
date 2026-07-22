#!/usr/bin/env python3
"""Benchmark the Stage B Area-mode ionospheric-parameter map on CPU vs CUDA.

Evaluates the full batched pipeline (MAGVAR geomagnetic field + geomagnetic
latitude + VERSY spherical-harmonic maps + the E/F1/F2/Es layer chain) over
global lat/lon grids of increasing resolution, timing each device.

Usage:  uv run python tools/bench_area_map.py
"""

import pathlib
import time

import numpy as np
import torch

from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_a.virtim import virtim
from voacap_gpu.stage_b.area import area_iono_map

COEFFS = (pathlib.Path(__file__).resolve().parents[1] / "reference"
          / "voacapl" / "voacapl" / "itshfbc" / "coeffs")
MONTH, SSN, GMT = 4, 120.0, 12.0


def _time(fn, device, reps=5):
    fn()  # warmup / JIT
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    st = redmap(COEFFS, "CCIR", MONTH, np.float32(SSN))
    ab_np = virtim(st, np.float32(GMT))
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"month={MONTH} ssn={SSN:.0f} gmt={GMT:.0f}  devices={devices}")
    print(f"{'grid':>7} {'points':>9}  " +
          "  ".join(f"{d:>9}" for d in devices) + "  speedup")
    for step in (2.0, 1.0, 0.5, 0.25):
        lats = np.arange(-89.5, 90, step, dtype=np.float32)
        lons = np.arange(-180, 180, step, dtype=np.float32)
        la, lo = np.meshgrid(lats, lons, indexing="ij")
        n = la.size
        times = {}
        for device in devices:
            ab = torch.from_numpy(ab_np).to(device)
            lat_t = torch.tensor(la.ravel(), device=device)
            lon_t = torch.tensor(lo.ravel(), device=device)
            times[device] = _time(
                lambda: area_iono_map(st, ab, MONTH, SSN, GMT, lat_t, lon_t),
                device)
        row = f"{step:>5}deg {n:>9}  " + "  ".join(
            f"{times[d]*1e3:7.1f}ms" for d in devices)
        if "cuda" in times:
            row += f"  {times['cpu']/times['cuda']:.1f}x"
        print(row)


if __name__ == "__main__":
    main()
