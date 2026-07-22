#!/usr/bin/env python3
"""Deterministic case grid for the GAIN ground-reflection (ITR<0) dump.

Sweeps realistic ground constants -- sea water (sigma ~5 mhos/m, eps ~80),
wet/average/dry land (sigma ~1e-3..3e-2, eps ~4..30), fresh water and ice --
across the HF band and all elevation angles.  Edge cases pin the branches:
SIGMA<=0 (no ground data -> 0), DELTA at/below the 1e-8 grazing floor (-> 6 dB),
and near-90-degree incidence.  stdlib-only.

Usage: gen_gain_cases.py <outdir>
gain_cases.csv columns:  ig,sigma,er,delta_rad,fmc
"""

import pathlib
import sys

SEED = 20260723
D2R = 0.0174532925199433

# (sigma mhos/m, relative permittivity) for typical ground types
GROUNDS = [
    (5.0, 80.0),       # sea water
    (0.01, 30.0),      # fresh water / marsh
    (0.03, 30.0),      # wet ground
    (0.005, 15.0),     # average ground
    (0.002, 10.0),     # poor / dry ground
    (0.001, 4.0),      # very dry / rocky
    (0.0001, 3.0),     # ice
]


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def row(ig, sigma, er, delta, fmc):
    return f"{int(ig)},{f(sigma)},{f(er)},{f(delta)},{f(fmc)}"


def cases():
    rows = []
    # every ground type across the band at a spread of elevations
    for ig, (sigma, er) in enumerate(GROUNDS, start=1):
        for fmc in (2.0, 7.0, 14.0, 30.0):
            for deg in (0.5, 3.0, 10.0, 30.0, 60.0, 89.0):
                rows.append(row(min(ig, 5), sigma, er, deg * D2R, fmc))
    # edge cases: no ground data, grazing floor, near-vertical
    rows.append(row(1, 0.0, 15.0, 10.0 * D2R, 10.0))       # sigma == 0
    rows.append(row(2, -1.0, 15.0, 10.0 * D2R, 10.0))      # sigma < 0
    rows.append(row(3, 0.005, 15.0, 0.0, 10.0))            # delta == 0 -> 6 dB
    rows.append(row(4, 0.005, 15.0, 1.0e-9, 10.0))         # below the floor
    rows.append(row(5, 0.005, 15.0, 1.0e-8, 10.0))         # at the floor
    rows.append(row(1, 5.0, 80.0, 89.999 * D2R, 25.0))     # near vertical
    # random fill
    rnd = lcg(SEED)
    for _ in range(200):
        ig = 1 + int(rnd(0.0, 5.0)) % 5
        sigma, er = GROUNDS[int(rnd(0.0, float(len(GROUNDS)))) % len(GROUNDS)]
        sigma = sigma * rnd(0.5, 2.0)
        er = er * rnd(0.8, 1.2)
        delta = rnd(0.0005, 1.5)
        fmc = rnd(2.0, 30.0)
        rows.append(row(ig, sigma, er, delta, fmc))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "gain_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"gain: {len(rows)} cases")


if __name__ == "__main__":
    main()
