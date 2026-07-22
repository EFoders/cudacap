#!/usr/bin/env python3
"""Deterministic case grid for the Module-2 layer-parameter dump
(stdlib-only): the TIMVAR -> F2VAR -> ESIND -> IONSET chain that turns the
VERSY maps into E/F1/F2/Es layer parameters for one sample area (KM=1).

Usage: gen_layer_cases.py <outdir>

layers_cases.csv: month,ssn,gmt,clat,clong,glat,gmdip,gyz
  angles in radians; clat/clong/glat/gmdip geographic+geomagnetic, gyz the
  F2 half-thickness correction.  Geometry is sampled independently of the
  real magnetic model on purpose — the layer routines are deterministic
  functions of these inputs, and independent sampling gives the branch
  coverage (F1 present/absent, twilight, F1->F2 merge) a physically
  correlated grid would miss.
"""

import pathlib
import sys

SEED = 20260724


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def row(month, ssn, gmt, clat, clong, glat, gmdip, gyz):
    return (f"{month},{f(ssn)},{f(gmt)},{f(clat)},{f(clong)},"
            f"{f(glat)},{f(gmdip)},{f(gyz)}")


def cases():
    rows = []
    # Crafted: local-noon (small zenith -> F1 present), high-lat/night
    # (F1 absent), equator, and twilight-ish geometries across seasons.
    crafted = [
        # month, ssn, gmt, clat, clong, glat, gmdip, gyz
        (6, 100, 18, 0.20, -1.5708, 0.35, 0.30, 1.36),   # equatorial noon
        (6, 100, 18, 0.70, -1.5708, 0.90, 1.00, 1.40),   # midlat noon
        (12, 15, 6, 1.30, -1.5708, 1.35, 1.45, 1.20),    # high-lat winter
        (1, 160, 0, -0.70, 3.1416, -0.90, -1.00, 1.50),  # S midlat, dateline
        (3, 100, 12, 0.00, 0.0, 0.05, 0.02, 1.30),       # equator, midnight-ish
        (9, 50, 21, 0.90, -2.30, 1.10, 1.20, 1.45),      # near twilight
        (7, 100, 15, 1.20, 2.5, 1.30, 1.40, 1.10),       # high lat, day
        (4, 30, 9, -0.30, 1.0, -0.40, -0.30, 1.25),      # S low lat
    ]
    for c in crafted:
        rows.append(row(*c))

    rnd = lcg(SEED)
    months = (1, 3, 4, 6, 7, 9, 10, 12)
    ssns = (0, 15, 50, 100, 150, 160)
    for i in range(160):
        month = months[int(rnd(0, len(months) - 1e-9))]
        ssn = ssns[int(rnd(0, len(ssns) - 1e-9))]
        rows.append(row(
            month, ssn, rnd(0.0, 23.999),
            rnd(-1.40, 1.40), rnd(-3.14159, 3.14159),
            rnd(-1.50, 1.50), rnd(-1.50, 1.50), rnd(0.80, 2.00)))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "layers_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"layers: {len(rows)} cases")


if __name__ == "__main__":
    main()
