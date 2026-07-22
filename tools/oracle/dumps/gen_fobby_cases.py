#!/usr/bin/env python3
"""Deterministic case grid for the FOBBY (Module-4 reflectrix) dump
(stdlib-only).

FOBBY's arithmetic depends on the ionogram (FVERT/HTRUE, built from injected
FI/YI/HI + IEDP) and a single take-off angle per case (ANG(1)=ang, NANG=1 in
the driver).  Angles are sampled across the physical HF take-off range plus a
few steep ones so the COS floor is exercised without ballooning IFOB into
unphysical magnitudes.

Usage: gen_fobby_cases.py <outdir>
fobby_cases.csv:  fi1,fi2,fi3,yi1,yi2,yi3,hi1,hi2,hi3,iedp,ang
"""

import pathlib
import sys

SEED = 20260722


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


# a fixed spread of take-off angles (deg) crossing the whole ANG table range
ANGLES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 24.0, 30.0,
          38.0, 48.0, 60.0, 70.0]


def cases():
    rows = []
    # a few crafted layer geometries (F1 present/absent, both IEDP branches)
    layers = [
        (3.0, 4.5, 9.0, 20, 40, 100, 110, 190, 350, 0),
        (2.0, 0.0, 6.0, 20, 0, 90, 110, 0, 300, -1),
        (4.5, 7.0, 14.0, 21, 50, 130, 110, 220, 480, 0),
        (2.5, 0.0, 4.0, 22, 0, 120, 110, 0, 300, -1),
    ]
    for L in layers:
        for ang in ANGLES:
            rows.append(",".join(f(v) for v in (*L, ang)))

    rnd = lcg(SEED)
    for i in range(120):
        fi1 = rnd(1.0, 4.5)
        f1_present = rnd(0.0, 1.0) < 0.55
        fi3 = rnd(max(fi1 + 1.0, 4.0), 15.0)
        if f1_present:
            fi2 = rnd(fi1 + 0.3, min(fi1 + 3.5, fi3 - 0.3))
            yi2 = rnd(30.0, 52.0)
            hi2 = rnd(160.0, 225.0)
        else:
            fi2 = yi2 = hi2 = 0.0
        yi1 = rnd(17.0, 23.0)
        yi3 = rnd(55.0, 145.0)
        hi1 = rnd(105.0, 115.0)
        hi3 = rnd(max(280.0, hi2 + 60.0), 490.0)
        iedp = -1 if rnd(0.0, 1.0) < 0.5 else 0
        ang = rnd(0.0, 70.0)
        rows.append(",".join(f(v) for v in
                    (fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, iedp, ang)))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "fobby_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"fobby: {len(rows)} cases")


if __name__ == "__main__":
    main()
