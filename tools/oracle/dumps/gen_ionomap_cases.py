#!/usr/bin/env python3
"""Deterministic case grids for the Module-2 ionospheric-map dumps
(stdlib-only): VIRTIM (diurnal Fourier -> AB) and VERSY (geographic
series -> GAMMA), both downstream of REDMAP's load + SSN blend.

Usage: gen_ionomap_cases.py <outdir>

virtim_cases.csv: icoeff,month,ssn,gmt          (icoeff 0=CCIR 1=URSI)
versy_cases.csv:  icoeff,month,ssn,gmt,clat,clong,gmdip   (radians)
"""

import pathlib
import sys

SEED = 20260723


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def virtim_cases():
    rows = []
    for icoeff in (0, 1):
        for month in (1, 7):
            for ssn in (15, 100, 160):
                for gmt in (1, 4, 7, 10, 13, 16, 19, 22):
                    rows.append(f"{icoeff},{month},{f(ssn)},{f(gmt)}")
    return rows


def versy_cases():
    rows = []
    groups = [(0, 7, 100, 16), (1, 1, 15, 4)]
    crafted = [
        # (clat, clong, gmdip) radians: equator, mid, high lat, poles,
        # east-longitude conversion branch, dip extremes/zero
        (0.0, 0.0, 0.0),
        (0.0, -0.001, 0.3),
        (0.6981317, 2.0943951, 1.1),      # 40N, 120E
        (-0.6981317, -2.0943951, -1.1),
        (1.2217305, 3.1415926, 1.4),      # 70N, ~180
        (-1.2217305, -3.1415926, -1.4),
        (1.5533430, -0.5, 1.45),          # 89N
        (-1.5533430, 0.5, -1.45),
        (1.5707964, 1.0, 1.5),            # f32 pi/2 — GOB underflow region
        (-1.5707964, -1.0, -1.5),
        (0.2, -3.0, 0.05),
        (0.9, 6.2, 0.9),                  # clong just under 2pi (unusual +)
    ]
    for grp in groups:
        icoeff, month, ssn, gmt = grp
        for (cl, cg, dip) in crafted:
            rows.append(f"{icoeff},{month},{f(ssn)},{f(gmt)},"
                        f"{f(cl)},{f(cg)},{f(dip)}")
        rnd = lcg(SEED + icoeff)
        for _ in range(100):
            rows.append(
                f"{icoeff},{month},{f(ssn)},{f(gmt)},"
                f"{f(rnd(-1.55, 1.55))},{f(rnd(-3.14159, 3.14159))},"
                f"{f(rnd(-1.5, 1.5))}")
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    for name, gen in [("virtim", virtim_cases), ("versy", versy_cases)]:
        rows = gen()
        with open(outdir / f"{name}_cases.csv", "w", newline="\n") as fh:
            fh.write("\n".join(rows) + "\n")
        print(f"{name}: {len(rows)} cases")


if __name__ == "__main__":
    main()
