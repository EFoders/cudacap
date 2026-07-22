#!/usr/bin/env python3
"""Deterministic case grid for the ANOIS1 (1 MHz atmospheric noise) dump.

Sweeps the receiver mean time GMTR across the six 4-hour block boundaries and
centres (so the KJ block pick and the JK adjacent-block choice, incl. the
cyclic wraps, all fire), the receiver latitude RLAT and longitude RLONG (both
signs, so the CEG east/west branch fires), and the month (selecting different
NOISY maps).  RLONGD is kept consistent with RLONG.  A few IZF=1 cases force
the no-data-base fallback.  stdlib-only.

Usage: gen_anois1_cases.py <outdir>
anois1_cases.csv columns:  month,ssn,gmtr,rlat_rad,rlong_rad,rlongd_deg,izf
"""

import pathlib
import sys

SEED = 20260722
R2D = 57.295779513
PI = 3.14159265358979


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def row(month, ssn, gmtr, rlat, rlong, izf):
    rlongd = rlong * R2D
    return (f"{int(month)},{f(ssn)},{f(gmtr)},{f(rlat)},{f(rlong)},"
            f"{f(rlongd)},{int(izf)}")


def cases():
    rows = []
    # GMTR across block centres (2,6,10,14,18,22) and boundaries, both hemis
    gmtrs = [0.0, 1.9, 2.0, 2.1, 4.0, 5.9, 6.0, 8.0, 10.0, 12.0, 14.0,
             16.0, 18.0, 20.0, 21.9, 22.0, 22.1, 23.9]
    for gm in gmtrs:
        rows.append(row(3, 100.0, gm, 0.7, 1.2, 0))     # east
        rows.append(row(9, 100.0, gm, -0.7, -1.2, 0))   # west
    # no-data fallback
    for m in (1, 6, 11):
        rows.append(row(m, 50.0, 10.0, 0.5, 0.5, 1))
    # random fill
    rnd = lcg(SEED)
    for _ in range(150):
        month = 1 + int(rnd(0.0, 12.0)) % 12
        ssn = rnd(8.0, 160.0)
        gmtr = rnd(0.0, 24.0)
        rlat = rnd(-1.55, 1.55)
        rlong = rnd(-PI, PI)
        izf = 1 if rnd(0.0, 1.0) < 0.04 else 0
        rows.append(row(month, ssn, gmtr, rlat, rlong, izf))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "anois1_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"anois1: {len(rows)} cases")


if __name__ == "__main__":
    main()
