#!/usr/bin/env python3
"""Deterministic case grid for the SYSSY (system-loss table) dump (stdlib-only).

Exercises SYSSY's integer index arithmetic against the real SYS/PERR tables:
  * geomagnetic latitude G (radians) swept across the KJ truncation boundaries
    (|G| deg near 32.5/37.5/.../77.5) in BOTH hemispheres (G<0 shifts the
    local-time block by KK=8 and the prediction block by ND=3);
  * reflection-area local time PT across the J / LJ / LD block boundaries,
    including the J==0 midnight wrap (LJ->8);
  * distance index NN in {2, 5};
  * a handful of IZF=1 cases forcing the no-data fallback.
Then a random fill for breadth.

Usage: gen_syssy_cases.py <outdir>
syssy_cases.csv columns:  month,g_rad,pt,nn,izf
"""

import pathlib
import sys

SEED = 20260722
D2R = 0.0174532925199433


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def row(month, g_rad, pt, nn, izf):
    return f"{int(month)},{f(g_rad)},{f(pt)},{int(nn)},{int(izf)}"


def cases():
    rows = []
    # G swept across KJ boundaries, both hemispheres, both NN
    g_deg = [5, 20, 32, 33, 37, 38, 42, 43, 47, 48, 52, 57, 62, 63,
             72, 77, 78, 85, 90]
    for gd in g_deg:
        for sign in (1.0, -1.0):
            for nn in (2, 5):
                rows.append(row(6, sign * gd * D2R, 12.0, nn, 0))
    # PT across J / LJ / LD block boundaries (incl. midnight wrap)
    pts = [0.0, 0.4, 0.6, 1.4, 2.4, 2.6, 3.0, 5.4, 5.6, 6.0, 8.9, 9.0,
           11.4, 12.0, 14.9, 15.0, 17.6, 18.0, 20.9, 21.0, 23.4, 23.9]
    for pt in pts:
        for nn in (2, 5):
            rows.append(row(3, 45.0 * D2R, pt, nn, 0))
            rows.append(row(9, -55.0 * D2R, pt, nn, 0))
    # no-data fallback
    for m in (1, 5, 11):
        rows.append(row(m, 40.0 * D2R, 10.0, 2, 1))
        rows.append(row(m, -20.0 * D2R, 3.0, 5, 1))
    # random fill for breadth
    rnd = lcg(SEED)
    for _ in range(300):
        month = 1 + int(rnd(0.0, 12.0)) % 12
        g_rad = rnd(-1.55, 1.55)
        pt = rnd(0.0, 24.0)
        nn = 2 if rnd(0.0, 1.0) < 0.5 else 5
        izf = 1 if rnd(0.0, 1.0) < 0.05 else 0
        rows.append(row(month, g_rad, pt, nn, izf))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "syssy_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"syssy: {len(rows)} cases")


if __name__ == "__main__":
    main()
