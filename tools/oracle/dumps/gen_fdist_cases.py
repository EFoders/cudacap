#!/usr/bin/env python3
"""Deterministic case grid for the FDIST (modes-per-hop) dump.

Same ionosphere/geometry sweep as the FINDF dump (which builds the /REFLX/ ray
table FDIST consumes), plus GFRAC -- the hop distance as a FRACTION of this
case's maximum ray distance DMAXKM, which the driver scales into GHOP.  Sizing
the hop relative to each case's own ray table (rather than an absolute km
range) is what actually lands it inside the table, exercising the exact-hit,
bracket-interpolate and skip-past paths plus the AMIND rejection; GFRAC>1
exercises the "beyond DMAXKM -> no modes" exit.  stdlib-only.

Usage: gen_fdist_cases.py <outdir>
Columns: fi1,fi2,fi3,yi1,yi2,yi3,hi1,hi2,hi3,iedp,gcdkm,amind,freq,gfrac
"""

import pathlib
import sys

SEED = 20260723
RZ = 6370.0


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def cases():
    rows = []
    rnd = lcg(SEED)
    for _ in range(180):
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
        gcdkm = rnd(400.0, 16000.0)
        amind = 0.0 if rnd(0.0, 1.0) < 0.25 else rnd(0.1, 3.0)
        u = rnd(0.0, 1.0)
        if u < 0.45:
            freq = fi3 * rnd(0.25, 0.9)
        elif u < 0.8:
            freq = fi3 * rnd(0.9, 1.03)
        else:
            freq = fi3 * rnd(1.03, 1.3)
        freq = max(freq, 0.5)
        # hop distance as a fraction of this case's DMAXKM (the driver scales
        # it); >1 exercises the "beyond max distance -> no modes" exit
        gfrac = rnd(1.0, 1.15) if rnd(0.0, 1.0) < 0.15 else rnd(0.05, 0.99)
        rows.append(",".join(f(v) for v in
                    (fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3)) +
                    f",{int(iedp)},{f(gcdkm)},{f(amind)},{f(freq)},{f(gfrac)}")
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "fdist_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"fdist: {len(rows)} cases")


if __name__ == "__main__":
    main()
