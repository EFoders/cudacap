#!/usr/bin/env python3
"""Deterministic case grid for the CURMUF (Module-3 MUF/FOT) dump
(stdlib-only).

CURMUF's arithmetic depends on the layer parameters (injected controlled
FI/YI/HI, to isolate it from the VERSY floor), the path geometry (GCDKM,
AMIND), and — only through F2DIS — on the F2D table (loaded by a real
REDMAP call for the given month/ssn) plus the sample-area CLAT/CLCK.

Usage: gen_curmuf_cases.py <outdir>
curmuf_cases.csv:
  month,ssn,gcdkm,amind,clat,clck,fi1,fi2,fi3,yi1,yi2,yi3,hi1,hi2,hi3,
  fs1,fs2,fs3   (Es deciles; fs2=0 = no sporadic E; HS fixed 110 in driver)
"""

import pathlib
import sys

SEED = 20260726


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
    # month, ssn, gcdkm, amind, clat, clck, fi1,fi2,fi3, yi1,yi2,yi3,
    # hi1,hi2,hi3
    crafted = [
        # ...fi/yi/hi..., fs1, fs2, fs3
        (6, 100, 1301, 3.0, 0.69, 12.0, 3.0, 4.5, 9.0, 20, 40, 100,
         110, 190, 350, 1.5, 2.2, 3.0),                   # midlat day, short
        (1, 15, 8000, 3.0, 0.35, 2.0, 2.0, 0.0, 6.0, 20, 0, 90,
         110, 0, 300, 0.0, 0.0, 0.0),                     # long night, no Es
        (9, 160, 300, 5.0, -0.4, 15.0, 4.5, 7.0, 14.0, 21, 50, 130,
         110, 220, 480, 3.0, 4.0, 5.2),                   # short, strong Es
        (3, 50, 3000, 1.0, 1.20, 6.0, 3.5, 5.0, 8.0, 18, 45, 110,
         112, 200, 320, 2.5, 3.6, 4.0),                   # Es near E (XES>0)
        (12, 100, 12000, 3.0, -1.10, 20.0, 2.5, 0.0, 10.0, 20, 0, 120,
         110, 0, 420, 1.0, 1.5, 5.0),                     # very long, wide Es
    ]
    for c in crafted:
        rows.append(",".join(f(v) for v in c))

    rnd = lcg(SEED)
    months = (1, 3, 6, 9, 12)
    ssns = (0, 15, 50, 100, 150, 160)
    for i in range(140):
        month = months[int(rnd(0, len(months) - 1e-9))]
        ssn = ssns[int(rnd(0, len(ssns) - 1e-9))]
        gcdkm = rnd(200.0, 13000.0)
        amind = rnd(0.5, 10.0)
        clat = rnd(-1.40, 1.40)
        clck = rnd(0.0, 24.0)
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
        if rnd(0.0, 1.0) < 0.6:  # sporadic E present
            fs2 = rnd(1.0, 5.0)
            fs1 = rnd(max(0.1, fs2 - 1.2), fs2 - 0.1)
            fs3 = rnd(fs2 + 0.1, fs2 + 1.6)
        else:
            fs1 = fs2 = fs3 = 0.0
        rows.append(",".join(f(v) for v in
                    (month, ssn, gcdkm, amind, clat, clck,
                     fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3,
                     fs1, fs2, fs3)))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "curmuf_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"curmuf: {len(rows)} cases")


if __name__ == "__main__":
    main()
