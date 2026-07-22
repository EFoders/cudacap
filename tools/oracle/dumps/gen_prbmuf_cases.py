#!/usr/bin/env python3
"""Deterministic case grid for the PRBMUF probability dump (stdlib-only).

Sweeps the operating frequency FMHZ around the layer MUF FGO (so Z = FMHZ-FGO
takes both signs and large magnitudes to exercise the |Z|<=5 clamp), the median
placement FSET (negative/zero exercise the step branch; positive the normal
approximation), the layer IL 1..4, and the decile spreads SIGL/SIGU (small
values drive SIG to its 0.001 floor).  A few hand-picked edge cases pin the
branch boundaries.

Usage: gen_prbmuf_cases.py <outdir>
prbmuf_cases.csv columns:  fmhz,fgo,fset,il,siglil,siguil
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


def cases():
    rows = []
    # hand-picked edges: step branch (fset<=0) both sides; z==0; tiny sig
    edges = [
        (5.0, 10.0, 0.0, 3, 0.1, 0.1),    # fset==0, z<0 -> qx=1
        (15.0, 10.0, 0.0, 3, 0.1, 0.1),   # fset==0, z>0 -> qx=0
        (10.0, 10.0, 0.0, 3, 0.1, 0.1),   # fset==0, z==0 -> qx=1
        (10.0, 10.0, 12.0, 3, 0.1, 0.1),  # z==0, fset>0
        (40.0, 8.0, 12.0, 3, 0.05, 0.05),  # huge z/sig -> clamp
        (1.0, 40.0, 12.0, 2, 0.05, 0.05),  # huge -z/sig -> clamp
        (12.0, 10.0, 30.0, 4, 0.02, 0.02),  # small sig -> 0.001 floor
    ]
    rows.extend(",".join(f(v) if i != 3 else str(int(v))
                for i, v in enumerate(c)) for c in edges)
    rnd = lcg(SEED)
    for i in range(240):
        fgo = rnd(2.0, 40.0)
        fmhz = rnd(1.0, 45.0)
        # ~1 in 6 cases takes the step branch (fset <= 0)
        fset = rnd(-6.0, 0.0) if rnd(0.0, 1.0) < 0.17 else rnd(0.5, 40.0)
        il = 1 + int(rnd(0.0, 4.0)) % 4
        siglil = rnd(0.02, 0.30)
        siguil = rnd(0.02, 0.30)
        rows.append(",".join((f(fmhz), f(fgo), f(fset), str(il),
                              f(siglil), f(siguil))))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "prbmuf_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"prbmuf: {len(rows)} cases")


if __name__ == "__main__":
    main()
