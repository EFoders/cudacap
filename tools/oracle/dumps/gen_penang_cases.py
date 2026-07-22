#!/usr/bin/env python3
"""Deterministic case grid for the PENANG (penetration angle) dump.

Controlled FI/YI/HI + IEDP build the ionogram (as for the ALOSFV dump); the
operating FREQ is sampled relative to the F2 critical FI(3) so every branch
fires: FREQ well below foF2 (E/F1/F2 all reflect), near foF2 (the F2-nose probe
of indices 28/29/30), and above foF2 (full F2 penetration -> DELPEN=0 and the
DSKPKM/DMAXKM=-1 flags).  Both F1-present and F1-absent ionospheres, both IEDP
models.  stdlib-only.

Usage: gen_penang_cases.py <outdir>
penang_cases.csv columns:  fi1,fi2,fi3,yi1,yi2,yi3,hi1,hi2,hi3,iedp,freq
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
    rnd = lcg(SEED)
    for _ in range(200):
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
        # FREQ relative to foF2: 40% low (reflect), 40% near foF2 (nose probe),
        # 20% above foF2 (full penetration)
        u = rnd(0.0, 1.0)
        if u < 0.4:
            freq = fi3 * rnd(0.25, 0.9)
        elif u < 0.8:
            freq = fi3 * rnd(0.92, 1.03)
        else:
            freq = fi3 * rnd(1.03, 1.35)
        freq = max(freq, 0.5)
        rows.append(",".join(f(v) for v in
                    (fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3)) +
                    f",{int(iedp)},{f(freq)}")
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "penang_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"penang: {len(rows)} cases")


if __name__ == "__main__":
    main()
