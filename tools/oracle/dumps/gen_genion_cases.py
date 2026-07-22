#!/usr/bin/env python3
"""Deterministic case grid for the GENION (Module-4 ionogram) dump
(stdlib-only).

GENION's arithmetic depends only on the E/F1/F2 layer parameters (injected
controlled FI/YI/HI, isolating it from the VERSY floor) and the IEDP flag
selecting the height model.  Each case carries an explicit IEDP so both the
full-Gaussian (IEDP<0) and fast (IEDP>=0) branches are covered, crossed with
F1-present and F1-absent geometries.

Usage: gen_genion_cases.py <outdir>
genion_cases.csv:  fi1,fi2,fi3,yi1,yi2,yi3,hi1,hi2,hi3,iedp
  (fi2=0 -> no F1 layer; iedp<0 -> full Gaussian, iedp>=0 -> fast model)
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
    # fi1,fi2,fi3, yi1,yi2,yi3, hi1,hi2,hi3, iedp -- crafted, each branch
    crafted = [
        # midlat day, F1 present -- fast then full
        (3.0, 4.5, 9.0, 20, 40, 100, 110, 190, 350, 0),
        (3.0, 4.5, 9.0, 20, 40, 100, 110, 190, 350, -1),
        # night, no F1 -- fast then full
        (2.0, 0.0, 6.0, 20, 0, 90, 110, 0, 300, 0),
        (2.0, 0.0, 6.0, 20, 0, 90, 110, 0, 300, -1),
        # strong high foF2 with F1
        (4.5, 7.0, 14.0, 21, 50, 130, 110, 220, 480, 0),
        (4.5, 7.0, 14.0, 21, 50, 130, 110, 220, 480, -1),
        # thin layers, F1 just above E (near-merge)
        (3.5, 5.0, 8.0, 18, 45, 110, 112, 200, 320, 0),
        (3.5, 5.0, 8.0, 18, 45, 110, 112, 200, 320, -1),
        # low foF2, no F1
        (2.5, 0.0, 4.0, 22, 0, 120, 110, 0, 300, 0),
        (2.5, 0.0, 4.0, 22, 0, 120, 110, 0, 300, -1),
    ]
    for c in crafted:
        rows.append(",".join(f(v) for v in c))

    rnd = lcg(SEED)
    for i in range(130):
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
        rows.append(",".join(f(v) for v in
                    (fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, iedp)))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "genion_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"genion: {len(rows)} cases")


if __name__ == "__main__":
    main()
