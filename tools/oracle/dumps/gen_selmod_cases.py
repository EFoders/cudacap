#!/usr/bin/env python3
"""Deterministic case grid for the SELMOD (sample-area selection) dump.

Covers every branch: KFX==1 (JMODE=1 regardless of FI); KFX==2 (E-critical
tiebreak FI(1,1) vs FI(1,2), including equal); KFX>=3 (F2-critical DELFI =
FI(3,1)-FI(3,3) with |DELFI| straddling the 0.01 MHz threshold, and the E
tiebreak FI(1,1) vs FI(1,3) when the F2s match).  stdlib-only.

Usage: gen_selmod_cases.py <outdir>
selmod_cases.csv columns:  kfx,fi11,fi12,fi13,fi31,fi33
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


def row(kfx, fi11, fi12, fi13, fi31, fi33):
    return f"{int(kfx)},{f(fi11)},{f(fi12)},{f(fi13)},{f(fi31)},{f(fi33)}"


def cases():
    rows = []
    # KFX==1 : JMODE always 1
    rows.append(row(1, 3.0, 5.0, 2.0, 8.0, 9.0))
    rows.append(row(1, 5.0, 1.0, 9.0, 4.0, 4.0))
    # KFX==2 : FI(1,1) vs FI(1,2)
    rows.append(row(2, 3.0, 2.0, 0.0, 0.0, 0.0))   # >  -> 2
    rows.append(row(2, 2.0, 3.0, 0.0, 0.0, 0.0))   # <  -> 1
    rows.append(row(2, 3.0, 3.0, 0.0, 0.0, 0.0))   # == -> 1
    # KFX>=3 : DELFI near the 0.01 threshold + E tiebreak
    for kfx in (3, 4, 5):
        rows.append(row(kfx, 3.0, 0.0, 2.0, 9.0, 9.005))   # |d|<.01, fi11>fi13 ->3
        rows.append(row(kfx, 2.0, 0.0, 3.0, 9.0, 9.005))   # |d|<.01, fi11<fi13 ->1
        rows.append(row(kfx, 2.0, 0.0, 3.0, 9.0, 9.02))    # d<0 (-.02) ->1
        rows.append(row(kfx, 2.0, 0.0, 3.0, 9.03, 9.0))    # d>0 (+.03) ->3
        rows.append(row(kfx, 3.0, 0.0, 2.0, 9.011, 9.0))   # d just >.01 ->3
        rows.append(row(kfx, 3.0, 0.0, 2.0, 9.0, 9.009))   # |d|<.01 ->E tiebreak
    # random fill
    rnd = lcg(SEED)
    for _ in range(120):
        kfx = 1 + int(rnd(0.0, 5.0)) % 5
        fi11 = rnd(1.0, 6.0)
        fi12 = rnd(1.0, 6.0)
        fi13 = rnd(1.0, 6.0)
        fi31 = rnd(4.0, 15.0)
        # ~1/4 of the time make FI(3,3) within ~0.01 of FI(3,1)
        if rnd(0.0, 1.0) < 0.25:
            fi33 = fi31 + rnd(-0.015, 0.015)
        else:
            fi33 = rnd(4.0, 15.0)
        rows.append(row(kfx, fi11, fi12, fi13, fi31, fi33))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "selmod_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"selmod: {len(rows)} cases")


if __name__ == "__main__":
    main()
