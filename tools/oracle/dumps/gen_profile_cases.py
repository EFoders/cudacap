#!/usr/bin/env python3
"""Deterministic case grid for the profile/bending primitives dump
(stdlib-only): LECDEN (electron-density profile) + GETHP (virtual heights)
+ BENDY + PEN.

These routines depend only on the E/F1/F2 layer parameters, so the cases
feed controlled FI/YI/HI directly (bypassing the coefficient chain) to
isolate the primitives from the upstream VERSY libm floor.  Constraints
kept physical so the layer geometry is well-posed: FI1<FI3, FI2<FI3 (or
F1 absent = FI2/YI2/HI2 = 0), HI1<HI2<HI3, all Y>0.

Usage: gen_profile_cases.py <outdir>
profile_cases.csv: fi1,fi2,fi3,yi1,yi2,yi3,hi1,hi2,hi3
"""

import pathlib
import sys

SEED = 20260725


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
    # Crafted: F1 present (day), F1 absent (night), thin/thick layers,
    # F1 close to F2 (linear-layer branch), E close to F1.
    crafted = [
        # fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3
        (3.0, 4.5, 9.0, 20, 40, 100, 110, 190, 350),   # typical day
        (2.0, 0.0, 6.0, 20, 0, 90, 110, 0, 300),       # night, no F1
        (4.0, 6.8, 7.2, 20, 45, 110, 110, 200, 320),   # F1 near F2
        (1.5, 2.0, 12.0, 22, 35, 140, 108, 165, 460),  # E near F1, high F2
        (3.5, 5.0, 5.5, 18, 48, 70, 112, 210, 280),    # thin F2
        (2.5, 0.0, 10.0, 20, 0, 120, 110, 0, 420),     # no F1, thick
        (4.5, 7.0, 14.0, 21, 50, 130, 110, 220, 480),  # strong day, high ssn
        (1.2, 0.0, 4.2, 20, 0, 80, 110, 0, 260),       # weak night
    ]
    for c in crafted:
        rows.append(",".join(f(v) for v in c))

    rnd = lcg(SEED)
    for i in range(140):
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
        rows.append(",".join(f(v) for v in
                    (fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3)))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "profile_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"profile: {len(rows)} cases")


if __name__ == "__main__":
    main()
