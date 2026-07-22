#!/usr/bin/env python3
"""Deterministic case grid for the GAIN antenna-branch (ITR>=0) dump.

Sweeps the bilinear lookup over both table axes and pins its edges:

* frequency exactly on a cell boundary (XFMC == 0) and just either side of it,
  where INT(FMC) steps;
* the top frequency cell, where MIN0(30,I+1) collapses IP1 onto I and the
  interpolation degenerates to a flat hold;
* elevation on and between whole degrees, and at/past 90 where MIN0(91,..)
  does the same to JP1;
* the 15 MHz seam between /cantenna/ entries 1 and 3, inclusive on both sides
  so entry 1 wins the tie;
* frequencies outside every entry's range, and a terminal with no entry at
  all, which must return 0 gain rather than indexing the table.

Nothing here drives FMC below 1 MHz or DELTA below 0: GAIN clamps only the top
of each axis, so those index ARRAY out of bounds in the Fortran.

stdlib-only.  Usage: gen_gainant_cases.py <outdir>
gainant_cases.csv columns:  itr,delta_rad,fmc
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


def row(itr, delta, fmc):
    return f"{int(itr)},{f(delta)},{f(fmc)}"


# elevations in degrees: cell boundaries, mid-cell, and the top clamp
DEGREES = (0.0, 0.25, 1.0, 1.5, 6.0, 12.75, 45.0, 67.3, 89.0, 89.999,
           90.0, 95.0)
# frequencies in MHz: boundaries, either side of them, the seam, the top cell
FREQS = (2.0, 2.0001, 2.5, 3.0, 7.0, 7.5, 9.999, 10.0, 10.001, 14.9, 15.0,
         15.0001, 21.3, 29.0, 29.5, 30.0)


def cases():
    rows = []
    for itr in (1, 2):
        for fmc in FREQS:
            for deg in DEGREES:
                rows.append(row(itr, deg * D2R, fmc))
    # no match: below every entry's range, and a terminal with no entry
    for fmc in (1.5, 1.999):
        rows.append(row(1, 10.0 * D2R, fmc))
        rows.append(row(2, 10.0 * D2R, fmc))
    for fmc in (5.0, 20.0):
        rows.append(row(3, 10.0 * D2R, fmc))
        rows.append(row(0, 10.0 * D2R, fmc))
    # random fill across the interior of both axes
    rnd = lcg(SEED)
    for _ in range(200):
        itr = 1 + int(rnd(0.0, 2.0)) % 2
        rows.append(row(itr, rnd(0.0, 90.0) * D2R, rnd(2.0, 30.0)))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "gainant_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"gainant: {len(rows)} cases")


if __name__ == "__main__":
    main()
