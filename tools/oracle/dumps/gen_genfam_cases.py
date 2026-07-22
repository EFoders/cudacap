#!/usr/bin/env python3
"""Deterministic case grid for the GENFAM (atmospheric-noise) dump.

GENFAM takes explicit arguments and reads only the FAM/DUD tables REDMAP puts
in /TWO/, so every case is a direct call.  What has to be covered:

* **Both hemispheres** -- the sign of Y2 alone shifts IBK by 6 into the
  southern half of the tables, so every block is exercised twice.
* **All six local-time blocks**, since IBK indexes the tables directly.
* **The two frequency clamps**, which are the part of this routine most
  likely to be got wrong: X is pinned at log10(20) above 20 MHz for all five
  DUD polynomials, and again at 1.0 above 10 MHz for the fifth (sigma-Fam)
  alone.  Cases sit either side of and exactly on 10 and 20 MHz.
* **Months**, which select which DUD/FAM tables are loaded at all.
* **Z**, the 1 MHz Fam, which enters only through the CZ inversion and so
  scales FA without touching the deciles -- swept over the range ANOIS1's
  NOISY maps actually produce.

Cases are grouped by month: the driver reloads the tables only when the month
changes, and the whole file is one long run of REDMAP calls otherwise.

stdlib-only.  Usage: gen_genfam_cases.py <outdir>
genfam_cases.csv columns:  month,y2_rad,iblk,freq,z
"""

import pathlib
import sys

SEED = 20260723

# either side of, and exactly on, both Report 322 curve endpoints
FREQS_CLAMP = [1.0, 2.0, 5.0, 9.99, 10.0, 10.01, 14.0,
               19.99, 20.0, 20.01, 25.0, 30.0, 55.0]
# the frequencies GENOIS itself will mostly ask for
FREQS_MAIN = [2.0, 5.0, 10.0, 20.0, 30.0]
ZS = [0.0, 30.0, 60.0, 90.0, 130.0]


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def row(month, y2, iblk, freq, z):
    return f"{int(month)},{f(y2)},{int(iblk)},{f(freq)},{f(z)}"


def cases():
    rows = []
    # every block x hemisphere, over the working frequency range
    for iblk in range(1, 7):
        for y2 in (0.7, -0.7):
            for freq in FREQS_MAIN:
                for z in (30.0, 90.0):
                    rows.append(row(6, y2, iblk, freq, z))
    # the frequency clamps, in four seasons and both hemispheres
    for month in (1, 4, 7, 10):
        for y2 in (0.5, -0.5):
            for freq in FREQS_CLAMP:
                rows.append(row(month, y2, 3, freq, 60.0))
    # every month, so every loaded table is touched
    for month in range(1, 13):
        for iblk in (1, 4):
            for y2 in (1.2, -1.2):
                for freq in (7.5, 25.0):
                    rows.append(row(month, y2, iblk, freq, 70.0))
    # Z sweep: FA scales with it, the deciles do not
    for z in ZS:
        for iblk in (2, 5):
            rows.append(row(8, 0.3, iblk, 6.0, z))
            rows.append(row(8, -0.3, iblk, 6.0, z))
    # random fill for breadth
    rnd = lcg(SEED)
    for _ in range(300):
        month = 1 + int(rnd(0.0, 12.0)) % 12
        y2 = rnd(-1.55, 1.55)
        iblk = 1 + int(rnd(0.0, 6.0)) % 6
        freq = rnd(1.0, 55.0)
        z = rnd(0.0, 140.0)
        rows.append(row(month, y2, iblk, freq, z))
    # group by month so the driver reloads the tables as rarely as possible
    rows.sort(key=lambda r: int(r.split(",", 1)[0]))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "genfam_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"genfam: {len(rows)} cases")


if __name__ == "__main__":
    main()
