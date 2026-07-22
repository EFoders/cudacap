#!/usr/bin/env python3
"""Case grid for the SETLNG (sample-area replication) dump (stdlib-only).

SETLNG's behaviour is selected entirely by KFX, so the grid just sweeps KFX
across all three branches -- KFX<=1 (one computed area -> 2..5 / ionogram
2..3), KFX==2 (three areas -> 4..5, ionogram 2->3), KFX>=3 (no-op) -- with a
few SALT values that shift the injected sentinels so cases are distinct.  The
sentinel VALUES themselves are generated identically on both sides from
(KFX, SALT); the CSV only needs to carry those two.

Usage: gen_setlng_cases.py <outdir>
setlng_cases.csv columns:  kfx,salt
"""

import pathlib
import sys

KFXS = [-1, 0, 1, 2, 3, 4, 5]
SALTS = [0, 3, 7]


def cases():
    return [f"{kfx},{salt}" for kfx in KFXS for salt in SALTS]


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "setlng_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"setlng: {len(rows)} cases")


if __name__ == "__main__":
    main()
