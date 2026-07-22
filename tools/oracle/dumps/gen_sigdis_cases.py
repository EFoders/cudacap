#!/usr/bin/env python3
"""Deterministic case grid for the SIGDIS (signal-distribution) dump.

Each case supplies REDMAP inputs (month/SSN -> SYS/PERR/F2D), the sample-area
count KFX (1..3) and circuit-mode area JMODE (<=KFX), the path length GCDKM
(spanning the 2500 km SYSSY distance-index threshold), the ionogram model flag
IEDP, three areas' layer parameters FI/YI/HI + geomagnetic latitude GLAT (both
hemispheres, so the path-average latitude reaches every FTAB branch) + local
time CLCK, and the F2/Es distribution medians YFOT/YHPF/YMUF and deciles
SIGL/SIGU (layer 3=F2, 4=Es; Es zeroed on ~40% of cases to exercise the
"no Es" branches).  stdlib-only.

Usage: gen_sigdis_cases.py <outdir>
Columns: month,ssn,kfx,jmode,gcdkm,iedp,
  {fi1,fi2,fi3,yi1,yi2,yi3,hi1,hi2,hi3,glat,clck}x3,
  yfot3,yfot4,yhpf3,yhpf4,ymuf3,ymuf4,sigl3,sigl4,sigu3,sigu4
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


def area(rnd):
    fi1 = rnd(1.0, 4.0)
    fi3 = rnd(max(fi1 + 1.0, 4.0), 15.0)
    if rnd(0.0, 1.0) < 0.55:
        fi2 = rnd(fi1 + 0.3, min(fi1 + 3.0, fi3 - 0.3))
        yi2 = rnd(30.0, 50.0)
        hi2 = rnd(160.0, 220.0)
    else:
        fi2 = yi2 = hi2 = 0.0
    yi1 = rnd(18.0, 24.0)
    yi3 = rnd(60.0, 140.0)
    hi1 = rnd(105.0, 115.0)
    hi3 = rnd(max(280.0, hi2 + 60.0), 450.0)
    glat = rnd(-1.4, 1.4)   # radians, both hemispheres
    clck = rnd(0.0, 24.0)
    return [fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, glat, clck]


def cases():
    rows = []
    rnd = lcg(SEED)
    for _ in range(130):
        month = 1 + int(rnd(0.0, 12.0)) % 12
        ssn = rnd(8.0, 160.0)
        kfx = 1 + int(rnd(0.0, 3.0)) % 3
        jmode = 1 + int(rnd(0.0, float(kfx))) % kfx
        gcdkm = rnd(400.0, 8000.0)
        iedp = -1 if rnd(0.0, 1.0) < 0.5 else 0
        vals = [str(month), f(ssn), str(kfx), str(jmode), f(gcdkm), str(iedp)]
        for _is in range(3):
            vals.extend(f(v) for v in area(rnd))
        # F2 (layer 3): FOT < MUF < HPF
        ymuf3 = rnd(5.0, 35.0)
        yfot3 = ymuf3 * rnd(0.75, 0.92)
        yhpf3 = ymuf3 * rnd(1.02, 1.18)
        # Es (layer 4): present on ~60% of cases
        if rnd(0.0, 1.0) < 0.60:
            ymuf4 = rnd(2.0, 15.0)
            yfot4 = ymuf4 * rnd(0.75, 0.95)
            yhpf4 = ymuf4 * rnd(1.02, 1.20)
        else:
            ymuf4 = yfot4 = yhpf4 = 0.0
        sigl3, sigu3 = rnd(0.05, 0.25), rnd(0.05, 0.25)
        sigl4, sigu4 = rnd(0.05, 0.25), rnd(0.05, 0.25)
        vals.extend(f(v) for v in (yfot3, yfot4, yhpf3, yhpf4, ymuf3, ymuf4,
                                   sigl3, sigl4, sigu3, sigu4))
        rows.append(",".join(vals))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "sigdis_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"sigdis: {len(rows)} cases")


if __name__ == "__main__":
    main()
