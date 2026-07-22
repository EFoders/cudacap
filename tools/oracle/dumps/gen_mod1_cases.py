#!/usr/bin/env python3
"""Deterministic case grids for the Module-1 tier-B dumps (stdlib-only):
MAGVAR(+MAGFIN), DISTXY, SANG, GEOM.

Usage: gen_mod1_cases.py <outdir>   (writes <name>_cases.csv files)

Values are decimal strings both consumers parse identically into REAL*4 /
float32 — the string in the CSV is the canonical case value (same
convention as gen_dazel_cases.py).
"""

import math
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


def rad(deg) -> float:
    return math.radians(float(deg))


def magvar_cases():
    rows = []
    lats = [-90.0, -89.9, -89.8972, -85.0, -60.0, -30.0, 0.0,
            30.0, 60.0, 85.0, 89.8972, 89.9, 90.0]
    lons = [-180.0, -90.0, -0.001, 0.0, 90.0, 179.9]
    for lat in lats:
        for lon in lons:
            rows.append((rad(lat), rad(lon)))
    # MAGFIN clamps |lat| at RD=1.56905124 rad: straddle it exactly
    for latr in (1.56905124, -1.56905124, 1.569, -1.569, 1.5705, -1.5705):
        rows.append((latr, rad(45.0)))
    rnd = lcg(SEED)
    for _ in range(150):
        rows.append((rad(rnd(-89.99, 89.99)), rad(rnd(-180.0, 180.0))))
    return [f"{f(a)},{f(b)}" for a, b in rows]


def distxy_cases():
    rows = []
    ends = [
        (51.50, -0.13, 48.85, 2.35),      # short
        (35.80, -5.90, 44.90, 20.50),     # medium
        (34.05, -118.24, 35.68, 139.77),  # long, date line
        (-33.87, 151.21, -33.45, -70.67),
        (64.84, -147.72, 60.17, 24.94),   # polar
        (51.50, -0.13, -36.85, 174.76),   # near-antipodal
        (10.0, -45.0, 20.0, 40.0),
    ]
    for nd in (2, 3, 5, 11, 21):
        for (tla, tlo, rla, rlo) in ends:
            ids = range(1, nd + 1) if nd <= 5 else (1, 2, nd // 2, nd - 1, nd)
            for i in ids:
                rows.append((i, nd, rad(tla), rad(tlo), rad(rla), rad(rlo), 0))
    # longitude wrap fix-ups: tlon beyond pi so interpolated rlongd > 360
    for i in (2, 3, 4):
        rows.append((i, 5, rad(80.0), 5.5, rad(70.0), 6.0, 0))
        rows.append((i, 5, rad(-80.0), -5.5, rad(-70.0), -6.0, 1))
    rnd = lcg(SEED + 1)
    for _ in range(150):
        nd = 2 + int(rnd(0, 20))
        i = 1 + int(rnd(0, nd))
        rows.append((i, nd, rad(rnd(-89.0, 89.0)), rad(rnd(-180.0, 180.0)),
                     rad(rnd(-89.0, 89.0)), rad(rnd(-180.0, 180.0)),
                     int(rnd(0, 2))))
    return [f"{r[0]},{r[1]},{f(r[2])},{f(r[3])},{f(r[4])},{f(r[5])},{r[6]}"
            for r in rows]


def sang_cases():
    gcdkms = [1.0, 500.0, 1999.9, 2000.0, 2000.5, 3999.0, 4000.0, 4001.0,
              5999.0, 6000.5, 7999.0, 8000.5, 9999.0, 10000.5, 11999.0,
              12000.5, 13999.0, 14000.5, 16000.0, 19999.0, 25000.0]
    aminds = [-3.0, 0.0, 0.05, 0.5, 3.0, 8.0, 15.0, 25.0, 41.2, 50.0,
              66.0, 76.4, 80.0, 81.4, 81.6, 85.0, 100.0]
    return [f"{f(g)},{f(a)}" for g in gcdkms for a in aminds]


def geom_cases():
    POLE = (79.5, -69.0)  # oracle's database/north_pole.txt
    rows = []

    def add(tla, tlo, rla, rlo, npsl=0, pole=POLE):
        rows.append((pole[0], pole[1], tla, tlo, rla, rlo, npsl))

    paths = [
        (51.50, -0.13, 48.85, 2.35),
        (35.80, -5.90, 44.90, 20.50),
        (34.05, -118.24, 35.68, 139.77),
        (-33.87, 151.21, -33.45, -70.67),
        (64.84, -147.72, 60.17, 24.94),
        (35.68, 139.77, -33.87, 151.21),
        (51.50, -0.13, -36.85, 174.76),
    ]
    for p in paths:
        add(*p)
        add(*p, npsl=1)
        add(*p, pole=(78.5, -69.0))
        add(*p, pole=(82.0, -82.0))
    # same-point nudge branch (ddx=0.03), both rlatd signs and the boundary
    add(10.0, 20.0, 10.01, 20.02)
    add(-10.0, 20.0, -10.01, 20.02)
    add(10.0, 20.0, 10.03, 20.03)
    add(10.0, 20.0, 10.031, 20.0)
    add(0.0, 0.0, 0.0, 0.0)
    add(0.0, 0.0, -0.001, 0.001)
    # poles: near-pole branch (cos(lat) < 1e-7) and the >89.9 rlong reset
    for lat in (90.0, -90.0, 89.99999, 89.95, -89.95):
        add(lat, 30.0, 20.0, 40.0)
        add(20.0, 40.0, lat, 30.0)
    add(90.0, 0.0, -90.0, 0.0)
    # KM band boundaries (same-longitude paths, dlat sets distance)
    for dlat in (17.9, 17.99, 18.01, 18.1, 35.9, 35.97, 36.05, 54.0, 72.0):
        add(0.0, 10.0, dlat, 10.0)
    # DLONG wrap and near-antipodal clamp
    add(10.0, -170.0, 20.0, 170.0)
    add(10.0, 170.0, 20.0, -170.0)
    add(10.0, 20.0, -10.0, -160.0)
    add(45.0, 0.0, -45.0, 180.0)
    rnd = lcg(SEED + 2)
    for k in range(200):
        add(rnd(-89.5, 89.5), rnd(-180.0, 180.0),
            rnd(-89.5, 89.5), rnd(-180.0, 180.0),
            npsl=1 if k % 7 == 3 else 0)
    return [f"{f(r[0])},{f(r[1])},{f(r[2])},{f(r[3])},{f(r[4])},{f(r[5])},"
            f"{r[6]}" for r in rows]


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    for name, gen in [("magvar", magvar_cases), ("distxy", distxy_cases),
                      ("sang", sang_cases), ("geom", geom_cases)]:
        rows = gen()
        with open(outdir / f"{name}_cases.csv", "w", newline="\n") as fh:
            fh.write("\n".join(rows) + "\n")
        print(f"{name}: {len(rows)} cases")


if __name__ == "__main__":
    main()
