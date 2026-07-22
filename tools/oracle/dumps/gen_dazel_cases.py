#!/usr/bin/env python3
"""Generate the DAZEL test-case grid (deterministic, stdlib-only).

Emits dazel_cases.csv rows: mode,tlat,tlon,tht,rlat,rlon,rht,taz,dgc
- mode 0: tlat..rht are inputs; taz/dgc columns are ignored (overwritten).
- mode 1: tlat,tlon,tht,rht,taz,dgc are inputs; rlat/rlon are outputs.

The grid deliberately exercises every branch of dazel.for: same-point (three
height sub-cases), same-longitude (both delat signs), the DELON +/-360
normalization loops, date-line and equator crossings, polar and
near-antipodal paths, plus a seeded pseudorandom sample. Separations below
DAZEL's documented ~5e-5 degree validity floor are excluded except the
same-point branch (which short-circuits before the unstable trig).

DAZEL's mode-1 path divides by sin(colat)*sin(b): poles and zero/antipodal
distances are excluded there (undefined in the Fortran too).
"""

import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "dazel_cases.csv"


def lcg(seed=12345):
    x = seed
    while True:
        x = (48271 * x) % 2147483647
        yield x / 2147483647.0


def fmt(v):
    return f"{v:.9g}"


rows = []


def m0(tlat, tlon, tht, rlat, rlon, rht):
    rows.append((0, tlat, tlon, tht, rlat, rlon, rht, 0.0, 0.0))


def m1(tlat, tlon, tht, rht, taz, dgc):
    rows.append((1, tlat, tlon, tht, 0.0, 0.0, rht, taz, dgc))


# --- mode 0: curated branch coverage ---
# same point (label 24): delht =0, >0, <0
m0(40.0, -105.0, 100.0, 40.0, -105.0, 100.0)
m0(40.0, -105.0, 0.0, 40.0, -105.0, 3000.0)
m0(40.0, -105.0, 3000.0, 40.0, -105.0, 0.0)
# within the 1e-5 same-point window but not identical
m0(40.0, -105.0, 0.0, 40.000005, -105.000005, 0.0)
# same longitude (label 40), both delat signs, equator crossing, long N-S
m0(40.0, -105.0, 0.0, 50.0, -105.0, 0.0)
m0(50.0, -105.0, 0.0, 40.0, -105.0, 0.0)
m0(-30.0, 10.0, 0.0, 10.0, 10.0, 0.0)
m0(-60.0, 170.0, 200.0, 75.0, 170.0, 0.0)
# generic paths
m0(35.8, -5.9, 0.0, 44.9, 20.5, 0.0)          # TANGIER -> BELGRADE (test01)
m0(34.05, -118.25, 100.0, 35.68, 139.69, 50.0)  # LA -> Tokyo (date line)
m0(-33.87, 151.21, 0.0, -33.45, -70.66, 500.0)  # Sydney -> Santiago
m0(80.0, -30.0, 0.0, 85.0, 150.0, 0.0)          # over the pole
m0(10.0, 20.0, 0.0, -10.0, -159.99, 0.0)        # near-antipodal
m0(45.0, 7.0, 0.0, 45.001, 7.001, 0.0)          # short path
m0(0.0, 0.0, 0.0, 0.0, 90.0, 0.0)               # both on equator
m0(0.5, -0.5, 0.0, -0.5, 0.5, 0.0)              # equator crossing
m0(-89.9, 0.0, 0.0, 89.9, 0.0, 0.0)             # pole-to-pole, same lon
m0(-90.0, 12.0, 0.0, 45.0, 13.0, 0.0)           # lat clamp branch (<= -90)
m0(90.0, 12.0, 0.0, -45.0, 13.0, 0.0)           # lat clamp branch (>= 90)
# DELON normalization loop coverage (inputs beyond +/-180)
m0(10.0, 350.0, 0.0, 20.0, 20.0, 0.0)
m0(10.0, -350.0, 0.0, 20.0, -20.0, 0.0)
m0(10.0, 719.0, 0.0, 20.0, -719.0, 0.0)
# height variations on one geometry
for tht, rht in [(0.0, 0.0), (100.0, 0.0), (0.0, 1500.0), (2000.0, 3000.0)]:
    m0(35.8, -5.9, tht, 44.9, 20.5, rht)

# --- mode 0: pseudorandom sample ---
r = lcg(20260721)
heights = [0.0, 50.0, 300.0, 3000.0]
for _ in range(200):
    tlat = round(-89.0 + 178.0 * next(r), 4)
    tlon = round(-180.0 + 360.0 * next(r), 4)
    rlat = round(-89.0 + 178.0 * next(r), 4)
    rlon = round(-180.0 + 360.0 * next(r), 4)
    tht = heights[int(next(r) * 4) % 4]
    rht = heights[int(next(r) * 4) % 4]
    m0(tlat, tlon, tht, rlat, rlon, rht)

# --- mode 1: curated ---
for taz in [0.0, 45.0, 90.0, 135.0, 179.0, 181.0, 270.0, 359.0]:
    m1(35.8, -5.9, 0.0, 0.0, taz, 2500.0)
for dgc in [100.0, 1000.0, 10000.0, 19000.0]:
    m1(-33.87, 151.21, 100.0, 2000.0, 77.0, dgc)
m1(0.0, 0.0, 0.0, 0.0, 90.0, 10007.0)   # quarter circumference along equator
m1(60.0, 120.0, 0.0, 0.0, 200.0, 8000.0)

# --- mode 1: pseudorandom sample ---
for _ in range(80):
    tlat = round(-80.0 + 160.0 * next(r), 4)
    tlon = round(-180.0 + 360.0 * next(r), 4)
    taz = round(360.0 * next(r), 4)
    dgc = round(100.0 + 18000.0 * next(r), 4)
    tht = heights[int(next(r) * 4) % 4]
    rht = heights[int(next(r) * 4) % 4]
    m1(tlat, tlon, tht, rht, taz, dgc)

with open(OUT, "w", newline="\n") as f:
    for row in rows:
        f.write(",".join([str(row[0])] + [fmt(v) for v in row[1:]]) + "\n")

print(f"{len(rows)} cases -> {OUT}")
