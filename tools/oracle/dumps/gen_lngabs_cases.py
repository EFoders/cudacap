#!/usr/bin/env python3
"""Deterministic case grid for the BABS + TABS + CONVH dump.

The three leaves LNGPAT calls to price its geometry.  All three are pure --
they read /FRQ/, /CON/, /GEOG/ and /SIGD/ and write only their output
arguments -- so one grid drives all three, with FREQ, GYZ(3), DEL and HP
shared exactly as LNGPAT shares them.

What has to be reached:

BABS  (DEL,HP,DF -> AF)
* **DF negative, DF exactly zero, DF positive.**  The arithmetic IF puts the
  zero case with the dead branch, so DF = 0 returns the 1000 marker.
* **HP either side of 70 AND EXACTLY ON IT.**  BABS's ``IF(HP-70.) 105,115,115``
  sends 70.0 to the LIVE arm.
* ACAV at zero (a real answer of exactly 0.0) and at realistic magnitudes.
* DEL large enough that PSI = PIO2-PHE-DEL is near zero and XMHOP blows up --
  the routine's only ill-conditioned corner.  Stopped short of DEL = PIO2,
  where PSI underflows to exactly zero and the division is infinite.

TABS  (DEL,HP,DM -> AM)
* the same three DM arms, with the guard written in the mirrored order.
* **HP exactly 70 taking the DEAD arm**, which is the opposite of BABS from
  the same input -- ``IF(HP-70.) 110,110,115``.
* TABS has no ACAV: its index is the hardcoded 0.1, so an ACAV of zero must
  still leave AM non-zero.

CONVH (GD,PHE,DEL,HP,RAY -> CH,PTOT)
* **|sin(GD)| under the 1e-6 floor**: GD at 0, at 1e-7, and at both float32
  neighbours of pi (blkdat's PI, 3.1415926, and 3.1415927).
* **GM = GD-2*PSI under the 0.001 floor** and above it.
* **CH at the 15 dB ceiling** and below it.  Note the ceiling makes the
  sin floor unobservable -- any GD close enough to the antipode for the floor
  to bite has already clamped -- so both are probed independently.

FREQ + GYZ(3) is kept at or above zero throughout: powf of a negative base
with a fractional exponent is NaN, and neither routine guards it.  DEL is
kept below PIO2 for the same reason (PSI -> 0), and RAY non-negative and DEL
inside a right angle so CONVH's ALOG10 argument stays positive.

stdlib-only.  Usage: gen_lngabs_cases.py <outdir>
"""

import pathlib
import sys

SEED = 20260727

COLUMNS = ["freq", "gyz3", "acav", "del", "hp", "df", "dm", "gd", "phe",
           "ray"]

DEFAULTS = {
    "freq": 14.0, "gyz3": 1.358, "acav": 0.4, "del": 0.15, "hp": 300.0,
    "df": 1.2, "dm": 0.6, "gd": 2.4, "phe": 1.35, "ray": 900.0,
}

# float32 neighbours of pi, and blkdat's own truncated PI
PI_LOW = 3.1415925
PI_BLK = 3.1415926
PI_HIGH = 3.1415927
PIO2 = 1.570796326


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def case(**kw) -> str:
    unknown = set(kw) - set(COLUMNS)
    if unknown:                       # "del" is a keyword: **{"del": x} only
        raise KeyError(f"not a column: {sorted(unknown)}")
    d = dict(DEFAULTS)
    d.update(kw)
    return ",".join(f(d[c]) for c in COLUMNS)


def cases():
    rows = []
    # --- BABS/TABS: the leg arc either side of zero and exactly on it ------
    for leg in (-3.0, -1.0, -1e-6, 0.0, 1e-6, 0.01, 0.5, 1.2, 3.0):
        rows.append(case(df=leg, dm=leg))
        rows.append(case(df=leg, dm=1.0))          # only BABS's arm dead
        rows.append(case(df=1.0, dm=leg))          # only TABS's arm dead
    # --- the 70 km boundary from both sides and exactly on it -------------
    for hp in (-10.0, 0.0, 30.0, 69.9, 69.99999, 70.0, 70.00001, 70.1,
               90.0, 110.0, 300.0, 500.0):
        rows.append(case(hp=hp))
        rows.append(case(hp=hp, **{"del": 0.0}))
    # --- ACAV: zero (AF is exactly 0) and a spread of magnitudes ----------
    for acav in (0.0, 0.001, 0.05, 0.4, 1.0, 3.0, 12.0):
        rows.append(case(acav=acav))
    # --- FREQ + GYZ(3): zero base, small, large ---------------------------
    for freq, gyz3 in ((0.0, 0.0), (0.5, 0.0), (2.0, 0.5), (2.0, 1.358),
                       (7.0, 1.358), (14.0, 0.5), (14.0, 1.8), (30.0, 1.0),
                       (40.0, 1.8), (0.2, 0.2)):
        rows.append(case(freq=freq, gyz3=gyz3))
    # --- DEL: from zero up to where PSI nearly vanishes --------------------
    for d in (0.0, 0.02, 0.15, 0.4, 0.7854, 1.0, 1.3, 1.45, 1.5, 1.55):
        rows.append(case(**{"del": d}))
        rows.append(case(**{"del": d, "hp": 71.0}))
    # --- CONVH: |sin(GD)| under and over the 1e-6 floor -------------------
    for gd in (0.0, 1e-7, 5e-7, 1e-6, 2e-6, 1e-4, 0.001, 0.5, 1.5707963,
               2.5, 3.0, 3.1, PI_LOW, PI_BLK, PI_HIGH):
        rows.append(case(gd=gd))
        rows.append(case(gd=gd, ray=0.0, phe=0.0))
    # --- CONVH: the GM floor, either side of 0.001 ------------------------
    # GM = GD - 2*(PIO2 - DEL - PHE); pick GD to land on/around the floor.
    for target in (-2.0, -0.1, -0.001, 0.0, 0.0005, 0.001, 0.002, 0.05, 1.0):
        d, p = 0.3, 0.9
        gd = target + 2.0 * (PIO2 - d - p)
        rows.append(case(gd=max(gd, 0.0), phe=p, **{"del": d}))
    # --- CONVH: the 15 dB ceiling, either side ----------------------------
    for ray, gd in ((0.0, 1.0), (10.0, 1.0), (100.0, 1.0), (900.0, 1.0),
                    (3000.0, 1.0), (0.0, 0.3), (900.0, 3.0), (900.0, 0.02),
                    (0.0, 3.14), (5000.0, 3.14)):
        rows.append(case(ray=ray, gd=gd, phe=0.6, **{"del": 0.3}))
    # --- CONVH: PHE across its range --------------------------------------
    for p in (0.0, 0.1, 0.5, 1.0, 1.3, 1.5):
        rows.append(case(phe=p))
        rows.append(case(phe=p, gd=0.4))
    # --- random fill for breadth ------------------------------------------
    rnd = lcg(SEED)
    for _ in range(320):
        rows.append(case(
            freq=rnd(0.0, 42.0),
            gyz3=rnd(0.0, 2.0),
            acav=rnd(0.0, 6.0),
            hp=rnd(40.0, 520.0),
            df=rnd(-0.4, 3.2),
            dm=rnd(-0.4, 3.2),
            gd=rnd(0.0, PI_BLK),
            phe=rnd(0.0, 1.5),
            ray=rnd(0.0, 3200.0),
            **{"del": rnd(0.0, 1.5)},
        ))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "lngabs_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"lngabs: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
