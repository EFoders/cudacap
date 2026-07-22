#!/usr/bin/env python3
"""Deterministic case grid for the GEOTIM + CNGTIM dump.

CNGTIM is a division and a wrap, so what has to be reached is every place the
wrap changes its mind:

* **XTHETA exactly 0**, the early return that leaves TIME untouched -- and
  therefore UNNORMALISED, unlike every other longitude.
* **XTHETA negative**, which is folded by 360 first, and the extremes -359.9
  and -0.1.
* the two THETA thresholds, **180** and **360**, from both sides and exactly
  on them.  They set FINT1/FINT2 and so pick which of the three offset
  formulae runs; 360 is the only case where FINT2 is non-zero and the
  ``TIME - FINT2*24`` subtraction at label 105 does anything.
* **ISW either way**, since the same longitude has to round-trip.
* TIME driven far enough out of range that the normalising loop has to run
  more than one iteration in each direction, and TIME landing exactly on 0
  and on 24 (the loop's bounds are inclusive at both ends).
* the day-change return at -1, 0 and +1.

GEOTIM then wants:

* **ITIM either side of 0**, which swaps which of GMT and XLMT is derived.
* **the hour reaching IT = 0**, where GEOTIM rewrites it to 24.
* **KM at 0**, where the sample-area loop does not run at all, and at its
  full 5.
* transmitter and receiver longitudes that disagree, so GMT, XLMT and GMTR
  are three different numbers.

stdlib-only.  Usage: gen_geotim_cases.py <outdir>
"""

import pathlib
import sys

SEED = 20260728
NC = 5
D2R = 0.01745329251

COLUMNS = (["jt", "itim", "tlong", "rlong", "km"]
           + [f"clong{i}" for i in range(1, NC + 1)]
           + ["ctime", "ctheta", "cisw"])

INTS = {"jt", "itim", "km", "cisw"}

DEFAULTS = {"jt": 12, "itim": 0, "tlong": 0.0, "rlong": 0.0, "km": 3,
            "ctime": 12.0, "ctheta": 45.0, "cisw": 1}
for _i in range(1, NC + 1):
    DEFAULTS[f"clong{_i}"] = 0.0


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
    d = dict(DEFAULTS)
    for key, val in kw.items():
        if key == "clong":
            for i in range(1, NC + 1):
                d[f"clong{i}"] = val[i - 1] if i <= len(val) else 0.0
        else:
            d[key] = val
    return ",".join(str(int(d[c])) if c in INTS else f(d[c]) for c in COLUMNS)


# A realistic spread of sample-area longitudes, in radians as /RON/ holds them
SAMPLES = [d * D2R for d in (-118.2, -60.0, 0.13, 75.0, 139.8)]


def cases():
    rows = []
    # --- CNGTIM: the reference longitude ------------------------------------
    for theta in (0.0, 0.001, -0.001, -0.1, 15.0, 90.0, 179.999, 180.0,
                  180.001, 270.0, 359.999, 360.0, -15.0, -90.0, -180.0,
                  -270.0, -359.9):
        for isw in (1, -1):
            rows.append(case(ctheta=theta, cisw=isw))
    # --- CNGTIM: the time, including several wraps in each direction --------
    for t in (0.0, 0.001, 1.0, 12.0, 23.0, 23.999, 24.0, -0.001, -5.0,
              -30.0, -50.0, 26.0, 48.0, 60.0):
        for theta, isw in ((45.0, 1), (315.0, 1), (45.0, -1), (315.0, -1)):
            rows.append(case(ctime=t, ctheta=theta, cisw=isw))
    # --- GEOTIM: ITIM either side of zero, over the whole hour range --------
    for jt in range(1, 25):
        for itim in (-1, 0, 1):
            rows.append(case(jt=jt, itim=itim, tlong=-0.13 * D2R,
                             rlong=2.35 * D2R, km=3, clong=SAMPLES))
    # --- GEOTIM: hours that reach IT = 0 and get rewritten to 24 ------------
    for tlongd in (-179.0, -90.0, 0.0, 90.0, 179.0):
        for jt in (1, 24):
            rows.append(case(jt=jt, itim=0, tlong=tlongd * D2R,
                             rlong=-tlongd * D2R, km=3, clong=SAMPLES))
            rows.append(case(jt=jt, itim=-1, tlong=tlongd * D2R,
                             rlong=-tlongd * D2R, km=3, clong=SAMPLES))
    # --- GEOTIM: KM from none to all ----------------------------------------
    for km in (0, 1, 2, 3, 4, 5):
        rows.append(case(km=km, clong=SAMPLES, tlong=-1.2, rlong=2.4))
    # --- GEOTIM: a transmitter at exactly Greenwich, the untouched arm ------
    rows.append(case(tlong=0.0, rlong=2.35 * D2R, km=3, clong=SAMPLES))
    rows.append(case(tlong=0.0, rlong=0.0, km=1, clong=[0.0]))
    rows.append(case(jt=24, tlong=0.0, rlong=0.0, km=1, clong=[0.0]))

    # --- random fill for breadth --------------------------------------------
    rnd = lcg(SEED)
    for _ in range(240):
        km = int(rnd(0.0, 6.0)) % 6
        rows.append(case(
            jt=1 + int(rnd(0.0, 24.0)) % 24,
            itim=(-1 if rnd(0.0, 1.0) < 0.5 else 0),
            tlong=rnd(-3.1415, 3.1415), rlong=rnd(-3.1415, 3.1415),
            km=km, clong=[rnd(-3.1415, 3.1415) for _ in range(NC)],
            ctime=rnd(-40.0, 60.0), ctheta=rnd(-360.0, 360.0),
            cisw=(1 if rnd(0.0, 1.0) < 0.5 else -1)))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "geotim_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"geotim: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
