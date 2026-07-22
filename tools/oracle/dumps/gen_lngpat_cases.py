#!/usr/bin/env python3
"""Deterministic case grid for the LNGPAT dump.

LNGPAT assembles the long path's single mode from the one ray SELTXR chose at
each end.  It calls CONVH, GETTOP, TABS, BABS (all validated by their own
dumps), plus PRBMUF and PWRDB, and everything else it reads is injected.

What has to be reached:

* **LTXRGM(1) or LTXRGM(2) not positive**, the early return where NOTHING is
  written -- the dump's sentinels have to survive intact.
* **DIKM at or below -1 km**, the arm that sets both losses to zero outright,
  and DIKM just above it, where the same zero comes out the long way (GETTOP
  refuses, DM is zero, DF is floored to zero, and both TABS and BABS return
  their 1000 marker into a zero-length leg).
* **an over-the-top segment existing and not existing** -- the night-day-night
  FI/HI configuration with a frequency inside the E penetration window, and
  the ordinary case where GETTOP returns zero.
* **HPX below, at and above 70 km**, which splits BABS from TABS (BABS keeps
  70, TABS does not).
* **the FHOP and HN floors at 1**, both of which need the end legs to be a
  large fraction of the whole path.
* **the 25 dB decile ceilings**, from both sides.
* **NMODE(1) equal to and different from NMODE(2)**: equal skips the second
  PRBMUF pass entirely, different averages the two layers' spreads.
* **PWRDB finding a transmitter entry and not finding one** (the 30 dBW
  fallback), since the field strength and the signal power both use it.
* **PROB(1) above and below PROB(2)**, the AMIN1 on the last line.
* **PRBMUF unsaturated**, which needs care.  LNGPAT calls it with FGO == FSET,
  so PRBMUF's ``FGO*SIGL/FSET`` collapses to SIGL in absolute MHz -- and the
  real SIGL is a tenth of a megahertz, which puts a 14 MHz signal eighty
  sigma from a 22 MHz MUF and saturates every one of the six probabilities to
  0 or 1.  DSLF and DSUF then come out exactly zero and the decile
  calculation is untested.  So a whole group sweeps FREQ across YMUF with
  wide spreads, and the random fill draws SIGL/SIGU up to 6 MHz.

Held away from the grid, and why:

* NMODE outside 1..4 indexes YMUF off the end of a four-element array.  The
  port raises there; the oracle is never asked to read out of bounds.
* HPFLX at or below zero would put SPHE at or above 1 and SQRT(1-SPHE^2)
  into the negatives.  Kept positive.
* GDFLX at zero divides by zero in HN and FHOP.  Kept positive.
* YMUF/YFOT/YHPF at or below zero send PRBMUF into its step branch, which
  can return exactly 0 and make -10*ALOG10(P) infinite.  Kept positive;
  PRBMUF's step branch has its own dump.
* ITXRCP(2) is never 1 -- LUFFY sets it to KFX and rewrites 1 to 2 -- so the
  transmitter slot (ray, area 1) and the receiver slot can never collide and
  the driver can inject them independently.

stdlib-only.  Usage: gen_lngpat_cases.py <outdir>
"""

import pathlib
import sys

SEED = 20260729

INTS = ["l1", "l2", "itxrcp2", "imode1", "imode2", "numants"]
FLOATS = (
    ["freq", "gcd"]
    + ["delfx1", "delfx2", "hpflx1", "hpflx2", "gdflx1", "gdflx2"]
    + ["andvx1", "andvx2", "advx1", "advx2", "aofx1", "aofx2",
       "grlosx1", "grlosx2", "tgainx1", "tgainx2"]
    + ["efflp", "asm", "dsl", "dsu", "rcnse", "acav", "gyz3", "pwr",
       "prob1", "prob2", "delopt"]
    + ["hi1", "hi2", "hi3", "fi1", "fi2", "fi3"]
    + [f"{a}{j}" for a in ("ymuf", "yfot", "yhpf", "fvmuf", "sigl", "sigu")
       for j in (1, 2, 3, 4)]
)
COLUMNS = INTS + FLOATS

DEFAULTS = {
    "l1": 4, "l2": 7, "itxrcp2": 3, "imode1": 3, "imode2": 3, "numants": 1,
    "freq": 14.0, "gcd": 2.4,
    "delfx1": 8.0, "delfx2": 11.0, "hpflx1": 280.0, "hpflx2": 320.0,
    "gdflx1": 3200.0, "gdflx2": 3400.0,
    "andvx1": 22.0, "andvx2": 26.0, "advx1": 4.0, "advx2": 5.0,
    "aofx1": 0.0, "aofx2": 1.5, "grlosx1": 3.0, "grlosx2": 4.0,
    "tgainx1": 9.0, "tgainx2": 7.0,
    "efflp": 0.8, "asm": 1.5, "dsl": 4.0, "dsu": 5.0, "rcnse": -145.0,
    "acav": 0.4, "gyz3": 1.358, "pwr": 40.0, "prob1": 0.8, "prob2": 0.6,
    "delopt": 3.0,
    "hi1": 110.0, "hi2": 110.0, "hi3": 110.0,
    "fi1": 0.6, "fi2": 0.8, "fi3": 0.6,
}
for _j, _v in enumerate([26.0, 24.0, 22.0, 12.0], 1):
    DEFAULTS[f"ymuf{_j}"] = _v
    DEFAULTS[f"yfot{_j}"] = _v * 0.85
    DEFAULTS[f"yhpf{_j}"] = _v * 1.15
    DEFAULTS[f"fvmuf{_j}"] = _v * 0.35
    DEFAULTS[f"sigl{_j}"] = 0.12
    DEFAULTS[f"sigu{_j}"] = 0.09

# the FI/HI that make GETTOP return a real over-the-top segment: a daylit
# middle penetrating at 4.198 MHz between ends penetrating at 3.149
NIGHT_DAY_NIGHT = {"fi1": 0.6, "fi2": 0.8, "fi3": 0.6}
OTT_FREQ = 3.5


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
    if unknown:
        raise KeyError(f"not a column: {sorted(unknown)}")
    d = dict(DEFAULTS)
    d.update(kw)
    return ",".join(str(int(d[c])) if c in INTS else f(d[c])
                    for c in COLUMNS)


def cases():
    rows = []
    # --- the early return: nothing may be written -------------------------
    for l1, l2 in ((0, 7), (4, 0), (0, 0), (-1, 7), (4, -3), (1, 1)):
        rows.append(case(l1=l1, l2=l2))
    # --- DIKM either side of -1 km, and around zero -----------------------
    # DI = GCD - DT - DR, so GCD is tuned to put DIKM where we want it.
    # DT+DR = .5*(3200+3400)/6370 = 0.51805 rad.
    base = 0.5 * (3200.0 + 3400.0) / 6370.0
    for dikm in (-4000.0, -1000.0, -1.001, -1.0, -0.999, -0.5, 0.0, 0.5,
                 1.0, 100.0, 999.0, 1001.0, 3000.0, 9000.0):
        rows.append(case(gcd=base + dikm / 6370.0))
    # --- an over-the-top segment, and none --------------------------------
    for gcd in (2.0, 2.4, 2.9, 3.1):
        rows.append(case(freq=OTT_FREQ, gcd=gcd, **NIGHT_DAY_NIGHT))
        rows.append(case(freq=14.0, gcd=gcd, **NIGHT_DAY_NIGHT))
        rows.append(case(freq=OTT_FREQ, gcd=gcd, fi1=0.9, fi2=0.5, fi3=0.9))
    # --- HPX below, exactly at, and above 70 km ---------------------------
    for h1, h2 in ((20.0, 20.0), (40.0, 60.0), (60.0, 79.9), (60.0, 80.0),
                   (60.0, 80.1), (70.0, 70.0), (69.0, 71.0), (100.0, 40.0),
                   (110.0, 120.0), (300.0, 320.0)):
        rows.append(case(hpflx1=h1, hpflx2=h2))
        rows.append(case(hpflx1=h1, hpflx2=h2, freq=OTT_FREQ,
                         **NIGHT_DAY_NIGHT))
    # --- the FHOP and HN floors at 1 --------------------------------------
    # both need the end legs to swallow the path: big GDFLX, small GCD
    for gd, gcd in ((3900.0, 0.62), (3900.0, 1.24), (3900.0, 1.30),
                    (2000.0, 0.32), (500.0, 0.09), (500.0, 2.4),
                    (3900.0, 2.4)):
        rows.append(case(gdflx1=gd, gdflx2=gd, gcd=gcd))
    # --- the 25 dB decile ceilings, from both sides ------------------------
    for dsl, dsu, fv in ((0.0, 0.0, 0.35), (4.0, 5.0, 0.35), (24.0, 24.0,
                         0.35), (26.0, 26.0, 0.35), (4.0, 5.0, 0.02),
                         (4.0, 5.0, 0.9)):
        kw = {f"fvmuf{j}": DEFAULTS[f"ymuf{j}"] * fv for j in (1, 2, 3, 4)}
        rows.append(case(dsl=dsl, dsu=dsu, **kw))
        rows.append(case(dsl=dsl, dsu=dsu, freq=30.0, **kw))
    # --- NMODE equal and different at the two ends ------------------------
    for m1 in (1, 2, 3, 4):
        for m2 in (1, 2, 3, 4):
            rows.append(case(imode1=m1, imode2=m2))
    # --- PWRDB: an entry that matches, and the 30 dBW fallback ------------
    for numants, pwr in ((1, 40.0), (1, 20.0), (1, 55.0), (0, 40.0)):
        rows.append(case(numants=numants, pwr=pwr))
    # --- PROB(1) above, below and equal to PROB(2) ------------------------
    for p1, p2 in ((0.9, 0.2), (0.2, 0.9), (0.5, 0.5), (0.0, 1.0)):
        rows.append(case(prob1=p1, prob2=p2))
    # --- FREQ across the band, either side of each YMUF -------------------
    for freq in (2.0, 5.0, 11.9, 12.0, 12.1, 21.9, 22.0, 22.1, 24.0, 26.0,
                 30.0, 40.0):
        rows.append(case(freq=freq))
        rows.append(case(freq=freq, imode1=1, imode2=4))
    # --- PRBMUF unsaturated: wide spreads, FREQ swept across the MUF ------
    wide = {}
    for j in (1, 2, 3, 4):
        wide[f"sigl{j}"] = 2.0
        wide[f"sigu{j}"] = 1.5
    for freq in (10.0, 14.0, 16.0, 18.0, 18.7, 20.0, 21.0, 22.0, 23.0,
                 25.3, 26.0, 28.0, 32.0):
        rows.append(case(freq=freq, **wide))
        rows.append(case(freq=freq, imode1=1, imode2=3, **wide))
        rows.append(case(freq=freq, imode1=4, imode2=4, **wide))
    # --- random fill for breadth ------------------------------------------
    rnd = lcg(SEED)
    for _ in range(320):
        ott = rnd(0.0, 1.0) < 0.25
        rows.append(case(
            l1=1 + int(rnd(0.0, 45.0)) % 45,
            l2=1 + int(rnd(0.0, 45.0)) % 45,
            itxrcp2=2 + int(rnd(0.0, 2.0)) % 2,
            imode1=1 + int(rnd(0.0, 4.0)) % 4,
            imode2=1 + int(rnd(0.0, 4.0)) % 4,
            numants=(1 if rnd(0.0, 1.0) < 0.9 else 0),
            freq=(OTT_FREQ if ott else rnd(2.0, 40.0)),
            gcd=rnd(0.3, 3.14),
            delfx1=rnd(0.0, 60.0), delfx2=rnd(0.0, 60.0),
            hpflx1=rnd(20.0, 450.0), hpflx2=rnd(20.0, 450.0),
            gdflx1=rnd(200.0, 4200.0), gdflx2=rnd(200.0, 4200.0),
            andvx1=rnd(0.0, 120.0), andvx2=rnd(0.0, 120.0),
            advx1=rnd(0.0, 40.0), advx2=rnd(0.0, 40.0),
            aofx1=rnd(0.0, 25.0), aofx2=rnd(0.0, 25.0),
            grlosx1=rnd(0.0, 25.0), grlosx2=rnd(0.0, 25.0),
            tgainx1=rnd(-12.0, 22.0), tgainx2=rnd(-12.0, 22.0),
            efflp=rnd(0.0, 6.0), asm=rnd(-12.0, 12.0),
            dsl=rnd(0.0, 28.0), dsu=rnd(0.0, 28.0),
            rcnse=rnd(-175.0, -110.0),
            acav=rnd(0.0, 3.0), gyz3=rnd(0.3, 2.0), pwr=rnd(10.0, 60.0),
            prob1=rnd(0.0, 1.0), prob2=rnd(0.0, 1.0),
            hi1=rnd(90.0, 135.0), hi2=rnd(90.0, 135.0),
            hi3=rnd(90.0, 135.0),
            fi1=rnd(0.3, 1.1), fi2=rnd(0.3, 1.1), fi3=rnd(0.3, 1.1),
            **{f"ymuf{j}": rnd(4.0, 42.0) for j in (1, 2, 3, 4)},
            **{f"yfot{j}": rnd(3.0, 38.0) for j in (1, 2, 3, 4)},
            **{f"yhpf{j}": rnd(5.0, 48.0) for j in (1, 2, 3, 4)},
            **{f"fvmuf{j}": rnd(1.0, 18.0) for j in (1, 2, 3, 4)},
            **{f"sigl{j}": rnd(0.02, 6.0) for j in (1, 2, 3, 4)},
            **{f"sigu{j}": rnd(0.02, 6.0) for j in (1, 2, 3, 4)},
        ))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "lngpat_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"lngpat: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
