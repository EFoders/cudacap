#!/usr/bin/env python3
"""Deterministic case grid for the FRQCOM + SETLUF dump.

FRQCOM is a six-way branch on two numbers, so the grid is built around where
those two land:

    XFL = max(2, min(EMUF, F2MUF))   the lower MUF, the knee to straddle
    XFH = min(HPF, 40)               the highest usable frequency

* **case 1**  XFH <= 2      no usable band.  The one arm that leaves slot 13
                            alone, so the case seeds it with a sentinel.
* **case 2**  XFL <= 2      nighttime, the whole band above the lower MUF.
* **case 3**  XFH <= 22     the knee planted on an interior slot.  NE is
                            swept from its smallest to its largest SAFE value.
* **case 4**  XFL <= 4      knee between 2 and 4 MHz.
* **case 5**  otherwise     two ladders meeting at the knee.  NE saturates at
                            7, so XFL is swept either side of 16 MHz.
* **case 6**  unreachable.  Its guard is ``(FHIGH-2.)-FLOW`` with both DATA
                            constants, i.e. the constant 36.

Every boundary is probed from both sides and exactly on it, since all five
live comparisons are arithmetic IFs whose zero arm goes with the LOW branch.

IFREQ picks the entry: -10 fills slot 1 with FOT and returns, 0 and -1 build
the list, and anything positive skips FRQCOM entirely so the driver just
drops the MUF into slot 12 -- which is also the only way SETLUF sees the
user's own frequencies.

SETLUF then wants:
* a frequency that reaches the required reliability, at the FIRST slot, an
  interior one and the last -- the scan stops at the first hit.
* nothing reaching it, so the fallback runs and XLUF comes back negative.
* **slot 1 blank**.  RF and REL are seeded from FREL(1)/RELIAB(1) before the
  loop and the loop's skip does not protect the seed, so a blank slot 1 leaves
  RELIAB(1) setting the bar and a non-positive RF able to reach XLUF.
* LUFP at 0 and at 100, the ends of the required-reliability range.

NOT GENERATED: case 3 where NE would exceed 10.  The Fortran writes
FREA(NE+1) with nothing bounding NE, so a lower MUF far above the highest
usable frequency runs off the end of the common block.  ``_ne_case3`` filters
those out; the port raises there instead.

stdlib-only.  Usage: gen_frqcom_cases.py <outdir>
"""

import pathlib
import sys

SEED = 20260727
NF = 11
FLOW, FHIGH = 2.0, 40.0
NE_LIMIT = 10                 # keep well inside the FREA(NE+1) write at 12

COLUMNS = (["it", "ifreq", "allmuf", "fot", "emuf", "f2muf", "hpf", "lufp"]
           + [f"frel{i}" for i in range(1, NF + 1)] + ["fr13"]
           + [f"rel{i}" for i in range(1, NF + 1)])

INTS = {"it", "ifreq", "lufp"}

DEFAULTS = {"it": 12, "ifreq": -1, "allmuf": 18.4, "fot": 14.7,
            "emuf": 9.2, "f2muf": 21.5, "hpf": 24.3, "lufp": 90,
            "fr13": -3333.0}
for _i in range(1, NF + 1):
    DEFAULTS[f"frel{_i}"] = 0.0
    DEFAULTS[f"rel{_i}"] = 0.0


def _ne_case3(emuf, f2muf, hpf) -> int:
    """NE as case 3 computes it, or 0 when case 3 does not run."""
    xfl = max(FLOW, min(emuf, f2muf))
    xfh = min(hpf, FHIGH)
    if xfh <= FLOW or xfl <= FLOW or xfh > FLOW + 20.0:
        return 0
    return int((xfl - FLOW) / ((xfh - FLOW) / 9.0)) + 2


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def case(**kw):
    d = dict(DEFAULTS)
    d.update(kw)
    if _ne_case3(d["emuf"], d["f2muf"], d["hpf"]) > NE_LIMIT:
        return None               # would write past the end of FREA
    return ",".join(str(int(d[c])) if c in INTS else f(d[c]) for c in COLUMNS)


def add(rows, **kw):
    row = case(**kw)
    if row is not None:
        rows.append(row)


def freqs(vals):
    out = {}
    for i in range(1, NF + 1):
        out[f"frel{i}"] = vals[i - 1] if i <= len(vals) else 0.0
    return out


def rels(vals):
    out = {}
    for i in range(1, NF + 1):
        out[f"rel{i}"] = vals[i - 1] if i <= len(vals) else 0.0
    return out


BASE_F = [3.2, 5.1, 7.3, 9.6, 11.8, 14.2, 16.5, 18.9, 21.4, 24.0, 27.5]


def cases():
    rows = []
    # --- the IFREQ dispatch --------------------------------------------------
    for ifreq in (-10, -1, 0, 1, 10):
        add(rows, ifreq=ifreq, **freqs(BASE_F), **rels([0.9] * NF))
    # --- case 1: XFH <= 2, and it leaves slot 13 alone -----------------------
    for hpf in (0.0, 1.0, 1.999, 2.0):
        add(rows, hpf=hpf, emuf=1.5, f2muf=1.5)
        add(rows, hpf=hpf, emuf=30.0, f2muf=30.0)
    add(rows, hpf=2.0, emuf=1.5, f2muf=1.5, fr13=17.25)
    # --- case 2: XFL <= 2 (nighttime) ---------------------------------------
    for emuf in (0.0, 1.0, 2.0, -3.0):
        for hpf in (2.001, 5.0, 22.0, 30.0, 40.0, 55.0):
            add(rows, emuf=emuf, f2muf=emuf + 0.5, hpf=hpf)
    # --- case 3: XFH <= 22, NE swept over its safe range ---------------------
    for xfh in (2.001, 4.0, 8.0, 15.0, 21.999, 22.0):
        for xfl in (2.001, 2.5, 3.0, 4.0, 6.0, 9.0, 13.0, 18.0, 21.0):
            add(rows, emuf=xfl, f2muf=xfl + 3.0, hpf=xfh)
    # --- case 4: 2 < XFL <= 4 with XFH > 22 ----------------------------------
    for xfl in (2.001, 2.5, 3.0, 3.999, 4.0):
        for xfh in (22.001, 25.0, 33.0, 40.0, 48.0):
            add(rows, emuf=xfl, f2muf=xfl + 1.0, hpf=xfh)
    # --- case 5: XFL > 4, NE = (XFL-2)/2 truncated then capped at 7 ----------
    for xfl in (4.001, 4.5, 6.0, 7.999, 8.0, 10.0, 12.0, 15.999, 16.0,
                16.001, 20.0, 28.0, 39.0):
        for xfh in (22.001, 30.0, 40.0, 60.0):
            add(rows, emuf=xfl, f2muf=xfl + 2.0, hpf=xfh)
    # --- which of EMUF/F2MUF wins the AMIN1, and the HPF clamp ---------------
    add(rows, emuf=7.0, f2muf=19.0, hpf=26.0)
    add(rows, emuf=19.0, f2muf=7.0, hpf=26.0)
    add(rows, emuf=12.0, f2muf=12.0, hpf=41.0)
    # --- the hour index, since XLUF is written at IT -------------------------
    for it in (1, 2, 13, 23, 24):
        add(rows, it=it, **freqs(BASE_F), **rels([0.95] * NF), ifreq=1)

    # --- SETLUF: where the required reliability is first reached ------------
    for hit in range(NF):
        r = [0.1] * NF
        r[hit] = 0.99
        add(rows, ifreq=1, **freqs(BASE_F), **rels(r))
    # ... nothing reaches it, so the fallback picks the best ------------------
    add(rows, ifreq=1, **freqs(BASE_F),
        **rels([0.10, 0.55, 0.31, 0.42, 0.28, 0.13, 0.07, 0.04, 0.02,
                0.01, 0.005]))
    add(rows, ifreq=1, **freqs(BASE_F), **rels([0.0] * NF))
    # ... exactly ON the requirement (the .GE. arm) ---------------------------
    for lufp in (0, 1, 50, 89, 90, 91, 99, 100):
        add(rows, ifreq=1, lufp=lufp, **freqs(BASE_F),
            **rels([0.30, 0.60, 0.90, 0.90, 0.85, 0.70, 0.50, 0.30, 0.20,
                    0.10, 0.05]))
    # ... slot 1 blank: the seed is taken anyway ------------------------------
    for f1 in (0.0, -1.0):
        add(rows, ifreq=1, **freqs([f1] + BASE_F[1:]),
            **rels([0.99] + [0.10] * (NF - 1)))
        add(rows, ifreq=1, **freqs([f1] + BASE_F[1:]),
            **rels([0.40] + [0.10, 0.35, 0.38, 0.20] + [0.05] * 6))
        add(rows, ifreq=1, **freqs([f1] + BASE_F[1:]),
            **rels([0.40] + [0.10, 0.35, 0.55, 0.20] + [0.05] * 6))
    # ... holes in the middle of the list ------------------------------------
    add(rows, ifreq=1, **freqs([3.2, 0.0, 7.3, 0.0, 11.8, 0.0, 16.5, 0.0,
                                21.4, 0.0, 27.5]),
        **rels([0.1, 0.9, 0.2, 0.9, 0.3, 0.9, 0.4, 0.9, 0.5, 0.9, 0.6]))

    # --- random fill for breadth --------------------------------------------
    rnd = lcg(SEED)
    for _ in range(260):
        add(rows,
            it=1 + int(rnd(0.0, 24.0)) % 24,
            ifreq=(-10 if rnd(0.0, 1.0) < 0.1 else
                   (1 if rnd(0.0, 1.0) < 0.45 else -1)),
            allmuf=rnd(2.0, 45.0), fot=rnd(1.0, 35.0),
            emuf=rnd(-2.0, 32.0), f2muf=rnd(-2.0, 45.0),
            hpf=rnd(-1.0, 55.0), lufp=int(rnd(0.0, 101.0)),
            fr13=rnd(-10.0, 30.0),
            **freqs(sorted(rnd(0.0, 30.0) for _ in range(NF))),
            **rels([rnd(0.0, 1.0) for _ in range(NF)]))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "frqcom_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"frqcom: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
