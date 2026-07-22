#!/usr/bin/env python3
"""Deterministic case grid for the ESMOD + ESREG dump.

Both routines are driven together in the order LUFFY calls them, on disjoint
/ZON/ slots (ESMOD 4-5, ESREG 6-7), so the dump also pins that neither
disturbs the other's.

Every input is injected -- as with REGMOD, ESMOD reads plain scalars and
interpolates no upstream table, so the FINDF/IFOB replay rule does not apply.
The branches that have to be reached:

* **The sample-area scan**, which picks the area with the LOWEST FS(2,.).
  With KM > 1 the minimum is placed at each end and in the middle.
* **The three early returns**: FREQ at or below FSDEAD (the IFOB-derived low
  cut-off, itself clamped to 3 MHz), FS(2,K) non-positive, and a path longer
  than two Es hops.
* **IHSRT 1 vs 2** -- one Es hop reaches about 2300 km for HS = 110, so a
  path beyond that starts at IHOP = 2 and writes slot 4 while leaving slot 5
  at its presets.  That is the case most likely to be mis-ported as "slot 4
  is the one-hop mode".
* **The AMIND cut**, which skips a hop mid-loop and leaves its slot preset.
* **The ADX correction**, taken only when FI(1,K) exceeds the Es mode's
  vertical-incidence frequency, i.e. an Es layer below the E layer.
* **The 0.90 PRBMUF clamps** on all three reflection probabilities, reached
  by putting the frequency well below the Es MUF.
* **The 25 dB decile clamp**.
* **KM 1/3/5**, which changes the ground-loss average as well as the scan.
* **The antenna ranges**, so both GAIN's no-match return and PWRDB's 30 dBW
  default are exercised.

Column order is fixed and shared with dump_esmod.f and test_esmod.py.

stdlib-only.  Usage: gen_esmod_cases.py <outdir>
"""

import pathlib
import sys

SEED = 20260724
RZ = 6370.0

COLUMNS = (
    ["jmode", "freq", "gcdkm", "amind", "km", "ifob13",
     "acav", "afe", "bfe", "asm", "dsl", "dsu", "rcnse",
     "sigl4", "sigu4", "ymuf4", "yfot4", "yhpf4"]
    + [f"fs1_{i}" for i in range(1, 6)]
    + [f"fs2_{i}" for i in range(1, 6)]
    + [f"fs3_{i}" for i in range(1, 6)]
    + [f"hs{i}" for i in range(1, 6)]
    + [f"fi1_{i}" for i in range(1, 6)]
    + [f"gyz{i}" for i in range(1, 6)]
    + [f"sigpat{i}" for i in range(1, 6)]
    + [f"epspat{i}" for i in range(1, 6)]
    + ["pwrdba", "txlo", "txhi", "rxlo", "rxhi"]
)

GROUNDS = [(5.0, 80.0), (0.01, 30.0), (0.03, 30.0), (0.005, 15.0),
           (0.002, 10.0)]

DEFAULTS = {
    "jmode": 1, "freq": 10.0, "gcdkm": 1500.0, "amind": 3.0, "km": 1,
    "ifob13": 2500, "acav": 0.12, "afe": 0.5, "bfe": 1.5, "asm": 3.0,
    "dsl": 8.0, "dsu": 9.0, "rcnse": -145.0,
    "sigl4": 0.15, "sigu4": 0.20, "ymuf4": 14.0, "yfot4": 11.0,
    "yhpf4": 17.0,
    "pwrdba": 40.0, "txlo": 2.0, "txhi": 30.0, "rxlo": 2.0, "rxhi": 30.0,
}
for _i in range(1, 6):
    DEFAULTS[f"fs1_{_i}"] = 4.0
    DEFAULTS[f"fs2_{_i}"] = 5.0
    DEFAULTS[f"fs3_{_i}"] = 7.0
    DEFAULTS[f"hs{_i}"] = 110.0
    DEFAULTS[f"fi1_{_i}"] = 3.0
    DEFAULTS[f"gyz{_i}"] = 1.2
    DEFAULTS[f"sigpat{_i}"] = GROUNDS[_i - 1][0]
    DEFAULTS[f"epspat{_i}"] = GROUNDS[_i - 1][1]


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
    d.update(kw)
    return ",".join(f(d[c]) for c in COLUMNS)


def spread(name, values):
    """Per-sample-area values from a 5-long list.

    Names that already end in a digit (fs1, fs2, fs3, fi1) take an underscore
    before the area index so the column stays readable.
    """
    sep = "_" if name[-1].isdigit() else ""
    return {f"{name}{sep}{i + 1}": v for i, v in enumerate(values)}


def cases():
    rows = []
    # --- hop count: one Es hop reaches ~2300 km at HS=110 -----------------
    for gcdkm in (200.0, 1000.0, 2200.0, 2290.0, 2300.0, 2400.0, 3500.0,
                  4500.0, 4590.0, 4600.0, 6000.0, 12000.0):
        rows.append(case(gcdkm=gcdkm))
    # the same sweep with a higher and a lower Es layer, which moves SDMAX
    for hs in (95.0, 100.0, 120.0, 140.0):
        for gcdkm in (1500.0, 2400.0, 4000.0, 5200.0):
            rows.append(case(gcdkm=gcdkm, **spread("hs", [hs] * 5)))
    # --- FSDEAD cut-off: IFOB/1000 clamped at 3 MHz ------------------------
    for ifob in (500, 1500, 2999, 3000, 3001, 9000):
        for freq in (0.6, 1.5, 2.9, 3.0, 3.1, 8.0):
            rows.append(case(ifob13=ifob, freq=freq))
    # --- no Es layer at all ------------------------------------------------
    for fs2 in (0.0, -1.0, -5.0):
        rows.append(case(**spread("fs2", [fs2] * 5)))
    # --- sample-area scan: the minimum at each end and in the middle -------
    for km in (3, 5):
        for pos in range(km):
            fs2 = [6.0] * 5
            fs2[pos] = 2.5
            rows.append(case(km=km, **spread("fs2", fs2)))
        # a distinct HS per area, so which K won is visible in HP
        rows.append(case(km=km, **spread("hs", [95.0, 105.0, 115.0,
                                                125.0, 135.0]),
                         **spread("fs2", [9.0, 7.0, 3.0, 8.0, 6.0])))
    # ties: the scan keeps the FIRST, since only a strict decrease moves K
    rows.append(case(km=5, **spread("fs2", [4.0] * 5),
                     **spread("hs", [100.0, 108.0, 116.0, 124.0, 132.0])))
    # --- AMIND cut, on either side of the take-off angle -------------------
    for amind in (0.0, 1.0, 3.0, 6.0, 10.0, 20.0, 45.0, 89.0):
        rows.append(case(amind=amind, gcdkm=1500.0))
        rows.append(case(amind=amind, gcdkm=3800.0))
    # --- ADX: an Es layer below the E layer --------------------------------
    for fi1 in (0.5, 2.0, 5.0, 9.0, 14.0):
        for freq in (4.0, 9.0, 16.0):
            rows.append(case(freq=freq, **spread("fi1", [fi1] * 5)))
    # --- the 0.90 PRBMUF clamps and the 25 dB decile clamp -----------------
    for freq in (3.5, 4.0, 5.0, 6.0, 12.0, 25.0):
        rows.append(case(freq=freq, **spread("fs2", [8.0] * 5),
                         **spread("fs1", [7.0] * 5),
                         **spread("fs3", [11.0] * 5)))
    for dsl, dsu in ((0.0, 0.0), (12.0, 14.0), (24.0, 24.0), (40.0, 60.0)):
        rows.append(case(dsl=dsl, dsu=dsu, freq=4.0))
    # --- antenna ranges: GAIN no-match and PWRDB's 30 dBW default ----------
    for freq in (3.0, 11.9, 12.0, 18.0, 18.1, 29.0):
        rows.append(case(freq=freq, txlo=12.0, txhi=18.0))
        rows.append(case(freq=freq, rxlo=12.0, rxhi=18.0))
        rows.append(case(freq=freq, txlo=12.0, txhi=18.0,
                         rxlo=12.0, rxhi=18.0))
    # --- ground constants over all five areas ------------------------------
    for km in (1, 3, 5):
        rows.append(case(km=km, gcdkm=1800.0))
        rows.append(case(km=km, gcdkm=3600.0))
        rows.append(case(km=km, **spread("sigpat", [0.0] * 5)))
    # --- random fill for breadth -------------------------------------------
    rnd = lcg(SEED)
    for _ in range(320):
        km = 1 + int(rnd(0.0, 5.0)) % 5
        fs2 = [rnd(0.0, 14.0) for _ in range(5)]
        rows.append(case(
            jmode=1 + int(rnd(0.0, 3.0)) % 3,
            freq=rnd(1.0, 30.0),
            gcdkm=rnd(100.0, 7000.0),
            amind=rnd(0.0, 25.0),
            km=km,
            ifob13=int(rnd(200.0, 9000.0)),
            acav=rnd(0.02, 1.2),
            afe=rnd(-2.0, 4.0), bfe=rnd(0.0, 6.0), asm=rnd(0.0, 12.0),
            dsl=rnd(0.0, 20.0), dsu=rnd(0.0, 20.0), rcnse=rnd(-190.0, -90.0),
            sigl4=rnd(0.05, 0.4), sigu4=rnd(0.05, 0.4),
            ymuf4=rnd(3.0, 30.0), yfot4=rnd(2.0, 25.0), yhpf4=rnd(4.0, 35.0),
            pwrdba=rnd(10.0, 60.0),
            **spread("fs1", [v * rnd(0.6, 0.95) for v in fs2]),
            **spread("fs2", fs2),
            **spread("fs3", [v * rnd(1.05, 1.6) for v in fs2]),
            **spread("hs", [rnd(90.0, 145.0) for _ in range(5)]),
            **spread("fi1", [rnd(0.5, 12.0) for _ in range(5)]),
            **spread("gyz", [rnd(0.7, 1.8) for _ in range(5)]),
        ))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "esmod_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"esmod: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
