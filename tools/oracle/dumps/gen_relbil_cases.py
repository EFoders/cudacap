#!/usr/bin/env python3
"""Deterministic case grid for the RELBIL dump.

RELBIL reads a list of up to twenty candidate modes out of /allMODE/, so the
cases are mode lists.  What has to be covered:

* **NMMOD 0** (immediate return), **1** (the single-mode branch, which skips
  the power sum entirely) and 2..20 (the combined branch).
* **HP <= 70**, the "not really a mode" test, including a case where EVERY
  mode fails it and one where only the first does.  With one mode that also
  reaches the stale-read path: D10R/D50R/D90R are never assigned, and
  SNRLW/SNRUP/SNPR come from whatever the previous frequency left -- which
  is why the case file injects those three as well as dumping them.
* **The two-pass most-reliable-mode selection**: reliabilities far apart (the
  max wins outright), within the 0.05 window with differing hop counts (fewer
  hops wins), within the window with equal hop counts (better S/N wins), and
  the pathological ordering where a naive single pass would walk away from
  the true maximum.
* **Z <= 0 vs Z > 0**, which switches the reliability between the lower and
  upper spread -- driven by putting the required S/N either side of the
  modes' median.
* **The ZEXP > -10 floor**, reached by making one mode 150 dB weaker than
  the strongest so its term is dropped from the power sums.
* **The 0.2 and 30 dB clamps** on both combined deciles, and the 30 dB cap
  on D90R.
* **LUFP either side of 50**, which selects which spread the required-power
  calculation uses, and past 95 where ITM saturates at 10.
* **NMODE 1..5**, so every LAYTYP label is produced.

stdlib-only.  Usage: gen_relbil_cases.py <outdir>
"""

import pathlib
import sys

SEED = 20260725
NM = 20                         # /allMODE/ slots

SCALARS = ["ifx", "freq", "rsn", "lufp", "du", "dl", "rcnse", "nmmod",
           "pwrdba", "txlo", "txhi", "d10r", "d50r", "d90r"]
ARRAYS = ["hp", "sn", "tllow", "tlhgh", "hn", "sigpow", "fldst", "tgain",
          "rgain", "b", "timed", "tloss", "prob", "eff", "nmode"]
COLUMNS = SCALARS + [f"{a}{i}" for a in ARRAYS for i in range(1, NM + 1)]

INTS = {"ifx", "lufp", "nmmod"}

DEFAULTS = {
    "ifx": 3, "freq": 14.0, "rsn": 24.0, "lufp": 90, "du": 9.7, "dl": 6.0,
    "rcnse": -145.0, "nmmod": 3, "pwrdba": 40.0, "txlo": 2.0, "txhi": 30.0,
    "d10r": -1.0, "d50r": -2.0, "d90r": -3.0,
}
# a plausible mode list: three F2 modes of increasing hop count
_BASE = {
    "hp": [300.0, 310.0, 320.0], "sn": [22.0, 19.0, 15.0],
    "tllow": [9.0, 10.0, 11.0], "tlhgh": [8.0, 9.0, 10.0],
    "hn": [1.0, 2.0, 3.0], "sigpow": [-120.0, -124.0, -130.0],
    "fldst": [30.0, 26.0, 20.0], "tgain": [8.0, 7.5, 6.0],
    "rgain": [5.0, 4.5, 3.0], "b": [12.0, 20.0, 28.0],
    "timed": [11.0, 12.0, 13.0], "tloss": [160.0, 164.0, 170.0],
    "prob": [0.9, 0.8, 0.6], "eff": [-1.7, -1.7, -1.7],
    "nmode": [3.0, 3.0, 3.0],
}
for _a in ARRAYS:
    _vals = _BASE[_a] + [0.0] * NM
    for _i in range(1, NM + 1):
        DEFAULTS[f"{_a}{_i}"] = _vals[_i - 1]


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def fmt(col, val) -> str:
    if col in INTS or col.startswith("nmode"):
        return str(int(val))
    return f(val)


def case(**kw) -> str:
    d = dict(DEFAULTS)
    for key, val in kw.items():
        if key in ARRAYS:                     # a whole per-mode array
            for i in range(1, NM + 1):
                d[f"{key}{i}"] = val[i - 1] if i <= len(val) else 0.0
        else:
            d[key] = val
    return ",".join(fmt(c, d[c]) for c in COLUMNS)


def modes(n, **kw):
    """An n-mode list; each array not named defaults to a spread."""
    out = {"nmmod": n}
    for a in ARRAYS:
        if a in kw:
            out[a] = kw[a]
            continue
        base = _BASE[a]
        step = (base[1] - base[0]) if len(base) > 1 else 1.0
        out[a] = [base[0] + step * i for i in range(n)]
    return out


def cases():
    rows = []
    # --- mode count: none, one, and up to the full twenty ------------------
    rows.append(case(nmmod=0))
    rows.append(case(nmmod=-1))
    for n in (1, 2, 3, 4, 6, 10, 15, 19, 20):
        rows.append(case(**modes(n)))
    # --- HP <= 70: the "not really a mode" test ----------------------------
    rows.append(case(**modes(1, hp=[50.0])))          # stale D10R/D50R/D90R
    rows.append(case(**modes(1, hp=[70.0])))          # the boundary is <=
    rows.append(case(**modes(1, hp=[70.01])))
    rows.append(case(**modes(3, hp=[50.0, 60.0, 65.0])))
    rows.append(case(**modes(3, hp=[50.0, 300.0, 310.0])))
    rows.append(case(**modes(4, hp=[300.0, 40.0, 310.0, 60.0])))
    # seeded stale values, so the stale read is visibly the seed
    for d10, d50, d90 in ((0.5, 12.0, 4.0), (25.0, -3.0, 29.0),
                          (0.0, 0.0, 0.0)):
        rows.append(case(d10r=d10, d50r=d50, d90r=d90, **modes(1, hp=[50.0])))
    # --- most-reliable-mode selection --------------------------------------
    # far apart: the outright maximum wins
    rows.append(case(**modes(4, sn=[5.0, 30.0, 6.0, 7.0])))
    # within 0.05 with differing hop counts: fewer hops wins
    rows.append(case(**modes(4, sn=[24.0, 24.05, 24.02, 24.01],
                             hn=[4.0, 3.0, 2.0, 1.0])))
    rows.append(case(**modes(4, sn=[24.0, 24.05, 24.02, 24.01],
                             hn=[1.0, 2.0, 3.0, 4.0])))
    # within 0.05 with equal hop counts: better median S/N wins
    rows.append(case(**modes(4, sn=[24.0, 24.4, 24.2, 24.1],
                             hn=[2.0, 2.0, 2.0, 2.0])))
    # the ordering a naive single pass mishandles: the max is last and a
    # near-tie with fewer hops sits before it
    rows.append(case(**modes(5, sn=[10.0, 24.3, 11.0, 12.0, 24.4],
                             hn=[3.0, 1.0, 3.0, 3.0, 2.0])))
    # every reliability identical
    rows.append(case(**modes(5, sn=[20.0] * 5, hn=[2.0] * 5)))
    # reliabilities all essentially zero, so XREL hits its 1e-8 floor
    rows.append(case(rsn=200.0, **modes(5)))
    # --- Z either side of zero ---------------------------------------------
    for rsn in (-40.0, 0.0, 15.0, 19.0, 22.0, 30.0, 60.0, 120.0):
        rows.append(case(rsn=rsn, **modes(3)))
        rows.append(case(rsn=rsn, **modes(1)))
    # --- the ZEXP > -10 floor in the power sums ----------------------------
    rows.append(case(**modes(3, sigpow=[-100.0, -260.0, -110.0],
                             fldst=[40.0, -120.0, 30.0])))
    rows.append(case(**modes(4, sigpow=[-100.0, -101.0, -260.0, -300.0])))
    # every term dropped but the reference: the sums stay at one
    rows.append(case(**modes(3, sigpow=[-100.0, -400.0, -400.0],
                             fldst=[40.0, -400.0, -400.0])))
    # --- the 0.2 / 30 dB clamps on the combined deciles --------------------
    for tl in (0.0, 0.05, 0.2, 5.0, 25.0, 40.0):
        rows.append(case(**modes(3, tllow=[tl] * 3, tlhgh=[tl] * 3)))
    rows.append(case(du=0.0, dl=0.0,
                     **modes(3, tllow=[0.0] * 3, tlhgh=[0.0] * 3)))
    rows.append(case(du=40.0, dl=40.0, **modes(3)))
    # --- LUFP: ITM either side of 50 and past the saturation point ---------
    for lufp in (0, 1, 10, 45, 49, 50, 51, 55, 90, 95, 99, 100):
        rows.append(case(lufp=lufp, **modes(3)))
        rows.append(case(lufp=lufp, **modes(1)))
    # --- every LAYTYP label ------------------------------------------------
    for layer in (1, 2, 3, 4, 5):
        rows.append(case(**modes(2, nmode=[layer, layer])))
    # --- PWRDB's 30 dBW default, for the recomputed transmission loss ------
    rows.append(case(freq=1.0, **modes(3)))
    rows.append(case(txlo=20.0, txhi=25.0, **modes(3)))
    # --- the frequency column index ----------------------------------------
    for ifx in (1, 2, 7, 12, 13):
        rows.append(case(ifx=ifx, **modes(3)))
    # --- random fill for breadth -------------------------------------------
    rnd = lcg(SEED)
    for _ in range(300):
        n = 1 + int(rnd(0.0, 20.0)) % 20
        sig = [rnd(-190.0, -80.0) for _ in range(n)]
        rows.append(case(
            ifx=1 + int(rnd(0.0, 13.0)) % 13,
            freq=rnd(2.0, 30.0),
            rsn=rnd(-20.0, 80.0),
            lufp=int(rnd(0.0, 100.0)),
            du=rnd(0.0, 20.0), dl=rnd(0.0, 20.0), rcnse=rnd(-190.0, -90.0),
            pwrdba=rnd(10.0, 60.0),
            d10r=rnd(0.0, 30.0), d50r=rnd(-20.0, 40.0), d90r=rnd(0.0, 40.0),
            nmmod=n,
            hp=[rnd(40.0, 450.0) for _ in range(n)],
            sn=[rnd(-30.0, 50.0) for _ in range(n)],
            tllow=[rnd(0.0, 26.0) for _ in range(n)],
            tlhgh=[rnd(0.0, 26.0) for _ in range(n)],
            hn=[float(1 + int(rnd(0.0, 6.0)) % 6) for _ in range(n)],
            sigpow=sig,
            fldst=[s + rnd(140.0, 160.0) for s in sig],
            tgain=[rnd(-10.0, 20.0) for _ in range(n)],
            rgain=[rnd(-10.0, 20.0) for _ in range(n)],
            b=[rnd(1.0, 60.0) for _ in range(n)],
            timed=[rnd(3.0, 40.0) for _ in range(n)],
            tloss=[rnd(90.0, 240.0) for _ in range(n)],
            prob=[rnd(0.0, 1.0) for _ in range(n)],
            eff=[rnd(-6.0, 0.0) for _ in range(n)],
            nmode=[float(1 + int(rnd(0.0, 5.0)) % 5) for _ in range(n)],
        ))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "relbil_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"relbil: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
