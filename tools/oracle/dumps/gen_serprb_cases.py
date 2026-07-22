#!/usr/bin/env python3
"""Deterministic case grid for the SERPRB + MPATH dump.

Both read /allMODE/ and write one /SON/ column, so the cases are mode lists
again.  What has to be reached:

SERPRB
* **LUFP either side of 50**, which swaps the whole set of spreads: above,
  the noise UPPER decile with the signal LOWER decile and XLH = -1; below,
  the mirror image with XLH = +1.  Also LUFP past 95, where ITM saturates.
* **HP <= 70**, the no-mode default, including every mode dead -- which
  leaves D10S and D50S at zero because their locals were never assigned.
* the prediction-error terms ADS/SUS/SLS and SIGM/SYGU/SYGL at zero and at
  realistic magnitudes, since the flat 2 dB DR keeps D10SA non-zero even
  when every one of them vanishes.
* **which mode wins the maximum**: first, last and middle.

MPATH
* **DMP or PMP non-positive** -- no multipath requested, early return.
* **METHOD 20 + MSPEC 121 beyond 7000 km**, the area-coverage exclusion.
* the delay test either side of DMP, including exactly on it (where the
  mode is SKIPPED, since the comparison is <=).
* the power test either side of the MRM's power less PMP.
* **HP <= 0**, a slot that does not hold a mode at all -- note this is a
  different threshold from SERPRB's 70 km, so a mode can be live for one
  routine and dead for the other.
* no qualifying mode at all, leaving PROBMP at its 0.001 default.

NMMOD is never zero: with no modes SERPRB reads uninitialised locals, and
the port raises there rather than reproducing it.

stdlib-only.  Usage: gen_serprb_cases.py <outdir>
"""

import pathlib
import sys

SEED = 20260726
NM = 20

SCALARS = ["ifx", "rsn", "lufp", "du", "dl", "sigm", "sygu", "sygl",
           "ads", "sus", "sls", "nmmod", "nrel", "dmp", "pmp", "gcdkm",
           "method", "mspec"]
ARRAYS = ["hp", "sn", "tllow", "tlhgh", "sigpow", "timed", "rely"]
COLUMNS = SCALARS + [f"{a}{i}" for a in ARRAYS for i in range(1, NM + 1)]

INTS = {"ifx", "lufp", "nmmod", "nrel", "method", "mspec"}

DEFAULTS = {
    "ifx": 4, "rsn": 24.0, "lufp": 90, "du": 9.7, "dl": 6.0, "sigm": 5.4,
    "sygu": 1.5, "sygl": 1.5, "ads": 5.6, "sus": 0.6, "sls": 1.63,
    "nmmod": 3, "nrel": 1, "dmp": 0.85, "pmp": 10.0, "gcdkm": 3000.0,
    "method": 30, "mspec": 0,
}
_BASE = {
    "hp": [300.0, 310.0, 320.0], "sn": [22.0, 19.0, 15.0],
    "tllow": [9.0, 10.0, 11.0], "tlhgh": [8.0, 9.0, 10.0],
    "sigpow": [-120.0, -124.0, -130.0], "timed": [11.0, 12.0, 13.0],
    "rely": [0.9, 0.7, 0.4],
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


def case(**kw) -> str:
    d = dict(DEFAULTS)
    for key, val in kw.items():
        if key in ARRAYS:
            for i in range(1, NM + 1):
                d[f"{key}{i}"] = val[i - 1] if i <= len(val) else 0.0
        else:
            d[key] = val
    return ",".join(str(int(d[c])) if c in INTS else f(d[c])
                    for c in COLUMNS)


def modes(n, **kw):
    out = {"nmmod": n}
    for a in ARRAYS:
        if a in kw:
            out[a] = kw[a]
            continue
        base = _BASE[a]
        step = base[1] - base[0]
        out[a] = [base[0] + step * i for i in range(n)]
    return out


def cases():
    rows = []
    # --- SERPRB: LUFP either side of 50, and the ITM saturation ------------
    for lufp in (0, 1, 10, 45, 49, 50, 51, 55, 80, 90, 95, 99, 100):
        rows.append(case(lufp=lufp, **modes(3)))
        rows.append(case(lufp=lufp, **modes(1)))
    # --- SERPRB: dead modes ------------------------------------------------
    rows.append(case(**modes(1, hp=[50.0])))
    rows.append(case(**modes(1, hp=[70.0])))          # boundary is inclusive
    rows.append(case(**modes(1, hp=[70.01])))
    rows.append(case(**modes(3, hp=[50.0, 60.0, 65.0])))   # every mode dead
    rows.append(case(**modes(4, hp=[300.0, 40.0, 310.0, 60.0])))
    # --- SERPRB: which mode takes the maximum ------------------------------
    for pos in range(4):
        sn = [0.0, 0.0, 0.0, 0.0]
        sn[pos] = 60.0
        rows.append(case(nrel=pos + 1, **modes(4, sn=sn)))
    # --- SERPRB: the prediction-error terms --------------------------------
    rows.append(case(sigm=0.0, sygu=0.0, sygl=0.0, ads=0.0, sus=0.0,
                     sls=0.0, **modes(3)))
    rows.append(case(sigm=0.0, sygu=0.0, sygl=0.0, ads=0.0, sus=0.0,
                     sls=0.0, du=0.0, dl=0.0,
                     **modes(3, tllow=[0.0] * 3, tlhgh=[0.0] * 3)))
    for scale in (0.5, 2.0, 6.0):
        rows.append(case(sigm=5.4 * scale, sygu=1.5 * scale,
                         sygl=1.5 * scale, ads=5.6 * scale,
                         sus=0.6 * scale, sls=1.63 * scale, **modes(3)))
    # --- SERPRB: RSN either side of the median -----------------------------
    for rsn in (-30.0, 0.0, 15.0, 22.0, 30.0, 90.0):
        rows.append(case(rsn=rsn, **modes(3)))
    # --- MPATH: not requested ----------------------------------------------
    for dmp, pmp in ((0.0, 10.0), (0.85, 0.0), (0.0, 0.0), (-1.0, 10.0)):
        rows.append(case(dmp=dmp, pmp=pmp, **modes(4)))
    # --- MPATH: the area-coverage exclusion --------------------------------
    for method, mspec, gcdkm in ((20, 121, 7001.0), (20, 121, 7000.0),
                                 (20, 121, 6000.0), (20, 0, 9000.0),
                                 (30, 121, 9000.0), (20, 121, 12000.0)):
        rows.append(case(method=method, mspec=mspec, gcdkm=gcdkm,
                         **modes(4, timed=[11.0, 14.0, 17.0, 20.0])))
    # --- MPATH: the delay test, either side of DMP and exactly on it -------
    for dt in (0.0, 0.5, 0.849, 0.85, 0.851, 2.0, 9.0):
        rows.append(case(**modes(4, timed=[11.0, 11.0 + dt, 11.0 + 2 * dt,
                                           11.0 + 3 * dt])))
    # --- MPATH: the power test, either side of MRM - PMP -------------------
    for dp in (0.0, 5.0, 9.9, 10.0, 10.1, 25.0):
        rows.append(case(**modes(4, timed=[11.0, 15.0, 19.0, 23.0],
                                 sigpow=[-120.0, -120.0 - dp,
                                         -120.0 - dp, -120.0 - dp])))
    # --- MPATH: HP <= 0 (a different threshold from SERPRB's 70) -----------
    rows.append(case(**modes(4, hp=[300.0, 0.0, -1.0, 310.0],
                             timed=[11.0, 15.0, 19.0, 23.0])))
    rows.append(case(**modes(4, hp=[300.0, 40.0, 60.0, 310.0],
                             timed=[11.0, 15.0, 19.0, 23.0])))
    # --- MPATH: NREL somewhere other than the first ------------------------
    for nrel in (1, 2, 3, 4):
        rows.append(case(nrel=nrel, **modes(4,
                                            timed=[11.0, 15.0, 19.0, 23.0])))
    # --- random fill for breadth -------------------------------------------
    rnd = lcg(SEED)
    for _ in range(300):
        n = 1 + int(rnd(0.0, 20.0)) % 20
        rows.append(case(
            ifx=1 + int(rnd(0.0, 13.0)) % 13,
            rsn=rnd(-20.0, 80.0),
            lufp=int(rnd(0.0, 100.0)),
            du=rnd(0.0, 20.0), dl=rnd(0.0, 20.0),
            sigm=rnd(0.0, 9.0), sygu=rnd(0.0, 7.0), sygl=rnd(0.0, 7.0),
            ads=rnd(0.0, 9.0), sus=rnd(0.0, 3.0), sls=rnd(0.0, 3.0),
            nmmod=n, nrel=1 + int(rnd(0.0, float(n))) % n,
            dmp=rnd(-0.5, 3.0), pmp=rnd(-2.0, 30.0),
            gcdkm=rnd(200.0, 14000.0),
            method=(20 if rnd(0.0, 1.0) < 0.5 else 30),
            mspec=(121 if rnd(0.0, 1.0) < 0.4 else 0),
            hp=[rnd(-10.0, 450.0) for _ in range(n)],
            sn=[rnd(-30.0, 50.0) for _ in range(n)],
            tllow=[rnd(0.0, 26.0) for _ in range(n)],
            tlhgh=[rnd(0.0, 26.0) for _ in range(n)],
            sigpow=[rnd(-190.0, -80.0) for _ in range(n)],
            timed=[rnd(3.0, 40.0) for _ in range(n)],
            rely=[rnd(0.0, 1.0) for _ in range(n)],
        ))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "serprb_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"serprb: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
