#!/usr/bin/env python3
"""Deterministic case grid for the INMUF dump.

Extends the REGMOD case format -- INMUF calls REGMOD, so every REGMOD input is
still needed -- with the /MUFS/ fields REGMOD does not read (DELMUF, AFMUF,
NHOPMF, MODMUF), the hop count IHOP, and the profile parameters that let the
driver build a real ionogram with LECDEN + GENION.

The ionogram itself is NOT generated here.  XLIN interpolates FVERT/HPRIM, so
by the FINDF/IFOB rule the table has to come from the oracle: the driver runs
the real chain and dumps FVERT/HPRIM/HTRUE/AFAC alongside the outputs, and the
Python test injects those columns verbatim.

Branches to cover, beyond REGMOD's:

* **Zero distance**, entered when DELMUF(MODMUF) >= 89.9, and its three
  sub-cases: an exact hit in the FVERT table, an interpolated hit, and a
  frequency above every table value (which writes no mode at all).
* **The over-the-MUF insert**, which needs a layer absent from /MODES/, the
  frequency at or above that layer's MUF, and IHOP matching NHOPMF.
* **The per-hop MUF recompute**, reached only when IHOP differs from
  NHOPMF(MODMUF), and which then has to clear its own FREQ >= ESD test.
* **The five-layer cutoff** that stops both insert loops.
* **The mode compaction**, including a virtual height below the 70 km floor.

stdlib-only.  Usage: gen_inmuf_cases.py <outdir>
"""

import pathlib
import sys

from gen_regmod_cases import COLUMNS as REGMOD_COLUMNS
from gen_regmod_cases import DEFAULTS as REGMOD_DEFAULTS
from gen_regmod_cases import f, lcg

SEED = 20260723

EXTRA = (["ihop", "modmuf"]
         + [f"delmuf{i}" for i in range(1, 5)]
         + [f"afmuf{i}" for i in range(1, 5)]
         + [f"nhopmf{i}" for i in range(1, 5)]
         + ["fi2", "fi3", "yi1", "yi2", "yi3", "hi1", "hi2", "hi3", "iedp"])

COLUMNS = list(REGMOD_COLUMNS) + EXTRA

DEFAULTS = dict(REGMOD_DEFAULTS)
DEFAULTS.update({
    "ihop": 2, "modmuf": 3,
    "fi2": 4.0, "fi3": 9.0, "yi1": 20.0, "yi2": 40.0, "yi3": 90.0,
    "hi1": 110.0, "hi2": 200.0, "hi3": 320.0, "iedp": -1,
})
for _i in range(1, 5):
    DEFAULTS[f"delmuf{_i}"] = 12.0
    DEFAULTS[f"afmuf{_i}"] = 0.5
    DEFAULTS[f"nhopmf{_i}"] = 2


def case(**kw) -> str:
    d = dict(DEFAULTS)
    d.update(kw)
    return ",".join(f(d[c]) for c in COLUMNS)


# a mode list with nothing in it, so the insert paths have room to work
_EMPTY = {f"hpmod{i}": -1.0 for i in range(1, 7)}
# ... and one that fills every slot, so the >= 5 cutoffs fire
_FULL = {f"hpmod{i}": 250.0 + 20.0 * i for i in range(1, 7)}


def cases():
    rows = []
    # --- zero distance: DELMUF(MODMUF) >= 89.9 ------------------------------
    for freq in (2.5, 5.0, 9.0, 13.0, 28.0):
        rows.append(case(freq=freq, **_EMPTY,
                         **{f"delmuf{i}": 89.9 for i in range(1, 5)}))
    # just either side of the 89.9 threshold
    for dm in (89.89, 89.9, 89.91, 90.0):
        rows.append(case(**_EMPTY, **{f"delmuf{i}": dm for i in range(1, 5)}))
    # zero distance reached the other way: hop matches but no mode survives
    rows.append(case(ihop=2, modmuf=3, **_EMPTY,
                     **{f"nhopmf{i}": 2 for i in range(1, 5)}))
    # --- over-the-MUF insert: frequency at/above the layer MUF, hop matches -
    for freq in (5.0, 13.6, 14.0, 14.4, 20.0):
        rows.append(case(freq=freq, ihop=2, **_EMPTY,
                         **{f"nhopmf{i}": 2 for i in range(1, 5)}))
    # the EPS = .4001 slack around the MUF comparison
    for freq in (13.5999, 13.6, 13.6001):
        rows.append(case(freq=freq, ihop=2, **_EMPTY,
                         **{f"ymuf{i}": 14.0 for i in range(1, 5)},
                         **{f"nhopmf{i}": 2 for i in range(1, 5)}))
    # no F1 layer -> the ILY == 2 arm must be skipped
    rows.append(case(fi2=-1.0, ihop=2, **_EMPTY,
                     **{f"nhopmf{i}": 2 for i in range(1, 5)}))
    rows.append(case(fi2=0.0, ihop=2, **_EMPTY,
                     **{f"nhopmf{i}": 2 for i in range(1, 5)}))
    # --- per-hop MUF recompute: IHOP != NHOPMF(MODMUF) and IHOP > NHOPMF(JH)
    for ihop in (2, 3, 4, 6):
        rows.append(case(ihop=ihop, modmuf=3, **_EMPTY,
                         **{f"nhopmf{i}": 1 for i in range(1, 5)}))
    for freq in (4.0, 8.0, 12.0, 18.0, 26.0):
        rows.append(case(freq=freq, ihop=4, modmuf=3, **_EMPTY,
                         **{f"nhopmf{i}": 1 for i in range(1, 5)}))
    # --- the >= 5 cutoff in both insert loops -------------------------------
    rows.append(case(ihop=2, **_FULL, **{f"nhopmf{i}": 2 for i in range(1, 5)}))
    rows.append(case(ihop=4, modmuf=3, **_FULL,
                     **{f"nhopmf{i}": 1 for i in range(1, 5)}))
    # --- compaction: virtual heights straddling the 70 km floor -------------
    rows.append(case(hpmod1=69.9, hpmod2=70.0, hpmod3=70.1,
                     hpmod4=-1.0, hpmod5=300.0, hpmod6=-1.0))
    rows.append(case(**{f"hpmod{i}": 50.0 for i in range(1, 7)}))
    # --- MODMUF picks a different layer -------------------------------------
    for modmuf in (1, 2, 3):
        rows.append(case(modmuf=modmuf, ihop=2, **_EMPTY))
    # --- random fill ---------------------------------------------------------
    rnd = lcg(SEED)
    for _ in range(200):
        nhop = 1 + int(rnd(0.0, 5.0)) % 5
        gcdkm = rnd(500.0, 3000.0) * nhop
        fi1 = rnd(1.0, 5.0)
        fi2 = fi1 + rnd(-1.0, 2.0)
        fi3 = max(fi2, fi1) + rnd(0.5, 6.0)
        kw = {
            "jmode": 1 + int(rnd(0.0, 3.0)) % 3,
            "freq": rnd(2.0, 30.0),
            "gcdkm": gcdkm,
            "ghopkm": gcdkm / nhop,
            "km": 1 + int(rnd(0.0, 5.0)) % 5,
            "ihop": nhop,
            "modmuf": 1 + int(rnd(0.0, 3.0)) % 3,
            "fi1": fi1, "fi2": fi2, "fi3": fi3,
            "yi1": rnd(10.0, 30.0), "yi2": rnd(25.0, 55.0),
            "yi3": rnd(60.0, 120.0),
            "hi1": rnd(100.0, 120.0), "hi2": rnd(170.0, 220.0),
            "hi3": rnd(250.0, 380.0),
            "iedp": -1 if rnd(0.0, 1.0) < 0.5 else 0,
            "gyz": rnd(0.5, 1.6),
            "fs1": rnd(1.0, 8.0), "fs2": rnd(-1.0, 12.0),
            "hs": rnd(95.0, 115.0),
            "ifob13": int(rnd(300.0, 8000.0)),
            "acav": rnd(0.01, 0.6), "htloss": rnd(60.0, 130.0),
            "xnuz": rnd(5.0, 20.0), "hnu": rnd(4.0, 14.0),
            "xve": rnd(0.5, 1.5), "afe": rnd(-1.0, 2.0),
            "bfe": rnd(0.2, 3.0), "asm": rnd(0.0, 8.0),
            "dsl": rnd(4.0, 14.0), "dsu": rnd(4.0, 14.0),
            "rcnse": rnd(-165.0, -120.0), "pwrdba": rnd(20.0, 60.0),
        }
        for i in range(1, 5):
            ymuf = rnd(6.0, 26.0)
            kw[f"sigl{i}"] = rnd(0.08, 0.30)
            kw[f"sigu{i}"] = rnd(0.08, 0.30)
            kw[f"ymuf{i}"] = ymuf
            kw[f"yfot{i}"] = ymuf * rnd(0.70, 0.95)
            kw[f"yhpf{i}"] = ymuf * rnd(1.05, 1.35)
            kw[f"hpmuf{i}"] = rnd(180.0, 420.0)
            kw[f"htmuf{i}"] = rnd(100.0, 360.0)
            kw[f"fvmuf{i}"] = rnd(2.0, 16.0)
            kw[f"afmuf{i}"] = rnd(0.0, 2.0)
            # a quarter of the cases sit at the 89.9 zero-distance threshold
            kw[f"delmuf{i}"] = (rnd(88.0, 91.0) if rnd(0.0, 1.0) < 0.25
                                else rnd(2.0, 40.0))
            kw[f"nhopmf{i}"] = 1 + int(rnd(0.0, 5.0)) % 5
        for i in range(1, 7):
            unused = rnd(0.0, 1.0) < 0.5
            kw[f"hpmod{i}"] = -1.0 if unused else rnd(60.0, 450.0)
            kw[f"delmod{i}"] = rnd(0.0, 45.0)
            kw[f"htmod{i}"] = rnd(90.0, 400.0)
            kw[f"fvmod{i}"] = fi1 * rnd(0.4, 2.5)
            kw[f"itmod{i}"] = 1 + int(rnd(0.0, 3.0)) % 3
            kw[f"afmod{i}"] = rnd(0.0, 2.0)
        for i in range(1, 6):
            kw[f"sigpat{i}"] = rnd(0.001, 5.0)
            kw[f"epspat{i}"] = rnd(3.0, 80.0)
        rows.append(case(**kw))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "inmuf_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"inmuf: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
