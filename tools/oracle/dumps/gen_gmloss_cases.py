#!/usr/bin/env python3
"""Deterministic case grid for the GMLOSS + SELTXR dump.

LUFFY calls GMLOSS (which calls SETTXR) and then SELTXR, and this drives the
three in that order.  The inputs are SETTXR's grid -- imported from
gen_settxr_cases so the two stay in step -- plus the four SELTXR needs:
DELOPT, GMIN, YMIN and the /INFORM/ mask.

What has to be reached, on top of everything SETTXR's own grid already covers:

* **the give-up path at label 95**, where SELTXR walks off the end of the
  filled rays or hits a dead one and DECREMENTS its answer.
* **the find-first search running and not running.**  It only starts when
  ray 1's hop count is under 0.9, and it advances the index without
  recomputing that hop count -- so the case grid needs ray 1 both short and
  long enough.
* **all three tie-breaks**: the gain-minus-loss window GMIN, the specular
  window YMIN, and the take-off-angle comparison, each taken and not taken.
  GMIN and YMIN are swept from zero (nothing is indistinguishable, so only
  the first test can decide) up to values that make everything tie.
* **the partial-hop fold at label 116/130**, either side of its threshold.
* **INFO bit 4 set and clear.**  Set, the whole selection runs twice, and the
  second pass inherits XHPM and YHPM from the first -- so this is not a
  no-op and the dump has to carry both.
* **GMLOSS's two tests either side of their thresholds and exactly on them**:
  HPFLX at 70 and DELFX at AMIND both LIVE here, which is the opposite of
  SELTXR's answer for the same ray.

Held away from the grid, on top of SETTXR's own exclusions:

* IAFTXR at exactly 45 with every ray existing and none clearing AMIND drives
  SELTXR's find-first search to ray 46, which the Fortran reads out of the
  column.  The port raises; NFILL is capped at 44 in the random fill and the
  explicit 45-ray cases keep a take-off-angle ramp that breaks the search
  early.

stdlib-only.  Usage: gen_gmloss_cases.py <outdir>
"""

import pathlib
import sys

import gen_settxr_cases as S

SEED = 20260731

INTS = S.INTS + ["info"]
FLOATS = S.FLOATS + ["delopt", "gmin", "ymin"]
COLUMNS = INTS + FLOATS

DEFAULTS = dict(S.DEFAULTS)
DEFAULTS.update({"info": 0, "delopt": 3.0, "gmin": 3.0, "ymin": 0.1})


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
    # --- INFO bit 4: the single and the double selection pass ------------
    for info in (0, 4, 5, 12, 3):
        rows.append(case(info=info))
        rows.append(case(info=info, gdflx0_1=4800.0, gdflx0_2=4800.0))
    # --- the give-up path: no ray past the first clears the tests --------
    for h0, dh, nf in ((100.0, 0.0, 1), (100.0, 0.0, 2), (100.0, -20.0, 6),
                       (100.0, 8.0, 3), (60.0, 8.0, 12), (0.0, 0.0, 12)):
        rows.append(case(hpflx0_1=h0, dhpflx_1=dh, nfill1=nf,
                         hpflx0_2=h0, dhpflx_2=dh, nfill2=nf,
                         gdflx0_1=4800.0, gdflx0_2=4800.0))
    # --- ray 1's hop count either side of 0.9 ----------------------------
    # XHPM = min(GCDKM,4000)/GDFLX(1,K); 4000/4444 = 0.9
    for gd in (1000.0, 4000.0, 4444.0, 4444.5, 4445.0, 4800.0, 5000.0):
        rows.append(case(gdflx0_1=gd, gdflx0_2=gd, dgdflx_1=0.0,
                         dgdflx_2=0.0))
        rows.append(case(gdflx0_1=gd, gdflx0_2=gd, dgdflx_1=-40.0,
                         dgdflx_2=-40.0))
    # --- the three tie-breaks, from "nothing ties" to "everything ties" --
    for gmin in (0.0, 0.5, 3.0, 20.0, 500.0):
        for ymin in (0.0, 0.1, 0.5, 2.0):
            rows.append(case(gmin=gmin, ymin=ymin))
    # --- DELOPT across the take-off-angle range --------------------------
    for delopt in (0.0, 3.0, 8.0, 20.0, 45.0, 89.0):
        rows.append(case(delopt=delopt))
        rows.append(case(delopt=delopt, gmin=500.0, ymin=2.0))
    # --- the partial-hop fold, either side of 0.5 and of 0.05 ------------
    for gd in (2000.0, 2666.0, 2667.0, 2700.0, 3000.0, 3200.0, 3810.0):
        rows.append(case(gdflx0_1=gd, gdflx0_2=gd, dgdflx_1=0.0,
                         dgdflx_2=0.0, gmin=500.0))
        rows.append(case(gdflx0_1=gd, gdflx0_2=gd, dgdflx_1=0.0,
                         dgdflx_2=0.0, gmin=500.0, info=4))
    # --- GMLOSS's two tests, exactly on their thresholds -----------------
    for h0 in (69.9, 70.0, 70.1):
        rows.append(case(hpflx0_1=h0, hpflx0_2=h0, dhpflx_1=0.0,
                         dhpflx_2=0.0))
    for amind in (0.0, 2.0, 3.8, 3.7999997, 20.0):
        rows.append(case(amind=amind))
    # --- inherit SETTXR's own structural grid ----------------------------
    # (its rows are a prefix of these columns; re-emit them with the four
    # SELTXR inputs at their blkdat defaults, plus a diagnostic variant)
    for row in S.cases():
        vals = dict(zip(S.COLUMNS, row.split(",")))
        kw = {c: (int(vals[c]) if c in S.INTS else float(vals[c]))
              for c in S.COLUMNS}
        if kw["nfill1"] > 44:
            kw["nfill1"] = 44
        if kw["nfill2"] > 44:
            kw["nfill2"] = 44
        rows.append(case(**kw))
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "gmloss_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"gmloss: {len(rows)} cases, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
