#!/usr/bin/env python3
"""Deterministic call sequence for the ALLMODES dump.

ALLMODES accumulates, so a case is not independent: /allMODE/ carries over and
NMMOD is both the running count and the write cursor.  The cases are therefore
a *sequence* of calls, and the driver deliberately does not clear the common
block between them.  Each block opens with a reset (IFLG=0) and then appends.

What has to be covered:

* each IFLG arm -- 0 reset, 1 (slot 1 plus the reserved NMMOD+2 slot), 2 (the
  sporadic-E pair at slots 4-5), and the default (slots 1-6);
* the HP > 0 test that decides whether a /ZON/ slot is live, including a call
  where nothing is live and NMMOD must not move;
* accumulation up to slot 20 without passing it, since the Fortran bound-checks
  nothing and would write past the array.

The generator tracks NMMOD exactly as ALLMODES does and forces a reset before a
block could overflow, keeping the oracle inside defined behaviour.

stdlib-only.  Usage: gen_allmodes_cases.py <outdir>
"""

import pathlib
import sys

SEED = 20260724
NSLOT = 20

# /ZON/ fields ALLMODES reads, in the order the driver reads them
ZON_FIELDS = ("abps", "crel", "fldst", "hn", "hp", "prob", "rely", "rgain",
              "sigpow", "sn", "spro", "tgain", "timed", "tloss", "b", "fslos",
              "grlos", "adv", "obf", "tllow", "tlhgh", "eff", "nmode")

COLUMNS = ["iflg", "fval"] + [f"{n}{i}" for n in ZON_FIELDS
                              for i in range(1, 8)]


def lcg(seed):
    state = seed

    def rnd(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        return lo + (state >> 11) / 2**53 * (hi - lo)

    return rnd


def f(v) -> str:
    return f"{float(v):.9g}"


def row(iflg, fval, zon) -> str:
    vals = [float(iflg), float(fval)]
    for n in ZON_FIELDS:
        vals += [float(x) for x in zon[n]]
    return ",".join(f(v) for v in vals)


def make_zon(rnd, live):
    """A /ZON/ block whose slots in `live` have a positive virtual height."""
    zon = {}
    for n in ZON_FIELDS:
        zon[n] = [rnd(-50.0, 250.0) for _ in range(7)]
    zon["nmode"] = [float(1 + int(rnd(0.0, 4.0)) % 4) for _ in range(7)]
    zon["hp"] = [(rnd(80.0, 450.0) if (i + 1) in live else -1.0)
                 for i in range(7)]
    return zon


def _appended(iflg, live) -> int:
    """How many slots this call will append, mirroring ALLMODES."""
    ist, lst = (1, 1) if iflg == 1 else ((4, 5) if iflg == 2 else (1, 6))
    return sum(1 for i in range(ist, lst + 1) if i in live)


def cases():
    rnd = lcg(SEED)
    rows = []
    nmmod = 0

    def emit(iflg, live, fval=0.0):
        nonlocal nmmod
        # IFLG=1 also writes slot NMMOD+2, so leave room for it
        need = _appended(iflg, live) + (2 if iflg == 1 else 0)
        if iflg != 0 and nmmod + need > NSLOT:
            rows.append(row(0, 14.0, make_zon(rnd, set())))
            nmmod = 0
        rows.append(row(iflg, fval, make_zon(rnd, live)))
        nmmod = 0 if iflg == 0 else nmmod + _appended(iflg, live)

    ALL = {1, 2, 3, 4, 5, 6, 7}
    # --- each arm on its own, from a clean reset --------------------------
    for iflg in (1, 2, 3):
        emit(0, set(), 12.0)
        emit(iflg, ALL)
    # --- nothing live: NMMOD must not move --------------------------------
    emit(0, set(), 7.5)
    emit(3, set())
    emit(2, set())
    emit(1, set())
    # --- only slot 7 live, which no arm reaches ---------------------------
    emit(0, set(), 9.0)
    emit(3, {7})
    # --- the IFLG=2 pair, one side at a time ------------------------------
    for live in ({4}, {5}, {4, 5}, {1, 4, 5}):
        emit(0, set(), 21.0)
        emit(2, live)
    # --- IFLG=1 with slot 1 live and not ----------------------------------
    for live in ({1}, {2}, {1, 2}, set()):
        emit(0, set(), 4.0)
        emit(1, live)
    # --- accumulate toward slot 20 ----------------------------------------
    emit(0, set(), 28.0)
    for _ in range(6):
        emit(3, ALL)
    # --- interleaved arms without a reset ---------------------------------
    emit(0, set(), 18.0)
    emit(3, {1, 3, 5})
    emit(2, {4, 5})
    emit(1, {1, 2})
    emit(3, {2, 6})
    # --- random fill -------------------------------------------------------
    for _ in range(180):
        r = rnd(0.0, 1.0)
        if r < 0.18:
            emit(0, set(), rnd(2.0, 30.0))
            continue
        iflg = 1 if r < 0.4 else (2 if r < 0.62 else 3)
        live = {i for i in range(1, 8) if rnd(0.0, 1.0) < 0.55}
        emit(iflg, live)
    return rows


def main():
    outdir = pathlib.Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    rows = cases()
    with open(outdir / "allmodes_cases.csv", "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"allmodes: {len(rows)} calls, {len(COLUMNS)} columns")


if __name__ == "__main__":
    main()
