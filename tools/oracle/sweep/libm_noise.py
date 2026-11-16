"""Is an outlier within reach of the float32 libm floor?

glibc's ``sinf`` / ``cosf`` / ``acosf`` / ``asinf`` / ``atanf`` (what the
gfortran oracle calls) are faithful but not correctly rounded; cudacap's host
math (``voacap_gpu.scalar.f32math``) is correctly rounded.  So the two differ
by one ULP on a small fraction of calls (docs/f32_libm.md).  This script asks,
for each outlier cell, whether perturbations of exactly that kind can move the
answer as far as the observed cudacap-vs-voacapl difference:

* it re-runs the cell's hour on the scalar reference with every one of those
  five functions replaced, in every ``voacap_gpu.scalar`` module, by a copy
  that returns the neighbouring float32 (up or down at random) with
  probability ``--p``, for ``--seeds`` seeds;
* ``--mode replay`` starts from the CAPTURED LUFFY entry state (noise inside
  LUFFY only); ``--mode e2e`` builds the circuit from the deck (noise in the
  setup chain too);
* ``--modules fobby`` (say) confines the noise to the named modules, which
  attributes an outlier to one routine;
* it prints, per cell, the observed |cudacap - voacapl|, the largest
  |noisy - clean| over the seeds, and how often a hop count changed.

A spread of the same order as the observed difference says the outlier is the
libm floor amplified by the model's conditioning; a spread orders of
magnitude smaller says something else is going on.

    uv run python tools/oracle/sweep/libm_noise.py --decks D --dumps D \
        --outliers build/sweep/results/outliers.tsv --mode replay [deck ...]
"""

import argparse
import copy
import csv
import pathlib
import sys
import warnings

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from compare_sweep import ALL_SON, build_deck, parse_deck        # noqa: E402
from tests.scalar.test_luffy_replay import _apply                 # noqa: E402
from tests.test_engine import SON_ABS_BOUND, _SON_NUM, _antennas, _rows  # noqa: E402
from voacap_gpu import engine as _engine                          # noqa: E402
from voacap_gpu.scalar import f32math                             # noqa: E402
from voacap_gpu.scalar.circuit import Circuit                     # noqa: E402
from voacap_gpu.scalar.f32math import F32                         # noqa: E402
from voacap_gpu.scalar.luffy import luffy                         # noqa: E402
from voacap_gpu.scalar.redmap import redmap                       # noqa: E402

NOISY = ("sin32", "cos32", "acos32", "asin32", "atan32")
# --funcs all adds the rest of f32math's transcendental set (glibc's new
# expf/logf/powf are much closer to correctly rounded, so these are off by
# default)
MORE = ("exp32", "log32", "log10_32", "pow32")
_ORIG = {n: getattr(f32math, n) for n in NOISY + MORE}


_ONLY = None          # --modules: restrict the noise to these scalar modules


def _modules():
    return [m for k, m in list(sys.modules.items())
            if k.startswith("voacap_gpu.scalar") and m is not None
            and (_ONLY is None or k.rsplit(".", 1)[-1] in _ONLY)]


def install_noise(rng, p, names=NOISY):
    """Swap every module-level reference to the ``names`` functions."""
    def make(fn):
        def noisy(*x):
            y = fn(*x)
            if rng.random() < p:
                y = np.nextafter(np.float32(y),
                                 np.float32(np.inf if rng.random() < 0.5
                                            else -np.inf))
            return y
        return noisy
    repl = {n: make(_ORIG[n]) for n in names}
    for m in _modules():
        for n in names:
            if getattr(m, n, None) is _ORIG[n]:
                setattr(m, n, repl[n])
    return repl


def remove_noise(repl):
    back = {id(f): n for n, f in repl.items()}
    for m in _modules():
        for k, v in list(vars(m).items()):
            if id(v) in back:
                setattr(m, k, _ORIG[back[id(v)]])


def worst_cells(path):
    """deck -> (hour, slot, field, observed |d|) of its worst bound ratio."""
    best = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            h, s = int(r["hour"]), int(r["slot"])
            cands = [(abs(float(r[f"{x}_got"]) - float(r[f"{x}_exp"]))
                      / SON_ABS_BOUND[x], x) for x in _SON_NUM]
            ratio, fld = max(cands)
            if r["nhp_got"] != r["nhp_exp"]:
                ratio, fld = max(ratio, 1.0), "nhp"
            d = (abs(float(r[f"{fld}_got"]) - float(r[f"{fld}_exp"]))
                 if fld != "nhp" else 0.0)
            if r["deck"] not in best or ratio > best[r["deck"]][0]:
                best[r["deck"]] = (ratio, h, s, fld, d)
    return {k: v[1:] for k, v in best.items()}


def run_cell(decks, dumps, name, hour, mode):
    """A closure that runs the cell's hour once and returns (son, nhp)."""
    d = parse_deck(decks / f"{name}.dat")
    cant = _antennas(_rows(dumps / f"{name}_ant.csv"))
    if mode == "replay":
        ins = _rows(dumps / f"{name}_in.csv")
        c = Circuit(iono=redmap(_engine._COEFFS_DIR, d["coeff"], d["month"],
                                F32(d["ssn"])))
        c.cant = cant
        c.muf.xluf[:] = F32(-1.0)
        for v in ins[:hour - 1]:           # carry-over, noise-free
            _apply(c, v)
            luffy(c, int(v["ipflag"]))
        base = c

        def once():
            x = copy.deepcopy(base)
            _apply(x, ins[hour - 1])
            luffy(x, int(ins[hour - 1]["ipflag"]))
            return x.son
    else:
        def once():
            cs, ips = build_deck(d, cant)
            luffy(cs[hour - 1], ips[hour - 1])
            return cs[hour - 1].son
    return once


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", required=True, type=pathlib.Path)
    ap.add_argument("--dumps", required=True, type=pathlib.Path)
    ap.add_argument("--outliers", required=True, type=pathlib.Path)
    ap.add_argument("--mode", choices=("replay", "e2e"), default="replay")
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--funcs", default=",".join(NOISY),
                    help="comma list from NOISY + MORE, or 'all'")
    ap.add_argument("--modules", default=None,
                    help="comma list of voacap_gpu.scalar module names to "
                    "perturb (default: all), e.g. fobby")
    ap.add_argument("deck", nargs="*")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore", UserWarning)
    global _ONLY
    _ONLY = set(a.modules.split(",")) if a.modules else None
    funcs = (NOISY + MORE if a.funcs == "all"
             else tuple(x for x in a.funcs.split(",") if x))
    assert set(funcs) <= set(_ORIG), funcs
    cells = worst_cells(a.outliers)
    names = a.deck or sorted(cells)
    print("deck\thour\tslot\tfield\tobserved\tnoise_max\tratio\tnhp_flips")
    for name in names:
        hour, slot, fld, obs = cells[name]
        once = run_cell(a.decks, a.dumps, name, hour, a.mode)
        clean = once()
        ref = {x: float(getattr(clean, x)[slot]) for x in ALL_SON}
        nhp0 = int(clean.nhp[slot])
        spread, flips = 0.0, 0
        for s in range(a.seeds):
            repl = install_noise(np.random.default_rng(1000 + s), a.p,
                                 funcs)
            try:
                son = once()
            finally:
                remove_noise(repl)
            flips += int(son.nhp[slot]) != nhp0
            if fld != "nhp":
                spread = max(spread, abs(float(getattr(son, fld)[slot])
                                         - ref[fld]))
        ratio = spread / obs if obs else float("nan")
        print(f"{name}\t{hour}\t{slot}\t{fld}\t{obs:.3g}\t{spread:.3g}\t"
              f"{ratio:.2f}\t{flips}/{a.seeds}", flush=True)


if __name__ == "__main__":
    main()
