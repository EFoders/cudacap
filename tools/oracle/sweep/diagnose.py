"""Bisect one sweep deck's disagreement: setup chain or LUFFY?

For each hour of a deck this prints three things:

1. **setup** -- cudacap's built circuit against the captured LUFFY entry state
   (the fields ``tests/test_engine.py::test_setup_chain_vs_oracle`` checks,
   plus the discrete KM / KFX / NANG / IPFLAG), worst floored ULP and where;
2. **e2e** -- the end-to-end /SON/ (raw deck -> build_circuit -> LUFFY, the
   sweep's configuration) against the oracle;
3. **replay** -- LUFFY (the scalar reference) run from the CAPTURED entry state
   (``tests/scalar/test_luffy_replay._apply``, one Circuit carried across the
   hours as the replay test does), against the oracle.

If ``replay`` is clean and ``e2e`` is not, the error enters in the setup chain
and LUFFY amplifies it; if ``replay`` shows it too, it is inside LUFFY.

    uv run python tools/oracle/sweep/diagnose.py --decks D --dumps D sw0008 [--hours 1,3]
    uv run python tools/oracle/sweep/diagnose.py --decks D --dumps D --table sw0008 sw0080 ...
"""

import argparse
import pathlib
import sys
import warnings

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from compare_sweep import (ALL_SON, build_deck, oracle_arrays,  # noqa: E402
                           parse_deck, son_arrays, ulp_floored)
from tests.scalar.test_luffy_replay import _apply                 # noqa: E402
from tests.test_engine import (SON_ABS_BOUND, _MUF_HR, _MUF_LAY,  # noqa: E402
                               _PER_AREA, _SON_NUM, _antennas, _rows)
from voacap_gpu import engine as _engine                          # noqa: E402
from voacap_gpu.engine import predict_circuits                    # noqa: E402
from voacap_gpu.scalar.circuit import Circuit                     # noqa: E402
from voacap_gpu.scalar.f32math import F32                         # noqa: E402
from voacap_gpu.scalar.luffy import luffy                         # noqa: E402
from voacap_gpu.scalar.redmap import redmap                       # noqa: E402


def setup_checks(circ):
    lay, muf = circ.layers, circ.muf
    checks = {"gcd": circ.gcd, "gcdkm": circ.gcdkm, "gmtr": circ.time.gmtr,
              "rlongd": circ.rlongd}
    for k in range(1, lay.km + 1):
        for nm in _PER_AREA:
            checks[f"{nm}{k:02d}"] = getattr(lay, nm)[k]
        for ly in range(1, 4):
            checks[f"fi{ly}{k}"] = lay.fi[ly, k]
            checks[f"yi{ly}{k}"] = lay.yi[ly, k]
            checks[f"hi{ly}{k}"] = lay.hi[ly, k]
    it = circ.time.it
    for nm in _MUF_HR:
        checks[nm] = getattr(muf, nm)[it]
    for ly in range(1, 5):
        for nm in _MUF_LAY:
            checks[f"{nm}{ly:02d}"] = getattr(muf, nm)[ly]
    for i in range(1, 41):
        checks[f"ang{i:02d}"] = circ.rays.ang[i]
    return checks


def worst(g, o, nf, h):
    best = (0.0, "-")
    for nm in _SON_NUM:
        a = np.abs(g[nm][h].astype(np.float64) - o[nm][h].astype(np.float64))
        r = a / SON_ABS_BOUND[nm]
        j = int(np.argmax(r))
        if r[j] > best[0]:
            best = (float(r[j]), f"{nm}{j + 1} {g[nm][h, j]!r} vs "
                    f"{o[nm][h, j]!r} (|d| {a[j]:.3g})")
    nhp = [j + 1 for j in range(nf) if g["nhp"][h, j] != o["nhp"][h, j]]
    return best, nhp


GEOM_FIELDS = ("gcd", "gcdkm", "rlongd", "clat", "clong", "glat", "rd")


def _group(field):
    base = field.rstrip("0123456789")
    if base in GEOM_FIELDS:
        return "geom"
    if base in ("fi", "yi", "hi"):
        return "layers"
    return "muf/other"


def analyse(decks, dumps, name, backend="cpu"):
    """Per-hour records: setup-chain diff, e2e and replay worst/bound."""
    d = parse_deck(decks / f"{name}.dat")
    ins = _rows(dumps / f"{name}_in.csv")
    outs = _rows(dumps / f"{name}_out.csv")
    cant = _antennas(_rows(dumps / f"{name}_ant.csv"))
    nf = len(d["freqs"])
    cs, ips = build_deck(d, cant)
    # LUFFY rewrites parts of the circuit state, so snapshot the setup first
    setup = [(setup_checks(c), c.layers.km, c.layers.kfx, c.rays.nang)
             for c in cs]
    predict_circuits(cs, ips, backend=backend)
    g = son_arrays(cs, nf)
    o = oracle_arrays(outs, nf)

    rep = Circuit(iono=redmap(_engine._COEFFS_DIR, d["coeff"], d["month"],
                              F32(d["ssn"])))
    rep.cant = cant
    rep.muf.xluf[:] = F32(-1.0)
    rg = {nm: np.zeros((24, nf), np.float32) for nm in ALL_SON}
    rg["nhp"] = np.zeros((24, nf), np.int32)
    for h, v in enumerate(ins):
        _apply(rep, v)
        luffy(rep, int(v["ipflag"]))
        for nm in ALL_SON:
            rg[nm][h] = np.asarray(getattr(rep.son, nm)[1:nf + 1])
        rg["nhp"][h] = np.asarray(rep.son.nhp[1:nf + 1])

    recs = []
    for h, v in enumerate(ins):
        checks, km, kfx, nang = setup[h]
        disc = [(k, int(x), int(v[k])) for k, x in
                (("km", km), ("kfx", kfx), ("nang", nang), ("ipflag", ips[h]))
                if int(x) != int(v[k])]
        sw, swhere, nexact = 0.0, "-", 0
        gw = {"geom": 0.0, "layers": 0.0, "muf/other": 0.0}
        for k, x in checks.items():
            u = float(ulp_floored(np.float32([x]), np.float32([v[k]]))[0])
            nexact += u == 0.0
            gw[_group(k)] = max(gw[_group(k)], u)
            if u > sw:
                sw, swhere = u, f"{k} {F32(x)!r} vs {F32(v[k])!r}"
        e2e, e2e_nhp = worst(g, o, nf, h)
        rp, rp_nhp = worst(rg, o, nf, h)
        recs.append(dict(hour=h + 1, setup_exact=nexact, setup_n=len(checks),
                         setup_worst=sw, setup_where=swhere, groups=gw,
                         discrete=disc, e2e=e2e, e2e_nhp=e2e_nhp, replay=rp,
                         replay_nhp=rp_nhp))
    return d, ins[0], recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", required=True, type=pathlib.Path)
    ap.add_argument("--dumps", required=True, type=pathlib.Path)
    ap.add_argument("deck", nargs="+")
    ap.add_argument("--hours", default=None)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--table", action="store_true",
                    help="one line per deck (its worst e2e hour)")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore", UserWarning)
    if a.table:
        print("deck	hour	e2e/bound	e2e_nhp	replay/bound	replay_nhp	"
              "setup_worst_ulp	geom_ulp	layers_ulp	muf_ulp	discrete	"
              "e2e_where	setup_where")
    for name in a.deck:
        d, v0, recs = analyse(a.decks, a.dumps, name, a.backend)
        if a.table:
            r = max(recs, key=lambda r: (bool(r["e2e_nhp"]), r["e2e"][0]))
            print("	".join(map(str, [
                name, r["hour"], f"{r['e2e'][0]:.2f}", r["e2e_nhp"] or "-",
                f"{r['replay'][0]:.2f}", r["replay_nhp"] or "-",
                f"{r['setup_worst']:.0f}", f"{r['groups']['geom']:.0f}",
                f"{r['groups']['layers']:.0f}",
                f"{r['groups']['muf/other']:.0f}", r["discrete"] or "-",
                r["e2e"][1], r["setup_where"]])), flush=True)
            continue
        print(f"{name}: tx {d['tx']} rx {d['rx']} npsl {d['npsl']} "
              f"M{d['month']} SSN {d['ssn']:g} METHOD {d['method']} "
              f"{d['coeff']} FPROB {d['fprob']} freqs {d['freqs']}")
        print(f"  SYSTEM noise {d['noise']:g} amind {d['amind']} lufp "
              f"{d['lufp']:g} rsn {d['rsn']} pmp {d['pmp']} dmp {d['dmp']}; "
              f"gcdkm captured {v0['gcdkm']}")
        hours = (None if a.hours is None
                 else {int(x) for x in a.hours.split(",")})
        for r in recs:
            if hours and r["hour"] not in hours:
                continue
            disc = r["discrete"]
            print(f" h{r['hour']:2d} setup {r['setup_exact']}/{r['setup_n']} "
                  f"exact, worst {r['setup_worst']:.0f} ULP {r['setup_where']}"
                  f"{' DISCRETE ' + str(disc) if disc else ''}")
            print(f"      e2e    worst/bound {r['e2e'][0]:.2f} {r['e2e'][1]}"
                  + (f" NHP slots {r['e2e_nhp']}" if r["e2e_nhp"] else ""))
            print(f"      replay worst/bound {r['replay'][0]:.2f} "
                  f"{r['replay'][1]}"
                  + (f" NHP slots {r['replay_nhp']}" if r["replay_nhp"] else ""))


if __name__ == "__main__":
    main()
