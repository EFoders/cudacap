"""Compare cudacap with voacapl over the accuracy-sweep decks.

For every captured deck (``run_sweep.sh`` output), rebuild the circuit from
the deck's RAW cards -- coordinates, NPSL, month, SSN, coefficients, FPROB,
SYSTEM, FREQUENCY, METHOD, parsed here in the reader's fixed columns -- the
way ``tests/test_engine.py::run_deck`` does: one FRESH ``build_circuit`` per
hour sharing one REDMAP, the captured /cantenna/ from ``_ant.csv``.  Run every
deck on CUDA (``predict_circuits``) and compare each /SON/ field with the
captured exit state.  Then:

* ``--cpu-n`` decks (random, seeded) again on ``backend='cpu'``, checked
  bit for bit against the CUDA run;
* ``--own-n`` decks (random, seeded) with the antennas built by cudacap
  itself (``build_circuit(cant=None)`` -> ``engine.isotropic_cant`` with the
  card's isotrope gains as ``tx_gos`` / ``rx_gos`` and its power), compared
  with the oracle and with the captured-antenna run.

The raw card values are also cross-checked against the captured LUFFY entry
state (rsn, amind, lufp, pmp, dmp, noise, psc, month, ssn, frel, method), so a
deck-parsing slip cannot hide in the comparison.

Usage (Windows, repo root)::

    uv run python tools/oracle/sweep/compare_sweep.py \
        --decks //wsl.localhost/FedoraLinux-42/home/<you>/sweep/decks \
        --dumps //wsl.localhost/FedoraLinux-42/home/<you>/sweep/dumps \
        --out build/sweep/results

Writes into ``--out``: ``summary.json`` (everything the report quotes),
``tables.md`` (the report's tables), ``decks.tsv`` (per-deck worst errors),
``outliers.tsv`` (every cell over a bound, with a hop-count disagreement or
a non-zero status) and ``cells.npz`` (every compared cell, got and oracle).
"""

import argparse
import json
import pathlib
import sys
import time
import warnings

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tests.test_engine import (SON_ABS_BOUND, ULP_BAR, _SON_NUM,  # noqa: E402
                               _antennas, _deck_method, _rows)
from voacap_gpu import engine as _engine                          # noqa: E402
from voacap_gpu.engine import (POLE_DEFAULT, System, build_circuit,  # noqa: E402
                               predict_circuits)
from voacap_gpu.scalar.f32math import F32                         # noqa: E402
from voacap_gpu.scalar.redmap import redmap                       # noqa: E402

SEED = 20261115

# every numeric /SON/ column the shim captures (nhp is discrete, separate);
# only _SON_NUM carries a CI bound
ALL_SON = ("angle", "angler", "cprob", "dblos", "dblosl", "dblosu", "dbu",
           "delay", "dbw", "xnynois", "probmp", "reliab", "sndb", "snpr",
           "snrlw", "snrup", "sprob", "vhigh", "rneff")
assert set(_SON_NUM) <= set(ALL_SON)

PATH_BUCKETS = ((0, 1000), (1000, 3000), (3000, 7000), (7000, 10000),
                (10000, 15000), (15000, 20000), (20000, 40100))
LAT_BUCKETS = ((0, 20), (20, 40), (40, 60), (60, 75), (75, 90.01))
SSN_BUCKETS = ((0, 50), (50, 100), (100, 150), (150, 201))


# ---------------------------------------------------------------------------
# The raw deck, in decred.for's fixed columns
# ---------------------------------------------------------------------------

def _f(s):
    s = s.strip()
    return float(s) if s else 0.0


def parse_deck(path):
    """The cards gen_sweep.py writes, read as DECRED reads them."""
    d = {"name": path.stem}
    ants = {}
    for line in path.read_text().splitlines():
        card, line = line[:10].strip(), line.ljust(80)
        if card == "COEFFS":
            d["coeff"] = line[10:14]
        elif card == "MONTH":
            d["month"] = int(_f(line[15:20]))
        elif card == "SUNSPOT":
            d["ssn"] = _f(line[10:15])
        elif card == "CIRCUIT":
            sg = lambda v, h: -_f(v) if h in "SW" else _f(v)  # noqa: E731
            d["tx"] = (sg(line[10:15], line[15]), sg(line[16:25], line[25]))
            d["rx"] = (sg(line[26:35], line[35]), sg(line[36:45], line[45]))
            d["npsl"] = int(_f(line[50:55]))
        elif card == "SYSTEM":
            (d["pwx"], d["noise"], d["amind"], d["lufp"], d["rsn"], d["pmp"],
             d["dmp"]) = (_f(line[c:c + 5]) for c in range(10, 45, 5))
        elif card == "FPROB":
            d["fprob"] = tuple(_f(line[c:c + 5]) for c in range(10, 30, 5))
        elif card == "ANTENNA":
            iantr = int(_f(line[15:20]))
            ants[iantr] = dict(design=_f(line[30:40]), file=line[41:62].strip(),
                               beam=_f(line[63:68]), pwr=_f(line[68:78]))
        elif card == "FREQUENCY":
            fr = [_f(line[c:c + 5]) for c in range(10, 65, 5)]
            d["freqs"] = [f for f in fr if f > 0.0]
        elif card == "METHOD":
            d["method"] = int(_f(line[10:15]))
    # ANTCALC's isotrope gain: TX from the design-frequency field, RX from the
    # power field when non-zero (antcalc.for:119,136)
    d["tx_gain"] = ants[1]["design"]
    d["rx_gain"] = ants[2]["pwr"] if ants[2]["pwr"] != 0.0 else ants[2]["design"]
    d["pwrkw"] = ants[1]["pwr"] if ants[1]["pwr"] > 0.0 else 1.0
    d["tx_beam"], d["rx_beam"] = ants[1]["beam"], ants[2]["beam"]
    return d


def deck_system(d, own_antennas=False):
    kw = dict(required_snr=d["rsn"], min_takeoff_deg=d["amind"],
              required_reliability=int(d["lufp"]),
              multipath_power_db=d["pmp"], multipath_delay_ms=d["dmp"],
              man_made_noise=d["noise"], fprob=d["fprob"])
    if own_antennas:
        kw.update(power_kw=d["pwrkw"], tx_gos=d["tx_gain"],
                  rx_gos=d["rx_gain"])
    return System(**kw)


def cross_check(d, v0, ins):
    """Raw card values vs the captured LUFFY entry state; list of slips."""
    bad = []
    pairs = [("rsn", d["rsn"]), ("amind", d["amind"] if d["amind"] > 0 else 3.0),
             ("lufp", int(d["lufp"])), ("pmp", d["pmp"]), ("dmp", d["dmp"]),
             ("noise", d["noise"]), ("month", d["month"]), ("ssn", d["ssn"]),
             ("npsl", d["npsl"])]
    pairs += [(f"psc{i + 1:02d}", x) for i, x in enumerate(d["fprob"])]
    pairs += [(f"frel{i + 1:02d}", x) for i, x in enumerate(d["freqs"])]
    for k, x in pairs:
        if F32(v0[k]) != F32(x):
            bad.append(f"{k}: card {x!r} vs entry {v0[k]!r}")
    if sum(1 for i in range(1, 12) if v0[f"frel{i:02d}"] > 0) != len(d["freqs"]):
        bad.append("nf differs")
    if _deck_method(v0) != d["method"]:
        bad.append(f"method: card {d['method']} vs entry {_deck_method(v0)}")
    if [int(v["it"]) for v in ins] != list(range(1, 25)):
        bad.append("hours are not 1..24")
    return bad


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------

def build_deck(d, cant, own_antennas=False):
    """24 fresh circuits, one REDMAP, as ``predict`` hands them over."""
    st = redmap(_engine._COEFFS_DIR, d["coeff"], d["month"], F32(d["ssn"]))
    system = deck_system(d, own_antennas)
    circuits, ipflags = [], []
    for hour in range(1, 25):
        c, ip = build_circuit(
            d["tx"], d["rx"], d["month"], F32(d["ssn"]), hour, d["freqs"],
            method=d["method"], system=system,
            cant=None if own_antennas else cant, coeff=d["coeff"],
            pole=POLE_DEFAULT, npsl=d["npsl"], iono=st,
            tx_bearing=d["tx_beam"] if own_antennas else None,
            rx_bearing=d["rx_beam"] if own_antennas else None)
        circuits.append(c)
        ipflags.append(ip)
    return circuits, ipflags


def son_arrays(circuits, nf):
    out = {nm: np.array([np.asarray(getattr(c.son, nm)[1:nf + 1], np.float32)
                         for c in circuits]) for nm in ALL_SON}
    out["nhp"] = np.array([np.asarray(c.son.nhp[1:nf + 1], np.int32)
                           for c in circuits])
    return out


def oracle_arrays(outs, nf):
    o = {nm: np.array([[r[f"{nm}{j:02d}"] for j in range(1, nf + 1)]
                       for r in outs], np.float32) for nm in ALL_SON}
    o["nhp"] = np.array([[int(r[f"nhp{j:02d}"]) for j in range(1, nf + 1)]
                         for r in outs], np.int32)
    return o


def regime(method, ipflag, gcdkm):
    if ipflag == 200:
        return "long (>=10000 km)"
    if method == 30 and gcdkm >= 7000.0:
        return "smooth (7000-10000 km, M30)"
    return "short"


def _bucket(x, buckets):
    for lo, hi in buckets:
        if lo <= x < hi:
            return f"{lo:g}-{hi:g}" if hi < 40000 else f">={lo:g}"
    return "?"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ulp_floored(g, e):
    """Vectorised tests.scalar.dumputil.ulp_diff_floored (floor 1.0)."""
    g = g.astype(np.float32)
    e = e.astype(np.float32)
    den = np.spacing(np.maximum(np.abs(e), np.float32(1.0))).astype(np.float64)
    d = np.abs(g.astype(np.float64) - e.astype(np.float64)) / den
    return np.where(g == e, 0.0, d)


def field_stats(err):
    if err.size == 0:
        return dict(n=0)
    return dict(n=int(err.size), max=float(err.max()),
                p999=float(np.quantile(err, 0.999)),
                p99=float(np.quantile(err, 0.99)),
                median=float(np.median(err)),
                exact=float(np.mean(err == 0.0)))


def group_stats(cells, mask):
    """Headline numbers for the cells selected by ``mask``."""
    ok = mask & cells["status_ok"]
    r = dict(decks=int(np.unique(cells["deck"][mask]).size),
             cells=int(mask.sum()), status_bad=int((mask & ~cells["status_ok"]).sum()))
    r["nhp_agree"] = float(np.mean(cells["nhp_got"][ok] == cells["nhp_exp"][ok])) if ok.any() else float("nan")
    ex = [cells[f"abs_{nm}"][ok] == 0.0 for nm in _SON_NUM]
    r["bitexact"] = float(np.mean(np.concatenate(ex))) if ok.any() else float("nan")
    for nm in _SON_NUM:
        a = cells[f"abs_{nm}"][ok]
        r[f"max_{nm}"] = float(a.max()) if a.size else float("nan")
        r[f"over_{nm}"] = int((a > SON_ABS_BOUND[nm]).sum())
    r["over_any"] = int(sum(r[f"over_{nm}"] for nm in _SON_NUM))
    return r


_T = {}


def _timed(mod, name):
    """Accumulate wall time spent in ``mod.name`` into ``_T[name]``.

    ``predict_circuits`` imports the driver functions at call time, so
    wrapping the module attributes is enough to split its time into the
    per-circuit host pack / unpack and the device launch (transfers + kernel).
    """
    fn = getattr(mod, name)

    def wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _T[name] = _T.get(name, 0.0) + time.perf_counter() - t0
    setattr(mod, name, wrap)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decks", required=True, type=pathlib.Path)
    ap.add_argument("--dumps", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--chunk", type=int, default=100,
                    help="decks per predict_circuits launch")
    ap.add_argument("--cpu-n", type=int, default=100)
    ap.add_argument("--own-n", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore", UserWarning)

    runlog = [ln.split("\t") for ln in
              (a.dumps / "RUNLOG.tsv").read_text().splitlines()[1:]]
    skipped = [(r[0], r[1]) for r in runlog if r[1] != "ok"]
    names = sorted(r[0] for r in runlog if r[1] == "ok")
    if a.limit:
        names = names[:a.limit]
    print(f"{len(names)} captured decks, {len(skipped)} not captured")

    # warm-up: compile / load the kernel outside the timed region
    d0 = parse_deck(a.decks / f"{names[0]}.dat")
    c0, i0 = build_deck(d0, None, own_antennas=True)
    predict_circuits(c0[:1], i0[:1], backend="cuda")
    from voacap_gpu.cuda import driver as _driver
    for nm in ("pack_circuit", "unpack_circuit", "shared_iono_tables",
               "shared_antenna_tables", "launch_luffy"):
        _timed(_driver, nm)

    meta, slips = [], {}
    col = {k: [] for k in ("deck", "hour", "slot", "freq", "nhp_got",
                           "nhp_exp", "status_ok")}
    for nm in ALL_SON:
        col[f"got_{nm}"], col[f"exp_{nm}"] = [], []
    t_read = t_setup = t_gpu = 0.0
    gpu_son = {}                       # deck -> CUDA son arrays (for CPU check)
    ch_hours = 0
    for lo in range(0, len(names), a.chunk):
        batch = names[lo:lo + a.chunk]
        circuits, ipflags, spans = [], [], []
        exps = {}
        for name in batch:
            t0 = time.perf_counter()
            ins = _rows(a.dumps / f"{name}_in.csv")
            outs = _rows(a.dumps / f"{name}_out.csv")
            antrows = _rows(a.dumps / f"{name}_ant.csv")
            t1 = time.perf_counter()
            d = parse_deck(a.decks / f"{name}.dat")
            cant = _antennas(antrows)
            cs, ips = build_deck(d, cant)
            t2 = time.perf_counter()
            t_read += t1 - t0
            t_setup += t2 - t1
            s = cross_check(d, ins[0], ins)
            if s:
                slips[name] = s
            nf = len(d["freqs"])
            exps[name] = (d, oracle_arrays(outs, nf), ins[0], ips[0])
            spans.append((name, len(circuits), nf))
            circuits += cs
            ipflags += ips
        t0 = time.perf_counter()
        _, status = predict_circuits(circuits, ipflags, backend="cuda")
        t_gpu += time.perf_counter() - t0
        ch_hours += len(circuits)
        for name, start, nf in spans:
            d, o, v0, ip = exps[name]
            cs = circuits[start:start + 24]
            g = son_arrays(cs, nf)
            stt = np.asarray(status[start:start + 24])
            gpu_son[name] = (g, stt)
            k = len(meta)
            gcd = float(cs[0].gcdkm)
            meta.append(dict(
                name=name, method=d["method"], coeff=d["coeff"],
                es=d["fprob"][3], npsl=d["npsl"], gcdkm=gcd, ipflag=int(ip),
                regime=regime(d["method"], int(ip), gcd), ssn=d["ssn"],
                month=d["month"], nf=nf,
                maxlat=max(abs(d["tx"][0]), abs(d["rx"][0])),
                tx=d["tx"], rx=d["rx"], freqs=d["freqs"]))
            H = 24
            col["deck"].append(np.full(H * nf, k, np.int32))
            col["hour"].append(np.repeat(np.arange(1, 25, dtype=np.int32), nf))
            col["slot"].append(np.tile(np.arange(1, nf + 1, dtype=np.int32), H))
            col["freq"].append(np.tile(np.array(d["freqs"], np.float32), H))
            col["nhp_got"].append(g["nhp"].ravel())
            col["nhp_exp"].append(o["nhp"].ravel())
            col["status_ok"].append(np.repeat(stt == 0, nf))
            for f in ALL_SON:
                col[f"got_{f}"].append(g[f].ravel())
                col[f"exp_{f}"].append(o[f].ravel())
        print(f"  {lo + len(batch)}/{len(names)} decks  setup {t_setup:.1f}s "
              f"gpu {t_gpu:.1f}s", flush=True)
        del circuits

    cells = {k: np.concatenate(v) for k, v in col.items()}
    for f in ALL_SON:
        cells[f"abs_{f}"] = np.abs(cells[f"got_{f}"].astype(np.float64)
                                   - cells[f"exp_{f}"].astype(np.float64))
    ulp = np.max(np.stack([ulp_floored(cells[f"got_{f}"], cells[f"exp_{f}"])
                           for f in _SON_NUM]), axis=0)
    cells["ulp_max"] = ulp
    np.savez_compressed(a.out / "cells.npz", **cells)

    ok = cells["status_ok"]
    allmask = np.ones_like(ok)
    summary = dict(seed=a.seed, decks=len(names), skipped=skipped,
                   circuit_hours=int(ch_hours), cells=int(ok.size),
                   cells_status_bad=int((~ok).sum()), card_slips=slips)
    summary["fields"] = {f: field_stats(cells[f"abs_{f}"][ok]) for f in ALL_SON}
    summary["ulp_worst"] = float(ulp[ok].max())
    summary["ulp_over_bar"] = int((ulp[ok] > ULP_BAR).sum())
    summary["overall"] = group_stats(cells, allmask)
    dk = cells["deck"]

    def by(key):
        vals = sorted({key(m) for m in meta}, key=str)
        res = {}
        for v in vals:
            idx = np.array([i for i, m in enumerate(meta) if key(m) == v])
            res[str(v)] = group_stats(cells, np.isin(dk, idx))
        return res

    summary["by"] = {
        "method": by(lambda m: m["method"]),
        "regime": by(lambda m: m["regime"]),
        "path km": by(lambda m: _bucket(m["gcdkm"], PATH_BUCKETS)),
        "coeff/Es": by(lambda m: f"{m['coeff']} Es {m['es']:.2f}"),
        "max |lat|": by(lambda m: _bucket(m["maxlat"], LAT_BUCKETS)),
        "SSN": by(lambda m: _bucket(m["ssn"], SSN_BUCKETS)),
        "path": by(lambda m: "long (L)" if m["npsl"] else "short (S)"),
    }
    summary["timing"] = dict(
        oracle_csv_read_s=t_read, host_setup_s=t_setup, gpu_predict_s=t_gpu,
        circuit_hours=int(ch_hours),
        gpu_us_per_circuit_hour=1e6 * t_gpu / ch_hours,
        predict_split_s=dict(_T), chunk_decks=a.chunk,
        total_us_per_circuit_hour=1e6 * (t_gpu + t_setup) / ch_hours)

    # --- per-deck table and outliers ---------------------------------------
    with open(a.out / "decks.tsv", "w", newline="\n") as f:
        f.write("deck\tmethod\tregime\tgcdkm\tcoeff\tes\tssn\tmonth\tmaxlat\t"
                "npsl\tnf\tnhp_mismatch\tstatus_bad\tbitexact\t"
                + "\t".join(f"max_{x}" for x in _SON_NUM) + "\n")
        for k, m in enumerate(meta):
            s = group_stats(cells, dk == k)
            okk = (dk == k) & ok
            nm_ = int((cells["nhp_got"][okk] != cells["nhp_exp"][okk]).sum())
            m["nhp_mismatch"], m["over"] = nm_, s["over_any"]
            f.write("\t".join(map(str, [
                m["name"], m["method"], m["regime"], f"{m['gcdkm']:.1f}",
                m["coeff"], m["es"], m["ssn"], m["month"],
                f"{m['maxlat']:.2f}", m["npsl"], m["nf"], nm_,
                s["status_bad"], f"{s['bitexact']:.4f}"]
                + [f"{s[f'max_{x}']:.3e}" for x in _SON_NUM])) + "\n")

    bad = ~ok | (cells["nhp_got"] != cells["nhp_exp"])
    for x in _SON_NUM:
        bad |= cells[f"abs_{x}"] > SON_ABS_BOUND[x]
    with open(a.out / "outliers.tsv", "w", newline="\n") as f:
        f.write("deck\thour\tslot\tfreq\tstatus_ok\tnhp_got\tnhp_exp\t"
                "over\tmethod\tregime\tgcdkm\tcoeff\tes\tssn\tmonth\ttx\trx\t"
                + "\t".join(f"{x}_got\t{x}_exp" for x in _SON_NUM) + "\n")
        for i in np.flatnonzero(bad):
            m = meta[dk[i]]
            over = [x for x in _SON_NUM
                    if cells[f"abs_{x}"][i] > SON_ABS_BOUND[x]]
            f.write("\t".join(map(str, [
                m["name"], cells["hour"][i], cells["slot"][i],
                f"{cells['freq'][i]:.2f}", bool(ok[i]), cells["nhp_got"][i],
                cells["nhp_exp"][i], ",".join(over) or "-", m["method"],
                m["regime"], f"{m['gcdkm']:.1f}", m["coeff"], m["es"],
                m["ssn"], m["month"], m["tx"], m["rx"]]
                + [f"{cells[f'got_{x}'][i]}\t{cells[f'exp_{x}'][i]}"
                   for x in _SON_NUM])) + "\n")
    summary["outlier_cells"] = int(bad.sum())
    summary["outlier_decks"] = sorted({meta[dk[i]]["name"]
                                       for i in np.flatnonzero(bad)})
    summary["nhp_mismatch_cells"] = int((ok & (cells["nhp_got"]
                                               != cells["nhp_exp"])).sum())

    # --- CPU backend == CUDA, bit for bit ----------------------------------
    rng = np.random.default_rng(a.seed + 1)
    cpu_names = sorted(rng.choice(names, size=min(a.cpu_n, len(names)),
                                  replace=False).tolist())
    t_cpu, cpu_hours, cpu_diff = 0.0, 0, []
    for name in cpu_names:
        d = parse_deck(a.decks / f"{name}.dat")
        cant = _antennas(_rows(a.dumps / f"{name}_ant.csv"))
        cs, ips = build_deck(d, cant)
        t0 = time.perf_counter()
        _, st = predict_circuits(cs, ips, backend="cpu")
        t_cpu += time.perf_counter() - t0
        cpu_hours += len(cs)
        g = son_arrays(cs, len(d["freqs"]))
        gg, gst = gpu_son[name]
        diffs = [f for f in (*ALL_SON, "nhp")
                 if not np.array_equal(g[f].view(np.int32) if f != "nhp" else g[f],
                                       gg[f].view(np.int32) if f != "nhp" else gg[f])]
        if diffs or not np.array_equal(np.asarray(st), gst):
            cpu_diff.append((name, diffs))
    summary["cpu"] = dict(decks=len(cpu_names), circuit_hours=cpu_hours,
                          identical_decks=len(cpu_names) - len(cpu_diff),
                          differing=cpu_diff, seconds=t_cpu,
                          ms_per_circuit_hour=1e3 * t_cpu / max(cpu_hours, 1))
    print(f"CPU==CUDA: {len(cpu_names) - len(cpu_diff)}/{len(cpu_names)} "
          f"decks bit-identical; {1e3 * t_cpu / max(cpu_hours, 1):.1f} "
          "ms/circuit-hour", flush=True)

    # --- cudacap's own antenna builder --------------------------------------
    rng = np.random.default_rng(a.seed + 2)
    own_names = sorted(rng.choice(names, size=min(a.own_n, len(names)),
                                  replace=False).tolist())
    own_cells = {f: [] for f in _SON_NUM}
    own_same, own_nhp_bad, own_n_cells, own_offaz = 0, 0, 0, []
    circuits, ipflags, spans = [], [], []
    for name in own_names:
        d = parse_deck(a.decks / f"{name}.dat")
        cs, ips = build_deck(d, None, own_antennas=True)
        spans.append((name, len(circuits), len(d["freqs"])))
        circuits += cs
        ipflags += ips
        antrows = _rows(a.dumps / f"{name}_ant.csv")
        cap = {int(r["iant"]): r["offazim"] for r in antrows}
        own_offaz.append(max(
            abs((float(cs[0].cant.offazim[i]) - cap[i] + 180.0) % 360.0 - 180.0)
            for i in (1, 2)))
    _T.clear()
    _, status = predict_circuits(circuits, ipflags, backend="cuda")
    own_diff_decks = []
    for name, start, nf in spans:
        g = son_arrays(circuits[start:start + 24], nf)
        gg, gst = gpu_son[name]
        o = oracle_arrays(_rows(a.dumps / f"{name}_out.csv"), nf)
        same = all(np.array_equal(g[f].view(np.int32), gg[f].view(np.int32))
                   for f in ALL_SON) and np.array_equal(g["nhp"], gg["nhp"])
        own_same += same
        if not same:
            own_diff_decks.append(name)
        stt = np.asarray(status[start:start + 24]) == 0
        okm = np.repeat(stt[:, None], nf, axis=1)
        own_nhp_bad += int(((g["nhp"] != o["nhp"]) & okm).sum())
        own_n_cells += int(okm.sum())
        for f in _SON_NUM:
            own_cells[f].append(np.abs(g[f].astype(np.float64)
                                       - o[f].astype(np.float64))[okm])
    summary["own_antennas"] = dict(
        decks=len(own_names), cells=own_n_cells,
        identical_to_captured_cant=own_same, differing=own_diff_decks,
        nhp_mismatch=own_nhp_bad,
        offazim_worst_deg=float(max(own_offaz)) if own_offaz else None,
        fields={f: field_stats(np.concatenate(v)) for f, v in own_cells.items()})
    print(f"own antennas: {own_same}/{len(own_names)} decks bit-identical to "
          "the captured-cant run", flush=True)

    (a.out / "summary.json").write_text(json.dumps(summary, indent=1,
                                                   default=str))
    write_tables(summary, a.out / "tables.md")
    print(f"wrote {a.out}")


def write_tables(s, path):
    L = []
    L.append("| field | units | max | p99.9 | p99 | median | bit-exact | bound | over |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    units = dict(angle="deg", angler="deg", cprob="frac", dblos="dB",
                 dblosl="dB", dblosu="dB", dbu="dB(uV/m)", delay="ms",
                 dbw="dBW", xnynois="dBW", probmp="frac", reliab="frac",
                 sndb="dB-Hz", snpr="dB", snrlw="dB", snrup="dB",
                 sprob="frac", vhigh="km", rneff="dB")
    for f in ALL_SON:
        st = s["fields"][f]
        b = SON_ABS_BOUND.get(f)
        over = s["overall"].get(f"over_{f}", "")
        L.append(f"| {f} | {units[f]} | {st['max']:.3g} | {st['p999']:.3g} | "
                 f"{st['p99']:.3g} | {st['median']:.3g} | "
                 f"{100 * st['exact']:.2f}% | {b if b is not None else '-'} | "
                 f"{over} |")
    L.append("")
    for key, groups in s["by"].items():
        L.append(f"**By {key}**")
        L.append("")
        L.append("| " + key + " | decks | cells | bit-exact | nhp agree | "
                 + " | ".join(f"max {x}" for x in
                              ("dbw", "sndb", "reliab", "angle", "delay",
                               "vhigh")) + " | cells over a bound |")
        L.append("|---" * 12 + "|")
        for g, r in groups.items():
            L.append(f"| {g} | {r['decks']} | {r['cells']} | "
                     f"{100 * r['bitexact']:.2f}% | {100 * r['nhp_agree']:.3f}% | "
                     + " | ".join(f"{r[f'max_{x}']:.2g}" for x in
                                  ("dbw", "sndb", "reliab", "angle", "delay",
                                   "vhigh"))
                     + f" | {r['over_any']} |")
        L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
