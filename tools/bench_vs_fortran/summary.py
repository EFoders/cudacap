"""Like-for-like GPU vs Fortran summary from the dated result files here.

Reads the newest ``results_<date>_fortran.txt`` (from ``bench_fortran.sh``)
and ``results_<date>_gpu.json`` (from ``bench_gpu.py``) in this directory,
or the files given, and prints one table: circuit-hours per second and
microseconds per (circuit, hour, frequency) evaluation, bit-exact and
fast-math, plus the GPU / 32-thread voacapl ratio.

Both sides run the SAME workload definition (``gen_bench_decks.py``): Ottawa
TX, receivers on the 2-degree world grid, 9 frequencies, month 7, SSN 120,
CCIR, METHOD 30, isotropic antennas, hours 1-24.  voacapl runs one process
per 24-hour circuit (how it runs a deck), so its numbers include one process
start-up per circuit; the one-hour decks size that start-up separately.

    uv run python tools/bench_vs_fortran/summary.py
    uv run python tools/bench_vs_fortran/summary.py --gpu path/to/gpu.json
"""

import argparse
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent

# gen_bench_decks.py: 640 single-circuit 24-hour decks, 64 one-hour decks,
# 9 frequencies per circuit-hour
N_PARALLEL = 640
N_SERIAL = 64
HOURS = 24
NFREQ = 9


def _newest(pattern):
    files = sorted(HERE.glob(pattern))
    return files[-1] if files else None


def _date_of(path):
    m = re.search(r"results_(\d{4}-\d{2}-\d{2})_", path.name)
    return m.group(1) if m else "?"


def parse_fortran(path):
    """``bench_fortran.sh`` output -> dict of best-of-reps timings (s)."""
    reps = {}
    meta = {}
    for line in pathlib.Path(path).read_text().splitlines():
        m = re.match(r"(\w+) rep\d+: ([\d.]+) s", line)
        if m:
            reps.setdefault(m.group(1), []).append(float(m.group(2)))
        elif line.startswith("cpu:"):
            meta["cpu"] = line[4:].split(", workers")[0].strip()
        elif line.startswith("gfortran:"):
            meta["gfortran"] = line[9:].strip()
    best = {k: min(v) for k, v in reps.items()}
    out = dict(meta, date=_date_of(pathlib.Path(path)), path=str(path))
    t_par = best["parallel_24h_x640"]
    out["parallel"] = dict(s=t_par, chs=N_PARALLEL * HOURS / t_par,
                           us_chf=1e6 * t_par / (N_PARALLEL * HOURS * NFREQ))
    if "serial_24h_x64" in best:
        t24 = best["serial_24h_x64"]
        out["serial"] = dict(s=t24, chs=N_SERIAL * HOURS / t24,
                             us_chf=1e6 * t24 / (N_SERIAL * HOURS * NFREQ))
        if "serial_1h_x64" in best:
            t1 = best["serial_1h_x64"]
            # t(1 h) = start-up + 1 h; per-hour compute = (t24 - t1) / 23
            per_h = (t24 - t1) / N_SERIAL / (HOURS - 1)
            out["serial_compute_only"] = dict(
                chs=1.0 / per_h, us_chf=1e6 * per_h / NFREQ)
            out["startup_ms"] = 1e3 * (t1 / N_SERIAL - per_h)
    return out


def _best_run(runs, fast):
    sel = {k: v for k, v in runs.items() if f"fast={fast}" in k}
    if not sel:
        return None, None
    k = max(sel, key=lambda k: sel[k]["chs"])
    return k, sel[k]


def parse_gpu(path):
    """``bench_gpu.py`` JSON -> best world-grid and 640-circuit rows."""
    j = json.loads(pathlib.Path(path).read_text())
    nf = j.get("world_grid", {}).get("freqs", NFREQ)
    out = dict(device=j.get("device", "?"), date=j.get("date") or
               _date_of(pathlib.Path(path)), path=str(path), freqs=nf)
    runs = j["world_grid"]["runs"]
    for fast in (False, True):
        k, r = _best_run(runs, fast)
        if r is not None:
            out[f"world_{'fast' if fast else 'exact'}"] = dict(
                config=k, chs=r["chs"], us_chf=1e6 / (r["chs"] * nf))
    fw = j.get("fortran_workload_gpu", {})
    # the 2026-11-15 file holds one bit-exact run; newer ones hold both modes
    modes = {"exact": fw} if "chs" in fw else fw
    for mode in ("exact", "fast"):
        r = modes.get(mode)
        if r:
            out[f"f640_{mode}"] = dict(chs=r["chs"], us_chf=1e6 / (r["chs"] * nf))
    out["world_circuit_hours"] = j["world_grid"]["circuit_hours"]
    return out


def load(fortran=None, gpu=None):
    """(fortran, gpu) parsed dicts from the given or newest files; None if absent."""
    fpath = pathlib.Path(fortran) if fortran else _newest("results_*_fortran.txt")
    gpath = pathlib.Path(gpu) if gpu else _newest("results_*_gpu.json")
    f = parse_fortran(fpath) if fpath and fpath.exists() else None
    g = parse_gpu(gpath) if gpath and gpath.exists() else None
    return f, g


def table(f, g):
    """The like-for-like summary as printable lines."""
    lines = []
    if f is None or g is None:
        return ["no bench_vs_fortran results found in " + str(HERE)]
    lines.append(f"like-for-like (tools/bench_vs_fortran): GPU {g['device']} "
                 f"[{g['date']}] vs voacapl on {f.get('cpu', '?')} "
                 f"[{f['date']}]; {NFREQ} freqs/circuit-hour")
    lines.append(f"  {'what':<44} {'circuit-h/s':>12} {'us/(c,h,f)':>11}"
                 f" {'vs 32-thr':>9}")
    ref = f["parallel"]["chs"]

    def row(label, r):
        if r is None:
            return
        lines.append(f"  {label:<44} {r['chs']:>12,.0f} {r['us_chf']:>11.3f}"
                     f" {r['chs'] / ref:>8.2f}x")

    row("voacapl 32 threads, 640 circuits", f["parallel"])
    row("voacapl 1 thread (incl. process start-up)", f.get("serial"))
    row("voacapl 1 thread, compute only (est.)", f.get("serial_compute_only"))
    row(f"GPU world grid {g['world_circuit_hours']:,} ch, bit-exact",
        g.get("world_exact"))
    row(f"GPU world grid {g['world_circuit_hours']:,} ch, fast-math",
        g.get("world_fast"))
    row("GPU, the same 640 circuits, bit-exact", g.get("f640_exact"))
    row("GPU, the same 640 circuits, fast-math", g.get("f640_fast"))
    if "startup_ms" in f:
        lines.append(f"  (voacapl process start-up ~{f['startup_ms']:.1f} ms "
                     f"per circuit, from the one-hour decks)")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fortran", help="bench_fortran.sh output (default: newest)")
    ap.add_argument("--gpu", help="bench_gpu.py JSON (default: newest)")
    a = ap.parse_args()
    print("\n".join(table(*load(a.fortran, a.gpu))))


if __name__ == "__main__":
    main()
