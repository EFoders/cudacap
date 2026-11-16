"""GPU/CPU timings for the GPU-vs-Fortran benchmark (run on a quiet machine).

Times, on the same workload definition as ``bench_fortran.sh``
(``gen_bench_decks.py``: Ottawa TX, 9 frequencies, month 7, SSN 120, CCIR,
METHOD 30, isotropic antennas, hours 1-24):

1. the 2-degree world grid through the production device path
   (``PipelinedSetupLuffy``), bit-exact and fast-math, two chunk sizes each;
2. the Fortran benchmark's own 640 circuits x 24 h, same path, bit-exact and
   fast-math (the strictly like-for-like row);
3. ``predict()`` latency for one circuit x 24 h x 9 f on both backends.

Every throughput is reported as circuit-hours per second AND as microseconds
per (circuit, hour, frequency) evaluation, so it compares directly with decks
of any frequency count.  Results go to ``results_<date>_gpu.json`` here
(``--out`` to override); an existing file is never overwritten without
``--force``, so a noisy check run cannot clobber a quiet-machine result.
The like-for-like table (``summary.py``) is printed at the end, against the
newest ``results_*_fortran.txt``.

    uv run python tools/bench_vs_fortran/gen_bench_decks.py     # once
    uv run python tools/bench_vs_fortran/bench_gpu.py
"""
import argparse
import datetime
import json
import pathlib
import sys
import time

import numpy as np
import cupy as cp

from voacap_gpu import engine
from voacap_gpu.cuda.driver import (SETUP_CONFIG_DTYPE, SETUP_PARAMS_DTYPE, IonoSlot,
                                     PipelinedSetupLuffy, shared_antenna_tables, shared_iono_tables)
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.gridjob import grid_cells

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summary  # noqa: E402

FREQS = (3.5, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.9, 28.0)
NF = len(FREQS)
TX = (45.42, -75.70)


def params_for(points, hours=range(1, 25)):
    pts = np.asarray(points, np.float64); nc = len(pts); hs = list(hours)
    p = np.zeros(nc * len(hs), SETUP_PARAMS_DTYPE)
    p["tlatd"], p["tlongd"] = TX
    p["rlatd"] = np.tile(pts[:, 0], len(hs)); p["rlongd"] = np.tile(pts[:, 1], len(hs))
    p["hour"] = np.repeat(hs, nc)
    return p


def best(fn, reps=3):
    fn(); cp.cuda.runtime.deviceSynchronize(); b = 1e9
    for _ in range(reps):
        t = time.perf_counter(); fn(); cp.cuda.runtime.deviceSynchronize(); b = min(b, time.perf_counter() - t)
    return b


def rate(n, t):
    """Throughput row: seconds, circuit-hours/s, us per (circuit, hour, freq)."""
    return dict(s=round(t, 4), chs=round(n / t), us_chf=round(1e6 * t / (n * NF), 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=pathlib.Path,
                    help="result JSON (default: results_<date>_gpu.json here)")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    ap.add_argument("--note", default="", help="free-text note stored in the JSON "
                    "(e.g. 'GPU shared, timings noisy')")
    a = ap.parse_args()
    date = datetime.date.today().isoformat()
    out_path = a.out or HERE / f"results_{date}_gpu.json"
    if out_path.exists() and not a.force:
        raise SystemExit(f"{out_path} exists; pass --force or --out")
    rx_file = HERE / "decks" / "rx_points.npy"
    if not rx_file.exists():
        raise SystemExit(f"{rx_file} missing: run gen_bench_decks.py first")

    out = {"date": date, "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
           "note": a.note}
    slot = IonoSlot(redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(120.0)), 7, 120.0)
    iono = shared_iono_tables([slot.iono])
    cfgs = np.zeros(1, SETUP_CONFIG_DTYPE); cfgs[0] = engine.setup_config(engine.System(), method=30, freqs=FREQS)
    ant = shared_antenna_tables([engine.isotropic_cant(power_kw=1.0)])

    # 1) world grid, production path
    lat, lon, *_ = grid_cells(-90, 90, -180, 180, 2.0)
    pw = params_for(np.c_[lat, lon]); n = len(pw)
    res = {}
    for fm in (False, True):
        for chunk, ns in ((32768, 2), (65536, 2)):
            pipe = PipelinedSetupLuffy(chunk, ns, [slot], iono, fast_math=fm)
            t = best(lambda: pipe.run(pw, cfgs, ant, np.zeros(n, np.int32)))
            free, total = cp.cuda.runtime.memGetInfo()
            res[f"fast={fm} chunk={chunk}x{ns}"] = dict(rate(n, t), mem_gb=round((total - free) / 2**30, 1))
            del pipe; cp.get_default_memory_pool().free_all_blocks()
    out["world_grid"] = dict(circuit_hours=n, freqs=NF, runs=res)

    # 2) the Fortran benchmark's 640 circuits x 24 h, same path (like-for-like)
    pts = np.load(rx_file)
    p640 = params_for(pts); n6 = len(p640)
    fw = dict(circuit_hours=n6)
    for mode, fm in (("exact", False), ("fast", True)):
        pipe = PipelinedSetupLuffy(16384, 2, [slot], iono, fast_math=fm)
        t = best(lambda: pipe.run(p640, cfgs, ant, np.zeros(n6, np.int32)), reps=5)
        fw[mode] = rate(n6, t)
        del pipe; cp.get_default_memory_pool().free_all_blocks()
    out["fortran_workload_gpu"] = fw

    # 3) predict() latency, one circuit x 24 h x 9 f
    rx = tuple(pts[0])
    for be in ("cuda", "cpu"):
        engine.predict(TX, rx, freqs=FREQS, month=7, ssn=120, backend=be)
        ts = []
        for _ in range(3 if be == "cpu" else 10):
            t0 = time.perf_counter(); engine.predict(TX, rx, freqs=FREQS, month=7, ssn=120, backend=be)
            ts.append(time.perf_counter() - t0)
        out[f"predict_1x24h_{be}_ms"] = round(1e3 * min(ts), 1)

    print(json.dumps(out, indent=1))
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {out_path}\n")
    f, _ = summary.load()
    print("\n".join(summary.table(f, summary.parse_gpu(out_path))))


if __name__ == "__main__":
    main()
