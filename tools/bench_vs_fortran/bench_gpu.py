"""GPU/CPU timings for the final benchmark summary (run on a quiet machine)."""
import sys, time, json, pathlib
import numpy as np
import cupy as cp
from voacap_gpu import engine
from voacap_gpu.cuda.driver import (SETUP_CONFIG_DTYPE, SETUP_PARAMS_DTYPE, IonoSlot,
                                     PipelinedSetupLuffy, shared_antenna_tables, shared_iono_tables)
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.gridjob import grid_cells

HERE = pathlib.Path(__file__).resolve().parent
FREQS = (3.5, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.9, 28.0)
TX = (45.42, -75.70)
out = {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}
slot = IonoSlot(redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(120.0)), 7, 120.0)
iono = shared_iono_tables([slot.iono])
cfgs = np.zeros(1, SETUP_CONFIG_DTYPE); cfgs[0] = engine.setup_config(engine.System(), method=30, freqs=FREQS)
ant = shared_antenna_tables([engine.isotropic_cant(power_kw=1.0)])


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


# 1) world grid, production path
lat, lon, *_ = grid_cells(-90, 90, -180, 180, 2.0)
pw = params_for(np.c_[lat, lon]); n = len(pw)
res = {}
for fm in (False, True):
    for chunk, ns in ((32768, 2), (65536, 2)):
        pipe = PipelinedSetupLuffy(chunk, ns, [slot], iono, fast_math=fm)
        t = best(lambda: pipe.run(pw, cfgs, ant, np.zeros(n, np.int32)))
        free, total = cp.cuda.runtime.memGetInfo()
        res[f"fast={fm} chunk={chunk}x{ns}"] = dict(s=round(t, 3), chs=round(n / t), mem_gb=round((total - free) / 2**30, 1))
        del pipe; cp.get_default_memory_pool().free_all_blocks()
out["world_grid"] = dict(circuit_hours=n, freqs=len(FREQS), runs=res)

# 2) the Fortran benchmark's 640 circuits x 24 h, same path (like-for-like)
pts = np.load(HERE / "decks" / "rx_points.npy")
p640 = params_for(pts); n6 = len(p640)
pipe = PipelinedSetupLuffy(16384, 2, [slot], iono)
t = best(lambda: pipe.run(p640, cfgs, ant, np.zeros(n6, np.int32)), reps=5)
out["fortran_workload_gpu"] = dict(circuit_hours=n6, s=round(t, 4), chs=round(n6 / t))
del pipe; cp.get_default_memory_pool().free_all_blocks()

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
(HERE / "bench_gpu.json").write_text(json.dumps(out, indent=1))
