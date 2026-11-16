"""LUFFY kernel benchmark: world grid + random order, kernel-only and end to end.

    uv run python tools/kernel_perf/bench_kernel.py                 # all
    uv run python tools/kernel_perf/bench_kernel.py --reps 5 --no-fast
    uv run python tools/kernel_perf/bench_kernel.py --json out.json

Workloads (both the shapes the review measured):

* ``world``  -- the 2-degree world grid (16,471 cells) x 24 UTC hours =
  395,304 circuit-hours, Ottawa transmitter, CCIR July SSN 120, METHOD 30,
  nine frequencies, isotropic antennas, in the natural hour-major order a
  grid job submits.
* ``random`` -- 65,536 circuit-hours drawn from the same grid x hours, in
  RANDOM order (the worst case for warp divergence).

Timings, each best-of-``--reps`` after one warm-up:

* ``e2e``    -- wall time of ``PipelinedSetupLuffy(65536, 2).run`` (the
  production path), host staging and result extraction included.
* ``kernel`` -- the sum of CUDA-event times of the ``luffy_batch`` launches
  alone, measured on a ``SetupLuffy(65536)`` rig chunk by chunk (setup kernel
  timed separately and reported as ``setup``).

The GPU is shared, so timings are noisy: compare best-of-N runs made under
comparable load, and re-run before believing a small difference.  The script
also prints ``luffy_batch``'s register count and local-memory size.
"""

import argparse
import json
import sys
import time

import numpy as np

FREQS = (3.5, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.9, 28.0)
CHUNK = 65536


def apply_defines(defs):
    """Experiment hook: append ``-D<def>`` to the strict NVRTC options.

    The production option set is untouched unless this is called; the CuPy
    cache keys on the options, so each variant compiles once."""
    if not defs:
        return
    import voacap_gpu.cuda.compile as C
    C.STRICT_OPTS = C.STRICT_OPTS + tuple(f"-D{d}" for d in defs)


def build(kind):
    from voacap_gpu import engine
    from voacap_gpu.cuda.driver import (SETUP_CONFIG_DTYPE, SETUP_PARAMS_DTYPE,
                                        IonoSlot, shared_antenna_tables,
                                        shared_iono_tables)
    from voacap_gpu.gridjob import grid_cells
    from voacap_gpu.scalar.redmap import redmap

    slot = IonoSlot(redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(120.0)),
                    7, 120.0)
    iono = shared_iono_tables([slot.iono])
    cfgs = np.zeros(1, SETUP_CONFIG_DTYPE)
    cfgs[0] = engine.setup_config(engine.System(), method=30, freqs=FREQS)
    ant = shared_antenna_tables([engine.isotropic_cant(power_kw=1.0)])
    lat, lon, *_ = grid_cells(-90, 90, -180, 180, 2.0)
    nc = len(lat)
    rlat, rlon = np.tile(lat, 24), np.tile(lon, 24)
    hour = np.repeat(np.arange(1, 25), nc)
    if kind == "random":
        idx = np.random.default_rng(0).choice(nc * 24, CHUNK, replace=False)
        rlat, rlon, hour = rlat[idx], rlon[idx], hour[idx]
    n = len(hour)
    p = np.zeros(n, SETUP_PARAMS_DTYPE)
    p["tlatd"], p["tlongd"] = 45.42, -75.70
    p["rlatd"], p["rlongd"], p["hour"] = rlat, rlon, hour
    return [slot], iono, cfgs, ant, p, np.zeros(n, np.int32)


class _Timed:
    """Wrap a cupy RawKernel function so each launch is bracketed by events
    on the current stream."""

    def __init__(self, fn):
        self.fn, self.ev = fn, []

    def __getattr__(self, k):
        return getattr(self.fn, k)

    def __call__(self, *a, **k):
        import cupy as cp
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        self.fn(*a, **k)
        e.record()
        self.ev.append((s, e))

    def take_ms(self):
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
        t = sum(cp.cuda.get_elapsed_time(s, e) for s, e in self.ev)
        self.ev = []
        return t


def bench_e2e(wl, fast_math, reps):
    import cupy as cp
    from voacap_gpu.cuda.driver import PipelinedSetupLuffy
    slots, iono, cfgs, ant, p, ai = wl
    pipe = PipelinedSetupLuffy(CHUNK, 2, slots, iono, fast_math=fast_math)
    pipe.run(p, cfgs, ant, ai)
    cp.cuda.runtime.deviceSynchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        res, _m = pipe.run(p, cfgs, ant, ai)
        cp.cuda.runtime.deviceSynchronize()
        ts.append(time.perf_counter() - t0)
    del pipe
    cp.get_default_memory_pool().free_all_blocks()
    return ts, res


def bench_kernel(wl, fast_math, reps):
    import cupy as cp
    from voacap_gpu.cuda.driver import SetupLuffy
    slots, iono, cfgs, ant, p, ai = wl
    rig = SetupLuffy(CHUNK, slots, iono, ant, fast_math=fast_math)
    rig._setup_fn = _Timed(rig._setup_fn)
    rig._luffy_fn = _Timed(rig._luffy_fn)
    fn = rig._luffy_fn.fn
    n = len(p)
    chunks = [(a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]

    def once():
        for a, b in chunks:
            pp, aa = p[a:b], ai[a:b]
            if b - a < CHUNK:
                pad = CHUNK - (b - a)
                pp = np.concatenate([pp, np.repeat(pp[:1], pad)])
                aa = np.concatenate([aa, np.repeat(aa[:1], pad)])
            rig._upload(pp, cfgs, aa).launch()
        return rig._setup_fn.take_ms(), rig._luffy_fn.take_ms()

    once()
    su, lu = [], []
    for _ in range(reps):
        s, k = once()
        su.append(s)
        lu.append(k)
    info = dict(regs=int(fn.num_regs), local=int(fn.local_size_bytes),
                shared=int(fn.shared_size_bytes))
    del rig
    cp.get_default_memory_pool().free_all_blocks()
    return su, lu, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-fast", action="store_true")
    ap.add_argument("--only-fast", action="store_true")
    ap.add_argument("--workloads", default="world,random")
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--skip-kernel", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--define", action="append", default=[],
                    help="extra -D for the kernel build (experiments)")
    ap.add_argument("--carveout", type=int, default=-1,
                    help="experiment: luffy_batch's preferred shared-memory "
                         "carveout percent (0 = max L1)")
    ap.add_argument("--maxrreg", type=int, default=0,
                    help="experiment: --maxrregcount for the kernel build")
    args = ap.parse_args()
    apply_defines(args.define)
    if args.maxrreg:
        import voacap_gpu.cuda.compile as C
        C.STRICT_OPTS = C.STRICT_OPTS + (f"--maxrregcount={args.maxrreg}",)

    fms = [False, True]
    if args.no_fast:
        fms = [False]
    if args.only_fast:
        fms = [True]
    out = dict(label=args.label, rows=[])
    for kind in args.workloads.split(","):
        wl = build(kind)
        n = len(wl[4])
        for fm in fms:
            if args.carveout >= 0:
                from voacap_gpu.cuda.driver import load_kernel
                load_kernel(fast_math=fm).get_function(
                    "luffy_batch").preferred_shared_memory_carveout =                     args.carveout
            row = dict(workload=kind, n=n, fast_math=fm)
            if not args.skip_kernel:
                su, lu, info = bench_kernel(wl, fm, args.reps)
                row.update(setup_ms=min(su), kernel_ms=min(lu),
                           kernel_all_ms=lu, **info)
            if not args.skip_e2e:
                ts, _res = bench_e2e(wl, fm, args.reps)
                row.update(e2e_s=min(ts), e2e_all_s=ts)
            out["rows"].append(row)
            msg = f"{kind:6s} n={n:7d} fast_math={fm!s:5}"
            if "kernel_ms" in row:
                msg += (f" | setup {row['setup_ms']:7.1f} ms | LUFFY kernel "
                        f"{row['kernel_ms']:7.1f} ms ({n / row['kernel_ms'] * 1e3:,.0f}"
                        f" ch/s) regs {row['regs']} local {row['local']} B")
            if "e2e_s" in row:
                msg += (f" | e2e {row['e2e_s']:.3f} s "
                        f"({n / row['e2e_s']:,.0f} ch/s) "
                        f"all={['%.3f' % t for t in row['e2e_all_s']]}")
            print(msg, flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
