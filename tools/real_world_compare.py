"""Real-world runtime: Stage C GPU vs the measured Fortran, in MILLISECONDS.

Not speedup factors -- wall-clock ms for a whole workload, which is what a
user waits on.

MEMORY DISCIPLINE (learned the hard way): ``LuffyArena(n)`` allocates device
buffers at the FULL workload width -- ~55 KB per circuit-hour, so ~10 GB at
193k -- while ``PipelinedLuffy`` only ever holds ``nstreams * chunk`` of them.
So this never builds a full-n arena, never holds two big GPU objects at once,
and frees each before building the next.  The serial baseline is timed on ONE
saturating 32,256-hour batch and multiplied out; the pipeline is timed on the
real multi-chunk workload, which is the case its overlap actually helps.

Three GPU configurations, each timed end to end (host gather + both PCIe legs
+ kernel), against the Fortran baselines in tools/results/fortran_baseline.md:

  serial      LuffyArena, per-batch, bit-exact
  pipelined   PipelinedLuffy over the whole workload, bit-exact
  pipe+fast   pipelined AND -DVOACAP_FAST_MATH (not bit-exact; 99.34% of cells
              exact, no discrete decision moves -- test_fastmath.py)
"""

import argparse
import gc
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FORT_1CORE_MS_CH = 16.68 / 24.0        # measured, ms per circuit-hour
FORT_32CORE_MS_CH = 0.97 / 24.0
BATCH = 32256                          # a saturating batch for this kernel


def _free():
    import cupy as cp
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _best(fn, reps):
    import cupy as cp
    fn()
    cp.cuda.runtime.deviceSynchronize()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        cp.cuda.runtime.deviceSynchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def _mem_mb():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    return (total - free) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=161280,
                    help="circuit-hours in the pipelined workload (multiple "
                         "of the 32,256 batch so overlap engages)")
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()

    import cupy as cp

    from voacap_gpu.stage_c import driver
    from tests.stage_c.test_luffy_replay import _replay

    w = _replay()
    n0 = len(w["entries"])
    iono = driver.shared_iono_tables(w["ionos"])
    ant = driver.shared_antenna_tables(w["cants"])

    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    n = args.n
    nb = max(1, round(n / BATCH))
    n = nb * BATCH
    print(f"device {name}; workload {n:,} circuit-hours = {nb} x {BATCH:,} "
          f"({n / 24:,.0f} circuit-equivalents)\n")

    def slab(size):
        idx = np.arange(size) % n0
        return (w["entries"][idx], w["entries_rays"][idx], w["ipflags"][idx],
                w["iono_idx"][idx], w["ant_idx"][idx])

    # ---- correctness on a SMALL n: pipeline must equal serial ----
    sb = slab(BATCH)
    ar = driver.LuffyArena(BATCH, iono, ant)
    r_ser = ar.run(sb[0], sb[1], sb[2], sb[3], sb[4])
    t_batch = _best(lambda: ar.run(sb[0], sb[1], sb[2], sb[3], sb[4]),
                    args.reps)
    del ar
    _free()

    pp = driver.PipelinedLuffy(BATCH, 2, iono, ant)
    r_pip = pp.run(sb[0], sb[1], sb[2], sb[3], sb[4])

    def _eq(a, b):
        for f in a.dtype.names:
            if not np.array_equal(np.ascontiguousarray(a[f]).view(np.uint8),
                                  np.ascontiguousarray(b[f]).view(np.uint8)):
                return f
        return None
    son_bad = _eq(r_ser.son, r_pip.son)
    stat_ok = np.array_equal(r_ser.status, r_pip.status)
    print(f"correctness (on {BATCH:,}): pipelined vs serial son "
          f"{'OK' if son_bad is None else son_bad}, status "
          f"{'OK' if stat_ok else 'DIFFERS'}")
    if son_bad is not None or not stat_ok:
        raise SystemExit("pipeline is NOT bit-identical to serial -- abort")
    del pp, r_ser, r_pip
    _free()
    print(f"  (fast-math deviation pinned by test_fastmath.py; peak device "
          f"mem so far {_mem_mb():.0f} MB)\n")

    # ---- pipelined, whole workload, bit-exact ----
    big = slab(n)
    pipe = driver.PipelinedLuffy(BATCH, 2, iono, ant)
    t_pipe = _best(lambda: pipe.run(big[0], big[1], big[2], big[3], big[4]),
                   args.reps)
    peak_pipe = _mem_mb()
    del pipe
    _free()

    # ---- pipelined + fast math ----
    pipef = driver.PipelinedLuffy(BATCH, 2, iono, ant, fast_math=True)
    t_pipef = _best(lambda: pipef.run(big[0], big[1], big[2], big[3], big[4]),
                    args.reps)
    del pipef
    _free()

    # serial over the whole workload = per-batch time x number of batches
    t_serial = t_batch * nb

    def row(label, total_ms, exact):
        per_ckt = total_ms / n * 24
        tag = "bit-exact" if exact else "fast-math"
        print(f"  {label:<20} {total_ms:9.1f} ms   {per_ckt:7.3f} ms/circuit"
              f"   [{tag}]")

    print(f"WALL-CLOCK for {n:,} circuit-hours "
          f"(host gather + PCIe + kernel, best of {args.reps}):")
    row("GPU serial", t_serial * 1e3, True)
    row("GPU pipelined", t_pipe * 1e3, True)
    row("GPU pipe+fastmath", t_pipef * 1e3, False)
    print()
    fort1 = FORT_1CORE_MS_CH * n
    fort32 = FORT_32CORE_MS_CH * n
    row("Fortran 1 core", fort1, True)
    row("Fortran 32 cores", fort32, True)

    print(f"\n  pipeline device footprint: {peak_pipe:.0f} MB for the whole "
          f"{n:,}-hour workload (2 streams x {BATCH:,}, independent of n)")
    print("\nhead-to-head (whole workload wall-clock):")
    print(f"  pipe+fastmath vs Fortran 1 core : {fort1 / (t_pipef * 1e3):6.1f}x")
    print(f"  pipe+fastmath vs Fortran 32 core: {fort32 / (t_pipef * 1e3):6.2f}x")
    print(f"  pipelined     vs Fortran 32 core: {fort32 / (t_pipe * 1e3):6.2f}x"
          f"  (bit-exact)")


if __name__ == "__main__":
    main()
