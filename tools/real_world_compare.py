"""LEGACY replay benchmark. This is NOT the current GPU-vs-Fortran comparison.

======================================================================
LEGACY.  The like-for-like benchmark is tools/bench_vs_fortran/ (see the
README there; results are in results_<date>_*).  On 2026-11-15 it measured
32-thread voacapl at 63,700 circuit-hours/s and the GPU world grid at 292k
(bit-exact) / 371k (fast-math) circuit-hours/s, i.e. 4.6x / 5.8x.
======================================================================

What this script measures: the Stage C LUFFY kernel REPLAYING pre-built
circuits.  These are the 21 golden oracle decks' captured LUFFY entry states
(tests/cuda/test_luffy_replay.py, 9 frequencies per circuit-hour), tiled up
to ``--n`` circuit-hours.  The timed region is host gather + both PCIe legs +
kernel.  Circuit SETUP (deck parsing, REDMAP, GEOM, the ionosphere chain) is
NOT timed, because the entry states exist before the clock starts.

Why its "x vs Fortran" ratios (it printed ~9x / ~15x vs 32 cores) are not
like-for-like:

* The Fortran side is NOT measured here.  It is a stored constant,
  ``FORT_32CORE_MS_CH = 0.97 / 24`` ms per circuit-hour, taken from
  tools/results/fortran_baseline.md.  That is voacapl on bench28.dat, a deck
  with 28 FREQUENCIES (this replay has 9), timed per circuit WITH one process
  start-up each (~3 ms per process).  Per circuit-hour, this charges Fortran
  for ~3x the frequencies the GPU computes.
* The GPU side skips the setup chain, which voacapl's time includes.

At the end, the script prints the like-for-like table if tools/bench_vs_fortran
has result files.  That table uses the same workload definition, runs setup on
the device, and normalises both sides per (circuit, hour, frequency), so the
two can be read side by side.  History is kept: the script and its stored
baseline still run unchanged.

---------------------------------------------------------------------------
Original description.  Real-world runtime: Stage C GPU vs the measured
Fortran, in MILLISECONDS.

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

# LEGACY stored baseline (tools/results/fortran_baseline.md): bench28.dat,
# 28 frequencies per circuit-hour, one process start-up per circuit.
FORT_1CORE_MS_CH = 16.68 / 24.0        # measured 2026-07-23, ms per circuit-hour
FORT_32CORE_MS_CH = 0.97 / 24.0
FORT_BASELINE_NFREQ = 28
BENCH_VS_FORTRAN = pathlib.Path(__file__).resolve().parent / "bench_vs_fortran"
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


def _like_for_like_lines():
    """The tools/bench_vs_fortran summary table, or a pointer if absent."""
    import importlib.util
    pointer = ("the current benchmark is tools/bench_vs_fortran/ "
               "(README.md there)")
    try:
        spec = importlib.util.spec_from_file_location(
            "_bvf_summary", BENCH_VS_FORTRAN / "summary.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        f, g = mod.load()
    except Exception as e:                       # missing or unreadable files
        return [f"no like-for-like results loaded ({type(e).__name__}: {e}); "
                + pointer]
    if f is None or g is None:
        return ["no like-for-like result files found; " + pointer]
    return mod.table(f, g)


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

    from voacap_gpu.cuda import driver
    from tests.cuda.test_luffy_replay import _replay

    w = _replay()
    n0 = len(w["entries"])
    iono = driver.shared_iono_tables(w["ionos"])
    ant = driver.shared_antenna_tables(w["cants"])

    print("LEGACY replay benchmark: pre-built circuits, setup not timed, "
          "Fortran side a stored 28-frequency baseline.")
    print("The like-for-like benchmark is tools/bench_vs_fortran/ (its table "
          "is printed at the end).\n")
    fr = w["entries"]["frq"]["frel"][:, 1:12]         # FREL(1..11)
    nfreq = float(np.mean((fr > 0).sum(axis=1)))
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
    print("\nLEGACY head-to-head (whole workload wall-clock). NOT like-for-like:"
          f"\n  the GPU replays {nfreq:g}-freq pre-built circuits with setup "
          f"untimed; Fortran is the stored {FORT_BASELINE_NFREQ}-freq baseline "
          "incl. process start-up")
    print(f"  pipe+fastmath vs Fortran 1 core : {fort1 / (t_pipef * 1e3):6.1f}x")
    print(f"  pipe+fastmath vs Fortran 32 core: {fort32 / (t_pipef * 1e3):6.2f}x")
    print(f"  pipelined     vs Fortran 32 core: {fort32 / (t_pipe * 1e3):6.2f}x"
          f"  (bit-exact)")

    # the same timings per (circuit, hour, frequency) evaluation
    def us_chf(total_ms, nf):
        return 1e3 * total_ms / (n * nf)
    print(f"\nper (circuit, hour, frequency) evaluation, us (replay {nfreq:g} "
          f"freqs, legacy baseline {FORT_BASELINE_NFREQ}):")
    print(f"  GPU replay pipelined (bit-exact): {us_chf(t_pipe * 1e3, nfreq):7.3f}")
    print(f"  GPU replay pipe+fastmath        : {us_chf(t_pipef * 1e3, nfreq):7.3f}")
    print(f"  Fortran 32 core (legacy stored) : "
          f"{us_chf(fort32, FORT_BASELINE_NFREQ):7.3f}")

    print("\n" + "\n".join(_like_for_like_lines()))


if __name__ == "__main__":
    main()
