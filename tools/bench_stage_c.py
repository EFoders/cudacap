"""How fast is one-thread-per-circuit-hour, really?

Stage C exists because of a measurement: identical REGMOD arithmetic ran at
3.75 us/lane on 10,000 Stage B lanes and 0.05 us/lane on 1,000,000 -- a 75x
spread that was PyTorch overhead, not physics.  The thesis was that a scalar
CUDA kernel keeping state in registers would collect most of that gap without
needing a million-lane batch to do it.  This measures whether it did.

The workload is real: the 21 replay decks' captured entry states, replicated
to fill a batch.  Every thread runs a complete LUFFY call -- frequency sweep,
hop loop, both path models, the blend -- so this is end-to-end throughput,
not a microbenchmark of one routine.

Two costs are reported separately because they scale differently:

* KERNEL is the compute.
* PACK/UNPACK is the 55,652 bytes of per-thread state crossing PCIe.  That is
  a harness cost here, not an intrinsic one: a production driver builds the
  state on-device from card inputs and never ships it.  Reported so the
  kernel number cannot be mistaken for a whole-pipeline number.

THE UNIT MATTERS.  The Fortran baseline is 17.5 ms per circuit measured on
this machine (tools/bench_scenario.py), and that circuit is 24 hours x 28
FREQUENCIES.  The replay decks fill only 10 frequency slots, so quoting
per-circuit-hour would flatter Stage C by 2.8x.  Everything below is
normalised to one (circuit, hour, frequency) -- 26.0 us for the Fortran on
one core -- which is the unit that compares like for like.

MEASURED RESULT: 1.77-1.9 us per evaluation, saturating from ~32k threads
on.  That is **13.7x one Fortran core** -- and a 32-core box running the
original Fortran is still roughly 2x faster than this.  The one-thread-per
-circuit thesis did NOT deliver the 5-20x it was pitched at.

WHAT IS *NOT* THE BOTTLENECK (all measured, all negative results, recorded
here so nobody re-runs them):

* **Occupancy / register pressure.**  The kernel uses 168 registers, which
  caps residency at 390 threads/SM (~32,760 on the chip) -- suspiciously
  exactly where throughput saturates.  It is a coincidence.  Forcing
  --maxrregcount 128/96/72 raises residency to 43k/57k/76k and the runtime
  does not move (202.7 -> 201.3 ms): flat to within noise.
* **Warp divergence between propagation models.**  A mixed batch puts
  SHORT, LONG and SMOOTH circuit-hours in one warp, and those are different
  MODELS, so the warp should execute all three serially.  Sorting the batch
  so each warp is regime-homogeneous is worth only 1.06x (605 -> 571 ms),
  and pure single-regime batches land in the same band (SHORT 546, LONG
  585, SMOOTH 647 ms).

So the cost is the scalar work itself -- a long, branchy, serially
dependent program per thread -- not an overhead with a lever on it.  The
remaining ideas are real but none is cheap: shrink the 55,652 B per-thread
state (RaysState/IFOB is 30.7 kB of it), split LUFFY into several kernels
so each has a smaller live set, or revisit the approximations in Tier 3 of
the restructuring plan.
"""

import argparse
import copy
import pathlib
import sys
import time

import numpy as np

# the replay harness lives under tests/, which is not an installed package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FORTRAN_MS_PER_CIRCUIT = 17.5               # 24 h x 28 freq, METHOD 30
FORTRAN_US_PER_EVAL = FORTRAN_MS_PER_CIRCUIT * 1000.0 / (24.0 * 28.0)


def _workload():
    """The replay's own case set -- the exact states the correctness proof ran.

    Reused rather than rebuilt: ``_replay`` already walks each deck's 24 hours
    sequentially (COMMON persists between them) and snapshots the Circuit
    before each call, which is what turns a 24-step chain into 24 independent
    (entry, exit) pairs.  Benchmarking anything else would be measuring a
    workload nothing has verified.
    """
    from tests.stage_c.test_luffy_replay import _replay
    return _replay()


def _time_kernel(bufs, ipflags, iono, iono_idx, ant, ant_idx, reps, block):
    import cupy as cp

    from voacap_gpu.stage_c.driver import launch_luffy

    launch_luffy(bufs, ipflags, iono, iono_idx, ant, ant_idx, block=block)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        launch_luffy(bufs, ipflags, iono, iono_idx, ant, ant_idx, block=block)
    cp.cuda.runtime.deviceSynchronize()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[504, 2016, 8064, 32256, 129024])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    from voacap_gpu.stage_c.driver import (CIRCUIT_DTYPE, circuit_state_sizeof,
                                           shared_antenna_tables,
                                           shared_iono_tables)

    w = _workload()
    base = w["entries"]
    n0 = len(base)
    base_ipf = w["ipflags"]
    iono = shared_iono_tables(w["ionos"])
    ant = shared_antenna_tables(w["cants"])
    base_ii, base_ai = w["iono_idx"], w["ant_idx"]

    print(f"per-thread state: {circuit_state_sizeof():,} B "
          f"(host dtype {CIRCUIT_DTYPE.itemsize:,} B)")
    tags = w["tags"]
    mix = {t: tags.count(t) for t in sorted(set(tags))}
    print(f"base workload: {n0} real circuit-hours from the replay decks, "
          f"mix {mix}\n")

    # how many frequency slots each circuit-hour actually evaluates
    frel = base["frq"]["frel"]
    nfreq = float((frel[:, 1:13] > 0.0).sum(axis=1).mean())
    print(f"live frequency slots per circuit-hour: {nfreq:.1f} "
          f"(the Fortran baseline circuit runs 28)\n")

    print(f"{'threads':>9} {'kernel ms':>10} {'us/circ-hr':>11} "
          f"{'us/eval':>9} {'vs 1 core':>10} {'state GB':>9}")
    print("-" * 64)
    for n in args.sizes:
        reps = max(1, n0 * args.reps // max(n, 1)) if n > 20000 else args.reps
        idx = np.arange(n) % n0
        bufs = base[idx]
        ipf = base_ipf[idx]
        gb = n * CIRCUIT_DTYPE.itemsize / 1e9
        if gb > 9.0:
            print(f"{n:>9,}  skipped -- {gb:.1f} GB of state exceeds the "
                  "budget on a 16 GB card")
            continue
        try:
            sec = _time_kernel(bufs, ipf, iono, base_ii[idx], ant,
                               base_ai[idx], reps, args.block)
        except Exception as exc:                      # OOM or launch failure
            print(f"{n:>9,}  failed: {type(exc).__name__}: {exc}")
            continue
        us = sec * 1e6 / n
        per_eval = us / nfreq
        speedup = FORTRAN_US_PER_EVAL / per_eval
        print(f"{n:>9,} {sec * 1e3:>10.2f} {us:>11.3f} {per_eval:>9.3f} "
              f"{speedup:>9.1f}x {gb:>9.2f}")

    print(f"\nFortran baseline {FORTRAN_US_PER_EVAL:.1f} us per "
          "(circuit, hour, frequency), measured, ONE core.")
    print("A 32-core box running the same Fortran is ~32x that, so divide "
          "the last column by 32 for the honest whole-machine comparison.")
    print("Kernel time only; see the module docstring on pack/unpack.")


if __name__ == "__main__":
    main()
