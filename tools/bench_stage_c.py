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

TWO COLUMNS, AND THE DIFFERENCE BETWEEN THEM IS THE HEADLINE:

* KERNEL is the compute, timed with the batch already resident.
* TRIP is the same launch plus the 55,652 bytes of per-thread state crossing
  PCIe in both directions.  That is a harness cost, not an intrinsic one: a
  production driver builds the state on-device from card inputs and never
  ships it.

Earlier revisions of this file timed only the round trip and called it
"kernel", which understated Stage C by a factor of five.  At 32,256 threads
the round trip is ~600 ms of which the KERNEL is ~125 ms; the other 475 ms is
1.8 GB up and 1.8 GB back at PCIe speed.

THE UNIT MATTERS.  The Fortran baseline is 17.5 ms per circuit measured on
this machine (tools/bench_scenario.py), and that circuit is 24 hours x 28
FREQUENCIES.  The replay decks fill only 10 frequency slots, so quoting
per-circuit-hour would flatter Stage C by 2.8x.  Everything below is
normalised to one (circuit, hour, frequency) -- 26.0 us for the Fortran on
one core -- which is the unit that compares like for like.

MEASURED RESULT: 0.395 us per evaluation of KERNEL at 32,256 threads and
0.268 us at 129,024 -- **66x and 97x one Fortran core** -- so throughput is
still climbing at 129k rather than saturating at 32k as the round-trip
numbers used to suggest.  The round trip is 1.73-1.87 us per evaluation over
the same range, which is a statement about PCIe and about this harness, not
about the kernel.

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
* **Array-of-structures state layout, for /RAYS/ at least.**  COMMON /RAYS/
  is 30,672 of the 55,652 bytes per thread, and at that stride a warp-wide
  read of one float costs 32 transactions.  It was moved to
  structure-of-arrays (cuda/raysview.cuh); with the kernel code held
  IDENTICAL and only the stride changed, the kernel goes 129.1 -> 124.9 ms
  at 32,256 threads.  **1.03x.**  The same 1.03x appears at 504, 2,016 and
  8,064 threads, so it is not a residency artefact.
* ... and the reason that lever is so short is that the kernel is nowhere
  near bandwidth-bound.  Kernel time barely moves with batch size -- 63 ms
  at 504 threads, 79 ms at 8,064, 127 ms at 32,256: 64x the work for 2x the
  time.  A single circuit-hour costs ~62 ms of serially dependent latency on
  one lane, and everything up to full residency is free because the machine
  is hiding that latency behind other threads.  Coalescing helps a kernel
  that is waiting on bandwidth; this one is waiting on its own dependency
  chain, and the dependency chain is the program.

So the cost is the scalar work itself -- a long, branchy, serially
dependent program per thread -- not an overhead with a lever on it.  The
remaining ideas are real but none is cheap: split LUFFY into several kernels
so each has a smaller live set and a shorter chain, or revisit the
approximations in Tier 3 of the restructuring plan.  Converting the other
fifteen structs to SoA is NOT indicated by the /RAYS/ result.
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


def _time(fn, reps):
    import cupy as cp

    fn()
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    cp.cuda.runtime.deviceSynchronize()
    return (time.perf_counter() - t0) / reps


def _time_kernel(bufs, rays, ipflags, iono, iono_idx, ant, ant_idx, reps,
                 block):
    """(kernel seconds, round-trip seconds) for one batch.

    The two are reported separately because they are nothing like each other:
    at 32k threads the round trip is ~600 ms of which the KERNEL is ~125 ms.
    The rest is the per-thread state crossing PCIe, which a production driver
    would not pay -- see the module docstring.
    """
    import cupy as cp

    from voacap_gpu.stage_c.driver import launch_luffy, stage_luffy

    run, state = stage_luffy(bufs, rays, ipflags, iono, iono_idx, ant,
                             ant_idx, block=block)
    kern = _time(run, reps)
    # the staged batch has to GO before the round trip is timed: holding both
    # resident at 129k threads spills the pool and measures the allocator
    del run, state
    cp.get_default_memory_pool().free_all_blocks()
    trip = _time(lambda: launch_luffy(bufs, rays, ipflags, iono, iono_idx,
                                      ant, ant_idx, block=block), reps)
    cp.get_default_memory_pool().free_all_blocks()
    return kern, trip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[504, 2016, 8064, 32256, 129024])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    from voacap_gpu.stage_c.driver import (CIRCUIT_DTYPE,
                                           STATE_BYTES_PER_THREAD,
                                           circuit_state_sizeof,
                                           shared_antenna_tables,
                                           shared_iono_tables)

    w = _workload()
    base = w["entries"]
    base_rays = w["entries_rays"]
    n0 = len(base)
    base_ipf = w["ipflags"]
    iono = shared_iono_tables(w["ionos"])
    ant = shared_antenna_tables(w["cants"])
    base_ii, base_ai = w["iono_idx"], w["ant_idx"]

    print(f"per-thread state: {STATE_BYTES_PER_THREAD:,} B "
          f"= CircuitState {circuit_state_sizeof():,} B (host dtype "
          f"{CIRCUIT_DTYPE.itemsize:,} B) + SoA /RAYS/ "
          f"{base_rays.dtype.itemsize:,} B")
    tags = w["tags"]
    mix = {t: tags.count(t) for t in sorted(set(tags))}
    print(f"base workload: {n0} real circuit-hours from the replay decks, "
          f"mix {mix}\n")

    # how many frequency slots each circuit-hour actually evaluates
    frel = base["frq"]["frel"]
    nfreq = float((frel[:, 1:13] > 0.0).sum(axis=1).mean())
    print(f"live frequency slots per circuit-hour: {nfreq:.1f} "
          f"(the Fortran baseline circuit runs 28)\n")

    print(f"{'threads':>9} {'kernel ms':>10} {'us/eval':>9} {'vs 1 core':>10} "
          f"| {'trip ms':>9} {'us/eval':>9} {'vs 1 core':>10} {'state GB':>9}")
    print("-" * 88)
    for n in args.sizes:
        reps = max(1, n0 * args.reps // max(n, 1)) if n > 20000 else args.reps
        idx = np.arange(n) % n0
        bufs = base[idx]
        rays = base_rays[idx]
        ipf = base_ipf[idx]
        gb = n * STATE_BYTES_PER_THREAD / 1e9
        if gb > 9.0:
            print(f"{n:>9,}  skipped -- {gb:.1f} GB of state exceeds the "
                  "budget on a 16 GB card")
            continue
        try:
            kern, trip = _time_kernel(bufs, rays, ipf, iono, base_ii[idx],
                                      ant, base_ai[idx], reps, args.block)
        except Exception as exc:                      # OOM or launch failure
            print(f"{n:>9,}  failed: {type(exc).__name__}: {exc}")
            continue
        ke = kern * 1e6 / n / nfreq
        te = trip * 1e6 / n / nfreq
        print(f"{n:>9,} {kern * 1e3:>10.2f} {ke:>9.3f} "
              f"{FORTRAN_US_PER_EVAL / ke:>9.1f}x | {trip * 1e3:>9.2f} "
              f"{te:>9.3f} {FORTRAN_US_PER_EVAL / te:>9.1f}x {gb:>9.2f}")

    print(f"\nFortran baseline {FORTRAN_US_PER_EVAL:.1f} us per "
          "(circuit, hour, frequency), measured, ONE core.")
    print("A 32-core box running the same Fortran is ~32x that, so divide "
          "the speedups by 32 for the honest whole-machine comparison.")
    print("KERNEL is the compute; TRIP adds the per-thread state crossing "
          "PCIe both ways, which a production driver would not pay.")


if __name__ == "__main__":
    main()
