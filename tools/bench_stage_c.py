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
  PCIe in both directions.

Earlier revisions of this file timed only the round trip and called it
"kernel", which understated Stage C by a factor of five.  At 32,256 threads
the round trip is ~600 ms of which the KERNEL is ~125 ms; the other 475 ms is
1.8 GB up and 1.8 GB back at PCIe speed.

AND MOST OF THOSE 1.8 GB NEVER NEEDED TO GO: ``--trip``
--------------------------------------------------------
That per-thread state is almost entirely SCRATCH -- /RAYS/ IFOB alone is
30,504 bytes, 55% of it, and LUFFY's own FOBBY fills it during setup.  Only
3,228 bytes are read before they are written, and only 1,700 are wanted back
(/SON/, XLUF, JMODE and the status).  ``stage_c/driver.py`` declares that
input set -- measured field by field, pinned by
``tests/stage_c/test_input_set.py`` -- and ``--trip`` A/Bs the whole state
both ways against it, with and without a resident ``LuffyArena``.  Both
columns return bit-identical results; the only thing that differs is how much
of the bus they use.  Measured (min of 5, ms, on a card another process was
also using, so the KERNEL column is roughly 3x its idle value):

    circuit-hours     504     8,064    32,256
    transfer, full   18.8     134.9     514.7
    transfer, arena  18.6      27.1      98.3
    speedup          1.01x     4.97x     5.24x

**5.2x off the bus at 32,256**, and nothing at 504 -- at that size the whole
batch is 28 MB and what is left is launch latency and the shared tables,
which neither column avoids.  The round trip moves less than the transfer
does because the kernel is most of it: 901 -> 472 ms, 1.91x, and about 2.9x
if the kernel is the ~127 ms an idle card gives.  The remaining ~98 ms is
roughly a third host-side gather, a third the two PCIe hops, and a third the
scatter/gather kernels; pinned host buffers are the obvious next lever and
have not been tried.

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

WHAT *IS* A LEVER, AND ONLY BELOW FULL OCCUPANCY: ``--warp``
------------------------------------------------------------
``cuda/luffy_warp.cuh`` shortens the chain itself -- one WARP per
circuit-hour, lane L on frequency L+1, so the twelve DO-265 trips run side by
side over one ionosphere and one reflectrix.  It is bit-identical to
``luffy_batch`` on all 504 replay circuit-hours, every field of every struct
(``tests/stage_c/test_luffy_warp.py``).  ``--warp`` A/Bs the two, kernel
only.  Measured at 4 warps/block:

    circuit-hours     504    2,016    4,032    8,064   32,256
    thread  us/eval  12.87    3.52     1.89     1.01     0.40
    warp    us/eval   2.22    1.34     1.16     1.04     0.98
    speedup          5.81x   2.63x    1.63x    0.97x    0.42x

The setup phase (SELMOD, ANOIS1, the ionosphere build, SETLNG, SIGDIS) does
not parallelise and is 11-12% of the kernel -- measured by giving the kernel
a runtime DO-265 trip count, so the register allocation is held identical --
which caps the design at 5.2-5.4x.  It reaches 5.8x on the empty machine and
loses on the full one, and both are the same fact: the thread kernel hides
its latency with thousands of INDEPENDENT circuit-hours, and when there are
enough of those it does not need a shorter chain, while the warp kernel has
given up 20 of every 32 lanes and 254 registers against 168 to get one.  The
crossover is around 8,000 circuit-hours.  Neither kernel replaces the other.
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


def _time_warp(bufs, rays, ipflags, iono, iono_idx, ant, ant_idx, reps, wpb,
               groups=1):
    """Kernel-only seconds for ``luffy_batch_warp`` — one WARP per
    circuit-hour, lane L on frequency L+1 (cuda/luffy_warp.cuh).

    ``groups=2`` packs TWO circuit-hours into each warp, sixteen lanes each,
    lifting lane use from 12/32 to 24/32.  It costs a little at small batch
    (the setup then splits over 16 lanes instead of 32) and wins at large,
    which is the whole point: it moves the crossover against the thread
    kernel a long way right.
    """
    import cupy as cp

    from voacap_gpu.stage_c.driver import stage_luffy

    run, state = stage_luffy(bufs, rays, ipflags, iono, iono_idx, ant,
                             ant_idx, warp=True, warps_per_block=wpb,
                             groups=groups)
    t = _time(run, reps)
    del run, state
    cp.get_default_memory_pool().free_all_blocks()
    return t


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


def _best(fn, reps):
    """Min of `reps`, not the mean: this machine's GPU is often shared."""
    import cupy as cp

    fn()
    cp.cuda.runtime.deviceSynchronize()
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def _trip_ab(args, base, base_rays, base_ipf, base_ii, base_ai, iono, ant,
             n0, nfreq):
    """The PCIe A/B: whole state both ways vs the measured input set.

    The transfer columns are timed WITHOUT the kernel, because that is the
    only part this comparison changes and because the kernel dominates the
    round trip whenever another process is using the card:

    * UP / DOWN   ``stage_luffy`` and ``fetch_luffy``, whole state --
                  55,652 bytes a thread each way.
    * ARENA UP/DOWN  ``LuffyArena.upload`` and ``.results()`` -- 3,228 bytes
                  up and 1,700 back, into buffers that already exist, so the
                  ~1.8 GB of scratch allocation and memset and the shared
                  tables are not paid per batch.
    * TRIP        the two ends plus the launch, ``launch_luffy`` against
                  ``LuffyArena.run``.

    Both return the SAME numbers; ``tests/stage_c/test_input_set.py`` is what
    says so.
    """
    import cupy as cp

    from voacap_gpu.stage_c.driver import (RESULT_BYTES_PER_THREAD,
                                           STATE_BYTES_PER_THREAD,
                                           UPLOAD_BYTES_PER_THREAD,
                                           LuffyArena, fetch_luffy,
                                           launch_luffy, stage_luffy)

    print(f"{'circ-hours':>11} {'kernel':>8} | {'up':>7} {'down':>7} "
          f"{'xfer':>7} | {'a-up':>7} {'a-down':>7} {'a-xfer':>7} "
          f"| {'xfer x':>7} | {'trip':>8} {'a-trip':>8} {'trip x':>7}")
    print("-" * 104)
    for n in args.sizes:
        reps = max(2, n0 * args.reps // max(n, 1)) if n > 20000 else args.reps
        idx = np.arange(n) % n0
        bufs, rays = base[idx], base_rays[idx]
        ipf, ii, ai = base_ipf[idx], base_ii[idx], base_ai[idx]
        try:
            up = _best(lambda: stage_luffy(bufs, rays, ipf, iono, ii, ant, ai,
                                           block=args.block), reps)
            run, state = stage_luffy(bufs, rays, ipf, iono, ii, ant, ai,
                                     block=args.block)
            kern = _best(run, reps)
            down = _best(lambda: fetch_luffy(state), reps)
            del run, state
            cp.get_default_memory_pool().free_all_blocks()
            trip = _best(lambda: launch_luffy(bufs, rays, ipf, iono, ii, ant,
                                              ai, block=args.block), reps)
            cp.get_default_memory_pool().free_all_blocks()
            arena = LuffyArena(n, iono, ant, block=args.block)
            aup = _best(lambda: arena.upload(bufs, rays, ipf, ii, ai), reps)
            arena.launch()
            adown = _best(arena.results, reps)
            atrip = _best(lambda: arena.run(bufs, rays, ipf, ii, ai), reps)
            del arena
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as exc:
            print(f"{n:>11,}  failed: {type(exc).__name__}: {exc}")
            continue
        x, ax = up + down, aup + adown
        print(f"{n:>11,} {kern * 1e3:>8.1f} | {up * 1e3:>7.1f} "
              f"{down * 1e3:>7.1f} {x * 1e3:>7.1f} | {aup * 1e3:>7.1f} "
              f"{adown * 1e3:>7.1f} {ax * 1e3:>7.1f} | {x / ax:>6.2f}x "
              f"| {trip * 1e3:>8.1f} {atrip * 1e3:>8.1f} "
              f"{trip / atrip:>6.2f}x")
    print(f"\nper circuit-hour: {STATE_BYTES_PER_THREAD:,} B of state, of "
          f"which {UPLOAD_BYTES_PER_THREAD:,} B is INPUT and "
          f"{RESULT_BYTES_PER_THREAD:,} B is RESULT.  Times are the MIN over "
          f"{args.reps} reps, in ms.")
    print("At small N the arena wins nothing: the transfer is a few tens of "
          "MB and what is left is launch latency and the shared tables, "
          "which neither column can avoid.")


def _warp_ab(args, base, base_rays, base_ipf, base_ii, base_ai, iono, ant,
             n0, nfreq):
    """The A/B: thread-per-circuit-hour vs warp-per-circuit-hour, kernel-only.

    Both columns time the SAME staged batch with the state already resident,
    so the difference is the kernel and nothing else.  The two are bit-
    identical on all 504 replay circuit-hours
    (``tests/stage_c/test_luffy_warp.py``), so this is a like-for-like
    comparison of two ways of computing the same numbers.
    """
    import cupy as cp

    from voacap_gpu.stage_c.driver import STATE_BYTES_PER_THREAD

    print(f"{'circ-hours':>11} {'thread':>9} {'us/ev':>7} "
          f"| {'warp':>9} {'us/ev':>7} | {'packed':>9} {'us/ev':>7} "
          f"| {'best':>7} {'vs 1 core':>10}")
    print("-" * 88)
    for n in args.sizes:
        reps = max(1, n0 * args.reps // max(n, 1)) if n > 20000 else args.reps
        idx = np.arange(n) % n0
        bufs, rays = base[idx], base_rays[idx]
        ipf, ii, ai = base_ipf[idx], base_ii[idx], base_ai[idx]
        if n * STATE_BYTES_PER_THREAD / 1e9 > 9.0:
            print(f"{n:>11,}  skipped -- state exceeds the budget")
            continue
        try:
            kern, _ = _time_kernel(bufs, rays, ipf, iono, ii, ant, ai, reps,
                                   args.block)
            cp.get_default_memory_pool().free_all_blocks()
            wk = _time_warp(bufs, rays, ipf, iono, ii, ant, ai, reps,
                            args.wpb)
            cp.get_default_memory_pool().free_all_blocks()
            pk = _time_warp(bufs, rays, ipf, iono, ii, ant, ai, reps,
                            args.wpb, groups=2)
        except Exception as exc:
            print(f"{n:>11,}  failed: {type(exc).__name__}: {exc}")
            continue
        ke = kern * 1e6 / n / nfreq
        we = wk * 1e6 / n / nfreq
        pe = pk * 1e6 / n / nfreq
        best = min(ke, we, pe)
        print(f"{n:>11,} {kern * 1e3:>9.2f} {ke:>7.3f} "
              f"| {wk * 1e3:>9.2f} {we:>7.3f} | {pk * 1e3:>9.2f} {pe:>7.3f} "
              f"| {ke / best:>6.2f}x {FORTRAN_US_PER_EVAL / best:>9.1f}x")
    print(f"\n{args.wpb} warps/block.  KERNEL-only, batch already resident.")
    print("'packed' is the warp kernel with VC_WARP_GROUPS=2 -- two "
          "circuit-hours per warp, 24 of 32 lanes busy.")
    print("'best' is the fastest of the three against the thread kernel; "
          "'vs 1 core' uses that best.")
    print(f"Fortran baseline {FORTRAN_US_PER_EVAL:.1f} us per "
          "(circuit, hour, frequency), measured, ONE core.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[504, 2016, 8064, 32256, 129024])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--warp", action="store_true",
                    help="A/B the warp-per-circuit-hour kernel against the "
                         "thread-per-circuit-hour one, kernel-only")
    ap.add_argument("--trip", action="store_true",
                    help="A/B the ROUND TRIP: whole state both ways vs the "
                         "measured input set, with and without an arena")
    ap.add_argument("--wpb", type=int, default=4,
                    help="warps per block for the warp kernel")
    args = ap.parse_args()

    from voacap_gpu.stage_c.driver import (CIRCUIT_DTYPE,
                                           RESULT_BYTES_PER_THREAD,
                                           STATE_BYTES_PER_THREAD,
                                           UPLOAD_BYTES_PER_THREAD,
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

    print(f"of that state, {UPLOAD_BYTES_PER_THREAD:,} B is INPUT and "
          f"{RESULT_BYTES_PER_THREAD:,} B is RESULT — "
          f"{STATE_BYTES_PER_THREAD / UPLOAD_BYTES_PER_THREAD:.1f}x and "
          f"{STATE_BYTES_PER_THREAD / RESULT_BYTES_PER_THREAD:.1f}x less "
          "than the whole thing\n")

    if args.trip:
        _trip_ab(args, base, base_rays, base_ipf, base_ii, base_ai, iono, ant,
                 n0, nfreq)
        return

    if args.warp:
        _warp_ab(args, base, base_rays, base_ipf, base_ii, base_ai, iono, ant,
                 n0, nfreq)
        return

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
