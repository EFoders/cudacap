"""What -DVOACAP_FAST_MATH buys, kernel-only, on the replay workload.

The accuracy side of the same switch is measured and asserted in
``tests/stage_c/test_fastmath.py``; this is only the clock.  It reuses the
replay's own 504 circuit-hours (the exact states the correctness proof runs)
replicated to fill each batch, and times the KERNEL with the state already
resident -- the round trip is PCIe, not compute, and would dilute the answer.

The A/B is INTERLEAVED and reported as the MINIMUM over trials: this machine
is shared with other work, and the minimum is the contention-free estimate for
each side.  Both sides compile the same translation unit with the same strict
options (--fmad=false, --prec-div, --prec-sqrt); the only difference is the
one -D.

MEASURED (RTX 5080, thread-per-circuit-hour kernel, 168 registers both ways)::

    circuit-hours      504     8,064    32,256
    strict      ms  127.72     75.09    121.49
    fast        ms   42.74     31.65     64.21
    speedup          2.99x     2.37x     1.89x

Absolute times inflate a lot when something else is on the card (32,256 has
been seen at 399 ms strict / 214 ms fast), but the RATIO barely moves --
1.77x to 1.89x at 32,256 across every run, either side of the 1.93x measured
independently.  Quote the ratio, not the milliseconds.

The win SHRINKS with batch size, which is the tell for what it is: at full
occupancy the machine is already hiding the f64 latency behind thousands of
independent circuit-hours, so what is left is the f64 units' throughput
(1/64 of f32 on this card).  Below full occupancy the kernel is latency-bound
on one thread's own serial dependency chain, and the binary64 transcendentals
are the long links in it -- so fast math shortens the chain as well, and pays
nearly 3x.

Usage:  uv run python tools/bench_fastmath.py [--sizes 504 8064 32256]
"""

import argparse
import pathlib
import sys
import time

import numpy as np

# the replay harness lives under tests/, which is not an installed package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _time(fn, reps):
    import cupy as cp

    fn()
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    cp.cuda.runtime.deviceSynchronize()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[504, 8064, 32256])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    import cupy as cp

    import voacap_gpu.stage_c.driver as drv
    from tests.stage_c.test_fastmath import fast_kernel_module, fast_math
    from tests.stage_c.test_luffy_replay import _replay

    w = _replay()
    base, base_rays = w["entries"], w["entries_rays"]
    n0 = len(base)
    iono = drv.shared_iono_tables(w["ionos"])
    ant = drv.shared_antenna_tables(w["cants"])

    for tag, mod in (("strict", drv.load_kernel()),
                     ("fast", fast_kernel_module())):
        f = mod.get_function("luffy_batch")
        print(f"{tag:>7}: {f.num_regs} registers, "
              f"{f.local_size_bytes} B local")

    print(f"\n{'circ-hours':>11} {'strict ms':>10} {'fast ms':>10} "
          f"{'speedup':>9}")
    print("-" * 43)
    for n in args.sizes:
        idx = np.arange(n) % n0
        call = (base[idx], base_rays[idx], w["ipflags"][idx], iono,
                w["iono_idx"][idx], ant, w["ant_idx"][idx])
        reps = args.reps if n <= 8064 else max(1, args.reps // 2)
        run_s, st_s = drv.stage_luffy(*call, block=args.block)
        with fast_math():
            run_f, st_f = drv.stage_luffy(*call, block=args.block)
        ts, tf = [], []
        for _ in range(args.trials):
            ts.append(_time(run_s, reps))
            tf.append(_time(run_f, reps))
        del run_s, st_s, run_f, st_f
        cp.get_default_memory_pool().free_all_blocks()
        a, b = min(ts), min(tf)
        print(f"{n:>11,} {a * 1e3:>10.2f} {b * 1e3:>10.2f} {a / b:>8.2f}x")

    print("\nKERNEL only, batch already resident, minimum of "
          f"{args.trials} interleaved trials.")
    print("Accuracy cost of the same switch: tests/stage_c/test_fastmath.py")


if __name__ == "__main__":
    main()
