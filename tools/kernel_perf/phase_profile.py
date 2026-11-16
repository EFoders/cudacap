"""Per-phase clock64() profile of the thread LUFFY kernel (no Nsight needed).

    uv run python tools/kernel_perf/phase_profile.py                # both
    uv run python tools/kernel_perf/phase_profile.py --kind world --n 65536

Compiles ``luffy_batch`` with ``-DVOACAP_LUFFY_PROFILE`` (cuda/kernels/
luffyprof.cuh), runs one SetupLuffy batch, and prints for each phase:

* share   -- the phase's clock64() cycles summed over threads, as a share of
             the TOTAL phase (the whole LUFFY body).  Phases nest (HOPS holds
             FDIST/INMUF/ALLM), so the leaf rows are what add up; "other" is
             TOTAL minus the leaves (SmoothStore init, loop control).
* cyc/ent -- mean cycles per entry
* ent/thr -- entries per thread (e.g. frequencies x passes)
* active  -- mean active-lane fraction (popc(__activemask()) / 32) at entry:
             1.0 means the whole warp arrives together; low values are
             divergence (lanes on another path, or already finished).

It also checks that the instrumented build's answers are bit-identical to the
production build's on the same batch (the counters must not perturb anything)
and reports the instrumented kernel's slowdown.
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import bench_kernel as B  # noqa: E402

PHASES = ["AREA", "NOISE1", "FINDF", "FDIST", "INMUF", "ALLM", "ES", "LONG",
          "NOISE2", "RELBIL", "POST", "AFTER", "HOPS", "TOTAL",
          # sub-phases: of AREA, then of LONG (not leaves)
          "a.LECDEN", "a.GENION", "a.FOBBY", "a.ALOSFV", "a.SETLNG",
          "a.SIGDIS", "l.FINDF2", "l.GMLOSS", "l.SELTXR", "l.LNGPAT",
          "KERNEL"]
LEAVES = ["AREA", "NOISE1", "FINDF", "FDIST", "INMUF", "ALLM", "ES", "LONG",
          "NOISE2", "RELBIL", "POST", "AFTER"]
NPH = len(PHASES)

_EXTRA = r'''
extern "C" __global__ void luffy_prof_set(unsigned long long* buf, int n) {
    vc_prof_buf = buf;
    vc_prof_n = n;
}
'''


def profiled_module(fast_math=False):
    from voacap_gpu.cuda import driver
    from voacap_gpu.cuda.compile import load
    opts = ("-DVOACAP_LUFFY_PROFILE",)
    if fast_math:
        opts += ("-DVOACAP_FAST_MATH",)
    return load(driver._KERNEL_SRC + _EXTRA, opts)


def run(kind, n, offset=0, fast_math=False, reps=3):
    import cupy as cp
    from voacap_gpu.cuda.driver import SetupLuffy
    slots, iono, cfgs, ant, p, ai = B.build(kind)
    if kind == "world":
        p, ai = p[offset:offset + n], ai[offset:offset + n]
    else:
        p, ai = p[:n], ai[:n]
    n = len(p)

    # production answer and kernel time
    rig = SetupLuffy(n, slots, iono, ant, fast_math=fast_math)
    rig._luffy_fn = B._Timed(rig._luffy_fn)
    ref = rig.run(p, cfgs, ai)
    rig._luffy_fn.take_ms()
    t_prod = min((rig.run(p, cfgs, ai), rig._luffy_fn.take_ms())[1]
                 for _ in range(reps))

    mod = profiled_module(fast_math)
    fn = mod.get_function("luffy_batch")
    buf = cp.zeros((3 * NPH, n), np.uint64)
    mod.get_function("luffy_prof_set")((1,), (1,), (buf, np.int32(n)))
    cp.cuda.runtime.deviceSynchronize()
    rig._luffy_fn = B._Timed(fn)
    t_prof = []
    for _ in range(reps):
        buf.fill(0)
        got = rig.run(p, cfgs, ai)
        t_prof.append(rig._luffy_fn.take_ms())
    counts = cp.asnumpy(buf).astype(np.float64)
    mod.get_function("luffy_prof_set")((1,), (1,), (buf, np.int32(0)))
    cp.cuda.runtime.deviceSynchronize()

    same = all(np.array_equal(np.ascontiguousarray(ref[0].son[f]).view(np.uint8),
                              np.ascontiguousarray(got[0].son[f]).view(np.uint8))
               for f in ref[0].son.dtype.names) and \
        np.array_equal(ref[0].status, got[0].status)
    del rig
    cp.get_default_memory_pool().free_all_blocks()

    cyc = counts[:NPH].sum(axis=1)
    act = counts[NPH:2 * NPH].sum(axis=1)
    ent = counts[2 * NPH:].sum(axis=1)
    tot = cyc[PHASES.index("TOTAL")]
    print(f"\n=== {kind} n={n} offset={offset} fast_math={fast_math} ===")
    print(f"production kernel {t_prod:.1f} ms, instrumented "
          f"{min(t_prof):.1f} ms; instrumented answers bit-identical: {same}")
    print(f"{'phase':8s} {'share':>7s} {'cyc/ent':>10s} {'ent/thr':>8s} "
          f"{'active':>7s}")
    rows = {}
    for i, ph in enumerate(PHASES):
        share = cyc[i] / tot if tot else 0.0
        cpe = cyc[i] / ent[i] if ent[i] else 0.0
        af = act[i] / (32.0 * ent[i]) if ent[i] else 0.0
        rows[ph] = dict(share=share, cyc_per_entry=cpe,
                        entries_per_thread=ent[i] / n, active=af)
        print(f"{ph:8s} {100 * share:6.1f}% {cpe:10.0f} {ent[i] / n:8.2f} "
              f"{af:7.3f}")
    leaf = sum(cyc[PHASES.index(ph)] for ph in LEAVES)
    print(f"{'other':8s} {100 * (tot - leaf) / tot:6.1f}%")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="both",
                    choices=("both", "world", "random"))
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    kinds = ("world", "random") if args.kind == "both" else (args.kind,)
    for k in kinds:
        run(k, args.n, args.offset, args.fast)


if __name__ == "__main__":
    main()
