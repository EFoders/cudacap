"""One setup + one LUFFY launch on a representative chunk, for Nsight Compute.

    ncu --kernel-name luffy_batch --launch-count 1 --set full \\
        -o tools/kernel_perf/ncu/world_full \\
        uv run python tools/kernel_perf/profile_one.py --kind world --n 65536

``--kind world`` takes the first ``--n`` circuit-hours of the 2-degree world
grid in natural order (see bench_kernel.py); ``--kind random`` a random draw
from it.  ``--fast`` profiles the fast-math build.  Nothing is timed here:
ncu serialises and replays the kernel, so its wall time means nothing.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import bench_kernel as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="world", choices=("world", "random"))
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--offset", type=int, default=0,
                    help="world only: first row of the chunk")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    import cupy as cp
    from voacap_gpu.cuda.driver import SetupLuffy
    slots, iono, cfgs, ant, p, ai = B.build(args.kind)
    a = args.offset if args.kind == "world" else 0
    p, ai = p[a:a + args.n], ai[a:a + args.n]
    rig = SetupLuffy(len(p), slots, iono, ant, fast_math=args.fast)
    res, _ = rig.run(p, cfgs, ai)
    cp.cuda.runtime.deviceSynchronize()
    print("rows", len(p), "status!=0:", int(np.count_nonzero(res.status)))


if __name__ == "__main__":
    main()
