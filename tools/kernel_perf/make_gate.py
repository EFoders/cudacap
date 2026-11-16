"""Regenerate the bit-exact gate digests in tests/cuda/test_bitexact_gate.py.

    uv run python tools/kernel_perf/make_gate.py            # rewrite digests
    uv run python tools/kernel_perf/make_gate.py --check    # compare only
    uv run python tools/kernel_perf/make_gate.py --runs 3 --only world,random

Every workload is run ``--runs`` times (default 3), each time with freshly
built rigs, and the digests are written ONLY if every run agrees: a digest
that is not reproducible is not a gate.  Strict (``fast_math=False``) and
fast-math digests are generated separately.

Regenerate deliberately.  The gate exists so that kernel performance work
does NOT move these; ``--check`` is what to run after a change.
"""

import argparse
import pathlib
import pprint
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.cuda import test_bitexact_gate as G  # noqa: E402

FAST_SET = ("world", "random", "icepac", "ant13", "refusals")
BEGIN = "# BEGIN GENERATED DIGESTS\n"
END = "# END GENERATED DIGESTS\n"


def compute(names, fast_math, runs):
    out = {}
    for name in names:
        t0 = time.perf_counter()
        first = None
        for r in range(runs):
            d = G.run_workload(name, fast_math)
            if first is None:
                first = d
            elif d != first:
                bad = sorted(k for k in d if d[k] != first[k])
                raise SystemExit(f"{name} fast_math={fast_math}: run {r} "
                                 f"differs from run 0 on {bad}; NOT stable")
        out[name] = first
        print(f"  {name:10s} fast_math={fast_math!s:5}  {runs} runs agree  "
              f"({time.perf_counter() - t0:5.1f} s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--check", action="store_true",
                    help="compare with the stored digests; write nothing")
    ap.add_argument("--only", default="",
                    help="comma-separated workload subset")
    ap.add_argument("--no-fast", action="store_true")
    ap.add_argument("--define", action="append", default=[],
                    help="extra -D for the kernel build (experiments; "
                         "only with --check)")
    args = ap.parse_args()
    if args.define:
        if not args.check:
            raise SystemExit("--define is for --check only: digests are "
                             "generated from the production build")
        import voacap_gpu.cuda.compile as C
        C.STRICT_OPTS = C.STRICT_OPTS + tuple(f"-D{d}" for d in args.define)

    names = [n for n in G.WORKLOADS
             if not args.only or n in args.only.split(",")]
    strict = compute(names, False, args.runs)
    fast = {} if args.no_fast else compute(
        [n for n in names if n in FAST_SET], True, args.runs)

    if args.check:
        bad = 0
        for stored, got, tag in ((G.DIGESTS, strict, "strict"),
                                 (G.FAST_DIGESTS, fast, "fast")):
            for name, d in got.items():
                want = stored.get(name)
                if want is None:
                    print(f"{tag:6s} {name:10s} NO STORED DIGEST")
                    bad += 1
                    continue
                moved = sorted(k for k in d if want.get(k) != d[k])
                print(f"{tag:6s} {name:10s} "
                      + ("identical" if not moved else f"MOVED: {moved}"))
                bad += bool(moved)
        raise SystemExit(1 if bad else 0)

    # merge into what is stored, so --only refreshes a subset
    allstrict = dict(G.DIGESTS)
    allstrict.update(strict)
    allfast = dict(G.FAST_DIGESTS)
    allfast.update(fast)
    path = pathlib.Path(G.__file__)
    src = path.read_text(encoding="utf-8")
    a, b = src.index(BEGIN) + len(BEGIN), src.index(END)
    block = ("DIGESTS = " + pprint.pformat(allstrict, width=79, sort_dicts=True)
             + "\nFAST_DIGESTS = "
             + pprint.pformat(allfast, width=79, sort_dicts=True) + "\n")
    path.write_text(src[:a] + block + src[b:], encoding="utf-8", newline="\n")
    print(f"wrote {len(allstrict)} strict + {len(allfast)} fast workloads "
          f"to {path}")


if __name__ == "__main__":
    main()
