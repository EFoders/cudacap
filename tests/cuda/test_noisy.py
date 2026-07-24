"""NOISY Stage C vs Stage A — the two-variable Fourier map evaluator.

No golden dump exists for NOISY alone (its oracle coverage is through the
ANOIS1 dump, which only reaches KJ 1..6), so the case set is generated the
way the Stage B tests generate theirs: random domain-valid latitudes and
longitudes across ALL map selectors KJ 1..8, which exercises the /TWO/
overlay (FAKP slices, FAKMAP, HMYM), both ABP sources and both series-limit
arms (LM/LN 29/15 vs 15/10).  Expected values come from Stage A in-process.

Unlike Stage B — whose torch SIN/COS are different functions and sit at a
measured non-zero floor — Stage C evaluates its transcendentals in binary64
through f32math.cuh exactly as Stage A does, so the bar here is bit-exact.
"""

import numpy as np

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.noisy import noisy
from voacap_gpu.scalar.redmap import redmap

from tests.scalar.dumputil import COEFFS_DIR
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "noisy.cuh"

extern "C" __global__ void drive(const float* fakp, const float* fakmap,
                                 const float* hmym, const float* fakabp,
                                 const float* abmap, const int* kj,
                                 const float* xla, const float* ceg,
                                 float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = noisy(fakp, fakmap, hmym, fakabp, abmap, kj[i], xla[i], ceg[i]);
}
'''


def test_noisy_bit_exact():
    st = redmap(COEFFS_DIR, "CCIR", 3, F32(100.0))
    rng = np.random.default_rng(17)
    n = 1200
    kj = rng.integers(1, 9, n).astype(np.int32)
    xla = rng.uniform(-89.0, 89.0, n).astype(np.float32)
    ceg = rng.uniform(0.0, 360.0, n).astype(np.float32)
    # pole-adjacent and meridian edges, on both series-limit arms
    kj[:6] = [1, 8, 7, 8, 6, 8]
    xla[:6] = np.float32([89.9, -89.9, 0.0, 0.0, 45.0, -45.0])
    ceg[:6] = np.float32([0.0, 0.0, 359.9, 180.0, 0.1, 359.9])

    expected = np.array(
        [noisy(st, int(k), x, c) for k, x, c in zip(kj, xla, ceg)],
        dtype=np.float32)

    mod = compile_test(_SRC)
    out = np.zeros(n, np.float32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [np.ascontiguousarray(st.fakp).ravel(),
                       np.ascontiguousarray(st.fakmap).ravel(),
                       np.ascontiguousarray(st.hmym).ravel(),
                       np.ascontiguousarray(st.fakabp).ravel(),
                       np.ascontiguousarray(st.abmap).ravel(),
                       kj, xla, ceg, out])
    got = fetch(dev[8])
    assert_bitexact(got, expected, "noisy",
                    cases=list(zip(kj.tolist(), xla.tolist(), ceg.tolist())))

    # every map selector, hence every overlay source (FAKP x6, FAKMAP, HMYM)
    # and both LM/LN limit arms, must actually be in the set
    assert set(kj.tolist()) == set(range(1, 9)), f"KJ only reaches {set(kj)}"
    assert (kj <= 6).any() and (kj == 7).any() and (kj == 8).any()
