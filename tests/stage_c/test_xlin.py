"""XLIN Stage C vs Stage A — the worked example for TABLE-taking routines.

Shows the layout rule in action: Stage A's 1-based (rows, cols) numpy arrays
pass to the device flat row-major with an explicit ncols, padding intact.
The tables are deliberately non-monotonic so all three segment classes
(increasing / flat / decreasing) and both endpoint clamps are exercised, and
the test asserts each class was actually hit rather than trusting the
generator.
"""

import numpy as np

from voacap_gpu.stage_a.xlin import xlin

from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "xlin.cuh"

extern "C" __global__ void drive(const float* x, const float* xn,
                                 const float* yn, int ntab, int ncols,
                                 const int* k, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = xlin(x[i], xn, yn, ntab, k[i], ncols);
}
'''

NTAB, NCOLS = 12, 4              # 1-based: rows 1..12 used, cols 1..3 used


def _tables(seed=5):
    rng = np.random.default_rng(seed)
    xn = np.zeros((NTAB + 1, NCOLS), np.float32)
    yn = np.zeros((NTAB + 1, NCOLS), np.float32)
    for kcol in range(1, NCOLS):
        steps = rng.choice([-1.5, 0.0, 0.0, 1.0, 2.5], NTAB - 1)
        xn[1:, kcol] = np.cumsum(np.r_[rng.uniform(0, 3), steps]
                                 ).astype(np.float32)
        yn[1:, kcol] = rng.uniform(-50, 50, NTAB).astype(np.float32)
    return xn, yn


def test_xlin_bit_exact():
    xn, yn = _tables()
    rng = np.random.default_rng(6)
    n = 30_000
    x = rng.uniform(xn[1:].min() - 3, xn[1:].max() + 3, n).astype(np.float32)
    k = rng.integers(1, NCOLS, n).astype(np.int32)
    # exact table hits, both clamps, and flat-segment probes
    x[:4] = np.float32([xn[3, 1], xn[1, 2] - 10.0, xn[NTAB, 2] + 10.0,
                        xn[5, 3]])
    k[:4] = [1, 2, 2, 3]

    expected = np.array([xlin(xi, xn, yn, NTAB, ki) for xi, ki
                         in zip(x[:4000], k[:4000])], dtype=np.float32)

    mod = compile_test(_SRC)
    out = np.zeros_like(x)
    dev = launch_rows(mod.get_function("drive"), n,
                      [x, xn.ravel(), yn.ravel(), np.int32(NTAB),
                       np.int32(NCOLS), k, out])
    got = fetch(dev[6])
    assert_bitexact(got[:4000], expected, "xlin", cases=x)

    # the generator must actually have produced all three segment classes
    d = xn[1:NTAB, 1:] - xn[2:NTAB + 1, 1:]
    assert (d < 0).any() and (d == 0).any() and (d > 0).any()
