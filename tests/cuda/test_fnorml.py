"""FNORML Stage C vs Stage A — the minimal worked example of the pattern.

One thread per case, expected values computed by calling Stage A in-process,
bit-exact assert.  The case set covers both signs, the 5-sigma clamp, exact
zero (the 110 path), and the dense middle where the polynomial's rounding
actually gets exercised.
"""

import numpy as np

from voacap_gpu.scalar.fnorml import fnorml

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "fnorml.cuh"

extern "C" __global__ void drive(const float* ypx, float* px, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    px[i] = fnorml(ypx[i]);
}
'''


def test_fnorml_bit_exact():
    rng = np.random.default_rng(23)
    x = rng.uniform(-8.0, 8.0, 100_000).astype(np.float32)
    x[:6] = np.float32([0.0, -0.0, 5.0, -5.0, 4.9999995, -4.9999995])
    expected = np.array([fnorml(v) for v in x[:5000]], dtype=np.float32)

    mod = compile_test(_SRC)
    out = np.zeros_like(x)
    dev = launch_rows(mod.get_function("drive"), len(x), [x, out])
    got = fetch(dev[1])
    assert_bitexact(got[:5000], expected, "fnorml", cases=x)

    # branch coverage: both signs and both sides of the clamp are in the set
    assert (x[:5000] < 0).any() and (x[:5000] > 0).any()
    assert (np.abs(x[:5000]) > 5.0).any() and (np.abs(x[:5000]) < 5.0).any()
