"""F2DIS Stage C vs Stage A — the F2 MUF decile spread.

Two launches: one against the real F2D(16,6,6) table REDMAP loads, and one
against a zeroed table to reach the "no ionospheric data" classical-15%
fallback.  The table indices come out of Fortran INT() truncation, so a
1-ULP difference picks a different cell entirely — which is what makes this
worth a bit-exact test rather than a tolerance.

The sweep covers both hemispheres (the J+3 southern shift), all three SSN
bands, both decile arms (FREQ <= FMUF and above), the ICC > 6 wrap, and the
|CL| range that saturates the ICL clamp at both ends.
"""

import numpy as np

from voacap_gpu import engine
from voacap_gpu.scalar.f2dis import f2dis
from voacap_gpu.scalar.redmap import redmap

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "f2dis.cuh"

extern "C" __global__ void drive(const float* f2d, const float* a,
                                 float* sig, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sig[i] = f2dis(f2d, a[i * 5 + 0], a[i * 5 + 1], a[i * 5 + 2],
                   a[i * 5 + 3], a[i * 5 + 4]);
}
'''


class _FakeState:
    """Just the F2D the routine reads, so the zero-table branch is reachable."""

    def __init__(self, f2d):
        self.f2d = f2d


def _args():
    rng = np.random.default_rng(17)
    n = 4000
    fmuf = rng.uniform(1.0, 40.0, n).astype(np.float32)
    ssn = np.concatenate([
        np.array([0.0, 50.0, 50.0001, 100.0, 100.0001, 200.0], np.float32),
        rng.uniform(0.0, 200.0, n - 6).astype(np.float32)])
    clat = np.concatenate([
        np.array([0.0, -1e-8, 1e-8, 1.5707, -1.5707, 0.7], np.float32),
        rng.uniform(-1.5707, 1.5707, n - 6).astype(np.float32)])
    # a handful of near-zero MUFs so SIG lands on its 0.001 floor on both
    # the table branch (SIG = |FMUF - ratio*FMUF|/1.28) and the fallback
    fmuf[300:306] = np.float32([1e-5, 1e-4, 2e-4, 5e-5, 1e-6, 3e-5])
    freq = rng.uniform(1.0, 40.0, n).astype(np.float32)
    freq[:200] = fmuf[:200]                      # the FREQ == FMUF boundary
    abb = np.concatenate([
        np.array([0.0, 4.0, 20.0, 23.9, 24.0, 30.0], np.float32),
        rng.uniform(0.0, 24.0, n - 6).astype(np.float32)])
    return np.stack([fmuf, ssn, clat, freq, abb], axis=1).astype(np.float32)


def _run(mod, f2d_flat, a):
    out = np.zeros(len(a), np.float32)
    dev = launch_rows(mod.get_function("drive"), len(a),
                      [f2d_flat, a.ravel(), out])
    return fetch(dev[2])


def test_f2dis_bit_exact():
    st = redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(100))
    a = _args()
    mod = compile_test(_SRC)

    real = np.ascontiguousarray(st.f2d, np.float32)
    assert real.shape == (16, 6, 6), real.shape
    assert real[0, 0, 0] > 0.0, "the real table must take the data branch"

    expected = np.array([f2dis(st, *row) for row in a], np.float32)
    got = _run(mod, real.ravel(), a)
    assert_bitexact(got, expected, "f2dis (real table)", cases=a)

    # ... and the no-data fallback
    zero = np.zeros((16, 6, 6), np.float32)
    zst = _FakeState(zero)
    expected0 = np.array([f2dis(zst, *row) for row in a], np.float32)
    got0 = _run(mod, zero.ravel(), a)
    assert_bitexact(got0, expected0, "f2dis (no-data fallback)", cases=a)

    # branch coverage
    assert (a[:, 2] <= 0).any() and (a[:, 2] > 0).any(), "both hemispheres"
    assert (a[:, 1] <= 50).any() and (a[:, 1] > 100).any(), "SSN bands"
    assert (a[:, 3] <= a[:, 0]).any() and (a[:, 3] > a[:, 0]).any(), "decile arms"
    assert (expected == np.float32(0.001)).any(), "the sig floor is reached"
