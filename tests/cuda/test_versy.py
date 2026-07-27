"""VERSY Stage C vs Stage A — the geographic coefficient series.

One thread per sample area, expected values from calling Stage A in-process,
bit-exact assert on all six GAMMA sets.  VERSY is the single most expensive
routine in the host setup chain, and its precision contract is delicate: X,
SX, G, CX and T are binary64 while Y, GOB, FJ, AB and GAMMA stay REAL*4, the
FJ*Y product is computed in single precision *before* widening, and GAMMA is
narrowed to binary32 on every accumulation.  A port that gets any of those
wrong fails here.

The case set sweeps both hemispheres and the full longitude range (so the
negative-longitude wrap at label 100 is live), includes near-pole latitudes
where cos(lat)^m underflows binary32 — the reason the Fortran uses real*8 —
and covers every IZ 1..6, which selects the Rawer-dip vs plain-latitude arm
and different IKIM loop extents.
"""

import numpy as np

from voacap_gpu import engine
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.versy import versy
from voacap_gpu.scalar.virtim import virtim

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "versy.cuh"

extern "C" __global__ void drive(const int* ikim, const float* ab,
                                 const float* coord, float* gamma, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g[7];
    versy(ikim, 6, ab, coord[i * 3 + 0], coord[i * 3 + 1], coord[i * 3 + 2],
          1, 6, g);
    for (int k = 0; k < 7; ++k)
        gamma[i * 7 + k] = g[k];
}
'''


def _cases():
    rng = np.random.default_rng(5)
    lat = np.concatenate([
        np.array([0.0, 89.99, -89.99, 89.0, -89.0, 45.0, -45.0], np.float32),
        rng.uniform(-89.9, 89.9, 400).astype(np.float32)])
    lon = np.concatenate([
        np.array([0.0, -0.001, 179.99, -179.99, 90.0, -90.0, 359.9], np.float32),
        rng.uniform(-180.0, 180.0, 400).astype(np.float32)])
    dip = np.concatenate([
        np.array([0.0, 80.0, -80.0, 12.0, -12.0, 60.0, -60.0], np.float32),
        rng.uniform(-85.0, 85.0, 400).astype(np.float32)])
    d2r = np.float32(0.01745329251)
    return np.stack([lat * d2r, lon * d2r, dip * d2r], axis=1).astype(np.float32)


def test_versy_bit_exact():
    st = redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(100))
    ab = virtim(st, np.float32(12.0))
    coord = _cases()
    n = len(coord)

    expected = np.zeros((n, 7), np.float32)
    for i, (clat, clong, gmdip) in enumerate(coord):
        expected[i] = versy(st, ab, clat, clong, gmdip, 1, 6)

    ikim = np.ascontiguousarray(st.ikim, np.int32)
    assert ikim.shape[1] == 6, f"IKIM is {ikim.shape}, kernel assumes ncols=6"

    mod = compile_test(_SRC)
    out = np.zeros((n, 7), np.float32)
    dev = launch_rows(mod.get_function("drive"), n, [
        ikim.ravel(), np.ascontiguousarray(ab, np.float32),
        coord.ravel(), out.ravel()])
    got = fetch(dev[3]).reshape(n, 7)

    assert_bitexact(got.ravel(), expected.ravel(), "versy gamma", cases=coord)

    # the series must actually be exercised, not all-zero
    assert np.abs(expected[:, 1:]).max() > 0.0
    # both longitude arms of label 100/105 are present
    assert (coord[:, 1] < 0).any() and (coord[:, 1] >= 0).any()
