"""DATR Stage C vs Stage A — the /DATR/ Gaussian-integration constants.

datr.cuh is pure data (the folded 20-point Gauss-Legendre rule GETHP sums
over, plus the NT/NPL/XNPL/TWDIV startup scalars).  The kernel reads every
slot back through the device tables and the test asserts bit-identity with
stage_a/datr.py — a wrong or mistyped literal anywhere in the tables fails
loudly here rather than as a diffuse GETHP mismatch.
"""

import numpy as np

from voacap_gpu.scalar import datr

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "datr.cuh"

extern "C" __global__ void dump_datr(float* xt, float* wt, float* scal,
                                     int n) {
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    for (int i = 0; i < 21; ++i) {
        xt[i] = DATR_XT[i];
        wt[i] = DATR_WT[i];
    }
    scal[0] = (float)DATR_NT;
    scal[1] = (float)DATR_NPL;
    scal[2] = DATR_XNPL;
    scal[3] = DATR_TWDIV;
}
'''


def test_datr_tables_bit_exact():
    mod = compile_test(_SRC)
    xt = np.zeros(21, np.float32)
    wt = np.zeros(21, np.float32)
    scal = np.zeros(4, np.float32)
    dev = launch_rows(mod.get_function("dump_datr"), 1, [xt, wt, scal])
    assert_bitexact(fetch(dev[0]), datr.XT, "DATR_XT")
    assert_bitexact(fetch(dev[1]), datr.WT, "DATR_WT")
    assert_bitexact(fetch(dev[2]),
                    np.array([datr.NT, datr.NPL, datr.XNPL, datr.TWDIV],
                             dtype=np.float32),
                    "DATR scalars")
    # the Stage A constants really are the frozen startup values GETHP assumes
    assert datr.NT == 20 and datr.NPL == 1
    assert datr.XNPL == np.float32(1.0) and datr.TWDIV == np.float32(0.5)
