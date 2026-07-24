"""PRBMUF Stage C vs Stage A — MUF-exceedance probability.

Driven on the tier-B `prbmuf` dump inputs, one packed MufState per thread
(only SIGL/SIGU matter, exactly as the Stage A dump test builds them).
PRBMUF is pure float32 arithmetic — a quartic Horner polynomial, a
reciprocal and a couple of divisions, no transcendentals — so the assert is
bit-exact with no residual at all.  The coverage asserts pin the FSET<=0
step branch, both sides of the median (SIGL vs SIGU), the |Z|<=5 clamp and
the SIG=0.001 floor, mirroring what the Stage A test's docstring claims of
the dump.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.mufstate import MufState
from voacap_gpu.scalar.prbmuf import prbmuf
from voacap_gpu.cuda.state import MUF_DTYPE, pack_muf

from tests.scalar.dumputil import load_dump
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("prbmuf")

_SRC = r'''
#include "prbmuf.cuh"

extern "C" __global__ void drive(const MufState* mufs, const float* fmhz,
                                 const float* fgo, const float* fset,
                                 const int* il, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = prbmuf(&mufs[i], fmhz[i], fgo[i], fset[i], il[i]);
}
'''


# extension cases (fmhz, fgo, fset, il, sigl, sigu): the dump never drives
# SIG below the 0.001 floor, so pin that branch (and re-pin the clamp) here
_EXTRA = [
    (9.9, 10.0, 1000.0, 2, 1e-5, 1e-5),    # below median, SIG floor + clamp
    (10.1, 10.0, 1000.0, 3, 1e-5, 1e-5),   # above median, SIG floor + clamp
    (10.5, 10.0, 10.0, 1, 0.15, 0.15),     # ordinary mid-polynomial probe
]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_prbmuf_dumps.sh")
def test_prbmuf_bit_exact():
    _, inputs, _ = DUMP
    rows = [tuple(r) for r in inputs] + _EXTRA
    n = len(rows)
    fmhz = np.array([np.float32(r[0]) for r in rows], np.float32)
    fgo = np.array([np.float32(r[1]) for r in rows], np.float32)
    fset = np.array([np.float32(r[2]) for r in rows], np.float32)
    il = np.array([int(r[3]) for r in rows], np.int32)
    sigl = np.array([np.float32(r[4]) for r in rows], np.float32)
    sigu = np.array([np.float32(r[5]) for r in rows], np.float32)

    mufs = np.zeros(n, dtype=MUF_DTYPE)
    expected = np.zeros(n, np.float32)
    for k in range(n):
        m = MufState()
        m.sigl[il[k]] = sigl[k]
        m.sigu[il[k]] = sigu[k]
        expected[k] = prbmuf(m, fmhz[k], fgo[k], fset[k], int(il[k]))
        mufs[k] = pack_muf(m)

    mod = compile_test(_SRC)
    out = np.zeros(n, np.float32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [mufs.view(np.uint8), fmhz, fgo, fset, il, out])
    assert_bitexact(fetch(dev[5]), expected, "prbmuf", cases=rows)

    # branch coverage, replicated in float32 exactly as the routine computes:
    z = fmhz - fgo
    live = fset > 0
    assert (~live).any(), "no FSET<=0 step-branch case"
    assert (live & (z <= 0)).any(), "no below-median (SIGL) case"
    assert (live & (z > 0)).any(), "no above-median (SIGU) case"
    # (FGO*SIG)/FSET is the routine's own association; FSET only read when live
    sig_raw = (np.where(z <= 0, fgo * sigl, fgo * sigu)
               / np.where(live, fset, np.float32(1.0)))
    assert (live & (sig_raw < F32(0.001))).any(), "SIG=0.001 floor unreached"
    zs = z / np.maximum(sig_raw, F32(0.001))
    assert (live & (np.abs(zs) > F32(5.0))).any(), "|Z|<=5 clamp unreached"
    assert (live & (np.abs(zs) < F32(5.0))).any()
