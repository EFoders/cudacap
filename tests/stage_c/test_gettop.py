"""GETTOP Stage C vs Stage A, on the gettop oracle grid.

How much of a long path the ray flies OVER the E layer.  Five tests in a row,
every one of which can zero the answer, so the coverage asserts below matter
more than the arithmetic: a port that got a comparison backwards would still
return plausible numbers on most rows.

Stage A measures this one ALL BIT-EXACT against the Fortran (615 cells), so
there is no libm slack to inherit and the Stage C bar is simply exact.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.gettop import gettop
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.lpathstate import LpathState
from voacap_gpu.stage_c.state import pack_layer, pack_lpath

from tests.stage_a.dumputil import load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("gettop")

COLUMNS = ["delopt", "freq", "dikm", "hi1", "hi2", "hi3", "fi1", "fi2", "fi3"]

_SRC = r'''
#include "gettop.cuh"

extern "C" __global__ void drive(const LayerState* lay,
                                 const LpathState* lp,
                                 const float* freq, const float* dikm,
                                 float* gmkm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    gmkm[i] = gettop(&lay[i], &lp[i], freq[i], dikm[i]);
}
'''


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_gettop_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    cases = [dict(zip(COLUMNS, (float(x) for x in inp))) for inp in inputs]
    n = len(cases)

    lays, lps = [], []
    exp = np.zeros(n, np.float32)
    for i, v in enumerate(cases):
        lay = LayerState()
        for j in (1, 2, 3):
            lay.hi[1, j] = F32(v[f"hi{j}"])
            lay.fi[1, j] = F32(v[f"fi{j}"])
        lp = LpathState(delopt=F32(v["delopt"]))
        lays.append(pack_layer(lay))
        lps.append(pack_lpath(lp))
        exp[i] = gettop(lay, lp, F32(v["freq"]), F32(v["dikm"]))

    freq = np.array([F32(v["freq"]) for v in cases], np.float32)
    dikm = np.array([F32(v["dikm"]) for v in cases], np.float32)
    out = np.zeros(n, np.float32)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [np.stack(lays).view(np.uint8),
                       np.stack(lps).view(np.uint8), freq, dikm, out])
    assert_bitexact(fetch(dev[4]), exp, "gmkm")

    # Both outcomes must be present, or the five guards are untested.
    assert (exp == 0.0).any(), "no case was rejected by a guard"
    assert (exp > 0.0).any(), "no case produced an over-the-top segment"
