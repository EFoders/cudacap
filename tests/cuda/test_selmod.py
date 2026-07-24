"""SELMOD Stage C vs Stage A — controlling sample-area selection.

Pure integer branching on KFX and FI comparisons, so JMODE is exact by
construction; the test's job is pinning the branch structure.  Inputs are
the selmod golden dump's 143 cases plus a directed grid that guarantees
every arm both ways (KFX 1..5, the KFX==2 E-critical tiebreak, the KFX>=3
F2 decision with |DELFI| on both sides of 0.01 MHz) independent of what the
dump happens to contain.  Expected values come from Stage A in-process.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.selmod import selmod
from voacap_gpu.cuda.state import LAYER_DTYPE, pack_layer

from tests.scalar.dumputil import load_dump
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("selmod")

_SRC = r'''
#include "selmod.cuh"

extern "C" __global__ void drive(const LayerState* s, int* jmode, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    jmode[i] = selmod(&s[i]);
}
'''


def _directed():
    """(kfx, fi11, fi12, fi13, fi31, fi33) covering every arm both ways."""
    cases = [(1, 5.0, 4.0, 3.0, 8.0, 7.0)]
    # KFX==2: E critical decides, both ways plus the tie (-> area 1)
    cases += [(2, 3.0, 2.0, 0.0, 0.0, 0.0),
              (2, 2.0, 3.0, 0.0, 0.0, 0.0),
              (2, 2.5, 2.5, 0.0, 0.0, 0.0)]
    for kfx in (3, 4, 5):
        # F2 controls, both signs of DELFI
        cases += [(kfx, 0.0, 0.0, 0.0, 6.0, 5.0),
                  (kfx, 0.0, 0.0, 0.0, 5.0, 6.0)]
        # F2 tie within 0.01 MHz -> E critical decides, both ways + exact tie
        cases += [(kfx, 3.0, 0.0, 2.0, 6.0, 6.005),
                  (kfx, 2.0, 0.0, 3.0, 6.005, 6.0),
                  (kfx, 2.5, 0.0, 2.5, 6.0, 6.0)]
    return cases


def _state(kfx, fi11, fi12, fi13, fi31, fi33):
    s = LayerState(kfx=int(kfx))
    s.fi[1, 1], s.fi[1, 2], s.fi[1, 3] = F32(fi11), F32(fi12), F32(fi13)
    s.fi[3, 1], s.fi[3, 3] = F32(fi31), F32(fi33)
    return s


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_selmod_dumps.sh")
def test_selmod_bit_exact():
    _, inputs, _ = DUMP
    raw = [tuple(float(x) for x in inp[:6]) for inp in inputs]
    raw += _directed()
    n = len(raw)
    buf = np.empty(n, dtype=LAYER_DTYPE)
    expected = np.empty(n, np.int32)
    seen = {"kfx1": 0, "kfx2_rx": 0, "kfx2_tx": 0,
            "tie_rx": 0, "tie_tx": 0, "f2_rx": 0, "f2_tx": 0}
    for i, c in enumerate(raw):
        s = _state(*c)
        buf[i] = pack_layer(s)
        expected[i] = selmod(s)
        kfx = int(c[0])
        if kfx == 1:
            seen["kfx1"] += 1
        elif kfx == 2:
            seen["kfx2_rx" if expected[i] == 2 else "kfx2_tx"] += 1
        else:
            tie = abs(F32(c[4]) - F32(c[5])) < F32(0.01)
            arm = "tie" if tie else "f2"
            seen[f"{arm}_rx" if expected[i] == 3 else f"{arm}_tx"] += 1

    mod = compile_test(_SRC)
    out = np.zeros(n, np.int32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [buf.view(np.uint8), out])
    assert_bitexact(fetch(dev[1]), expected, "selmod.jmode", cases=raw)

    for arm, c in seen.items():
        assert c, f"branch never reached: {arm} ({seen})"
