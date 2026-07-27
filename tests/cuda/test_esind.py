"""ESIND Stage C vs Stage A — sporadic-E deciles at every sample area.

Drives whole LayerStates (KM = 1..5) so the DO-100 loop and the KM < 1
early return are both live, and sweeps PSC(4) across 0.0 (Es disabled — the
VOACAP arm) and 0.70 (the ICEPAC arm), since that scaling is the only thing
between VERSY's output and FS.
"""

import numpy as np

from voacap_gpu import engine
from voacap_gpu.cuda.state import LAYER_DTYPE, pack_layer
from voacap_gpu.scalar.esind import esind
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.virtim import virtim

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "esind.cuh"

extern "C" __global__ void drive(LayerState* st, const int* ikim,
                                 const float* ab, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    esind(&st[i], ikim, 6, ab);
}
'''

D2R = np.float32(0.01745329251)


def _states(n, rng):
    out = []
    for i in range(n):
        st = LayerState()
        st.km = [0, 1, 2, 3, 4, 5][i % 6]        # 0 exercises the early return
        st.ssn = np.float32(rng.uniform(0.0, 200.0))
        st.psc[:] = np.float32(1.0)
        st.psc[4] = np.float32(0.70 if i % 2 else 0.0)
        for k in range(1, 6):
            st.clat[k] = np.float32(rng.uniform(-85.0, 85.0)) * D2R
            st.clong[k] = np.float32(rng.uniform(-180.0, 180.0)) * D2R
            st.gmdip[k] = np.float32(rng.uniform(-85.0, 85.0)) * D2R
        out.append(st)
    return out


def test_esind_bit_exact():
    rng = np.random.default_rng(71)
    ist = redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(100))
    ab = virtim(ist, np.float32(9.0))
    n = 240
    states = _states(n, rng)

    buf = np.zeros(n, LAYER_DTYPE)
    for i, s in enumerate(states):
        buf[i] = pack_layer(s)
    expected = np.zeros(n, LAYER_DTYPE)
    for i, s in enumerate(states):
        esind(s, ist, ab)
        expected[i] = pack_layer(s)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n, [
        buf, np.ascontiguousarray(ist.ikim, np.int32).ravel(),
        np.ascontiguousarray(ab, np.float32)])
    got = fetch(dev[0]).view(LAYER_DTYPE)

    for name in ("fs", "hs"):
        assert_bitexact(np.ascontiguousarray(got[name]).ravel(),
                        np.ascontiguousarray(expected[name]).ravel(),
                        f"esind {name}")

    # branch coverage
    assert (np.array([s.km for s in states]) == 0).any(), "the KM<1 return"
    assert (expected["hs"][:, 1] == np.float32(110.0)).any(), "the DO-100 body"
    assert (expected["fs"] != 0.0).any(), "a non-zero Es decile (PSC4=0.70)"
