"""EF1VAR Stage C vs Stage A — the E and F1 layer parameters.

Two launches, one against a midnight AB and one against a midday AB, so the
"bad map value, use the midnight value" arm (GAMMA(6) < 0.36) is reachable
as well as the ordinary daylight one.  The zenith sweep spans 0..180 degrees
so both sides of ZMAX are live — above it the three F1 slots are zeroed,
which is the state IONSET later keys on.
"""

import numpy as np

from voacap_gpu import engine
from voacap_gpu.cuda.state import LAYER_DTYPE, pack_layer
from voacap_gpu.scalar.ef1var import ef1var
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.virtim import virtim

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "ef1var.cuh"

extern "C" __global__ void drive(LayerState* st, const int* ikim,
                                 const float* ab, const float* anew,
                                 const float* bnew, const float* achi,
                                 const float* bchi, const float* cycen,
                                 float* ret, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ret[i] = ef1var(&st[i], ikim, 6, ab, anew, bnew, achi, bchi, cycen[i], 1);
}
'''

D2R = np.float32(0.01745329251)


def _states(n, rng):
    out, cyc = [], np.zeros(n, np.float32)
    for i in range(n):
        st = LayerState()
        st.km = 1
        st.ssn = np.float32(rng.uniform(0.0, 200.0))
        st.psc[:] = np.float32(1.0)
        st.clat[1] = np.float32(rng.uniform(-88.0, 88.0)) * D2R
        st.clong[1] = np.float32(rng.uniform(-180.0, 180.0)) * D2R
        st.gmdip[1] = np.float32(rng.uniform(-88.0, 88.0)) * D2R
        out.append(st)
        # zenith 0..180 deg, and both signs (the routine takes abs)
        cyc[i] = np.float32(rng.uniform(-np.pi, np.pi))
    return out, cyc


def _run(mod, ist, ab, states, cyc):
    n = len(states)
    buf = np.zeros(n, LAYER_DTYPE)
    for i, s in enumerate(states):
        buf[i] = pack_layer(s)
    exp = np.zeros(n, LAYER_DTYPE)
    exp_ret = np.zeros(n, np.float32)
    import copy
    for i, s in enumerate(states):
        s2 = copy.deepcopy(s)
        exp_ret[i] = ef1var(s2, ist, ab, cyc[i], 1)
        exp[i] = pack_layer(s2)

    ret = np.zeros(n, np.float32)
    f32 = lambda a: np.ascontiguousarray(a, np.float32).ravel()   # noqa: E731
    dev = launch_rows(mod.get_function("drive"), n, [
        buf, np.ascontiguousarray(ist.ikim, np.int32).ravel(), f32(ab),
        f32(ist.anew), f32(ist.bnew), f32(ist.achi), f32(ist.bchi),
        cyc, ret])
    got = fetch(dev[0]).view(LAYER_DTYPE)
    got_ret = fetch(dev[8])
    return got, got_ret, exp, exp_ret


def test_ef1var_bit_exact():
    rng = np.random.default_rng(83)
    ist = redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(100))
    mod = compile_test(_SRC)
    n = 300
    states, cyc = _states(n, rng)

    covered_f1_present = covered_f1_absent = False
    for gmt in (0.0, 12.0):
        ab = virtim(ist, np.float32(gmt))
        got, got_ret, exp, exp_ret = _run(mod, ist, ab, states, cyc)
        for name in ("fi", "yi", "hi", "zenang", "zenmax"):
            assert_bitexact(np.ascontiguousarray(got[name]).ravel(),
                            np.ascontiguousarray(exp[name]).ravel(),
                            f"ef1var {name} (gmt={gmt})")
        assert_bitexact(got_ret, exp_ret, f"ef1var return (gmt={gmt})")
        covered_f1_present |= bool((exp["fi"][:, 2, 1] > 0.0).any())
        covered_f1_absent |= bool((exp["fi"][:, 2, 1] == 0.0).any())

    assert covered_f1_present, "an F1 layer below ZMAX"
    assert covered_f1_absent, "an F1 layer above ZMAX (zeroed)"
