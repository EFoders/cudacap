"""IONSET Stage C vs Stage A — layer-parameter consistency and KFX packing.

IONSET is a pure transform of (FI, YI, HI, KM) into (FI, YI, HI, KFX), so the
cases are synthetic LayerStates rather than circuits: that reaches every arm
of the routine, which a handful of real paths would not.  The state crosses
to the device through the same ``pack_layer`` the production packer uses, so
this also pins the LayerState struct layout.

Branches covered: all three KM regimes (the KFX 1/2/3 shuffles), F1 present
vs absent, F1 merged into E, F1 merged into F2, the surviving-F1 arm with its
HI(2) = min(HI(2), HI(3)) clamp, and the YI(3) = min(YI(3), HDIF) floor.
"""

import numpy as np

from voacap_gpu.cuda.state import LAYER_DTYPE, pack_layer, unpack_layer
from voacap_gpu.scalar.ionset import ionset
from voacap_gpu.scalar.layerstate import LayerState

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "ionset.cuh"

extern "C" __global__ void drive(LayerState* st, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ionset(&st[i]);
}
'''


def _make_states(n, rng):
    """Synthetic layer states spanning IONSET's branch space."""
    states = []
    for i in range(n):
        st = LayerState()
        st.km = int([1, 2, 3, 4, 5][i % 5])
        # FI(layer, area): layer 1=E, 2=F1, 3=F2; areas 1..5 used
        fi = rng.uniform(0.5, 12.0, (4, 6)).astype(np.float32)
        yi = rng.uniform(5.0, 120.0, (4, 6)).astype(np.float32)
        hi = rng.uniform(80.0, 400.0, (4, 6)).astype(np.float32)
        # force each F1 outcome across the case set
        mode = i % 4
        if mode == 0:                       # F1 absent
            fi[2, :] = 0.0
        elif mode == 1:                     # F1 merged into E  (fi2-fi1<=0.2)
            fi[2, :] = fi[1, :] + np.float32(0.1)
        elif mode == 2:                     # F1 merged into F2 (fi3-fi2<=0.2)
            fi[2, :] = fi[3, :] - np.float32(0.1)
        else:                               # F1 survives: E < F1 < F2
            fi[1, :] = np.float32(2.0)
            fi[2, :] = np.float32(5.0)
            fi[3, :] = np.float32(9.0)
        st.fi[:] = fi
        st.yi[:] = yi
        st.hi[:] = hi
        states.append(st)
    return states


def test_ionset_bit_exact():
    rng = np.random.default_rng(31)
    n = 400
    states = _make_states(n, rng)

    buf = np.zeros(n, LAYER_DTYPE)
    for i, st in enumerate(states):
        buf[i] = pack_layer(st)
    before = buf.copy()

    expected = np.zeros(n, LAYER_DTYPE)
    for i, st in enumerate(states):
        ionset(st)                                  # Stage A, in place
        expected[i] = pack_layer(st)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n, [before])
    got = fetch(dev[0]).view(LAYER_DTYPE)

    for name in ("fi", "yi", "hi"):
        assert_bitexact(np.ascontiguousarray(got[name]).ravel(),
                        np.ascontiguousarray(expected[name]).ravel(),
                        f"ionset {name}")
    assert np.array_equal(got["kfx"], expected["kfx"]), "KFX"

    # branch coverage
    assert set(expected["kfx"].tolist()) == {1, 2, 3}, "all three KFX regimes"
    fi_e = expected["fi"]
    assert (fi_e[:, 2, 1:4] == 0.0).any(), "an F1 layer was zeroed"
    assert (fi_e[:, 2, 1:4] > 0.0).any(), "an F1 layer survived"
