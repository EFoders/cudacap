"""GEOTIM (+ the inlined CNGTIM) Stage C vs Stage A — the four time bases.

Each prediction hour is resolved as universal time, local mean time at the
transmitter, local mean time at the receiver, and local mean time at every
sample area.  ITIM's sign picks which of the first two the user supplied, so
both arms are driven here; the hour wrap that CNGTIM performs is exercised by
sweeping longitudes across the full +-180 range at every hour, including the
midnight case where IT truncates to 0 and is rewritten to 24.

Both state structs cross through the production ``pack_layer``/``pack_time``,
so this pins their struct layouts as well as the arithmetic.
"""

import numpy as np

from voacap_gpu.cuda.state import (LAYER_DTYPE, TIME_DTYPE, pack_layer,
                                    pack_time)
from voacap_gpu.scalar.geotim import geotim
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.timestate import TimeState

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "geotim.cuh"

extern "C" __global__ void drive(LayerState* lay, TimeState* tm,
                                 const int* jt, const float* lon, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    geotim(&lay[i], &tm[i], jt[i], lon[i * 2 + 0], lon[i * 2 + 1]);
}
'''

D2R = np.float32(0.01745329251)


def _cases(rng):
    out = []
    lons = [0.0, 179.9, -179.9, 90.0, -90.0, 45.0, -45.0, 120.0, -120.0]
    for jt in range(0, 25):                      # includes the IT->24 rewrite
        for itim in (1, -1):
            for tl in lons:
                out.append((jt, itim, tl,
                            float(rng.uniform(-180.0, 180.0)),
                            int(rng.integers(1, 6))))
    return out


def test_geotim_bit_exact():
    rng = np.random.default_rng(43)
    cases = _cases(rng)
    n = len(cases)

    lay_in = np.zeros(n, LAYER_DTYPE)
    tm_in = np.zeros(n, TIME_DTYPE)
    lay_exp = np.zeros(n, LAYER_DTYPE)
    tm_exp = np.zeros(n, TIME_DTYPE)
    lon = np.zeros((n, 2), np.float32)
    jts = np.zeros(n, np.int32)

    for i, (jt, itim, tl, rl, km) in enumerate(cases):
        lay = LayerState()
        lay.km = km
        lay.clong[1:km + 1] = rng.uniform(-np.pi, np.pi, km).astype(np.float32)
        tm = TimeState()
        tm.itim = itim
        lay_in[i] = pack_layer(lay)
        tm_in[i] = pack_time(tm)
        jts[i] = jt
        lon[i] = (np.float32(tl) * D2R, np.float32(rl) * D2R)

        geotim(lay, tm, jt, lon[i, 0], lon[i, 1])     # Stage A, in place
        lay_exp[i] = pack_layer(lay)
        tm_exp[i] = pack_time(tm)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [lay_in, tm_in, jts, lon.ravel()])
    lay_got = fetch(dev[0]).view(LAYER_DTYPE)
    tm_got = fetch(dev[1]).view(TIME_DTYPE)

    assert np.array_equal(tm_got["it"], tm_exp["it"]), "IT"
    for name in ("gmt", "gmtr", "utime", "xlmt"):
        assert_bitexact(np.ascontiguousarray(tm_got[name]).ravel(),
                        np.ascontiguousarray(tm_exp[name]).ravel(),
                        f"geotim {name}")
    assert_bitexact(np.ascontiguousarray(lay_got["clck"]).ravel(),
                    np.ascontiguousarray(lay_exp["clck"]).ravel(),
                    "geotim clck")

    # branch coverage
    assert (tm_in["itim"] < 0).any() and (tm_in["itim"] > 0).any(), "both ITIM arms"
    assert (tm_exp["it"] == 24).any(), "the midnight IT->24 rewrite"
