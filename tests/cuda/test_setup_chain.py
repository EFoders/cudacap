"""THE WHOLE per-hour setup chain on the device vs Stage A.

GEOTIM -> TIMVAR -> F2VAR -> ESIND -> IONSET -> CURMUF, run in one kernel
from real GEOM output, compared field-for-field against ``engine._setup_hour``
executed in-process on the same circuits.  Every LayerState, TimeState and
MufState field is compared bit-exactly, so this is the acceptance test for the
device-side setup: if it passes, the host no longer has to build circuits.

CURMUF is the routine under test here (the layer chain has its own test); its
LECDEN/GETHP/BENDY/PEN/F2DIS leaves were already on the device.  The circuits
span short/medium/long paths, both hemispheres, four seasons, quiet and active
sun, and every second hour, so the E/F1/F2 controlling-area choices, the
Martyn iteration's converged and out-of-tries exits, the no-F1 copy arm, and
both Es arms are all live.
"""

import numpy as np

from voacap_gpu import engine
from voacap_gpu.cuda.state import (LAYER_DTYPE, MUF_DTYPE, TIME_DTYPE,
                                    pack_layer, pack_muf, pack_time)
from voacap_gpu.engine import _install_geometry, _setup_hour
from voacap_gpu.scalar.circuit import Circuit
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.geom import geom
from voacap_gpu.scalar.noisy import noisy
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.virtim import virtim

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

POLE = (78.5, -69.0)

_SRC = r'''
#include "curmuf.cuh"
#include "esind.cuh"
#include "f2var.cuh"
#include "geotim.cuh"
#include "ionset.cuh"
#include "timvar.cuh"

extern "C" __global__ void drive(LayerState* lay, TimeState* tm, MufState* muf,
                                 const int* jt, const float* lon,
                                 const float* geo, const int* ikim,
                                 const float* abs_, const float* anew,
                                 const float* bnew, const float* achi,
                                 const float* bchi, const float* fakp,
                                 const float* fakmap, const float* hmym,
                                 const float* fakabp, const float* abmap,
                                 const float* f2d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* ab = abs_ + (size_t)i * 319;
    geotim(&lay[i], &tm[i], jt[i], lon[i * 2 + 0], lon[i * 2 + 1]);
    timvar(&lay[i], ikim, 6, ab, anew, bnew, achi, bchi,
           fakp, fakmap, hmym, fakabp, abmap, tm[i].gmt);
    f2var(&lay[i], ikim, 6, ab);
    esind(&lay[i], ikim, 6, ab);
    ionset(&lay[i]);
    curmuf(&lay[i], &muf[i], f2d, geo[i * 3 + 0], geo[i * 3 + 1],
           geo[i * 3 + 2], lay[i].ssn, tm[i].it, lay[i].iedp);
}
'''

_PATHS = [
    (51.50, -0.13, 48.85, 2.35),        # short  (KM=1)
    (40.00, -105.0, 40.00, -70.0),      # medium (KM=3)
    (40.00, -105.0, 50.00, 0.0),        # long   (KM=5)
    (34.05, -118.24, 35.68, 139.77),    # trans-Pacific
    (-33.87, 151.21, 64.80, -147.70),   # southern -> arctic
    (-23.00, 30.00, -35.00, 138.00),    # southern hemisphere
]


def _build(ist, tx_rx, month, ssn):
    wld = lambda a, b: noisy(ist, 7, a, b)          # noqa: E731
    tla, tlo, rla, rlo = tx_rx
    g = geom(tla, tlo, rla, rlo, 0, POLE[0], POLE[1], wld)
    circ = Circuit(iono=ist)
    circ.muf.xluf[:] = F32(-1.0)                    # SETOUT's preset
    circ.layers.month = month
    circ.layers.ssn = np.float32(ssn)
    circ.layers.psc[:] = np.array([0.0, 1.0, 1.0, 1.0, 0.7], np.float32)
    circ.time.itim = 1
    _install_geometry(circ, g, tla, tlo, rlo, np.float32(3.0))
    return circ


def _run_group(mod, ist, month, ssn):
    cases = [(path, jt) for path in _PATHS for jt in range(0, 25, 2)]
    n = len(cases)
    lay_in = np.zeros(n, LAYER_DTYPE)
    tm_in = np.zeros(n, TIME_DTYPE)
    muf_in = np.zeros(n, MUF_DTYPE)
    lay_exp = np.zeros(n, LAYER_DTYPE)
    tm_exp = np.zeros(n, TIME_DTYPE)
    muf_exp = np.zeros(n, MUF_DTYPE)
    jts = np.zeros(n, np.int32)
    lon = np.zeros((n, 2), np.float32)
    geo = np.zeros((n, 3), np.float32)
    abs_ = np.zeros((n, 319), np.float32)

    for i, (path, jt) in enumerate(cases):
        circ = _build(ist, path, month, ssn)
        lay_in[i] = pack_layer(circ.layers)
        tm_in[i] = pack_time(circ.time)
        muf_in[i] = pack_muf(circ.muf)
        jts[i] = jt
        lon[i] = (circ.tlong, circ.rlong)
        geo[i] = (circ.gcdkm, circ.gcd, circ.amind)
        abs_[i] = virtim(ist, np.float32(jt))       # itim=1 -> GMT == the hour

        _setup_hour(circ, jt)                       # Stage A, the real driver
        lay_exp[i] = pack_layer(circ.layers)
        tm_exp[i] = pack_time(circ.time)
        muf_exp[i] = pack_muf(circ.muf)

    f32 = lambda a: np.ascontiguousarray(a, np.float32).ravel()   # noqa: E731
    dev = launch_rows(mod.get_function("drive"), n, [
        lay_in, tm_in, muf_in, jts, lon.ravel(), geo.ravel(),
        np.ascontiguousarray(ist.ikim, np.int32).ravel(), abs_.ravel(),
        f32(ist.anew), f32(ist.bnew), f32(ist.achi), f32(ist.bchi),
        f32(ist.fakp), f32(ist.fakmap), f32(ist.hmym),
        f32(ist.fakabp), f32(ist.abmap), f32(ist.f2d)])
    got = (fetch(dev[0]).view(LAYER_DTYPE), fetch(dev[1]).view(TIME_DTYPE),
           fetch(dev[2]).view(MUF_DTYPE))
    exp = (lay_exp, tm_exp, muf_exp)

    tag = f"m{month}s{ssn:g}"
    for dt, g, e, label in zip((LAYER_DTYPE, TIME_DTYPE, MUF_DTYPE),
                               got, exp, ("layer", "time", "muf")):
        for name in dt.names:
            gv = np.ascontiguousarray(g[name]).ravel()
            ev = np.ascontiguousarray(e[name]).ravel()
            if gv.dtype.kind == "i":
                assert np.array_equal(gv, ev), f"{label}.{name} [{tag}]"
            else:
                assert_bitexact(gv, ev, f"{label}.{name} [{tag}]")
    return muf_exp


def test_setup_chain_bit_exact():
    mod = compile_test(_SRC)
    mods, no_f1, es_seen, es_absent = set(), False, False, False
    for month, ssn in ((1, 10.0), (4, 100.0), (7, 160.0), (10, 60.0)):
        ist = redmap(engine._COEFFS_DIR, "CCIR", month, np.float32(ssn))
        m = _run_group(mod, ist, month, ssn)
        mods |= set(m["modmuf"].tolist())
        # F1 == E exactly is the "no F1 layer" copy arm
        no_f1 |= bool((m["f1muf"] == m["emuf"]).any())
        es_seen |= bool((m["esmuf"] > 0.0).any())
        es_absent |= bool((m["esmuf"] == 0.0).any())

    assert mods, "a controlling mode was chosen"
    assert no_f1, "the no-F1 copy arm was exercised"
    assert es_seen and es_absent, "both sporadic-E arms were exercised"
