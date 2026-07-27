"""The per-hour layer chain on the device vs Stage A, composed end to end.

GEOTIM -> TIMVAR -> F2VAR -> ESIND -> IONSET, run in one kernel from real
GEOM output, compared field-for-field against the identical Stage A sequence.

This is the composition test for TIMVAR and F2VAR (which consume each
other's outputs, so isolated cases would not exercise them honestly), and it
re-validates GEOTIM/EF1VAR/ESIND/IONSET in the arrangement the production
setup kernel will actually use.  Every LayerState and TimeState field is
compared, so a routine that writes a field it should not is caught here even
if its own unit test passes.

The circuits span short/medium/long paths, both hemispheres, all four
seasons, quiet and active sun, and all 24 hours, so the twilight arms of
F2VAR and the day/night arms of TIMVAR are live.
"""

import numpy as np

from voacap_gpu import engine
from voacap_gpu.cuda.state import (LAYER_DTYPE, TIME_DTYPE, pack_layer,
                                    pack_time)
from voacap_gpu.engine import _install_geometry
from voacap_gpu.scalar.circuit import Circuit
from voacap_gpu.scalar.esind import esind
from voacap_gpu.scalar.f2var import f2var
from voacap_gpu.scalar.geom import geom
from voacap_gpu.scalar.geotim import geotim
from voacap_gpu.scalar.ionset import ionset
from voacap_gpu.scalar.noisy import noisy
from voacap_gpu.scalar.redmap import redmap
from voacap_gpu.scalar.timvar import timvar
from voacap_gpu.scalar.virtim import virtim

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

POLE = (78.5, -69.0)

_SRC = r'''
#include "esind.cuh"
#include "f2var.cuh"
#include "geotim.cuh"
#include "ionset.cuh"
#include "timvar.cuh"

extern "C" __global__ void drive(LayerState* lay, TimeState* tm,
                                 const int* jt, const float* lon,
                                 const int* ikim, const float* abs_,
                                 const float* anew, const float* bnew,
                                 const float* achi, const float* bchi,
                                 const float* fakp, const float* fakmap,
                                 const float* hmym, const float* fakabp,
                                 const float* abmap, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* ab = abs_ + (size_t)i * 319;
    geotim(&lay[i], &tm[i], jt[i], lon[i * 2 + 0], lon[i * 2 + 1]);
    timvar(&lay[i], ikim, 6, ab, anew, bnew, achi, bchi,
           fakp, fakmap, hmym, fakabp, abmap, tm[i].gmt);
    f2var(&lay[i], ikim, 6, ab);
    esind(&lay[i], ikim, 6, ab);
    ionset(&lay[i]);
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


def _build(ist, tx_rx, month, ssn, psc4):
    """A Circuit with geometry installed and nothing of the hour chain run."""
    wld = lambda a, b: noisy(ist, 7, a, b)          # noqa: E731
    tla, tlo, rla, rlo = tx_rx
    g = geom(tla, tlo, rla, rlo, 0, POLE[0], POLE[1], wld)
    circ = Circuit(iono=ist)
    circ.layers.month = month
    circ.layers.ssn = np.float32(ssn)
    circ.layers.psc[:] = np.array([0.0, 1.0, 1.0, 1.0, psc4], np.float32)
    circ.time.itim = 1
    _install_geometry(circ, g, tla, tlo, rlo, np.float32(3.0))
    return circ


def test_layer_chain_bit_exact():
    """One launch per (month, SSN): the REDMAP tables differ between them, so
    each group uploads its own coefficient set — sharing one would silently
    compare the device against the wrong month."""
    mod = compile_test(_SRC)
    seen_kfx = set()
    any_fi = False
    for month, ssn in ((1, 10.0), (4, 100.0), (7, 160.0), (10, 60.0)):
        ist = redmap(engine._COEFFS_DIR, "CCIR", month, np.float32(ssn))
        kfx, fi_any = _run_group(mod, ist, month, ssn)
        seen_kfx |= kfx
        any_fi |= fi_any
    assert any_fi, "critical frequencies were set"
    assert seen_kfx == {1, 2, 3}, f"all KFX regimes, got {seen_kfx}"


def _run_group(mod, ist, month, ssn):
    cases = [(path, jt) for path in _PATHS for jt in range(0, 25, 2)]
    n = len(cases)

    lay_in = np.zeros(n, LAYER_DTYPE)
    tm_in = np.zeros(n, TIME_DTYPE)
    lay_exp = np.zeros(n, LAYER_DTYPE)
    tm_exp = np.zeros(n, TIME_DTYPE)
    jts = np.zeros(n, np.int32)
    lon = np.zeros((n, 2), np.float32)
    abs_ = np.zeros((n, 319), np.float32)

    for i, (path, jt) in enumerate(cases):
        circ = _build(ist, path, month, ssn, 0.7)
        lay_in[i] = pack_layer(circ.layers)
        tm_in[i] = pack_time(circ.time)
        jts[i] = jt
        lon[i] = (circ.tlong, circ.rlong)
        # itim=1, so GMT is the requested hour and AB can be built on the host
        abs_[i] = virtim(ist, np.float32(jt))

        geotim(circ.layers, circ.time, jt, circ.tlong, circ.rlong)
        ab = virtim(ist, circ.time.gmt)
        timvar(circ.layers, ist, ab, circ.time.gmt)
        f2var(circ.layers, ist, ab)
        esind(circ.layers, ist, ab)
        ionset(circ.layers)
        lay_exp[i] = pack_layer(circ.layers)
        tm_exp[i] = pack_time(circ.time)

    f32 = lambda a: np.ascontiguousarray(a, np.float32).ravel()   # noqa: E731
    dev = launch_rows(mod.get_function("drive"), n, [
        lay_in, tm_in, jts, lon.ravel(),
        np.ascontiguousarray(ist.ikim, np.int32).ravel(), abs_.ravel(),
        f32(ist.anew), f32(ist.bnew), f32(ist.achi), f32(ist.bchi),
        f32(ist.fakp), f32(ist.fakmap), f32(ist.hmym),
        f32(ist.fakabp), f32(ist.abmap)])
    lay_got = fetch(dev[0]).view(LAYER_DTYPE)
    tm_got = fetch(dev[1]).view(TIME_DTYPE)

    # every field of both states, not just the ones we expect to change
    tag = f"m{month}s{ssn:g}"
    for name in LAYER_DTYPE.names:
        g = np.ascontiguousarray(lay_got[name]).ravel()
        e = np.ascontiguousarray(lay_exp[name]).ravel()
        if g.dtype.kind == "i":
            assert np.array_equal(g, e), f"layer.{name} [{tag}]"
        else:
            assert_bitexact(g, e, f"layer.{name} [{tag}]")
    for name in TIME_DTYPE.names:
        g = np.ascontiguousarray(tm_got[name]).ravel()
        e = np.ascontiguousarray(tm_exp[name]).ravel()
        if g.dtype.kind == "i":
            assert np.array_equal(g, e), f"time.{name} [{tag}]"
        else:
            assert_bitexact(g, e, f"time.{name} [{tag}]")

    # RAT is only written for areas 1..KM; the 1-based pad and unused slots
    # stay zero, so the floor applies to the written entries only.
    rat = lay_exp["rat"]
    assert (rat[rat > 0.0] >= 2.0).all(), f"RAT floor held [{tag}]"
    return set(lay_exp["kfx"].tolist()), bool((lay_exp["fi"] > 0.0).any())
