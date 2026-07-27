"""GEOM Stage C vs Stage A — path geometry, sample areas, ground constants.

One thread per circuit, expected values computed by calling Stage A
in-process on the same cases, bit-exact assert over every field GEOM writes.

The case set is built to reach every branch the routine has: all three
sample-area regimes (KM = 1 / 3 / 5), the polar-transmitter arm
(cos(tlat) < EPSLON), the near-pole reflection-area arm, the coincident-
endpoint nudge (DDX), the long-path option (NPSL=1), antimeridian-crossing
longitudes, and both land and sea ground constants.
"""

import numpy as np

from voacap_gpu import engine
from voacap_gpu.scalar.geom import geom as geom_ref
from voacap_gpu.scalar.noisy import noisy
from voacap_gpu.scalar.redmap import redmap

from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

POLE = (78.5, -69.0)          # engine.POLE_DEFAULT (the runtime fallback)

_SRC = r'''
#include "geom.cuh"

extern "C" __global__ void drive(const float* tll, const int* npsl,
                                 const float* fakp, const float* fakmap,
                                 const float* hmym, const float* fakabp,
                                 const float* abmap,
                                 float* scal, int* km, float* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    GeomResult g;
    geom(tll[i * 4 + 0], tll[i * 4 + 1], tll[i * 4 + 2], tll[i * 4 + 3],
         npsl[i], 78.5f, -69.0f, fakp, fakmap, hmym, fakabp, abmap, &g);
    float* s = scal + i * 8;
    s[0] = g.dlong; s[1] = g.gcd;  s[2] = g.gcdkm; s[3] = g.btr;
    s[4] = g.btrd;  s[5] = g.brtd; s[6] = g.alatd; s[7] = g.rlatd;
    km[i] = g.km;
    float* a = arr + i * 40;
    for (int k = 0; k < 5; ++k) {
        a[ 0 + k] = g.rd[k];
        a[ 5 + k] = g.clat[k];
        a[10 + k] = g.clong[k];
        a[15 + k] = g.glat[k];
        a[20 + k] = g.gyz[k];
        a[25 + k] = g.gmdip[k];
        a[30 + k] = g.sigpat[k];
        a[35 + k] = g.epspat[k];
    }
}
'''

# (tlat, tlon, rlat, rlon, npsl) — south/west negative
_CASES = [
    (51.50, -0.13, 48.85, 2.35, 0),        # short, KM=1 (343 km)
    (40.00, -105.0, 40.00, -70.0, 0),      # ~2960 km -> KM=3 band
    (0.00, 0.0, 0.00, 27.0, 0),            # ~3000 km equatorial -> KM=3
    (40.00, -105.0, 50.00, 0.0, 0),        # long, KM=5
    (34.05, -118.24, 35.68, 139.77, 0),    # trans-Pacific, dateline
    (51.50, -0.13, -36.85, 174.76, 1),     # long path (NPSL=1)
    (51.50, -0.13, -36.85, 174.76, 0),     # ... and its short twin
    (89.95, 10.0, 45.0, 20.0, 0),          # polar TX (cos(tlat) < EPSLON)
    (-89.95, 10.0, -45.0, 20.0, 0),        # polar TX, south
    (10.0, 20.0, 10.01, 20.01, 0),         # coincident endpoints -> DDX nudge
    (-10.0, 20.0, -10.01, 20.01, 0),       # ... negative-latitude arm
    (0.0, 0.0, 0.0, 90.0, 0),              # equatorial quarter-circle
    (-33.87, 151.21, 64.80, -147.70, 0),   # Sydney -> Fairbanks (very long)
    (60.0, 179.9, 60.0, -179.9, 0),        # straddles the antimeridian
    (20.0, 30.0, -20.0, -150.0, 0),        # near-antipodal
]


def test_geom_bit_exact():
    st = redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(100))
    wld = lambda a, b: noisy(st, 7, a, b)          # noqa: E731

    cases = np.array([c[:4] for c in _CASES], dtype=np.float32)
    npsl = np.array([c[4] for c in _CASES], dtype=np.int32)
    n = len(_CASES)

    exp_s = np.zeros((n, 8), np.float32)
    exp_km = np.zeros(n, np.int32)
    exp_a = np.zeros((n, 40), np.float32)
    for i, (tla, tlo, rla, rlo, ps) in enumerate(_CASES):
        g = geom_ref(tla, tlo, rla, rlo, ps, POLE[0], POLE[1], wld)
        exp_s[i] = [g.dlong, g.gcd, g.gcdkm, g.btr,
                    g.btrd, g.brtd, g.alatd, g.rlatd]
        exp_km[i] = g.km
        for j, nm in enumerate(("rd", "clat", "clong", "glat",
                                "gyz", "gmdip", "sigpat", "epspat")):
            exp_a[i, j * 5:(j + 1) * 5] = getattr(g, nm)

    mod = compile_test(_SRC)
    out_s = np.zeros((n, 8), np.float32)
    out_km = np.zeros(n, np.int32)
    out_a = np.zeros((n, 40), np.float32)
    dev = launch_rows(mod.get_function("drive"), n, [
        cases.ravel(), npsl,
        np.ascontiguousarray(st.fakp, np.float32).ravel(),
        np.ascontiguousarray(st.fakmap, np.float32).ravel(),
        np.ascontiguousarray(st.hmym, np.float32).ravel(),
        np.ascontiguousarray(st.fakabp, np.float32).ravel(),
        np.ascontiguousarray(st.abmap, np.float32).ravel(),
        out_s.ravel(), out_km, out_a.ravel()])

    got_s = fetch(dev[7]).reshape(n, 8)
    got_km = fetch(dev[8])
    got_a = fetch(dev[9]).reshape(n, 40)

    assert np.array_equal(got_km, exp_km), f"KM: {got_km} vs {exp_km}"
    assert_bitexact(got_s.ravel(), exp_s.ravel(), "geom scalars", cases=cases)
    assert_bitexact(got_a.ravel(), exp_a.ravel(), "geom arrays", cases=cases)

    # branch coverage the case set must actually reach
    assert set(exp_km.tolist()) == {1, 3, 5}, "all three KM regimes"
    assert (npsl == 1).any(), "long-path arm"
    sig = exp_a[:, 30:35]
    assert (sig == np.float32(0.001)).any(), "a land sample area"
    assert (sig == np.float32(5.0)).any(), "a sea sample area"
