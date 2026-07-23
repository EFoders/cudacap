"""PEN Stage C vs Stage A — parabolic-layer retardation term.

PEN has no golden dump of its own (the Stage A `profile` dump probes it once
per case through the LECDEN chain), so the case set here is the profile
dump's layer parameters plus randomized domain-valid FI/YI, probed at
frequency ratios that straddle the X = F/FI(I,K) >= 1.001 clamp on both
sides.  One packed LayerState per parameter row, one thread per (state,
layer, frequency) probe; expected values from Stage A in-process; bit-exact
with no residual (one ALOG both sides evaluate binary64-and-round-once).
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.pen import pen
from voacap_gpu.stage_c.state import LAYER_DTYPE, pack_layer

from tests.stage_a.dumputil import load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("profile")

_SRC = r'''
#include "pen.cuh"

extern "C" __global__ void drive(const LayerState* states, const int* sidx,
                                 const int* ii, const float* f, float* out,
                                 int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    out[t] = pen(&states[sidx[t]], ii[t], 1, f[t]);
}
'''

# ratios of F to FI(I,K): below, straddling, and above the 1.001 clamp
_RATIOS = [0.3, 0.9, 0.9995, 1.0005, 1.001, 1.05, 1.5, 3.0]


def _layer_rows():
    rows = [[float(x) for x in inp] for inp in DUMP[1]] if DUMP else []
    rng = np.random.default_rng(41)
    for _ in range(60):
        fi1 = rng.uniform(2.0, 4.0)
        fi3 = rng.uniform(6.0, 14.0)
        fi2 = 0.0 if rng.random() < 0.3 else fi1 + rng.uniform(0.3, 0.9) * (
            fi3 - fi1)
        rows.append([fi1, fi2, fi3,
                     rng.uniform(18.0, 22.0), rng.uniform(20.0, 50.0),
                     rng.uniform(50.0, 140.0),
                     rng.uniform(105.0, 115.0), rng.uniform(150.0, 200.0),
                     rng.uniform(280.0, 480.0)])
    return rows


def test_pen_bit_exact():
    rows = _layer_rows()
    states = np.zeros(len(rows), dtype=LAYER_DTYPE)
    objs = []
    for j, r in enumerate(rows):
        s = LayerState(km=1)
        s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(r[0]), F32(r[1]), F32(r[2])
        s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(r[3]), F32(r[4]), F32(r[5])
        s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(r[6]), F32(r[7]), F32(r[8])
        states[j] = pack_layer(s)
        objs.append(s)

    sidx, ii, f = [], [], []
    for j, s in enumerate(objs):
        for i in (1, 2, 3):
            if s.fi[i, 1] <= 0.0:
                continue  # PEN is never called on an absent layer
            for r in _RATIOS:
                sidx.append(j)
                ii.append(i)
                f.append(F32(r) * s.fi[i, 1])
    sidx = np.array(sidx, np.int32)
    ii = np.array(ii, np.int32)
    f = np.array(f, np.float32)
    n = len(f)

    expected = np.array([pen(objs[sidx[t]], int(ii[t]), 1, f[t])
                         for t in range(n)], dtype=np.float32)

    mod = compile_test(_SRC)
    out = np.zeros(n, np.float32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [states.view(np.uint8), sidx, ii, f, out])
    assert_bitexact(fetch(dev[4]), expected, "pen", cases=list(zip(sidx, ii)))

    # both sides of the X >= 1.001 clamp were exercised
    x = f / np.array([objs[sidx[t]].fi[ii[t], 1] for t in range(n)],
                     np.float32)
    assert (x < F32(1.001)).any(), "clamp branch unreached"
    assert (x > F32(1.001)).any(), "unclamped branch unreached"
    assert n > 2000
