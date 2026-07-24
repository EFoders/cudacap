"""GETHP Stage C vs Stage A — virtual/true reflection heights by Gaussian
integration over the LECDEN profile.

GETHP has no dump of its own; the Stage A `profile` dump probes it at four
ratios of foF2 through the LECDEN chain.  Here every profile-dump case (plus
randomized domain-valid layer parameters) is built into a LayerState by
Stage A's own LECDEN in-process, packed, and probed on the device at those
same ratios plus the below-profile floor and random frequencies.  Expected
values from Stage A in-process; bit-exact (GETHP is SQRT-only, and both
sides evaluate it correctly rounded).
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.gethp import gethp
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.lecden import lecden
from voacap_gpu.cuda.state import LAYER_DTYPE, pack_layer

from tests.scalar.dumputil import load_dump
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("profile")

_SRC = r'''
#include "gethp.cuh"

extern "C" __global__ void drive(const LayerState* states, const int* sidx,
                                 const float* fxx, float* hpx, float* htx,
                                 int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float hp, ht;
    gethp(&states[sidx[t]], fxx[t], &hp, &ht);
    hpx[t] = hp;
    htx[t] = ht;
}
'''


def _layer_rows():
    rows = [[float(x) for x in inp] for inp in DUMP[1]] if DUMP else []
    rng = np.random.default_rng(47)
    for _ in range(40):
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


def test_gethp_bit_exact():
    rows = _layer_rows()
    rng = np.random.default_rng(48)
    states = np.zeros(len(rows), dtype=LAYER_DTYPE)
    objs = []
    for j, r in enumerate(rows):
        s = LayerState(km=1)
        s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(r[0]), F32(r[1]), F32(r[2])
        s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(r[3]), F32(r[4]), F32(r[5])
        s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(r[6]), F32(r[7]), F32(r[8])
        lecden(s, 1)  # Stage A's own profile builder feeds GETHP
        states[j] = pack_layer(s)
        objs.append(s)

    sidx, fxx = [], []
    for j, s in enumerate(objs):
        fp = s.fi[3, 1]
        probes = [F32(0.005),               # below the profile (label 435)
                  F32(0.01),                # GENION's lowest D-E sample
                  F32(0.30) * fp, F32(0.60) * fp,   # the profile-dump probes
                  F32(0.90) * fp, F32(0.99) * fp]
        probes += [F32(v) * fp for v in rng.uniform(0.05, 0.995, 3)]
        for v in probes:
            sidx.append(j)
            fxx.append(v)
    sidx = np.array(sidx, np.int32)
    fxx = np.array(fxx, np.float32)
    n = len(fxx)

    expected_hp = np.zeros(n, np.float32)
    expected_ht = np.zeros(n, np.float32)
    for t in range(n):
        expected_hp[t], expected_ht[t] = gethp(objs[sidx[t]], fxx[t])

    mod = compile_test(_SRC)
    hpx = np.zeros(n, np.float32)
    htx = np.zeros(n, np.float32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [states.view(np.uint8), sidx, fxx, hpx, htx])
    assert_bitexact(fetch(dev[3]), expected_hp, "gethp hpx", cases=fxx)
    assert_bitexact(fetch(dev[4]), expected_ht, "gethp htx", cases=fxx)

    # branch coverage: the below-profile clamp really fired (HPX == HTX ==
    # HTR(1) == 70 km there) and the Gaussian path really integrated
    below = np.array([fxx[t] * fxx[t] <= objs[sidx[t]].fnsq[1, 1]
                      for t in range(n)])
    assert below.any(), "below-profile branch unreached"
    assert (~below).any(), "integration branch unreached"
    assert (expected_hp[below] == expected_ht[below]).all()
    assert n > 1500
