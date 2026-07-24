"""BABS + TABS + CONVH Stage C vs Stage A, on the lngabs oracle grid.

The three pure leaves LNGPAT calls to price its geometry, tested together
because the dump drives all three from one grid -- and shares DEL and HP
across them exactly as LNGPAT does, which is what makes the 70 km
disagreement visible from a single row: BABS's ``IF(HP-70.) 105,115,115``
sends a virtual height of exactly 70 km to the LIVE arm while TABS's
``IF(HP-70.) 110,110,115`` sends the same number to the dead one.

Stage A holds these to a LOOSE bar against the Fortran (CH only 97.9%
bit-exact, because ``10*ALOG10(SMALLC)`` with SMALLC near 1 amplifies a
1-ULP cosf disagreement more than a hundredfold).  Stage C's bar against
STAGE A is exact: both evaluate in binary64 and round once, so the
amplification applies to the same input and cancels.  A single differing
cell here is a porting bug, not libm.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.babs import babs
from voacap_gpu.scalar.convh import convh
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.signalstate import SigdState
from voacap_gpu.scalar.tabs import tabs
from voacap_gpu.cuda.state import pack_layer, pack_sigd

from tests.scalar.dumputil import load_dump
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("lngabs")

COLUMNS = ["freq", "gyz3", "acav", "del", "hp", "df", "dm", "gd", "phe",
           "ray"]

_SRC = r'''
#include "babs.cuh"
#include "tabs.cuh"
#include "convh.cuh"

extern "C" __global__ void drive(const LayerState* lay, const SigdState* sig,
                                 const float* freq, const float* del,
                                 const float* hp, const float* df,
                                 const float* dm, const float* gd,
                                 const float* phe, const float* ray,
                                 float* af, float* am, float* ch,
                                 float* ptot, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    af[i] = babs(&lay[i], &sig[i], freq[i], del[i], hp[i], df[i]);
    am[i] = tabs(&lay[i], freq[i], del[i], hp[i], dm[i]);
    convh(gd[i], phe[i], del[i], hp[i], ray[i], &ch[i], &ptot[i]);
}
'''


def _cases():
    _, inputs, _ = DUMP
    return [dict(zip(COLUMNS, (float(x) for x in inp))) for inp in inputs]


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_babs_tabs_convh_bit_exact_vs_stage_a():
    cases = _cases()
    n = len(cases)

    lays, sigs = [], []
    exp = {k: np.zeros(n, np.float32) for k in ("af", "am", "ch", "ptot")}
    for i, v in enumerate(cases):
        lay = LayerState()
        lay.gyz[3] = F32(v["gyz3"])
        sig = SigdState(acav=F32(v["acav"]))
        lays.append(pack_layer(lay))
        sigs.append(pack_sigd(sig))
        exp["af"][i] = babs(lay, sig, F32(v["freq"]), F32(v["del"]),
                            F32(v["hp"]), F32(v["df"]))
        exp["am"][i] = tabs(lay, F32(v["freq"]), F32(v["del"]),
                            F32(v["hp"]), F32(v["dm"]))
        exp["ch"][i], exp["ptot"][i] = convh(
            F32(v["gd"]), F32(v["phe"]), F32(v["del"]), F32(v["hp"]),
            F32(v["ray"]))

    col = {k: np.array([F32(v[k]) for v in cases], np.float32)
           for k in ("freq", "del", "hp", "df", "dm", "gd", "phe", "ray")}
    out = {k: np.zeros(n, np.float32) for k in ("af", "am", "ch", "ptot")}

    mod = compile_test(_SRC)
    dev = launch_rows(
        mod.get_function("drive"), n,
        [np.stack(lays).view(np.uint8), np.stack(sigs).view(np.uint8),
         col["freq"], col["del"], col["hp"], col["df"], col["dm"],
         col["gd"], col["phe"], col["ray"],
         out["af"], out["am"], out["ch"], out["ptot"]])

    for j, name in enumerate(("af", "am", "ch", "ptot"), start=10):
        assert_bitexact(fetch(dev[j]), exp[name], name)

    # The finding this grid exists to pin must actually be present.
    hp = col["hp"]
    at70 = (hp == F32(70.0)) & (col["df"] > 0.0) & (col["dm"] > 0.0)
    assert at70.any(), "the dump no longer carries an HP of exactly 70 km"
    assert (exp["af"][at70] != F32(1000.0)).any(), "BABS must keep 70 LIVE"
    assert (exp["am"][at70] == F32(1000.0)).all(), "TABS must let 70 DIE"

    # branch coverage: both dead arms and the live arm for each routine
    assert (col["df"] <= 0.0).any() and (col["df"] > 0.0).any()
    assert (col["dm"] <= 0.0).any() and (col["dm"] > 0.0).any()
    assert (hp < F32(70.0)).any() and (hp > F32(70.0)).any()
    assert (exp["ch"] == F32(15.0)).any(), "the 15 dB antipodal cap"
    assert (exp["ch"] < F32(15.0)).any()
