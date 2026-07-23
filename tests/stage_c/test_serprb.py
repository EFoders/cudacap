"""SERPRB Stage C vs Stage A, on the serprb oracle grid.

Service probability: RELBIL's normal test per mode, widened by everything
uncertain about the PREDICTION rather than about the hour.  The whole
mutated state is compared, not just SPROB: /allMODE/ SPRO for all 20 slots
(so a dead mode's 0.001 and an untouched slot's sentinel are both pinned),
/TON/ SPR, and the /DON/ D10S/D50S/D90S triple -- including that D90S is set
equal to D10S rather than being a decile of its own.

NMMOD <= 0 makes the Fortran read uninitialised D10SA/D50SA locals; Stage A
raises, so Stage C is never driven there and the dump never contains it.
That is asserted below rather than assumed.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.allmodestate import AllModeState
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.noisestate import AnoisState
from voacap_gpu.stage_a.serprb import serprb
from voacap_gpu.stage_a.signalstate import TonState
from voacap_gpu.stage_a.sonstate import SonState
from voacap_gpu.stage_c.state import (pack_allmode, pack_anois, pack_son,
                                      pack_ton)

from tests.stage_a.dumputil import load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("serprb")

SENTINEL = F32(-8888.0)
NM = 20
SCALARS = ["ifx", "rsn", "lufp", "du", "dl", "sigm", "sygu", "sygl",
           "ads", "sus", "sls", "nmmod", "nrel", "dmp", "pmp", "gcdkm",
           "method", "mspec"]
ARRAYS = ["hp", "sn", "tllow", "tlhgh", "sigpow", "timed", "rely"]
COLUMNS = SCALARS + [f"{a}{i}" for a in ARRAYS for i in range(1, NM + 1)]

_SRC = r'''
#include "serprb.cuh"

extern "C" __global__ void drive(AllModeState* allm, const AnoisState* anois,
                                 TonState* ton, SonState* son,
                                 const int* ifx, const float* rsn,
                                 const int* lufp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    serprb(&allm[i], &anois[i], &ton[i], &son[i], ifx[i], rsn[i], lufp[i]);
}
'''


def _build(v):
    """The entry state for one dump row, exactly as the Stage A test builds it."""
    allm = AllModeState()
    allm.nmmod = int(float(v["nmmod"]))
    allm.nrel = int(float(v["nrel"]))
    for i in range(1, NM + 1):
        for a in ARRAYS:
            getattr(allm, a)[i] = F32(v[f"{a}{i}"])
        allm.spro[i] = SENTINEL
    anois = AnoisState(du=F32(v["du"]), dl=F32(v["dl"]), sigm=F32(v["sigm"]),
                       sygu=F32(v["sygu"]), sygl=F32(v["sygl"]))
    ton = TonState(ads=F32(v["ads"]), sus=F32(v["sus"]), sls=F32(v["sls"]))
    son = SonState()
    son.sprob[1:] = SENTINEL
    return allm, anois, ton, son


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_serprb_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    cases = [dict(zip(COLUMNS, (float(x) for x in inp))) for inp in inputs]
    n = len(cases)
    assert all(int(float(v["nmmod"])) > 0 for v in cases), (
        "a dump row with NMMOD <= 0 would read uninitialised Fortran locals; "
        "Stage A raises there and Stage C must never be driven into it")

    pa, pn, pt, ps = [], [], [], []
    ea, es = [], []
    for v in cases:
        allm, anois, ton, son = _build(v)
        pa.append(pack_allmode(allm))
        pn.append(pack_anois(anois))
        pt.append(pack_ton(ton))
        ps.append(pack_son(son))
        serprb(allm, anois, ton, son, int(float(v["ifx"])), F32(v["rsn"]),
               int(float(v["lufp"])))
        ea.append(pack_allmode(allm))          # expected POST state
        es.append(pack_son(son))
    exp_allm = np.stack(ea)
    exp_son = np.stack(es)
    exp_ton = np.stack([pack_ton(t) for t in
                        [_build(v)[2] for v in cases]])
    exp_ton["spr"] = F32(0.0)                  # SERPRB zeroes it, always

    ifx = np.array([int(float(v["ifx"])) for v in cases], np.int32)
    rsn = np.array([F32(v["rsn"]) for v in cases], np.float32)
    lufp = np.array([int(float(v["lufp"])) for v in cases], np.int32)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [np.stack(pa).view(np.uint8), np.stack(pn).view(np.uint8),
                       np.stack(pt).view(np.uint8), np.stack(ps).view(np.uint8),
                       ifx, rsn, lufp])
    got_allm = fetch(dev[0]).view(exp_allm.dtype)
    got_ton = fetch(dev[2]).view(exp_ton.dtype)
    got_son = fetch(dev[3]).view(exp_son.dtype)

    for name in ("spro", "nmmod", "nrel"):
        assert_bitexact(got_allm[name], exp_allm[name], f"allm.{name}")
    assert_bitexact(got_ton["spr"], exp_ton["spr"], "ton.spr")
    for name in ("sprob", "d10s", "d50s", "d90s"):
        assert_bitexact(got_son[name], exp_son[name], f"son.{name}")

    # Whole-struct survival: nothing outside the documented outputs moved.
    untouched = [f for f in exp_allm.dtype.names if f != "spro"]
    for f in untouched:
        assert_bitexact(got_allm[f], exp_allm[f], f"allm.{f} (untouched)")

    # coverage: both tails, dead modes, and the D90S == D10S identity
    lu = lufp
    assert (lu >= 50).any() and (lu < 50).any(), "both reliability tails"
    hp1 = np.array([F32(v["hp1"]) for v in cases], np.float32)
    assert (hp1 <= F32(70.0)).any(), "no dead-mode (hp <= 70) row"
    assert np.array_equal(exp_son["d90s"], exp_son["d10s"])
