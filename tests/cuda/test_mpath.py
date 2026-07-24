"""MPATH Stage C vs Stage A — multipath probability.

MPATH has no dump of its own; the serprb golden dump carries its inputs
(the tier-B driver runs SERPRB then MPATH on the same injected /allMODE/),
and MPATH's only coupling to SERPRB is through fields the driver injects
directly — so the dump rows convert straight into MPATH cases without
running SERPRB.  A randomized-but-domain-valid set widens the sweep: every
skip arm (the NREL slot itself, dead modes, within-tolerance delays, weak
signals), the two early returns, and the qualifying-mode maximum.  Expected
values come from Stage A in-process; the assert is bit-identity of the
whole /SON/ struct, which also pins that ONLY PROBMP(ifx) is written.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.allmodestate import AllModeState
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.mpath import mpath
from voacap_gpu.scalar.sonstate import SonState
from voacap_gpu.cuda.state import (ALLMODE_DTYPE, SON_DTYPE, pack_allmode,
                                      pack_son)

from tests.scalar.dumputil import load_dump
from tests.scalar.test_serprb import ARRAYS, COLUMNS, NM
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("serprb")

_SRC = r'''
#include "mpath.cuh"

extern "C" __global__ void drive(const AllModeState* allm, SonState* son,
                                 const int* ifx, const float* dmp,
                                 const float* pmp, const float* gcdkm,
                                 const int* method, const int* mspec,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mpath(&allm[i], &son[i], ifx[i], dmp[i], pmp[i], gcdkm[i],
          method[i], mspec[i]);
}
'''


def _dump_cases():
    if DUMP is None:
        return []
    _, inputs, _ = DUMP
    cases = []
    for inp in inputs:
        v = dict(zip(COLUMNS, inp))
        allm = AllModeState()
        allm.nmmod = int(float(v["nmmod"]))
        allm.nrel = int(float(v["nrel"]))
        for i in range(1, NM + 1):
            for a in ARRAYS:
                getattr(allm, a)[i] = F32(v[f"{a}{i}"])
        cases.append((allm, int(float(v["ifx"])), F32(v["dmp"]),
                      F32(v["pmp"]), F32(v["gcdkm"]),
                      int(float(v["method"])), int(float(v["mspec"]))))
    return cases


def _random_cases(n=3000, seed=17):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        allm = AllModeState()
        nmmod = int(rng.integers(0, NM + 1))
        allm.nmmod = nmmod
        allm.nrel = int(rng.integers(1, max(nmmod, 1) + 1))
        for i in range(1, NM + 1):
            # a quarter of the slots dead (HP <= 0), the rest live
            allm.hp[i] = F32(rng.choice([-1.0, 0.0])) if rng.random() < 0.25 \
                else F32(rng.uniform(75.0, 400.0))
            allm.timed[i] = F32(rng.uniform(0.0, 40.0))
            allm.sigpow[i] = F32(rng.uniform(-200.0, -80.0))
            allm.rely[i] = F32(rng.uniform(0.0, 1.0))
        dmp = F32(rng.choice([0.0, -1.0])) if rng.random() < 0.15 \
            else F32(rng.uniform(0.05, 8.0))
        pmp = F32(0.0) if rng.random() < 0.1 \
            else F32(rng.uniform(1.0, 60.0))
        method = int(rng.choice([20, 25, 30]))
        mspec = int(rng.choice([121, 0]))
        gcdkm = F32(rng.uniform(100.0, 12000.0))
        cases.append((allm, int(rng.integers(1, 14)), dmp, pmp, gcdkm,
                      method, mspec))
    return cases


def test_mpath_bit_exact():
    cases = _dump_cases() + _random_cases()
    n = len(cases)
    rng = np.random.default_rng(4)
    allm_buf = np.empty(n, dtype=ALLMODE_DTYPE)
    son_buf = np.empty(n, dtype=SON_DTYPE)
    exp_son = np.empty(n, dtype=SON_DTYPE)
    ifx = np.empty(n, np.int32)
    dmp = np.empty(n, np.float32)
    pmp = np.empty(n, np.float32)
    gcdkm = np.empty(n, np.float32)
    method = np.empty(n, np.int32)
    mspec = np.empty(n, np.int32)
    seen = {"mp_off": 0, "mp_area_cut": 0, "mp_found": 0, "mp_none": 0,
            "skip_nrel": 0, "skip_dead": 0, "skip_close": 0, "skip_weak": 0,
            "empty_table": 0}
    for i, (allm, fx, d, p, g, me, ms) in enumerate(cases):
        son = SonState()
        # a live seed proves the routine overwrites PROBMP unconditionally
        son.probmp[1:] = rng.uniform(0.0, 1.0, 13).astype(np.float32)
        allm_buf[i], son_buf[i] = pack_allmode(allm), pack_son(son)
        ifx[i], dmp[i], pmp[i] = fx, d, p
        gcdkm[i], method[i], mspec[i] = g, me, ms
        mpath(allm, son, fx, d, p, g, me, ms)   # Stage A, in place
        exp_son[i] = pack_son(son)
        # which arms this case exercised, from the inputs
        if d <= 0.0 or p <= 0.0:
            seen["mp_off"] += 1
            continue
        if me == 20 and ms == 121 and g > 7000.0:
            seen["mp_area_cut"] += 1
            continue
        seen["mp_found" if son.probmp[fx] > F32(0.001) else "mp_none"] += 1
        seen["empty_table"] += allm.nmmod < 1
        nrel = allm.nrel
        ttim = allm.timed[nrel]
        limit = allm.sigpow[nrel] - p
        for im in range(1, allm.nmmod + 1):
            if im == nrel:
                seen["skip_nrel"] += 1
            elif allm.hp[im] <= 0.0:
                seen["skip_dead"] += 1
            elif abs(allm.timed[im] - ttim) <= d:
                seen["skip_close"] += 1
            elif allm.sigpow[im] < limit:
                seen["skip_weak"] += 1

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [allm_buf.view(np.uint8), son_buf.view(np.uint8),
                       ifx, dmp, pmp, gcdkm, method, mspec])
    got_son = fetch(dev[1]).reshape(n, SON_DTYPE.itemsize
                                    ).view(SON_DTYPE).reshape(n)
    for name in SON_DTYPE.names:
        assert_bitexact(got_son[name], exp_son[name], f"mpath.son.{name}")

    for arm, c in seen.items():
        assert c, f"branch never reached: {arm} ({seen})"
