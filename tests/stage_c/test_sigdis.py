"""SIGDIS Stage C vs Stage A, on the sigdis oracle grid.

Path-average absorption and system-loss deciles (part one, the KFX-area loop
over SYSSY) plus the D-E absorption-loss parameters and the median/decile
signal adjustments (part two, PRBMUF at the reference frequency FTAB).

The comparison is whole-struct on all three mutated states: /SIGD/ and /TON/
are seeded with sentinels so a field SIGDIS never writes (DSLF/ASMF/DSUF,
GOT/REL/SPR/NF) is proved untouched rather than coincidentally zero, and the
whole LayerState is compared so ABIY/ARTIC beyond KFX are proved untouched
too -- the Stage A test only checks the active areas.

SYS/PERR are shared, not per-thread, so they are stacked per distinct
(month, SSN) and indexed by thread, as in test_syssy.py.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.genion import genion
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.lecden import lecden
from voacap_gpu.stage_a.mufstate import MufState
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_a.sigdis import sigdis
from voacap_gpu.stage_a.signalstate import SigdState, TonState
from voacap_gpu.stage_c.state import (pack_layer, pack_muf, pack_sigd,
                                      pack_ton)

from tests.stage_a.dumputil import COEFFS_DIR, load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("sigdis")
SENTINEL = F32(-8888.0)

_SRC = r'''
#include "sigdis.cuh"

extern "C" __global__ void drive(LayerState* state, const MufState* muf,
                                 const float* sys_all, const float* perr_all,
                                 const int* mi, const float* f2d111,
                                 SigdState* sigd, TonState* ton,
                                 const float* gcdkm, const int* jmode,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sigdis(&state[i], &muf[i], sys_all + mi[i] * (9 * 16 * 6),
           perr_all + mi[i] * (9 * 4 * 6), f2d111[i], &sigd[i], &ton[i],
           gcdkm[i], jmode[i]);
}
'''


def _sentinel_sigd():
    return SigdState(dsl=SENTINEL, asm=SENTINEL, dsu=SENTINEL, aglat=SENTINEL,
                     dslf=SENTINEL, asmf=SENTINEL, dsuf=SENTINEL,
                     acav=SENTINEL, feav=SENTINEL, afe=SENTINEL, bfe=SENTINEL,
                     hnu=SENTINEL, htloss=SENTINEL, xnuz=SENTINEL,
                     xve=SENTINEL)


def _sentinel_ton():
    return TonState(adj=SENTINEL, ads=SENTINEL, gnos=SENTINEL, got=SENTINEL,
                    rel=SENTINEL, sl=SENTINEL, sls=SENTINEL, spr=SENTINEL,
                    su=SENTINEL, sus=SENTINEL, xnoise=SENTINEL,
                    znoise=SENTINEL, nf=-99)


def _entry(inp):
    """The entry LayerState/MufState for one dump row, as Stage A builds it."""
    kfx = int(inp[2])
    s = LayerState(kfx=kfx)
    s.iedp = int(inp[5])
    for a in range(1, 4):
        off = 6 + (a - 1) * 11
        vals = [float(x) for x in inp[off:off + 11]]
        s.fi[1, a], s.fi[2, a], s.fi[3, a] = (F32(v) for v in vals[0:3])
        s.yi[1, a], s.yi[2, a], s.yi[3, a] = (F32(v) for v in vals[3:6])
        s.hi[1, a], s.hi[2, a], s.hi[3, a] = (F32(v) for v in vals[6:9])
        s.glat[a], s.clck[a] = F32(vals[9]), F32(vals[10])
    for k in range(1, kfx + 1):
        lecden(s, k)
        genion(s, k)
    (yfot3, yfot4, yhpf3, yhpf4, ymuf3, ymuf4,
     sigl3, sigl4, sigu3, sigu4) = (float(x) for x in inp[39:49])
    m = MufState()
    m.yfot[3], m.yfot[4] = F32(yfot3), F32(yfot4)
    m.yhpf[3], m.yhpf[4] = F32(yhpf3), F32(yhpf4)
    m.ymuf[3], m.ymuf[4] = F32(ymuf3), F32(ymuf4)
    m.sigl[3], m.sigl[4] = F32(sigl3), F32(sigl4)
    m.sigu[3], m.sigu[4] = F32(sigu3), F32(sigu4)
    return s, m


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_sigdis_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    keys = sorted({(int(r[0]), r[1]) for r in inputs})
    maps = {k: redmap(COEFFS_DIR, "CCIR", k[0], F32(k[1])) for k in keys}
    kidx = {k: i for i, k in enumerate(keys)}
    sys_all = np.ascontiguousarray(
        np.stack([np.asarray(maps[k].sys, np.float32) for k in keys])).ravel()
    perr_all = np.ascontiguousarray(
        np.stack([np.asarray(maps[k].perr, np.float32) for k in keys])).ravel()

    # (state, muf, key, gcdkm, jmode) per case: the dump, plus one
    # deterministic extension.  FEAV never drops to 0.5 MHz on the oracle
    # grid, so the AFE/BFE zero arm and the Greg Hand 0.1 absorption floor
    # would both go uncovered; injecting a 0.4 MHz E-layer critical into
    # row 0's fully-built ionosphere reaches both, and Stage A and Stage C
    # see the identical injected state, so bit-exactness still means what
    # it means everywhere else.
    cases = []
    for r in inputs:
        s, m = _entry(r)
        cases.append((s, m, (int(r[0]), r[1]), F32(r[4]), int(r[3])))
    s, m = _entry(inputs[0])
    s.kfx = 1
    s.fi[1, 1] = F32(0.4)
    cases.append((s, m, (int(inputs[0][0]), inputs[0][1]),
                  F32(inputs[0][4]), 1))
    n = len(cases)

    p_state, p_muf, p_sigd, p_ton = [], [], [], []
    e_state, e_sigd, e_ton = [], [], []
    for s, m, key, gcd, jm in cases:
        sigd, ton = _sentinel_sigd(), _sentinel_ton()
        p_state.append(pack_layer(s))
        p_muf.append(pack_muf(m))
        p_sigd.append(pack_sigd(sigd))
        p_ton.append(pack_ton(ton))
        sigdis(s, m, maps[key], sigd, ton, gcd, jm)
        e_state.append(pack_layer(s))
        e_sigd.append(pack_sigd(sigd))
        e_ton.append(pack_ton(ton))
    exp_state = np.stack(e_state)
    exp_sigd = np.stack(e_sigd)
    exp_ton = np.stack(e_ton)

    mi = np.array([kidx[c[2]] for c in cases], np.int32)
    f2d111 = np.array([maps[c[2]].f2d[0, 0, 0] for c in cases], np.float32)
    gcdkm = np.array([c[3] for c in cases], np.float32)
    jmode = np.array([c[4] for c in cases], np.int32)
    kfx = np.array([c[0].kfx for c in cases], np.int32)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [np.stack(p_state).view(np.uint8),
                       np.stack(p_muf).view(np.uint8), sys_all, perr_all,
                       mi, f2d111, np.stack(p_sigd).view(np.uint8),
                       np.stack(p_ton).view(np.uint8), gcdkm, jmode])
    got_state = fetch(dev[0]).view(exp_state.dtype)
    got_muf = fetch(dev[1]).view(np.stack(p_muf).dtype)
    got_sigd = fetch(dev[6]).view(exp_sigd.dtype)
    got_ton = fetch(dev[7]).view(exp_ton.dtype)

    for name in exp_sigd.dtype.names:
        assert_bitexact(got_sigd[name], exp_sigd[name], f"sigd.{name}")
    for name in exp_ton.dtype.names:
        assert_bitexact(got_ton[name], exp_ton[name], f"ton.{name}")
    for name in exp_state.dtype.names:      # ABIY/ARTIC out, rest untouched
        assert_bitexact(got_state[name], exp_state[name], f"state.{name}")
    for name in got_muf.dtype.names:        # /MUFS/ is read-only here
        assert_bitexact(got_muf[name], np.stack(p_muf)[name],
                        f"muf.{name} (read-only)")

    # SIGDIS writes neither the scratch trio nor the rest of /TON/
    for name in ("dslf", "asmf", "dsuf"):
        assert (exp_sigd[name] == SENTINEL).all(), f"sigd.{name} moved"
    for name in ("gnos", "got", "rel", "spr", "xnoise", "znoise"):
        assert (exp_ton[name] == SENTINEL).all(), f"ton.{name} moved"
    for k in range(n):                      # areas past KFX stay untouched
        assert (exp_state["abiy"][k, kfx[k] + 1:] == F32(0.0)).all()

    # branch coverage
    feav = exp_sigd["feav"]
    glav = exp_sigd["aglat"] * F32(57.295779513)
    seen = {
        "idp_long": int((gcdkm > F32(2500.0)).sum()),
        "idp_short": int((gcdkm <= F32(2500.0)).sum()),
        "afe_high": int((feav - F32(2.0) > 0).sum()),
        "afe_zero": int(((feav - F32(2.0) <= 0)
                         & (feav - F32(0.5) <= 0)).sum()),
        "afe_ramp": int(((feav - F32(2.0) <= 0)
                         & (feav - F32(0.5) > 0)).sum()),
        "ftab_low": int((glav - F32(40.0) <= 0).sum()),
        "ftab_ramp": int(((glav - F32(40.0) > 0)
                          & (glav - F32(50.0) <= 0)).sum()),
        "ftab_polar": int((glav - F32(50.0) > 0).sum()),
        "es_median": int((np.stack(p_muf)["ymuf"][:, 4] > 0).sum()),
        "no_es_median": int((np.stack(p_muf)["ymuf"][:, 4] <= 0).sum()),
        "es_fot": int((np.stack(p_muf)["yfot"][:, 4] > 0).sum()),
        "no_es_fot": int((np.stack(p_muf)["yfot"][:, 4] <= 0).sum()),
        "es_hpf": int((np.stack(p_muf)["yhpf"][:, 4] > 0).sum()),
        "no_es_hpf": int((np.stack(p_muf)["yhpf"][:, 4] <= 0).sum()),
        "asm_floor": int((exp_sigd["asm"] == F32(0.0)).sum()),
        "asm_free": int((exp_sigd["asm"] > F32(0.0)).sum()),
        "dsu_free": int((exp_sigd["dsu"] > F32(0.5)).sum()),
        "dsl_free": int((exp_sigd["dsl"] > F32(1.0)).sum()),
        "kfx_multi": int((kfx > 1).sum()),
    }
    for arm, cnt in seen.items():
        assert cnt, f"branch never reached: {arm} ({seen})"
    # the ABIY 0.1 floor: raw values below it must appear somewhere
    raw = np.array([[F32(-0.04) + np.float32(np.exp(np.float64(
        F32(-2.937) + F32(0.8445) * exp_state["fi"][k, 1, a])))
        for a in range(1, kfx[k] + 1)] for k in range(n)], dtype=object)
    assert any(float(v) < 0.1 for row in raw for v in row), \
        "the Greg Hand 0.1 absorption floor is never exercised"
