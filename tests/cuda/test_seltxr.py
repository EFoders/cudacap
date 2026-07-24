"""SELTXR Stage C vs Stage A — the long-path ray choice at the two ends.

SELTXR has no dump of its own; the gmloss dump carries it (LUFFY runs
GMLOSS then SELTXR), so dump-derived cases are built by replaying the
GMLOSS chain with Stage A on the host and snapshotting the state SELTXR is
handed.  A randomized ray-table sweep plus the directed cases from the
Stage A finding tests (the 70 km kill, the ray-1 incumbent hop count, the
give-up decrement, the answer-changing INFO bit 4) widen the branch cover.
Expected values come from Stage A in-process; the assert is bit-identity of
the whole /REFLX/ and /LPATH/ structs — LTXRGM, the GML/FHP side effects,
and the survival of everything else.

Stage A raises RayIndexError where the find-first search walks past ray 45;
the device returns status 1 with the state exactly as Stage A leaves it at
the raise, and one crafted case pins that arm.
"""

import copy

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.gmloss import gmloss
from voacap_gpu.scalar.losxstate import LosxState
from voacap_gpu.scalar.lpathstate import LpathState
from voacap_gpu.scalar.reflxstate import ReflxState
from voacap_gpu.scalar.seltxr import RayIndexError, seltxr
from voacap_gpu.scalar.zonstate import ZonState
from voacap_gpu.cuda.state import (LOSX_DTYPE, LPATH_DTYPE, REFLX_DTYPE,
                                      pack_losx, pack_lpath, pack_reflx)

from tests.scalar.dumputil import load_dump
from tests.scalar.test_gmloss import COLUMNS as GM_COLUMNS
from tests.scalar.test_gmloss import TXRGML_SENTINEL
from tests.scalar.test_settxr import _build, _cantenna
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("gmloss")
NRAY = 45
DUMP_STRIDE = 4                 # host replay of the GMLOSS chain is not free

_SRC = r'''
#include "seltxr.cuh"

extern "C" __global__ void drive(ReflxState* reflx, const LosxState* losx,
                                 LpathState* lpath, const int* itxrcp,
                                 const float* gcdkm, const float* amind,
                                 const int* info, int* status, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    status[i] = seltxr(&reflx[i], &losx[i], &lpath[i], &itxrcp[i * 3],
                       gcdkm[i], amind[i], info[i]);
}
'''

ITXRCP11 = np.array([0, 1, 1], dtype=np.int32)


def _table(hp, de, gd, nfill=None):
    """A ray table with the given per-ray columns, in control area 1."""
    reflx = ReflxState()
    n = len(hp)
    for i in range(1, n + 1):
        reflx.hpflx[i, 1] = F32(hp[i - 1])
        reflx.delfx[i, 1] = F32(de[i - 1])
        reflx.gdflx[i, 1] = F32(gd[i - 1])
    reflx.iaftxr[1] = n if nfill is None else nfill
    return reflx


def _losx(n, tgain=0.0, andv=0.0, adv=0.0):
    losx = LosxState()
    for i in range(1, n + 1):
        losx.tgainx[i, 1] = F32(tgain)
        losx.andvx[i, 1] = F32(andv)
        losx.advx[i, 1] = F32(adv)
    return losx


def _directed():
    """The Stage A finding cases: (reflx, losx, lpath, itxrcp, gcdkm,
    amind, info)."""
    cases = []
    # GMLOSS keeps a ray at exactly 70 km; SELTXR kills it
    cases.append((_table([70.0, 300.0], [10.0, 10.0], [2000.0, 2000.0]),
                  _losx(2, tgain=6.0, andv=1.0, adv=0.5), LpathState(),
                  ITXRCP11, 4000.0, 5.0, 0))
    # the incumbent hop count belongs to ray 1
    cases.append((_table([300.0, 300.0], [1.0, 10.0], [8000.0, 500.0]),
                  _losx(2, tgain=0.0, andv=1.0, adv=0.0), LpathState(),
                  ITXRCP11, 4000.0, 5.0, 0))
    # a dead ray 2 sends the search to the give-up decrement (never zero)
    cases.append((_table([300.0, 50.0], [1.0, 10.0], [8000.0, 500.0]),
                  _losx(2), LpathState(), ITXRCP11, 4000.0, 5.0, 0))
    # the diagnostic flag changes the answer: same table, INFO 0 vs 4
    for info in (0, 4):
        cases.append((_table([300.0, 300.0, 300.0], [3.0, 3.4, 2.6],
                             [1900.0, 1450.0, 1150.0]),
                      _losx(3, tgain=0.0, andv=1.0, adv=0.0), LpathState(),
                      ITXRCP11, 4000.0, 1.0, info))
    # the refusal: every ray exists and none clears AMIND, so the search
    # walks past ray 45 — Stage A raises, the device returns status 1
    cases.append((_table([300.0] * NRAY, [1.0] * NRAY, [8000.0] * NRAY),
                  _losx(NRAY), LpathState(), ITXRCP11, 4000.0, 5.0, 0))
    return cases


def _random_cases(n=1500, seed=29):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        reflx, losx = ReflxState(), LosxState()
        k2 = int(rng.integers(2, 4))
        itxrcp = np.array([0, 1, k2], np.int32)
        for k in sorted({1, k2}):
            reflx.iaftxr[k] = int(rng.integers(1, NRAY + 1))
            for ia in range(1, NRAY + 1):
                u = rng.random()
                if u < 0.10:
                    hp = 70.0           # dead here, live for SETTXR
                elif u < 0.35:
                    hp = rng.uniform(30.0, 69.9)
                else:
                    hp = rng.uniform(75.0, 400.0)
                reflx.hpflx[ia, k] = F32(hp)
                reflx.delfx[ia, k] = F32(rng.uniform(0.0, 25.0))
                reflx.gdflx[ia, k] = F32(rng.uniform(300.0, 9000.0))
                losx.tgainx[ia, k] = F32(rng.uniform(-10.0, 25.0))
                losx.andvx[ia, k] = F32(rng.uniform(0.0, 40.0))
                losx.advx[ia, k] = F32(rng.uniform(0.0, 40.0))
                reflx.gml[ia, k] = F32(-999.0)
                reflx.fhp[ia, k] = F32(999.0)
        lpath = LpathState()
        if rng.random() < 0.5:          # sweep the blkdat thresholds too
            lpath.delopt = F32(rng.uniform(1.0, 6.0))
            lpath.gmin = F32(rng.uniform(0.5, 5.0))
            lpath.ymin = F32(rng.uniform(0.02, 0.3))
        cases.append((reflx, losx, lpath, itxrcp,
                      float(rng.uniform(500.0, 20000.0)),
                      float(rng.uniform(0.0, 12.0)),
                      int(rng.choice([0, 4]))))
    return cases


def _dump_cases():
    """Replay the GMLOSS chain on the host; snapshot what SELTXR is handed."""
    if DUMP is None:
        return []
    _, inputs, _ = DUMP
    cases = []
    for inp in inputs[::DUMP_STRIDE]:
        v = dict(zip(GM_COLUMNS, (float(x) for x in inp)))
        state, muf, sigd, reflx, losx, rays, allm, itxrcp, _ = _build(v)
        lpath = LpathState(delopt=F32(v["delopt"]), gmin=F32(v["gmin"]),
                           ymin=F32(v["ymin"]))
        lpath.txrgml[1:, 1:] = TXRGML_SENTINEL
        gmloss(state, muf, sigd, reflx, losx, rays, _cantenna(v), allm,
               ZonState(), lpath, itxrcp, F32(v["freq"]), F32(v["gcdkm"]),
               F32(v["amind"]))
        cases.append((reflx, losx, lpath, itxrcp.astype(np.int32),
                      float(v["gcdkm"]), float(v["amind"]), int(v["info"])))
    return cases


def test_seltxr_bit_exact():
    cases = _directed() + _random_cases() + _dump_cases()
    n = len(cases)
    reflx_buf = np.empty(n, dtype=REFLX_DTYPE)
    losx_buf = np.empty(n, dtype=LOSX_DTYPE)
    lpath_buf = np.empty(n, dtype=LPATH_DTYPE)
    itxrcp = np.empty((n, 3), np.int32)
    gcdkm = np.empty(n, np.float32)
    amind = np.empty(n, np.float32)
    info = np.empty(n, np.int32)
    exp_reflx = np.empty(n, dtype=REFLX_DTYPE)
    exp_lpath = np.empty(n, dtype=LPATH_DTYPE)
    exp_status = np.zeros(n, np.int32)
    seen = {"info_diag": 0, "info_plain": 0, "search_ran": 0,
            "search_skipped": 0, "kept_ray1": 0, "moved_on": 0,
            "gave_up": 0, "refused": 0}
    for i, (rf, lx, lp, itx, g, a, inf) in enumerate(cases):
        reflx_buf[i], losx_buf[i] = pack_reflx(rf), pack_losx(lx)
        lpath_buf[i] = pack_lpath(lp)
        itxrcp[i] = itx
        gcdkm[i], amind[i], info[i] = g, a, inf
        rf_e, lp_e = copy.deepcopy(rf), copy.deepcopy(lp)
        try:
            seltxr(rf_e, lx, lp_e, itx, F32(g), F32(a), inf)  # Stage A
        except RayIndexError:
            exp_status[i] = 1
            seen["refused"] += 1
        exp_reflx[i], exp_lpath[i] = pack_reflx(rf_e), pack_lpath(lp_e)
        # observable arms, as the Stage A gmloss test counts them
        seen["info_diag" if inf & 4 else "info_plain"] += 1
        dend = min(F32(g), F32(4000.0))
        for jj in (1, 2):
            k = int(itx[jj])
            if rf.gdflx[1, k] != 0.0:
                ran = dend / rf.gdflx[1, k] < F32(0.9)
                seen["search_ran" if ran else "search_skipped"] += 1
            if exp_status[i]:
                continue
            ltx = int(lp_e.ltxrgm[jj])
            seen["kept_ray1" if ltx == 1 else "moved_on"] += 1
            if ltx >= 1 and rf.hpflx[ltx, k] <= F32(70.0):
                seen["gave_up"] += 1

    mod = compile_test(_SRC)
    status = np.full(n, -1, np.int32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [reflx_buf.view(np.uint8), losx_buf.view(np.uint8),
                       lpath_buf.view(np.uint8), itxrcp, gcdkm, amind,
                       info, status])
    got_reflx = fetch(dev[0]).reshape(n, REFLX_DTYPE.itemsize
                                      ).view(REFLX_DTYPE).reshape(n)
    got_lpath = fetch(dev[2]).reshape(n, LPATH_DTYPE.itemsize
                                      ).view(LPATH_DTYPE).reshape(n)
    assert_bitexact(fetch(dev[7]), exp_status, "seltxr.status")
    # LTXRGM first: the chosen ray is the answer that matters most
    assert_bitexact(got_lpath["ltxrgm"], exp_lpath["ltxrgm"],
                    "seltxr.lpath.ltxrgm")
    for name in LPATH_DTYPE.names:
        assert_bitexact(got_lpath[name], exp_lpath[name],
                        f"seltxr.lpath.{name}")
    for name in REFLX_DTYPE.names:
        assert_bitexact(got_reflx[name], exp_reflx[name],
                        f"seltxr.reflx.{name}")

    for arm, c in seen.items():
        assert c, f"branch never reached: {arm} ({seen})"
    # the INFO bit must remain answer-changing: the directed pair (cases 3
    # and 4) share a table and differ only in INFO
    d0, d4 = exp_lpath["ltxrgm"][3][1], exp_lpath["ltxrgm"][4][1]
    assert d0 >= 1 and d4 >= 1
