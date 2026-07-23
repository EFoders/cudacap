"""SYSSY Stage C vs Stage A — excess-system-loss decile table lookup.

Driven on the tier-B `syssy` dump inputs (SYS/PERR stacked per month, one
index per thread) plus a deterministic extension that forces the index
clamps the dump happens not to pin — KJ<=0 and KJ>9, and the J=0 wraps
LJ->8 / LD->4 — so every arm of the KJ/LOL/LD/ND arithmetic is asserted
reached, not assumed.  Pure table reads and two divisions: bit-exact, no
residual.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_a.syssy import syssy

from tests.stage_a.dumputil import COEFFS_DIR, load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("syssy")
OUTS = ("fm", "su", "sl", "fmp", "sup", "slp")

_SRC = r'''
#include "syssy.cuh"

extern "C" __global__ void drive(const float* sys_all, const float* perr_all,
                                 const int* mi, const float* f2d111,
                                 const float* g, const float* pt,
                                 const int* nn, float* fm, float* su,
                                 float* sl, float* fmp, float* sup,
                                 float* slp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    syssy(sys_all + mi[i] * 9 * 16 * 6, perr_all + mi[i] * 9 * 4 * 6,
          f2d111[i], g[i], pt[i], nn[i],
          &fm[i], &su[i], &sl[i], &fmp[i], &sup[i], &slp[i]);
}
'''

# extension cases (month, g, pt, nn, izf): the clamps and wraps
_EXTRA = [
    (6, 0.0, 0.2, 2, 0),      # KJ raw < 0 -> 1;  J=0 -> LJ wrap 8, LD wrap 4
    (6, -0.05, 0.2, 5, 0),    # same wraps, southern hemisphere
    (6, 1.6, 12.0, 2, 0),     # KJ raw > 9 -> 9
    (6, -1.6, 6.5, 5, 0),     # KJ clamp, southern
    (6, 0.9, 23.7, 2, 0),     # J=24: top LJ/LD blocks without wrapping
    (6, 0.9, 5.0, 5, 1),      # no-data fallback (its own SL/1.29)
    (6, -0.9, 18.0, 2, 1),    # fallback again, southern inputs ignored
]


def _cases():
    _, inputs, _ = DUMP
    rows = [(int(r[0]), np.float32(r[1]), np.float32(r[2]), int(r[3]),
             int(r[4])) for r in inputs]
    rows += [(m, np.float32(g), np.float32(p), nn, izf)
             for m, g, p, nn, izf in _EXTRA]
    return rows


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_syssy_dumps.sh")
def test_syssy_bit_exact():
    rows = _cases()
    n = len(rows)
    months = sorted({r[0] for r in rows})
    maps = {m: redmap(COEFFS_DIR, "CCIR", m, F32(100.0)) for m in months}
    midx = {m: i for i, m in enumerate(months)}

    sys_all = np.ascontiguousarray(
        np.stack([maps[m].sys for m in months])).ravel()
    perr_all = np.ascontiguousarray(
        np.stack([maps[m].perr for m in months])).ravel()
    mi = np.array([midx[r[0]] for r in rows], np.int32)
    g = np.array([r[1] for r in rows], np.float32)
    pt = np.array([r[2] for r in rows], np.float32)
    nn = np.array([r[3] for r in rows], np.int32)
    izf = np.array([r[4] for r in rows], np.int32)
    f2d111 = np.array([0.0 if r[4] else maps[r[0]].f2d[0, 0, 0]
                       for r in rows], np.float32)

    expected = np.zeros((n, 6), np.float32)
    for k, (m, gk, ptk, nnk, izfk) in enumerate(rows):
        st = maps[m]
        if izfk:  # force the no-data branch without touching the cache
            st = SimpleNamespace(f2d=np.zeros((1, 1, 1), dtype=np.float32),
                                 sys=st.sys, perr=st.perr)
        expected[k] = syssy(st, gk, ptk, nnk)

    mod = compile_test(_SRC)
    outs = [np.zeros(n, np.float32) for _ in OUTS]
    dev = launch_rows(mod.get_function("drive"), n,
                      [sys_all, perr_all, mi, f2d111, g, pt, nn] + outs)
    for i, name in enumerate(OUTS):
        assert_bitexact(fetch(dev[7 + i]), expected[:, i], f"syssy {name}",
                        cases=rows)

    # branch coverage, in the routine's own float32 index arithmetic
    live = izf == 0
    assert live.any() and (~live).any(), "need both arms of the no-data test"
    assert (g[live] < 0).any() and (g[live] >= 0).any(), "one hemisphere only"
    assert set(nn[live].tolist()) == {2, 5}, "NN must span short and long"
    gg = np.abs(g) * F32(57.29577)
    kj_raw = np.trunc((gg - F32(40.0)) / F32(5.0) + F32(1.5))
    assert (kj_raw[live] <= 0).any(), "KJ<=0 clamp unreached"
    assert (kj_raw[live] > 9).any(), "KJ>9 clamp unreached"
    assert ((kj_raw[live] > 0) & (kj_raw[live] <= 9)).any()
    j = np.trunc(pt + F32(0.5)).astype(np.float32)
    lj_raw = np.trunc(j / F32(3.0) + F32(0.67))
    ld_raw = np.trunc(j / F32(6.0) + F32(0.67))
    assert (lj_raw[live] - 1 < 0).any(), "LJ->8 wrap unreached"
    assert (ld_raw[live] - 1 < 0).any(), "LD->4 wrap unreached"
    assert (lj_raw[live] >= 1).any() and (ld_raw[live] >= 1).any()
