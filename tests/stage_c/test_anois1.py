"""ANOIS1 Stage C vs Stage A, on the anois1 oracle grid.

The 1 MHz atmospheric noise at the receiver: two NOISY map evaluations at the
local-time blocks bracketing GMTR.  The whole /ANOIS/ struct is compared, not
just ATNU/ATNY -- the block indices KJ/JK and the times CC/TM are outputs too,
and on the no-data-base cases (F2D(1,1,1) <= 0) they must be left STALE, which
only a whole-struct compare against a sentinel-seeded entry state can prove.

The NOISY maps are shared, not per-thread, so they are stacked per distinct
(month, SSN) and indexed by thread, exactly as test_syssy.py stacks SYS/PERR.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.anois1 import anois1
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.noisestate import AnoisState
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_c.state import pack_anois

from tests.stage_a.dumputil import COEFFS_DIR, load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("anois1")
SENTINEL = F32(-8888.0)

_SRC = r'''
#include "anois1.cuh"

extern "C" __global__ void drive(AnoisState* anois, const float* fakp_all,
                                 const float* fakmap_all,
                                 const float* hmym_all,
                                 const float* fakabp_all,
                                 const float* abmap_all, const int* mi,
                                 const float* f2d111, const float* gmtr,
                                 const float* rlong, const float* rlongd,
                                 const float* rlat, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int m = mi[i];
    anois1(&anois[i], fakp_all + m * (29 * 16 * 6), fakmap_all + m * (29 * 16),
           hmym_all + m * (29 * 16), fakabp_all + m * (2 * 6),
           abmap_all + m * (2 * 3), f2d111[i], gmtr[i], rlong[i], rlongd[i],
           rlat[i]);
}
'''


def _entry_state():
    """A fully sentinel-seeded /ANOIS/ so a stale field is visibly stale."""
    return AnoisState(atnu=SENTINEL, atny=SENTINEL, cc=SENTINEL,
                      tm=SENTINEL, rcnse=SENTINEL, du=SENTINEL, dl=SENTINEL,
                      sigm=SENTINEL, sygu=SENTINEL, sygl=SENTINEL,
                      kj=-77, jk=-88)


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_anois1_bit_exact_vs_stage_a():
    _, inputs, _ = DUMP
    cases = [(int(r[0]), r[1], F32(r[2]), F32(r[3]), F32(r[4]), F32(r[5]),
              int(r[6])) for r in inputs]           # month ssn gmtr rlat
    n = len(cases)                                  # rlong rlongd izf

    keys = sorted({(c[0], c[1]) for c in cases})
    maps = {k: redmap(COEFFS_DIR, "CCIR", k[0], F32(k[1])) for k in keys}
    kidx = {k: i for i, k in enumerate(keys)}
    order = [maps[k] for k in keys]

    def stack(attr):
        return np.ascontiguousarray(
            np.stack([np.asarray(getattr(s, attr), np.float32)
                      for s in order])).ravel()

    fakp = stack("fakp")
    fakmap = stack("fakmap")
    hmym = stack("hmym")
    fakabp = stack("fakabp")
    abmap = stack("abmap")

    mi = np.array([kidx[(c[0], c[1])] for c in cases], np.int32)
    f2d111 = np.array([F32(0.0) if c[6] else maps[(c[0], c[1])].f2d[0, 0, 0]
                       for c in cases], np.float32)
    gmtr = np.array([c[2] for c in cases], np.float32)
    rlat = np.array([c[3] for c in cases], np.float32)
    rlong = np.array([c[4] for c in cases], np.float32)
    rlongd = np.array([c[5] for c in cases], np.float32)
    izf = np.array([c[6] for c in cases], np.int32)

    pre, exp = [], []
    for k, c in enumerate(cases):
        a = _entry_state()
        pre.append(pack_anois(a))
        st = maps[(c[0], c[1])]
        if c[6]:            # force the no-data branch without touching cache
            st = _NoData()
        anois1(a, st, gmtr[k], rlong[k], rlongd[k], rlat[k])
        exp.append(pack_anois(a))
    exp = np.stack(exp)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [np.stack(pre).view(np.uint8), fakp, fakmap, hmym,
                       fakabp, abmap, mi, f2d111, gmtr, rlong, rlongd, rlat])
    got = fetch(dev[0]).view(exp.dtype)

    for name in exp.dtype.names:            # whole struct, outputs and not
        assert_bitexact(got[name], exp[name], f"anois.{name}", cases=cases)

    # the no-data early return really did leave CC/TM/KJ/JK stale
    stale = izf != 0
    assert stale.any() and (~stale).any(), "need both arms of the F2D test"
    assert (exp["cc"][stale] == SENTINEL).all()
    assert (exp["kj"][stale] == -77).all() and (exp["jk"][stale] == -88).all()
    assert (exp["atnu"][stale] == F32(0.0)).all()

    # branch coverage on the block arithmetic, over the live cases only
    live = ~stale
    cc = gmtr[live]
    assert (cc - F32(22.0) >= 0).any(), "KJ = 6 default arm unreached"
    assert (cc - F32(22.0) < 0).any(), "computed KJ arm unreached"
    tm = np.float32(4 * exp["kj"][live] - 2)
    assert (cc < tm).any() and (cc > tm).any() and (cc == tm).any(), \
        "one of the three CC-vs-TM arms unreached"
    assert (exp["jk"][live] == 6).any(), "the JK <= 0 wrap unreached"
    assert (exp["jk"][live] == 1).any(), "the JK > 6 wrap unreached"
    assert (rlong[live] < 0).any() and (rlong[live] >= 0).any(), \
        "one longitude hemisphere unreached"
    assert set(exp["kj"][live].tolist()) == set(range(1, 7)), "block coverage"


class _NoData:
    """Stage A's IZF stand-in: F2D(1,1,1) = 0 and nothing else touched."""
    f2d = np.zeros((1, 1, 1), dtype=np.float32)
