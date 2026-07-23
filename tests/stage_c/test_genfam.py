"""GENFAM Stage C vs Stage A — atmospheric-noise frequency dependence.

Driven on the tier-B `genfam` dump inputs (FAM/DUD stacked per month, one
index per thread), bit-exact against Stage A in-process — both sides take
ALOG10 and 2.**X through binary64, so unlike the oracle comparison there is
no libm residual to sit above.  The behavioural claims the Stage A test
carries — every table column 1..12 reached, the 20 MHz decile clamp and the
10 MHz sigma-Fam clamp holding the curve ends flat, and the hemisphere shift
— are asserted on the DEVICE outputs.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.genfam import genfam
from voacap_gpu.stage_a.redmap import redmap

from tests.stage_a.dumputil import COEFFS_DIR, load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("genfam")
OUTS = ("fa", "dua", "dla", "dms", "dus", "dls")

_SRC = r'''
#include "genfam.cuh"

extern "C" __global__ void drive(const float* fam_all, const float* dud_all,
                                 const int* mi, const float* y2,
                                 const int* iblk, const float* freq,
                                 const float* z, float* fa, float* dua,
                                 float* dla, float* dms, float* dus,
                                 float* dls, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    genfam(fam_all + mi[i] * 14 * 12, dud_all + mi[i] * 5 * 12 * 5,
           y2[i], iblk[i], freq[i], z[i],
           &fa[i], &dua[i], &dla[i], &dms[i], &dus[i], &dls[i]);
}
'''


def _run(mod, months, maps, cases):
    """cases: (month, y2, iblk, freq, z) tuples -> dict of device outputs."""
    midx = {m: i for i, m in enumerate(months)}
    n = len(cases)
    fam_all = np.ascontiguousarray(
        np.stack([maps[m].fam for m in months])).ravel()
    dud_all = np.ascontiguousarray(
        np.stack([maps[m].dud for m in months])).ravel()
    mi = np.array([midx[c[0]] for c in cases], np.int32)
    y2 = np.array([c[1] for c in cases], np.float32)
    iblk = np.array([c[2] for c in cases], np.int32)
    freq = np.array([c[3] for c in cases], np.float32)
    z = np.array([c[4] for c in cases], np.float32)
    outs = [np.zeros(n, np.float32) for _ in OUTS]
    dev = launch_rows(mod.get_function("drive"), n,
                      [fam_all, dud_all, mi, y2, iblk, freq, z] + outs)
    return {name: fetch(dev[7 + i]) for i, name in enumerate(OUTS)}


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_genfam_dumps.sh")
def test_genfam_bit_exact():
    _, inputs, _ = DUMP
    cases = [(int(r[0]), np.float32(r[1]), int(r[2]), np.float32(r[3]),
              np.float32(r[4])) for r in inputs]
    months = sorted({c[0] for c in cases})
    maps = {m: redmap(COEFFS_DIR, "CCIR", m, F32(100.0)) for m in months}

    expected = np.array([genfam(maps[m], y2, ib, fq, z)
                         for m, y2, ib, fq, z in cases], np.float32)

    mod = compile_test(_SRC)
    got = _run(mod, months, maps, cases)
    for i, name in enumerate(OUTS):
        assert_bitexact(got[name], expected[:, i], f"genfam {name}",
                        cases=cases)

    # every FAM/DUD table column must actually be selected
    blocks = {c[2] + (6 if c[1] < 0.0 else 0) for c in cases}
    assert blocks == set(range(1, 13)), f"unreached table column: {blocks}"
    freq = np.array([c[3] for c in cases], np.float32)
    assert (freq > 20.0).any(), "20 MHz decile clamp unreached"
    assert ((freq > 10.0) & (freq <= 20.0)).any(), "10 MHz DMS clamp unreached"
    assert (freq <= 10.0).any()


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_frequency_clamps_hold_the_curve_ends_on_device():
    """Report 322's curves stop at 20 MHz, sigma-Fam's at 10 — flat above."""
    months = [6]
    maps = {6: redmap(COEFFS_DIR, "CCIR", 6, F32(100.0))}
    mod = compile_test(_SRC)
    cases = [(6, np.float32(0.5), 3, np.float32(f), np.float32(60.0))
             for f in (20.0, 25.0, 40.0, 55.0, 10.0, 12.0, 18.0)]
    got = _run(mod, months, maps, cases)
    for k in (1, 2, 3):                      # 25, 40, 55 vs 20 MHz
        for name in ("dua", "dla", "dms", "dus", "dls"):
            assert got[name][k] == got[name][0], f"{name} moved above 20 MHz"
        assert got["fa"][k] != got["fa"][0], "FA must still track frequency"
    for k in (5, 6):                         # 12, 18 vs 10 MHz
        assert got["dms"][k] == got["dms"][4], "DMS moved above 10 MHz"
        assert got["dua"][k] != got["dua"][4], "DUA must track to 20 MHz"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_southern_hemisphere_shifts_the_table_column_on_device():
    """Only the SIGN of Y2 matters, and it selects blocks 7..12."""
    months = [2]
    maps = {2: redmap(COEFFS_DIR, "CCIR", 2, F32(100.0))}
    mod = compile_test(_SRC)
    cases = [(2, np.float32(y2), 2, np.float32(8.0), np.float32(70.0))
             for y2 in (0.01, 1.4, -0.01, -1.4)]
    got = _run(mod, months, maps, cases)
    for name in OUTS:
        assert got[name][0] == got[name][1], "north depends on |Y2|"
        assert got[name][2] == got[name][3], "south depends on |Y2|"
    assert any(got[name][0] != got[name][2] for name in OUTS)
    assert got["dua"][0] != got["dua"][2]
