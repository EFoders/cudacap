"""FINDF Stage C vs Stage A — the frequency ray-table search, one thread/case.

Cases come from the findf golden dump exactly as tests/stage_a/test_findf.py
builds them: ionogram via LECDEN+GENION plus the synthetic AFAC ramp, and the
ORACLE'S own reflectrix (ifob.csv) + NANG injected rather than recomputed —
the same isolation Stage A uses so FINDF's logic is tested independently of
FOBBY's integer-kHz quantisation floor.  Expected values are Stage A run
in-process.  The device gets packed LayerState/ReflxState/RaysState structs,
one triple per thread (FINDF calls PENANG on-device), and the assert is
bit-identity on EVERY ReflxState field — the whole ray table including stale
slots above IAFTXR, the skip/max scalars, DELPEN and IAFTXR.

Coverage (measured on this dump, asserted below): both the F1-present and
F1-absent layer sequences, skip rays from all three layers, the label-360
next-layer cusp insertion (DELFX + 0.001), and ray counts up to 44 — the
IAF > 45 table-full exit is not reached by this dump.
"""

import csv

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.findf import findf
from voacap_gpu.stage_a.genion import genion
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.lecden import lecden
from voacap_gpu.stage_a.raysstate import RaysState
from voacap_gpu.stage_a.reflxstate import ReflxState

from voacap_gpu.stage_c.state import DTYPES, PACK

from tests.stage_a.dumputil import GOLDEN, load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("findf")

_SRC = r'''
#include "findf.cuh"

extern "C" __global__ void drive(const LayerState* ls, ReflxState* rx,
                                 const RaysState* ry, const float* freq,
                                 const float* amind, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // rays_view_of() is the stride-1 view: byte-for-byte the packed struct
    // this test has always built, which is what proves the SoA refactor inert.
    findf(&ls[i], &rx[i], rays_view_of(&ry[i]), 1, freq[i], amind[i]);
}
'''


def _load_ifob():
    p = GOLDEN / "findf" / "ifob.csv"
    if not p.exists():
        return None
    with open(p, newline="") as f:
        return [[int(x) for x in row] for row in csv.reader(f)]


IFOB_ROWS = _load_ifob()


def _pack_cases(objs, sname):
    buf = np.zeros(len(objs), dtype=DTYPES[sname])
    for i, o in enumerate(objs):
        buf[i] = PACK[sname](o)
    return buf


def _build_case(inp, ifob_row):
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3 = (float(x) for x in inp[0:9])
    s = LayerState(km=1)
    s.iedp = int(inp[9])
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(fi1), F32(fi2), F32(fi3)
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(yi1), F32(yi2), F32(yi3)
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(hi1), F32(hi2), F32(hi3)
    lecden(s, 1)
    genion(s, 1)
    for ih in range(1, 31):
        s.afac[ih, 1] = F32(0.1) + F32(0.03) * F32(ih)
    r = RaysState()
    r.nang = ifob_row[0]                       # oracle NANG
    flat = ifob_row[1:]                        # 40x30 IFOB, IA outer, IH inner
    for ia in range(1, 41):
        for ih in range(1, 31):
            r.ifob[ia, ih, 1] = flat[(ia - 1) * 30 + (ih - 1)]
    return s, ReflxState(), r, np.float32(inp[12]), np.float32(inp[11])


@pytest.mark.skipif(DUMP is None or IFOB_ROWS is None,
                    reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_findf_dumps.sh")
def test_findf_bit_exact():
    header, inputs, _ = DUMP
    assert len(header) == 8 + 7 * 45
    layers, reflxs, rayss, freqs, aminds = [], [], [], [], []
    for inp, ifob_row in zip(inputs, IFOB_ROWS):
        s, rx, ry, fq, am = _build_case(inp, ifob_row)
        layers.append(s)
        reflxs.append(rx)
        rayss.append(ry)
        freqs.append(fq)
        aminds.append(am)
    n = len(layers)
    freqs = np.array(freqs, dtype=np.float32)
    aminds = np.array(aminds, dtype=np.float32)

    # pack device inputs BEFORE Stage A mutates the states
    ls_in = _pack_cases(layers, "LayerState")
    rx_in = _pack_cases(reflxs, "ReflxState")
    ry_in = _pack_cases(rayss, "RaysState")

    # expected: Stage A in-process on the same cases
    for s, rx, ry, fq, am in zip(layers, reflxs, rayss, freqs, aminds):
        findf(s, rx, ry, 1, fq, am)
    rx_exp = _pack_cases(reflxs, "ReflxState")

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [ls_in.view(np.uint8), rx_in.view(np.uint8),
                       ry_in.view(np.uint8), freqs, aminds])
    got = fetch(dev[1]).view(DTYPES["ReflxState"])
    assert got.shape == (n,)
    for name in DTYPES["ReflxState"].names:
        assert_bitexact(np.ascontiguousarray(got[name]),
                        np.ascontiguousarray(rx_exp[name]),
                        f"findf reflx.{name}")

    # branch coverage, measured on the Stage A expected values
    fi2 = np.array([s.fi[2, 1] for s in layers], dtype=np.float32)
    assert (fi2 <= 0).any() and (fi2 > 0).any()   # both layer sequences
    iaftxr = rx_exp["iaftxr"][:, 1]
    assert iaftxr.min() >= 1
    # skip rays chosen from all three layers
    assert set(rx_exp["iskp"][:, 1].tolist()) == {1, 2, 3}
    # label-360 next-layer cusp: DELFX(IAF) = DELFX(IAF-1) + .001 somewhere
    cusp = 0
    for i in range(n):
        d = rx_exp["delfx"][i, 1:iaftxr[i] + 1, 1]
        cusp += int((d[1:] == (d[:-1] + np.float32(0.001))).sum())
    assert cusp > 0
    # the IAF > 45 table-full exit is NOT reached by this dump — assert it
    # stays that way rather than claiming coverage.
    assert iaftxr.max() < 45
