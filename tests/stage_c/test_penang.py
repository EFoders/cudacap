"""PENANG Stage C vs Stage A — per-layer penetration angles, one thread/case.

Cases come from the penang golden dump exactly as tests/stage_a/test_penang.py
builds them (ionogram via LECDEN+GENION, DSKPKM/DMAXKM preset as FINDF does);
expected values are Stage A run in-process on the same cases.  The device gets
the packed LayerState/ReflxState structs (see state.py), one pair per thread,
and the assert is bit-identity on EVERY ReflxState field — DELPEN and the
skip/max flags penang writes, and the untouched remainder.

Coverage (measured on this dump, asserted below): E penetrated, F1 absent /
present / penetrated, F2 reflects (89.9), the 90-degree exits, the near-nose
probe that still reflects, and the CDEL > .999999 zero-angle clamp.  The F2
nose-fully-penetrates branch (label 175: DELPEN=0 with DSKPKM/DMAXKM=-1) is
NOT reached by this dump — asserted still unreachable rather than claimed.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.genion import genion
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.lecden import lecden
from voacap_gpu.stage_a.penang import penang
from voacap_gpu.stage_a.reflxstate import ReflxState

from voacap_gpu.stage_c.state import DTYPES, PACK

from tests.stage_a.dumputil import load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("penang")

_SRC = r'''
#include "penang.cuh"

extern "C" __global__ void drive(const LayerState* ls, ReflxState* rx,
                                 const float* freq, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    penang(&ls[i], &rx[i], 1, freq[i]);
}
'''


def _pack_cases(objs, sname):
    buf = np.zeros(len(objs), dtype=DTYPES[sname])
    for i, o in enumerate(objs):
        buf[i] = PACK[sname](o)
    return buf


def _build_case(inp):
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3 = (float(x) for x in inp[0:9])
    s = LayerState(km=1)
    s.iedp = int(inp[9])
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(fi1), F32(fi2), F32(fi3)
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(yi1), F32(yi2), F32(yi3)
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(hi1), F32(hi2), F32(hi3)
    lecden(s, 1)
    genion(s, 1)
    r = ReflxState()
    r.dskpkm[1] = F32(10000.0)
    r.dmaxkm[1] = F32(0.0)
    return s, r, np.float32(inp[10])


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_penang_dumps.sh")
def test_penang_bit_exact():
    header, inputs, _ = DUMP
    assert header == ["delpen1", "delpen2", "delpen3", "dskpkm", "dmaxkm"]
    layers, reflxs, freqs = [], [], []
    for inp in inputs:
        s, r, fq = _build_case(inp)
        layers.append(s)
        reflxs.append(r)
        freqs.append(fq)
    n = len(layers)
    freqs = np.array(freqs, dtype=np.float32)

    # pack device inputs BEFORE Stage A mutates the states
    ls_in = _pack_cases(layers, "LayerState")
    rx_in = _pack_cases(reflxs, "ReflxState")

    # expected: Stage A in-process on the same cases
    for s, r, fq in zip(layers, reflxs, freqs):
        penang(s, r, 1, fq)
    rx_exp = _pack_cases(reflxs, "ReflxState")

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [ls_in.view(np.uint8), rx_in.view(np.uint8), freqs])
    got = fetch(dev[1]).view(DTYPES["ReflxState"])
    assert got.shape == (n,)
    for name in DTYPES["ReflxState"].names:
        assert_bitexact(np.ascontiguousarray(got[name]),
                        np.ascontiguousarray(rx_exp[name]),
                        f"penang reflx.{name}")

    # branch coverage, measured on the Stage A expected values
    dp1 = rx_exp["delpen"][:, 1, 1]
    dp2 = rx_exp["delpen"][:, 2, 1]
    dp3 = rx_exp["delpen"][:, 3, 1]
    fi2 = np.array([s.fi[2, 1] for s in layers], dtype=np.float32)
    fv30 = np.array([s.fvert[30, 1] for s in layers], dtype=np.float32)
    assert (dp1 == np.float32(89.9)).any()          # E penetrated
    assert (fi2 <= 0).any() and (fi2 > 0).any()     # F1 absent / present
    assert ((dp2 == np.float32(89.9)) & (fi2 > 0)).any()   # F1 penetrated
    assert (dp3 == np.float32(89.9)).any()          # F2 reflects (label 160)
    assert (dp3 == np.float32(90.0)).any()          # 90-degree exit paths
    assert ((dp1 == 0.0) | (dp2 == 0.0)).any()      # CDEL > .999999 clamp
    # label-156 nose probe that still reflects (0 < DELPEN3 < 89.9)
    nose = (fv30 - freqs + np.float32(0.0001)) < 0
    assert (nose & (dp3 > 0) & (dp3 < np.float32(89.9))).any()
    # label 175 (F2 nose fully penetrates, DSKPKM/DMAXKM = -1): UNREACHABLE
    # in this dump — assert it stays that way rather than claiming coverage.
    assert not (rx_exp["dskpkm"][:, 1] == np.float32(-1.0)).any()
