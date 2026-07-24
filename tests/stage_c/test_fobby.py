"""FOBBY Stage C vs Stage A — the reflectrix build, one thread per case.

Cases come from the fobby golden dump exactly as tests/stage_a/test_fobby.py
builds them (ionogram via LECDEN+GENION, NANG=1 with the dumped take-off
angle); expected values are Stage A run in-process.  The device gets the
packed LayerState/RaysState structs, one pair per thread, and the assert is
bit-identity on EVERY RaysState field — the 30 IFOB integer-kHz cells FOBBY
writes (the ``(int)(1000.0f * freq)`` truncation), plus NANG/ANG and the
untouched IFOB remainder.  Both IEDP ionogram branches are in the case set
(asserted); the AMAX1(.000001, SQCOS) clamp is not exercised by physical
ionograms, exactly as in Stage A's dump.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.fobby import fobby
from voacap_gpu.stage_a.genion import genion
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.lecden import lecden
from voacap_gpu.stage_a.raysstate import RaysState

from voacap_gpu.stage_c.state import DTYPES, PACK

from tests.stage_a.dumputil import load_dump
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("fobby")

_SRC = r'''
#include "fobby.cuh"

extern "C" __global__ void drive(const LayerState* ls, RaysState* rs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // rays_view_of() is the stride-1 view: byte-for-byte the packed struct
    // this test has always built, which is what proves the SoA refactor inert.
    fobby(&ls[i], rays_view_of(&rs[i]), 1);
}
'''


def _pack_cases(objs, sname):
    buf = np.zeros(len(objs), dtype=DTYPES[sname])
    for i, o in enumerate(objs):
        buf[i] = PACK[sname](o)
    return buf


def _build_case(inp):
    (fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3, iedp, ang) = (
        float(x) for x in inp)
    s = LayerState(km=1)
    s.iedp = int(round(iedp))
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = F32(fi1), F32(fi2), F32(fi3)
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = F32(yi1), F32(yi2), F32(yi3)
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = F32(hi1), F32(hi2), F32(hi3)
    lecden(s, 1)
    genion(s, 1)
    rays = RaysState(nang=1)
    rays.ang[1] = F32(ang)
    return s, rays


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_fobby_dumps.sh")
def test_fobby_bit_exact():
    header, inputs, _ = DUMP
    assert len(header) == 30
    layers, rayss = [], []
    for inp in inputs:
        s, r = _build_case(inp)
        layers.append(s)
        rayss.append(r)
    n = len(layers)

    # pack device inputs BEFORE Stage A mutates the states
    ls_in = _pack_cases(layers, "LayerState")
    rs_in = _pack_cases(rayss, "RaysState")

    # expected: Stage A in-process on the same cases
    for s, r in zip(layers, rayss):
        fobby(s, r, 1)
    rs_exp = _pack_cases(rayss, "RaysState")

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [ls_in.view(np.uint8), rs_in.view(np.uint8)])
    got = fetch(dev[1]).view(DTYPES["RaysState"])
    assert got.shape == (n,)
    for name in DTYPES["RaysState"].names:
        assert_bitexact(np.ascontiguousarray(got[name]),
                        np.ascontiguousarray(rs_exp[name]),
                        f"fobby rays.{name}")

    # coverage: both IEDP ionogram branches and a spread of take-off angles
    iedps = np.array([int(inp[9]) for inp in inputs])
    assert (iedps < 0).any() and (iedps >= 0).any()
    angs = np.array([float(inp[10]) for inp in inputs])
    assert angs.min() < 1.0 and angs.max() > 45.0
