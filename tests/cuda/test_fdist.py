"""FDIST Stage C vs Stage A — up to six modes per hop, one thread per case.

Cases come from the fdist golden dump exactly as tests/stage_a/test_fdist.py
builds them: the /REFLX/ ray table is injected verbatim from the dump (the
same isolation Stage A uses to decouple FDIST from FOBBY/FINDF), GHOP is the
oracle's, and expected values are Stage A run in-process.  The device gets
packed ReflxState/ModesState structs plus per-case FREQ/AMIND, one thread per
case, and the assert is bit-identity on NUMMOD and EVERY ModesState field —
all six mode slots including the HPMOD=-1/ITMOD=5 presets of unused slots.

Coverage (measured on this dump, asserted below): the DHOPKM >= DMAXKM early
exit, mode counts 0..5, and the label-185 interpolation path with all three
layers appearing in ITMOD.  Two branches are NOT reached by this dump and are
asserted still unreachable rather than claimed: the label-165 take-the-table-
entry-as-is path (every found mode is interpolated), and the label-305
below-AMIND rejection (no negated interpolated HPMOD appears).
"""

import numpy as np
import pytest

from voacap_gpu.scalar.fdist import fdist
from voacap_gpu.scalar.modesstate import ModesState
from voacap_gpu.scalar.reflxstate import ReflxState

from voacap_gpu.cuda.state import DTYPES, PACK

from tests.scalar.dumputil import load_dump
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("fdist")

_RAY_BASE = 2               # dmaxkm, ghop, then the 7 ray arrays
_OUT_BASE = _RAY_BASE + 7 * 45   # index of nummod (unused here; layout doc)

_SRC = r'''
#include "fdist.cuh"

extern "C" __global__ void drive(const ReflxState* rx, ModesState* md,
                                 const float* freq, const float* amind,
                                 int* nummod, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    nummod[i] = fdist(&rx[i], &md[i], 1, freq[i], amind[i]);
}
'''


def _pack_cases(objs, sname):
    buf = np.zeros(len(objs), dtype=DTYPES[sname])
    for i, o in enumerate(objs):
        buf[i] = PACK[sname](o)
    return buf


def _build_case(inp, exp):
    amind, freq = np.float32(inp[11]), np.float32(inp[12])
    reflx = ReflxState()
    reflx.dmaxkm[1] = np.float32(exp[0])
    for b, arr in enumerate([reflx.delfx, reflx.gdflx, reflx.hpflx,
                             reflx.htflx, reflx.fvflx, reflx.afflx]):
        base = _RAY_BASE + b * 45
        for j in range(1, 46):
            arr[j, 1] = np.float32(exp[base + j - 1])
    base = _RAY_BASE + 6 * 45
    for j in range(1, 46):
        reflx.imode[j, 1] = int(round(float(exp[base + j - 1])))
    modes = ModesState()
    modes.ghop = np.float32(exp[1])   # GHOP the oracle used
    return reflx, modes, freq, amind


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_fdist_dumps.sh")
def test_fdist_bit_exact():
    header, inputs, rows = DUMP
    assert len(header) == 2 + 7 * 45 + 1 + 6 * 6
    reflxs, modess, freqs, aminds = [], [], [], []
    for inp, exp in zip(inputs, rows):
        rx, md, fq, am = _build_case(inp, exp)
        reflxs.append(rx)
        modess.append(md)
        freqs.append(fq)
        aminds.append(am)
    n = len(reflxs)
    freqs = np.array(freqs, dtype=np.float32)
    aminds = np.array(aminds, dtype=np.float32)

    # pack device inputs BEFORE Stage A mutates the states
    rx_in = _pack_cases(reflxs, "ReflxState")
    md_in = _pack_cases(modess, "ModesState")

    # expected: Stage A in-process on the same cases
    nummod_exp = np.array(
        [fdist(rx, md, 1, fq, am)
         for rx, md, fq, am in zip(reflxs, modess, freqs, aminds)],
        dtype=np.int32)
    md_exp = _pack_cases(modess, "ModesState")

    mod = compile_test(_SRC)
    nummod_out = np.zeros(n, dtype=np.int32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [rx_in.view(np.uint8), md_in.view(np.uint8),
                       freqs, aminds, nummod_out])
    assert_bitexact(fetch(dev[4]), nummod_exp, "fdist nummod")
    got = fetch(dev[1]).view(DTYPES["ModesState"])
    assert got.shape == (n,)
    for name in DTYPES["ModesState"].names:
        assert_bitexact(np.ascontiguousarray(got[name]),
                        np.ascontiguousarray(md_exp[name]),
                        f"fdist modes.{name}")

    # branch coverage, measured on the Stage A expected values
    dhopkm = md_exp["ghop"] * np.float32(6370.0)
    dmaxkm = rx_in["dmaxkm"][:, 1]
    assert (dhopkm >= dmaxkm).any()               # early exit taken
    assert (dhopkm < dmaxkm).any()                # ... and the search taken
    assert set(nummod_exp.tolist()) == {0, 1, 2, 3, 4, 5}
    hp = md_exp["hpmod"][:, 1:7, 1]
    it = md_exp["itmod"][:, 1:7, 1]
    assert set(it[hp > 0].tolist()) == {1, 2, 3}  # all layers found
    assert (hp == np.float32(-1.0)).any()         # unused-slot preset intact
    # UNREACHABLE in this dump (assert rather than claim): the label-165
    # exact-table-entry path — every found mode's DELMOD is re-derived by the
    # label-185 geometry, never copied from DELFX — and the label-305
    # below-AMIND rejection (no negated interpolated HPMOD, only the -1s).
    delfx_in = rx_in["delfx"][:, :, 1]
    for i in range(n):
        found = hp[i] > 0
        assert not np.isin(md_exp["delmod"][i, 1:7, 1][found],
                           delfx_in[i]).any()
    assert not ((hp < 0) & (hp != np.float32(-1.0))).any()
