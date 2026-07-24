"""ALOSFV Stage C vs Stage A — the reflectrix deviative-loss table.

Two tests.  The first isolates ALOSFV: each tier-B alosfv dump case (plus
jittered variants) is built by Stage A's LECDEN + GENION in-process — so
ALOSFV's YI(2,K) input is LECDEN's ADJUSTED value, exactly the Stage A
signature contract — snapshot-packed with its MufState (HPMUF/HTMUF), and
the device ALOSFV's whole mutated LayerState is compared bit-exactly against
Stage A's.  The second runs the WHOLE chain LECDEN -> GENION -> ALOSFV on
the device from the raw layer parameters, proving the three ports compose,
with the same rare numpy-``**2``-power carve-out test_lecden.py documents.

Coverage mirrors the Stage A dump test: both IEDP branches, F1 present and
absent, nonzero AFAC cells (the collision-frequency EXP path) and zeroed
below-MUF cells all asserted present.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.alosfv import alosfv
from voacap_gpu.scalar.f32math import F32, sqrt32
from voacap_gpu.scalar.genion import genion
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.lecden import lecden
from voacap_gpu.scalar.mufstate import MufState
from voacap_gpu.cuda.state import (LAYER_DTYPE, MUF_DTYPE, pack_layer,
                                      pack_muf)

from tests.scalar.dumputil import load_dump
from tests.cuda.harness import (compile_test, fetch, launch_rows,
                                   requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("alosfv")

_SRC = r'''
#include "lecden.cuh"
#include "genion.cuh"
#include "alosfv.cuh"

extern "C" __global__ void drive(LayerState* states, const MufState* mufs,
                                 int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    alosfv(&states[t], &mufs[t], 1);
}

extern "C" __global__ void chain(LayerState* states, const MufState* mufs,
                                 int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    lecden(&states[t], 1);
    genion(&states[t], 1);
    alosfv(&states[t], &mufs[t], 1);
}
'''


def _cases():
    """(16-value rows: 9 layer params, iedp, hpmuf1..3, htmuf1..3)."""
    rows = [[float(x) for x in inp] for inp in DUMP[1]]
    rng = np.random.default_rng(61)
    for _ in range(40):  # jittered variants stay in the dump's valid domain
        base = rows[int(rng.integers(0, len(DUMP[1])))]
        jit = [v * (1.0 + rng.uniform(-0.02, 0.02)) if v != 0.0 else 0.0
               for v in base]
        jit[9] = base[9]  # iedp is a flag, not a float to jitter
        rows.append(jit)
    return rows


def _build(row):
    """Stage A LECDEN + GENION chain for one case -> (state, muf)."""
    s = LayerState(km=1)
    s.iedp = int(round(row[9]))
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = (F32(v) for v in row[0:3])
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = (F32(v) for v in row[3:6])
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = (F32(v) for v in row[6:9])
    m = MufState()
    m.hpmuf[1], m.hpmuf[2], m.hpmuf[3] = (F32(v) for v in row[10:13])
    m.htmuf[1], m.htmuf[2], m.htmuf[3] = (F32(v) for v in row[13:16])
    lecden(s, 1)
    genion(s, 1)
    return s, m


def _pow2_risk(row):
    """test_lecden.py's carve-out: the case reaches Stage A's
    ``(fi1/fi2) ** 2`` and numpy's f32 power differs there from t*t."""
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3 = (F32(v) for v in row[:9])
    if not fi2 > 0.0:
        return False
    hb2 = hi3 - yi3
    fc1 = fi2 * fi2
    fc2 = fi3 * fi3
    htw = hb2 + yi3 * (F32(1.0) - sqrt32(F32(1.0) - fc1 / fc2))
    if htw > hi2 + F32(0.001):
        return False
    if hb2 - (hi2 - yi2) < 0.0:
        return False
    t = fi1 / fi2
    return (t ** 2).tobytes() != (t * t).tobytes()


def _assert_states_bitexact(got_u8, expected, name, skip=frozenset()):
    got = np.frombuffer(got_u8.tobytes(), dtype=expected.dtype)
    nc = len(expected)
    bad = {}
    for fname in expected.dtype.names:
        g = np.ascontiguousarray(got[fname]).view(np.uint32).reshape(nc, -1)
        e = np.ascontiguousarray(
            expected[fname]).view(np.uint32).reshape(nc, -1)
        for j in np.flatnonzero((g != e).any(axis=1)):
            bad.setdefault(int(j), []).append(fname)
    unexplained = {j: fs for j, fs in bad.items() if j not in skip}
    assert not unexplained, (
        f"{name}: {len(unexplained)}/{nc} cases differ from Stage A: "
        + "; ".join(f"case {j}: {fs}"
                    for j, fs in sorted(unexplained.items())[:10]))


def _coverage(rows, afacs):
    f1 = np.array([r[1] > 0.0 for r in rows])
    iedp = np.array([int(round(r[9])) for r in rows])
    assert f1.any(), "F1-present branch unreached"
    assert (~f1).any(), "no-F1 branch unreached"
    assert (iedp < 0).any() and (iedp >= 0).any()
    nonzero = int((afacs > 0.0).sum())
    zero = int((afacs == 0.0).sum())
    assert nonzero > 0, "collision-frequency EXP path unreached"
    assert zero > 0, "below-MUF zero path unreached"
    return nonzero


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_alosfv_dumps.sh")
def test_alosfv_bit_exact():
    rows = _cases()
    n = len(rows)
    inputs = np.zeros(n, dtype=LAYER_DTYPE)
    mufs = np.zeros(n, dtype=MUF_DTYPE)
    expected = np.zeros(n, dtype=LAYER_DTYPE)
    afacs = np.zeros((n, 30), np.float32)
    for j, row in enumerate(rows):
        s, m = _build(row)
        inputs[j] = pack_layer(s)  # snapshot BEFORE alosfv
        mufs[j] = pack_muf(m)
        alosfv(s, m, 1)            # Stage A in-process
        expected[j] = pack_layer(s)
        afacs[j] = s.afac[1:31, 1]

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [inputs.view(np.uint8), mufs.view(np.uint8)])
    _assert_states_bitexact(fetch(dev[0]), expected, "alosfv")
    _coverage(rows, afacs)


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_alosfv_dumps.sh")
def test_chain_lecden_genion_alosfv_bit_exact():
    """The full device chain from raw layer parameters composes bit-exactly:
    LECDEN's adjusted YI(2) flows into GENION's ionogram and ALOSFV's loss
    table inside one kernel, matching the same Stage A chain."""
    rows = _cases()
    n = len(rows)
    inputs = np.zeros(n, dtype=LAYER_DTYPE)
    mufs = np.zeros(n, dtype=MUF_DTYPE)
    expected = np.zeros(n, dtype=LAYER_DTYPE)
    flagged = set()
    for j, row in enumerate(rows):
        s = LayerState(km=1)
        s.iedp = int(round(row[9]))
        s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = (F32(v) for v in row[0:3])
        s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = (F32(v) for v in row[3:6])
        s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = (F32(v) for v in row[6:9])
        m = MufState()
        m.hpmuf[1], m.hpmuf[2], m.hpmuf[3] = (F32(v) for v in row[10:13])
        m.htmuf[1], m.htmuf[2], m.htmuf[3] = (F32(v) for v in row[13:16])
        inputs[j] = pack_layer(s)  # RAW state: device runs the whole chain
        mufs[j] = pack_muf(m)
        lecden(s, 1)
        genion(s, 1)
        alosfv(s, m, 1)
        expected[j] = pack_layer(s)
        if _pow2_risk(row):
            flagged.add(j)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("chain"), n,
                      [inputs.view(np.uint8), mufs.view(np.uint8)])
    if flagged:
        print(f"chain: {len(flagged)} case(s) carved out for the numpy "
              f"f32 **2 power-vs-multiply divergence: {sorted(flagged)}")
    assert len(flagged) / n < 0.02, "power-vs-multiply carve-out not rare"
    _assert_states_bitexact(fetch(dev[0]), expected, "lecden->genion->alosfv",
                            skip=flagged)
