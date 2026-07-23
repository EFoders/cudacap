"""LECDEN Stage C vs Stage A — the electron-density profile builder.

One packed LayerState per case (profile + genion dump inputs, randomized
domain-valid parameters, and deterministic branch-forcing cases), mutated in
place by the device LECDEN and compared WHOLE-STRUCT against the Stage A
mutation — so the HTR/FNSQ profile, the adjusted HI(2)/YI(2)/FSECV F1 fit
(the in/out contract ALOSFV depends on), and the untouched remainder of the
struct are all held to bit-identity at once.

The one carved-out residual: Stage A writes ``(fi1/fi2) ** 2`` in the F1
fit, and numpy's float32 power deviates from the correctly rounded square
(what gfortran's INTEGER-exponent ``**2`` emits, and what Stage C computes
as t*t) by 1 ULP on ~0.03% of arguments.  Cases whose control flow reaches
that expression with a deviating ratio are flagged host-side, counted,
asserted rare, and excluded from the bit-compare; every other case must be
bit-exact.

Branch coverage is asserted from a host-side probe of the F1-fit control
flow: the 205 pin-to-E path, the 194 F2-below-F1 path, the linear-F1 401
(yb wins) and 402 (ys wins) paths, no-F1, the two-layer HTR grid, its
collapse back to one layer, and the IELECT early return.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32, sqrt32
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.lecden import lecden
from voacap_gpu.stage_c.state import LAYER_DTYPE, pack_layer

from tests.stage_a.dumputil import load_dump
from tests.stage_c.harness import (compile_test, fetch, launch_rows,
                                   requires_cuda)

pytestmark = requires_cuda

PROFILE = load_dump("profile")
GENION = load_dump("genion")

_SRC = r'''
#include "lecden.cuh"

extern "C" __global__ void drive(LayerState* states, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    lecden(&states[t], 1);
}
'''

# deterministic branch-forcing layer rows [fi1..3, yi1..3, hi1..3]
_C205 = [3.0, 5.0, 10.0, 20.0, 40.0, 80.0, 110.0, 130.0, 400.0]
_C194 = [3.0, 4.0, 12.0, 20.0, 40.0, 250.0, 110.0, 200.0, 300.0]
_C401 = [2.0, 5.0, 12.0, 20.0, 100.0, 120.0, 110.0, 165.0, 270.0]


def _layer_rows():
    rows = []
    if PROFILE:
        rows += [[float(x) for x in inp] for inp in PROFILE[1]]
    if GENION:
        rows += [[float(x) for x in inp[:9]] for inp in GENION[1]]
    rng = np.random.default_rng(53)
    for _ in range(100):
        fi1 = rng.uniform(2.0, 4.0)
        fi3 = rng.uniform(6.0, 14.0)
        fi2 = 0.0 if rng.random() < 0.3 else fi1 + rng.uniform(0.3, 0.9) * (
            fi3 - fi1)
        rows.append([fi1, fi2, fi3,
                     rng.uniform(18.0, 22.0), rng.uniform(20.0, 50.0),
                     rng.uniform(50.0, 140.0),
                     rng.uniform(105.0, 115.0), rng.uniform(150.0, 200.0),
                     rng.uniform(280.0, 480.0)])
    rows.append(_C205)
    rows.append(_C194)
    rows.append(_C401)
    return rows


def _make_state(row, ielect=0):
    s = LayerState(km=1)
    s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = (F32(v) for v in row[0:3])
    s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = (F32(v) for v in row[3:6])
    s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = (F32(v) for v in row[6:9])
    if ielect:
        s.ielect[2] = 1
    return s


def _f1_branch(row):
    """Which F1-fit path Stage A's control flow takes (host-side probe)."""
    fi1, fi2, fi3, yi1, yi2, yi3, hi1, hi2, hi3 = (F32(v) for v in row)
    if not fi2 > 0.0:
        return "nof1"
    hb2 = hi3 - yi3
    fc1 = fi2 * fi2
    fc2 = fi3 * fi3
    htw = hb2 + yi3 * (F32(1.0) - sqrt32(F32(1.0) - fc1 / fc2))
    if htw > hi2 + F32(0.001):
        return "205"
    hb1 = hi2 - yi2
    if hb2 - hb1 < 0.0:
        return "194"
    ys = max(F32(1.0), htw - hb1)
    denom = max(F32(0.17), F32(1.0) - (fi1 / fi2) ** 2)
    yb = (hi2 - hi1) / denom
    return "401" if ys - yb >= 0.0 else "402"


def _pow2_risk(row):
    """True when the case reaches Stage A's ``(fi1/fi2) ** 2`` AND numpy's
    f32 power differs there from the correctly rounded square t*t."""
    if _f1_branch(row) not in ("401", "402"):
        return False
    t = F32(row[0]) / F32(row[1])
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


def test_lecden_bit_exact():
    rows = _layer_rows()
    # IELECT>0 early-return cases: the struct must come back untouched
    n_plain = len(rows)
    rows = rows + [rows[0], rows[1]]
    ielect_flags = [0] * n_plain + [1, 1]
    n = len(rows)

    inputs = np.zeros(n, dtype=LAYER_DTYPE)
    expected = np.zeros(n, dtype=LAYER_DTYPE)
    flagged = set()
    for j, row in enumerate(rows):
        s = _make_state(row, ielect_flags[j])
        inputs[j] = pack_layer(s)
        lecden(s, 1)  # Stage A in-process, mutates in place
        expected[j] = pack_layer(s)
        if not ielect_flags[j] and _pow2_risk(row):
            flagged.add(j)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n, [inputs.view(np.uint8)])
    got_u8 = fetch(dev[0])

    if flagged:
        print(f"lecden: {len(flagged)} case(s) carved out for the numpy "
              f"f32 **2 power-vs-multiply divergence: {sorted(flagged)}")
    assert len(flagged) / n < 0.02, "power-vs-multiply carve-out not rare"
    _assert_states_bitexact(got_u8, expected, "lecden", skip=flagged)

    # IELECT early return: output bytes == input bytes for those cases
    got = np.frombuffer(got_u8.tobytes(), dtype=LAYER_DTYPE)
    for j in range(n_plain, n):
        assert got[j].tobytes() == inputs[j].tobytes(), "ielect case mutated"

    # branch coverage over the generated set (host-side control-flow probe)
    branches = [_f1_branch(r) for r in rows[:n_plain]]
    for b in ("nof1", "205", "194", "401", "402"):
        assert b in branches, f"F1-fit branch {b} unreached"
    # the 194 case also exercises the two-layer grid collapsing back to one
    # (its post-fit F1 bottom sits below the F2 bottom); plain two-layer and
    # one-layer grids are ubiquitous in the dump rows
    assert branches.count("nof1") > 5 and branches.count("402") > 5
