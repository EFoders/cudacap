"""GENION Stage C vs Stage A — the Module-4 ionogram generator.

Cases are the tier-B genion dump inputs (both IEDP branches, F1 present and
absent) plus randomized domain-valid layer parameters.  Each case is built
into a LayerState by Stage A's own LECDEN in-process, snapshot-packed as the
device input, and the device GENION's whole mutated struct is compared
bit-exactly against Stage A's GENION on the identical post-LECDEN state —
FVERT, HPRIM, HTRUE, and the untouched remainder all at once.  Because the
device consumes the host-LECDEN profile, no numpy-power carve-out applies
here; the assert is bit-exact with no residual.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.genion import genion
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.lecden import lecden
from voacap_gpu.stage_c.state import LAYER_DTYPE, pack_layer

from tests.stage_a.dumputil import load_dump
from tests.stage_c.harness import (compile_test, fetch, launch_rows,
                                   requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("genion")

_SRC = r'''
#include "genion.cuh"

extern "C" __global__ void drive(LayerState* states, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    genion(&states[t], 1);
}
'''


def _cases():
    rows = []
    if DUMP:
        rows += [[float(x) for x in inp] for inp in DUMP[1]]  # 9 params+iedp
    rng = np.random.default_rng(59)
    for _ in range(60):
        fi1 = rng.uniform(2.0, 4.0)
        fi3 = rng.uniform(6.0, 14.0)
        fi2 = 0.0 if rng.random() < 0.3 else fi1 + rng.uniform(0.3, 0.9) * (
            fi3 - fi1)
        rows.append([fi1, fi2, fi3,
                     rng.uniform(18.0, 22.0), rng.uniform(20.0, 50.0),
                     rng.uniform(50.0, 140.0),
                     rng.uniform(105.0, 115.0), rng.uniform(150.0, 200.0),
                     rng.uniform(280.0, 480.0),
                     float(rng.choice([-1, 0]))])
    return rows


def _assert_states_bitexact(got_u8, expected, name):
    got = np.frombuffer(got_u8.tobytes(), dtype=expected.dtype)
    nc = len(expected)
    bad = {}
    for fname in expected.dtype.names:
        g = np.ascontiguousarray(got[fname]).view(np.uint32).reshape(nc, -1)
        e = np.ascontiguousarray(
            expected[fname]).view(np.uint32).reshape(nc, -1)
        for j in np.flatnonzero((g != e).any(axis=1)):
            bad.setdefault(int(j), []).append(fname)
    assert not bad, (
        f"{name}: {len(bad)}/{nc} cases differ from Stage A: "
        + "; ".join(f"case {j}: {fs}" for j, fs in sorted(bad.items())[:10]))


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_genion_dumps.sh")
def test_genion_bit_exact():
    rows = _cases()
    n = len(rows)
    inputs = np.zeros(n, dtype=LAYER_DTYPE)
    expected = np.zeros(n, dtype=LAYER_DTYPE)
    for j, row in enumerate(rows):
        s = LayerState(km=1)
        s.iedp = int(round(row[9]))
        s.fi[1, 1], s.fi[2, 1], s.fi[3, 1] = (F32(v) for v in row[0:3])
        s.yi[1, 1], s.yi[2, 1], s.yi[3, 1] = (F32(v) for v in row[3:6])
        s.hi[1, 1], s.hi[2, 1], s.hi[3, 1] = (F32(v) for v in row[6:9])
        lecden(s, 1)               # Stage A profile feeds both sides
        inputs[j] = pack_layer(s)  # snapshot BEFORE genion
        genion(s, 1)               # Stage A in-process
        expected[j] = pack_layer(s)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n, [inputs.view(np.uint8)])
    _assert_states_bitexact(fetch(dev[0]), expected, "genion")

    # branch coverage: full-Gaussian (600) vs fast (705), and within the
    # fast branch both the BENDY/PEN no-F1 arm (720) and the GETHP F1 arm
    # (730), all present in the case set
    iedp = np.array([int(round(r[9])) for r in rows])
    f1 = np.array([r[1] > 0.0 for r in rows])
    assert (iedp < 0).any(), "full-Gaussian branch unreached"
    assert (iedp >= 0).any(), "fast branch unreached"
    assert ((iedp >= 0) & ~f1).any(), "fast BENDY/PEN (no F1) arm unreached"
    assert ((iedp >= 0) & f1).any(), "fast GETHP (F1) arm unreached"
    assert ((iedp < 0) & ~f1).any() and ((iedp < 0) & f1).any()
