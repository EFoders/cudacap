"""SETLNG Stage C vs Stage A — sample-area replication.

Pure array replication; the whole content is the KFX branch and the
deliberately asymmetric index ranges (scalars to areas 2..5 or 4..5, the
ionogram only to 2..3 or 3).  Cases reuse the golden dump's (KFX, salt)
grid and the Stage A test's sentinel-injection formula, so every touched
and every deliberately UNtouched cell carries a distinct exact-integer
value; the assert is bit-identity of the complete LayerState and RaysState
structs — 12 kB of witnesses per case — against Stage A run in-process.
A synthetic KFX 0..5 grid runs even without the dump.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.raysstate import RaysState
from voacap_gpu.stage_a.setlng import setlng
from voacap_gpu.stage_c.state import (LAYER_DTYPE, RAYS_DTYPE, pack_layer,
                                      pack_rays)

from tests.stage_a.dumputil import load_dump
from tests.stage_a.test_setlng import _inject
from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("setlng")

_SRC = r'''
#include "setlng.cuh"

extern "C" __global__ void drive(LayerState* s, RaysState* r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    setlng(&s[i], &r[i]);
}
'''


def _run_cases(cases):
    """cases: (kfx, salt) pairs.  Device vs Stage A, full-struct compare."""
    n = len(cases)
    s_buf = np.empty(n, dtype=LAYER_DTYPE)
    r_buf = np.empty(n, dtype=RAYS_DTYPE)
    exp_s = np.empty(n, dtype=LAYER_DTYPE)
    exp_r = np.empty(n, dtype=RAYS_DTYPE)
    for i, (kfx, salt) in enumerate(cases):
        s = LayerState(kfx=int(kfx))
        r = RaysState()
        _inject(s, r, int(salt))
        s_buf[i], r_buf[i] = pack_layer(s), pack_rays(r)
        setlng(s, r)                        # Stage A, in place
        exp_s[i], exp_r[i] = pack_layer(s), pack_rays(r)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [s_buf.view(np.uint8), r_buf.view(np.uint8)])
    got_s = fetch(dev[0]).reshape(n, LAYER_DTYPE.itemsize
                                  ).view(LAYER_DTYPE).reshape(n)
    got_r = fetch(dev[1]).reshape(n, RAYS_DTYPE.itemsize
                                  ).view(RAYS_DTYPE).reshape(n)
    for name in LAYER_DTYPE.names:
        assert_bitexact(got_s[name], exp_s[name], f"setlng.layer.{name}")
    for name in RAYS_DTYPE.names:
        assert_bitexact(got_r[name], exp_r[name], f"setlng.rays.{name}")

    # all three KFX arms must be present in the case set
    kfxs = [int(k) for k, _ in cases]
    assert any(k <= 1 for k in kfxs), "one-area arm never reached"
    assert any(k == 2 for k in kfxs), "three-area arm never reached"
    assert any(k >= 3 for k in kfxs), "no-op arm never reached"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_setlng_dumps.sh")
def test_setlng_bit_exact_on_the_dump_grid():
    _, inputs, _ = DUMP
    _run_cases([(int(inp[0]), int(inp[1])) for inp in inputs])


def test_setlng_bit_exact_on_a_synthetic_grid():
    """KFX 0..5 with three salts each — runs even without the dump."""
    _run_cases([(kfx, salt) for kfx in range(6) for salt in (1, 4, 7)])
