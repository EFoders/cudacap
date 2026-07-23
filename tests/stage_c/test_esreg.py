"""ESREG Stage C vs Stage A — the preset that is the whole routine.

ESREG's mixed Es-F physics is dead code behind an unconditional RETURN; the
reachable behaviour is presetting /ZON/ slots 6 and 7 to the no-mode
markers.  Cases seed every /ZON/ cell with a distinct random value so the
full-struct bit-compare pins both the writes (slots 6-7, CREL = +1000 where
ESMOD's slots get -1000) and the non-writes (slots 1-5 and 0 survive
untouched).  Expected values come from Stage A in-process.
"""

import numpy as np

from voacap_gpu.stage_a.esreg import esreg
from voacap_gpu.stage_a.zonstate import ZonState
from voacap_gpu.stage_c.state import ZON_DTYPE, pack_zon

from tests.stage_c.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

_SRC = r'''
#include "esreg.cuh"

extern "C" __global__ void drive(ZonState* zon, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    esreg(&zon[i]);
}
'''


def test_esreg_bit_exact():
    rng = np.random.default_rng(11)
    n = 256
    zon_buf = np.empty(n, dtype=ZON_DTYPE)
    exp_buf = np.empty(n, dtype=ZON_DTYPE)
    for i in range(n):
        zon = ZonState()
        for name in ZON_DTYPE.names:
            arr = getattr(zon, name)
            if name == "nmode":
                arr[:] = rng.integers(-9, 9, arr.shape).astype(np.int32)
            else:
                arr[:] = rng.uniform(-2000.0, 2000.0,
                                     arr.shape).astype(np.float32)
        zon_buf[i] = pack_zon(zon)
        esreg(zon)                          # Stage A, in place
        exp_buf[i] = pack_zon(zon)

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n, [zon_buf.view(np.uint8)])
    got = fetch(dev[0]).reshape(n, ZON_DTYPE.itemsize
                                ).view(ZON_DTYPE).reshape(n)
    for name in ZON_DTYPE.names:
        assert_bitexact(got[name], exp_buf[name], f"esreg.zon.{name}")

    # the shape of the answer, pinned explicitly: slots 1..5 survive, 6..7
    # carry the markers, and CREL is +1000 (ESMOD's Es slots get -1000)
    assert (got["crel"][:, 6:8] == np.float32(1000.0)).all()
    assert (got["hp"][:, 6:8] == np.float32(-1.0)).all()
    assert (got["nmode"][:, 6:8] == 5).all()
    assert (got["abps"][:, :6] == zon_buf["abps"][:, :6]).all(), "slot 1..5"
