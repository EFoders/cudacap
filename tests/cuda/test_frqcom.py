"""FRQCOM Stage C vs Stage A — frequency-complement synthesis.

The golden dump drives FRQCOM + SETLUF together; SETLUF is another chunk,
so this test replays FRQCOM's share: every dump row whose IFREQ dispatches
into FRQCOM (ifreq <= 0 — the ifreq > 0 rows are HFMUFS's passthrough, not
FRQCOM's code, and are skipped).  The full 30-slot FREL row is compared so
the untouched slots' survival is part of the assert — including case 1
leaving slot 13 alone where cases 2-5 zero it.  A crafted case-3 overflow
pins the refusal arm: Stage A raises FrequencyIndexError after the slot-12
write, the device returns status 1 with the same partial state.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.frqcom import FrequencyIndexError, frqcom
from voacap_gpu.scalar.frqstate import FrqState
from voacap_gpu.scalar.mufstate import MufState
from voacap_gpu.cuda.state import MUF_DTYPE, pack_muf

from tests.scalar.dumputil import load_dump
from tests.scalar.test_frqcom import COLUMNS, NF, SEED_FREL, _which_case
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("frqcom")
NCOL = 30                       # FREL's full width rides along per case

_SRC = r'''
#include "frqcom.cuh"

extern "C" __global__ void drive(const MufState* muf, float* frel,
                                 const int* ifreq, const int* it,
                                 int* status, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    status[i] = frqcom(&muf[i], &frel[i * 30], ifreq[i], it[i]);
}
'''


def _case(v):
    """One dump row -> (MufState, seeded FREL row, ifreq, it)."""
    it = int(float(v["it"]))
    muf = MufState()
    for name in ("allmuf", "fot", "emuf", "f2muf", "hpf"):
        getattr(muf, name)[it] = F32(v[name])
    frq = FrqState()
    frq.frel[:] = SEED_FREL
    for i in range(1, NF + 1):
        frq.frel[i] = F32(v[f"frel{i}"])
    frq.frel[12] = SEED_FREL
    frq.frel[13] = F32(v["fr13"])
    return muf, frq.frel, int(float(v["ifreq"])), it


def _overflow_case():
    """XFL far above XFH drives case 3's NE past FREA's 13 elements."""
    muf = MufState()
    muf.allmuf[12] = F32(18.0)
    muf.emuf[12] = muf.f2muf[12] = F32(21.0)
    muf.hpf[12] = F32(2.5)
    frel = np.full(NCOL, SEED_FREL, np.float32)
    return muf, frel, -1, 12


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_frqcom_dumps.sh")
def test_frqcom_bit_exact():
    _, inputs, _ = DUMP
    cases = []
    seen = {"case1": 0, "case2": 0, "case3": 0, "case4": 0, "case5": 0,
            "fot_only": 0, "case1_kept_slot13": 0, "refused": 0}
    for inp in inputs:
        v = dict(zip(COLUMNS, inp))
        if int(float(v["ifreq"])) > 0:
            continue                # HFMUFS passthrough, not FRQCOM's code
        cases.append(_case(v) + (v,))
    cases.append(_overflow_case() + (None,))

    n = len(cases)
    muf_buf = np.empty(n, dtype=MUF_DTYPE)
    frel = np.empty((n, NCOL), np.float32)
    ifreq = np.empty(n, np.int32)
    it = np.empty(n, np.int32)
    exp_frel = np.empty((n, NCOL), np.float32)
    exp_status = np.zeros(n, np.int32)
    for i, (muf, row, ifq, hour, v) in enumerate(cases):
        muf_buf[i] = pack_muf(muf)
        frel[i] = row
        ifreq[i], it[i] = ifq, hour
        exp = row.copy()
        try:
            frqcom(muf, exp, ifq, hour)     # Stage A, in place
        except FrequencyIndexError:
            exp_status[i] = 1
            seen["refused"] += 1
        exp_frel[i] = exp
        if v is not None:
            if ifq == -10:
                seen["fot_only"] += 1
            else:
                arm = _which_case(float(v["emuf"]), float(v["f2muf"]),
                                  float(v["hpf"]))
                seen[arm] += 1
                if arm == "case1":
                    seen["case1_kept_slot13"] += exp[13] == F32(v["fr13"])

    mod = compile_test(_SRC)
    status = np.full(n, -1, np.int32)
    dev = launch_rows(mod.get_function("drive"), n,
                      [muf_buf.view(np.uint8), frel, ifreq, it, status])
    assert_bitexact(fetch(dev[4]), exp_status, "frqcom.status")
    assert_bitexact(fetch(dev[1]), exp_frel, "frqcom.frel")

    for arm, c in seen.items():
        assert c, f"branch never reached: {arm} ({seen})"
