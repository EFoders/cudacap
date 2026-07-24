"""ALLMODES Stage C vs Stage A — the /allMODE/ accumulator.

The golden dump is a call SEQUENCE (NMMOD is both count and cursor and
accumulates across calls), so cases are built by threading one Stage A
AllModeState through every record and snapshotting the state BEFORE each
call: each device thread then replays exactly one call from its own
pre-state, and the assert is bit-identity of both mutated structs (/ZON/
included — the IFLG=1 arm writes HP(2)=HP(1) back), the returned FREQ, and
the overflow status.  Two crafted overflow cases pin the refusal arms:
Stage A raises AllModesOverflow, the device returns status 1 with the state
exactly as Stage A leaves it at the raise.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.allmodes import AllModesOverflow, allmodes
from voacap_gpu.scalar.allmodestate import AllModeState
from voacap_gpu.scalar.zonstate import ZonState
from voacap_gpu.cuda.state import (ALLMODE_DTYPE, ZON_DTYPE, pack_allmode,
                                      pack_zon)

from tests.scalar.dumputil import load_dump
from tests.scalar.test_allmodes import _zon
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

DUMP = load_dump("allmodes")
FREQ_SENTINEL = np.float32(-3333.0)

_SRC = r'''
#include "allmodes.cuh"

extern "C" __global__ void drive(ZonState* zon, AllModeState* allm,
                                 const int* iflg, const float* fval,
                                 float* freq, int* status, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    status[i] = allmodes(&zon[i], &allm[i], iflg[i], fval[i], &freq[i]);
}
'''


def _overflow_cases():
    """The two refusal arms; the comparison is AT the Stage A raise."""
    cases = []
    # copy-loop overflow: 18 accumulated + 6 live modes walks past slot 20
    allm = AllModeState()
    allm.nmmod = 18
    zon = ZonState()
    zon.hp[1:8] = np.float32(300.0)
    cases.append((zon, allm, 3, 0.0))
    # IFLG=1 marker overflow: slot NMMOD+2 past the end, before the copy —
    # but after the HP(2)=HP(1) write-back and the 32000 sentinel
    allm = AllModeState()
    allm.nmmod = 19
    zon = ZonState()
    zon.hp[1] = np.float32(250.0)
    cases.append((zon, allm, 1, 0.0))
    return cases


def _build_cases():
    """(zon_pre, allm_pre, iflg, fval) and Stage A's post-state per call."""
    _, inputs, _ = DUMP
    allm = AllModeState()          # threaded through the whole sequence
    cases, expected = [], []
    for inp in inputs:
        iflg = int(float(inp[0]))
        fval = float(inp[1])
        zon = _zon(inp)
        pre = (pack_zon(zon), pack_allmode(allm), iflg, fval)
        got = allmodes(zon, allm, iflg, fval)
        freq = np.float32(got) if iflg == 0 else FREQ_SENTINEL
        cases.append(pre)
        expected.append((pack_zon(zon), pack_allmode(allm), freq, 0))
    for zon, allm, iflg, fval in _overflow_cases():
        pre = (pack_zon(zon), pack_allmode(allm), iflg, fval)
        with pytest.raises(AllModesOverflow):
            allmodes(zon, allm, iflg, fval)
        cases.append(pre)
        expected.append((pack_zon(zon), pack_allmode(allm), FREQ_SENTINEL, 1))
    return cases, expected


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_allmodes_dumps.sh")
def test_allmodes_bit_exact():
    cases, expected = _build_cases()
    n = len(cases)
    zon_buf = np.empty(n, dtype=ZON_DTYPE)
    allm_buf = np.empty(n, dtype=ALLMODE_DTYPE)
    iflg = np.empty(n, np.int32)
    fval = np.empty(n, np.float32)
    for i, (z, a, fl, fv) in enumerate(cases):
        zon_buf[i], allm_buf[i] = z, a
        iflg[i], fval[i] = fl, np.float32(fv)
    freq = np.full(n, FREQ_SENTINEL, np.float32)
    status = np.full(n, -1, np.int32)

    exp_zon = np.empty(n, dtype=ZON_DTYPE)
    exp_allm = np.empty(n, dtype=ALLMODE_DTYPE)
    exp_freq = np.empty(n, np.float32)
    exp_status = np.empty(n, np.int32)
    for i, (z, a, fq, st) in enumerate(expected):
        exp_zon[i], exp_allm[i] = z, a
        exp_freq[i], exp_status[i] = fq, st

    mod = compile_test(_SRC)
    dev = launch_rows(mod.get_function("drive"), n,
                      [zon_buf.view(np.uint8), allm_buf.view(np.uint8),
                       iflg, fval, freq, status])
    got_zon = fetch(dev[0]).reshape(n, ZON_DTYPE.itemsize
                                    ).view(ZON_DTYPE).reshape(n)
    got_allm = fetch(dev[1]).reshape(n, ALLMODE_DTYPE.itemsize
                                     ).view(ALLMODE_DTYPE).reshape(n)

    assert_bitexact(fetch(dev[5]), exp_status, "allmodes.status")
    assert_bitexact(fetch(dev[4]), exp_freq, "allmodes.freq")
    for name in ZON_DTYPE.names:
        assert_bitexact(got_zon[name], exp_zon[name], f"allmodes.zon.{name}")
    for name in ALLMODE_DTYPE.names:
        assert_bitexact(got_allm[name], exp_allm[name],
                        f"allmodes.allm.{name}")

    # branch coverage, mirroring the Stage A test: every IFLG arm ran, the
    # accumulation genuinely walked deep into the table, and both refusal
    # arms fired
    seen = {0: 0, 1: 0, 2: 0, 3: 0}
    for fl in iflg:
        seen[int(fl) if int(fl) in (0, 1, 2) else 3] += 1
    assert all(seen[i] for i in (0, 1, 2, 3)), f"unreached IFLG arm: {seen}"
    assert int(exp_allm["nmmod"].max()) >= 18, "accumulation stayed shallow"
    assert int((exp_status == 1).sum()) == 2, "both overflow arms must fire"
