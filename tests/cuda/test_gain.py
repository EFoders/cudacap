"""GAIN Stage C vs Stage A — both branches behind the one entry point.

The ground branch (ITR<0) drives on the tier-B `gain` dump inputs, one
LayerState struct per thread; the antenna branch (ITR>=0) drives on the
`gainant` dump inputs against the same three-entry /cantenna/ the Stage A
test builds from sample.14, tables passed as shared flat pointers.  Expected
values come from Stage A in-process; the assert is bit-exact everywhere.

Where Stage A RAISES (the missing-bottom-clamp out-of-bounds read, a real
Fortran defect, and the unported area-coverage path) the device function
cannot, so it clamps its indices and reports through `oob_flag` instead —
tested to fire on exactly the cases Stage A raises on, and never elsewhere.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.gain import antenna_gain, ground_loss
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.readant import read_antenna
from voacap_gpu.cuda.state import LAYER_DTYPE, pack_layer

from tests.scalar.dumputil import ANTENNAS_DIR, load_dump
from tests.cuda.harness import (assert_bitexact, compile_test, fetch,
                                   launch_rows, requires_cuda)

pytestmark = requires_cuda

GROUND = load_dump("gain")
ANT = load_dump("gainant")
SAMPLE = ANTENNAS_DIR / "sample.14"

_SRC = r'''
#include "gain.cuh"

extern "C" __global__ void drive_ground(const LayerState* states,
                                        const int* ig, const float* delta,
                                        const float* fmc, float* rain, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rain[i] = ground_loss(&states[i], ig[i], delta[i], fmc[i]);
}

extern "C" __global__ void drive_ant(const float* array, const float* aeff,
                                     const int* iats, const float* xfqs,
                                     const float* xfqe, const float* offazim,
                                     int numants, const int* itr,
                                     const float* delta, const float* fmc,
                                     float* rain, float* eff, int* oob,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    antenna_gain(array, aeff, iats, xfqs, xfqe, offazim, numants,
                 itr[i], delta[i], fmc[i], &rain[i], &eff[i], &oob[i]);
}
'''


def _cantenna() -> CantennaState:
    """The three entries dump_gainant.f installs — same as the Stage A test."""
    array, aeff = read_antenna(SAMPLE).p2p_tables()
    c = CantennaState()
    for iat in (1, 2, 3):
        c.array[1:31, 1:92, iat] = array
        c.aeff[1:31, iat] = aeff
        c.offazim[iat] = np.float32(0.0)
    # entry 3 is offset so that selecting the wrong entry cannot go unnoticed
    c.array[1:31, 1:92, 3] = array + np.float32(3.0)
    c.aeff[1:31, 3] = aeff + np.float32(0.5)
    c.numants = 3
    c.iats[1], c.xfqs[1], c.xfqe[1] = 1, np.float32(2.0), np.float32(15.0)
    c.iats[2], c.xfqs[2], c.xfqe[2] = 2, np.float32(2.0), np.float32(30.0)
    c.iats[3], c.xfqs[3], c.xfqe[3] = 1, np.float32(15.0), np.float32(30.0)
    return c


def _flat_tables(c: CantennaState):
    """The shared /cantenna/ tables as (31,92,A+1)/(31,A+1)/(A+1,) flats."""
    na = c.numants + 1
    return [np.ascontiguousarray(c.array[:, :, :na]).ravel(),
            np.ascontiguousarray(c.aeff[:, :na]).ravel(),
            np.ascontiguousarray(c.iats[:na]).astype(np.int32),
            np.ascontiguousarray(c.xfqs[:na]),
            np.ascontiguousarray(c.xfqe[:na]),
            np.ascontiguousarray(c.offazim[:na]),
            np.int32(c.numants)]


def _launch_ant(mod, cant, itr, delta, fmc):
    n = len(itr)
    rain = np.zeros(n, np.float32)
    eff = np.zeros(n, np.float32)
    oob = np.full(n, -1, np.int32)
    dev = launch_rows(mod.get_function("drive_ant"), n,
                      _flat_tables(cant) + [itr, delta, fmc, rain, eff, oob])
    return fetch(dev[10]), fetch(dev[11]), fetch(dev[12])


@pytest.mark.skipif(GROUND is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_gain_dumps.sh")
def test_ground_loss_bit_exact():
    _, inputs, _ = GROUND
    n = len(inputs)
    ig = np.array([int(r[0]) for r in inputs], np.int32)
    sigma = np.array([np.float32(r[1]) for r in inputs], np.float32)
    er = np.array([np.float32(r[2]) for r in inputs], np.float32)
    delta = np.array([np.float32(r[3]) for r in inputs], np.float32)
    fmc = np.array([np.float32(r[4]) for r in inputs], np.float32)

    states = np.zeros(n, dtype=LAYER_DTYPE)
    expected = np.zeros(n, np.float32)
    for k in range(n):
        s = LayerState()
        s.sigpat[ig[k]] = sigma[k]
        s.epspat[ig[k]] = er[k]
        expected[k] = ground_loss(s, int(ig[k]), delta[k], fmc[k])
        states[k] = pack_layer(s)

    mod = compile_test(_SRC)
    out = np.zeros(n, np.float32)
    dev = launch_rows(mod.get_function("drive_ground"), n,
                      [states.view(np.uint8), ig, delta, fmc, out])
    assert_bitexact(fetch(dev[4]), expected, "ground_loss", cases=inputs)

    # branch coverage: no-ground-data preset, the grazing floor, and the
    # ordinary computed path must all be in the set
    assert (sigma <= 0).any(), "no SIGMA<=0 (no ground data) case"
    assert ((sigma > 0) & (delta <= F32(0.00000001))).any(), "no grazing case"
    assert ((sigma > 0) & (delta > F32(0.00000001))).any()


@pytest.mark.skipif(ANT is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_gainant_dumps.sh")
def test_antenna_gain_bit_exact():
    _, inputs, _ = ANT
    cant = _cantenna()
    # the dump cases, plus the Stage A test's explicit no-match probes
    cases = ([(int(r[0]), np.float32(r[1]), np.float32(r[2])) for r in inputs]
             + [(1, np.float32(0.2), np.float32(1.5)),    # below every range
                (3, np.float32(0.2), np.float32(10.0))])  # no such terminal
    itr = np.array([c[0] for c in cases], np.int32)
    delta = np.array([c[1] for c in cases], np.float32)
    fmc = np.array([c[2] for c in cases], np.float32)

    exp = [antenna_gain(cant, int(t), d, f)
           for t, d, f in zip(itr, delta, fmc)]
    exp_rain = np.array([e[0] for e in exp], np.float32)
    exp_eff = np.array([e[1] for e in exp], np.float32)

    mod = compile_test(_SRC)
    rain, eff, oob = _launch_ant(mod, cant, itr, delta, fmc)
    assert_bitexact(rain, exp_rain, "antenna_gain rain", cases=cases)
    assert_bitexact(eff, exp_eff, "antenna_gain eff", cases=cases)
    # Stage A raised nowhere on this set, so no lane may flag
    assert (oob == 0).all(), f"oob fired on {np.flatnonzero(oob).tolist()}"

    # guard against a grid that accidentally matches nothing: the miss arm
    # AND a healthy majority of real lookups must both be present
    matched = eff != np.float32(0.0)
    assert matched.sum() > len(inputs) // 2
    assert (~matched).any()


@pytest.mark.skipif(ANT is None, reason="oracle dump missing")
def test_below_table_sets_oob_flag_where_stage_a_raises():
    """GAIN clamps only the top of each axis; the bottom read is refused.

    Stage A raises IndexError on FMC < 1 or DELTA < 0; the device function
    cannot, so it must set oob_flag = 1 on exactly those lanes (and only
    those), clamping its own reads in bounds.
    """
    cant = _cantenna()
    cant.xfqs[1] = np.float32(0.0)      # let a sub-1 MHz case through
    cases = [(1, np.float32(0.2), np.float32(0.5)),    # I = 0
             (1, np.float32(-0.2), np.float32(10.0)),  # J < 1
             (1, np.float32(0.2), np.float32(10.0))]   # clean control lane
    for t, d, f in cases[:2]:
        with pytest.raises(IndexError):
            antenna_gain(cant, t, d, f)
    ctrl = antenna_gain(cant, cases[2][0], cases[2][1], cases[2][2])

    mod = compile_test(_SRC)
    itr = np.array([c[0] for c in cases], np.int32)
    delta = np.array([c[1] for c in cases], np.float32)
    fmc = np.array([c[2] for c in cases], np.float32)
    rain, eff, oob = _launch_ant(mod, cant, itr, delta, fmc)
    assert oob.tolist() == [1, 1, 0]
    assert rain[2] == np.float32(ctrl[0]) and eff[2] == np.float32(ctrl[1])


@pytest.mark.skipif(ANT is None, reason="oracle dump missing")
def test_area_coverage_sets_flag_2_where_stage_a_raises():
    """OFFAZIM == -999 selects /Careaant/, which is not ported; refuse it."""
    cant = _cantenna()
    cant.offazim[2] = np.float32(-999.0)
    with pytest.raises(NotImplementedError):
        antenna_gain(cant, 2, 0.2, 10.0)

    mod = compile_test(_SRC)
    itr = np.array([2, 1], np.int32)          # entry 2 is the receiver
    delta = np.array([0.2, 0.2], np.float32)
    fmc = np.array([10.0, 10.0], np.float32)
    rain, eff, oob = _launch_ant(mod, cant, itr, delta, fmc)
    assert oob.tolist() == [2, 0]
    # the refused lane keeps the preset 0/0
    assert rain[0] == np.float32(0.0) and eff[0] == np.float32(0.0)
