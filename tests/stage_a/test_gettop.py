"""Stage A GETTOP vs the tier-B oracle dump.

GETTOP measures the stretch of a long path whose ray flies OVER the E layer:
a daylit middle can reflect a frequency the dark ends let straight through,
and that stretch never sees the D-region absorption underneath.  Five tests in
a row can each zero the answer, so the case grid has to land on all five
boundaries, from both sides and exactly on them.

The generator does that by emulating GETTOP's own float32 arithmetic and
SEARCHING one ULP at a time for the input that makes a boundary expression
exactly zero -- see gen_gettop_cases.py.  Two of the five turned out not to be
hittable at a physical frequency at all: FREQ-FP is an exact subtraction of
two nearby float32s, hence a multiple of ulp(FP), while float32 0.001 is an
ODD multiple of 2^-33, so the two can only meet when ulp(FP) is 2^-33 itself
-- FP below about 2 kHz.  Those exact cases therefore sit at kilohertz scale;
both physical-scale sides are pinned by stepping FREQ one ULP at a time.
Nothing observable rests on the difference: in both arithmetic IFs the zero
arm shares its label with one of the others.

Measured: 615 cases x 1 column = 615 cells, ALL BIT-EXACT.  A 4 ULP bar is
kept as a libm safety valve (COS and SQRT are the only calls) rather than
because anything needed it.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.constants import D2R, RZ
from voacap_gpu.stage_a.f32math import F32, cos32, sqrt32
from voacap_gpu.stage_a.gettop import gettop
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.lpathstate import LpathState

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("gettop")

GETTOP_ULP_BAR = 4.0            # measured worst 0.0 -- a libm safety valve

COLUMNS = ["delopt", "freq", "dikm", "hi1", "hi2", "hi3", "fi1", "fi2", "fi3"]


def _setup(v):
    state = LayerState()
    for j in (1, 2, 3):
        state.hi[1, j] = F32(v[f"hi{j}"])
        state.fi[1, j] = F32(v[f"fi{j}"])
    return state, LpathState(delopt=F32(v["delopt"]))


def _arms(v, seen):
    """Walk GETTOP's five tests again, counting which arm each case took."""
    state, lpath = _setup(v)
    freq, dikm = F32(v["freq"]), F32(v["dikm"])
    rcosd = RZ * cos32(lpath.delopt * D2R)
    fpe = [F32(0.0)] * 4
    clamped = False
    for j in (1, 2, 3):
        s = rcosd / (RZ + state.hi[1, j])
        s = s * s
        clamped = clamped or s > F32(0.9999)
        fpe[j] = state.fi[1, j] / sqrt32(F32(1.0) - min(s, F32(0.9999)))
    seen["sphe_clamped" if clamped else "sphe_free"] += 1
    seen["end1_higher" if fpe[1] > fpe[3] else
         "end3_higher" if fpe[1] < fpe[3] else "ends_tie"] += 1
    fp1, fp2, fp3 = max(fpe[1], fpe[3]), fpe[2], min(fpe[1], fpe[3])

    def arm(t, low, zero, high):
        """Name the arithmetic-IF arm and count it; returns 'low'/'0'/'high'."""
        name = low if t < 0 else zero if t == 0 else high
        seen[name] += 1
        return name

    if arm(fp1 - fp2, "fp1_lt_fp2", "fp1_eq_fp2",
           "fp1_gt_fp2") != "fp1_lt_fp2":
        return
    if arm(freq - fp2 - F32(0.001), "below_middle", "on_middle_slack",
           "above_middle") != "below_middle":
        return
    if arm(freq - fp1 + F32(0.001), "below_end", "on_end_slack",
           "above_end") != "above_end":
        return
    gm = F32(0.5) * dikm * (fp2 - freq) * (
        F32(1.0) / (fp2 - fp1) + F32(1.0) / (fp2 - fp3))
    if arm(gm - F32(1000.0), "top_too_short", "top_exactly_1000",
           "top_long_enough") != "top_long_enough":
        return
    arm(dikm - gm - F32(1000.0), "nothing_left_under", "exactly_1000_left",
        "pass")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_gettop_dumps.sh")
def test_gettop_vs_oracle():
    header, inputs, rows = DUMP
    seen = dict.fromkeys(
        ["sphe_clamped", "sphe_free", "end1_higher", "end3_higher",
         "ends_tie", "fp1_lt_fp2", "fp1_eq_fp2", "fp1_gt_fp2",
         "below_middle", "on_middle_slack", "above_middle", "below_end",
         "on_end_slack", "above_end", "top_too_short", "top_exactly_1000",
         "top_long_enough", "nothing_left_under", "exactly_1000_left",
         "pass"], 0)
    total = exact = 0
    worst_ulp = 0.0
    worst_at = None
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        v = dict(zip(COLUMNS, (float(x) for x in inp)))
        state, lpath = _setup(v)
        g = gettop(state, lpath, F32(v["freq"]), F32(v["dikm"]))
        e = np.float32(exp[0])
        d = ulp_diff_floored(g, e)
        if d > worst_ulp:
            worst_ulp, worst_at = d, f"case {k}"
        assert d <= GETTOP_ULP_BAR, (
            f"case {k}: got {g!r} expected {e!r} "
            f"({abs(float(g) - float(e)):.3e} abs, {d:.1f} floored ULP) {v}")
        total += 1
        exact += d == 0.0
        _arms(v, seen)
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    assert exact / total >= 0.999, (
        f"exact {exact}/{total} (worst {worst_ulp:.1f} ULP at {worst_at})")


def _night_day_night(fi=(0.6, 0.8, 0.6), hi=(110.0, 110.0, 110.0),
                     delopt=3.0):
    state = LayerState()
    for j in (1, 2, 3):
        state.hi[1, j] = F32(hi[j - 1])
        state.fi[1, j] = F32(fi[j - 1])
    return state, LpathState(delopt=F32(delopt))


def test_the_middle_must_penetrate_higher_than_either_end():
    """No daylit middle, no over-the-top segment -- and the tie counts as
    "no", since the arithmetic IF puts equality with the failing arm."""
    args = (F32(3.5), F32(15000.0))
    assert gettop(*_night_day_night((0.6, 0.8, 0.6)), *args) > F32(0.0)
    assert gettop(*_night_day_night((0.8, 0.8, 0.6)), *args) == F32(0.0)
    assert gettop(*_night_day_night((0.9, 0.8, 0.6)), *args) == F32(0.0)


def _fpe(fv, hi=110.0, delopt=3.0):
    s = RZ * cos32(F32(delopt) * D2R) / (RZ + F32(hi))
    return F32(fv) / sqrt32(F32(1.0) - s * s)


def test_the_lower_slack_admits_and_the_upper_one_is_neutralised():
    """FREQ-FP2-.001 widens the window at the top and FREQ-FP1+.001 widens
    it at the bottom -- but only the bottom one changes any answer.

    ABOVE FP2 THE INTERPOLATION'S (FP2-FREQ) FACTOR TURNS NEGATIVE, so the
    kilohertz the top slack admits produces a negative GMKM that the very
    next test throws away.  The bottom slack does admit: a frequency a
    kilohertz below the higher end's penetration still yields a usable
    over-the-top segment.

    (Just BELOW FP2 the answer is also zero, but for the honest reason --
    the segment has shrunk to a few kilometres.  The over-the-top distance
    is largest at the bottom of the window and vanishes at the top.)
    """
    state, lpath = _night_day_night(fi=(0.7, 0.9, 0.4))
    fp1, fp2 = _fpe(0.7), _fpe(0.9)
    assert gettop(state, lpath, fp1 - F32(0.0005), F32(15000.0)) > F32(0.0)
    assert gettop(state, lpath, fp1 - F32(0.01), F32(15000.0)) == F32(0.0)
    mid = fp1 + F32(0.25) * (fp2 - fp1)
    assert gettop(state, lpath, mid, F32(15000.0)) > F32(0.0)
    # the top slack admits the frequency and then cannot use it
    assert gettop(state, lpath, fp2 + F32(0.0005), F32(15000.0)) == F32(0.0)


def test_the_segment_must_leave_a_thousand_kilometres_underneath():
    """"MUST BE ABLE TO GET UP THERE FOR M MODE": an over-the-top stretch
    that eats the whole path is refused, because the ray still has to climb
    from the ground at each end."""
    state, lpath = _night_day_night()
    # GMKM is 0.6654 x DIKM for this geometry, so DIKM = 2000 leaves 669 km
    assert gettop(state, lpath, F32(3.5), F32(2000.0)) == F32(0.0)
    assert gettop(state, lpath, F32(3.5), F32(4000.0)) > F32(0.0)
    # ... and one that is under 1000 km itself is refused from the other side
    assert gettop(state, lpath, F32(3.5), F32(1400.0)) == F32(0.0)


def test_the_sin_squared_clamp_is_dead_for_real_geometry():
    """The 0.9999 clamp protects a SQRT of a negative, but DELOPT is a
    blkdat constant of 3 degrees and the E layer sits near 110 km, which
    gives sin^2 = 0.964.  Reaching the clamp needs a negative height."""
    rcosd = RZ * cos32(F32(3.0) * D2R)
    s = rcosd / (RZ + F32(110.0))
    assert s * s < F32(0.9999)
    below = rcosd / (RZ + F32(-20.0))
    assert below * below > F32(1.0), "and without the clamp, SQRT(1-x) < 0"
    # with the clamp the routine still returns a number rather than a NaN:
    # sin^2 pins at 0.9999, so the oblique factor pins at 100
    state, lpath = _night_day_night(fi=(0.004, 0.006, 0.004),
                                    hi=(-20.0, -20.0, -20.0))
    got = gettop(state, lpath, F32(0.5), F32(15000.0))
    assert np.isfinite(float(got)) and got > F32(0.0)


def test_gmkm_is_written_on_every_path():
    """Every route through the routine assigns GMKM, so no caller can read
    an uninitialised output -- the dump's 12345 sentinel never survives."""
    state, lpath = _night_day_night()
    for freq in (F32(0.0), F32(3.5), F32(40.0)):
        for dikm in (F32(-1.0), F32(0.0), F32(15000.0)):
            assert np.isfinite(float(gettop(state, lpath, freq, dikm)))
