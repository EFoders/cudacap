"""Stage A BABS + TABS + CONVH vs the tier-B oracle dump.

The three pure leaves LNGPAT calls to price its geometry.  None of them
touches state another writes, so the dump drives all three from one grid --
and that grid shares DEL and HP across all three exactly as LNGPAT does,
which is what makes the 70 km disagreement visible from a single row.

That disagreement is the finding worth pinning: ``IF(HP-70.) 105,115,115`` in
BABS sends a virtual height of exactly 70 km to the LIVE arm, while
``IF(HP-70.) 110,110,115`` in TABS sends the same number to the dead one.  So
one row of this dump can carry a real absorbing-leg loss and a 1000 dB
over-the-top loss from the same HP.

Measured: 469 cases x 4 columns = 1,876 cells, 99.41% bit-exact, worst 4.0
floored ULP (9.5e-7 absolute).  AF and PTOT are bit-exact throughout.  The
residual is all in CH: ``10*ALOG10(SMALLC)`` with SMALLC near 1 amplifies a
1-ULP cosf/sinf disagreement by more than a hundred times, because
d(log10)/dx is 0.43 there while the log10 itself is near zero.  That is the
[[f32-libm-noise]] floor with the routine's own conditioning on top, not a
transliteration error -- PTOT, which shares COS(DEL) with nothing, matches
bit for bit on every case.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.babs import babs
from voacap_gpu.scalar.constants import PIO2, RZ
from voacap_gpu.scalar.convh import convh
from voacap_gpu.scalar.f32math import F32, cos32, sin32
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.signalstate import SigdState
from voacap_gpu.scalar.tabs import tabs

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("lngabs")

LNGABS_ULP_BAR = 16.0           # measured worst 4.0, all of it in CH
LNGABS_ABS_BAR = 1.0e-5         # measured worst 9.5e-7
# Per-column bit-exact floors.  Measured: af 1.0000, am 0.9979, ch 0.9787,
# ptot 1.0000.  CH gets its own floor because the ALOG10 amplification above
# is real and unfixable at this bar; the other three are effectively exact.
EXACT_FLOOR = {"af": 0.99, "am": 0.99, "ch": 0.95, "ptot": 0.99}

COLUMNS = ["freq", "gyz3", "acav", "del", "hp", "df", "dm", "gd", "phe",
           "ray"]


def _run(inp):
    v = dict(zip(COLUMNS, (float(x) for x in inp)))
    state = LayerState()
    state.gyz[3] = F32(v["gyz3"])
    sigd = SigdState(acav=F32(v["acav"]))
    got = {}
    got["af"] = babs(state, sigd, F32(v["freq"]), F32(v["del"]),
                     F32(v["hp"]), F32(v["df"]))
    got["am"] = tabs(state, F32(v["freq"]), F32(v["del"]), F32(v["hp"]),
                     F32(v["dm"]))
    got["ch"], got["ptot"] = convh(F32(v["gd"]), F32(v["phe"]),
                                   F32(v["del"]), F32(v["hp"]), F32(v["ray"]))
    return got, v


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_lngabs_dumps.sh")
def test_babs_tabs_convh_vs_oracle():
    header, inputs, rows = DUMP
    worst_ulp = 0.0
    worst_at = None
    colexact = {name: 0 for name in header}
    coltotal = {name: 0 for name in header}
    seen = {"babs_leg_dead": 0, "babs_leg_zero": 0, "babs_hp_dead": 0,
            "babs_live": 0, "babs_hp_is_70": 0, "babs_acav_zero": 0,
            "tabs_leg_dead": 0, "tabs_leg_zero": 0, "tabs_hp_dead": 0,
            "tabs_live": 0, "tabs_hp_is_70": 0,
            "convh_sin_floored": 0, "convh_sin_free": 0,
            "convh_gm_floored": 0, "convh_gm_free": 0,
            "convh_ch_clamped": 0, "convh_ch_free": 0}
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got, v = _run(inp)
        df, dm, hp = F32(v["df"]), F32(v["dm"]), F32(v["hp"])
        seen["babs_leg_dead"] += df <= 0.0
        seen["babs_leg_zero"] += df == 0.0
        seen["babs_hp_dead"] += df > 0.0 and hp < F32(70.0)
        seen["babs_live"] += df > 0.0 and hp >= F32(70.0)
        seen["babs_hp_is_70"] += df > 0.0 and hp == F32(70.0)
        seen["babs_acav_zero"] += df > 0.0 and F32(v["acav"]) == 0.0
        seen["tabs_leg_dead"] += dm <= 0.0
        seen["tabs_leg_zero"] += dm == 0.0
        seen["tabs_hp_dead"] += dm > 0.0 and hp <= F32(70.0)
        seen["tabs_live"] += dm > 0.0 and hp > F32(70.0)
        seen["tabs_hp_is_70"] += dm > 0.0 and hp == F32(70.0)
        gd, phe, del_ = F32(v["gd"]), F32(v["phe"]), F32(v["del"])
        floored = abs(sin32(gd)) < F32(0.000001)
        seen["convh_sin_floored"] += floored
        seen["convh_sin_free"] += not floored
        gm_raw = gd - F32(2.0) * (PIO2 - del_ - phe)
        seen["convh_gm_floored"] += gm_raw < F32(0.001)
        seen["convh_gm_free"] += gm_raw >= F32(0.001)
        seen["convh_ch_clamped"] += got["ch"] == F32(15.0)
        seen["convh_ch_free"] += got["ch"] < F32(15.0)
        for name, e_str in zip(header, exp):
            g = np.float32(got[name])
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            a = abs(float(g) - float(e))
            if d > worst_ulp:
                worst_ulp, worst_at = d, f"case {k} {name}"
            assert d <= LNGABS_ULP_BAR and a <= LNGABS_ABS_BAR, (
                f"case {k} {name}: got {g!r} expected {e!r} "
                f"({a:.3e} abs, {d:.1f} floored ULP)")
            coltotal[name] += 1
            colexact[name] += d == 0.0
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    for name in header:
        rate = colexact[name] / coltotal[name]
        assert rate >= EXACT_FLOOR[name], (
            f"{name}: exact {colexact[name]}/{coltotal[name]} = {rate:.4f} "
            f"(worst overall {worst_ulp:.1f} ULP at {worst_at})")


def _leaf_state(gyz3=1.358, acav=0.4):
    state = LayerState()
    state.gyz[3] = F32(gyz3)
    return state, SigdState(acav=F32(acav))


def test_babs_and_tabs_disagree_at_exactly_70_km():
    """The twins' arithmetic IFs put HP == 70 on opposite sides."""
    state, sigd = _leaf_state()
    af = babs(state, sigd, F32(14.0), F32(0.15), F32(70.0), F32(1.2))
    am = tabs(state, F32(14.0), F32(0.15), F32(70.0), F32(1.2))
    assert af < F32(1.0), "BABS keeps 70 km"
    assert am == F32(1000.0), "TABS kills 70 km"
    # one float32 step below, both are dead; one step above, both are live
    lo = np.nextafter(np.float32(70.0), np.float32(0.0))
    hi = np.nextafter(np.float32(70.0), np.float32(100.0))
    assert babs(state, sigd, F32(14.0), F32(0.15), lo, F32(1.2)) == F32(1000.0)
    assert tabs(state, F32(14.0), F32(0.15), lo, F32(1.2)) == F32(1000.0)
    assert babs(state, sigd, F32(14.0), F32(0.15), hi, F32(1.2)) < F32(1.0)
    assert tabs(state, F32(14.0), F32(0.15), hi, F32(1.2)) < F32(1.0)


def test_a_zero_length_leg_is_dead_not_lossless():
    """Both arithmetic IFs put the zero case with the dead branch."""
    state, sigd = _leaf_state()
    assert babs(state, sigd, F32(14.0), F32(0.15), F32(300.0),
                F32(0.0)) == F32(1000.0)
    assert tabs(state, F32(14.0), F32(0.15), F32(300.0),
                F32(0.0)) == F32(1000.0)


def test_tabs_ignores_acav_entirely():
    """TABS's index is the hardcoded 0.1, so a zero ACAV changes nothing --
    where BABS returns exactly zero loss for the same input."""
    hot, sigd_hot = _leaf_state(acav=3.0)
    cold, sigd_cold = _leaf_state(acav=0.0)
    args = (F32(14.0), F32(0.15), F32(300.0), F32(1.2))
    assert tabs(hot, *args) == tabs(cold, *args)
    assert babs(cold, sigd_cold, *args) == F32(0.0)
    assert babs(hot, sigd_hot, *args) > F32(0.0)


def test_the_no_loss_marker_is_per_kilometre():
    """1000 is a loss PER KM, and LNGPAT multiplies it back by the leg
    length -- so a dead leg costs 1000 dB for every kilometre of itself,
    not a flat 1000 dB."""
    state, sigd = _leaf_state()
    af = babs(state, sigd, F32(14.0), F32(0.15), F32(50.0), F32(0.5))
    assert af == F32(1000.0)
    assert af * F32(0.5) * RZ > F32(3.0e6)


def test_convh_ceiling_hides_the_sin_floor():
    """DENOM's 1e-6 floor never decides anything on a physical path: any GD
    close enough to the antipode to reach it has already sent CH past the
    15 dB ceiling.  Both clamps are transliterated; only one is observable.

    For this geometry CH does not come off the ceiling until GD reaches about
    0.0086 rad -- where |sin(GD)| is over eight thousand times the floor.
    """
    args = (F32(0.5), F32(0.3), F32(300.0), F32(900.0))
    on_floor = convh(F32(0.0), *args)            # DENOM = 1e-6 exactly
    just_free = convh(F32(2.0e-6), *args)        # DENOM = |sin(GD)|, barely
    assert on_floor[0] == F32(15.0) and just_free[0] == F32(15.0)
    assert convh(F32(0.01), *args)[0] < F32(15.0)
    # and the group path is unaffected by either clamp
    assert on_floor[1] == just_free[1] == F32(2.0) * F32(900.0) + (
        RZ + F32(300.0)) * F32(0.001)


def test_convh_gm_floor_stops_a_negative_arc():
    """GM is what is left of the great-circle arc after the two end legs;
    on a short GD that is negative, and the floor turns it into 0.001 --
    leaving PTOT as the two straight legs plus a token 6.7 km of arc.

    CH then comes out NEGATIVE: an unfocused short path diverges, and the
    routine's only floor on that is the ALOG10 argument staying positive.
    """
    ch, ptot = convh(F32(0.1), F32(0.0), F32(0.0), F32(300.0), F32(0.0))
    assert ptot == (RZ + F32(300.0)) * F32(0.001)
    assert ch < F32(-19.0)


def test_convh_psi_is_associated_differently_from_babs():
    """CONVH writes PIO2-DEL-PHE where BABS and TABS write PIO2-PHE-DEL.
    Identical in exact arithmetic; a distinct float32 number in places."""
    del_, phe = F32(1.4999999), F32(0.07000001)
    assert (PIO2 - del_) - phe != (PIO2 - phe) - del_
