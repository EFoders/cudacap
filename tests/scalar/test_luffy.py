"""Stage A LUFFY — the driver's own decisions.

LUFFY calls eighteen routines that are each already validated against the
oracle, so what is left to test here is the control flow between them: which
sample area gets its ionosphere built, how many hops get swept, what happens
at the end of the frequency list, and where the port refuses.

The composed behaviour is a separate exercise: it needs the whole upstream
state at LUFFY's entry, which is an end-to-end replay rather than a tier-B
dump.  That lives in test_luffy_replay.py.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.circuit import Circuit
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.f32math import log10_32, pow32
from voacap_gpu.scalar.lsmooth import SmoothStore
from voacap_gpu.scalar.luffy import (_after_loop, _luf_found, _min_hops,
                                      _no_luf, _select_area)


def _hops(n1, n2, n3):
    return np.array([0, n1, n2, n3, 0], dtype=np.int32)


def _plausible():
    """A Circuit whose layers are merely non-degenerate.

    These tests are about which area gets built, not about what comes out of
    it -- but a Circuit straight from the constructor has zero semithicknesses,
    and LECDEN and ALOSFV both divide by those.  The resulting profile is
    meaningless and unused; it just keeps the arithmetic finite so a real
    warning from somewhere else would still stand out."""
    c = Circuit()
    c.layers.km = 5
    for k in range(1, 6):
        for ly in (1, 2, 3):
            c.layers.fi[ly, k] = F32(3.0 + ly)
            c.layers.yi[ly, k] = F32(20.0 + 10.0 * ly)
            c.layers.hi[ly, k] = F32(100.0 + 50.0 * ly)
    return c


def test_min_hops_skips_f1_when_it_carries_no_modes():
    """NHOPMF(2) is the F1 layer, which often has no modes at all; its slot
    is then 0 and must not win the minimum.  E and F2 are taken as they are."""
    assert _min_hops(_hops(3, 2, 4)) == 2
    assert _min_hops(_hops(3, 0, 4)) == 3, "F1 with no modes is skipped"
    assert _min_hops(_hops(3, 0, 1)) == 1, "but F2 is not, even at zero"
    assert _min_hops(_hops(2, 5, 0)) == 0
    assert _min_hops(_hops(1, 1, 1)) == 1


def test_the_long_path_builds_an_ionosphere_at_BOTH_ends():
    """The short path has one controlling area; the long path has two.

    Over 10,000 km the ends are in different places and different local
    times, so each gets its own ionosphere -- and ITXRCP records which area
    belongs to which end, where the short path leaves it alone."""
    for ipfg in (200, 400):
        c = _plausible()
        c.layers.kfx = 3    # IONSET only ever sets 1, 2 or 3
        c.frq.jmode = 2
        k, xmdl = _select_area(c, ipfg)
        assert c.jlong == 1, "the long-path flag"
        assert c.frq.itxrcp[1] == 1, "transmitter end is area 1"
        assert c.frq.itxrcp[2] == 3 and k == 3, "receiver end is KFX"
        assert xmdl == "L"

    c = _plausible()
    c.layers.kfx = 3
    c.frq.jmode = 2
    k, xmdl = _select_area(c, 100)
    assert c.jlong == -1 and k == 2, "the short path stays on SELMOD's area"
    assert xmdl == "S"
    assert c.frq.itxrcp[1] == 0 and c.frq.itxrcp[2] == 0, "and sets no ends"


def test_the_smoothing_band_recycles_into_the_second_model():
    """METHOD 30 between 7000 and 10000 km runs BOTH models and blends them.

    _after_loop returns True to mean "go back to label 83 and run the other
    one", which is the recycle.  It says that only on the FIRST pass: coming
    back with IPFLAG already 200 means both passes are done, so it blends
    instead."""
    for km in (7000.0, 8809.0, 10000.0):
        c = Circuit(mspec=121, gcdkm=F32(km), method=20)
        assert _after_loop(c, 100, SmoothStore()) is True, f"{km} km recycles"


def test_the_bands_either_side_of_the_smoothing_range_do_not_recycle():
    for km in (6999.0, 343.0, 10001.0, 18334.0):
        c = Circuit(mspec=121, gcdkm=F32(km), method=20)
        assert _after_loop(c, 100, SmoothStore()) is False
    # MSPEC 0 never smooths at any distance, and METHOD 25 leaves first.
    assert _after_loop(Circuit(mspec=0, gcdkm=F32(8809.0), method=20), 100,
                       SmoothStore()) is False
    assert _after_loop(Circuit(mspec=121, gcdkm=F32(8809.0), method=25), 100,
                       SmoothStore()) is False


def test_the_smoothing_blend_joins_the_two_models_continuously():
    """SMOOTH = SSPLD + 10*log10(DISINT*(10^(DELPOW/10) - 1) + 1).

    At DISINT 0 (7000 km) that collapses to the short path's own level and at
    DISINT 1 (10000 km) to the long path's, so the interpolation meets each
    pure model exactly where that model takes over."""
    def smooth(gcdkm, sspld, slpld):
        disint = F32((F32(gcdkm) - F32(7000.0)) / F32(3000.0))
        delpow = F32(slpld) - F32(sspld)
        delx = pow32(F32(10.0), delpow / F32(10.0))
        return F32(sspld) + F32(10.0) * log10_32(
            disint * (delx - F32(1.0)) + F32(1.0))

    assert float(smooth(7000.0, -120.0, -110.0)) == pytest.approx(-120.0,
                                                                 abs=1e-3)
    assert float(smooth(10000.0, -120.0, -110.0)) == pytest.approx(-110.0,
                                                                  abs=1e-3)
    mid = float(smooth(8500.0, -120.0, -110.0))
    assert -120.0 < mid < -110.0, "and lands between them in the band"
    # The blend is on POWER, so the midpoint sits well above the dB midpoint.
    assert mid > -115.0


def test_the_luf_interpolates_between_the_bracketing_frequencies():
    c = Circuit()
    c.time.it = 7
    c.frq.frea[3] = F32(7.0)
    c.frq.frea[4] = F32(9.0)
    c.son.reliab[3] = F32(0.80)
    c.son.reliab[4] = F32(0.92)
    _luf_found(c, 4, F32(0.90))
    assert c.muf.xluf[7] == pytest.approx(8.6666667, abs=1e-5)
    assert c.frq.frea[13] == c.muf.xluf[7]


def test_a_first_frequency_that_works_is_the_luf_outright():
    c = Circuit()
    c.time.it = 7
    c.frq.freq = F32(3.2)
    c.frq.frea[1] = F32(3.2)
    _luf_found(c, 1, F32(0.90))
    assert c.muf.xluf[7] == F32(3.2) and c.frq.frea[13] == F32(3.2)


def test_no_luf_takes_the_LAST_frequency_to_beat_slot_one_not_the_best():
    """REL is seeded from RELIAB(1) and never reassigned inside the loop, so
    the scan keeps overwriting IG.  The source flags this itself, asking
    whether the test should have been RELIAB(IF).gt.RELIAB(IG)."""
    c = Circuit()
    c.time.it = 7
    for i, (f, r) in enumerate(
            [(3.2, 0.10), (5.1, 0.80), (7.3, 0.55), (9.6, 0.30)], start=1):
        c.frq.frea[i] = F32(f)
        c.son.reliab[i] = F32(r)
    _no_luf(c)
    assert c.muf.xluf[7] == F32(-9.6), "9.6 MHz is last past slot 1, not best"
    assert c.frq.frea[13] == F32(9.6)
    # With a real running maximum it would have picked 5.1 MHz.
    assert c.son.reliab[2] > c.son.reliab[4]


def test_no_luf_with_nothing_beating_slot_one_keeps_slot_one():
    c = Circuit()
    c.time.it = 7
    for i, (f, r) in enumerate(
            [(3.2, 0.80), (5.1, 0.10), (7.3, 0.55)], start=1):
        c.frq.frea[i] = F32(f)
        c.son.reliab[i] = F32(r)
    _no_luf(c)
    assert c.muf.xluf[7] == F32(-3.2)
