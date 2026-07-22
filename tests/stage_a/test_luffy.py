"""Stage A LUFFY — the driver's own decisions.

LUFFY calls eighteen routines that are each already validated against the
oracle, so what is left to test here is the control flow between them: which
sample area gets its ionosphere built, how many hops get swept, what happens
at the end of the frequency list, and where the port refuses.

The composed behaviour is a separate exercise: it needs the whole upstream
state at LUFFY's entry, which is an end-to-end replay rather than a tier-B
dump.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.circuit import Circuit
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.luffy import (LongPathNotImplemented, _after_loop,
                                      _luf_found, _min_hops, _no_luf,
                                      _select_area)


def _hops(n1, n2, n3):
    return np.array([0, n1, n2, n3, 0], dtype=np.int32)


def test_min_hops_skips_f1_when_it_carries_no_modes():
    """NHOPMF(2) is the F1 layer, which often has no modes at all; its slot
    is then 0 and must not win the minimum.  E and F2 are taken as they are."""
    assert _min_hops(_hops(3, 2, 4)) == 2
    assert _min_hops(_hops(3, 0, 4)) == 3, "F1 with no modes is skipped"
    assert _min_hops(_hops(3, 0, 1)) == 1, "but F2 is not, even at zero"
    assert _min_hops(_hops(2, 5, 0)) == 0
    assert _min_hops(_hops(1, 1, 1)) == 1


def test_the_long_path_arms_refuse():
    for ipfg in (200, 400):
        with pytest.raises(LongPathNotImplemented):
            _select_area(Circuit(), ipfg)


def test_the_smoothing_recycle_refuses():
    """METHOD 30 between 7000 and 10000 km runs the short path, then the long
    path, then blends them.  That is the ordinary behaviour of a long-haul
    circuit, not a corner -- so it has to refuse rather than return early."""
    for km in (7000.0, 8809.0, 10000.0):
        with pytest.raises(LongPathNotImplemented):
            _after_loop(Circuit(mspec=121, gcdkm=F32(km), method=20), 100)


def test_the_bands_either_side_of_the_smoothing_range_do_not():
    for km in (6999.0, 343.0, 10001.0, 18334.0):
        _after_loop(Circuit(mspec=121, gcdkm=F32(km), method=20), 100)
    # MSPEC 0 never smooths at any distance, and METHOD 25 leaves first.
    _after_loop(Circuit(mspec=0, gcdkm=F32(8809.0), method=20), 100)
    _after_loop(Circuit(mspec=121, gcdkm=F32(8809.0), method=25), 100)


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
