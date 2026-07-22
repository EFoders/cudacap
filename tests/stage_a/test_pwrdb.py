"""PWRDB -- transmitter power lookup.

PWRDB has no floating-point arithmetic, only comparisons and a table read, so
there is nothing for a tier-B dump to measure that these assertions do not
already pin.  Its *integration* is oracle-backed anyway: the REGMOD dump feeds
FLDST and SIGPOW through it on all 262 cases, covering both a transmitter in
range and one out of range (which falls back to 30 dBW).

What is left, and what these tests are for, is the 2-30 MHz clamp -- which
applies to the *range test only*, never to the returned power.  Getting that
backwards would silently change which antenna entry matches at band edges.
"""

import numpy as np

from voacap_gpu.stage_a.antennastate import CantennaState
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.pwrdb import pwrdb


def _cant(*entries) -> CantennaState:
    """entries: (iats, xfqs, xfqe, pwrdba) tuples, installed from index 1."""
    c = CantennaState()
    for i, (iats, lo, hi, p) in enumerate(entries, start=1):
        c.iats[i] = iats
        c.xfqs[i] = F32(lo)
        c.xfqe[i] = F32(hi)
        c.pwrdba[i] = F32(p)
    c.numants = len(entries)
    return c


def test_default_is_30_dbw_when_nothing_matches():
    assert pwrdb(CantennaState(), 14.0) == np.float32(30.0)
    assert pwrdb(_cant((1, 2.0, 10.0, 47.0)), 20.0) == np.float32(30.0)


def test_receiver_entries_are_skipped():
    """Only IATS == 1 counts; a receiver covering the band must not match."""
    cant = _cant((2, 2.0, 30.0, 55.0), (1, 2.0, 30.0, 43.0))
    assert pwrdb(cant, 14.0) == np.float32(43.0)
    assert pwrdb(_cant((2, 2.0, 30.0, 55.0)), 14.0) == np.float32(30.0)


def test_first_matching_entry_wins():
    cant = _cant((1, 2.0, 15.0, 41.0), (1, 2.0, 30.0, 52.0))
    assert pwrdb(cant, 10.0) == np.float32(41.0)
    assert pwrdb(cant, 20.0) == np.float32(52.0)


def test_range_is_inclusive_at_both_ends():
    cant = _cant((1, 5.0, 15.0, 44.0))
    assert pwrdb(cant, 5.0) == np.float32(44.0)
    assert pwrdb(cant, 15.0) == np.float32(44.0)
    assert pwrdb(cant, 4.999) == np.float32(30.0)
    assert pwrdb(cant, 15.001) == np.float32(30.0)


def test_clamp_applies_to_the_range_test_not_the_result():
    """0.5 MHz is tested as 2 MHz, so a 2-30 entry matches and returns ITS dBW.

    The returned power is the table value untouched -- the clamp never scales
    it, and a frequency far outside the band still reports the entry's power.
    """
    cant = _cant((1, 2.0, 30.0, 46.0))
    assert pwrdb(cant, 0.5) == np.float32(46.0)
    assert pwrdb(cant, 1.9) == np.float32(46.0)
    assert pwrdb(cant, 45.0) == np.float32(46.0)
    # an entry that excludes the clamped endpoint still misses
    assert pwrdb(_cant((1, 3.0, 30.0, 46.0)), 0.5) == np.float32(30.0)
    assert pwrdb(_cant((1, 2.0, 25.0, 46.0)), 45.0) == np.float32(30.0)
