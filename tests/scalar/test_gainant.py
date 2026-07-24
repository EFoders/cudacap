"""Stage A GAIN antenna branch (ITR>=0, point-to-point) vs the tier-B oracle.

The lookup is pure float32 multiply-add over a table -- no transcendental, no
cancellation -- so unlike the ground-reflection branch this one has no
[[f32-libm-noise]] floor to sit above.  The bar is bit-exact.

/cantenna/ is loaded from sample.14 rather than recomputed, matching the
oracle driver: the FINDF experience is that recomputing a table the routine
under test then interpolates is how libm noise gets amplified into a real
error.  Here the table is read by the parser the READANT dump already pinned.

The grid drives both interpolation axes plus the branches that are easy to get
wrong by transcription: MIN0 clamping the top frequency and elevation cells
into a flat hold, the inclusive 15 MHz seam between two /cantenna/ entries,
and the "no antenna matched" early return that must yield 0 dB rather than an
out-of-range index.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.gain import antenna_gain
from voacap_gpu.scalar.readant import read_antenna

from .dumputil import ANTENNAS_DIR, load_dump

DUMP = load_dump("gainant")
SAMPLE = ANTENNAS_DIR / "sample.14"


def _cantenna() -> CantennaState:
    """The three entries dump_gainant.f installs, built the same way."""
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


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_gainant_dumps.sh")
def test_gain_antenna_vs_oracle():
    header, inputs, rows = DUMP
    assert header == ["rain", "eff"]
    cant = _cantenna()
    matched = 0
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        itr = int(inp[0])
        delta, fmc = (float(x) for x in inp[1:3])
        got_rain, got_eff = antenna_gain(cant, itr, delta, fmc)
        exp_rain, exp_eff = np.float32(exp[0]), np.float32(exp[1])
        assert got_rain == exp_rain, (f"case {k} rain: got {got_rain!r} "
                                      f"expected {exp_rain!r} in {inp}")
        assert got_eff == exp_eff, (f"case {k} eff: got {got_eff!r} "
                                    f"expected {exp_eff!r} in {inp}")
        matched += got_eff != np.float32(0.0)
    assert len(rows) == 592
    # guard against a grid that accidentally matches nothing and passes trivially
    assert matched > len(rows) // 2, f"only {matched} cases selected an antenna"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_no_matching_antenna_returns_zero():
    cant = _cantenna()
    assert antenna_gain(cant, 1, 0.2, 1.5) == (0.0, 0.0)    # below every range
    assert antenna_gain(cant, 3, 0.2, 10.0) == (0.0, 0.0)   # no such terminal


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_below_table_is_refused_not_wrapped():
    """GAIN clamps only the top of each axis; Python must not index backwards."""
    cant = _cantenna()
    cant.xfqs[1] = np.float32(0.0)          # let a sub-1 MHz case through
    with pytest.raises(IndexError):
        antenna_gain(cant, 1, 0.2, 0.5)
    with pytest.raises(IndexError):
        antenna_gain(cant, 1, -0.2, 10.0)
