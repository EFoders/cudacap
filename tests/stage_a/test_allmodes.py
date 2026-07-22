"""Stage A ALLMODES vs the tier-B oracle dump.

ALLMODES does no arithmetic -- only assignment and index selection -- so the
bar is bit-exact everywhere, and any mismatch is a bookkeeping error rather
than a numerical one.

The dump is a *call sequence*, not independent cases: /allMODE/ accumulates
across calls and NMMOD is both the running count and the write cursor, so the
test threads one AllModeState through every record in order.  That means a
single wrong index poisons everything after it, which is exactly the property
worth having here.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.allmodes import AllModesOverflow, allmodes
from voacap_gpu.stage_a.allmodestate import AllModeState
from voacap_gpu.stage_a.zonstate import ZonState

from .dumputil import load_dump

DUMP = load_dump("allmodes")

# /ZON/ fields the case file carries, in order (gen_allmodes_cases.py)
ZON_FIELDS = ("abps", "crel", "fldst", "hn", "hp", "prob", "rely", "rgain",
              "sigpow", "sn", "spro", "tgain", "timed", "tloss", "b", "fslos",
              "grlos", "adv", "obf", "tllow", "tlhgh", "eff", "nmode")

# /allMODE/ arrays the dump carries, in order (dump_allmodes.f)
Z_FIELDS = ("abps", "crel", "fldst", "hn", "hp", "prob", "rely", "rgain",
            "sigpow", "sn", "spro", "tgain", "timed", "tloss", "b", "fslos",
            "grlos", "adv", "obf", "tllow", "tlhgh", "eff")

_O_Z = 0                      # 22 arrays x 20
_O_ZNMODE = 22 * 20           # 20
_O_ZONHP = _O_ZNMODE + 20     # 7
_O_NMMOD = _O_ZONHP + 7
_O_FREQ = _O_NMMOD + 1


def _zon(row) -> ZonState:
    """The /ZON/ block one case record injects."""
    zon = ZonState()
    for j, name in enumerate(ZON_FIELDS):
        arr = getattr(zon, name)
        for i in range(1, 8):
            v = float(row[2 + j * 7 + i - 1])
            arr[i] = int(round(v)) if name == "nmode" else np.float32(v)
    return zon


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_allmodes_dumps.sh")
def test_allmodes_vs_oracle():
    header, inputs, rows = DUMP
    assert header[_O_NMMOD] == "nmmod" and header[_O_FREQ] == "freq"
    assert header[_O_ZONHP] == "zonhp1"
    allm = AllModeState()          # threaded through the whole sequence
    freq = np.float32(0.0)
    seen = {0: 0, 1: 0, 2: 0, 3: 0}
    peak = 0
    for kc, (inp, exp) in enumerate(zip(inputs, rows)):
        iflg = int(float(inp[0]))
        fval = float(inp[1])
        zon = _zon(inp)
        got = allmodes(zon, allm, iflg, fval)
        if iflg == 0:
            freq = got
        else:
            assert got is None, f"case {kc}: only the reset returns a FREQ"
        seen[iflg if iflg in seen else 3] += 1
        peak = max(peak, allm.nmmod)

        assert allm.nmmod == int(float(exp[_O_NMMOD])), (
            f"case {kc} (iflg={iflg}) nmmod: got {allm.nmmod} "
            f"expected {int(float(exp[_O_NMMOD]))}")
        assert freq == np.float32(exp[_O_FREQ]), (
            f"case {kc} freq: got {freq!r} expected {exp[_O_FREQ]}")
        for j, name in enumerate(Z_FIELDS):
            arr = getattr(allm, name)
            for i in range(1, 21):
                e = np.float32(exp[_O_Z + j * 20 + i - 1])
                assert arr[i] == e, (f"case {kc} (iflg={iflg}) z{name}[{i}]: "
                                     f"got {arr[i]!r} expected {e!r}")
        for i in range(1, 21):
            e = int(float(exp[_O_ZNMODE + i - 1]))
            assert int(allm.nmode[i]) == e, (
                f"case {kc} (iflg={iflg}) znmode[{i}]: "
                f"got {int(allm.nmode[i])} expected {e}")
        # the IFLG=1 arm writes HP(2) = HP(1) back into /ZON/
        for i in range(1, 8):
            e = np.float32(exp[_O_ZONHP + i - 1])
            assert zon.hp[i] == e, (f"case {kc} (iflg={iflg}) /ZON/ hp[{i}]: "
                                    f"got {zon.hp[i]!r} expected {e!r}")
    assert all(seen[i] for i in (0, 1, 2, 3)), f"unreached IFLG arm: {seen}"
    assert peak >= 18, f"accumulation only reached slot {peak} of 20"


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_reset_clears_to_the_no_mode_markers():
    """A reset must restore all 20 slots, not just the ones in use."""
    allm = AllModeState()
    zon = ZonState()
    for i in range(1, 8):
        zon.hp[i] = np.float32(300.0)
    allmodes(zon, allm, 3)
    assert allm.nmmod == 6
    assert allmodes(zon, allm, 0, 21.5) == np.float32(21.5)
    assert allm.nmmod == 0
    assert np.all(allm.tloss[1:21] == np.float32(99999.0))
    assert np.all(allm.tllow[1:21] == np.float32(999.0))
    assert np.all(allm.tlhgh[1:21] == np.float32(999.0))
    assert np.all(allm.hp[1:21] == np.float32(-1.0))


def test_overflow_is_refused_not_wrapped():
    """Past 20 slots the Fortran writes into the next common-block variable."""
    allm = AllModeState()
    zon = ZonState()
    for i in range(1, 8):
        zon.hp[i] = np.float32(300.0)
    for _ in range(3):
        allmodes(zon, allm, 3)          # 6 each -> 18
    assert allm.nmmod == 18
    with pytest.raises(AllModesOverflow):
        allmodes(zon, allm, 3)          # would reach 24
