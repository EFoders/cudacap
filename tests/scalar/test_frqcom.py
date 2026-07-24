"""Stage A FRQCOM + SETLUF vs the tier-B oracle dump.

The driver builds the frequency list with FRQCOM before LUFFY and reads it
back with SETLUF after, so the dump drives them in that order: FRQCOM's FREL
IS SETLUF's input.  Both routines' outputs are compared separately, so a
failure still localises.

Measured: 399 cases x 15 columns = 5,985 cells, ALL BIT-EXACT.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.frqcom import FrequencyIndexError, frqcom
from voacap_gpu.scalar.frqstate import FrqState
from voacap_gpu.scalar.mufstate import MufState
from voacap_gpu.scalar.setluf import setluf
from voacap_gpu.scalar.sonstate import SonState

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("frqcom")

FRQCOM_ULP_BAR = 4.0            # measured worst 0.0 -- a libm safety valve
NF = 11
FLOW, FHIGH = 2.0, 40.0
SEED_FREL = F32(-4444.0)
SEED_XLUF = F32(-5555.0)

COLUMNS = (["it", "ifreq", "allmuf", "fot", "emuf", "f2muf", "hpf", "lufp"]
           + [f"frel{i}" for i in range(1, NF + 1)] + ["fr13"]
           + [f"rel{i}" for i in range(1, NF + 1)])


def _which_case(emuf, f2muf, hpf) -> str:
    """FRQCOM's six-way branch, from the two bracketing frequencies."""
    xfl = max(FLOW, min(emuf, f2muf))
    xfh = min(hpf, FHIGH)
    if xfh <= FLOW:
        return "case1"
    if xfl <= FLOW:
        return "case2"
    if xfh <= FLOW + 20.0:
        return "case3"
    if xfl <= FLOW + 2.0:
        return "case4"
    return "case5"          # case 6's guard is the constant 36 -- see frqcom.py


def _run(inp):
    v = dict(zip(COLUMNS, inp))
    it = int(float(v["it"]))
    ifreq = int(float(v["ifreq"]))
    muf = MufState()
    for name in ("allmuf", "fot", "emuf", "f2muf", "hpf"):
        getattr(muf, name)[it] = F32(v[name])
    muf.xluf[it] = SEED_XLUF
    frq = FrqState()
    frq.frel[:] = SEED_FREL
    for i in range(1, NF + 1):
        frq.frel[i] = F32(v[f"frel{i}"])
    frq.frel[12] = SEED_FREL
    frq.frel[13] = F32(v["fr13"])
    son = SonState()
    for i in range(1, NF + 1):
        son.reliab[i] = F32(v[f"rel{i}"])

    if ifreq <= 0:                          # HFMUFS's dispatch, verbatim
        frqcom(muf, frq.frel, ifreq, it)
    else:
        frq.frel[12] = muf.allmuf[it]
    got = {f"qfrel{i:02d}": frq.frel[i] for i in range(1, 14)}
    setluf(muf, frq, son, int(float(v["lufp"])), it)
    got["xluf"] = muf.xluf[it]
    got["luf13"] = frq.frel[13]
    return got, v


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_frqcom_dumps.sh")
def test_frqcom_setluf_vs_oracle():
    header, inputs, rows = DUMP
    total = exact = 0
    worst_ulp = 0.0
    worst_at = None
    seen = {"case1": 0, "case2": 0, "case3": 0, "case4": 0, "case5": 0,
            "fot_only": 0, "passthrough": 0, "case1_kept_slot13": 0,
            "luf_at_first": 0, "luf_later": 0, "no_luf": 0,
            "blank_slot1": 0, "seed_survived": 0}
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got, v = _run(inp)
        ifreq = int(float(v["ifreq"]))
        arm = _which_case(float(v["emuf"]), float(v["f2muf"]),
                          float(v["hpf"]))
        if ifreq > 0:
            seen["passthrough"] += 1
        elif ifreq == -10:
            seen["fot_only"] += 1
        else:
            seen[arm] += 1
            if arm == "case1":
                seen["case1_kept_slot13"] += (
                    got["qfrel13"] == F32(v["fr13"]))
        xluf = float(got["xluf"])
        seen["no_luf"] += xluf <= 0.0
        seen["luf_at_first"] += (xluf > 0.0
                                 and xluf == float(got["qfrel01"]))
        seen["luf_later"] += (xluf > 0.0
                              and xluf != float(got["qfrel01"]))
        seen["blank_slot1"] += float(v["frel1"]) <= 0.0
        seen["seed_survived"] += got["xluf"] == SEED_XLUF

        for name, e_str in zip(header, exp):
            g = got[name]
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            if d > worst_ulp:
                worst_ulp, worst_at = d, f"case {k} {name}"
            assert d <= FRQCOM_ULP_BAR, (
                f"case {k} {name}: got {g!r} expected {e!r} "
                f"({abs(float(g) - float(e)):.3e} abs, {d:.1f} floored ULP)")
            total += 1
            exact += d == 0.0
    for arm, n in seen.items():
        if arm == "seed_survived":
            assert n == 0, "SETLUF left XLUF unwritten, which it cannot do"
            continue
        assert n, f"branch never reached: {arm} ({seen})"
    assert exact / total >= 0.999, (
        f"exact {exact}/{total} (worst {worst_ulp:.1f} ULP at {worst_at})")


def _muf(**kw):
    muf = MufState()
    for name, val in kw.items():
        getattr(muf, name)[12] = F32(val)
    return muf


def test_case_one_alone_leaves_the_luf_slot_alone():
    """Cases 2-5 zero FREA(13); case 1, the no-band arm, does not."""
    frq = FrqState()
    frq.frel[13] = F32(17.25)
    frqcom(_muf(allmuf=18.0, emuf=1.5, f2muf=1.5, hpf=1.0),
           frq.frel, -1, 12)
    assert frq.frel[13] == F32(17.25)
    assert frq.frel[1] == F32(2.0) and frq.frel[11] == F32(22.0)

    frq = FrqState()
    frq.frel[13] = F32(17.25)
    frqcom(_muf(allmuf=18.0, emuf=1.5, f2muf=1.5, hpf=30.0),
           frq.frel, -1, 12)                # case 2
    assert frq.frel[13] == F32(0.0)


def test_case_three_past_the_end_is_refused():
    """A lower MUF far above the highest usable frequency drives NE past
    FREA's 13 elements, where the Fortran writes off the end of the common."""
    frq = FrqState()
    with pytest.raises(FrequencyIndexError):
        frqcom(_muf(allmuf=18.0, emuf=21.0, f2muf=21.0, hpf=2.5),
               frq.frel, -1, 12)


def test_the_fot_arm_touches_only_slots_one_and_twelve():
    frq = FrqState()
    frq.frel[:] = F32(-4444.0)
    frqcom(_muf(allmuf=18.4, fot=14.7), frq.frel, -10, 12)
    assert frq.frel[1] == F32(14.7) and frq.frel[12] == F32(18.4)
    assert all(frq.frel[i] == F32(-4444.0) for i in [*range(2, 12), 13])


def _setluf(freqs, rels, lufp=90):
    muf = MufState()
    frq = FrqState()
    son = SonState()
    for i, (f, r) in enumerate(zip(freqs, rels), start=1):
        frq.frel[i] = F32(f)
        son.reliab[i] = F32(r)
    setluf(muf, frq, son, lufp, 12)
    return muf.xluf[12], frq.frel[13]


def test_the_luf_is_the_lowest_frequency_that_makes_the_grade():
    xluf, luf13 = _setluf([3.2, 5.1, 7.3, 9.6], [0.2, 0.5, 0.95, 0.99])
    assert xluf == F32(7.3) and luf13 == F32(7.3)


def test_no_luf_reports_the_best_frequency_negated():
    xluf, luf13 = _setluf([3.2, 5.1, 7.3, 9.6], [0.2, 0.5, 0.4, 0.1])
    assert xluf == F32(-5.1), "5.1 MHz came closest"
    assert luf13 == F32(5.1), "and slot 13 keeps the magnitude"


def test_the_search_is_seeded_from_slot_one_even_when_it_is_blank():
    """RF and REL come from FREL(1)/RELIAB(1) before the loop, and the loop's
    "no frequency here" skip does not protect the seed."""
    # Slot 1 is blank but its reliability still sets the bar: 0.30 and 0.25
    # both fail to beat it, so RF stays at the blank slot's own value.
    xluf, luf13 = _setluf([0.0, 5.1, 7.3], [0.40, 0.30, 0.25])
    assert xluf == F32(0.0) and luf13 == F32(0.0)
    # Raise 7.3 MHz past the seed and it becomes the fallback.
    xluf, luf13 = _setluf([0.0, 5.1, 7.3], [0.40, 0.30, 0.55])
    assert xluf == F32(-7.3) and luf13 == F32(7.3)


def test_required_reliability_is_inclusive():
    assert _setluf([3.2, 5.1], [0.90, 0.95], lufp=90)[0] == F32(3.2)
    assert _setluf([3.2, 5.1], [0.899, 0.95], lufp=90)[0] == F32(5.1)
