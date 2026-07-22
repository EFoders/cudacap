"""Stage A ESMOD + ESREG vs the tier-B oracle dump.

The two run together in LUFFY's order over disjoint /ZON/ slots -- ESMOD
writes 4 and 5, ESREG writes 6 and 7 -- so the dump pins both the arithmetic
and the fact that neither disturbs the other's slots.

The oracle presets all seven slots to 12345. (NMODE -9) before calling, so a
field neither routine writes comes back visibly untouched rather than merely
zero; the Python side seeds the same sentinel.  That is what makes the
"ESREG does nothing but preset" claim testable rather than merely asserted:
if any of its 280 lines of dead mixed Es-F code ran, slot 6 would carry
computed values instead of the markers.

Two bars, as for REGMOD: an absolute one and a floored-ULP one.  Fields that
only copy an input, or are constants, are asserted bit-exact.

Measured: 42,862 of 42,872 cells bit-exact (99.98%), worst 4.8e-7 absolute
and 2.2 floored ULP.  Only ABPS and GRLOS drift at all, and only at 1-2 ULP:
ABPS from the 1/SQRT(1 - SINP**2) grazing-angle term, GRLOS from summing
GAIN's Fresnel result over the sample areas.  TLOSS stays exact despite
containing both, because a 3e-7 perturbation is far below one ULP at the
150 dB scale it lands on -- the same conditioning that made REGMOD's FLDST
read 128 floored ULP at 3.6e-5 absolute, running the other way.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.antennastate import CantennaState
from voacap_gpu.stage_a.esmod import esmod
from voacap_gpu.stage_a.esreg import esreg
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.mufstate import MufState
from voacap_gpu.stage_a.noisestate import AnoisState
from voacap_gpu.stage_a.raysstate import RaysState
from voacap_gpu.stage_a.readant import read_antenna
from voacap_gpu.stage_a.signalstate import SigdState
from voacap_gpu.stage_a.zonstate import ZonState

from .dumputil import ANTENNAS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("esmod")

SENTINEL = F32(12345.0)
SENTINEL_NMODE = -9

ESMOD_ABS_BAR = 1.0e-5          # measured worst 4.8e-7
ESMOD_ULP_BAR = 16.0            # measured worst 2.2
# copies of an input, or constants: no arithmetic can drift them
ESMOD_EXACT = {"nmode", "adv", "hn", "hp", "crel", "rely", "spro", "eff",
               "rgain", "tgain"}

FIELDS = ("abps", "adv", "b", "crel", "eff", "fldst", "fslos", "grlos", "hn",
          "hp", "nmode", "obf", "prob", "rely", "rgain", "sigpow", "sn",
          "spro", "tgain", "timed", "tllow", "tlhgh", "tloss")
SLOTS = (4, 5, 6, 7)

# column order in gen_esmod_cases.py
COLUMNS = (
    ["jmode", "freq", "gcdkm", "amind", "km", "ifob13",
     "acav", "afe", "bfe", "asm", "dsl", "dsu", "rcnse",
     "sigl4", "sigu4", "ymuf4", "yfot4", "yhpf4"]
    + [f"fs1_{i}" for i in range(1, 6)]
    + [f"fs2_{i}" for i in range(1, 6)]
    + [f"fs3_{i}" for i in range(1, 6)]
    + [f"hs{i}" for i in range(1, 6)]
    + [f"fi1_{i}" for i in range(1, 6)]
    + [f"gyz{i}" for i in range(1, 6)]
    + [f"sigpat{i}" for i in range(1, 6)]
    + [f"epspat{i}" for i in range(1, 6)]
    + ["pwrdba", "txlo", "txhi", "rxlo", "rxhi"]
)

RZ = F32(6370.0)

_ANT = None


def _cantenna(txlo, txhi, rxlo, rxhi, pwrdba) -> CantennaState:
    """sample.14 as both /cantenna/ terminals, over per-case ranges."""
    global _ANT
    if _ANT is None:
        _ANT = read_antenna(ANTENNAS_DIR / "sample.14").p2p_tables()
    array, aeff = _ANT
    cant = CantennaState()
    cant.numants = 2
    cant.iats[1], cant.iats[2] = 1, 2
    cant.xfqs[1], cant.xfqe[1] = F32(txlo), F32(txhi)
    cant.xfqs[2], cant.xfqe[2] = F32(rxlo), F32(rxhi)
    cant.offazim[1] = cant.offazim[2] = F32(0.0)
    for iat in (1, 2):
        cant.array[1:31, 1:92, iat] = array
        cant.aeff[1:31, iat] = aeff
    cant.pwrdba[1] = F32(pwrdba)
    return cant


def _seeded_zon() -> ZonState:
    """All seven slots at the oracle's sentinel, not zero."""
    zon = ZonState()
    for name in FIELDS:
        if name == "nmode":
            zon.nmode[1:8] = SENTINEL_NMODE
        else:
            getattr(zon, name)[1:8] = SENTINEL
    return zon


def _run(inp):
    """One case: build the injected state, run ESMOD then ESREG."""
    v = dict(zip(COLUMNS, inp))
    state = LayerState()
    state.km = int(float(v["km"]))
    state.kfx = 1
    for i in range(1, 6):
        state.fs[1, i] = F32(v[f"fs1_{i}"])
        state.fs[2, i] = F32(v[f"fs2_{i}"])
        state.fs[3, i] = F32(v[f"fs3_{i}"])
        state.hs[i] = F32(v[f"hs{i}"])
        state.fi[1, i] = F32(v[f"fi1_{i}"])
        state.gyz[i] = F32(v[f"gyz{i}"])
        state.sigpat[i] = F32(v[f"sigpat{i}"])
        state.epspat[i] = F32(v[f"epspat{i}"])
    muf = MufState()
    muf.sigl[4] = F32(v["sigl4"])
    muf.sigu[4] = F32(v["sigu4"])
    muf.ymuf[4] = F32(v["ymuf4"])
    muf.yfot[4] = F32(v["yfot4"])
    muf.yhpf[4] = F32(v["yhpf4"])
    sigd = SigdState(acav=F32(v["acav"]), afe=F32(v["afe"]),
                     bfe=F32(v["bfe"]), asm=F32(v["asm"]),
                     dsl=F32(v["dsl"]), dsu=F32(v["dsu"]))
    anois = AnoisState(rcnse=F32(v["rcnse"]))
    jmode = int(float(v["jmode"]))
    rays = RaysState()
    rays.ifob[1, 3, jmode] = int(float(v["ifob13"]))
    cant = _cantenna(v["txlo"], v["txhi"], v["rxlo"], v["rxhi"], v["pwrdba"])
    zon = _seeded_zon()
    gcdkm = F32(v["gcdkm"])
    esmod(state, muf, sigd, anois, rays, cant, zon,
          F32(v["freq"]), jmode, gcdkm / RZ, gcdkm, F32(v["amind"]))
    esreg(zon)
    return zon


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_esmod_dumps.sh")
def test_esmod_esreg_vs_oracle():
    header, inputs, rows = DUMP
    assert header == [f"{n}{s}" for n in FIELDS for s in SLOTS]
    total = exact = 0
    worst_abs = worst_ulp = 0.0
    worst_at = None
    seen = {"one_hop": 0, "two_hop": 0, "no_mode": 0, "amind_cut": 0,
            "slot5_live": 0, "slot5_preset": 0}
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        zon = _run(inp)
        got = {f"{n}{s}": getattr(zon, n)[s] for n in FIELDS for s in SLOTS}
        seen["one_hop"] += zon.hn[4] == 1.0
        seen["two_hop"] += zon.hn[4] == 2.0
        seen["no_mode"] += zon.hn[4] == -1.0
        seen["amind_cut"] += zon.hn[4] == -1.0 and zon.hn[5] > 0
        seen["slot5_live"] += zon.hn[5] > 0
        seen["slot5_preset"] += zon.hn[5] == -1.0
        for name, e_str in zip(header, exp):
            g = got[name]
            e = np.float32(e_str)
            if name.rstrip("4567") in ESMOD_EXACT:
                assert g == e, (f"case {k} {name}: got {g!r} expected {e!r} "
                                f"(must be bit-exact)")
                total += 1
                exact += 1
                continue
            a = abs(float(g) - float(e))
            d = ulp_diff_floored(g, e)
            if a > worst_abs:
                worst_abs, worst_at = a, f"case {k} {name}"
            worst_ulp = max(worst_ulp, d)
            assert a <= ESMOD_ABS_BAR and d <= ESMOD_ULP_BAR, (
                f"case {k} {name}: got {g!r} expected {e!r} "
                f"({a:.3e} abs, {d:.1f} floored ULP)")
            total += 1
            exact += a == 0.0
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    assert exact / total >= 0.999, (
        f"exact {exact}/{total} (worst {worst_abs:.3e} abs at {worst_at}, "
        f"{worst_ulp:.1f} ULP)")


def test_esreg_only_presets_and_only_slots_6_and_7():
    """ESREG's 280 lines of mixed Es-F code are dead: unconditional RETURN."""
    zon = _seeded_zon()
    esreg(zon)
    for s in (1, 2, 3, 4, 5):
        assert zon.abps[s] == SENTINEL and zon.nmode[s] == SENTINEL_NMODE, (
            f"ESREG touched slot {s}")
    for s in (6, 7):
        assert zon.hn[s] == F32(-1.0) and zon.hp[s] == F32(-1.0)
        assert zon.nmode[s] == 5
        assert zon.tloss[s] == F32(1000.0)
        assert zon.crel[s] == F32(1000.0)      # ESMOD's slots get -1000
        assert not any(getattr(zon, n)[s] == SENTINEL for n in FIELDS), (
            f"ESREG left a field of slot {s} unwritten")


def test_esmod_presets_both_slots_before_any_early_return():
    """The three early returns must still leave slots 4 and 5 readable."""
    state = LayerState()
    state.km = 1
    state.hs[1] = F32(110.0)
    state.fs[2, 1] = F32(-1.0)             # no Es layer -> early return
    rays = RaysState()
    rays.ifob[1, 3, 1] = 2500
    zon = _seeded_zon()
    esmod(state, MufState(), SigdState(), AnoisState(), rays,
          CantennaState(), zon, F32(10.0), 1, F32(0.3), F32(1900.0),
          F32(3.0))
    for s in (4, 5):
        assert zon.hn[s] == F32(-1.0) and zon.nmode[s] == 5
        assert zon.crel[s] == F32(-1000.0)  # ESREG's slots get +1000
        assert not any(getattr(zon, n)[s] == SENTINEL for n in FIELDS)
    for s in (1, 2, 3, 6, 7):
        assert zon.abps[s] == SENTINEL, f"ESMOD touched slot {s}"


def test_two_hop_path_writes_slot_4_not_slot_5():
    """A path needing two Es hops runs IHOP = 2,2 -- one iteration, slot 4.

    Slot 4 is "the first mode found", not "the one-hop mode"; slot 5 keeps
    its presets.  This is the detail most likely to be mis-ported.
    """
    state = LayerState()
    state.km = 1
    state.hs[1] = F32(110.0)
    state.fs[1, 1], state.fs[2, 1], state.fs[3, 1] = (F32(4.0), F32(5.0),
                                                      F32(7.0))
    state.fi[1, 1] = F32(3.0)
    state.gyz[1] = F32(1.2)
    state.sigpat[1], state.epspat[1] = F32(5.0), F32(80.0)
    muf = MufState()
    muf.sigl[4], muf.sigu[4] = F32(0.15), F32(0.20)
    muf.ymuf[4], muf.yfot[4], muf.yhpf[4] = F32(14.0), F32(11.0), F32(17.0)
    sigd = SigdState(acav=F32(0.12), afe=F32(0.5), bfe=F32(1.5),
                     asm=F32(3.0), dsl=F32(8.0), dsu=F32(9.0))
    rays = RaysState()
    rays.ifob[1, 3, 1] = 2500

    def run(gcdkm):
        zon = _seeded_zon()
        esmod(state, muf, sigd, AnoisState(rcnse=F32(-145.0)), rays,
              _cantenna(2.0, 30.0, 2.0, 30.0, 40.0), zon,
              F32(10.0), 1, F32(gcdkm) / RZ, F32(gcdkm), F32(0.0))
        return zon

    short = run(1500.0)                       # IHSRT = 1 -> IHOP 1 and 2
    assert short.hn[4] == 1.0 and short.hn[5] == 2.0
    far = run(3500.0)                         # IHSRT = 2 -> IHOP 2 only
    assert far.hn[4] == 2.0, "the two-hop mode belongs in slot 4"
    assert far.hn[5] == -1.0, "slot 5 must keep its presets"
    assert far.nmode[5] == 5 and far.tloss[5] == F32(1000.0)
    too_far = run(9000.0)                     # IHSRT > 2 -> no Es mode at all
    assert too_far.hn[4] == -1.0 and too_far.hn[5] == -1.0
