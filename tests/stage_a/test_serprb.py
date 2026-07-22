"""Stage A SERPRB + MPATH vs the tier-B oracle dump.

Both read /allMODE/ and write one /SON/ column, and they run in sequence in
LUFFY, so the dump drives them together.  NREL is injected rather than
produced, which keeps both independent of RELBIL's mode selection -- a
failure here is these two routines, not the one before them.

The two routines disagree about what a mode is, which is worth pinning: for
SERPRB a slot is dead at HP <= 70 km, for MPATH at HP <= 0.  A slot between
those two thresholds is live for one and dead for the other, and the case
grid puts modes there.

Measured: 375 cases x 26 columns = 9,750 cells, ALL BIT-EXACT.  One ULP bar
as a libm safety valve plus the exactness floor is the whole tolerance story.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.allmodestate import AllModeState
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.mpath import mpath
from voacap_gpu.stage_a.noisestate import AnoisState
from voacap_gpu.stage_a.serprb import NoModesError, serprb
from voacap_gpu.stage_a.signalstate import TonState
from voacap_gpu.stage_a.sonstate import SonState

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("serprb")

SERPRB_ULP_BAR = 4.0            # measured worst 0.0 -- a libm safety valve
SENTINEL = F32(-8888.0)
NM = 20

SCALARS = ["ifx", "rsn", "lufp", "du", "dl", "sigm", "sygu", "sygl",
           "ads", "sus", "sls", "nmmod", "nrel", "dmp", "pmp", "gcdkm",
           "method", "mspec"]
ARRAYS = ["hp", "sn", "tllow", "tlhgh", "sigpow", "timed", "rely"]
COLUMNS = SCALARS + [f"{a}{i}" for a in ARRAYS for i in range(1, NM + 1)]


def _run(inp):
    v = dict(zip(COLUMNS, inp))
    allm = AllModeState()
    allm.nmmod = int(float(v["nmmod"]))
    allm.nrel = int(float(v["nrel"]))
    for i in range(1, NM + 1):
        for a in ARRAYS:
            getattr(allm, a)[i] = F32(v[f"{a}{i}"])
        allm.spro[i] = SENTINEL
    anois = AnoisState(du=F32(v["du"]), dl=F32(v["dl"]), sigm=F32(v["sigm"]),
                       sygu=F32(v["sygu"]), sygl=F32(v["sygl"]))
    ton = TonState(ads=F32(v["ads"]), sus=F32(v["sus"]), sls=F32(v["sls"]),
                   spr=F32(-7777.0))
    son = SonState(d10s=F32(-6666.0), d50s=F32(-6666.0), d90s=F32(-6666.0))
    son.sprob[1:] = SENTINEL
    son.probmp[1:] = SENTINEL
    ifx = int(float(v["ifx"]))
    serprb(allm, anois, ton, son, ifx, F32(v["rsn"]),
           int(float(v["lufp"])))
    mpath(allm, son, ifx, F32(v["dmp"]), F32(v["pmp"]), F32(v["gcdkm"]),
          int(float(v["method"])), int(float(v["mspec"])))
    got = {f"spro{i:02d}": allm.spro[i] for i in range(1, NM + 1)}
    got["sprob"] = son.sprob[ifx]
    got["probmp"] = son.probmp[ifx]
    got["d10s"], got["d50s"], got["d90s"] = son.d10s, son.d50s, son.d90s
    got["spr"] = ton.spr
    return got, allm, son, v


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_serprb_dumps.sh")
def test_serprb_mpath_vs_oracle():
    header, inputs, rows = DUMP
    total = exact = 0
    worst_ulp = 0.0
    worst_at = None
    seen = {"lufp_under_50": 0, "lufp_over_50": 0, "dead_mode": 0,
            "all_dead": 0, "mp_off": 0, "mp_area_cut": 0, "mp_found": 0,
            "mp_none": 0, "live_for_one_only": 0}
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got, allm, son, v = _run(inp)
        n = int(float(v["nmmod"]))
        hps = [float(v[f"hp{i}"]) for i in range(1, n + 1)]
        lufp = int(float(v["lufp"]))
        ifx = int(float(v["ifx"]))
        seen["lufp_under_50"] += lufp < 50
        seen["lufp_over_50"] += lufp >= 50
        seen["dead_mode"] += any(h <= 70.0 for h in hps)
        seen["all_dead"] += all(h <= 70.0 for h in hps)
        seen["live_for_one_only"] += any(0.0 < h <= 70.0 for h in hps)
        seen["mp_off"] += (float(v["dmp"]) <= 0.0 or float(v["pmp"]) <= 0.0)
        seen["mp_area_cut"] += (int(float(v["method"])) == 20
                                and int(float(v["mspec"])) == 121
                                and float(v["gcdkm"]) > 7000.0)
        seen["mp_found"] += son.probmp[ifx] > 0.001
        seen["mp_none"] += son.probmp[ifx] == F32(0.001)
        for name, e_str in zip(header, exp):
            g = got[name]
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            if d > worst_ulp:
                worst_ulp, worst_at = d, f"case {k} {name}"
            assert d <= SERPRB_ULP_BAR, (
                f"case {k} {name}: got {g!r} expected {e!r} "
                f"({abs(float(g) - float(e)):.3e} abs, {d:.1f} floored ULP)")
            total += 1
            exact += d == 0.0
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    assert exact / total >= 0.999, (
        f"exact {exact}/{total} (worst {worst_ulp:.1f} ULP at {worst_at})")


def _allm(n, **kw):
    allm = AllModeState()
    allm.nmmod = n
    allm.nrel = kw.get("nrel", 1)
    for i in range(1, n + 1):
        allm.hp[i] = F32(kw.get("hp", [300.0] * n)[i - 1])
        allm.sn[i] = F32(kw.get("sn", [20.0] * n)[i - 1])
        allm.tllow[i] = F32(9.0)
        allm.tlhgh[i] = F32(8.0)
        allm.sigpow[i] = F32(kw.get("sigpow", [-120.0] * n)[i - 1])
        allm.timed[i] = F32(kw.get("timed", [11.0] * n)[i - 1])
        allm.rely[i] = F32(kw.get("rely", [0.5] * n)[i - 1])
    return allm


def test_no_modes_is_refused():
    """With NMMOD <= 0 the Fortran reads uninitialised D10SA/D50SA locals."""
    allm = AllModeState()
    allm.nmmod = 0
    with pytest.raises(NoModesError):
        serprb(allm, AnoisState(), TonState(), SonState(), 3, F32(24.0), 90)


def test_the_two_routines_disagree_about_a_dead_mode():
    """SERPRB kills a mode at HP <= 70 km; MPATH only at HP <= 0."""
    allm = _allm(2, hp=[300.0, 40.0], timed=[11.0, 30.0], rely=[0.5, 0.77])
    son = SonState()
    serprb(allm, AnoisState(du=F32(9.7), dl=F32(6.0), sigm=F32(5.4),
                            sygu=F32(1.5), sygl=F32(1.5)),
           TonState(ads=F32(5.6), sus=F32(0.6), sls=F32(1.63)),
           son, 3, F32(24.0), 90)
    assert allm.spro[2] == F32(0.001), "40 km is not a mode for SERPRB"
    mpath(allm, son, 3, F32(0.85), F32(10.0), F32(3000.0), 30, 0)
    assert son.probmp[3] == F32(0.77), "but it still counts as multipath"


def test_multipath_keeps_modes_OUTSIDE_the_delay_tolerance():
    """The header comment says "within"; the code skips within and keeps
    outside, which is what multipath interference actually means."""
    son = SonState()
    close = _allm(2, timed=[11.0, 11.5], rely=[0.5, 0.9])
    mpath(close, son, 3, F32(0.85), F32(10.0), F32(3000.0), 30, 0)
    assert son.probmp[3] == F32(0.001), "0.5 ms apart is not multipath"
    far = _allm(2, timed=[11.0, 14.0], rely=[0.5, 0.9])
    mpath(far, son, 3, F32(0.85), F32(10.0), F32(3000.0), 30, 0)
    assert son.probmp[3] == F32(0.9)


def test_multipath_not_requested():
    son = SonState()
    allm = _allm(2, timed=[11.0, 30.0], rely=[0.5, 0.9])
    for dmp, pmp in ((0.0, 10.0), (0.85, 0.0), (0.0, 0.0)):
        son.probmp[3] = F32(-1.0)
        mpath(allm, son, 3, F32(dmp), F32(pmp), F32(3000.0), 30, 0)
        assert son.probmp[3] == F32(0.001)


def test_d90s_is_a_copy_of_d10s_not_an_upper_decile():
    allm = _allm(2)
    son = SonState()
    serprb(allm, AnoisState(du=F32(9.7), dl=F32(6.0), sigm=F32(5.4),
                            sygu=F32(1.5), sygl=F32(1.5)),
           TonState(ads=F32(5.6), sus=F32(0.6), sls=F32(1.63)),
           son, 3, F32(24.0), 90)
    assert son.d90s == son.d10s and son.d10s > son.d50s
