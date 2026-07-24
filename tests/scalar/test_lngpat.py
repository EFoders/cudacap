"""Stage A LNGPAT vs the tier-B oracle dump.

LNGPAT is where the long path's ONE mode is built -- SELTXR's chosen ray at
each end, averaged into a single geometry and priced end to end.  It calls
CONVH, GETTOP, TABS, BABS, PRBMUF and PWRDB, and the dump links all six real
rather than stubbing them, so this pins the composition as well as the
arithmetic.

Two things the dump exists to prove and not merely assume:

* the **LTXRGM <= 0 early return writes nothing**.  Every output is preset to
  a distinctive sentinel (-8888, -4444, -3333, -2222) rather than zero, so
  "untouched" and "written zero" are different observations.
* **slot 2 of /allMODE/ is not a second mode**.  NMMOD comes back 1, but B(2)
  and NMODE(2) hold the reception angle and receiving layer of the SAME mode,
  and PROB(1)/PROB(2) arrive holding the two ends' MUFdays before the last
  line collapses them.

Measured: 478 cases x 25 columns = 11,950 cells, 99.90% bit-exact, worst 138
floored ULP (8.4e-3 absolute, on a transmission loss of 1020 dB).  Twelve
cells in two cases carry it, and both are the [[f32-libm-noise]] floor with
this routine's own conditioning on top:

* TLOSS and its three dependents inherit a 1-ULP RELATIVE difference in
  BABS's AF, multiplied back out by DF*RZ -- a 1e-7 relative error on a leg
  loss of 140,000 dB is 8e-3 absolute.
* TIMED is PTOT/VOFL, and PTOT's RAY term is RZ*COS(PHE+DEL)/SPHE, where
  PHE+DEL approaches a right angle by construction (PHE is always less than
  PIO2-DEL), so the cosine is a near-total cancellation.

Eighteen of the twenty-five columns are bit-exact on every case.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.allmodestate import AllModeState
from voacap_gpu.scalar.antennastate import CantennaState
from voacap_gpu.scalar.constants import D2R, RZ
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.gettop import gettop
from voacap_gpu.scalar.layerstate import LayerState
from voacap_gpu.scalar.lngpat import ModeIndexError, lngpat
from voacap_gpu.scalar.losxstate import LosxState
from voacap_gpu.scalar.lpathstate import LpathState
from voacap_gpu.scalar.mufstate import MufState
from voacap_gpu.scalar.noisestate import AnoisState
from voacap_gpu.scalar.reflxstate import ReflxState
from voacap_gpu.scalar.signalstate import SigdState

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("lngpat")

LNGPAT_ULP_BAR = 512.0          # measured worst 138.0, on TLOSS and kin
LNGPAT_ABS_BAR = 0.05           # measured worst 8.4e-3

INTS = ["l1", "l2", "itxrcp2", "imode1", "imode2", "numants"]
FLOATS = (
    ["freq", "gcd"]
    + ["delfx1", "delfx2", "hpflx1", "hpflx2", "gdflx1", "gdflx2"]
    + ["andvx1", "andvx2", "advx1", "advx2", "aofx1", "aofx2",
       "grlosx1", "grlosx2", "tgainx1", "tgainx2"]
    + ["efflp", "asm", "dsl", "dsu", "rcnse", "acav", "gyz3", "pwr",
       "prob1", "prob2", "delopt"]
    + ["hi1", "hi2", "hi3", "fi1", "fi2", "fi3"]
    + [f"{a}{j}" for a in ("ymuf", "yfot", "yhpf", "fvmuf", "sigl", "sigu")
       for j in (1, 2, 3, 4)]
)
COLUMNS = INTS + FLOATS
SENTINEL = F32(-8888.0)
PRESET = ("grlos", "hn", "hp", "rgain", "tgain", "eff", "timed", "tloss",
          "fslos", "adv", "abps", "obf", "sigpow", "fldst", "sn", "tllow",
          "tlhgh")
OUT = ["nmmod", "grlos1", "hn1", "hp1", "rgain1", "tgain1", "eff1", "timed1",
       "tloss1", "b1", "b2", "nmode1", "nmode2", "fslos1", "adv1", "abps1",
       "obf1", "sigpow1", "fldst1", "sn1", "tllow1", "tlhgh1", "prob1",
       "dslf", "dsuf"]


def _build(v):
    """The state blocks LNGPAT reads, exactly as dump_lngpat.f injects them."""
    state = LayerState()
    state.gyz[3] = F32(v["gyz3"])
    for j in (1, 2, 3):
        state.hi[1, j] = F32(v[f"hi{j}"])
        state.fi[1, j] = F32(v[f"fi{j}"])
    lpath = LpathState(delopt=F32(v["delopt"]))
    lpath.ltxrgm[1] = int(v["l1"])
    lpath.ltxrgm[2] = int(v["l2"])
    losx, reflx, muf = LosxState(), ReflxState(), MufState()
    l1, l2, k2 = int(v["l1"]), int(v["l2"]), int(v["itxrcp2"])
    if 1 <= l1 <= 45:
        reflx.delfx[l1, 1] = F32(v["delfx1"])
        reflx.hpflx[l1, 1] = F32(v["hpflx1"])
        reflx.gdflx[l1, 1] = F32(v["gdflx1"])
        reflx.imode[l1, 1] = int(v["imode1"])
        losx.andvx[l1, 1] = F32(v["andvx1"])
        losx.advx[l1, 1] = F32(v["advx1"])
        losx.aofx[l1, 1] = F32(v["aofx1"])
        losx.grlosx[l1, 1] = F32(v["grlosx1"])
        losx.tgainx[l1, 1] = F32(v["tgainx1"])
    if 1 <= l2 <= 45:
        reflx.delfx[l2, k2] = F32(v["delfx2"])
        reflx.hpflx[l2, k2] = F32(v["hpflx2"])
        reflx.gdflx[l2, k2] = F32(v["gdflx2"])
        reflx.imode[l2, k2] = int(v["imode2"])
        losx.andvx[l2, k2] = F32(v["andvx2"])
        losx.advx[l2, k2] = F32(v["advx2"])
        losx.aofx[l2, k2] = F32(v["aofx2"])
        losx.grlosx[l2, k2] = F32(v["grlosx2"])
        losx.tgainx[l2, k2] = F32(v["tgainx2"])
        losx.efflp[l2] = F32(v["efflp"])
    for j in (1, 2, 3, 4):
        muf.ymuf[j] = F32(v[f"ymuf{j}"])
        muf.yfot[j] = F32(v[f"yfot{j}"])
        muf.yhpf[j] = F32(v[f"yhpf{j}"])
        muf.fvmuf[j] = F32(v[f"fvmuf{j}"])
        muf.sigl[j] = F32(v[f"sigl{j}"])
        muf.sigu[j] = F32(v[f"sigu{j}"])
    sigd = SigdState(asm=F32(v["asm"]), dsl=F32(v["dsl"]), dsu=F32(v["dsu"]),
                     acav=F32(v["acav"]), dslf=F32(-3333.0),
                     dsuf=F32(-2222.0))
    anois = AnoisState(rcnse=F32(v["rcnse"]))
    cant = CantennaState()
    cant.numants = int(v["numants"])
    cant.iats[1] = 1
    cant.xfqs[1] = F32(0.0)
    cant.xfqe[1] = F32(100.0)
    cant.pwrdba[1] = F32(v["pwr"])
    allm = AllModeState()
    allm.nmmod = -4444
    allm.prob[1] = F32(v["prob1"])
    allm.prob[2] = F32(v["prob2"])
    for a in PRESET:
        getattr(allm, a)[1] = SENTINEL
    allm.b[1] = allm.b[2] = SENTINEL
    allm.nmode[1] = allm.nmode[2] = -8888
    return state, lpath, losx, reflx, muf, allm, sigd, anois, cant, k2


def _run(v):
    (state, lpath, losx, reflx, muf, allm, sigd, anois,
     cant, k2) = _build(v)
    lngpat(state, lpath, losx, reflx, muf, allm, sigd, anois, cant,
           F32(v["freq"]), F32(v["gcd"]), k2)
    got = {"nmmod": F32(allm.nmmod), "b1": allm.b[1], "b2": allm.b[2],
           "nmode1": F32(allm.nmode[1]), "nmode2": F32(allm.nmode[2]),
           "prob1": allm.prob[1], "dslf": sigd.dslf, "dsuf": sigd.dsuf}
    for a in PRESET:
        got[f"{a}1"] = getattr(allm, a)[1]
    return got, (state, lpath, losx, reflx, muf, allm, sigd)


def _arms(v, got, seen):
    l1, l2 = int(v["l1"]), int(v["l2"])
    if l1 <= 0 or l2 <= 0:
        seen["early_return"] += 1
        return
    seen["ran"] += 1
    state = LayerState()
    for j in (1, 2, 3):
        state.hi[1, j] = F32(v[f"hi{j}"])
        state.fi[1, j] = F32(v[f"fi{j}"])
    lpath = LpathState(delopt=F32(v["delopt"]))
    hpx = F32(0.5) * (F32(v["hpflx1"]) + F32(v["hpflx2"]))
    dt = F32(0.5) * F32(v["gdflx1"]) / RZ
    dr = F32(0.5) * F32(v["gdflx2"]) / RZ
    di = F32(v["gcd"]) - dt - dr
    dikm = di * RZ
    seen["hpx_is_70"] += hpx == F32(70.0)
    if dikm <= F32(-1.0):
        seen["dikm_at_or_below_-1"] += 1
        dm = df = F32(0.0)
    else:
        seen["dikm_above_-1"] += 1
        gmkm = gettop(state, lpath, F32(v["freq"]), dikm)
        seen["over_the_top" if gmkm > 0 else "no_over_the_top"] += 1
        dm = gmkm / RZ
        raw = di - dm
        seen["df_floored"] += raw < F32(0.0)
        df = max(raw, F32(0.0))
    seen["babs_dead" if (df <= 0 or hpx < F32(70.0)) else "babs_live"] += 1
    seen["tabs_dead" if (dm <= 0 or hpx <= F32(70.0)) else "tabs_live"] += 1
    fhop = df / (dr + dt)
    seen["fhop_floored" if fhop < F32(1.0) else "fhop_free"] += 1
    hn = F32(0.5) * F32(v["gcd"]) / (dr + dt)
    seen["hn_floored" if hn < F32(1.0) else "hn_free"] += 1
    seen["tllow_clamped" if got["tllow1"] == F32(25.0) else "tllow_free"] += 1
    seen["tlhgh_clamped" if got["tlhgh1"] == F32(25.0) else "tlhgh_free"] += 1
    seen["layers_same" if int(v["imode1"]) == int(v["imode2"])
         else "layers_differ"] += 1
    seen["pwrdb_found" if int(v["numants"]) >= 1 else "pwrdb_fallback"] += 1
    p1, p2 = F32(v["prob1"]), F32(v["prob2"])
    seen["prob_tx_wins" if p1 < p2 else "prob_rx_wins" if p2 < p1
         else "prob_tie"] += 1


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_lngpat_dumps.sh")
def test_lngpat_vs_oracle():
    header, inputs, rows = DUMP
    seen = dict.fromkeys(
        ["early_return", "ran", "hpx_is_70", "dikm_at_or_below_-1",
         "dikm_above_-1", "over_the_top", "no_over_the_top", "df_floored",
         "babs_dead", "babs_live", "tabs_dead", "tabs_live", "fhop_floored",
         "fhop_free", "hn_floored", "hn_free", "tllow_clamped", "tllow_free",
         "tlhgh_clamped", "tlhgh_free", "layers_same", "layers_differ",
         "pwrdb_found", "pwrdb_fallback", "prob_tx_wins", "prob_rx_wins",
         "prob_tie"], 0)
    total = exact = 0
    worst_ulp = 0.0
    worst_at = None
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        v = dict(zip(COLUMNS, (float(x) for x in inp)))
        got, _ = _run(v)
        _arms(v, got, seen)
        for name, e_str in zip(header, exp):
            g = np.float32(got[name])
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            a = abs(float(g) - float(e))
            if d > worst_ulp:
                worst_ulp, worst_at = d, f"case {k} {name}"
            assert d <= LNGPAT_ULP_BAR and a <= LNGPAT_ABS_BAR, (
                f"case {k} {name}: got {g!r} expected {e!r} "
                f"({a:.3e} abs, {d:.1f} floored ULP)")
            total += 1
            exact += d == 0.0
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    assert exact / total >= 0.99, (
        f"exact {exact}/{total} (worst {worst_ulp:.1f} ULP at {worst_at})")


# --- hand-built cases, independent of the dump ---------------------------

def _plain(**kw):
    """A workable long path: one ray at each end, 15000 km, F2 both ends.

    FREQ sits ON layer 3's MUF and the decile spreads are wide, because
    otherwise every PRBMUF in the routine saturates and the whole decile
    calculation collapses to zero -- SIGL/SIGU are absolute MHz here (LNGPAT
    passes FGO == FSET, so PRBMUF's ``FGO*SIGL/FSET`` is just SIGL), and a
    tenth of a megahertz puts a 14 MHz signal 80 sigma from a 22 MHz MUF.
    """
    v = {
        "l1": 4, "l2": 7, "itxrcp2": 3, "imode1": 3, "imode2": 3,
        "numants": 1, "freq": 22.0, "gcd": 2.4, "delfx1": 8.0,
        "delfx2": 11.0, "hpflx1": 280.0, "hpflx2": 320.0, "gdflx1": 3200.0,
        "gdflx2": 3400.0, "andvx1": 22.0, "andvx2": 26.0, "advx1": 4.0,
        "advx2": 5.0, "aofx1": 0.0, "aofx2": 1.5, "grlosx1": 3.0,
        "grlosx2": 4.0, "tgainx1": 9.0, "tgainx2": 7.0, "efflp": 0.8,
        "asm": 1.5, "dsl": 4.0, "dsu": 5.0, "rcnse": -145.0, "acav": 0.4,
        "gyz3": 1.358, "pwr": 40.0, "prob1": 0.8, "prob2": 0.6,
        "delopt": 3.0, "hi1": 110.0, "hi2": 110.0, "hi3": 110.0,
        "fi1": 0.6, "fi2": 0.8, "fi3": 0.6,
    }
    for j, base in enumerate([26.0, 24.0, 22.0, 12.0], 1):
        v[f"ymuf{j}"] = base
        v[f"yfot{j}"] = base * 0.85
        v[f"yhpf{j}"] = base * 1.15
        v[f"fvmuf{j}"] = base * 0.35
        v[f"sigl{j}"] = 2.0
        v[f"sigu{j}"] = 1.5
    v.update(kw)
    return v


def test_the_early_return_writes_nothing():
    """LTXRGM <= 0 at either end means SELTXR never chose a ray.  Every
    output keeps its sentinel -- including NMMOD, which is NOT set to 1."""
    for kw in ({"l1": 0}, {"l2": 0}, {"l1": -1, "l2": -1}):
        got, _ = _run(_plain(**kw))
        assert got["nmmod"] == F32(-4444)
        assert got["dslf"] == F32(-3333) and got["dsuf"] == F32(-2222)
        assert got["prob1"] == F32(0.8), "not even the AMIN1 on the last line"
        for name in OUT:
            if name not in ("nmmod", "prob1", "dslf", "dsuf"):
                assert got[name] == SENTINEL, name


def test_slot_2_carries_the_same_mode_not_a_second_one():
    """NMMOD is 1, yet B(2) and NMODE(2) are written: they are the RECEPTION
    angle and receiving layer of the one mode, not a second mode's."""
    got, _ = _run(_plain(delfx1=8.0, delfx2=11.0, imode1=3, imode2=1))
    assert got["nmmod"] == F32(1.0)
    assert got["b1"] == F32(8.0) and got["b2"] == F32(11.0)
    assert got["nmode1"] == F32(3.0) and got["nmode2"] == F32(1.0)


def test_prob_collapses_the_two_ends_with_amin1():
    """SETTXR leaves the transmitter end's MUFday in PROB(1) and the
    receiver's in PROB(2); the last line of LNGPAT takes the smaller."""
    assert _run(_plain(prob1=0.8, prob2=0.6))[0]["prob1"] == F32(0.6)
    assert _run(_plain(prob1=0.2, prob2=0.9))[0]["prob1"] == F32(0.2)


def test_lngpat_clobbers_the_sigd_deciles():
    """DSLF and DSUF are /SIGD/ path parameters used here as scratch."""
    got, _ = _run(_plain())
    assert got["dslf"] != F32(-3333) and got["dsuf"] != F32(-2222)


def test_one_layer_at_both_ends_skips_the_averaging_pass():
    """With NMODE(1) == NMODE(2) the second PRBMUF pass is skipped outright,
    so DSLF is that layer's spread rather than an average of two."""
    same = _run(_plain(imode1=3, imode2=3))[0]
    solo = _run(_plain(imode1=3, imode2=3, ymuf1=99.0, yfot1=88.0))[0]
    assert same["dslf"] == solo["dslf"], "layer 1's numbers cannot leak in"
    mixed = _run(_plain(imode1=3, imode2=1))[0]
    assert mixed["dslf"] != same["dslf"]
    # ... and the mix really is the mean of the two ends taken separately
    one = _run(_plain(imode1=1, imode2=1))[0]
    assert abs(float(mixed["dslf"])
               - 0.5 * (float(same["dslf"]) + float(one["dslf"]))) < 1e-5


def test_a_ray_with_no_layer_is_refused():
    """NMODE outside 1..4 indexes YMUF off the end of a four-element array.
    A ray SELTXR chose but FINDF never filled leaves IMODE at zero."""
    for kw in ({"imode1": 0}, {"imode2": 0}, {"imode1": 5}):
        with pytest.raises(ModeIndexError):
            _run(_plain(**kw))


def test_a_vanishing_ionospheric_distance_costs_nothing():
    """Two ways to reach the same zero: DIKM at or below -1 km sets both
    losses to zero outright, while DIKM just above it gets there the long
    way -- GETTOP refuses, DM is zero, DF is floored to zero, and BABS and
    TABS both return their 1000 dB/km marker into a leg of no length."""
    base = 0.5 * (3200.0 + 3400.0) / 6370.0
    short = _run(_plain(gcd=base - 4000.0 / 6370.0))[0]
    edge = _run(_plain(gcd=base - 0.5 / 6370.0))[0]
    # the free-space loss differs (different GCD), but neither carries a
    # 1000 dB/km leg into the transmission loss
    for got in (short, edge):
        assert float(got["tloss1"]) - float(got["fslos1"]) < 100.0
