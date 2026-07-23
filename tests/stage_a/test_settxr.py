"""Stage A SETTXR vs the tier-B oracle dump.

SETTXR prices all 45 rays at each of the two long-path control areas, and it
is the widest routine in the port: 523 output columns per case.  The /REFLX/
ray tables it reads are generated from a base and a step per column, so a
single case puts rays on both sides of every threshold in the routine and
sweeping the base moves the crossing.

Two structural things the dump exists to prove:

* **AOFX is the one /LOSX/ column SETTXR does not preset.**  TGAINX, ANDVX,
  ADVX and GRLOSX are all reset at the top of every ray; AOFX is written only
  inside the two layer branches.  The driver presets AOFX to -8888 and the
  other four to -7777, so a dead ray shows the routine's own presets next to
  AOFX's untouched sentinel.
* **the over-the-MUF fallback OVERWRITES ray 1 of /REFLX/.**  Ray 1's six
  columns and its IMODE are read back to see whether it fired.

The dump also pins the DISPATCH: settxr.for chooses between SETTXR_ORIG and
SETTXR_ALEX on the eighth character of the version string, and the driver
plants the vendored database/version.w32 value ("16.1207W"), so the oracle
runs the original -- which is what the port implements.

Measured: 298 cases x 523 columns = 155,854 cells, 99.947% bit-exact, worst 18
floored ULP (1.1e-5 absolute).  The residual is the [[f32-libm-noise]] floor
arriving through GAIN's ground-reflection branch, which is the longest
transcendental chain anywhere in the port -- GRLOSX is the weakest column at
99.77% and TGAINX, which shares no libm call with it, is bit-exact on all
26,820 cells.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.allmodestate import AllModeState
from voacap_gpu.stage_a.antennastate import CantennaState
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.layerstate import LayerState
from voacap_gpu.stage_a.losxstate import LosxState
from voacap_gpu.stage_a.mufstate import MufState
from voacap_gpu.stage_a.readant import read_antenna
from voacap_gpu.stage_a.raysstate import RaysState
from voacap_gpu.stage_a.reflxstate import ReflxState
from voacap_gpu.stage_a.settxr import ModmufError, settxr
from voacap_gpu.stage_a.signalstate import SigdState

from .dumputil import ANTENNAS_DIR, load_dump, ulp_diff_floored

DUMP = load_dump("settxr")

SETTXR_ULP_BAR = 64.0           # measured worst 18.0, in GRLOSX
SETTXR_ABS_BAR = 1.0e-3         # measured worst 1.1e-5

INTS = ["itxrcp2", "km", "modmuf", "nhopmf", "nfill1", "nfill2",
        "ifob13_1", "ifob13_2"]
RAMP = ["delfx0", "ddelfx", "hpflx0", "dhpflx", "htflx0", "dhtflx",
        "gdflx0", "dgdflx", "fvflx0", "dfvflx", "afflx0", "dafflx"]
FLOATS = (
    ["freq", "gcdkm", "amind", "htloss", "xnuz", "hnu", "afe", "bfe", "xve"]
    + ["fi1_1", "fi1_2", "gyz_1", "gyz_2", "abiy_1", "abiy_2",
       "fs2_1", "fs2_2", "hs_1", "hs_2"]
    + [f"sigpat{j}" for j in range(1, 6)]
    + [f"epspat{j}" for j in range(1, 6)]
    + [f"{a}{j}" for a in ("ymuf", "sigl", "sigu") for j in (1, 2, 3, 4)]
    + ["delmuf", "hpmuf", "htmuf", "fvmufm", "afmuf"]
    + [f"{r}_1" for r in RAMP] + [f"{r}_2" for r in RAMP]
    + ["txlo", "txhi", "rxlo", "rxhi", "pwrdba"]
)
COLUMNS = INTS + FLOATS

AOFX_SENTINEL = F32(-8888.0)
PRESET_SENTINEL = F32(-7777.0)
EFFLP_SENTINEL = F32(-5555.0)
NO_GAIN = F32(-10.0)            # SETTXR's own TGAINX preset
_ANT = None


def _cantenna(v) -> CantennaState:
    """sample.14 as both terminals, over the case's frequency ranges."""
    global _ANT
    if _ANT is None:
        _ANT = read_antenna(ANTENNAS_DIR / "sample.14").p2p_tables()
    array, aeff = _ANT
    cant = CantennaState()
    cant.numants = 2
    cant.iats[1], cant.iats[2] = 1, 2
    cant.xfqs[1], cant.xfqe[1] = F32(v["txlo"]), F32(v["txhi"])
    cant.xfqs[2], cant.xfqe[2] = F32(v["rxlo"]), F32(v["rxhi"])
    cant.offazim[1] = cant.offazim[2] = F32(0.0)
    for iat in (1, 2):
        cant.array[1:31, 1:92, iat] = array
        cant.aeff[1:31, iat] = aeff
    cant.pwrdba[1] = cant.pwrdba[2] = F32(v["pwrdba"])
    return cant


def _rampit(reflx, k, nfill, r):
    """dump_settxr.f's RAMPIT: base + step over NFILL rays, rest zeroed."""
    for ia in range(1, 46):
        if ia <= nfill:
            x = F32(ia - 1)
            reflx.delfx[ia, k] = F32(r[0]) + F32(r[1]) * x
            reflx.hpflx[ia, k] = F32(r[2]) + F32(r[3]) * x
            reflx.htflx[ia, k] = F32(r[4]) + F32(r[5]) * x
            reflx.gdflx[ia, k] = F32(r[6]) + F32(r[7]) * x
            reflx.fvflx[ia, k] = F32(r[8]) + F32(r[9]) * x
            reflx.afflx[ia, k] = F32(r[10]) + F32(r[11]) * x
            reflx.imode[ia, k] = 1 + (ia - 1) % 4
        else:
            for a in ("delfx", "hpflx", "htflx", "gdflx", "fvflx", "afflx"):
                getattr(reflx, a)[ia, k] = F32(0.0)
            reflx.imode[ia, k] = 0
    reflx.iaftxr[k] = nfill


def _build(v):
    k2 = int(v["itxrcp2"])
    state = LayerState()
    state.km = int(v["km"])
    state.kfx = k2
    for j in range(1, 6):
        state.sigpat[j] = F32(v[f"sigpat{j}"])
        state.epspat[j] = F32(v[f"epspat{j}"])
    state.fi[1, 1], state.fi[1, k2] = F32(v["fi1_1"]), F32(v["fi1_2"])
    state.gyz[1], state.gyz[k2] = F32(v["gyz_1"]), F32(v["gyz_2"])
    state.abiy[1], state.abiy[k2] = F32(v["abiy_1"]), F32(v["abiy_2"])
    state.fs[2, 1], state.fs[2, k2] = F32(v["fs2_1"]), F32(v["fs2_2"])
    state.hs[1], state.hs[k2] = F32(v["hs_1"]), F32(v["hs_2"])
    muf = MufState()
    muf.modmuf = int(v["modmuf"])
    for j in (1, 2, 3, 4):
        muf.ymuf[j] = F32(v[f"ymuf{j}"])
        muf.sigl[j] = F32(v[f"sigl{j}"])
        muf.sigu[j] = F32(v[f"sigu{j}"])
        muf.delmuf[j] = F32(v["delmuf"])
        muf.hpmuf[j] = F32(v["hpmuf"])
        muf.htmuf[j] = F32(v["htmuf"])
        muf.fvmuf[j] = F32(v["fvmufm"])
        muf.afmuf[j] = F32(v["afmuf"])
        muf.nhopmf[j] = int(v["nhopmf"])
    sigd = SigdState(htloss=F32(v["htloss"]), xnuz=F32(v["xnuz"]),
                     hnu=F32(v["hnu"]), afe=F32(v["afe"]),
                     bfe=F32(v["bfe"]), xve=F32(v["xve"]))
    rays = RaysState()
    rays.ifob[1, 3, 1] = int(v["ifob13_1"])
    rays.ifob[1, 3, k2] = int(v["ifob13_2"])
    reflx, losx = ReflxState(), LosxState()
    _rampit(reflx, 1, int(v["nfill1"]), [v[f"{r}_1"] for r in RAMP])
    _rampit(reflx, k2, int(v["nfill2"]), [v[f"{r}_2"] for r in RAMP])
    for kk in (1, 2, 3):
        for ia in range(1, 46):
            losx.aofx[ia, kk] = AOFX_SENTINEL
            for a in ("andvx", "advx", "grlosx", "tgainx"):
                getattr(losx, a)[ia, kk] = PRESET_SENTINEL
            reflx.gml[ia, kk] = F32(-6666.0)
            reflx.fhp[ia, kk] = F32(-6666.0)
    losx.efflp[1:46] = EFFLP_SENTINEL
    allm = AllModeState()
    allm.prob[1] = allm.prob[2] = F32(-4444.0)
    itxrcp = np.array([0, 1, k2], dtype=np.int32)
    return state, muf, sigd, reflx, losx, rays, allm, itxrcp, k2


def _run(v):
    state, muf, sigd, reflx, losx, rays, allm, itxrcp, k2 = _build(v)
    ray_in = reflx.hpflx.copy(), reflx.delfx.copy(), reflx.gdflx.copy()
    settxr(state, muf, sigd, reflx, losx, rays, _cantenna(v), allm, itxrcp,
           F32(v["freq"]), F32(v["gcdkm"]), F32(v["amind"]))
    ka = (1, k2)
    got = {}
    for name in ("andvx", "advx", "aofx", "grlosx", "tgainx"):
        arr = getattr(losx, name)
        for ik in (1, 2):
            for ia in range(1, 46):
                got[f"{name}{ik}_{ia:02d}"] = arr[ia, ka[ik - 1]]
    for ia in range(1, 46):
        got[f"efflp_{ia:02d}"] = losx.efflp[ia]
    for ik in (1, 2):
        k = ka[ik - 1]
        got[f"r1delfx{ik}"] = reflx.delfx[1, k]
        got[f"r1hpflx{ik}"] = reflx.hpflx[1, k]
        got[f"r1htflx{ik}"] = reflx.htflx[1, k]
        got[f"r1fvflx{ik}"] = reflx.fvflx[1, k]
        got[f"r1afflx{ik}"] = reflx.afflx[1, k]
        got[f"r1gdflx{ik}"] = reflx.gdflx[1, k]
        got[f"r1imode{ik}"] = F32(reflx.imode[1, k])
        for ia in (1, 23, 45):
            got[f"gml{ik}_{ia:02d}"] = reflx.gml[ia, k]
            got[f"fhp{ik}_{ia:02d}"] = reflx.fhp[ia, k]
    got["prob1"], got["prob2"] = allm.prob[1], allm.prob[2]
    return got, ray_in, ka, (state, muf, sigd, reflx, losx)


def _arms(v, got, ray_in, ka, seen):
    """Which arm each control area and each ray took."""
    hpflx0, delfx0, gdflx0 = ray_in
    dend = min(F32(4000.0), F32(v["gcdkm"]))
    amind = F32(v["amind"])
    for ik in (1, 2):
        k = ka[ik - 1]
        # --- the ray-existence scan -----------------------------------
        found = False
        for ia in range(1, 46):
            if hpflx0[ia, k] - F32(70.0) < 0.0:
                seen["scan_stops_at_dead_ray"] += 1
                break
            if dend / gdflx0[ia, k] - F32(0.9) < 0.0:
                seen["scan_skips_short_hop"] += 1
                continue
            if delfx0[ia, k] - amind >= 0.0:
                seen["scan_finds_a_mode"] += 1
                found = True
                break
        else:
            seen["scan_exhausts_45_rays"] += 1
        seen["fallback_fired" if not found else "fallback_skipped"] += 1
        if not found:
            assert got[f"r1hpflx{ik}"] == F32(v["hpmuf"]), "ray 1 overwritten"
        # --- the pricing loop, per ray ---------------------------------
        for ia in range(1, 46):
            if got[f"tgainx{ik}_{ia:02d}"] == NO_GAIN:
                seen["price_skipped"] += 1
                if got[f"aofx{ik}_{ia:02d}"] == AOFX_SENTINEL:
                    seen["aofx_sentinel_survives"] += 1
                continue
            seen["price_ran"] += 1
            if got[f"tgainx{ik}_{ia:02d}"] == F32(0.0):
                seen["gain_no_antenna_match"] += 1
            else:
                seen["gain_matched"] += 1
            if got[f"aofx{ik}_{ia:02d}"] == F32(0.0):
                seen["no_es_obscuration"] += 1
            else:
                seen["es_obscuration"] += 1
    seen[f"km_{int(v['km'])}"] += 1
    seen[f"modmuf_{int(v['modmuf'])}"] += 1
    seen[f"itxrcp2_{int(v['itxrcp2'])}"] += 1


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_settxr_dumps.sh")
def test_settxr_vs_oracle():
    header, inputs, rows = DUMP
    seen = dict.fromkeys(
        ["scan_stops_at_dead_ray", "scan_skips_short_hop",
         "scan_finds_a_mode", "scan_exhausts_45_rays", "fallback_fired",
         "fallback_skipped", "price_skipped", "price_ran",
         "aofx_sentinel_survives", "gain_matched", "gain_no_antenna_match",
         "es_obscuration", "no_es_obscuration", "km_1", "km_3", "km_5",
         "modmuf_1", "modmuf_2", "modmuf_3", "modmuf_4",
         "itxrcp2_2", "itxrcp2_3"], 0)
    total = exact = 0
    worst_ulp = 0.0
    worst_at = None
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        v = dict(zip(COLUMNS, (float(x) for x in inp)))
        got, ray_in, ka, _ = _run(v)
        _arms(v, got, ray_in, ka, seen)
        for name, e_str in zip(header, exp):
            g = np.float32(got[name])
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            a = abs(float(g) - float(e))
            if d > worst_ulp:
                worst_ulp, worst_at = d, f"case {k} {name}"
            assert d <= SETTXR_ULP_BAR and a <= SETTXR_ABS_BAR, (
                f"case {k} {name}: got {g!r} expected {e!r} "
                f"({a:.3e} abs, {d:.1f} floored ULP)")
            total += 1
            exact += d == 0.0
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    assert exact / total >= 0.995, (
        f"exact {exact}/{total} (worst {worst_ulp:.1f} ULP at {worst_at})")


# --- hand-built cases, independent of the dump ---------------------------

def _plain(**kw):
    v = {"itxrcp2": 3, "km": 5, "modmuf": 3, "nhopmf": 4, "nfill1": 12,
         "nfill2": 12, "ifob13_1": 4200, "ifob13_2": 3800,
         "freq": 14.0, "gcdkm": 15000.0, "amind": 3.0, "htloss": 90.0,
         "xnuz": 12.0, "hnu": 5.0, "afe": 0.7, "bfe": 1.3, "xve": 0.5,
         "fi1_1": 3.0, "fi1_2": 2.6, "gyz_1": 1.358, "gyz_2": 1.1,
         "abiy_1": 0.35, "abiy_2": 0.3, "fs2_1": 4.0, "fs2_2": 3.0,
         "hs_1": 110.0, "hs_2": 110.0, "delmuf": 6.0, "hpmuf": 320.0,
         "htmuf": 280.0, "fvmufm": 6.5, "afmuf": 1.1,
         "txlo": 2.0, "txhi": 30.0, "rxlo": 2.0, "rxhi": 30.0,
         "pwrdba": 40.0}
    for j, (s, e) in enumerate([(0.005, 15.0), (0.02, 12.0), (0.001, 20.0),
                                (0.01, 10.0), (0.003, 25.0)], 1):
        v[f"sigpat{j}"], v[f"epspat{j}"] = s, e
    for j, base in enumerate([26.0, 24.0, 22.0, 12.0], 1):
        v[f"ymuf{j}"], v[f"sigl{j}"], v[f"sigu{j}"] = base, 1.8, 1.4
    ramp = {"delfx0": 2.0, "ddelfx": 1.8, "hpflx0": 60.0, "dhpflx": 8.0,
            "htflx0": 70.0, "dhtflx": 5.0, "gdflx0": 4000.0,
            "dgdflx": -80.0, "fvflx0": 1.5, "dfvflx": 0.9,
            "afflx0": 0.8, "dafflx": 0.15}
    for key, val in ramp.items():
        v[f"{key}_1"] = v[f"{key}_2"] = val
    v.update(kw)
    return v


def test_aofx_is_the_one_column_settxr_does_not_preset():
    """TGAINX, ANDVX, ADVX and GRLOSX are reset at the top of every ray;
    AOFX is written only inside the two layer branches, so a ray that fails
    the existence tests keeps the previous frequency's obscuration."""
    got, _, _, _ = _run(_plain(nfill1=3, nfill2=3))
    # rays 4..45 are unfilled, hence dead
    for ia in (4, 20, 45):
        assert got[f"tgainx1_{ia:02d}"] == NO_GAIN
        assert got[f"andvx1_{ia:02d}"] == F32(1000.0)
        assert got[f"advx1_{ia:02d}"] == F32(1000.0)
        assert got[f"grlosx1_{ia:02d}"] == F32(0.0)
        assert got[f"aofx1_{ia:02d}"] == AOFX_SENTINEL, "untouched"


def test_the_over_the_muf_fallback_overwrites_ray_one():
    """No ray passes all three existence tests -> SETTXR plants the MUF mode
    in ray 1 of /REFLX/, clobbering whatever FINDF left there."""
    got, _, _, _ = _run(_plain(nfill1=0, nfill2=0))
    assert got["r1delfx1"] == F32(6.0) and got["r1hpflx1"] == F32(320.0)
    assert got["r1htflx1"] == F32(280.0) and got["r1fvflx1"] == F32(6.5)
    assert got["r1afflx1"] == F32(1.1)
    assert got["r1gdflx1"] == F32(15000.0) / F32(4.0)
    assert got["r1imode1"] == F32(3.0)
    # ... and with a live ray it is left alone.  Note the default ramp's
    # ray 1 is at 60 km, which is itself under the 70 km bar and so takes
    # the fallback -- the table has to start above it for the scan to run.
    kept, _, _, _ = _run(_plain(hpflx0_1=100.0, hpflx0_2=100.0))
    assert kept["r1hpflx1"] == F32(100.0), "ray 1's own virtual height"


def test_the_existence_scan_gives_up_at_the_first_dead_ray():
    """It does not skip past a ray under 70 km looking for a live one -- it
    stops there and takes the fallback, even with 44 good rays behind it."""
    got, _, _, _ = _run(_plain(hpflx0_1=10.0, dhpflx_1=40.0, nfill1=12))
    assert got["r1hpflx1"] == F32(320.0), "ray 1 at 10 km -> fallback"


def test_efflp_is_the_receiver_end_only():
    """SETTXR writes EFFlp on its JJ=2 pass alone, and indexes it by ray with
    no area subscript -- so it always holds the receiver end's efficiency."""
    got, _, _, _ = _run(_plain(rxlo=2.0, rxhi=30.0, txlo=2.0, txhi=30.0))
    assert got["efflp_01"] != EFFLP_SENTINEL
    # with no receiver antenna covering the frequency, GAIN returns 0
    missed, _, _, _ = _run(_plain(rxlo=20.0, rxhi=30.0))
    assert missed["efflp_01"] == F32(0.0)


def test_prob_is_the_same_number_for_every_ray():
    """PROB(JJ) is assigned inside the ray loop, so the last ray wins -- and
    that costs nothing, because PROS depends only on FREQ, YMUF(MODMUF) and
    MODMUF, none of which vary with the ray."""
    a, _, _, _ = _run(_plain(nfill1=1, nfill2=1))
    b, _, _, _ = _run(_plain(nfill1=12, nfill2=12))
    assert a["prob1"] == b["prob1"] and a["prob2"] == b["prob2"]


def test_gml_and_fhp_are_preset_across_all_45_rays():
    """SELTXR fills these; SETTXR only marks them unset, and it does so for
    the whole table, not just the rays it goes on to price."""
    got, _, _, _ = _run(_plain(nfill1=3, nfill2=3))
    for ik in (1, 2):
        for ia in (1, 23, 45):
            assert got[f"gml{ik}_{ia:02d}"] == F32(-999.0)
            assert got[f"fhp{ik}_{ia:02d}"] == F32(999.0)


def test_a_missing_controlling_layer_is_refused():
    """MODMUF indexes DELMUF and YMUF, both four elements; zero means CURMUF
    never ran and the Fortran reads off the end."""
    for modmuf in (0, 5, -1):
        with pytest.raises(ModmufError):
            _run(_plain(modmuf=modmuf, nfill1=0, nfill2=0))
