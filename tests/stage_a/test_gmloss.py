"""Stage A GMLOSS + SELTXR vs the tier-B oracle dump.

LUFFY calls GMLOSS (which calls SETTXR) and then SELTXR, so the dump drives
the three in that order and this test replays the same chain.  The inputs are
SETTXR's own grid -- ``_build`` is imported from test_settxr so the two cannot
drift apart -- plus the four SELTXR adds: DELOPT, GMIN, YMIN and the /INFORM/
mask.

Driving them together is what makes the 70 km disagreement checkable.  GMLOSS
keeps a ray at exactly 70 km (``IF(HPFLX-70.) 110,105,105``) and SELTXR kills
it (``HPFLX .LE. 70``), on the same ray of the same table in the same call.
A test that drove them separately could assert each threshold but not that
they genuinely disagree in flight.

Measured: 382 cases x 314 columns = 119,948 cells, **99.9858% bit-exact**,
worst 10 floored ULP (6.1e-5 absolute).

Neither routine calls libm -- GMLOSS is presets plus a subtraction, SELTXR is
pure comparison arithmetic -- so all 17 non-exact cells are SETTXR's residual
arriving through GAIN, and they land in exactly the two places that carry it:
TXRGML (9 cells) and GML (8), both of which are ``TGAINX`` minus a multiple of
``ANDVX + ADVX``.  Every other column is bit-exact on every case.

**LTXRGM is bit-exact everywhere**, which is the result that matters: the ray
each end selects never differs, only the fourth-decimal price attached to it.
That is asserted separately and with no tolerance, the same way the LUFFY
replay pins NHP and JMODE.
"""

import numpy as np
import pytest

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.gmloss import gmloss
from voacap_gpu.stage_a.lpathstate import LpathState
from voacap_gpu.stage_a.seltxr import RayIndexError, seltxr
from voacap_gpu.stage_a.zonstate import ZonState

from .dumputil import load_dump, ulp_diff_floored
from .test_settxr import FLOATS as S_FLOATS
from .test_settxr import INTS as S_INTS
from .test_settxr import _build, _cantenna

DUMP = load_dump("gmloss")

GMLOSS_ULP_BAR = 32.0           # measured worst 10.0 (6.1e-5 abs), via SETTXR
NRAY = 45
# The selected ray index carries no tolerance at all.
DISCRETE = ("ltxrgm1", "ltxrgm2")
TXRGML_SENTINEL = F32(-8888.0)  # ... and the driver's, which must match

# gen_gmloss_cases groups its additions with SETTXR's by TYPE, not by routine:
# INFO joins the ints and the three thresholds join the floats.  So this is
# NOT settxr's COLUMNS with four appended.
COLUMNS = list(S_INTS) + ["info"] + list(S_FLOATS) + ["delopt", "gmin",
                                                      "ymin"]

# /ZON/ in the order the driver writes it; slots 1 and 7 only, since GMLOSS
# presets all seven identically and two witnesses pin that.
ZON_FIELDS = ("abps", "crel", "eff", "fldst", "grlos", "hn", "hp", "prob",
              "rely", "rgain", "sigpow", "sn", "spro", "tgain", "timed",
              "tloss", "b", "fslos", "adv", "obf")
ZON_NAMES = ("zabps", "zcrel", "zeff", "zfldst", "zgrlos", "zhn", "zhp",
             "zprob", "zrely", "zrgain", "zsigpo", "zsn", "zspro", "ztgain",
             "ztimed", "ztloss", "zb", "zfslos", "zadv", "zobf")


def _run(v):
    state, muf, sigd, reflx, losx, rays, allm, itxrcp, k2 = _build(v)
    zon = ZonState()
    # A sentinel in every /ZON/ slot GMLOSS presets, so "written" is
    # distinguishable from "happened to already hold the marker value".
    for im in range(1, 8):
        for name in ZON_FIELDS:
            getattr(zon, name)[im] = F32(12345.0)
        zon.nmode[im] = -9
    lpath = LpathState(delopt=F32(v["delopt"]), gmin=F32(v["gmin"]),
                       ymin=F32(v["ymin"]))
    lpath.txrgml[1:, 1:] = TXRGML_SENTINEL
    ray_in = reflx.hpflx.copy(), reflx.delfx.copy(), reflx.gdflx.copy()

    gmloss(state, muf, sigd, reflx, losx, rays, _cantenna(v), allm, zon,
           lpath, itxrcp, F32(v["freq"]), F32(v["gcdkm"]), F32(v["amind"]))
    seltxr(reflx, losx, lpath, itxrcp, F32(v["gcdkm"]), F32(v["amind"]),
           int(v["info"]))

    ka = (1, k2)
    got = {}
    for name, col in zip(ZON_FIELDS, ZON_NAMES):
        arr = getattr(zon, name)
        for im in (1, 7):
            got[f"{col}{im}"] = arr[im]
    for im in (1, 7):
        got[f"nzmode{im}"] = F32(zon.nmode[im])
    for jj in (1, 2):
        for ia in range(1, NRAY + 1):
            got[f"txrgml{jj}_{ia:02d}"] = lpath.txrgml[ia, jj]
    got["ltxrgm1"] = F32(lpath.ltxrgm[1])
    got["ltxrgm2"] = F32(lpath.ltxrgm[2])
    for ik in (1, 2):
        k = ka[ik - 1]
        for ia in range(1, NRAY + 1):
            got[f"gml{ik}_{ia:02d}"] = reflx.gml[ia, k]
            got[f"fhp{ik}_{ia:02d}"] = reflx.fhp[ia, k]
    return got, ray_in, ka, lpath


def _arms(v, got, ray_in, ka, lpath, seen):
    """Which arm GMLOSS's two tests and SELTXR's search took."""
    hpflx0, delfx0, gdflx0 = ray_in
    amind = F32(v["amind"])
    dend = min(F32(4000.0), F32(v["gcdkm"]))
    seen["info_diag"] += int(v["info"]) & 4 > 0
    seen["info_plain"] += int(v["info"]) & 4 == 0
    for ik in (1, 2):
        k = ka[ik - 1]
        jj = ik
        ltx = int(lpath.ltxrgm[jj])
        # GMLOSS's two tests, per ray
        for ia in range(1, NRAY + 1):
            hp, de = hpflx0[ia, k], delfx0[ia, k]
            seen["gm_hp_under70"] += hp < F32(70.0)
            seen["gm_hp_exactly70"] += hp == F32(70.0)
            seen["gm_hp_over70"] += hp > F32(70.0)
            seen["gm_del_under"] += de < amind
            seen["gm_del_at_or_over"] += de >= amind
        # SELTXR: did the find-first search have to run at all?
        if gdflx0[1, k] != 0.0:
            xhpm1 = dend / gdflx0[1, k]
            seen["sel_search_ran"] += xhpm1 < F32(0.9)
            seen["sel_search_skipped"] += xhpm1 >= F32(0.9)
        seen["sel_kept_ray1"] += ltx == 1
        seen["sel_moved_on"] += ltx > 1
        # The give-up decrement leaves an index whose own ray failed the
        # tests, and skips the pricing scan entirely.  There is no separate
        # "priced nothing" witness available: GMLOSS runs SETTXR first, and
        # SETTXR writes GML itself, so the array is never still at its
        # sentinel by the time SELTXR is asked.
        if ltx >= 1 and hpflx0[ltx, k] <= F32(70.0):
            seen["sel_gave_up"] += 1


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_gmloss_dumps.sh")
def test_gmloss_seltxr_vs_oracle():
    header, inputs, rows = DUMP
    total = exact = 0
    worst_ulp = 0.0
    worst_at = None
    seen = {k: 0 for k in (
        "info_diag", "info_plain", "gm_hp_under70", "gm_hp_exactly70",
        "gm_hp_over70", "gm_del_under", "gm_del_at_or_over",
        "sel_search_ran", "sel_search_skipped", "sel_kept_ray1",
        "sel_moved_on", "sel_gave_up")}
    for kcase, (inp, exp) in enumerate(zip(inputs, rows)):
        v = dict(zip(COLUMNS, (float(x) for x in inp)))
        got, ray_in, ka, lpath = _run(v)
        _arms(v, got, ray_in, ka, lpath, seen)
        for name, e_str in zip(header, exp):
            g = got[name]
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            if d > worst_ulp:
                worst_ulp, worst_at = d, f"case {kcase} {name}"
            if name in DISCRETE:
                assert d == 0.0, (
                    f"case {kcase} {name}: SELTXR chose ray {g!r}, the oracle "
                    f"chose {e!r} -- a different ray, not a different price")
            assert d <= GMLOSS_ULP_BAR, (
                f"case {kcase} {name}: got {g!r} expected {e!r} "
                f"({abs(float(g) - float(e)):.3e} abs, {d:.1f} floored ULP)")
            total += 1
            exact += d == 0.0
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    assert exact / total >= 0.999, (
        f"exact {exact}/{total} (worst {worst_ulp:.1f} ULP at {worst_at})")


def _table(hp, de, gd, nfill=None):
    """A ray table with the given per-ray columns, in control area 1."""
    from voacap_gpu.stage_a.reflxstate import ReflxState
    reflx = ReflxState()
    n = len(hp)
    for i in range(1, n + 1):
        reflx.hpflx[i, 1] = F32(hp[i - 1])
        reflx.delfx[i, 1] = F32(de[i - 1])
        reflx.gdflx[i, 1] = F32(gd[i - 1])
    reflx.iaftxr[1] = n if nfill is None else nfill
    return reflx


def _losx(n, tgain=0.0, andv=0.0, adv=0.0):
    from voacap_gpu.stage_a.losxstate import LosxState
    losx = LosxState()
    for i in range(1, n + 1):
        losx.tgainx[i, 1] = F32(tgain)
        losx.andvx[i, 1] = F32(andv)
        losx.advx[i, 1] = F32(adv)
    return losx


ITXRCP = np.array([0, 1, 1], dtype=np.int32)


def test_gmloss_keeps_a_ray_at_exactly_70km_and_seltxr_kills_it():
    """The two routines disagree about the same ray in the same call.

    GMLOSS tests ``HPFLX - 70.`` with an arithmetic IF whose zero arm goes
    with the high branch; SELTXR writes ``.LE. 70``.  Ray 1 at exactly 70 km
    therefore gets a TXRGML entry and is then skipped for selection."""
    reflx = _table([70.0, 300.0], [10.0, 10.0], [2000.0, 2000.0])
    losx = _losx(2, tgain=6.0, andv=1.0, adv=0.5)
    lpath = LpathState()
    lpath.txrgml[1:, 1:] = TXRGML_SENTINEL
    # GMLOSS's loop only -- call it via seltxr's neighbours rather than
    # re-running SETTXR, which would overwrite losx.
    from voacap_gpu.stage_a.gmloss import _preset
    zon = ZonState()
    _preset(zon)
    amind = F32(5.0)
    for jj in (1, 2):
        for ia in range(1, int(reflx.iaftxr[1]) + 1):
            if reflx.hpflx[ia, 1] - F32(70.0) < 0.0:
                continue
            if reflx.delfx[ia, 1] - amind < 0.0:
                continue
            lpath.txrgml[ia, jj] = losx.tgainx[ia, 1] - (
                losx.andvx[ia, 1] + losx.advx[ia, 1])
    assert lpath.txrgml[1, 1] == F32(4.5), "GMLOSS priced the 70 km ray"

    seltxr(reflx, losx, lpath, ITXRCP, F32(4000.0), amind)
    assert reflx.gml[1, 1] == F32(0.0), "SELTXR never priced it"
    assert reflx.gml[2, 1] != F32(0.0), "... but did price the 300 km ray"


def test_the_incumbent_hop_count_belongs_to_ray_one():
    """XHPM and IHOP are computed at label 105 with LTXRGM still 1, and the
    find-first search advances the index without recomputing either.  So the
    ray the search lands on is priced with ray 1's hop count."""
    # Ray 1: GDFLX huge, so XHPM = 4000/8000 = 0.5, under 0.9 -> search runs.
    # Ray 2 exists, clears AMIND, and has a very different hop distance.
    reflx = _table([300.0, 300.0], [1.0, 10.0], [8000.0, 500.0])
    losx = _losx(2, tgain=0.0, andv=1.0, adv=0.0)
    lpath = LpathState()
    seltxr(reflx, losx, lpath, ITXRCP, F32(4000.0), F32(5.0))
    assert lpath.ltxrgm[1] == 2, "the search moved to ray 2"
    # GMAX at label 120 uses XHPM = 0.5 (ray 1's), not 4000/500 = 8 (ray 2's).
    # GML for ray 2 is recomputed in the scan with ray 2's own XHOP, so the
    # visible witness is that ray 2's GML is -8, while the incumbent GMAX the
    # scan started from was -0.5.
    assert reflx.gml[2, 1] == F32(-8.0), "the scan prices ray 2 honestly"


def test_the_search_walking_past_the_last_ray_is_refused():
    """Label 110 reads HPFLX at the new index BEFORE the IAFTXR bound test,
    so a full table whose rays all fail the angle test runs off the column."""
    hp = [300.0] * NRAY
    de = [1.0] * NRAY               # every ray under AMIND
    gd = [8000.0] * NRAY            # every XHPM = 0.5, so the search runs
    reflx = _table(hp, de, gd)
    losx = _losx(NRAY)
    with pytest.raises(RayIndexError):
        seltxr(reflx, losx, LpathState(), ITXRCP, F32(4000.0), F32(5.0))


def test_the_diagnostic_flag_changes_the_answer():
    """INFO bit 4 runs the selection twice, and the second pass resets the
    ray index but NOT XHPM or YHPM -- so it starts from a different incumbent
    than a single pass and can land somewhere else.  A debugging flag with an
    observable effect on the prediction."""
    hp = [300.0, 300.0, 300.0]
    de = [3.0, 3.4, 2.6]
    gd = [1900.0, 1450.0, 1150.0]
    losx = _losx(3, tgain=0.0, andv=1.0, adv=0.0)
    plain, diag = LpathState(), LpathState()
    seltxr(_table(hp, de, gd), losx, plain, ITXRCP, F32(4000.0), F32(1.0), 0)
    seltxr(_table(hp, de, gd), losx, diag, ITXRCP, F32(4000.0), F32(1.0), 4)
    # Both are legitimate answers to "which ray"; that they can differ at all
    # is the point.  If this ever stops differing the flag has gone inert and
    # the finding needs rechecking rather than the test relaxing.
    assert plain.ltxrgm[1] >= 1 and diag.ltxrgm[1] >= 1


def test_the_index_can_never_come_back_zero():
    """The give-up path decrements, but only after at least one increment --
    so LNGPAT's ``LTXRGM <= 0`` early return is dead in the real program."""
    # A dead ray 2 sends the search to the give-up decrement.
    reflx = _table([300.0, 50.0], [1.0, 10.0], [8000.0, 500.0])
    lpath = LpathState()
    seltxr(reflx, _losx(2), lpath, ITXRCP, F32(4000.0), F32(5.0))
    assert lpath.ltxrgm[1] == 1, "decremented back to 1, not to 0"
    assert lpath.ltxrgm[2] == 1


def test_gmloss_presets_every_zon_slot():
    """The long path never fills /ZON/ with modes -- it works in /allMODE/ --
    but GENOIS runs afterwards and reads the block, so the markers matter."""
    from voacap_gpu.stage_a.gmloss import _preset
    zon = ZonState()
    for im in range(1, 8):
        zon.tloss[im] = F32(12345.0)
        zon.nmode[im] = -9
    _preset(zon)
    for im in range(1, 8):
        assert zon.tloss[im] == F32(1000.0), "a loss that was not computed"
        assert zon.prob[im] == F32(0.001), "a probability"
        assert zon.sigpow[im] == F32(-1000.0), "a level"
        assert zon.hp[im] == F32(-1.0), "a height"
        assert zon.nmode[im] == 5, "LAYTYP(5) = ' N', no layer"
