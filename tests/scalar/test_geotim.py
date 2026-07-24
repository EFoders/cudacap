"""Stage A GEOTIM + CNGTIM vs the tier-B oracle dump.

CNGTIM is driven twice: directly, so its day-change return and its in-place
rewrite of TIME are both visible, and through GEOTIM, which calls it up to
seven times and throws every day-change away.

Measured: 431 cases x 12 columns = 5,172 cells, ALL BIT-EXACT.
"""

import numpy as np
import pytest

from voacap_gpu.scalar.circuit import Circuit
from voacap_gpu.scalar.cngtim import cngtim
from voacap_gpu.scalar.constants import D2R, PI, R2D
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.geotim import geotim

from .dumputil import load_dump, ulp_diff_floored

DUMP = load_dump("geotim")

GEOTIM_ULP_BAR = 4.0            # measured worst 0.0 -- a libm safety valve
NC = 5
SENTINEL = F32(-8888.0)

COLUMNS = (["jt", "itim", "tlong", "rlong", "km"]
           + [f"clong{i}" for i in range(1, NC + 1)]
           + ["ctime", "ctheta", "cisw"])


def _run(inp):
    v = dict(zip(COLUMNS, inp))
    c = Circuit()
    c.time.itim = int(float(v["itim"]))
    c.tlong = F32(v["tlong"])
    c.rlong = F32(v["rlong"])
    c.layers.km = int(float(v["km"]))
    for i in range(1, NC + 1):
        c.layers.clong[i] = F32(v[f"clong{i}"])
        c.layers.clck[i] = SENTINEL
    c.time.it = -9
    c.time.gmt = SENTINEL
    c.time.gmtr = SENTINEL
    c.time.utime[1:] = SENTINEL
    c.time.xlmt[1:] = SENTINEL

    ctime, cday = cngtim(F32(v["ctime"]), F32(v["ctheta"]),
                         int(float(v["cisw"])))
    geotim(c.layers, c.time, int(float(v["jt"])), c.tlong, c.rlong)

    iut = c.time.it if 1 <= c.time.it <= 24 else 24
    got = {"ctime": ctime, "cday": F32(cday), "it": F32(c.time.it),
           "gmt": c.time.gmt, "gmtr": c.time.gmtr,
           "utime": c.time.utime[iut], "xlmt": c.time.xlmt[iut]}
    for i in range(1, NC + 1):
        got[f"clck{i}"] = c.layers.clck[i]
    return got, c, v


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_geotim_dumps.sh")
def test_geotim_cngtim_vs_oracle():
    header, inputs, rows = DUMP
    total = exact = 0
    worst_ulp = 0.0
    worst_at = None
    seen = {"theta_zero": 0, "theta_negative": 0, "theta_under_180": 0,
            "theta_over_180": 0, "theta_full_turn": 0, "isw_forward": 0,
            "isw_back": 0, "day_advance": 0, "day_retard": 0, "day_same": 0,
            "itim_local": 0, "itim_universal": 0, "it_rewritten_to_24": 0,
            "no_samples": 0, "all_samples": 0}
    for k, (inp, exp) in enumerate(zip(inputs, rows)):
        got, c, v = _run(inp)
        theta = float(v["ctheta"])
        folded = 360.0 + theta if theta < 0.0 else theta
        seen["theta_zero"] += theta == 0.0
        seen["theta_negative"] += theta < 0.0
        seen["theta_under_180"] += 0.0 < folded < 180.0
        seen["theta_over_180"] += 180.0 <= folded < 360.0
        seen["theta_full_turn"] += folded >= 360.0
        seen["isw_forward"] += int(float(v["cisw"])) > 0
        seen["isw_back"] += int(float(v["cisw"])) < 0
        seen["day_advance"] += float(got["cday"]) > 0
        seen["day_retard"] += float(got["cday"]) < 0
        seen["day_same"] += float(got["cday"]) == 0
        seen["itim_local"] += c.time.itim < 0
        seen["itim_universal"] += c.time.itim >= 0
        seen["it_rewritten_to_24"] += (c.time.it == 24
                                       and float(c.time.gmt) < 24.0)
        seen["no_samples"] += c.layers.km == 0
        seen["all_samples"] += c.layers.km == NC

        for name, e_str in zip(header, exp):
            g = got[name]
            e = np.float32(e_str)
            d = ulp_diff_floored(g, e)
            if d > worst_ulp:
                worst_ulp, worst_at = d, f"case {k} {name}"
            assert d <= GEOTIM_ULP_BAR, (
                f"case {k} {name}: got {g!r} expected {e!r} "
                f"({abs(float(g) - float(e)):.3e} abs, {d:.1f} floored ULP)")
            total += 1
            exact += d == 0.0
    for arm, n in seen.items():
        assert n, f"branch never reached: {arm} ({seen})"
    assert exact / total >= 0.999, (
        f"exact {exact}/{total} (worst {worst_ulp:.1f} ULP at {worst_at})")


def test_greenwich_returns_the_time_untouched_and_unnormalised():
    """The XTHETA == 0 arm leaves before the wrap, so it also skips the range
    fix every other longitude gets."""
    assert cngtim(F32(30.0), F32(0.0), 1) == (F32(30.0), 0)
    assert cngtim(F32(-5.0), F32(0.0), -1) == (F32(-5.0), 0)
    # Any other longitude, however small, normalises.
    t, _ = cngtim(F32(30.0), F32(0.001), 1)
    assert 0.0 <= float(t) <= 24.0


def test_the_conversion_round_trips():
    for theta in (15.0, 90.0, 179.0, 181.0, 270.0, 359.0, -75.0):
        there, _ = cngtim(F32(9.0), F32(theta), 1)
        back, _ = cngtim(there, F32(theta), -1)
        assert float(back) == pytest.approx(9.0, abs=1e-4), theta


def test_east_and_west_of_the_half_turn_disagree_by_a_day():
    """THETA past 180 switches to the -24 formula, so the same physical
    meridian read as 190 degrees east gives the same clock as -170."""
    east, _ = cngtim(F32(12.0), F32(190.0), 1)
    west, _ = cngtim(F32(12.0), F32(-170.0), 1)
    assert east == west


def test_the_day_change_is_reported_but_geotim_drops_it():
    _, advance = cngtim(F32(23.0), F32(90.0), 1)
    assert advance == 1
    _, retard = cngtim(F32(1.0), F32(270.0), 1)
    assert retard == -1
    # GEOTIM keeps no trace of either: /TIME/ has no date field to put it in.
    c = Circuit()
    c.layers.km = 0
    geotim(c.layers, c.time, 23, F32(90.0) * D2R, F32(90.0) * D2R)
    assert not hasattr(c.time, "day")


def test_midnight_is_indexed_as_hour_24():
    """An hour that truncates to 0 belongs to the day that is ending.

    Only reachable when the user gave LOCAL time: on the ITIM >= 0 arm GMT is
    assigned the requested hour before any conversion touches it, so IT is
    just that hour and the rewrite would need JT itself to be 0."""
    c = Circuit()
    c.layers.km = 0
    c.time.itim = -1                      # LMT in, so GMT is derived
    geotim(c.layers, c.time, 12, F32(179.0) * D2R, F32(0.0))
    assert 0.0 < float(c.time.gmt) < 1.0
    assert c.time.it == 24 and c.time.utime[24] == c.time.gmt
    assert c.time.xlmt[24] == F32(12.0), "the hour the user asked for"

    c = Circuit()
    c.layers.km = 0
    geotim(c.layers, c.time, 12, F32(179.0) * D2R, F32(0.0))
    assert c.time.it == 12, "the universal arm never reaches the rewrite"


def test_no_sample_areas_leaves_the_clock_array_alone():
    c = Circuit()
    c.layers.km = 0
    c.layers.clck[1:] = F32(-1.0)
    geotim(c.layers, c.time, 12, F32(45.0) * D2R, F32(45.0) * D2R)
    assert all(c.layers.clck[i] == F32(-1.0) for i in range(1, NC + 1))
    assert float(c.time.gmtr) == pytest.approx(15.0, abs=1e-4)


def test_the_almost_24_guard_is_exactly_24_in_single_precision():
    """The epsilon in (TIME - 23.99999999)/24.0 is 200x smaller than the
    float32 spacing at 24, so the constant IS 24.0 and the guard is inert."""
    assert F32(23.99999999) == F32(24.0)
    assert float(np.spacing(F32(24.0))) > 1e-8


def test_blkdats_truncated_angle_constants_survive_float32():
    """blkdat writes /CON/ to eleven digits, short of double precision, and
    the port keeps those literals rather than computing them.  For the two
    GEOTIM uses that caution costs nothing -- D2R and R2D round to the SAME
    float32 as the exact conversion factors.  PI is where it bites: blkdat's
    3.1415926 and the true pi are different float32 numbers."""
    assert D2R == F32(np.radians(1.0)) and R2D == F32(np.degrees(1.0))
    assert PI != F32(np.pi), "the truncation is visible here, not in D2R/R2D"
