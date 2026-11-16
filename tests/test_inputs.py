"""The public boundary refuses bad inputs instead of computing something.

Each refusal below was, before ``voacap_gpu.validate`` existed, a silent wrong
answer or a cryptic downstream error (0.5 MHz -> 0 dBW with status 0, month
13 -> FileNotFoundError, hour 12.7 -> hour 12, noise -145 -> category
'remote', ...).  Also covered: the Fortran-faithful truncation warnings, the
DMP / PMP multipath field rename and its deprecated (crossed) aliases, and the
``model`` toggle refusing to override an explicit coeff / fprob.

None of these needs a GPU: every check fires before the kernel.
"""

import warnings

import numpy as np
import pytest

from voacap_gpu import engine, validate
from voacap_gpu.engine import System, build_circuit, predict, setup_config
from voacap_gpu.scalar.f32math import F32

TX, RX = (40.0, -105.0), (48.85, 2.35)
KW = dict(freqs=[14.0], month=6, ssn=120, hours=[12])


def _predict(**over):
    kw = dict(KW)
    kw.update(over)
    tx = kw.pop("tx", TX)
    rx = kw.pop("rx", RX)
    return predict(tx, rx, **kw)


def _build(system=None, **over):
    args = dict(tx=TX, rx=RX, month=6, ssn=120, hour=12, freqs=[14.0])
    args.update(over)
    return build_circuit(args["tx"], args["rx"], args["month"], args["ssn"],
                         args["hour"], args["freqs"], system=system)


# ---------------------------------------------------------------------------
# Frequencies, month, hours, SSN, coordinates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f", [0.5, 0.99, 30.01, 35.0, 60.0, float("nan"),
                               float("inf"), -14.0])
def test_frequency_out_of_range_raises(f):
    with pytest.raises(ValueError, match="frequency"):
        _predict(freqs=[7.0, f])
    with pytest.raises(ValueError, match="frequency"):
        _build(freqs=[f])


def test_frequency_bounds_accepted():
    assert validate.check_freqs([1.0, 30.0, np.float32(14.1)]) == \
        [1.0, 30.0, float(np.float32(14.1))]


def test_frequency_count_rule_kept():
    with pytest.raises(NotImplementedError):
        _predict(freqs=list(range(1, 13)))
    with pytest.raises(NotImplementedError):
        _predict(freqs=[])


@pytest.mark.parametrize("m", [0, 13, 6.5, -1, float("nan")])
def test_month_refused(m):
    with pytest.raises(ValueError, match="month"):
        _predict(month=m)
    with pytest.raises(ValueError, match="month"):
        _build(month=m)


def test_month_integral_forms_accepted():
    assert validate.check_month(np.int64(6)) == 6
    assert validate.check_month(6.0) == 6


@pytest.mark.parametrize("h", [0, 25, 12.7, -3, float("nan")])
def test_hour_refused(h):
    with pytest.raises(ValueError, match="hour"):
        _predict(hours=[1, h])
    with pytest.raises(ValueError, match="hour"):
        _build(hour=h)


def test_hours_integral_forms_accepted():
    assert validate.check_hours([np.int32(12), 12.0, np.float64(24.0), 1]) \
        == [12, 12, 24, 1]
    assert validate.check_hours(range(1, 25)) == list(range(1, 25))


def test_hours_empty_refused():
    with pytest.raises(ValueError, match="hours"):
        _predict(hours=[])


@pytest.mark.parametrize("s", [-50.0, -0.1, 300.5, 400.0, float("nan"),
                               float("inf")])
def test_ssn_refused(s):
    with pytest.raises(ValueError, match="ssn"):
        _predict(ssn=s)
    with pytest.raises(ValueError, match="ssn"):
        _build(ssn=s)


def test_ssn_bounds_accepted():
    assert validate.SSN_MAX == 300.0
    assert validate.check_ssn(0) == 0.0
    assert validate.check_ssn(np.float32(300.0)) == 300.0


@pytest.mark.parametrize("p", [(91.0, 0.0), (-90.5, 0.0), (float("nan"), 0.0),
                               (0.0, float("inf")), (0.0, float("nan"))])
def test_coordinates_refused(p):
    with pytest.raises(ValueError, match="tx"):
        _predict(tx=p)
    with pytest.raises(ValueError, match="rx"):
        _predict(rx=p)
    with pytest.raises(ValueError, match="tx"):
        _build(tx=p)


def test_coordinate_bounds_accepted():
    for p in ((90.0, 180.0), (-90.0, -180.0), (np.float32(51.5), -0.13)):
        assert validate.check_point("tx", p) is p      # in range: untouched


@pytest.mark.parametrize("lon,want", [(180.0, 180.0), (-180.0, -180.0),
                                      (190.0, -170.0), (-190.0, 170.0),
                                      (540.0, 180.0), (-540.0, -180.0),
                                      (725.5, 5.5), (359.0, -1.0)])
def test_wrap_lon(lon, want):
    """Longitude is circular: wrapped by whole turns, both endpoints kept
    (PropServer's normaliseLon / gridjob.normalise_lon rule)."""
    assert validate.wrap_lon(lon) == pytest.approx(want, abs=1e-9)
    assert validate.check_point("rx", (10.0, lon))[1] == pytest.approx(
        want, abs=1e-9)
    arr = validate.wrap_lon(np.array([lon, 0.0]))
    assert arr[0] == pytest.approx(want, abs=1e-9) and arr[1] == 0.0


def test_wrapped_longitude_gives_the_same_circuit():
    """A web map panned past the date line reports 190: same answer as -170."""
    a, _ = build_circuit((45.42, -75.70), (35.0, 190.0), 7, 120, 12, [14.0])
    b, _ = build_circuit((45.42, -75.70), (35.0, -170.0), 7, 120, 12, [14.0])
    assert float(a.gcdkm) == float(b.gcdkm)
    assert np.array_equal(np.asarray(a.layers.clong), np.asarray(b.layers.clong))
    assert float(a.rlongd) == float(b.rlongd) == -170.0


def test_coeff_spelling_refused():
    """REDMAP reads anything but the exact 'URSI' as CCIR -- refuse it."""
    with pytest.raises(ValueError, match="coeff"):
        _predict(coeff="ursi")
    with pytest.raises(ValueError, match="coeff"):
        build_circuit(TX, RX, 6, 120, 12, [14.0], coeff="ITU")


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad,match", [
    (dict(power_kw=0.0), "power_kw"),
    (dict(power_kw=-1.0), "power_kw"),
    (dict(power_kw=float("nan")), "power_kw"),
    (dict(required_snr=float("nan")), "required_snr"),
    (dict(required_snr=float("inf")), "required_snr"),
    (dict(required_reliability=0), "required_reliability"),
    (dict(required_reliability=100), "required_reliability"),
    (dict(required_reliability=99.5), "required_reliability"),
    (dict(man_made_noise=float("nan")), "man_made_noise"),
    (dict(man_made_noise=0.5), "man_made_noise"),
    (dict(min_takeoff_deg=float("nan")), "min_takeoff_deg"),
    (dict(tx_gos=float("inf")), "tx_gos"),
    (dict(fprob=(1.0, 1.0, 1.0)), "fprob"),
    (dict(multipath_delay_ms=float("nan")), "multipath_delay_ms"),
])
def test_system_refused(bad, match):
    s = System(**bad)
    with pytest.raises(ValueError, match=match):
        _predict(system=s)
    with pytest.raises(ValueError, match=match):
        _build(system=s)
    with pytest.raises(ValueError, match=match):
        setup_config(s, freqs=[14.0])


@pytest.mark.parametrize("noise", [-145.0, -5, -1000, -2.5, -0.5])
def test_noise_signed_dbw_refused(noise):
    with pytest.raises(ValueError, match="pass 145, not -145"):
        _predict(system=System(man_made_noise=noise))


@pytest.mark.parametrize("noise", [0, -1, -2, -3, -4, -2.0, 145, 164.0])
def test_noise_categories_and_levels_accepted(noise):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate.check_system(System(man_made_noise=noise))


def test_noise_truncation_warns_but_is_kept():
    s = System(man_made_noise=144.7)
    with pytest.warns(UserWarning, match="truncated to 144"):
        circ, _ = _build(system=s)
    assert circ.noise == 144                    # Fortran-faithful int()


def test_reliability_truncation_warns_but_is_kept():
    s = System(required_reliability=90.5)
    with pytest.warns(UserWarning, match="truncated to 90"):
        cfg = setup_config(s, freqs=[14.0])
    assert int(cfg["lufp"]) == 90


def test_truncation_warns_once_per_system():
    """predict re-checks inside build_circuit per hour; warn once, not 24x."""
    s = System(man_made_noise=144.7)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(3):
            validate.check_system(s)
    assert len([x for x in w if issubclass(x.category, UserWarning)]) == 1
    s.man_made_noise = 150.2                    # a new value warns afresh
    with pytest.warns(UserWarning, match="truncated to 150"):
        validate.check_system(s)


def test_integral_values_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate.check_system(System(man_made_noise=145.0,
                                     required_reliability=np.int64(90)))


def test_min_takeoff_nonpositive_is_3_degrees():
    _psc, amind = System(min_takeoff_deg=0.0)._psc()
    assert amind == F32(3.0)
    _psc, amind = System(min_takeoff_deg=-1.0)._psc()
    assert amind == F32(3.0)


# ---------------------------------------------------------------------------
# DMP / PMP: the multipath field names and their deprecated aliases
# ---------------------------------------------------------------------------

def test_default_multipath_is_the_golden_decks():
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # a bare System() never warns
        s = System()
    assert (s.dmp, s.pmp) == (0.1, 3.0)
    circ, _ = _build(system=s)
    assert circ.dmp == F32(0.1) and circ.pmp == F32(3.0)


def test_new_names_map_to_dmp_and_pmp():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = System(multipath_delay_ms=0.25, multipath_power_db=6.0)
    assert (s.dmp, s.pmp) == (0.25, 6.0)
    circ, _ = _build(system=s)
    assert circ.dmp == F32(0.25)                # DMP: max delay, ms
    assert circ.pmp == F32(6.0)                 # PMP: power tolerance, dB


def test_deprecated_names_keep_the_crossed_mapping():
    with pytest.warns(FutureWarning, match="CROSSED"):
        s = System(max_delay_ms=5.0)
    assert s.pmp == 5.0 and s.dmp == 0.1        # max_delay_ms -> PMP (dB)
    circ, _ = _build(system=s)
    assert circ.pmp == F32(5.0)
    with pytest.warns(FutureWarning, match="max_spread_db"):
        s = System(max_spread_db=0.2)
    assert s.dmp == 0.2 and s.pmp == 3.0        # max_spread_db -> DMP (ms)
    circ, _ = _build(system=s)
    assert circ.dmp == F32(0.2)


def test_deprecated_names_positional_still_work():
    """The old fields kept their position (4th, 5th) in the dataclass."""
    with pytest.warns(FutureWarning):
        s = System(73.0, 0.1, 90, 4.0, 0.3)
    assert (s.pmp, s.dmp) == (4.0, 0.3)


@pytest.mark.parametrize("kw", [
    dict(max_delay_ms=3.0, multipath_power_db=3.0),
    dict(max_spread_db=0.1, multipath_delay_ms=0.2),
    dict(max_delay_ms=3.0, multipath_delay_ms=0.1),
    dict(max_spread_db=0.1, multipath_power_db=3.0),
])
def test_old_and_new_names_together_raise(kw):
    with pytest.raises(ValueError, match="deprecated"):
        System(**kw)


def test_mutating_into_a_mix_is_caught_at_use():
    s = System(multipath_power_db=4.0)
    s.max_delay_ms = 5.0
    with pytest.raises(ValueError, match="deprecated"):
        _build(system=s)


def test_model_does_not_rewarn_deprecated_alias():
    with pytest.warns(FutureWarning):
        s = System(max_delay_ms=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _c, r = engine._resolve_model("icepac", "CCIR", s)
    assert r.pmp == 5.0 and r.fprob == (1.0, 1.0, 1.0, 0.70)
    assert s.fprob == (1.0, 1.0, 1.0, 0.0)      # the caller's is untouched


@pytest.mark.parametrize("kw", [
    dict(),
    dict(multipath_delay_ms=0.25, multipath_power_db=6.0),
    dict(max_delay_ms=5.0),
    dict(max_spread_db=0.2),
    dict(required_reliability=50, man_made_noise=-3, required_snr=10.0,
         min_takeoff_deg=0.0),
])
def test_setup_config_agrees_with_build_circuit(kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        s = System(**kw)
    for method in (30, 20, 25):
        cfg = setup_config(s, method=method, freqs=[7.0, 14.0])
        circ, _ = build_circuit(TX, RX, 6, 120, 12, [7.0, 14.0],
                                method=method, system=s)
        assert F32(cfg["dmp"]) == circ.dmp
        assert F32(cfg["pmp"]) == circ.pmp
        assert F32(cfg["rsn"]) == circ.rsn
        assert F32(cfg["amind"]) == circ.amind
        assert int(cfg["lufp"]) == circ.lufp
        assert int(cfg["noise"]) == circ.noise
        assert int(cfg["method"]) == circ.method
        assert int(cfg["mspec"]) == circ.mspec
        assert np.array_equal(np.asarray(cfg["psc"], np.float32),
                              np.asarray(circ.layers.psc, np.float32))


# ---------------------------------------------------------------------------
# The model toggle never silently overrides an explicit choice
# ---------------------------------------------------------------------------

def test_model_with_conflicting_coeff_raises():
    with pytest.raises(ValueError, match="coeff"):
        engine._resolve_model("voacap", "URSI", System())
    with pytest.raises(ValueError, match="coeff"):
        _predict(model="voacap", coeff="URSI")


@pytest.mark.parametrize("model,fprob", [
    ("voacap", (1.0, 1.0, 1.0, 0.70)),
    ("voacap", (0.9, 1.0, 1.0, 0.0)),
    ("icepac", (1.0, 1.0, 1.0, 0.5)),
    ("icepac", (0.9, 1.0, 1.0, 0.70)),
])
def test_model_with_conflicting_fprob_raises(model, fprob):
    with pytest.raises(ValueError, match="fprob"):
        engine._resolve_model(model, "CCIR", System(fprob=fprob))
    with pytest.raises(ValueError, match="fprob"):
        _predict(model=model, system=System(fprob=fprob))


@pytest.mark.parametrize("model,coeff,fprob,want", [
    ("voacap", "CCIR", (1.0, 1.0, 1.0, 0.0), "CCIR"),
    ("voacap", None, (1.0, 1.0, 1.0, 0.0), "CCIR"),
    ("icepac", "CCIR", (1.0, 1.0, 1.0, 0.0), "URSI"),   # CCIR == "not given"
    ("icepac", "URSI", (1.0, 1.0, 1.0, 0.70), "URSI"),
    ("icepac", "URSI", (1.0, 1.0, 1.0, np.float32(0.7)), "URSI"),
    ("ICEPAC", "URSI", (1.0, 1.0, 1.0, 0.0), "URSI"),
])
def test_model_consistent_inputs_accepted(model, coeff, fprob, want):
    c, s = engine._resolve_model(model, coeff, System(fprob=fprob))
    assert c == want
    assert s.fprob == ((1.0, 1.0, 1.0, 0.0) if want == "CCIR"
                       else (1.0, 1.0, 1.0, 0.70))


def test_model_none_passes_through():
    s = System(fprob=(1.0, 1.0, 1.0, 0.5))
    assert engine._resolve_model(None, "URSI", s) == ("URSI", s)


def test_unknown_model_still_not_implemented():
    with pytest.raises(NotImplementedError, match="voacap"):
        engine._resolve_model("nonsense", "CCIR", System())
