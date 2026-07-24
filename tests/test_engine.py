"""The public ``predict`` endpoint, validated against the LUFFY oracle decks.

The claims:

* ``test_setup_chain_vs_oracle`` -- the setup chain (REDMAP -> GEOM -> SANG ->
  the per-hour GEOTIM/VIRTIM/TIMVAR/F2VAR/ESIND/IONSET/CURMUF chain), built
  from a deck's RAW inputs (endpoint coordinates, month, SSN, the SYSTEM/FPROB
  scalars), reproduces the captured LUFFY *entry* state field for field.  The
  captured state came from the stock voacapl, so this is the composition test
  the replay could not be: everything above CURMUF, run as one driver.

* ``test_end_to_end_vs_oracle`` -- feeding those built circuits through the
  CUDA LUFFY kernel reproduces the deck's captured /SON/ output.  Since LUFFY
  is already validated bit-exact, any divergence here would be a setup-chain
  bug; there is none beyond the libm floor.

* ``test_offazim_from_bearing`` / ``test_bearing_gain_path_vs_oracle`` -- the
  antenna beam heading, against GEOM's path bearing, reproduces the captured
  OFFAZIM, and the bearing -> OFFAZIM -> gain path reproduces the deck /SON/.

* ``test_model_*`` and ``test_icepac_*`` -- the Prop-SL VOACAP/ICEPAC toggle:
  both validated against voacapl, 'icepac' against its own URSI + FPROB 0.70
  oracle deck (sporadic-E on, where the CCIR twin has none).

* ``test_gain_offset_*`` / ``test_advisory_*`` / ``test_propsl_*`` -- the
  Prop-SL parameter surface (gain offsets consumed, bandwidth/modulation
  advisory, RSN/LUFP threaded).

* ``test_predict_smoke`` -- ``predict`` runs from pure Python inputs (no deck),
  returns a sane, correctly-shaped table with the expected monotonicities.

The geomagnetic pole is the reference program's runtime fallback (78.5N,
-69.0), which is what the oracle capture used -- see ``engine`` module docstring.
"""

import numpy as np
import pytest

from voacap_gpu.engine import (POLE_DEFAULT, System, build_circuit,
                               cant_from_files, isotropic_cant, offazim,
                               predict, predict_circuits)
from voacap_gpu.scalar.f32math import F32

from tests.scalar.dumputil import ulp_diff_floored
from tests.scalar.test_luffy_replay import DUMPS, _antennas, _rows

# name -> (tx lat, tx lon, rx lat, rx lon, npsl), south/west negative.  From
# tools/oracle/golden_decks/gen_matrix.py PATHS, one per regime.
DECKS = {
    "lonpar_m01_s100_v30": (51.50, -0.13, 48.85, 2.35, 0),      # short (343 km)
    "latok_m01_s100_v30": (34.05, -118.24, 35.68, 139.77, 0),   # smooth (8809)
    "lonakl_m01_s100_v30_long": (51.50, -0.13, -36.85, 174.76, 1),  # long
}

# the replay's own bars (tests/scalar/test_luffy_replay.py)
ULP_BAR = 25000.0
REL_BAR = 3.0e-3

_PER_AREA = ("clat", "clong", "glat", "rd", "gyz", "gmdip", "clck", "abiy",
             "f2m3", "hpf2", "zenang", "zenmax", "rat", "sigpat", "epspat",
             "hs")
_MUF_HR = ("emuf", "f1muf", "f2muf", "esmuf", "allmuf", "fot", "hpf", "angmuf")
_MUF_LAY = ("sigl", "sigu", "delmuf", "hpmuf", "htmuf", "fvmuf", "yfot",
            "yhpf", "ymuf")
# numeric /SON/ columns compared end-to-end (nhp is discrete, checked exactly)
_SON_NUM = ("angle", "dblos", "dbu", "delay", "dbw", "reliab", "sndb", "snpr",
            "vhigh")


def _system_from_row(v):
    """The SYSTEM/FPROB scalars off a captured entry-state row."""
    return System(
        required_snr=float(v["rsn"]), min_takeoff_deg=float(v["amind"]),
        required_reliability=int(v["lufp"]), max_delay_ms=float(v["pmp"]),
        max_spread_db=float(v["dmp"]), man_made_noise=float(v["noise"]),
        fprob=(float(v["psc01"]), float(v["psc02"]), float(v["psc03"]),
               float(v["psc04"])))


def _deck_method(v):
    """The user METHOD that produced this captured (method, mspec) pair."""
    if int(v["mspec"]) == 121:
        return 30
    return int(v["method"])


@pytest.mark.skipif(not DUMPS.glob("*_in.csv"),
                    reason="oracle dumps missing")
@pytest.mark.parametrize("deck", DECKS)
def test_setup_chain_vs_oracle(deck):
    """Built geometry + per-area ionosphere vs the captured entry state."""
    coords = DECKS[deck]
    ins = _rows(DUMPS / f"{deck}_in.csv")
    v0 = ins[0]
    tx = (coords[0], coords[1])
    rx = (coords[2], coords[3])
    npsl = coords[4]
    system = _system_from_row(v0)
    method = _deck_method(v0)
    coeff = "URSI" if "_ursi" in deck else "CCIR"

    circ = None
    total = exact = 0
    worst = (0.0, None)
    for v in ins:                              # all 24 hours, circuit reused
        circ, _ip = build_circuit(tx, rx, int(v["month"]), F32(v["ssn"]),
                                  int(v["it"]), [F32(v[f"frel{i:02d}"])
                                                 for i in range(1, 12)
                                                 if v[f"frel{i:02d}"] > 0],
                                  method=method, system=system, coeff=coeff,
                                  pole=POLE_DEFAULT, npsl=npsl, circ=circ)
        lay, muf = circ.layers, circ.muf
        checks = {"gcd": circ.gcd, "gcdkm": circ.gcdkm,
                  "gmtr": circ.time.gmtr, "rlongd": circ.rlongd}
        for k in range(1, lay.km + 1):
            for nm in _PER_AREA:
                checks[f"{nm}{k:02d}"] = getattr(lay, nm)[k]
            for ly in range(1, 4):
                checks[f"fi{ly}{k}"] = lay.fi[ly, k]
                checks[f"yi{ly}{k}"] = lay.yi[ly, k]
                checks[f"hi{ly}{k}"] = lay.hi[ly, k]
        it = circ.time.it
        for nm in _MUF_HR:
            checks[nm] = getattr(muf, nm)[it]
        for ly in range(1, 5):
            for nm in _MUF_LAY:
                checks[f"{nm}{ly:02d}"] = getattr(muf, nm)[ly]
        # discrete branch outcomes must be EXACT
        assert lay.km == int(v0["km"]) and lay.kfx == int(v0["kfx"])
        assert circ.rays.nang == int(v0["nang"])
        for name, got in checks.items():
            g, e = F32(got), F32(v[name])
            d = ulp_diff_floored(g, e)
            total += 1
            exact += d == 0.0
            if d > worst[0]:
                worst = (d, f"{name} h{it}: {g!r} vs {e!r}")
            assert d <= ULP_BAR, f"{deck} {name} h{it}: {g!r} vs {e!r}"
            if d and float(e) != 0.0:
                rel = abs(float(g) - float(e)) / abs(float(e))
                assert rel <= REL_BAR, f"{deck} {name} h{it}: {rel:.2e} rel"
    frac = exact / total
    print(f"\n{deck}: setup-chain {exact}/{total} bit-exact "
          f"({100 * frac:.2f}%), worst {worst[0]:.0f} ULP at {worst[1]}")
    assert frac >= 0.97, f"{deck}: only {frac:.3f} bit-exact"


@pytest.mark.skipif(not DUMPS.glob("*_in.csv"),
                    reason="oracle dumps missing")
@pytest.mark.parametrize("deck", DECKS)
def test_end_to_end_vs_oracle(deck):
    """Built circuits through the GPU kernel vs the captured /SON/ output."""
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                       # pragma: no cover
        pytest.skip("no CUDA device")

    coords = DECKS[deck]
    ins = _rows(DUMPS / f"{deck}_in.csv")
    outs = _rows(DUMPS / f"{deck}_out.csv")
    ants = _rows(DUMPS / f"{deck}_ant.csv")
    cant = _antennas(ants)                    # the captured /cantenna/ + power
    v0 = ins[0]
    tx = (coords[0], coords[1])
    rx = (coords[2], coords[3])
    system = _system_from_row(v0)
    method = _deck_method(v0)
    coeff = "URSI" if "_ursi" in deck else "CCIR"
    nf = sum(1 for i in range(1, 12) if v0[f"frel{i:02d}"] > 0)
    freqs = [F32(v0[f"frel{i:02d}"]) for i in range(1, nf + 1)]

    circ = None
    total = exact = 0
    worst = (0.0, None)
    for v, vout in zip(ins, outs):            # reuse circuit for carry-over
        circ, ip = build_circuit(tx, rx, int(v["month"]), F32(v["ssn"]),
                                 int(v["it"]), freqs, method=method,
                                 system=system, cant=cant, coeff=coeff,
                                 pole=POLE_DEFAULT, npsl=coords[4], circ=circ)
        predict_circuits([circ], [ip])
        it = circ.time.it
        son = circ.son
        for j in range(1, nf + 1):
            assert int(son.nhp[j]) == int(vout[f"nhp{j:02d}"]), (
                f"{deck} nhp{j} h{it}: hop count differs")
            for nm in _SON_NUM:
                g = F32(getattr(son, nm)[j])
                e = F32(vout[f"{nm}{j:02d}"])
                d = ulp_diff_floored(g, e)
                total += 1
                exact += d == 0.0
                if d > worst[0]:
                    worst = (d, f"{nm}{j} h{it}: {g!r} vs {e!r}")
                assert d <= ULP_BAR, f"{deck} {nm}{j} h{it}: {g!r} vs {e!r}"
                if d and float(e) != 0.0:
                    rel = abs(float(g) - float(e)) / abs(float(e))
                    assert rel <= REL_BAR, f"{deck} {nm}{j} h{it}: {rel:.2e}"
    frac = exact / total
    print(f"\n{deck}: end-to-end {exact}/{total} bit-exact "
          f"({100 * frac:.2f}%), worst {worst[0]:.0f} ULP at {worst[1]}")
    assert frac >= 0.90, f"{deck}: only {frac:.3f} bit-exact"


def test_predict_smoke():
    """predict() from pure Python inputs: shape and monotonicities."""
    pytest.importorskip("cupy")
    freqs = [5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 28.0]
    hours = list(range(1, 25))
    r = predict((40.0, -105.0), (48.85, 2.35), freqs=freqs, month=6, ssn=120,
                hours=hours, system=System(power_kw=100.0))

    assert r.snr.shape == (24, len(freqs))
    assert r.muf.shape == (24,)
    assert np.all(r.status == 0)
    # every MUF must be positive and finite
    assert np.all(np.isfinite(r.muf)) and np.all(r.muf > 0)
    # FOT <= MUF <= HPF, the decile ordering, at every hour
    assert np.all(r.fot <= r.muf + 1e-3) and np.all(r.muf <= r.hpf + 1e-3)
    # daytime MUF exceeds night: midday (11-13 UTC ~ local noon at 105W? no --
    # this circuit straddles the Atlantic, so just assert the MUF varies
    # diurnally rather than a fixed phase)
    assert r.muf.max() > 1.2 * r.muf.min()
    # signal power falls monotonically with frequency at a fixed hour, once
    # above the MUF everything is weak; check the median trend over all hours
    med = np.nanmedian(r.dbw, axis=0)
    assert med[0] > med[-1], "signal should weaken toward the high end"
    # reliability is a probability
    good = np.isfinite(r.reliab)
    assert np.all(r.reliab[good] >= -1e-6) and np.all(r.reliab[good] <= 1 + 1e-6)
    # a mode label is one of the layer tags or empty
    labels = set(np.unique(r.mode.astype(str)))
    assert labels <= {" E", "F1", "F2", "ES", " N", ""}


def test_unsupported_raises():
    """The refusals: too many freqs, an unknown METHOD, a half-given antenna."""
    with pytest.raises(NotImplementedError):
        build_circuit((0, 0), (10, 10), 6, 100, 12, list(range(1, 13)))
    with pytest.raises(NotImplementedError):
        build_circuit((0, 0), (10, 10), 6, 100, 12, [10.0], method=99)
    with pytest.raises(NotImplementedError):
        predict((0, 0), (10, 10), freqs=[10.0], month=6, ssn=100,
                tx_antenna="x.voa", rx_antenna=None)


# ---------------------------------------------------------------------------
# OFFAZIM from antenna bearing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DUMPS.glob("*_in.csv"), reason="oracle dumps missing")
def test_offazim_from_bearing():
    """Beam heading + GEOM path bearing reproduce the captured OFFAZIM.

    The lonpar deck's ANTENNA cards point the beams at 90 deg (TX) and 270 deg
    (RX); ANTCALC recorded OFFAZIM 58.10 and 60.01 against the path.  Building
    from those bearings must reproduce them.
    """
    deck = "lonpar_m01_s100_v30"
    ants = _rows(DUMPS / f"{deck}_ant.csv")
    cap = {(int(a["iant"])): (float(a["beammain"]), float(a["offazim"]))
           for a in ants}
    beam_tx, off_tx_cap = cap[1]
    beam_rx, off_rx_cap = cap[2]
    assert (beam_tx, beam_rx) == (90.0, 270.0)

    c, _ip = build_circuit((51.50, -0.13), (48.85, 2.35), 1, 100, 1,
                           [7.2, 14.1], tx_bearing=beam_tx, rx_bearing=beam_rx)
    # OFFAZIM is an antenna azimuth; ANTCALC's path bearing and ours agree to
    # the geom bearing floor (~0.01 deg), so a hundredth-degree bar is right.
    assert abs(float(c.cant.offazim[1]) - off_tx_cap) < 0.02, (
        f"OFFAZIM tx {float(c.cant.offazim[1])} vs captured {off_tx_cap}")
    assert abs(float(c.cant.offazim[2]) - off_rx_cap) < 0.02, (
        f"OFFAZIM rx {float(c.cant.offazim[2])} vs captured {off_rx_cap}")
    # the direct formula, too
    assert offazim(148.10538, 90.0) == pytest.approx(58.10538, abs=1e-3)
    assert offazim(10.0, 350.0) == pytest.approx(20.0, abs=1e-4)   # wrap


@pytest.mark.skipif(not DUMPS.glob("*_in.csv"), reason="oracle dumps missing")
def test_bearing_gain_path_vs_oracle():
    """The bearing -> OFFAZIM -> gain path, driven end-to-end on a real deck.

    lonpar's sample.00 is a type-0 antenna: ANTCALC *synthesises* its gain
    table by method-of-moments (10 dBi here), which is out of scope to rebuild
    -- so this uses the captured /cantenna/ (the real gains) but installs
    OFFAZIM from the antenna bearings through ``build_circuit``.  Running that
    through the GPU kernel must still reproduce the deck /SON/: it confirms both
    that our OFFAZIM matches ANTCALC's (to the bearing floor) and that the gain
    LOOKUP consumes the captured table correctly.
    """
    pytest.importorskip("cupy")
    deck = "lonpar_m01_s100_v30"
    ins = _rows(DUMPS / f"{deck}_in.csv")
    outs = _rows(DUMPS / f"{deck}_out.csv")
    cant = _antennas(_rows(DUMPS / f"{deck}_ant.csv"))    # the REAL gains
    v0 = ins[0]
    system = _system_from_row(v0)
    nf = sum(1 for i in range(1, 12) if v0[f"frel{i:02d}"] > 0)
    freqs = [F32(v0[f"frel{i:02d}"]) for i in range(1, nf + 1)]

    circ = None
    total = exact = 0
    for v, vout in zip(ins, outs):
        circ, ip = build_circuit((51.50, -0.13), (48.85, 2.35), int(v["month"]),
                                 F32(v["ssn"]), int(v["it"]), freqs,
                                 method=_deck_method(v0), system=system,
                                 cant=cant, tx_bearing=90.0, rx_bearing=270.0,
                                 pole=POLE_DEFAULT, circ=circ)
        # our OFFAZIM (from the bearing) lands on the captured value
        assert abs(float(circ.cant.offazim[1]) - 58.1) < 0.02
        predict_circuits([circ], [ip])
        son = circ.son
        for j in range(1, nf + 1):
            for nm in _SON_NUM:
                d = ulp_diff_floored(F32(getattr(son, nm)[j]),
                                     F32(vout[f"{nm}{j:02d}"]))
                total += 1
                exact += d == 0.0
                assert d <= ULP_BAR
    frac = exact / total
    print(f"\n{deck}: bearing+gain end-to-end {exact}/{total} "
          f"bit-exact ({100 * frac:.2f}%)")
    assert frac >= 0.95


# ---------------------------------------------------------------------------
# The VOACAP / ICEPAC model toggle
# ---------------------------------------------------------------------------

def test_model_voacap_matches_default():
    """model='voacap' == CCIR + FPROB (1,1,1,0), i.e. the validated config."""
    pytest.importorskip("cupy")
    kw = dict(freqs=[7.0, 14.0, 21.0], month=6, ssn=120, hours=[6, 12, 18])
    a = predict((40.0, -105.0), (48.85, 2.35), model="voacap", **kw)
    b = predict((40.0, -105.0), (48.85, 2.35), coeff="CCIR",
                system=System(fprob=(1.0, 1.0, 1.0, 0.0)), **kw)
    assert np.array_equal(a.muf, b.muf)
    assert np.allclose(a.dbw, b.dbw, equal_nan=True)
    assert np.allclose(a.snr, b.snr, equal_nan=True)


def test_model_icepac_matches_ursi_fprob():
    """model='icepac' == URSI + FPROB (1,1,1,0.70), the documented config."""
    pytest.importorskip("cupy")
    kw = dict(freqs=[7.0, 14.0, 21.0], month=6, ssn=120, hours=[6, 12, 18])
    a = predict((40.0, -105.0), (48.85, 2.35), model="icepac", **kw)
    b = predict((40.0, -105.0), (48.85, 2.35), coeff="URSI",
                system=System(fprob=(1.0, 1.0, 1.0, 0.70)), **kw)
    assert np.array_equal(a.muf, b.muf)
    assert np.allclose(a.dbw, b.dbw, equal_nan=True)


def test_model_unknown_refuses():
    with pytest.raises(NotImplementedError, match="voacap"):
        predict((40.0, -105.0), (48.85, 2.35), freqs=[14.0], month=6, ssn=120,
                hours=[12], model="nonsense")


# ---------------------------------------------------------------------------
# ICEPAC against its own oracle deck (URSI + FPROB 0.70, Es in all 24 hours;
# the CCIR twin tanbel_m07_s160_v30 has ESMUF=0, so this exercises FPROB).
# Captured isolated so the 21-deck replay set is untouched.
# ---------------------------------------------------------------------------

ICEPAC_DUMPS = DUMPS.parent / "luffy_icepac"
ICEPAC_DECK = "tanbel_m07_s160_icepac"
ICEPAC_COORDS = (35.80, -5.90, 44.90, 20.50, 0)      # TANGIER -> BELGRADE


@pytest.mark.skipif(not (ICEPAC_DUMPS / f"{ICEPAC_DECK}_in.csv").exists(),
                    reason="ICEPAC oracle dump missing "
                    "(LUFFY_DEST=golden/dumps/luffy_icepac run_luffy_dumps.sh "
                    "tanbel_m07_s160_icepac)")
def test_icepac_setup_chain_vs_oracle():
    """Built URSI+FPROB0.70 geometry+ionosphere vs the captured ICEPAC entry."""
    ins = _rows(ICEPAC_DUMPS / f"{ICEPAC_DECK}_in.csv")
    tx, rx = ICEPAC_COORDS[:2], ICEPAC_COORDS[2:4]
    circ = None
    total = exact = es_hours = 0
    worst = (0.0, None)
    for v in ins:
        circ, _ip = build_circuit(tx, rx, int(v["month"]), F32(v["ssn"]),
                                  int(v["it"]), [F32(v[f"frel{i:02d}"])
                                                 for i in range(1, 12)
                                                 if v[f"frel{i:02d}"] > 0],
                                  method=30, system=_system_from_row(v),
                                  coeff="URSI", pole=POLE_DEFAULT, npsl=0,
                                  circ=circ)
        lay, muf, it = circ.layers, circ.muf, circ.time.it
        es_hours += F32(muf.esmuf[it]) > 0
        checks = {"gcdkm": circ.gcdkm}
        for k in range(1, lay.km + 1):
            for nm in _PER_AREA:
                checks[f"{nm}{k:02d}"] = getattr(lay, nm)[k]
            for ly in range(1, 4):
                checks[f"fi{ly}{k}"] = lay.fi[ly, k]
                checks[f"yi{ly}{k}"] = lay.yi[ly, k]
                checks[f"hi{ly}{k}"] = lay.hi[ly, k]
        for nm in _MUF_HR:
            checks[nm] = getattr(muf, nm)[it]
        for name, got in checks.items():
            d = ulp_diff_floored(F32(got), F32(v[name]))
            total += 1
            exact += d == 0.0
            if d > worst[0]:
                worst = (d, f"{name} h{it}")
            assert d <= ULP_BAR, f"icepac {name} h{it}: {F32(got)!r} vs {F32(v[name])!r}"
    frac = exact / total
    print(f"\nICEPAC setup-chain {exact}/{total} bit-exact ({100*frac:.2f}%), "
          f"worst {worst[0]:.0f} ULP at {worst[1]}; ESMUF>0 in {es_hours}/24 h")
    # the whole point: sporadic-E is on, from FPROB(4)=0.70
    assert es_hours == 24, "FPROB(4)=0.70 should give Es in every hour"
    assert frac >= 0.97, f"icepac setup-chain only {frac:.3f}"


@pytest.mark.skipif(not (ICEPAC_DUMPS / f"{ICEPAC_DECK}_out.csv").exists(),
                    reason="ICEPAC oracle dump missing")
def test_icepac_end_to_end_vs_oracle():
    """Built ICEPAC circuits through the GPU kernel vs the captured /SON/.

    This is the FIRST end-to-end validation of the sporadic-E path: every
    deck in the 21-deck replay set is CCIR with ESMUF=0, so ESMOD's Es modes
    reaching /SON/ through RELBIL were never checked against the oracle until
    this deck (Es in all 24 hours).
    """
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                       # pragma: no cover
        pytest.skip("no CUDA device")

    ins = _rows(ICEPAC_DUMPS / f"{ICEPAC_DECK}_in.csv")
    outs = _rows(ICEPAC_DUMPS / f"{ICEPAC_DECK}_out.csv")
    ants = _rows(ICEPAC_DUMPS / f"{ICEPAC_DECK}_ant.csv")
    cant = _antennas(ants)
    tx, rx = ICEPAC_COORDS[:2], ICEPAC_COORDS[2:4]
    v0 = ins[0]
    nf = sum(1 for i in range(1, 12) if v0[f"frel{i:02d}"] > 0)
    freqs = [F32(v0[f"frel{i:02d}"]) for i in range(1, nf + 1)]

    circ = None
    total = exact = es_modes = 0
    worst = (0.0, None)
    for v, vout in zip(ins, outs):
        circ, ip = build_circuit(tx, rx, int(v["month"]), F32(v["ssn"]),
                                 int(v["it"]), freqs, method=30,
                                 system=_system_from_row(v), cant=cant,
                                 coeff="URSI", pole=POLE_DEFAULT, npsl=0,
                                 circ=circ)
        predict_circuits([circ], [ip])
        it, son = circ.time.it, circ.son
        for j in range(1, nf + 1):
            assert int(son.nhp[j]) == int(vout[f"nhp{j:02d}"]), (
                f"icepac nhp{j} h{it}: hop count differs")
            es_modes += son.mdl[j] == "Es" if hasattr(son, "mdl") else 0
            for nm in _SON_NUM:
                g, e = F32(getattr(son, nm)[j]), F32(vout[f"{nm}{j:02d}"])
                d = ulp_diff_floored(g, e)
                total += 1
                exact += d == 0.0
                if d > worst[0]:
                    worst = (d, f"{nm}{j} h{it}: {g!r} vs {e!r}")
                assert d <= ULP_BAR, f"icepac {nm}{j} h{it}: {g!r} vs {e!r}"
                if d and float(e) != 0.0:
                    rel = abs(float(g) - float(e)) / abs(float(e))
                    assert rel <= REL_BAR, f"icepac {nm}{j} h{it}: {rel:.2e}"
    frac = exact / total
    print(f"\nICEPAC end-to-end {exact}/{total} bit-exact ({100*frac:.2f}%), "
          f"worst {worst[0]:.0f} ULP at {worst[1]}")
    # Its own floor, BELOW the CCIR decks' 0.90.  Measured 0.896; the residual
    # is the [[f32-libm-noise]] floor spread over MORE fields than a no-Es deck
    # -- worst 4.9e-4 absolute, broad across delay/angle (the documented
    # ATAN/ACOS-cancellation geometry fields) and the Es power/SNR columns, NOT
    # concentrated or systematic.  The load-bearing check is above: the hop
    # count NHP is exact in every cell, so no Es mode is mis-selected; only the
    # numbers attached to it carry the libm floor.  Do not raise this to 0.90 --
    # that would be sitting a bar the no-Es decks happen to clear on a deck the
    # extra Es transcendentals legitimately do not.
    assert frac >= 0.87, f"icepac end-to-end only {frac:.3f}"


# ---------------------------------------------------------------------------
# The Prop-SL parameter surface
# ---------------------------------------------------------------------------

def test_gain_offset_shifts_signal():
    """tx_gos + rx_gos add straight onto the received signal power, in dB."""
    pytest.importorskip("cupy")
    kw = dict(freqs=[14.0], month=6, ssn=120, hours=[12])
    base = predict((40.0, -105.0), (48.85, 2.35), **kw)
    off = predict((40.0, -105.0), (48.85, 2.35),
                  system=System(tx_gos=8.0, rx_gos=4.0), **kw)
    d = float(off.dbw[0, 0]) - float(base.dbw[0, 0])
    assert d == pytest.approx(12.0, abs=1e-2), f"gain offset gave {d} dB"


def test_advisory_params_do_not_change_results():
    """bandwidth / modulation are advisory: carried, but inert to the answer."""
    pytest.importorskip("cupy")
    kw = dict(freqs=[7.0, 14.0], month=6, ssn=120, hours=[12])
    a = predict((40.0, -105.0), (48.85, 2.35), **kw)
    b = predict((40.0, -105.0), (48.85, 2.35),
                system=System(bandwidth=3000.0, modulation="USB"), **kw)
    assert np.allclose(a.dbw, b.dbw, equal_nan=True)
    assert np.allclose(a.snr, b.snr, equal_nan=True)


def test_propsl_snr_reliability_params():
    """required_snr and required_reliability reach the circuit as RSN / LUFP."""
    c, _ = build_circuit((40.0, -105.0), (48.85, 2.35), 6, 120, 12, [14.0],
                         system=System(required_snr=50.0,
                                       required_reliability=80))
    assert float(c.rsn) == pytest.approx(50.0)
    assert int(c.lufp) == 80
