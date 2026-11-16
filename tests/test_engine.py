"""The public ``predict`` endpoint, validated against the LUFFY oracle decks.

The claims:

* ``test_setup_chain_vs_oracle`` -- the setup chain (REDMAP -> GEOM -> SANG ->
  the per-hour GEOTIM/VIRTIM/TIMVAR/F2VAR/ESIND/IONSET/CURMUF chain), built
  from a deck's RAW inputs (endpoint coordinates off the deck's CIRCUIT card,
  month, SSN, the SYSTEM/FPROB scalars), reproduces the captured LUFFY *entry*
  state field for field, on EVERY captured deck.  The captured state came
  from the stock voacapl, so this is the composition test the replay could
  not be: everything above CURMUF, run as one driver.

* ``test_end_to_end_vs_oracle`` -- feeding those built circuits through LUFFY
  (``cpu``: the scalar reference; ``cuda``: the production kernel) in exactly
  ``predict``'s configuration -- one FRESH circuit per hour sharing one REDMAP
  -- reproduces every deck's captured /SON/ output: hop counts exact, each
  numeric field within an ABSOLUTE bound (table below).  The ICEPAC deck is
  in the set, so this is also the end-to-end check of the sporadic-E path.

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

END-TO-END BOUNDS
-----------------
Measured worst |got - oracle| over the original 22 decks (the 21 in
``golden/dumps/luffy`` + ``luffy_icepac/tanbel_m07_s160_icepac``; the 13 newer
ICEPAC-mode decks in ``luffy_icepac`` are also run against these bounds, see
tests/test_icepac_oracle.py, with ushrot_* an expected failure) x 24 hours x
every requested frequency, fresh circuit per hour, captured antennas.  CPU and
CUDA measured IDENTICAL (bit-for-bit, every deck; see test_cpu_backend.py).
Each bound is ~5x the measured worst, rounded up to a clean number:

    field   units      measured worst  bound   worst cell
    dbw     dBW        1.663e-3        1e-2    sydstg_m01_s100_v30 h18 f4
    sndb    dB-Hz      1.663e-3        1e-2    (same cell)
    dbu     dB(uV/m)   1.663e-3        1e-2    (same cell)
    dblos   dB         1.663e-3        1e-2    (same cell)
    snpr    dB         1.663e-3        1e-2    (same cell)
    reliab  fraction   3.403e-5        2e-4    fbahel_m07_s15_v30 h22 f1
    angle   deg        2.527e-4        2e-3    latok_m01_s100_v20 h22 f2
    delay   ms         5.264e-4        3e-3    lonakl_m01_s100_v30 h8 f6
    vhigh   km         6.018e-2        0.4     lonakl_m01_s100_v30 h8 f6

These replace the old "% bit-exact" floor as the pass criterion (that
fraction is still printed): it measured how often the libm floor happens to
round the same way, not how close the answer is -- lonakl_m07_s15_v30 is
65.7% bit-exact with a worst error of a few thousandths of a dB.  The
floored-ULP bar (``ULP_BAR``, the replay's 25,000) is kept: it holds on every
deck (worst 8,704, lonakl_m01_s100_v30_long dbu).  The relative bar
(``REL_BAR``) is dropped end-to-end: on a near-zero field it is ill-posed --
the ICEPAC deck's snpr of -0.0054 dB is off by 1.5e-5 dB, which is 2.8e-3
relative, passing the 3e-3 bar by 6% while being 100x inside the absolute
bound.

The bounds were measured on Windows (numpy/MSVC f32 libm on the host setup
chain); the oracle is glibc gfortran.  See docs/f32_libm.md -- a different
host libm moves the floor, not the bound's order of magnitude.
"""

import functools
import time

import numpy as np
import pytest

from voacap_gpu import engine as _engine
from voacap_gpu.engine import (POLE_DEFAULT, System, build_circuit,
                               cant_from_files, isotropic_cant, offazim,
                               predict, predict_circuits)
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.redmap import redmap

from tests.scalar.dumputil import ulp_diff_floored
from tests.scalar.test_luffy_replay import DUMPS, _antennas, _rows

# The ICEPAC oracle deck (URSI + FPROB 0.70, Es in all 24 hours; the CCIR twin
# tanbel_m07_s160_v30 has ESMUF=0).  Captured isolated so the 21-deck replay
# set is untouched; its CIRCUIT card is the CCIR twin's.
ICEPAC_DUMPS = DUMPS.parent / "luffy_icepac"
ICEPAC_DECK = "tanbel_m07_s160_icepac"
ICEPAC_COORDS = (35.80, -5.90, 44.90, 20.50, 0)      # TANGIER -> BELGRADE

# The deck sources the dumps were captured from (golden/p2p/matrix/decks).
DECKDIR = DUMPS.parent.parent / "p2p" / "matrix" / "decks"

# The ICEPAC-mode deck matrix (tools/oracle/icepac/gen_icepac_decks.py, all
# URSI + FPROB 0.70) is captured into luffy_icepac too; its CIRCUIT cards are
# in golden/p2p/icepac_decks.  ICEPAC_DECKS = every deck in luffy_icepac.
ICEPAC_DECKDIR = DUMPS.parent.parent / "p2p" / "icepac_decks"
ICEPAC_DECKS = sorted(p.name[:-7] for p in ICEPAC_DUMPS.glob("*_in.csv"))

# Every captured LUFFY deck: the 21 replay decks plus the ICEPAC ones.
ALL_DECKS = sorted(p.name[:-7] for p in DUMPS.glob("*_in.csv")) + ICEPAC_DECKS

# Decks known to break the oracle bounds, with the diagnosed cause.  Kept in
# ALL_DECKS (test_cpu_backend's CPU == CUDA identity still holds on them) but
# xfail(strict) in the two oracle tests here, so a fix shows up as XPASS.
_OUT_OF_BOUNDS = {
    "ushrot_m01_s160_v30_ice": (
        "near-meridional path (dlon 0.17 deg): GEOM's control-point longitude "
        "is ACOS of an argument 7.7e-7 below 1, so a 1-ULP f32 difference "
        "there is 4.7e-5 rad (clong01; clck01 1504 floored ULP) and reaches "
        "angle as 6.5e-3 deg > 2e-3.  Not Es: LUFFY fed the captured entry "
        "state is within bounds, and the CCIR/FPROB-0 twin also breaks angle"),
}
_ORACLE_DECKS = [
    pytest.param(d, marks=pytest.mark.xfail(strict=True,
                                            reason=_OUT_OF_BOUNDS[d]))
    if d in _OUT_OF_BOUNDS else d for d in ALL_DECKS]

# the replay's ULP bar (tests/scalar/test_luffy_replay.py); holds end-to-end
# on every deck, worst 8,704 -- see END-TO-END BOUNDS
ULP_BAR = 25000.0

# Absolute end-to-end bounds per /SON/ field -- see END-TO-END BOUNDS above.
SON_ABS_BOUND = {
    "dbw": 1e-2, "sndb": 1e-2, "dbu": 1e-2, "dblos": 1e-2, "snpr": 1e-2,
    "reliab": 2e-4, "angle": 2e-3, "delay": 3e-3, "vhigh": 0.4,
}

# Setup-chain bars, measured the same way over all 22 decks (fresh circuit
# per hour): worst 172 floored ULP (lonakl_m01_s100_v30 delmuf01) and 2.28e-5
# relative (tanbel delmuf01); ~5x each, rounded up.  The old 97% bit-exact
# floor fails honestly on the two lonakl short-path decks (84.6% / 88.2%,
# DELMUF/angle-of-MUF libm noise), so it is printed, not asserted.
SETUP_ULP_BAR = 1000.0
SETUP_REL_BAR = 1.5e-4

_PER_AREA = ("clat", "clong", "glat", "rd", "gyz", "gmdip", "clck", "abiy",
             "f2m3", "hpf2", "zenang", "zenmax", "rat", "sigpat", "epspat",
             "hs")
_MUF_HR = ("emuf", "f1muf", "f2muf", "esmuf", "allmuf", "fot", "hpf", "angmuf")
_MUF_LAY = ("sigl", "sigu", "delmuf", "hpmuf", "htmuf", "fvmuf", "yfot",
            "yhpf", "ymuf")
# numeric /SON/ columns compared end-to-end (nhp is discrete, checked exactly)
_SON_NUM = ("angle", "dblos", "dbu", "delay", "dbw", "reliab", "sndb", "snpr",
            "vhigh")
assert set(SON_ABS_BOUND) == set(_SON_NUM)


def _system_from_row(v):
    """The SYSTEM/FPROB scalars off a captured entry-state row."""
    return System(
        required_snr=float(v["rsn"]), min_takeoff_deg=float(v["amind"]),
        required_reliability=int(v["lufp"]),
        multipath_power_db=float(v["pmp"]),
        multipath_delay_ms=float(v["dmp"]), man_made_noise=float(v["noise"]),
        fprob=(float(v["psc01"]), float(v["psc02"]), float(v["psc03"]),
               float(v["psc04"])))


def _deck_method(v):
    """The user METHOD that produced this captured (method, mspec) pair."""
    if int(v["mspec"]) == 121:
        return 30
    return int(v["method"])


def _dumps(deck):
    return ICEPAC_DUMPS if deck in ICEPAC_DECKS else DUMPS


def _coeff(deck):
    return "URSI" if (deck in ICEPAC_DECKS or "_ursi" in deck) else "CCIR"


def _circuit_card(deck):
    """(tx lat, tx lon, rx lat, rx lon, npsl) off the deck's CIRCUIT card.

    The card is ``CIRCUIT  51.50N  0.13W  36.85S  174.76E  L  1``: four
    hemisphere-suffixed coordinates (south/west negative) and the S/L path
    flag, where ``L`` is the long great circle, NPSL=1.  The ICEPAC deck was
    generated from its CCIR twin's card; the ICEPAC matrix decks carry their
    own, in ICEPAC_DECKDIR.
    """
    src = ICEPAC_DECKDIR / f"{deck}.dat"
    if not src.exists():
        src = DECKDIR / f"{deck.replace('_icepac', '_v30')}.dat"
    name = src.stem
    with open(src) as f:
        for line in f:
            if line.startswith("CIRCUIT"):
                tok = line.split()
                vals = [float(t[:-1]) * (-1.0 if t[-1] in "SW" else 1.0)
                        for t in tok[1:5]]
                assert tok[5] in ("S", "L"), line
                return (*vals, 1 if tok[5] == "L" else 0)
    raise LookupError(f"{name}.dat has no CIRCUIT card")


def _require_cuda():
    """Skip unless cupy imports AND sees a device.

    ``importorskip("cupy")`` alone is not enough: on a CPU-only CI runner the
    CuPy wheel installs fine and ``predict`` then dies with
    ``cudaErrorNoDevice``.  Same check as ``predict(backend='auto')``.
    """
    if not _engine._cuda_usable():
        pytest.skip("no usable CUDA device")


def _need(backend):
    """Skip a ``backend``-parametrised case whose backend is unavailable."""
    if backend == "cuda":
        _require_cuda()


def _deck_inputs(deck):
    """Raw inputs for one deck: (coords, system, method, coeff, freqs, ins)."""
    ins = _rows(_dumps(deck) / f"{deck}_in.csv")
    v0 = ins[0]
    nf = sum(1 for i in range(1, 12) if v0[f"frel{i:02d}"] > 0)
    freqs = [F32(v0[f"frel{i:02d}"]) for i in range(1, nf + 1)]
    return (_circuit_card(deck), _system_from_row(v0), _deck_method(v0),
            _coeff(deck), freqs, ins)


@functools.lru_cache(maxsize=None)
def run_deck(deck, backend):
    """One deck through LUFFY in ``predict``'s configuration; cached.

    A FRESH circuit per hour (no /SON/ carry-over), all sharing one REDMAP
    and the captured /cantenna/ -- exactly what ``predict`` hands
    ``predict_circuits``.  Returns ``(son, status, seconds)``: ``son`` maps
    each ``_SON_NUM`` field and ``nhp`` to an ``(hours, nf)`` array in
    /SON/ slot order, ``seconds`` is the LUFFY wall time.  Cached so the
    CPU-vs-CUDA identity test reuses these runs rather than repeating them.
    """
    coords, system, method, coeff, freqs, ins = _deck_inputs(deck)
    cant = _antennas(_rows(_dumps(deck) / f"{deck}_ant.csv"))
    v0 = ins[0]
    st = redmap(_engine._COEFFS_DIR, coeff, int(v0["month"]), F32(v0["ssn"]))
    circuits, ipflags = [], []
    for v in ins:
        c, ip = build_circuit(coords[:2], coords[2:4], int(v["month"]),
                              F32(v["ssn"]), int(v["it"]), freqs,
                              method=method, system=system, cant=cant,
                              coeff=coeff, pole=POLE_DEFAULT, npsl=coords[4],
                              iono=st)
        circuits.append(c)
        ipflags.append(ip)
    t0 = time.perf_counter()
    _, status = predict_circuits(circuits, ipflags, backend=backend)
    secs = time.perf_counter() - t0
    nf = len(freqs)
    son = {nm: np.array([np.asarray(getattr(c.son, nm)[1:nf + 1], np.float32)
                         for c in circuits]) for nm in _SON_NUM}
    son["nhp"] = np.array([[int(c.son.nhp[j]) for j in range(1, nf + 1)]
                           for c in circuits], np.int32)
    son["it"] = np.array([int(c.time.it) for c in circuits], np.int32)
    return son, np.asarray(status, np.int32), secs


_BACKENDS = ("cpu", "cuda")


@pytest.mark.skipif(not ALL_DECKS, reason="oracle dumps missing")
@pytest.mark.parametrize("deck", _ORACLE_DECKS)
def test_setup_chain_vs_oracle(deck):
    """Built geometry + per-area ionosphere vs the captured entry state."""
    coords, system, method, coeff, _freqs, ins = _deck_inputs(deck)
    v0 = ins[0]
    tx, rx, npsl = coords[:2], coords[2:4], coords[4]
    st = redmap(_engine._COEFFS_DIR, coeff, int(v0["month"]), F32(v0["ssn"]))

    total = exact = 0
    worst = (0.0, None)
    for v in ins:                    # all 24 hours, fresh circuit as predict
        circ, _ip = build_circuit(tx, rx, int(v["month"]), F32(v["ssn"]),
                                  int(v["it"]), [F32(v[f"frel{i:02d}"])
                                                 for i in range(1, 12)
                                                 if v[f"frel{i:02d}"] > 0],
                                  method=method, system=system, coeff=coeff,
                                  pole=POLE_DEFAULT, npsl=npsl, iono=st)
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
            assert d <= SETUP_ULP_BAR, f"{deck} {name} h{it}: {g!r} vs {e!r}"
            if d and float(e) != 0.0:
                rel = abs(float(g) - float(e)) / abs(float(e))
                assert rel <= SETUP_REL_BAR, (
                    f"{deck} {name} h{it}: {rel:.2e} rel")
    print(f"\n{deck}: setup-chain {exact}/{total} bit-exact "
          f"({100 * exact / total:.2f}%), worst {worst[0]:.0f} ULP at "
          f"{worst[1]}")


@pytest.mark.skipif(not ALL_DECKS, reason="oracle dumps missing")
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("deck", _ORACLE_DECKS)
def test_end_to_end_vs_oracle(deck, backend):
    """Raw inputs -> setup chain -> LUFFY vs the captured /SON/ output."""
    if backend == "cuda":
        _require_cuda()
    outs = _rows(_dumps(deck) / f"{deck}_out.csv")
    son, status, secs = run_deck(deck, backend)
    assert status.tolist() == [0] * len(outs), f"{deck}: status {status}"
    H, nf = son["nhp"].shape
    assert H == len(outs)

    total = exact = 0
    worst = {nm: (0.0, None) for nm in _SON_NUM}
    worst_ulp = (0.0, None)
    for h, vout in enumerate(outs):
        it = int(son["it"][h])
        for j in range(nf):
            assert int(son["nhp"][h, j]) == int(vout[f"nhp{j + 1:02d}"]), (
                f"{deck} nhp{j + 1} h{it}: hop count differs")
            for nm in _SON_NUM:
                g = F32(son[nm][h, j])
                e = F32(vout[f"{nm}{j + 1:02d}"])
                where = f"{nm}{j + 1} h{it}: {g!r} vs {e!r}"
                d = ulp_diff_floored(g, e)
                total += 1
                exact += d == 0.0
                if d > worst_ulp[0]:
                    worst_ulp = (d, where)
                assert d <= ULP_BAR, f"{deck} {where} ({d:.0f} ULP)"
                a = abs(float(g) - float(e))
                if a > worst[nm][0]:
                    worst[nm] = (a, where)
                assert a <= SON_ABS_BOUND[nm], (
                    f"{deck} {where}: |d| {a:.3e} > {SON_ABS_BOUND[nm]:g}")
    print(f"\n{deck} [{backend}]: end-to-end {exact}/{total} bit-exact "
          f"({100 * exact / total:.2f}%), worst {worst_ulp[0]:.0f} ULP; "
          f"LUFFY {1000 * secs / H:.1f} ms/circuit-hour; worst |d| "
          + ", ".join(f"{nm} {worst[nm][0]:.1e}" for nm in _SON_NUM))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_predict_smoke(backend):
    """predict() from pure Python inputs: shape and monotonicities."""
    _need(backend)
    freqs = [5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 28.0]
    hours = list(range(1, 25))
    r = predict((40.0, -105.0), (48.85, 2.35), freqs=freqs, month=6, ssn=120,
                hours=hours, system=System(power_kw=100.0), backend=backend)

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


def test_unsupported_raises(tmp_path):
    """The refusals: too many freqs, an unknown METHOD, a synthesised antenna.

    (A half-given antenna pair is no longer a refusal: ANTCALC handles each
    end independently, so pattern-TX + isotropic-RX is now the common case.)
    """
    with pytest.raises(NotImplementedError):
        build_circuit((0, 0), (10, 10), 6, 100, 12, list(range(1, 13)))
    with pytest.raises(NotImplementedError):
        build_circuit((0, 0), (10, 10), 6, 100, 12, [10.0], method=99)
    # type 1 is method-of-moments synthesis -- out of scope, must refuse
    synth = tmp_path / "synth.voa"
    synth.write_text("some dipole\n 3     3 parameters\n"
                     "  2.15  [ 1] Max Gain dBi..:\n"
                     "  1     [ 2] Antenna Type..:\n"
                     "  0.0   [ 3] Efficiency\n")
    with pytest.raises(NotImplementedError):
        predict((0, 0), (10, 10), freqs=[10.0], month=6, ssn=100,
                tx_antenna=str(synth), rx_antenna=None)


# ---------------------------------------------------------------------------
# OFFAZIM from antenna bearing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not any(DUMPS.glob("*_in.csv")), reason="oracle dumps missing")
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


@pytest.mark.skipif(not any(DUMPS.glob("*_in.csv")), reason="oracle dumps missing")
@pytest.mark.parametrize("backend", _BACKENDS)
def test_bearing_gain_path_vs_oracle(backend):
    """The bearing -> OFFAZIM -> gain path, driven end-to-end on a real deck.

    lonpar's sample.00 is a type-0 antenna: ANTCALC *synthesises* its gain
    table by method-of-moments (10 dBi here), which is out of scope to rebuild
    -- so this uses the captured /cantenna/ (the real gains) but installs
    OFFAZIM from the antenna bearings through ``build_circuit``.  Running that
    through the GPU kernel must still reproduce the deck /SON/: it confirms both
    that our OFFAZIM matches ANTCALC's (to the bearing floor) and that the gain
    LOOKUP consumes the captured table correctly.
    """
    _need(backend)
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
        predict_circuits([circ], [ip], backend=backend)
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

@pytest.mark.parametrize("backend", _BACKENDS)
def test_model_voacap_matches_default(backend):
    """model='voacap' == CCIR + FPROB (1,1,1,0), i.e. the validated config."""
    _need(backend)
    kw = dict(backend=backend, freqs=[7.0, 14.0, 21.0], month=6, ssn=120,
              hours=[6, 12, 18])
    a = predict((40.0, -105.0), (48.85, 2.35), model="voacap", **kw)
    b = predict((40.0, -105.0), (48.85, 2.35), coeff="CCIR",
                system=System(fprob=(1.0, 1.0, 1.0, 0.0)), **kw)
    assert np.array_equal(a.muf, b.muf)
    assert np.allclose(a.dbw, b.dbw, equal_nan=True)
    assert np.allclose(a.snr, b.snr, equal_nan=True)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_model_icepac_matches_ursi_fprob(backend):
    """model='icepac' == URSI + FPROB (1,1,1,0.70), the documented config."""
    _need(backend)
    kw = dict(backend=backend, freqs=[7.0, 14.0, 21.0], month=6, ssn=120,
              hours=[6, 12, 18])
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

# (ICEPAC_DUMPS / ICEPAC_DECK / ICEPAC_COORDS are defined at the top.)  Its
# end-to-end check is test_end_to_end_vs_oracle[tanbel_m07_s160_icepac-*],
# which replaced the carry-over-circuit test with its 87% bit-exact floor.


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


# ---------------------------------------------------------------------------
# The Prop-SL parameter surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", _BACKENDS)
def test_gain_offset_shifts_signal(backend):
    """tx_gos + rx_gos add straight onto the received signal power, in dB."""
    _need(backend)
    kw = dict(backend=backend, freqs=[14.0], month=6, ssn=120, hours=[12])
    base = predict((40.0, -105.0), (48.85, 2.35), **kw)
    off = predict((40.0, -105.0), (48.85, 2.35),
                  system=System(tx_gos=8.0, rx_gos=4.0), **kw)
    d = float(off.dbw[0, 0]) - float(base.dbw[0, 0])
    assert d == pytest.approx(12.0, abs=1e-2), f"gain offset gave {d} dB"


@pytest.mark.parametrize("backend", _BACKENDS)
def test_advisory_params_do_not_change_results(backend):
    """bandwidth / modulation are advisory: carried, but inert to the answer."""
    _need(backend)
    kw = dict(backend=backend, freqs=[7.0, 14.0], month=6, ssn=120, hours=[12])
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


# ---------------------------------------------------------------------------
# Type-13 azimuth-dependent antennas (ANTCALC antcalc.for:268-284 + DAZEL0)
# ---------------------------------------------------------------------------

from voacap_gpu.engine import antcalc_cant                      # noqa: E402
from voacap_gpu.scalar.dazel import dazel0                      # noqa: E402
from voacap_gpu.scalar.readant import parse_antenna             # noqa: E402


def _t13_text(gain_fn, maxgain=1.5, eff=-0.25):
    """Synthesise a type-13 .VOA: 360 azimuth rows x 91 elevations.

    ``gain_fn(az, el)`` -> dB; values are formatted f7.3 exactly as the stock
    files are, so the parse is the real fixed-column path.
    """
    lines = ["synthetic type 13 pattern",
             " 3     3 parameters",
             f"  {maxgain:.2f}  [ 1] Max Gain dBi..:",
             "  13    [ 2] Antenna Type..:",
             f"  {eff:.2f}  [ 3] Efficiency"]
    for az in range(360):
        vals = [gain_fn(az, el) for el in range(91)]
        for lo in range(0, 91, 10):
            chunk = vals[lo:lo + 10]
            lines.append(" " * 9 + "".join(f"{v:7.3f}" for v in chunk))
    return "\n".join(lines) + "\n"


def test_dazel0_bearings():
    """DAZEL0 basics: cardinal bearings, wrap, and agreement with DAZEL."""
    az, dgc = dazel0(0.0, 0.0, 0.0, 10.0)          # due east on the equator
    assert az == pytest.approx(90.0, abs=1e-4)
    assert dgc == pytest.approx(10.0 * 111.19, rel=1e-3)
    az, _ = dazel0(0.0, 0.0, 10.0, 0.0)            # due north
    assert az == pytest.approx(0.0, abs=1e-4)
    az, _ = dazel0(10.0, 0.0, 0.0, 0.0)            # due south
    assert az == pytest.approx(180.0, abs=1e-4)
    # antipodal nudge must keep the bearing defined (no NaN/exception)
    az, dgc = dazel0(10.0, 0.0, -10.0, 180.0)
    assert np.isfinite(az) and np.isfinite(dgc)
    # a general path agrees with the full DAZEL mode 0 to well under the
    # 1-degree azimuth table resolution
    from voacap_gpu.scalar.dazel import Azel, dazel
    z = Azel(ztlat=40.0, ztlon=-105.0, zrlat=50.0, zrlon=0.0)
    dazel(z, 0)
    az, dgc = dazel0(40.0, -105.0, 50.0, 0.0)
    assert az == pytest.approx(float(z.ztaz), abs=1e-3)
    assert dgc == pytest.approx(float(z.zdgc), rel=1e-4)


def test_t13_install_exact(tmp_path):
    """load_p2p_t13 reproduces ANTCALC's arm value-for-value.

    Checks the truncated azimuth index, the 359->0 wrap, the f32 interpolation
    order, the PARM(1) add, the -99.99 floor, the f7.3 quantisation, and that
    only rows MINFREQ..MAXFREQ are touched (row 1 keeps the zero preset).
    """
    from voacap_gpu.scalar.antennastate import CantennaState

    gain_fn = lambda az, el: (az % 36) / 10.0 - el / 100.0      # noqa: E731
    ant = parse_antenna(_t13_text(gain_fn, maxgain=1.5, eff=-0.25))
    assert ant.jant == 13

    cant = CantennaState()
    cant.load_p2p_t13(1, ant, 1, 123.4)
    off = F32(123.4)
    fract = F32(off - F32(123.0))
    g1 = ant.gains360[123]
    g2 = ant.gains360[124]
    expect = (g1 + (g2 - g1) * fract) + ant.parm[0]
    expect = np.maximum(expect, F32(-99.99))
    expect = (np.round(np.float64(expect) * 1000.0) / 1000.0).astype(np.float32)
    assert np.array_equal(np.asarray(cant.array[2, 1:92, 1]), expect)
    # every frequency row 2..30 carries the same cut; row 1 is untouched
    for f in (3, 17, 30):
        assert np.array_equal(np.asarray(cant.array[f, 1:92, 1]), expect)
    assert not np.any(np.asarray(cant.array[1, :, 1]))
    assert float(cant.aeff[1, 1]) == 0.0
    assert float(cant.aeff[2, 1]) == pytest.approx(-0.25)
    assert float(cant.xfqs[1]) == 2.0 and float(cant.xfqe[1]) == 30.0

    # 359 -> 0 wrap
    cant2 = CantennaState()
    cant2.load_p2p_t13(1, ant, 1, 359.5)
    w1, w2 = ant.gains360[359], ant.gains360[0]
    wexp = (w1 + (w2 - w1) * F32(F32(359.5) - F32(359.0))) + ant.parm[0]
    wexp = (np.round(np.float64(np.maximum(wexp, F32(-99.99))) * 1000.0)
            / 1000.0).astype(np.float32)
    assert np.array_equal(np.asarray(cant2.array[2, 1:92, 1]), wexp)

    # deep nulls floor at -99.99 (ANTCALC's gain-file clamp)
    deep = parse_antenna(_t13_text(lambda az, el: -99.9, maxgain=-50.0))
    cant3 = CantennaState()
    cant3.load_p2p_t13(1, deep, 1, 10.0)
    assert np.all(np.asarray(cant3.array[2, 1:92, 1]) == F32(-99.99))


def test_antcalc_cant_t13_azimuth_dependence(tmp_path):
    """A rotated beam installs a different azimuth slice; isotropic RX rides
    along as its own entry (the mixed case Prop-SL sends most often)."""
    f = tmp_path / "dir.t13"
    f.write_text(_t13_text(lambda az, el: 10.0 * np.cos(np.radians(az)),
                           maxgain=0.0, eff=0.0))
    ant = parse_antenna(f.read_text())
    tx, rx = (40.0, -105.0), (50.0, 0.0)
    ztaz, _ = dazel0(*tx, *rx)

    on_path = antcalc_cant(ant, None, tx, rx, tx_bearing=float(ztaz))
    rotated = antcalc_cant(ant, None, tx, rx,
                           tx_bearing=float(F32(ztaz) - F32(90.0)))
    # beam on the path: OFFAZIM 0 -> the +10 dB azimuth-0 slice
    assert float(on_path.offazim[1]) == pytest.approx(0.0, abs=1e-4)
    assert np.allclose(np.asarray(on_path.array[2, 1:92, 1]), 10.0, atol=5e-3)
    # beam 90 deg off the path: OFFAZIM 90 -> the 0 dB slice
    assert float(rotated.offazim[1]) == pytest.approx(90.0, abs=1e-3)
    assert np.allclose(np.asarray(rotated.array[2, 1:92, 1]), 0.0, atol=2e-2)
    # the isotropic RX entry: zero table, full 0..30 MHz bracket
    assert int(on_path.iats[2]) == 2
    assert not np.any(np.asarray(on_path.array[:, :, 2]))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_predict_t13_smoke(tmp_path, backend):
    """predict() end-to-end on the GPU with a directive type-13 TX pattern:
    rotating the beam 90 degrees off the path costs ~10 dB of signal."""
    _need(backend)
    f = tmp_path / "dir.t13"
    f.write_text(_t13_text(lambda az, el: 10.0 * np.cos(np.radians(az)),
                           maxgain=0.0, eff=0.0))
    tx, rx = (40.0, -105.0), (50.0, 0.0)
    ztaz, _ = dazel0(*tx, *rx)
    kw = dict(freqs=[14.0], month=7, ssn=100, hours=[12], backend=backend)

    on = predict(tx, rx, tx_antenna=str(f), tx_bearing=float(ztaz), **kw)
    off = predict(tx, rx, tx_antenna=str(f),
                  tx_bearing=float(F32(ztaz) - F32(90.0)), **kw)
    assert int(on.status[0]) == 0 and int(off.status[0]) == 0
    d = float(on.dbw[0, 0]) - float(off.dbw[0, 0])
    assert d == pytest.approx(10.0, abs=0.5), (
        f"rotating the beam changed signal by {d} dB, expected ~10")


def test_t13_offazim_360_wraps(tmp_path):
    """OFFAZIM landing on exactly 360.0 wraps to the azimuth-0 slice instead
    of indexing one past the table (a due-north path with beam 0)."""
    from voacap_gpu.scalar.antennastate import CantennaState

    ant = parse_antenna(_t13_text(lambda az, el: az / 100.0))
    cant = CantennaState()
    cant.load_p2p_t13(1, ant, 1, 360.0)          # must not raise
    q = (np.round(np.float64(ant.gains360[0] + ant.parm[0]) * 1000.0)
         / 1000.0).astype(np.float32)
    assert np.array_equal(np.asarray(cant.array[2, 1:92, 1]), q)
    # and through antcalc_cant with a due-north geometry + beam 0
    f = tmp_path / "n.t13"
    f.write_text(_t13_text(lambda az, el: az / 100.0))
    c = antcalc_cant(parse_antenna(f.read_text()), None,
                     (10.0, 0.0), (60.0, 0.0), tx_bearing=0.0)
    assert 0.0 <= float(c.offazim[1]) < 360.0
