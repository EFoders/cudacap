"""The public ``predict`` endpoint, validated against the LUFFY oracle decks.

Three tests, three claims:

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

* ``test_predict_smoke`` -- ``predict`` runs from pure Python inputs (no deck),
  returns a sane, correctly-shaped table with the expected monotonicities.

The geomagnetic pole is the reference program's runtime fallback (78.5N,
-69.0), which is what the oracle capture used -- see ``engine`` module docstring.
"""

import numpy as np
import pytest

from voacap_gpu.engine import (POLE_DEFAULT, System, build_circuit,
                               isotropic_cant, predict, predict_circuits)
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
