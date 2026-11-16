"""Antenna guard rails: frequency range, absolute-dBi tables, bearings, OFFAZIM.

* Type 14 installs only rows MINFREQ..MAXFREQ (ANTCALC's rule), with
  XFQS/XFQE bracketing them; isotropic entries keep 0..30.
* A requested frequency outside a FILE antenna's range raises (GAIN would
  silently substitute 0 dBi).
* ``*_table_absolute`` skips the header-parameter-1 add; a relative install
  peaking above 30 dBi warns.
* A type-13 file with bearing None (best-case gain) warns; so does a bearing
  that looks like radians.
* ``build_circuit`` keeps the OFFAZIM of a cant passed in by the caller.

All synthetic ``.VOA`` files are built in ``tmp_path``.
"""

import warnings

import numpy as np
import pytest

from voacap_gpu.engine import (antcalc_cant, build_circuit, cant_from_files,
                               predict)
from voacap_gpu.scalar.dazel import dazel0
from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.readant import parse_antenna

from tests.cuda.harness import requires_cuda

TX, RX = (40.0, -105.0), (50.0, 0.0)


def _t14_text(gain_fn, maxgain=0.0, eff_fn=lambda f: 0.0):
    """Synthesise a type-14 .VOA: 30 frequency rows of (eff + 91 gains).

    Each row is ``(2x,f6.1,(t10,10f7.3))``: the efficiency in columns 3-8,
    then 91 gains 10 to a record from column 10.  ``gain_fn(freq_mhz, el)``.
    """
    lines = ["synthetic type 14 pattern",
             " 3     3 parameters",
             f"  {maxgain:.2f}  [ 1] Max Gain dBi..:",
             "  14    [ 2] Antenna Type..:",
             "  0.00  [ 3] Efficiency"]
    for f in range(1, 31):
        vals = [gain_fn(f, el) for el in range(91)]
        for k, lo in enumerate(range(0, 91, 10)):
            chunk = "".join(f"{v:7.3f}" for v in vals[lo:lo + 10])
            head = f"  {eff_fn(f):6.1f} " if k == 0 else " " * 9
            lines.append(head + chunk)
    return "\n".join(lines) + "\n"


def _t13_text(gain_fn, maxgain=0.0, eff=0.0):
    """Synthesise a type-13 .VOA: 360 azimuth rows x 91 elevations."""
    lines = ["synthetic type 13 pattern",
             " 3     3 parameters",
             f"  {maxgain:.2f}  [ 1] Max Gain dBi..:",
             "  13    [ 2] Antenna Type..:",
             f"  {eff:.2f}  [ 3] Efficiency"]
    for az in range(360):
        vals = [gain_fn(az, el) for el in range(91)]
        for lo in range(0, 91, 10):
            lines.append(" " * 9 + "".join(f"{v:7.3f}" for v in vals[lo:lo + 10]))
    return "\n".join(lines) + "\n"


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def _flat14(tmp_path, gain=5.0, maxgain=5.0, name="flat.t14"):
    return _write(tmp_path, name, _t14_text(lambda f, el: gain, maxgain))


def _ztaz():
    return float(dazel0(*TX, *RX)[0])


# ---------------------------------------------------------------------------
# 1. Frequency range
# ---------------------------------------------------------------------------

def test_t14_parse_round_trip(tmp_path):
    """The generator writes the real fixed-column format."""
    ant = parse_antenna(_t14_text(lambda f, el: f + el / 100.0, maxgain=1.0,
                                  eff_fn=lambda f: -f / 10.0))
    assert ant.jant == 14
    assert ant.gains30[4, 7] == F32(5.07)          # 5 MHz, 7 degrees
    assert ant.eff30[9] == F32(-1.0)               # 10 MHz


def test_t14_installs_only_minfreq_to_maxfreq(tmp_path):
    ant = parse_antenna(_t14_text(lambda f, el: f / 2 + el / 100.0, maxgain=1.0,
                                  eff_fn=lambda f: -f / 10.0))
    c = antcalc_cant(ant, None, TX, RX)            # default range 2..30
    table = (ant.gains30 + ant.parm[0]).astype(np.float32)
    # ANTCALC's gainNN.dat f7.3 round trip (load_p2p models it)
    table = (np.round(np.float64(np.maximum(table, np.float32(-99.99))) * 1000.0)
             / 1000.0).astype(np.float32)
    assert not np.any(c.array[1, :, 1]) and float(c.aeff[1, 1]) == 0.0
    for f in (2, 17, 30):
        assert np.array_equal(c.array[f, 1:92, 1], table[f - 1])
        assert c.aeff[f, 1] == ant.eff30[f - 1]
    assert float(c.xfqs[1]) == 2.0 and float(c.xfqe[1]) == 30.0
    # the isotropic RX entry keeps its 0..30 bracket
    assert float(c.xfqs[2]) == 0.0 and float(c.xfqe[2]) == 30.0

    n = antcalc_cant(ant, None, TX, RX, tx_freq_range=(5, 20))
    for f in range(1, 31):
        row = n.array[f, 1:92, 1]
        if 5 <= f <= 20:
            assert np.array_equal(row, table[f - 1])
        else:
            assert not np.any(row) and float(n.aeff[f, 1]) == 0.0
    assert float(n.xfqs[1]) == 5.0 and float(n.xfqe[1]) == 20.0


def test_cant_from_files_freq_range(tmp_path):
    p = _write(tmp_path, "r.t14", _t14_text(lambda f, el: f, maxgain=0.0))
    c = cant_from_files(p, p, rx_freq_range=(3, 12))
    assert float(c.xfqs[1]) == 2.0 and float(c.xfqe[1]) == 30.0
    assert float(c.xfqs[2]) == 3.0 and float(c.xfqe[2]) == 12.0
    assert not np.any(c.array[13:, :, 2]) and not np.any(c.array[:3, :, 2])
    assert np.all(c.array[12, 1:92, 2] == F32(12.0))


def test_type13_freq_range(tmp_path):
    ant = parse_antenna(_t13_text(lambda az, el: 3.0))
    c = antcalc_cant(ant, None, TX, RX, tx_bearing=_ztaz(),
                     tx_freq_range=(4, 9))
    assert float(c.xfqs[1]) == 4.0 and float(c.xfqe[1]) == 9.0
    assert np.all(c.array[4:10, 1:92, 1] == F32(3.0))
    assert not np.any(c.array[:4, :, 1]) and not np.any(c.array[10:, :, 1])


@pytest.mark.parametrize("rng", [(0, 30), (2, 31), (20, 10), (2.5, 30)])
def test_bad_freq_range_refused(tmp_path, rng):
    ant = parse_antenna(_t14_text(lambda f, el: 0.0))
    with pytest.raises(ValueError, match="frequency range"):
        antcalc_cant(ant, None, TX, RX, tx_freq_range=rng)


def test_predict_t14_above_range_raises(tmp_path):
    """A frequency above the type-14 TX's MAXFREQ: previously silently
    isotropic, now refused.  (35 MHz is now caught earlier by predict()'s
    1..30 MHz input check, so narrow the antenna range to exercise this one.)"""
    p = _flat14(tmp_path)
    with pytest.raises(ValueError, match=r"tx antenna.*25 MHz"):
        predict(TX, RX, freqs=[14.0, 25.0], month=6, ssn=100, hours=[12],
                tx_antenna=str(p), tx_antenna_freq_range=(2, 20))
    with pytest.raises(ValueError, match=r"35"):
        predict(TX, RX, freqs=[14.0, 35.0], month=6, ssn=100, hours=[12],
                tx_antenna=str(p))


def test_predict_t13_below_range_raises(tmp_path):
    """1.5 MHz on a type-13 RX: below MINFREQ 2, refused and the end named."""
    p = _write(tmp_path, "d.t13", _t13_text(lambda az, el: 3.0))
    with pytest.raises(ValueError, match=r"rx antenna.*1\.5 MHz"):
        predict(TX, RX, freqs=[1.5], month=6, ssn=100, hours=[12],
                rx_antenna=str(p), rx_bearing=90.0)


def test_predict_range_kwarg_honoured(tmp_path):
    """A narrowed range refuses a frequency the default range would allow."""
    p = _flat14(tmp_path)
    with pytest.raises(ValueError, match=r"tx antenna.*25 MHz"):
        predict(TX, RX, freqs=[25.0], month=6, ssn=100, hours=[12],
                tx_antenna=str(p), tx_antenna_freq_range=(2, 20))


def test_isotropic_ends_never_raise_on_freq():
    antcalc_cant(None, None, TX, RX, freqs=[1.0, 35.0])        # no raise


# ---------------------------------------------------------------------------
# 2. Absolute-dBi tables
# ---------------------------------------------------------------------------

def test_t14_absolute_flag(tmp_path):
    """Flat 5 dB table, header 5: +10 dBi as VOACAP adds it, +5 absolute."""
    ant = parse_antenna(_flat14(tmp_path).read_text())
    rel = antcalc_cant(ant, None, TX, RX)
    ab = antcalc_cant(ant, None, TX, RX, tx_table_absolute=True)
    assert np.all(rel.array[2:31, 1:92, 1] == F32(10.0))
    assert np.all(ab.array[2:31, 1:92, 1] == F32(5.0))
    p = _flat14(tmp_path)
    assert np.all(cant_from_files(p, p, rx_table_absolute=True)
                  .array[2:31, 1:92, 2] == F32(5.0))


def test_t13_absolute_flag(tmp_path):
    """A 12 dBi-peak table with header 12: +24 relative, +12 absolute."""
    ant = parse_antenna(_t13_text(lambda az, el: 12.0, maxgain=12.0))
    kw = dict(tx_bearing=_ztaz())
    assert np.allclose(antcalc_cant(ant, None, TX, RX, **kw)
                       .array[2, 1:92, 1], 24.0)
    assert np.allclose(antcalc_cant(ant, None, TX, RX,
                                    tx_table_absolute=True, **kw)
                       .array[2, 1:92, 1], 12.0)


def test_peak_above_30_dbi_warns(tmp_path):
    hot = parse_antenna(_t13_text(lambda az, el: 12.0, maxgain=20.0))   # 32
    with pytest.warns(UserWarning, match="absolute"):
        antcalc_cant(hot, None, TX, RX, tx_bearing=_ztaz())
    hot14 = parse_antenna(_flat14(tmp_path, gain=16.0, maxgain=16.0)
                          .read_text())                               # 32
    with pytest.warns(UserWarning, match="32.00 dBi"):
        antcalc_cant(None, hot14, TX, RX)
    # neither the absolute install nor a 24 dBi relative one warns
    ok = parse_antenna(_t13_text(lambda az, el: 12.0, maxgain=12.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        antcalc_cant(hot, None, TX, RX, tx_bearing=_ztaz(),
                     tx_table_absolute=True)
        antcalc_cant(ok, hot14, TX, RX, tx_bearing=_ztaz(),
                     rx_table_absolute=True)


@requires_cuda
def test_predict_absolute_flag_end_to_end(tmp_path):
    """Through the GPU: the flat-5 table buys +10 dB relative, +5 absolute."""
    p = _flat14(tmp_path)
    kw = dict(freqs=[14.0], month=6, ssn=100, hours=[12])
    iso = predict(TX, RX, **kw)
    rel = predict(TX, RX, tx_antenna=str(p), **kw)
    ab = predict(TX, RX, tx_antenna=str(p), tx_table_absolute=True, **kw)
    base = float(iso.dbw[0, 0])
    assert float(rel.dbw[0, 0]) - base == pytest.approx(10.0, abs=1e-2)
    assert float(ab.dbw[0, 0]) - base == pytest.approx(5.0, abs=1e-2)


# ---------------------------------------------------------------------------
# 3. Bearing guard rails
# ---------------------------------------------------------------------------

def test_t13_bearing_none_warns(tmp_path):
    ant = parse_antenna(_t13_text(lambda az, el: 3.0))
    with pytest.warns(UserWarning, match=r"tx_bearing is None.*best-case"):
        antcalc_cant(ant, None, TX, RX)
    with pytest.warns(UserWarning, match=r"rx_bearing is None"):
        antcalc_cant(None, ant, TX, RX)


def test_bearing_radians_warns(tmp_path):
    ant = parse_antenna(_t13_text(lambda az, el: 3.0))
    with pytest.warns(UserWarning, match="radians"):
        antcalc_cant(ant, None, TX, RX, tx_bearing=1.2)


def test_no_bearing_warning_when_not_type13(tmp_path):
    """Isotropic / type 14 ignore the bearing, so no warning; nor do a 0 or
    a plainly-degrees bearing on type 13."""
    t14 = parse_antenna(_flat14(tmp_path, gain=0.0, maxgain=0.0).read_text())
    t13 = parse_antenna(_t13_text(lambda az, el: 3.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        antcalc_cant(t14, None, TX, RX)
        antcalc_cant(t14, None, TX, RX, tx_bearing=1.0)
        antcalc_cant(t13, None, TX, RX, tx_bearing=0.0)
        antcalc_cant(t13, None, TX, RX, tx_bearing=135.0)


# ---------------------------------------------------------------------------
# 4. Single OFFAZIM computation
# ---------------------------------------------------------------------------

def test_build_circuit_keeps_caller_offazim(tmp_path):
    """A cant from antcalc_cant keeps the OFFAZIM its type-13 slice was
    installed at, even if build_circuit is handed (different) bearings."""
    ant = parse_antenna(_t13_text(lambda az, el: 10.0 * np.cos(np.radians(az))))
    ztaz = _ztaz()
    cant = antcalc_cant(ant, None, TX, RX, tx_bearing=ztaz - 90.0,
                        rx_bearing=123.0)
    off = cant.offazim.copy()
    arr = cant.array.copy()
    c, _ = build_circuit(TX, RX, 6, 100, 12, [14.0], cant=cant,
                         tx_bearing=ztaz - 45.0, rx_bearing=10.0)
    assert c.cant is cant
    assert np.array_equal(c.cant.offazim, off)
    assert float(c.cant.offazim[1]) == pytest.approx(90.0, abs=1e-3)
    assert np.array_equal(c.cant.array, arr)


def test_build_circuit_isotropic_offazim_from_bearing():
    """With no cant, build_circuit still derives OFFAZIM from the bearing."""
    ztaz = _ztaz()
    c, _ = build_circuit(TX, RX, 6, 100, 12, [14.0], tx_bearing=ztaz - 30.0)
    assert float(c.cant.offazim[1]) == pytest.approx(30.0, abs=0.05)
    assert float(c.cant.offazim[2]) == 0.0
