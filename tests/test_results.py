"""``engine.Results``: status, the not-computed mask, the lookup API, and the
grid reducer's handling of the NaNs that mask introduces.

The GPU tests drive the real CUDA LUFFY kernel; the rest are pure numpy.
"""

import math

import numpy as np
import pytest

from voacap_gpu import engine
from voacap_gpu.engine import (Results, System, _assemble, antcalc_cant,
                               build_circuit, predict, predict_circuits)
from voacap_gpu.gridjob import CellReducer
from voacap_gpu.scalar.antennastate import AREA_COVERAGE

from tests.cuda.harness import requires_cuda

TOKYO = (35.68, 139.77)
SYDNEY = (-33.87, 151.21)          # ~7,820 km: inside the smoothing band
LONDON = (51.50, -0.13)
PARIS = (48.85, 2.35)              # 343 km: pure short path

NUMERIC = ("snr", "dbw", "reliab", "field_dbu")


def _cant(tx, rx, system):
    return antcalc_cant(None, None, tx, rx, power_kw=system.power_kw,
                        tx_gos=system.tx_gos, rx_gos=system.rx_gos)


def _run_direct(tx, rx, freqs, hours, month=7, ssn=160):
    """build_circuit + predict_circuits + _assemble, bypassing predict()'s
    input validation, so slots predict() would refuse can still be exercised.
    """
    system = System()
    freqs_f = np.array(freqs, np.float32)
    cant = _cant(tx, rx, system)
    circuits, ipflags = [], []
    muf, fot, hpf = (np.empty(len(hours), np.float32) for _ in range(3))
    # build_circuit now validates frequencies (1..30 MHz), so build with a
    # valid stand-in and then write the deliberate out-of-range values straight
    # into the FREL slots LUFFY reads -- the kernel path is what is under test.
    build_f = np.where((freqs_f >= 1.0) & (freqs_f <= 30.0), freqs_f,
                       np.float32(10.0)).astype(np.float32)
    for k, h in enumerate(hours):
        c, ip = build_circuit(tx, rx, month, ssn, h, build_f, system=system,
                              cant=cant)
        for j, f in enumerate(freqs_f, start=1):
            c.frq.frel[j] = f
        it = c.time.it
        muf[k], fot[k], hpf[k] = (c.muf.allmuf[it], c.muf.fot[it],
                                  c.muf.hpf[it])
        circuits.append(c)
        ipflags.append(ip)
    _, status = predict_circuits(circuits, ipflags)
    raw_nhp = np.array([[int(c.son.nhp[j + 1]) for j in range(len(freqs))]
                        for c in circuits])
    raw_dbw = np.array([[float(c.son.dbw[j + 1]) for j in range(len(freqs))]
                        for c in circuits])
    r = _assemble(circuits, hours, freqs_f, muf, fot, hpf,
                  system.required_reliability, status=status)
    return r, status, raw_nhp, raw_dbw


def _assert_masked(r, invalid):
    for name in NUMERIC:
        a = r.array(name)
        assert np.all(np.isnan(a[invalid])), name
        assert np.all(np.isfinite(a[~invalid])), name
    assert all(m == "" for m in r.mode[invalid])


# ---------------------------------------------------------------------------
# Task 1: status reaches Results
# ---------------------------------------------------------------------------

@requires_cuda
def test_kernel_refusal_reaches_results_through_predict(monkeypatch):
    """An area-coverage receive antenna (OFFAZIM == -999) is unported: GAIN
    flags it (oob_flag 2) and GENOIS turns that into status 3 on the very
    first frequency.  predict() must carry that status into Results, and the
    whole hour must be invalid.

    The sentinel goes in AFTER build_circuit, which writes OFFAZIM itself."""
    real = engine.build_circuit

    def area_coverage(*a, **kw):
        c, ip = real(*a, **kw)
        c.cant.offazim[2] = AREA_COVERAGE          # the receive entry
        return c, ip

    monkeypatch.setattr(engine, "build_circuit", area_coverage)
    r = predict(LONDON, PARIS, freqs=[7.0, 14.0], month=6, ssn=100,
                hours=[1, 13])
    assert r.status.dtype == np.int32 and r.status.shape == (2,)
    assert np.all(r.status == 3), r.status
    assert not r.valid.any()
    _assert_masked(r, ~r.valid)
    assert np.all(np.isnan(r.luf))
    rec = r.at(13, 14.0)
    assert rec["status"] == 3 and rec["valid"] is False


@requires_cuda
def test_status_from_predict_circuits_is_not_discarded(monkeypatch):
    """Whatever predict_circuits returns is what Results carries, per hour."""
    real = engine.predict_circuits

    def fake(circuits, ipflags, fast_math=False):
        circuits, status = real(circuits, ipflags, fast_math=fast_math)
        status = np.array(status, np.int32)
        status[1] = 22
        return circuits, status

    monkeypatch.setattr(engine, "predict_circuits", fake)
    r = predict(LONDON, PARIS, freqs=[7.0, 14.0], month=6, ssn=100,
                hours=[1, 2, 3])
    assert r.status.tolist() == [0, 22, 0]
    assert r.valid[[0, 2]].all() and not r.valid[1].any()
    _assert_masked(r, ~r.valid)


@requires_cuda
def test_sub_1mhz_is_a_refusal_not_zeros():
    """The reported '0.5 MHz returns snr=0, dbw=0, status 0' was status 2 being
    thrown away: GENOIS's receive-antenna GAIN lookup leaves the table below
    1 MHz (Stage A raises IndexError there).  On a smoothing-band path the
    blend never runs, so even the slots before the refusal are not answers."""
    r, status, raw_nhp, raw_dbw = _run_direct(TOKYO, SYDNEY, [7.0, 14.0, 0.5],
                                              hours=[1, 12])
    assert np.all(status == 2), status
    assert np.all(raw_dbw[:, 2] == 0.0)          # the preset that looked real
    assert np.all(raw_nhp[:, 2] == -1)
    assert not r.valid.any()
    _assert_masked(r, ~r.valid)


# ---------------------------------------------------------------------------
# Task 2: the valid mask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tx,rx", [(LONDON, PARIS), (TOKYO, SYDNEY)],
                         ids=["short", "smoothing"])
@requires_cuda
def test_empty_slot_is_invalid_and_nan(tx, rx):
    """A FREL <= 0 slot is LUFFY's "no frequency in this slot" skip: status
    stays 0, NHP keeps its -1 preset, the /SON/ presets (0 dBW...) remain.
    Only that cell may be invalid; its neighbours are untouched."""
    freqs = [7.0, 0.0, 14.0]
    r, status, raw_nhp, raw_dbw = _run_direct(tx, rx, freqs, hours=[1, 12])
    assert np.all(status == 0)
    assert np.all(raw_nhp[:, 1] == -1)
    assert np.all(raw_dbw[:, 1] == 0.0)
    expect = np.ones_like(r.valid)
    expect[:, 1] = False
    np.testing.assert_array_equal(r.valid, expect)
    _assert_masked(r, ~r.valid)
    # the computed neighbours are exactly what the kernel wrote
    np.testing.assert_array_equal(r.dbw[:, [0, 2]],
                                  raw_dbw[:, [0, 2]].astype(np.float32))
    np.testing.assert_array_equal(r.nhp, raw_nhp)


@requires_cuda
def test_ordinary_circuit_is_all_valid():
    """Tokyo -> Sydney, every hour: nothing an ordinary request returns may be
    flagged, including the very negative over-the-MUF cells."""
    r = predict(TOKYO, SYDNEY, freqs=[7.0, 14.0, 21.0], month=7, ssn=160,
                hours=range(1, 25))
    assert np.all(r.status == 0)
    assert r.valid.shape == (24, 3) and r.valid.all()
    for name in NUMERIC:
        assert np.all(np.isfinite(r.array(name))), name
    assert np.all(r.nhp >= 1)
    assert all(m != "" for m in r.mode.ravel())
    # the over-the-MUF floor is real data, and stays
    assert r.dbw.min() < -400.0
    assert r.reliab.min() > 0.0


@requires_cuda
def test_predict_smoke_path_is_all_valid():
    """The test_engine smoke circuit is likewise fully valid."""
    r = predict((40.0, -105.0), PARIS, freqs=[5.0, 7.0, 10.0, 14.0, 18.0,
                                              21.0, 28.0],
                month=6, ssn=120, hours=range(1, 25),
                system=System(power_kw=100.0))
    assert np.all(r.status == 0) and r.valid.all()


# ---------------------------------------------------------------------------
# Task 3: the Results API (pure numpy)
# ---------------------------------------------------------------------------

def _toy(status=(0, 0)):
    H, F = 2, 3
    nhp = np.array([[1, 2, -1], [3, 1, 1]], np.int32)
    mode = np.array([["F2", " E", ""], ["F2", "F2", "F1"]], object)
    rel = np.array([[0.95, 0.5, np.nan], [0.1, 0.92, 0.99]], np.float32)
    z = np.zeros((H, F), np.float32)
    return Results(hours=np.array([5, 6], np.int32),
                   freqs=np.array([7.1, 14.0, 21.0], np.float32),
                   muf=np.array([20.0, 22.0], np.float32),
                   fot=np.array([17.0, 18.0], np.float32),
                   hpf=np.array([25.0, 26.0], np.float32),
                   luf=np.array([7.1, 14.0], np.float32),
                   snr=z + 10, dbw=z - 120, reliab=rel, field_dbu=z + 5,
                   nhp=nhp, mode=mode, status=np.array(status, np.int32))


def test_at_requires_an_exact_frequency():
    r = _toy()
    assert r.at(5, 14.0)["freq"] == 14.0
    assert r.at(5, 7.1)["freq"] == pytest.approx(7.1, abs=1e-6)  # float32
    assert r.at(5, 14.00005)["freq"] == 14.0                     # within 1e-4
    with pytest.raises(KeyError, match="15.9"):
        r.at(5, 15.9)                       # used to snap silently to 14
    with pytest.raises(KeyError, match="14.001"):
        r.at(5, 14.001)


def test_at_missing_hour_is_a_keyerror():
    r = _toy()
    with pytest.raises(KeyError, match="hour 7"):
        r.at(7, 14.0)


def test_at_and_records_carry_valid_and_status():
    r = _toy(status=(0, 4))
    rec = r.at(5, 21.0)
    assert rec["valid"] is False and rec["status"] == 0 and rec["nhp"] == -1
    assert r.at(5, 7.1)["valid"] is True
    assert r.at(6, 7.1)["valid"] is False and r.at(6, 7.1)["status"] == 4
    recs = r.to_records()
    assert len(recs) == 6
    assert all({"valid", "status"} <= set(x) for x in recs)
    assert [x["valid"] for x in recs] == [True, True, False,
                                         False, False, False]


def test_default_valid_derives_from_nhp_and_status():
    r = _toy(status=(0, 0))
    np.testing.assert_array_equal(
        r.valid, [[True, True, False], [True, True, True]])


def test_lowest_reliable_freq_is_luf():
    r = _toy()
    assert r.lowest_reliable_freq is r.luf


def test_lowest_reliable_freq_skips_invalid_cells():
    """_assemble's LUF scan: NaN never meets the threshold."""
    class _Son:
        def __init__(self, rel, nhp):
            n = 14
            self.sndb = np.zeros(n, np.float32)
            self.dbw = np.full(n, -100.0, np.float32)
            self.dbu = np.zeros(n, np.float32)
            self.reliab = np.zeros(n, np.float32)
            self.reliab[1:4] = rel
            self.nhp = np.full(n, -1, np.int32)
            self.nhp[1:4] = nhp
            self.mode = ["F2"] * n

    class _C:
        def __init__(self, rel, nhp):
            self.son = _Son(rel, nhp)

    freqs = np.array([3.0, 7.0, 14.0], np.float32)
    # the 3 MHz slot was not computed; its stale 0.99 must not make it the LUF
    circs = [_C([0.99, 0.0, 0.95], [-1, 1, 1]),
             _C([0.0, 0.5, 0.95], [1, 1, 1])]
    v = np.zeros(2, np.float32)
    r = _assemble(circs, [1, 2], freqs, v, v, v, 90,
                  status=np.array([0, 0], np.int32))
    assert r.valid.tolist() == [[False, True, True], [True, True, True]]
    assert r.lowest_reliable_freq.tolist() == [14.0, 14.0]
    assert math.isnan(r.reliab[0, 0]) and r.mode[0, 0] == ""
    with pytest.raises(ValueError, match="status"):
        _assemble(circs, [1, 2], freqs, v, v, v, 90,
                  status=np.zeros(3, np.int32))


# ---------------------------------------------------------------------------
# Task 4: CellReducer and non-finite samples
# ---------------------------------------------------------------------------

def _feed(red, ok, snr, dbw, dbu, rel, f, cells=None):
    n = len(snr)
    red.feed(np.arange(n) if cells is None else cells, np.asarray(ok, bool),
             np.asarray(snr, float), np.asarray(dbw, float),
             np.asarray(dbu, float), np.asarray(rel, float), f)


@pytest.mark.parametrize("mode", ["snr", "signalpower", "fieldstrength",
                                  "reliability"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_sample_equals_a_failed_circuit(mode, bad):
    """Feeding a non-finite value anywhere in a sample == feeding ok=False."""
    rng = np.random.default_rng(7)
    n = 40
    blocks = []
    for f in (7.0, 14.0, 7.0, 21.0, 7.0):
        vals = [rng.normal(0, 20, n) for _ in range(4)]
        blocks.append((f, vals, rng.random(n) < 0.3))
    for monthly in (False, True):
        a = CellReducer(n, mode, monthly_median=monthly, default_freq=3.0)
        b = CellReducer(n, mode, monthly_median=monthly, default_freq=3.0)
        for k, (f, vals, poison) in enumerate(blocks):
            which = k % 4                     # poison a different column
            pv = [v.copy() for v in vals]
            pv[which][poison] = bad
            _feed(a, np.ones(n, bool), *pv, f)
            _feed(b, ~poison, *vals, f)
        for x, y in zip(a.finalise(), b.finalise()):
            np.testing.assert_array_equal(x, y)
            assert np.all(np.isfinite(x))


def test_nan_does_not_win_or_poison_the_mean():
    red = CellReducer(1, "snr", monthly_median=True)
    _feed(red, [True], [10.0], [-120.0], [5.0], [50.0], 7.0)
    _feed(red, [True], [np.nan], [np.nan], [np.nan], [np.nan], 14.0)
    _feed(red, [True], [8.0], [-121.0], [4.0], [40.0], 7.0)
    _feed(red, [True], [np.nan], [-1.0], [1.0], [1.0], 7.0)  # one NaN value
    snr, dbw, dbu, rel, freq = red.finalise()
    assert freq[0] == 7.0
    assert snr[0] == pytest.approx(9.0) and dbw[0] == pytest.approx(-120.5)
    assert dbu[0] == pytest.approx(4.5) and rel[0] == pytest.approx(45.0)


def test_all_nan_cell_gets_the_sentinels():
    red = CellReducer(2, "signalpower", default_freq=9.0)
    _feed(red, [True, True], [np.nan, 1.0], [np.nan, -130.0], [np.nan, 2.0],
          [np.nan, 3.0], 14.0)
    snr, dbw, dbu, rel, freq = red.finalise()
    assert (snr[0], dbw[0], dbu[0], rel[0], freq[0]) == (-200.0, -200.0,
                                                         -200.0, 0.0, 9.0)
    assert dbw[1] == -130.0 and freq[1] == 14.0


def test_results_nan_cells_feed_cleanly():
    """End-to-end shape of the intended use: feed Results columns straight in;
    the invalid cell contributes nothing."""
    r = _toy()
    red = CellReducer(2, "reliability")
    for j, f in enumerate(r.freqs):
        _feed(red, r.status == 0, r.snr[:, j], r.dbw[:, j],
              r.field_dbu[:, j], 100.0 * r.reliab[:, j], float(f))
    _, _, _, rel, freq = red.finalise()
    assert np.all(np.isfinite(rel))
    assert freq.tolist() == [pytest.approx(7.1), 21.0]
