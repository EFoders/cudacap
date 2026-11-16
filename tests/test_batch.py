"""``predict_batch``: N circuits in one device run, equal to ``predict``.

The contract is that ``predict_batch(...).circuit(i)`` IS ``predict`` for
circuit ``i`` -- every Results field, floats bitwise -- whatever the batch
size, the internal ordering, the chunking, the small-batch / pipelined
choice, or the antenna blocking.  These tests hold it to that on random
mixes of the inputs (one TX to many RX, explicit pairs, per-circuit bearings
and long-path flags, type-13 at both ends, ICEPAC, METHOD 20/25/30), plus
the vectorised type-13 /cantenna/ build against ``antcalc_cant`` itself, the
input validation, and the CPU backend (which runs without a GPU).
"""

import warnings

import numpy as np
import pytest

from voacap_gpu import batch as B
from voacap_gpu import engine
from voacap_gpu.batch import BatchResults, predict_batch
from voacap_gpu.engine import System, predict

from tests.test_predict_fast import assert_results_identical

ANT = "tests/data/antennas"


def _require_cuda():
    if not engine._cuda_usable():
        pytest.skip("no usable CUDA device")


def _check_all(r, txs, rxs, kw, per=None, every=1):
    """r.circuit(i) == predict(...) for every ``every``-th circuit."""
    per = per or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(0, len(r), every):
            k = dict(kw)
            for name, arr in per.items():
                v = arr[i]
                if name.endswith("bearing"):
                    v = None if v is None or np.isnan(v) else float(v)
                elif name == "long_path":
                    v = bool(v)
                k[name] = v
            want = predict(tuple(txs[i]), tuple(rxs[i]), **k)
            assert_results_identical(r.circuit(i), want, f"circuit {i}")


def _rx_points(rng, n):
    return np.c_[rng.uniform(-75, 75, n), rng.uniform(-180, 180, n)]


KW = dict(freqs=[3.5, 7.1, 10.1, 14.2, 21.2, 28.3], month=7, ssn=120.0)


# ---------------------------------------------------------------------------
# equality with predict
# ---------------------------------------------------------------------------

def test_one_tx_many_rx_equals_predict():
    _require_cuda()
    rng = np.random.default_rng(1)
    rx = _rx_points(rng, 40)
    tx = (45.42, -75.70)
    r = predict_batch(tx, rx, **KW)
    assert isinstance(r, BatchResults) and len(r) == 40
    assert r.snr.shape == (40, 24, 6) and r.muf.shape == (40, 24)
    assert r.mode.dtype == np.int8 and r.status.dtype == np.int32
    _check_all(r, np.repeat([tx], 40, 0), rx, KW)


@pytest.mark.parametrize("seed", range(6))
def test_random_pairs_equal_predict(seed):
    """Explicit TX/RX pairs with per-circuit bearings and long-path flags,
    type-13 / type-14 / isotropic at either end, ICEPAC and METHOD mixes."""
    _require_cuda()
    rng = np.random.default_rng(50 + seed)
    n = int(rng.integers(5, 60))
    txs, rxs = _rx_points(rng, n), _rx_points(rng, n)
    ants = (None, f"{ANT}/sample.13", f"{ANT}/sample.14", f"{ANT}/sample.14x")
    kw = dict(freqs=sorted(rng.choice(np.arange(2.0, 30.0, 0.5),
                                      size=int(rng.integers(1, 12)),
                                      replace=False).tolist()),
              month=int(rng.integers(1, 13)),
              ssn=float(rng.uniform(0, 200)),
              hours=sorted(rng.choice(np.arange(1, 25), size=int(
                  rng.integers(1, 25)), replace=False).tolist()),
              method=int(rng.choice([20, 25, 30])),
              tx_antenna=ants[int(rng.integers(4))],
              rx_antenna=ants[int(rng.integers(4))],
              system=System(tx_gos=float(rng.uniform(-2, 2)),
                            rx_gos=float(rng.uniform(-2, 2))))
    if seed % 2:
        kw["model"] = "icepac"
    txb = np.where(rng.random(n) < 0.3, np.nan, rng.uniform(0, 360, n))
    rxb = np.where(rng.random(n) < 0.3, np.nan, rng.uniform(0, 360, n))
    lp = rng.random(n) < 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = predict_batch(txs, rxs, tx_bearing=txb, rx_bearing=rxb,
                          long_path=lp, **kw)
    _check_all(r, txs, rxs, kw,
               per=dict(tx_bearing=txb, rx_bearing=rxb, long_path=lp))
    assert np.array_equal(r.long_path, lp)
    assert r.hours.tolist() == kw["hours"]


def test_type13_both_ends_pipelined_and_blocked(monkeypatch):
    """Type 13 at both ends (a /cantenna/ per circuit), large enough for the
    pipelined path, with CANT_BLOCK forced small so the batch runs in
    several antenna blocks and the chunks are ragged."""
    _require_cuda()
    rng = np.random.default_rng(9)
    n = 70
    rx = _rx_points(rng, n)
    tx = (51.5, -0.13)
    kw = dict(KW, tx_antenna=f"{ANT}/sample.13", rx_antenna=f"{ANT}/sample.13",
              tx_bearing=270.0, rx_bearing=33.0)
    # blocks of 50 circuits: 1,200 rows pipelined, then 480 on the warp rig
    monkeypatch.setattr(B, "CANT_BLOCK", 50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = predict_batch(tx, rx, chunk=256, nstreams=2, **kw)
    _check_all(r, np.repeat([tx], n, 0), rx, kw, every=3)


def test_order_and_chunking_do_not_change_answers():
    """Spatial ordering, caller ordering, small vs pipelined path and the
    chunk width are all invisible in the result."""
    _require_cuda()
    rng = np.random.default_rng(4)
    rx = _rx_points(rng, 120)                # 2,880 rows: pipelined
    tx = (-33.87, 151.21)
    a = predict_batch(tx, rx, order="spatial", **KW)
    b = predict_batch(tx, rx, order="none", chunk=512, nstreams=2, **KW)
    c = predict_batch(tx, rx[:30], hours=range(1, 25), **KW)   # 720: small
    for name in ("snr", "dbw", "field_dbu", "reliab", "nhp", "mode", "valid",
                 "muf", "fot", "hpf", "luf", "status"):
        x, y, z = getattr(a, name), getattr(b, name), getattr(c, name)
        if x.dtype.kind == "f":
            x, y, z = (v.view(np.uint32) for v in (x, y, z))
        assert np.array_equal(x, y), name
        assert np.array_equal(x[:30], z), name
    _check_all(a, np.repeat([tx], 120, 0), rx, KW, every=11)


def test_fast_math_matches_predict_fast_math():
    """With fast_math both run the native-f32 libm through setup AND LUFFY,
    so they agree bitwise with each other (not with the exact answer)."""
    _require_cuda()
    rng = np.random.default_rng(12)
    rx = _rx_points(rng, 6)
    tx = (35.0, 139.0)
    r = predict_batch(tx, rx, fast_math=True, **KW)
    for i in range(6):
        assert_results_identical(
            r.circuit(i), predict(tx, tuple(rx[i]), fast_math=True, **KW),
            f"circuit {i}")


# ---------------------------------------------------------------------------
# the vectorised type-13 /cantenna/ build
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("gos", [0.0, 1.75])
def test_type13_tables_equal_antcalc(absolute, gos):
    """``_Antennas.tables`` lays out exactly what ``shared_antenna_tables``
    makes of ``antcalc_cant`` per circuit (array, aeff, OFFAZIM, ...), for
    paths whose OFFAZIM hits 0, the 359/0 wrap and fractional slices."""
    from voacap_gpu.cuda.driver import shared_antenna_tables
    from voacap_gpu.scalar.readant import read_antenna

    a13 = read_antenna(f"{ANT}/sample.13")
    a14 = read_antenna(f"{ANT}/sample.14")
    rng = np.random.default_rng(3)
    txs = np.c_[rng.uniform(-60, 60, 25), rng.uniform(-180, 180, 25)]
    rxs = np.c_[rng.uniform(-60, 60, 25), rng.uniform(-180, 180, 25)]
    txs[0], rxs[0] = (0.0, 0.0), (10.0, 0.0)          # due north: ztaz 0
    lp = rng.random(25) < 0.4
    txb = rng.uniform(0, 360, 25)
    txb[0] = 0.0
    txb[1] = np.nan
    rxb = np.full(25, np.nan)
    system = System(tx_gos=gos, rx_gos=-gos)
    kw = dict(tx_freq_range=(2, 30), rx_freq_range=(3, 28),
              tx_table_absolute=absolute, rx_table_absolute=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ants = B._Antennas(a13, a14, txs, rxs, lp, txb, rxb, system, kw,
                           np.array([5.0, 10.0], np.float32))
        want_cants = [engine.antcalc_cant(
            a13, a14, tuple(txs[i]), tuple(rxs[i]), power_kw=1.0,
            tx_gos=gos, rx_gos=-gos,
            tx_bearing=None if np.isnan(txb[i]) else float(txb[i]),
            rx_bearing=None, long_path=bool(lp[i]), **kw)
            for i in range(25)]
    got = ants.tables(ants.set_of)                  # one set per circuit
    # type-14 RX end: OFFAZIM is inert (area-coverage sentinel only) and
    # the batch shares the template's; compare it on the type-13 end only
    want = shared_antenna_tables(want_cants)
    for k in ("array", "aeff", "iats", "xfqs", "xfqe", "pwrdba"):
        assert np.array_equal(np.asarray(got[k]).view(np.uint32)
                              if np.asarray(got[k]).dtype == np.float32
                              else got[k],
                              np.asarray(want[k]).view(np.uint32)
                              if np.asarray(want[k]).dtype == np.float32
                              else want[k]), k
    off_g = np.asarray(got["offazim"]).reshape(25, 3)
    off_w = np.asarray(want["offazim"]).reshape(25, 3)
    assert np.array_equal(off_g[:, 1].view(np.uint32),
                          off_w[:, 1].view(np.uint32))
    assert got["numants"] == want["numants"]


# ---------------------------------------------------------------------------
# validation and the CPU backend (no GPU needed)
# ---------------------------------------------------------------------------

def test_bad_point_is_named_with_its_index():
    rx = np.array([[10.0, 10.0], [95.0, 0.0]])
    with pytest.raises(ValueError, match=r"rx\[1\] latitude"):
        predict_batch((0.0, 0.0), rx, backend="cpu", **KW)
    with pytest.raises(ValueError, match="finite"):
        predict_batch((0.0, np.nan), rx[:1], backend="cpu", **KW)


def test_shape_and_argument_errors():
    with pytest.raises(ValueError, match="same length"):
        predict_batch(np.zeros((3, 2)), np.zeros((2, 2)), backend="cpu", **KW)
    with pytest.raises(ValueError, match="long_path"):
        predict_batch((0, 0), (10, 10), long_path=[0, 1], backend="cpu", **KW)
    with pytest.raises(ValueError, match="order"):
        predict_batch((0, 0), (10, 10), order="random", backend="cpu", **KW)
    with pytest.raises(NotImplementedError):
        predict_batch((0, 0), (10, 10), freqs=list(range(2, 14)), month=1,
                      ssn=10, backend="cpu")
    with pytest.raises(NotImplementedError):
        predict_batch((0, 0), (10, 10), tx_antenna=f"{ANT}/sample.11",
                      backend="cpu", **KW)
    with pytest.raises(ValueError, match="month"):
        predict_batch((0, 0), (10, 10), freqs=[7.0], month=13, ssn=10,
                      backend="cpu")


def test_longitudes_are_wrapped_like_predict():
    r = predict_batch((10.0, 190.0), [(20.0, -200.0), (0.0, 180.0)],
                      freqs=[7.0], month=3, ssn=50, hours=[6],
                      backend="cpu")
    assert r.tx[0].tolist() == [10.0, -170.0]
    assert r.rx.tolist() == [[20.0, 160.0], [0.0, 180.0]]


def test_cpu_backend_equals_predict_cpu():
    rx = [(48.85, 2.35), (40.7, -74.0)]
    kw = dict(freqs=[7.0, 14.0], month=6, ssn=100.0, hours=[1, 13])
    r = predict_batch((51.50, -0.13), rx, backend="cpu", **kw)
    for i in range(2):
        assert_results_identical(
            r.circuit(i), predict((51.50, -0.13), rx[i], backend="cpu", **kw),
            f"cpu circuit {i}")
    assert r.mode_labels().shape == (2, 2, 2)


def test_cuda_equals_cpu():
    _require_cuda()
    rx = [(48.85, 2.35), (-36.85, 174.76)]
    kw = dict(freqs=[7.0, 14.0, 21.0], month=11, ssn=60.0, hours=[3, 15],
              tx_antenna=f"{ANT}/sample.13", tx_bearing=90.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = predict_batch((51.50, -0.13), rx, **kw)
        b = predict_batch((51.50, -0.13), rx, backend="cpu", **kw)
    for i in range(2):
        assert_results_identical(a.circuit(i), b.circuit(i), f"circuit {i}")


def test_engine_reexport():
    assert engine.predict_batch.__doc__
    r = engine.predict_batch((0.0, 0.0), (5.0, 5.0), freqs=[10.0], month=1,
                             ssn=0, hours=[12], backend="cpu")
    assert r.snr.shape == (1, 1, 1)
