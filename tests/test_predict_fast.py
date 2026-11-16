"""``predict(backend='cuda')``'s device-setup path against the host-build path.

``predict`` on CUDA no longer runs ``build_circuit`` per hour: the device
builds every hour's circuit from 32-byte parameter rows and the
warp-per-circuit-hour LUFFY kernel runs on it in place (engine, THE CUDA
PATH).  The host-build path it replaced -- ``build_circuit`` per hour, pack,
the thread-per-circuit kernel, unpack, ``_assemble`` -- is still in the
endpoint (``_predict_host``; the CPU backend's path).  These tests reach it on
CUDA by replacing predict's routing predicate (``endpoint._device_path_ok``)
to get the pre-device-path predict() as the reference, and require EVERY
Results field to be identical, bitwise for the floats (NaN patterns
included), on:

* the raw inputs (CIRCUIT card, SYSTEM, METHOD, coefficient set, month, SSN,
  frequencies, hours) of every captured LUFFY oracle deck -- the same decks
  ``tests/test_engine.py``'s all-deck tests run through ``build_circuit`` +
  ``predict_circuits``;
* random circuits mixing type-13 / type-14 / isotropic antennas from
  ``tests/data/antennas``, beam bearings, ICEPAC mode, long path, METHOD
  20/25/30, gain offsets, a non-default pole, hour subsets and wrapped
  longitudes.

Also here: the fast path is really taken (the host path is booby-trapped
while it runs), the cached rig is safe to share between threads, and the
public ``coeffs_dir()``.
"""

import threading
import warnings

import numpy as np
import pytest

from voacap_gpu import endpoint, engine
from voacap_gpu.engine import System, predict

from tests.test_engine import ALL_DECKS, _deck_inputs

ANT = "tests/data/antennas"
FIELDS = ("hours", "freqs", "muf", "fot", "hpf", "luf", "snr", "dbw",
          "reliab", "field_dbu", "nhp", "mode", "status", "valid")


def _require_cuda():
    if not engine._cuda_usable():
        pytest.skip("no usable CUDA device")


def assert_results_identical(got, want, what=""):
    """Every Results field equal; floats compared as bit patterns."""
    for name in FIELDS:
        g, w = getattr(got, name), getattr(want, name)
        assert g.dtype == w.dtype, (what, name, g.dtype, w.dtype)
        assert g.shape == w.shape, (what, name, g.shape, w.shape)
        if g.dtype.kind == "f":
            bad = g.view(np.uint32) != w.view(np.uint32)
        else:
            bad = g != w
        if bad.any():
            idx = np.argwhere(bad)[:5].tolist()
            raise AssertionError(
                f"{what}: Results.{name} differs at {int(bad.sum())} cells, "
                f"first {idx}: got {g[bad][:5]!r} want {w[bad][:5]!r}")


def _host_predict(monkeypatch, *args, **kwargs):
    """``predict`` on the HOST-build path: the routing predicate says "no
    device path", and the device path is booby-trapped so a silent use of it
    cannot pass for the host path."""
    def trap(*a, **k):
        raise AssertionError("predict took the device-setup path")

    with monkeypatch.context() as m:
        m.setattr(endpoint, "_device_path_ok", lambda long_path: False)
        m.setattr(endpoint, "_predict_device", trap)
        return predict(*args, **kwargs)


def _fast_predict(monkeypatch, *args, **kwargs):
    """``predict`` with the host path booby-trapped, so a silent fall-back
    to it cannot pass for the device path."""
    def trap(*a, **k):
        raise AssertionError("predict took the host-build path")

    with monkeypatch.context() as m:
        m.setattr(endpoint, "_predict_host", trap)
        assert endpoint._device_path_ok(False)
        return predict(*args, **kwargs)


def _both(monkeypatch, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = _host_predict(monkeypatch, *args, **kwargs)
        got = _fast_predict(monkeypatch, *args, **kwargs)
    return got, want


# ---------------------------------------------------------------------------
# every oracle deck's raw inputs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ALL_DECKS, reason="oracle dumps missing")
@pytest.mark.parametrize("deck", ALL_DECKS)
def test_deck_inputs_identical(deck, monkeypatch):
    _require_cuda()
    coords, system, method, coeff, freqs, ins = _deck_inputs(deck)
    v0 = ins[0]
    kw = dict(freqs=freqs, month=int(v0["month"]), ssn=np.float32(v0["ssn"]),
              hours=[int(v["it"]) for v in ins], system=system,
              method=method, coeff=coeff, long_path=bool(coords[4]))
    got, want = _both(monkeypatch, coords[:2], coords[2:4], **kw)
    assert_results_identical(got, want, deck)
    assert got.valid.any(), deck


# ---------------------------------------------------------------------------
# random circuits: antennas, ICEPAC, long path, METHOD 20/25/30, ...
# ---------------------------------------------------------------------------

_ANTS = (None, f"{ANT}/sample.13", f"{ANT}/sample.14", f"{ANT}/sample.14x")


def _random_case(rng):
    def pt():
        return (float(rng.uniform(-80, 80)), float(rng.uniform(-180, 180)))
    tx, rx = pt(), pt()
    if rng.random() < 0.15:                         # a longitude past 180
        rx = (rx[0], rx[1] + 360.0 * rng.choice([-1, 1]))
    kw = dict(freqs=sorted(rng.choice(np.arange(2.0, 30.0, 0.5),
                                      size=int(rng.integers(1, 12)),
                                      replace=False).tolist()),
              month=int(rng.integers(1, 13)),
              ssn=float(rng.choice([0.0, 15.0, 80.0, 150.0, 220.0])),
              method=int(rng.choice([20, 25, 30])),
              long_path=bool(rng.random() < 0.3))
    if rng.random() < 0.3:
        kw["model"] = "icepac"
    elif rng.random() < 0.2:
        kw["coeff"] = "URSI"
    if rng.random() < 0.6:
        kw["hours"] = sorted(rng.choice(np.arange(1, 25),
                                        size=int(rng.integers(1, 25)),
                                        replace=False).tolist())
    kw["tx_antenna"] = _ANTS[int(rng.integers(len(_ANTS)))]
    kw["rx_antenna"] = _ANTS[int(rng.integers(len(_ANTS)))]
    if rng.random() < 0.7:
        kw["tx_bearing"] = float(rng.uniform(0, 360))
    if rng.random() < 0.7:
        kw["rx_bearing"] = float(rng.uniform(0, 360))
    sys_kw = {}
    if rng.random() < 0.4:
        sys_kw.update(tx_gos=float(rng.uniform(-3, 3)),
                      rx_gos=float(rng.uniform(-3, 3)))
    if rng.random() < 0.4:
        sys_kw.update(required_snr=float(rng.uniform(40, 80)),
                      required_reliability=int(rng.integers(10, 99)),
                      man_made_noise=float(rng.choice([140.0, 145.0, 155.0,
                                                       -1.0, -3.0])),
                      power_kw=float(rng.choice([0.1, 1.0, 10.0, 100.0])),
                      min_takeoff_deg=float(rng.choice([0.1, 3.0, 5.0])))
    kw["system"] = System(**sys_kw)
    if rng.random() < 0.2:
        kw["pole"] = (79.5, -69.0)
    return tx, rx, kw


@pytest.mark.parametrize("seed", range(40))
def test_random_circuits_identical(seed, monkeypatch):
    _require_cuda()
    rng = np.random.default_rng(1000 + seed)
    tx, rx, kw = _random_case(rng)
    got, want = _both(monkeypatch, tx, rx, **kw)
    assert_results_identical(got, want, f"seed {seed}: {tx} {rx} {kw}")


@pytest.mark.parametrize("case", [
    dict(tx_antenna=f"{ANT}/sample.13", rx_antenna=f"{ANT}/sample.13",
         tx_bearing=300.0, rx_bearing=10.0, long_path=True, method=30),
    dict(tx_antenna=f"{ANT}/sample.13", model="icepac", method=25,
         tx_bearing=0.0),
    dict(rx_antenna=f"{ANT}/sample.14x", method=20, long_path=True,
         tx_antenna_freq_range=(3, 29), rx_antenna_freq_range=(2, 30)),
    dict(tx_antenna=f"{ANT}/sample.14", tx_table_absolute=True,
         rx_antenna=f"{ANT}/sample.13", rx_table_absolute=True,
         rx_bearing=123.4),
])
@pytest.mark.parametrize("path", [((51.50, -0.13), (-36.85, 174.76)),
                                  ((40.0, -105.0), (40.0, -70.0)),
                                  ((-33.87, 151.21), (64.80, -147.70))])
def test_antenna_cases_identical(case, path, monkeypatch):
    _require_cuda()
    freqs = [4.0, 7.1, 10.1, 14.2, 18.1, 21.2, 24.9, 28.3]
    if "tx_antenna_freq_range" in case:
        freqs = [4.0, 10.0, 21.0, 28.0]
    got, want = _both(monkeypatch, *path, freqs=freqs, month=10, ssn=110.0,
                      **case)
    assert_results_identical(got, want, f"{path} {case}")


def test_fast_math_is_kept_and_close(monkeypatch):
    """fast_math still works on the device path, and is still the opt-in
    NON-bit-exact libm.  On the device path it now covers the setup chain
    too (geometry / ionosphere / MUF are built on the device, with the same
    native-f32 libm, exactly as ``PipelinedSetupLuffy(fast_math=True)`` and
    ``predict_batch`` build them), where the host-build path built them on
    the host with the exact libm -- so the two fast_math answers are
    deliberately NOT compared bitwise.  The exact (default) answers are,
    everywhere else in this file."""
    _require_cuda()
    for path in (((51.50, -0.13), (48.85, 2.35)),
                 ((34.05, -118.24), (35.68, 139.77))):
        kw = dict(freqs=[5.0, 9.0, 15.0, 22.0], month=1, ssn=40.0)
        exact = predict(*path, **kw)
        fast = _fast_predict(monkeypatch, *path, fast_math=True, **kw)
        assert fast.valid.any()
        np.testing.assert_allclose(fast.muf, exact.muf, rtol=1e-4)
        np.testing.assert_allclose(fast.fot, exact.fot, rtol=1e-4)
        both = fast.valid & exact.valid
        assert both.mean() > 0.9
        np.testing.assert_allclose(fast.reliab[both], exact.reliab[both],
                                   atol=0.05)


def test_duplicate_and_unsorted_hours(monkeypatch):
    _require_cuda()
    got, want = _both(monkeypatch, (60.0, 25.0), (-1.3, 36.8),
                      freqs=[7.0, 14.0], month=4, ssn=60.0,
                      hours=[24, 1, 13, 13, 7])
    assert_results_identical(got, want)
    assert got.hours.tolist() == [24, 1, 13, 13, 7]


def test_more_hours_than_the_rig(monkeypatch):
    """More than 24 rows (duplicates) gets a wider rig, not truncation."""
    _require_cuda()
    hours = list(range(1, 25)) + [12, 6]
    got, want = _both(monkeypatch, (10.0, 10.0), (20.0, 40.0),
                      freqs=[10.0], month=8, ssn=100.0, hours=hours)
    assert_results_identical(got, want)


def test_errors_unchanged_on_the_device_path():
    """The device path raises where the host path did, with the same type."""
    _require_cuda()
    kw = dict(freqs=[7.0], month=6, ssn=100)
    with pytest.raises(NotImplementedError, match="METHOD"):
        predict((51.5, 0.0), (48.8, 2.3), method=9, **kw)
    with pytest.raises(NotImplementedError):
        predict((51.5, 0.0), (48.8, 2.3), tx_antenna=f"{ANT}/sample.11", **kw)
    with pytest.raises(ValueError, match="MINFREQ"):
        predict((51.5, 0.0), (48.8, 2.3), tx_antenna=f"{ANT}/sample.14",
                tx_antenna_freq_range=(10, 30), **kw)


def test_rig_is_safe_to_share_between_threads():
    """Concurrent predict() calls (same rig key, different circuits) give the
    sequential answers: the per-rig lock serialises submissions."""
    _require_cuda()
    rng = np.random.default_rng(7)
    cases = [((float(rng.uniform(-60, 60)), float(rng.uniform(-180, 180))),
              (float(rng.uniform(-60, 60)), float(rng.uniform(-180, 180))))
             for _ in range(12)]
    kw = dict(freqs=[3.5, 7.0, 14.0, 21.0], month=6, ssn=100.0)
    want = [predict(t, r, **kw) for t, r in cases]
    got = [None] * len(cases)
    errors = []

    def work(k):
        try:
            for _ in range(3):
                got[k] = predict(*cases[k], **kw)
        except Exception as e:                       # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work, args=(k,))
               for k in range(len(cases))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    for k in range(len(cases)):
        assert_results_identical(got[k], want[k], f"thread {k}")


def test_clear_device_cache_then_predict_again():
    _require_cuda()
    kw = dict(freqs=[7.0, 14.0], month=6, ssn=100.0)
    a = predict((51.5, -0.13), (40.7, -74.0), **kw)
    engine.clear_device_cache()
    assert len(engine._SMALL_RIGS) == 0
    b = predict((51.5, -0.13), (40.7, -74.0), **kw)
    assert_results_identical(a, b)


def test_coeffs_dir_is_public_and_matches_private():
    d = engine.coeffs_dir()
    assert d == engine._COEFFS_DIR
    assert d.is_dir()
    assert any(d.glob("coeff*.asc")) or any(d.iterdir())


def test_cpu_backend_unchanged():
    """The CPU backend is still the host build + scalar LUFFY: it never
    touches the device-setup path (which would import cupy)."""
    def trap(*a, **k):
        raise AssertionError("CPU backend took the device path")

    real = endpoint._predict_device
    endpoint._predict_device = trap
    try:
        r = predict((51.50, -0.13), (48.85, 2.35), freqs=[7.0], month=6,
                    ssn=100, hours=[12], backend="cpu")
    finally:
        endpoint._predict_device = real
    assert r.valid.all()
