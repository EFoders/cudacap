"""``predict(..., backend=...)``: the CPU backend (the scalar Stage-A LUFFY).

The claims:

* the CPU backend is BIT-IDENTICAL to the CUDA kernel -- every /SON/ field
  and hop count of every hour of all 22 captured oracle decks
  (``test_cpu_vs_cuda_all_decks``, reusing ``test_engine.run_deck``'s cached
  runs), and every ``Results`` array of ``predict`` on ordinary circuits;
* it never imports cupy (checked in a subprocess with cupy blocked);
* where the kernel returns a non-zero status the scalar code raises, and the
  CPU backend maps that back to the SAME status, so ``status`` / ``valid``
  read the same on both backends;
* ``'auto'`` falls back to CPU with a ``UserWarning``; ``fast_math`` and an
  unknown backend are refused.

The CUDA comparisons skip without cupy and a device; the rest always run.
"""

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

from voacap_gpu import engine
from voacap_gpu.engine import (STATUS_CPU_UNMAPPED,
                               STATUS_GENOIS_AREA_COVERAGE,
                               STATUS_GENOIS_GAIN_OOB, System, build_circuit,
                               predict, predict_circuits)
from voacap_gpu.scalar.antennastate import AREA_COVERAGE

from tests.test_engine import (ALL_DECKS, _SON_NUM, _require_cuda,
                               run_deck)

LONDON = (51.50, -0.13)
PARIS = (48.85, 2.35)              # 343 km: pure short path
TOKYO = (35.68, 139.77)
SYDNEY = (-33.87, 151.21)          # ~7,820 km: inside the smoothing band
BOULDER = (40.0, -105.0)
AUCKLAND = (-36.85, 174.76)

_RESULT_ARRAYS = ("muf", "fot", "hpf", "luf", "snr", "dbw", "reliab",
                  "field_dbu", "nhp", "status", "valid")


def _bits_equal(a, b):
    """Exact equality, NaN == NaN, -0.0 != 0.0 (compares the bit patterns)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == np.float32:
        return a.shape == b.shape and np.array_equal(a.view(np.int32),
                                                     b.view(np.int32))
    return np.array_equal(a, b)


def _assert_results_identical(a, b):
    for nm in _RESULT_ARRAYS:
        assert _bits_equal(a.array(nm), b.array(nm)), nm
    assert np.array_equal(a.mode.astype(str), b.mode.astype(str))


# ---------------------------------------------------------------------------
# Bit-identity with the CUDA kernel
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ALL_DECKS, reason="oracle dumps missing")
@pytest.mark.parametrize("deck", ALL_DECKS)
def test_cpu_vs_cuda_all_decks(deck):
    """Every /SON/ field + NHP, every hour, every frequency: same bits."""
    _require_cuda()
    cpu, st_cpu, _ = run_deck(deck, "cpu")
    gpu, st_gpu, _ = run_deck(deck, "cuda")
    assert st_cpu.tolist() == st_gpu.tolist()
    for nm in _SON_NUM + ("nhp",):
        assert _bits_equal(cpu[nm], gpu[nm]), (
            f"{deck} {nm}: CPU and CUDA differ at "
            f"{np.argwhere(cpu[nm] != gpu[nm])[:5].tolist()}")


@pytest.mark.parametrize("tx,rx,kw", [
    (TOKYO, SYDNEY, dict(freqs=[7.0, 10.0, 14.0, 21.0], hours=[3, 12, 20])),
    (LONDON, PARIS, dict(freqs=[3.5, 7.0, 14.0], hours=[1, 13])),
    (BOULDER, PARIS, dict(freqs=[5.0, 10.0, 14.0, 18.0, 28.0], hours=[6, 18],
                          system=System(power_kw=100.0))),
    (LONDON, AUCKLAND, dict(freqs=[10.0, 14.0, 21.0], hours=[8, 20],
                            long_path=True)),
    (BOULDER, PARIS, dict(freqs=[7.0, 14.0, 21.0], hours=[12],
                          model="icepac")),
], ids=["smoothing", "short", "transatlantic", "long_path", "icepac"])
def test_predict_cpu_vs_cuda(tx, rx, kw):
    """predict(backend='cpu') == predict(backend='cuda'), every array."""
    _require_cuda()
    kw = dict(month=7, ssn=100, **kw)
    cpu = predict(tx, rx, backend="cpu", **kw)
    gpu = predict(tx, rx, backend="cuda", **kw)
    assert np.all(cpu.status == 0) and cpu.valid.any()
    _assert_results_identical(cpu, gpu)


def test_predict_cpu_sane():
    """Without a GPU, the CPU path alone still gives a sane table."""
    r = predict(LONDON, PARIS, freqs=[3.5, 7.0, 14.0, 28.0], month=6, ssn=100,
                hours=[1, 13], backend="cpu")
    assert r.snr.shape == (2, 4)
    assert np.all(r.status == 0)
    assert r.valid.all()
    assert np.all((r.reliab >= 0) & (r.reliab <= 1))
    assert np.all(r.nhp >= 1)


# ---------------------------------------------------------------------------
# No cupy on the CPU path
# ---------------------------------------------------------------------------

def test_cpu_backend_never_imports_cupy():
    """With cupy blocked (sys.modules['cupy'] = None, so any import of it
    raises ImportError), importing the engine and predicting on the CPU
    works, and 'auto' falls back to CPU with a warning."""
    code = textwrap.dedent("""
        import sys, warnings
        sys.modules["cupy"] = None
        from voacap_gpu import engine
        kw = dict(freqs=[7.0, 14.0], month=6, ssn=100, hours=[1, 13])
        r = engine.predict((51.50, -0.13), (48.85, 2.35), backend="cpu", **kw)
        assert r.status.tolist() == [0, 0], r.status
        assert r.valid.all()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = engine.predict((51.50, -0.13), (48.85, 2.35), backend="auto",
                               **kw)
        assert any(issubclass(x.category, UserWarning)
                   and "falling back" in str(x.message) for x in w), w
        assert (a.dbw.view("i4") == r.dbw.view("i4")).all()
        assert sys.modules["cupy"] is None
        bad = sorted(m for m in sys.modules
                     if m.startswith(("cupy.", "cupyx", "voacap_gpu.cuda")))
        assert not bad, bad
        print("NO-CUPY-OK")
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "NO-CUPY-OK" in p.stdout


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_KW = dict(freqs=[7.0, 14.0], month=6, ssn=100, hours=[12])


def test_fast_math_on_cpu_refused():
    with pytest.raises(ValueError, match="fast_math"):
        predict(LONDON, PARIS, backend="cpu", fast_math=True, **_KW)
    c, ip = build_circuit(LONDON, PARIS, 6, 100, 12, [7.0])
    with pytest.raises(ValueError, match="fast_math"):
        predict_circuits([c], [ip], fast_math=True, backend="cpu")


def test_unknown_backend_refused():
    with pytest.raises(ValueError, match="backend"):
        predict(LONDON, PARIS, backend="gpu", **_KW)


def test_auto_falls_back_with_warning(monkeypatch):
    monkeypatch.setattr(engine, "_cuda_usable", lambda: False)
    with pytest.warns(UserWarning, match="falling back.*1000x slower"):
        a = predict(LONDON, PARIS, backend="auto", **_KW)
    b = predict(LONDON, PARIS, backend="cpu", **_KW)
    _assert_results_identical(a, b)


def test_auto_fast_math_without_gpu_refused(monkeypatch):
    monkeypatch.setattr(engine, "_cuda_usable", lambda: False)
    with pytest.warns(UserWarning), pytest.raises(ValueError,
                                                   match="fast_math"):
        predict(LONDON, PARIS, backend="auto", fast_math=True, **_KW)


def test_auto_uses_cuda_when_usable(monkeypatch):
    """With a device, 'auto' is the CUDA path: no warning, and it goes
    through predict_circuits (which the CPU path never calls)."""
    _require_cuda()
    calls = []
    real = engine.predict_circuits

    def spy(circuits, ipflags, fast_math=False):
        calls.append(len(circuits))
        return real(circuits, ipflags, fast_math=fast_math)

    monkeypatch.setattr(engine, "predict_circuits", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predict(LONDON, PARIS, backend="auto", **_KW)
    assert calls == [1]


# ---------------------------------------------------------------------------
# Status: scalar exceptions -> the kernel's codes
# ---------------------------------------------------------------------------

def _direct(tx, rx, freqs, hours, backend, mutate=None):
    """build_circuit + predict_circuits, with an optional post-build mutation
    (to reach states predict()'s validation refuses)."""
    build_f = [f if 1.0 <= f <= 30.0 else 10.0 for f in freqs]
    circuits, ipflags = [], []
    for h in hours:
        c, ip = build_circuit(tx, rx, 7, 160, h, build_f)
        for j, f in enumerate(freqs, start=1):
            c.frq.frel[j] = np.float32(f)
        if mutate:
            mutate(c)
        circuits.append(c)
        ipflags.append(ip)
    _, status = predict_circuits(circuits, ipflags, backend=backend)
    return np.asarray(status)


def _area_coverage(c):
    c.cant.offazim[2] = AREA_COVERAGE          # the receive entry


_STATUS_CASES = {
    # GENOIS's receive GAIN lookup below 1 MHz: Stage A IndexError, kernel 302
    "sub_1mhz": (TOKYO, SYDNEY, [7.0, 14.0, 0.5], None,
                 STATUS_GENOIS_GAIN_OOB),
    # area-coverage receive antenna: Stage A NotImplementedError, kernel 303
    "area_coverage": (LONDON, PARIS, [7.0, 14.0], _area_coverage,
                      STATUS_GENOIS_AREA_COVERAGE),
}


@pytest.mark.parametrize("case", _STATUS_CASES)
def test_cpu_status_matches_kernel_code(case):
    tx, rx, freqs, mutate, code = _STATUS_CASES[case]
    st = _direct(tx, rx, freqs, [1, 12], "cpu", mutate)
    assert st.tolist() == [code, code], st


@pytest.mark.parametrize("case", _STATUS_CASES)
def test_cpu_status_equals_cuda_status(case):
    _require_cuda()
    tx, rx, freqs, mutate, _ = _STATUS_CASES[case]
    cpu = _direct(tx, rx, freqs, [1, 12], "cpu", mutate)
    gpu = _direct(tx, rx, freqs, [1, 12], "cuda", mutate)
    assert cpu.tolist() == gpu.tolist()


def test_cpu_refusal_reaches_results_through_predict(monkeypatch):
    """As test_results' GPU twin: status 303 for every hour, nothing valid."""
    real = engine.build_circuit

    def area_coverage(*a, **kw):
        c, ip = real(*a, **kw)
        c.cant.offazim[2] = AREA_COVERAGE
        return c, ip

    monkeypatch.setattr(engine, "build_circuit", area_coverage)
    r = predict(LONDON, PARIS, freqs=[7.0, 14.0], month=6, ssn=100,
                hours=[1, 13], backend="cpu")
    assert r.status.tolist() == [STATUS_GENOIS_AREA_COVERAGE] * 2
    assert not r.valid.any()
    assert np.all(np.isnan(r.snr)) and np.all(np.isnan(r.luf))


def test_unidentified_refusal_is_the_cpu_code(monkeypatch):
    """A refusal raised outside any identifiable scalar callee is the
    documented CPU-only code; other hours are unaffected."""
    import voacap_gpu.scalar.luffy as L
    real = L.luffy

    def flaky(c, ipflag):
        if c.time.it == 12:
            raise IndexError("synthetic")
        return real(c, ipflag)

    monkeypatch.setattr(L, "luffy", flaky)
    r = predict(LONDON, PARIS, freqs=[7.0], month=6, ssn=100, hours=[1, 12],
                backend="cpu")
    assert r.status.tolist() == [0, STATUS_CPU_UNMAPPED]
    assert r.valid[0].all() and not r.valid[1].any()


def test_programming_errors_propagate(monkeypatch):
    """Only refusal-type exceptions become a status; a TypeError is a bug."""
    import voacap_gpu.scalar.luffy as L

    def broken(c, ipflag):
        raise TypeError("bug")

    monkeypatch.setattr(L, "luffy", broken)
    with pytest.raises(TypeError, match="bug"):
        predict(LONDON, PARIS, backend="cpu", **_KW)
