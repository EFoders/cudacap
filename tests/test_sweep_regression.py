"""The six accuracy-sweep outlier decks, end to end, on sweep-sized bounds.

``golden/dumps/luffy_sweep/`` holds six decks picked from the 2,200-deck
randomised accuracy sweep (docs/accuracy_sweep.md, seed 20261115): sw0112,
sw1539, sw0008, sw1704, sw1714 and sw0759.  Each is the raw ``.dat`` deck plus
voacapl's LUFFY capture (``_in.csv`` entry state, ``_out.csv`` /SON/ exit
state, ``_ant.csv`` /cantenna/).  The README in that directory says what each
one exercises.  Every one of them breaks a CI bound of
``tests/test_engine.py`` (``SON_ABS_BOUND``) or a hop count, and every
excursion was traced to the float32 libm floor -- glibc's ``cosf`` / ``sinf``
/ ``acosf`` round differently from cudacap's correctly rounded host math, and
GEOM's or FOBBY's conditioning amplifies the 1-ULP difference.  None is a port
bug, so they are NOT in ``test_engine.ALL_DECKS``; they live here, with their
own bounds.

What is run
-----------
Each deck goes from its RAW cards to /SON/: ``compare_sweep.parse_deck``
reads the deck in DECRED's fixed columns, ``compare_sweep.build_deck`` builds
one FRESH circuit per hour sharing one REDMAP with the captured /cantenna/
(``test_engine.run_deck``'s configuration, i.e. what ``predict`` hands
``predict_circuits``), and LUFFY runs on ``cpu`` (the scalar reference) and
on ``cuda`` (the production kernel) when a device is usable.  The parsed card
values are cross-checked against the captured entry state first, so a
deck-parsing slip cannot hide in the comparison.  Nothing here needs the
``reference/`` tree: the coefficients are the package's own.

The bounds (``SWEEP_ABS_BOUND``)
--------------------------------
They come from docs/accuracy_sweep.md, "CI bounds over the sweep": the bound
set that covers the WHOLE 2,200-deck sweep at about 2x its worst cell.

    field                      sweep worst   bound      this file's worst deck
    dbw/sndb/dblos/snpr        0.098 dB      0.25 dB    sw0112 h7 24.55 MHz
    dbu                        0.113 dB      0.25 dB    sw1539 h22 2.07 MHz
    reliab                     4.6e-4        1e-3       sw0112 h7
    angle                      0.0188 deg    0.04 deg   sw0112 h7
    delay                      6.5e-3 ms     0.015 ms   sw0112 h7
    vhigh                      0.65 km       1.5 km     sw0112 h7

Compare ``test_engine.SON_ABS_BOUND`` (0.01 dB, 2e-4, 2e-3 deg, 3e-3 ms,
0.4 km), which is ~5x the worst over 22 hand-picked decks and holds at the
sweep's 99.9th percentile but not as its worst case.

Hop counts must be exact, EXCEPT the one documented discrete flip of the sweep
(``NHP_FLIPS``): sw1704 (0.60N 80.81W to 72.03N 155.49E, 11,037 km, METHOD 20,
URSI), hours 8-11 at 15.12 MHz (/SON/ slot 2), where cudacap reports 1 hop and
voacapl 2.  GEOM's GCD is 1 ULP apart (``acosf``), so the long-path hop number
``HN = 0.5*GCD/(DR+DT)`` is 1.9999999 against 2.0 and RELBIL's ``INT(HN)``
truncates differently.  dbw agrees to 2e-3 dB there.  Those cells are pinned
to exactly those values; any other hop-count disagreement fails.

The regression pin (``SON_DIGEST``)
-----------------------------------
On top of the oracle bounds, cudacap's CURRENT output for each deck is pinned:
a SHA-256 over the float32 bytes of all 19 numeric /SON/ fields (``ALL_SON``)
plus ``nhp`` and the status vector, checked on both backends (the CPU backend
equals CUDA bit for bit, also asserted directly).  A digest change means the
numbers moved.  That may be a fix -- a glibc libm emulation (docs/f32_libm.md)
would move every one of these decks -- but it has to be a deliberate one:
rerun with ``-s`` to print the new digests and update the table with the
reason.  The digests were recorded on Windows (numpy's float64 libm in the host
setup chain).  A different host libm can legitimately move a last bit, so the
digest test asserts only on the platforms it was recorded for and skips,
printing the digest, elsewhere; the oracle-bound test runs everywhere.
"""

import functools
import hashlib
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from voacap_gpu.engine import predict_circuits

from tests.test_engine import _SON_NUM, _antennas, _need, _rows

REPO = pathlib.Path(__file__).resolve().parents[1]
SWEEP = REPO / "golden" / "dumps" / "luffy_sweep"


def _load_compare_sweep():
    """``tools/oracle/sweep/compare_sweep.py`` as a module (not a package)."""
    name = "_voacap_compare_sweep"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "tools" / "oracle" / "sweep" / "compare_sweep.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cs = _load_compare_sweep()

DECKS = ("sw0112", "sw1539", "sw0008", "sw1704", "sw1714", "sw0759")

# docs/accuracy_sweep.md, "CI bounds over the sweep": ~2x the sweep's worst
SWEEP_ABS_BOUND = {
    "dbw": 0.25, "sndb": 0.25, "dbu": 0.25, "dblos": 0.25, "snpr": 0.25,
    "reliab": 1e-3, "angle": 0.04, "delay": 0.015, "vhigh": 1.5,
}
assert set(SWEEP_ABS_BOUND) == set(_SON_NUM)

# The documented INT(HN) flip: {deck: {(hour, slot): (cudacap nhp, oracle nhp)}}
# hour is 1..24, slot is the 1-based /SON/ frequency slot (sw1704 slot 2 =
# 15.12 MHz).  Every other cell's hop count must agree exactly.
NHP_FLIPS = {
    "sw1704": {(8, 2): (1, 2), (9, 2): (1, 2), (10, 2): (1, 2),
               (11, 2): (1, 2)},
}

# SHA-256 of cudacap's /SON/ output per deck (see _digest), recorded on the
# platforms listed; cpu and cuda give the same digest.  Recorded 2026-11-15 at
# 41e06e8 (Windows 11, numpy from uv.lock, RTX 5080).
SON_DIGEST = {
    "win32": {
        "sw0112":
            "ee8f11ea2a9d88655fbfedce3d5d99216c22134675123a5cc37e86c8ab2d3925",
        "sw1539":
            "77335665a79491b9e6b4a56d24fa36824dd9c04272be7c38efb5d850ff6766ce",
        "sw0008":
            "11a6330f38132722a397bdb20ccc86a94b75c6182209a8a2b8447f4f35be2ac6",
        "sw1704":
            "12a6536e12ff0d568eb5d2cf78493eba212335b6bc6befe37a83dbb543b92f56",
        "sw1714":
            "d0e80a1fb72946ffdd6c024add8177c3f0dda196f589cdd72ffeb40799bc6c58",
        "sw0759":
            "741a15da3580ed2bbcc535be498657ed5d6f3fd0ca48129e69ea841f2a1710cb",
    },
}

_BACKENDS = ("cpu", "cuda")


def _deck(deck):
    return cs.parse_deck(SWEEP / f"{deck}.dat")


@functools.lru_cache(maxsize=None)
def run_sweep_deck(deck, backend):
    """One sweep deck, raw cards -> /SON/; cached across tests.

    Returns ``(son, status)``: ``son`` maps every ``ALL_SON`` field and
    ``nhp`` to an ``(24, nf)`` array in /SON/ slot order.
    """
    d = _deck(deck)
    cant = _antennas(_rows(SWEEP / f"{deck}_ant.csv"))
    circuits, ipflags = cs.build_deck(d, cant)
    _, status = predict_circuits(circuits, ipflags, backend=backend)
    son = cs.son_arrays(circuits, len(d["freqs"]))
    son["it"] = np.array([int(c.time.it) for c in circuits], np.int32)
    return son, np.asarray(status, np.int32)


def _digest(son, status):
    """SHA-256 over the float32 /SON/ fields, nhp and status, in fixed order."""
    h = hashlib.sha256()
    for nm in cs.ALL_SON:
        h.update(nm.encode())
        h.update(np.ascontiguousarray(son[nm], "<f4").tobytes())
    h.update(b"nhp")
    h.update(np.ascontiguousarray(son["nhp"], "<i4").tobytes())
    h.update(b"status")
    h.update(np.ascontiguousarray(status, "<i4").tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("deck", DECKS)
def test_sweep_deck_cards_match_capture(deck):
    """The raw-card parse agrees with voacapl's captured LUFFY entry state."""
    ins = _rows(SWEEP / f"{deck}_in.csv")
    assert cs.cross_check(_deck(deck), ins[0], ins) == []


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("deck", DECKS)
def test_sweep_deck_vs_oracle(deck, backend):
    """Raw cards -> LUFFY vs the captured /SON/, on the sweep-sized bounds."""
    _need(backend)
    d = _deck(deck)
    nf = len(d["freqs"])
    outs = _rows(SWEEP / f"{deck}_out.csv")
    son, status = run_sweep_deck(deck, backend)
    assert status.tolist() == [0] * 24, f"{deck}: status {status}"
    assert son["it"].tolist() == list(range(1, 25))
    o = cs.oracle_arrays(outs, nf)
    assert son["nhp"].shape == o["nhp"].shape == (24, nf)

    worst = {}
    for nm in _SON_NUM:
        a = np.abs(son[nm].astype(np.float64) - o[nm].astype(np.float64))
        h, j = np.unravel_index(int(np.argmax(a)), a.shape)
        worst[nm] = float(a[h, j])
        assert a.max() <= SWEEP_ABS_BOUND[nm], (
            f"{deck} [{backend}] {nm} h{h + 1} slot{j + 1}: "
            f"{son[nm][h, j]!r} vs {o[nm][h, j]!r}, |d| {a.max():.3e} > "
            f"{SWEEP_ABS_BOUND[nm]:g}")

    flips = NHP_FLIPS.get(deck, {})
    for (hour, slot), (got_exp, ora_exp) in flips.items():
        got = int(son["nhp"][hour - 1, slot - 1])
        ora = int(o["nhp"][hour - 1, slot - 1])
        assert (got, ora) == (got_exp, ora_exp), (
            f"{deck} [{backend}] pinned INT(HN) flip h{hour} slot{slot}: "
            f"cudacap/voacapl nhp {got}/{ora}, pinned {got_exp}/{ora_exp}")
    differ = {(int(h) + 1, int(j) + 1)
              for h, j in np.argwhere(son["nhp"] != o["nhp"])}
    assert differ == set(flips), (
        f"{deck} [{backend}]: hop counts differ at {sorted(differ)}, only "
        f"{sorted(flips)} are documented")
    print(f"\n{deck} [{backend}]: worst |d| "
          + ", ".join(f"{nm} {worst[nm]:.2e}" for nm in _SON_NUM))


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("deck", DECKS)
def test_sweep_deck_digest(deck, backend):
    """cudacap's current /SON/ for the deck has not drifted (SON_DIGEST)."""
    _need(backend)
    got = _digest(*run_sweep_deck(deck, backend))
    print(f"\n{deck} [{backend}] {sys.platform} SON digest {got}")
    pinned = SON_DIGEST.get(sys.platform)
    if pinned is None:
        pytest.skip(f"no SON digests recorded for {sys.platform}; got {got}")
    assert got == pinned[deck], (
        f"{deck} [{backend}]: /SON/ digest {got} != pinned {pinned[deck]}. "
        "cudacap's output for this deck changed; if deliberate, update "
        "SON_DIGEST with the reason")


@pytest.mark.parametrize("deck", DECKS)
def test_sweep_deck_cpu_equals_cuda(deck):
    """The CPU backend equals CUDA bit for bit on every /SON/ field."""
    _need("cuda")
    cpu, st_cpu = run_sweep_deck(deck, "cpu")
    gpu, st_gpu = run_sweep_deck(deck, "cuda")
    assert np.array_equal(st_cpu, st_gpu)
    for nm in cs.ALL_SON:
        assert np.array_equal(cpu[nm].view(np.uint32), gpu[nm].view(np.uint32)), (
            f"{deck}: {nm} differs between cpu and cuda")
    assert np.array_equal(cpu["nhp"], gpu["nhp"])
