"""The device grid reduction == ``gridjob.CellReducer`` == the Go, bit for bit.

``gridjob.DeviceCellReducer`` (cuda/kernels/gridreduce.cuh) moves PropServer's
``cellAgg`` onto the GPU so a grid sweep never ships its per-sample results to
the host.  The reduction is ORDER-DEPENDENT -- a new best restarts the hourly
accumulator, ties keep the incumbent, only later samples at the winning
frequency average -- so the risks are the same ones ``tests/test_gridjob.py``
guards for the host port, plus the device's own: chunks reaching the kernel
out of order (they run on different streams), a cell appearing twice in one
chunk, float64 sums accumulated in a different order or contracted into an
FMA, and the sanitiser/scale applied in the wrong place.

So the device answer is compared, with ``np.array_equal`` on float64 (no
tolerance anywhere), against

* ``CellReducer`` fed exactly the way the Prop-SL service feeds it
  (``gridjob.feed_chunk_host``), and
* the LITERAL statement-by-statement Go transcription in
  ``tests/test_gridjob.py`` (``GoCellAgg`` / ``go_reduce``),

on randomised sweeps that are built to produce ties (hard-quantised scores),
failed circuits, repeated winning frequencies, NaN/+-inf samples, cells seen
twice in one delivery and deliveries that split (hour, frequency-chunk)
blocks -- and then end to end on real LUFFY output from the pipelined rig.
"""

import types

import numpy as np
import pytest

from voacap_gpu import engine, gridjob
from voacap_gpu.cuda.driver import (SETUP_CONFIG_DTYPE, IonoSlot,
                                       PipelinedSetupLuffy,
                                       shared_antenna_tables,
                                       shared_iono_tables)
from voacap_gpu.gridjob import (CellReducer, DeviceCellReducer,
                                   feed_chunk_host, freq_chunks)
from voacap_gpu.scalar.redmap import redmap

from tests.cuda.harness import requires_cuda
from tests.test_gridjob import go_reduce

pytestmark = requires_cuda

NAMES = ("snr", "dbw", "dbu", "reliab", "freq")
MODES = ("snr", "reliability", "fieldstrength", "signalpower")

# A synthetic "state" row: four 14-slot float32 columns at these word offsets,
# with junk words around them so a wrong offset or slot reads garbage.
STRIDE = 61
OFFS = (2, 17, 31, 46)                  # sndb, dbw, dbu, reliab -- slot 0


def _assert_same(got, exp, tag):
    for name, g, e in zip(NAMES, got, exp):
        g = np.asarray(g, np.float64)
        e = np.asarray(e, np.float64)
        same = (g.view(np.uint64) == e.view(np.uint64))
        assert same.all(), (
            f"{name} differs [{tag}]: {int((~same).sum())}/{g.size} cells; "
            f"first at {int(np.flatnonzero(~same)[0])}: "
            f"got {g[~same][0]!r} expected {e[~same][0]!r}")


# ---------------------------------------------------------------------------
# a randomised sweep, as float32 /SON/ words
# ---------------------------------------------------------------------------

def make_sweep(rng, n_cells, hours, freqs, fail_rate=0.15, coarse=True,
               nonfinite=0.0):
    """Circuit-order float32 samples in a synthetic word block.

    Returns ``(chunks, words, status, cols)`` where ``cols`` are the four
    ``(n_circ, 14)`` float32 /SON/-like columns (reliab as a 0..1 fraction,
    as LUFFY stores it).  ``coarse`` quantises every column hard so TIES
    happen in every rendering mode; ``nonfinite`` is the per-sample rate of
    NaN / +inf / -inf injected into a random one of the four values.
    """
    chunks = freq_chunks(freqs)
    n_circ = hours * len(chunks) * n_cells
    shape = (n_circ, 14)
    snr = rng.uniform(-40, 60, shape)
    dbw = rng.uniform(-180, -60, shape)
    dbu = rng.uniform(-20, 90, shape)
    rel = rng.uniform(0, 1, shape)
    if coarse:
        snr = np.round(snr / 8.0) * 8.0
        dbw = np.round(dbw / 30.0) * 30.0
        dbu = np.round(dbu / 25.0) * 25.0
        rel = np.round(rel / 0.2) * 0.2
    cols = [a.astype(np.float32) for a in (snr, dbw, dbu, rel)]
    if nonfinite:
        bad = rng.random(shape) < nonfinite
        which = rng.integers(0, 4, shape)
        kind = rng.choice(np.array([np.nan, np.inf, -np.inf], np.float32),
                          shape)
        for k in range(4):
            m = bad & (which == k)
            cols[k][m] = kind[m]
    status = np.where(rng.random(n_circ) < fail_rate,
                      rng.integers(1, 5, n_circ), 0).astype(np.int32)

    words = rng.integers(-2**31, 2**31 - 1, (n_circ, STRIDE),
                         dtype=np.int64).astype(np.int32)   # junk everywhere
    for k, off in enumerate(OFFS):
        words[:, off:off + 14] = cols[k].view(np.int32)
    return chunks, words, status, cols


def fake_results(cols, status, lo, hi):
    """A ``LuffyResults``-shaped object for rows [lo, hi) of the sweep."""
    son = np.zeros(hi - lo, [(k, np.float32, (14,))
                             for k in gridjob.GRID_OUTPUTS])
    for k, c in zip(gridjob.GRID_OUTPUTS, cols):
        son[k] = c[lo:hi]
    return types.SimpleNamespace(son=son, status=status[lo:hi])


def host_reduce(chunks, cols, status, n_cells, mode, monthly, span,
                sanitize=True, default_freq=14.0):
    red = CellReducer(n_cells, mode, monthly_median=monthly,
                      default_freq=default_freq)
    n = status.size
    for lo in range(0, n, span):
        hi = min(n, lo + span)
        feed_chunk_host(red, chunks, n_cells, lo, hi,
                        fake_results(cols, status, lo, hi), sanitize=sanitize)
    return red.finalise()


def device_reduce(chunks, words, status, n_cells, mode, monthly, span,
                  sanitize=True, default_freq=14.0):
    red = DeviceCellReducer(n_cells, chunks, mode, monthly_median=monthly,
                            default_freq=default_freq, sanitize=sanitize)
    n = status.size
    for lo in range(0, n, span):
        hi = min(n, lo + span)
        red.feed_words(words[lo:hi], status[lo:hi], OFFS)
    assert red.offset == n
    return red.finalise()


def go_samples(chunks, cols, status, n_cells, sanitize):
    """The sweep as the Go sees it: one (cell, ok, point) per sample."""
    snr, dbw, dbu, rel = (c.astype(np.float64) for c in cols)
    if sanitize:
        snr, dbw, dbu = (np.where(np.isfinite(a), a, -200.0)
                         for a in (snr, dbw, dbu))
        rel = np.where(np.isfinite(rel), rel, 0.0)
    rel = rel * 100.0
    out = []
    for g in range(status.size):
        b, cell = divmod(g, n_cells)
        for j, f in enumerate(chunks[b % len(chunks)]):
            k = 1 + j
            out.append((cell, status[g] == 0,
                        (snr[g, k], dbw[g, k], dbu[g, k], rel[g, k], f)))
    return out


# ---------------------------------------------------------------------------
# synthetic sweeps: device vs CellReducer vs the Go
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("hours,freqs,monthly", [
    (1, [14.0], False),                                   # single point
    (1, [3.0, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.0, 28.0], False),
    (4, [3.0, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.0, 28.0], True),
    (24, [7.0, 14.0, 21.0], True),                        # every hour
    (3, [float(2 + i) for i in range(29)], True),          # 3 freq-chunks
])
def test_device_matches_cellreducer_and_go(mode, hours, freqs, monthly):
    """Finite samples (sanitised): all three reductions agree exactly."""
    rng = np.random.default_rng(abs(hash((mode, hours, len(freqs)))) % 2**31)
    n_cells = 137
    chunks, words, status, cols = make_sweep(rng, n_cells, hours, freqs,
                                             nonfinite=0.03)
    tag = f"{mode} {hours}h {len(freqs)}f monthly={monthly}"

    exp_go = go_reduce(go_samples(chunks, cols, status, n_cells, True),
                       n_cells, mode, monthly, default_freq=14.0)
    exp = host_reduce(chunks, cols, status, n_cells, mode,
                      monthly, span=97)
    _assert_same(exp, exp_go, tag + " (host vs Go: the premise)")
    got = device_reduce(chunks, words, status, n_cells,
                        mode, monthly, span=97)
    _assert_same(got, exp, tag)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("monthly", [False, True])
def test_nonfinite_samples_are_skipped_like_cellreducer(mode, monthly):
    """sanitize=False: raw NaN/+-inf reach the scan and must be DROPPED.

    The Go never saw a NaN, so the reference here is CellReducer's own
    extension; on the finite subset the Go is checked too (a sweep with the
    non-finite samples marked not-ok is a Go-legal sweep).
    """
    rng = np.random.default_rng(7 + monthly + 10 * MODES.index(mode))
    n_cells = 113
    freqs = [3.0, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.0, 28.0, 30.0, 4.0,
             6.0]                                         # 2 freq-chunks
    chunks, words, status, cols = make_sweep(rng, n_cells, 5, freqs,
                                             nonfinite=0.2)
    exp = host_reduce(chunks, cols, status, n_cells, mode,
                      monthly, span=250, sanitize=False)
    got = device_reduce(chunks, words, status, n_cells,
                        mode, monthly, span=250,
                        sanitize=False)
    _assert_same(got, exp, f"raw non-finite {mode} monthly={monthly}")

    samples = go_samples(chunks, cols, status, n_cells, False)
    samples = [(c, ok and all(np.isfinite(pt[:4])), pt)
               for c, ok, pt in samples]
    exp_go = go_reduce(samples, n_cells, mode, monthly, default_freq=14.0)
    _assert_same(got, exp_go, f"raw non-finite vs Go {mode}")


def test_delivery_span_does_not_change_the_answer():
    """Spans that split blocks, and spans holding a cell 2-3 times."""
    rng = np.random.default_rng(4242)
    n_cells, hours = 91, 6
    freqs = [3.0, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.0, 28.0]
    chunks, words, status, cols = make_sweep(rng, n_cells, hours, freqs,
                                             nonfinite=0.05)
    ref = go_reduce(go_samples(chunks, cols, status, n_cells, True), n_cells,
                    "snr", True, 14.0)
    for span in (1, 7, n_cells - 1, n_cells, n_cells + 1, 2 * n_cells + 5,
                 10_000):
        got = device_reduce(chunks, words, status, n_cells, "snr", True, span)
        _assert_same(got, ref, f"span {span}")


def test_order_dependence_is_reproduced():
    """The device must NOT compute the order-free answer.

    Reversing the hours changes PropServer's answer (the accumulator reset
    makes the mean depend on what came after the winner).  The device has to
    change with it, identically.
    """
    rng = np.random.default_rng(99)
    n_cells, hours = 64, 8
    freqs = [7.0, 14.0, 21.0]
    chunks, words, status, cols = make_sweep(rng, n_cells, hours, freqs)
    blk = n_cells * len(chunks)
    rev = np.concatenate([np.arange(h * blk, (h + 1) * blk)
                          for h in reversed(range(hours))])
    a = device_reduce(chunks, words, status, n_cells, "snr", True, 50)
    b = device_reduce(chunks, words[rev], status[rev], n_cells, "snr", True,
                      50)
    cols_r = [c[rev] for c in cols]
    _assert_same(b, host_reduce(chunks, cols_r, status[rev], n_cells, "snr",
                                True, 50), "reversed hours")
    assert any(not np.array_equal(x, y) for x, y in zip(a, b)), (
        "reversing the hours changed nothing: the sweep is not exercising "
        "the order-dependent path")


def test_ties_failures_and_sentinels_by_hand():
    """The four rules on a hand-built 3-cell, 2-hour, 2-frequency sweep."""
    n_cells, chunks = 3, [[7.0, 14.0]]
    # rows: hour 1 cells 0..2, then hour 2 cells 0..2
    snr = np.full((6, 14), np.nan, np.float32)
    rel = np.zeros((6, 14), np.float32)
    snr[0, 1:3] = (10.0, 10.0)      # cell 0 h1: tie -> 7 MHz (earliest) wins
    snr[3, 1:3] = (4.0, 10.0)       # cell 0 h2: 7 MHz 4 is not the winner's...
    snr[1, 1:3] = (20.0, 30.0)      # cell 1 h1: 14 MHz wins at 30
    snr[4, 1:3] = (99.0, 99.0)      # cell 1 h2: FAILED circuit -> ignored
    snr[2, 1:3] = (5.0, 5.0)        # cell 2 h1: failed
    snr[5, 1:3] = (5.0, 5.0)        # cell 2 h2: failed -> sentinels
    status = np.array([0, 0, 2, 0, 1, 3], np.int32)
    dbw = np.full_like(snr, -100.0)
    dbu = np.full_like(snr, 20.0)
    words = np.zeros((6, STRIDE), np.int32)
    for k, c in zip(OFFS, (snr, dbw, dbu, rel)):
        words[:, k:k + 14] = c.view(np.int32)
    red = DeviceCellReducer(n_cells, chunks, "snr", monthly_median=True,
                            default_freq=9.5)
    red.feed_words(words, status, OFFS)
    s, w, u, r, f = red.finalise()
    # cell 0: winner 7 MHz @10 (tie kept the incumbent), then h2 7 MHz @4
    # averages in: (10 + 4) / 2 = 7; 14 MHz @10 never displaces (not > 10).
    assert (f[0], s[0]) == (7.0, 7.0)
    assert (f[1], s[1]) == (14.0, 30.0)
    assert (s[2], w[2], u[2], r[2], f[2]) == (-200.0, -200.0, -200.0, 0.0,
                                              9.5)


# ---------------------------------------------------------------------------
# end to end: real LUFFY output through the pipelined rig
# ---------------------------------------------------------------------------

FREQS13 = [3.5, 5.0, 7.0, 10.0, 12.0, 14.0, 16.0, 18.0, 21.0, 24.9, 28.0,
           6.0, 9.0]                                  # 2 freq-chunks: 11 + 2


@pytest.fixture(scope="module")
def sweep():
    lat, lon, _ls, _os = gridjob.grid_cells(-60.0, 60.0, -180.0, 180.0, 20.0)
    n_cells = lat.size
    hours = [1, 7, 13, 19]
    chunks = freq_chunks(FREQS13)
    slot = IonoSlot(redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(120.0)),
                    7, 120.0)
    iono = shared_iono_tables([slot.iono])
    cfgs = np.zeros(len(chunks), SETUP_CONFIG_DTYPE)
    for j, ch in enumerate(chunks):
        cfgs[j] = engine.setup_config(engine.System(), method=30, freqs=ch)
    ant = shared_antenna_tables([engine.isotropic_cant(power_kw=1.0)])
    params = gridjob.sweep_params((45.42, -75.70), lat, lon, hours,
                                  len(chunks))
    ant_idx = np.zeros(len(params), np.int32)
    return dict(n_cells=n_cells, chunks=chunks, slot=slot, iono=iono,
                cfgs=cfgs, ant=ant, params=params, ant_idx=ant_idx,
                hours=hours)


@pytest.mark.parametrize("chunk,ns", [(300, 3), (4096, 2), (97, 1),
                                      (1000, 2)])
def test_end_to_end_matches_the_service_host_path(sweep, chunk, ns):
    """run_grid(device=True) == the service's on_chunk + CellReducer."""
    s = sweep
    rig = PipelinedSetupLuffy(chunk, ns, [s["slot"]], s["iono"])
    args = (rig, s["params"], s["cfgs"], s["ant"], s["ant_idx"], s["n_cells"],
            s["chunks"])
    for mode in ("snr", "reliability"):
        exp = gridjob.run_grid(*args, rendering_mode=mode,
                               monthly_median=True, default_freq=14.0,
                               device=False, host_outputs=None)   # FULL
        got = gridjob.run_grid(*args, rendering_mode=mode,
                               monthly_median=True, default_freq=14.0,
                               device=True)
        _assert_same(got, exp, f"e2e {mode} chunk={chunk} ns={ns}")
        slim = gridjob.run_grid(*args, rendering_mode=mode,
                                monthly_median=True, default_freq=14.0,
                                device=False)                     # slim
        _assert_same(slim, exp, f"e2e slim-host {mode} chunk={chunk}")
    # the sweep is not trivial: averaging actually happened somewhere
    red = DeviceCellReducer(s["n_cells"], s["chunks"], "snr",
                            monthly_median=True)
    rig.run(s["params"], s["cfgs"], s["ant"], s["ant_idx"], reduce=red)
    assert int(red.count.max()) > 1


def test_end_to_end_split_runs_and_progress(sweep):
    """One reducer across several runs (by hour) == one run; progress fires."""
    s = sweep
    rig = PipelinedSetupLuffy(512, 2, [s["slot"]], s["iono"])
    whole = gridjob.run_grid(rig, s["params"], s["cfgs"], s["ant"],
                             s["ant_idx"], s["n_cells"], s["chunks"],
                             "fieldstrength", monthly_median=True,
                             device=False, host_outputs=None)
    red = DeviceCellReducer(s["n_cells"], s["chunks"], "fieldstrength",
                            monthly_median=True)
    per_hour = s["n_cells"] * len(s["chunks"])
    seen = []
    for h in range(len(s["hours"])):
        sl = slice(h * per_hour, (h + 1) * per_hour)
        rig.run(s["params"][sl], s["cfgs"], s["ant"], s["ant_idx"][sl],
                reduce=red, on_chunk=lambda lo, hi, r, m: seen.append(
                    (lo, hi, r, m)))
    assert red.offset == len(s["params"])
    _assert_same(red.finalise(), whole, "split by hour")
    assert all(r is None and m is None for _lo, _hi, r, m in seen)

    ticks = []
    gridjob.run_grid(rig, s["params"], s["cfgs"], s["ant"], s["ant_idx"],
                     s["n_cells"], s["chunks"], device=True,
                     on_progress=lambda d, t: ticks.append((d, t)))
    assert ticks[-1] == (len(s["params"]), len(s["params"]))
    assert [d for d, _ in ticks] == sorted(d for d, _ in ticks)


def test_end_to_end_reduce_with_slim_rows(sweep):
    """reduce= and outputs= together: the rows still arrive, and agree."""
    s = sweep
    rig = PipelinedSetupLuffy(700, 3, [s["slot"]], s["iono"])
    red = DeviceCellReducer(s["n_cells"], s["chunks"], "snr",
                            monthly_median=True)
    host = CellReducer(s["n_cells"], "snr", monthly_median=True)

    def on_chunk(lo, hi, res, m3):
        assert res.son.dtype.names == gridjob.GRID_OUTPUTS
        assert m3.shape == (hi - lo, 3)
        feed_chunk_host(host, s["chunks"], s["n_cells"], lo, hi, res)

    rig.run(s["params"], s["cfgs"], s["ant"], s["ant_idx"], reduce=red,
            outputs=gridjob.GRID_OUTPUTS, on_chunk=on_chunk)
    _assert_same(red.finalise(), host.finalise(), "reduce + slim rows")


def test_end_to_end_raw_values(sweep):
    """sanitize=False on real output == CellReducer fed the raw /SON/."""
    s = sweep
    rig = PipelinedSetupLuffy(1024, 2, [s["slot"]], s["iono"])
    args = (rig, s["params"], s["cfgs"], s["ant"], s["ant_idx"], s["n_cells"],
            s["chunks"])
    exp = gridjob.run_grid(*args, "signalpower", monthly_median=True,
                           device=False, sanitize=False)
    got = gridjob.run_grid(*args, "signalpower", monthly_median=True,
                           device=True, sanitize=False)
    _assert_same(got, exp, "e2e raw")
