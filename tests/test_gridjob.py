"""The server-side grid reduction == PropServer's, statement for statement.

``gridjob.CellReducer`` moves a reduction that used to live in Go
(``models/voacap_grid.go``'s ``cellAgg``) onto this side of the wire, and
vectorises it in the process.  Two things could go wrong that a self-check
would not catch: the vectorisation could be a faithful port of the WRONG
semantics, or it could quietly reorder an order-dependent scan.

So the reference below is a LITERAL transcription of the Go -- one sample at a
time, same branches, same strict comparison, same accumulator reset -- written
from the source rather than from ``CellReducer``, and the two are compared on
randomised sweeps.  The generator deliberately produces ties, failures, and
repeated winning frequencies, because those are the three cases where a
plausible-looking reduction diverges.
"""

import numpy as np
import pytest

from voacap_gpu.gridjob import (CellReducer, block_spans, clamp_lat,
                                   freq_chunks, go_round, grid_cells,
                                   normalise_lon)


# ---------------------------------------------------------------------------
# The reference: models/voacap_grid.go, transcribed
# ---------------------------------------------------------------------------

class GoCellAgg:
    """`cellAgg` verbatim — has/best/bestScore + the four hour accumulators."""

    def __init__(self):
        self.has = False
        self.best = None                  # (snr, dbw, dbu, rel, freq)
        self.best_score = 0.0             # Go's zero value; !has guards it
        self.hour_count = 0
        self.sum_sig = 0.0
        self.sum_snr = 0.0
        self.sum_fs = 0.0
        self.sum_rel = 0.0

    def consider(self, pt, score, is_monthly_median):
        if not self.has or score > self.best_score:
            self.has = True
            self.best = pt
            self.best_score = score
            if is_monthly_median:
                self.sum_sig = pt[1]
                self.sum_snr = pt[0]
                self.sum_fs = pt[2]
                self.sum_rel = pt[3]
                self.hour_count = 1
            return
        if is_monthly_median and pt[4] == self.best[4]:
            self.sum_sig += pt[1]
            self.sum_snr += pt[0]
            self.sum_fs += pt[2]
            self.sum_rel += pt[3]
            self.hour_count += 1

    def finalise(self, default_freq):
        if not self.has:
            return (-200.0, -200.0, -200.0, 0.0, default_freq)
        snr, dbw, dbu, rel, freq = self.best
        if self.hour_count > 1:
            n = float(self.hour_count)
            dbw = self.sum_sig / n
            snr = self.sum_snr / n
            dbu = self.sum_fs / n
            rel = self.sum_rel / n
        return (snr, dbw, dbu, rel, freq)


_GOAL_COL = {"snr": 0, "signalpower": 1, "fieldstrength": 2,
             "reliability": 3}


def go_reduce(samples, n_cells, mode, monthly, default_freq):
    """Drive the reference over the whole sweep, in circuit order."""
    col = _GOAL_COL.get(mode.lower(), 0)
    agg = [GoCellAgg() for _ in range(n_cells)]
    for cell, ok, pt in samples:
        if not ok:
            continue
        agg[cell].consider(pt, pt[col], monthly)
    out = [a.finalise(default_freq) for a in agg]
    return [np.array([o[k] for o in out], np.float64) for k in range(5)]


# ---------------------------------------------------------------------------
# A randomised sweep, and the two ways of consuming it
# ---------------------------------------------------------------------------

def make_sweep(rng, n_cells, hours, freqs, fail_rate=0.15, coarse=True):
    """Circuit-order samples plus the per-circuit arrays the reducer takes.

    ``coarse`` quantises the scores hard so TIES actually happen -- the strict
    ``>`` in the Go means a tie keeps the earlier sample, and a reduction that
    used ``>=`` would pass every test built on continuous scores.
    """
    chunks = freq_chunks(freqs)
    blocks = [(h, ci) for h in range(hours) for ci in range(len(chunks))]
    n_circ = len(blocks) * n_cells
    nf = max(len(c) for c in chunks)

    snr = rng.uniform(-40, 60, (n_circ, nf))
    dbw = rng.uniform(-180, -60, (n_circ, nf))
    dbu = rng.uniform(-20, 90, (n_circ, nf))
    rel = rng.uniform(0, 1, (n_circ, nf)) * 100.0
    if coarse:
        snr = np.round(snr / 8.0) * 8.0
        rel = np.round(rel / 20.0) * 20.0
    status = (rng.random(n_circ) < fail_rate).astype(np.int32)

    samples = []
    for g in range(n_circ):
        b, cell = divmod(g, n_cells)
        _h, ci = blocks[b]
        for j, f in enumerate(chunks[ci]):
            samples.append((cell, status[g] == 0,
                            (snr[g, j], dbw[g, j], dbu[g, j], rel[g, j], f)))
    return chunks, blocks, snr, dbw, dbu, rel, status, samples


def feed_in_chunks(red, chunks, n_cells, snr, dbw, dbu, rel, status, span):
    """Push the sweep through ``CellReducer`` in arbitrary delivery spans."""
    n_circ = snr.shape[0]
    for lo in range(0, n_circ, span):
        hi = min(n_circ, lo + span)
        for b, r0, r1, cells in block_spans(lo, hi, n_cells):
            ci = b % len(chunks)
            rows = slice(lo + r0, lo + r1)
            ok = status[rows] == 0
            for j, f in enumerate(chunks[ci]):
                red.feed(cells, ok, snr[rows, j], dbw[rows, j],
                         dbu[rows, j], rel[rows, j], f)


@pytest.mark.parametrize("mode", ["snr", "reliability", "fieldstrength",
                                  "signalpower"])
@pytest.mark.parametrize("hours,freqs,monthly", [
    (1, [14.0], False),                                   # single point
    (1, [3.0, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.0, 28.0], False),
    (4, [3.0, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.0, 28.0], True),
    (24, [7.0, 14.0, 21.0], True),                        # every hour
    (3, [float(2 + i) for i in range(29)], True),          # 3 freq-chunks
])
def test_reduction_matches_the_go_transcription(mode, hours, freqs, monthly):
    rng = np.random.default_rng(abs(hash((mode, hours, len(freqs)))) % 2**31)
    n_cells = 137
    chunks, _blocks, snr, dbw, dbu, rel, status, samples = make_sweep(
        rng, n_cells, hours, freqs)

    exp = go_reduce(samples, n_cells, mode, monthly, default_freq=14.0)
    red = CellReducer(n_cells, mode, monthly_median=monthly, default_freq=14.0)
    feed_in_chunks(red, chunks, n_cells, snr, dbw, dbu, rel, status, span=97)
    got = red.finalise()

    names = ("snr", "dbw", "dbu", "reliab", "freq")
    for name, g, e in zip(names, got, exp):
        assert np.array_equal(g, e), (
            f"{name} differs from the Go reduction "
            f"[{mode}, {hours}h, {len(freqs)}f, monthly={monthly}]: "
            f"{int((g != e).sum())}/{g.size} cells")


def test_delivery_span_does_not_change_the_answer():
    """Chunk boundaries are an implementation detail of the GPU pipeline.

    The reducer is fed whatever spans the streams happen to deliver, so a span
    that splits a (hour, frequency-chunk) block must give the same answer as
    one that does not.
    """
    rng = np.random.default_rng(4242)
    n_cells, hours = 91, 6
    freqs = [3.0, 5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 24.0, 28.0]
    chunks, _b, snr, dbw, dbu, rel, status, samples = make_sweep(
        rng, n_cells, hours, freqs)
    ref = go_reduce(samples, n_cells, "snr", True, 14.0)

    for span in (1, 7, n_cells - 1, n_cells, n_cells + 1, 2 * n_cells, 10_000):
        red = CellReducer(n_cells, "snr", monthly_median=True,
                          default_freq=14.0)
        feed_in_chunks(red, chunks, n_cells, snr, dbw, dbu, rel, status, span)
        for name, g, e in zip(("snr", "dbw", "dbu", "reliab", "freq"),
                              red.finalise(), ref):
            assert np.array_equal(g, e), f"{name} at span {span}"


def test_a_failed_circuit_contributes_nothing():
    """status != 0 skips the WHOLE circuit, not just a bad frequency."""
    red = CellReducer(2, "snr", monthly_median=True, default_freq=9.0)
    cells = np.array([0, 1])
    ok = np.array([True, False])
    red.feed(cells, ok, np.array([10.0, 99.0]), np.array([-100.0, -1.0]),
             np.array([20.0, 80.0]), np.array([50.0, 100.0]), 14.0)
    snr, dbw, dbu, rel, freq = red.finalise()
    assert snr[0] == 10.0 and freq[0] == 14.0
    assert (snr[1], dbw[1], dbu[1], rel[1], freq[1]) == (-200.0, -200.0,
                                                         -200.0, 0.0, 9.0)


def test_ties_keep_the_earliest_sample():
    """Go's test is strictly `>`, so an equal score does not displace."""
    red = CellReducer(1, "snr", monthly_median=False, default_freq=0.0)
    c, ok = np.array([0]), np.array([True])
    red.feed(c, ok, np.array([25.0]), np.array([-100.0]), np.array([10.0]),
             np.array([50.0]), 7.0)
    red.feed(c, ok, np.array([25.0]), np.array([-111.0]), np.array([11.0]),
             np.array([55.0]), 14.0)        # same score, later
    snr, dbw, dbu, rel, freq = red.finalise()
    assert freq[0] == 7.0 and dbw[0] == -100.0, "the incumbent survived a tie"


def test_only_the_winning_frequency_averages():
    """And only samples that arrive AFTER the winner was established."""
    red = CellReducer(1, "snr", monthly_median=True, default_freq=0.0)
    c, ok = np.array([0]), np.array([True])

    def s(v, f):
        red.feed(c, ok, np.array([v]), np.array([v]), np.array([v]),
                 np.array([v]), f)

    s(10.0, 7.0)      # discarded: superseded below
    s(30.0, 14.0)     # the winner
    s(20.0, 7.0)      # wrong frequency: ignored
    s(10.0, 14.0)     # winning frequency, after the winner: averaged
    snr, _dbw, _dbu, _rel, freq = red.finalise()
    assert freq[0] == 14.0
    assert snr[0] == 20.0, f"(30 + 10) / 2 expected, got {snr[0]}"


# ---------------------------------------------------------------------------
# The grid geometry
# ---------------------------------------------------------------------------

def go_cells_reference(min_lat, max_lat, min_lon, max_lon, step):
    """gpuGrid's cell construction, transcribed with Python scalars."""
    def rnd(x):
        return np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5)

    def clamp(v):
        return 90.0 if v > 90 else (-90.0 if v < -90 else v)

    def norm(v):
        while v > 180:
            v -= 360
        while v < -180:
            v += 360
        return v

    lat_steps = max(1, int(rnd((max_lat - min_lat) / step)) + 1)
    lon_steps = max(1, int(rnd((max_lon - min_lon) / step)) + 1)
    lat, lon = [], []
    for i in range(lat_steps * lon_steps):
        lat.append(clamp(min_lat + (i // lon_steps) * step))
        lon.append(norm(min_lon + (i % lon_steps) * step))
    return np.array(lat), np.array(lon), lat_steps, lon_steps


@pytest.mark.parametrize("bounds,step", [
    ((-90.0, 90.0, -180.0, 180.0), 2.0),      # the frontend's world grid
    ((-90.0, 90.0, -180.0, 180.0), 5.0),
    ((-90.0, 90.0, -180.0, 180.0), 1.0),
    ((20.0, 60.0, -130.0, -60.0), 0.5),       # a regional box
    ((-33.0, 12.5, 100.0, 179.5), 2.5),       # off-grid bounds and step
    ((0.0, 0.0, 0.0, 0.0), 1.0),              # degenerate: one cell
])
def test_grid_cells_match_the_go_construction(bounds, step):
    mn_la, mx_la, mn_lo, mx_lo = bounds
    lat, lon, ls, os_ = grid_cells(mn_la, mx_la, mn_lo, mx_lo, step)
    e_lat, e_lon, e_ls, e_os = go_cells_reference(mn_la, mx_la, mn_lo, mx_lo,
                                                  step)
    assert (ls, os_) == (e_ls, e_os), "step counts"
    assert np.array_equal(lat, e_lat), "cell latitudes"
    assert np.array_equal(lon, e_lon), "cell longitudes"


def test_go_round_is_half_away_from_zero():
    """np.round would be half-to-even and give a different step count."""
    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    assert np.array_equal(go_round(x), np.array([-3., -2., -1., 1., 2., 3.]))


def test_freq_chunks_respect_the_frel_capacity():
    assert freq_chunks(list(range(1, 12))) == [list(range(1, 12))]
    c = freq_chunks([float(i) for i in range(29)])
    assert [len(x) for x in c] == [11, 11, 7]
    with pytest.raises(ValueError):
        freq_chunks([])


def test_helpers_wrap_and_clamp_like_go():
    assert clamp_lat(np.array([91.0, -91.0, 45.0])).tolist() == [90., -90., 45.]
    # BOTH endpoints survive: Go's loops are strict (`> 180`, `< -180`), so
    # +180 and -180 are already in range and are left exactly as they are.
    # A modulo-based wrap would fold +180 onto -180 and move the world grid's
    # entire last column.
    assert normalise_lon(np.array([181.0, -181.0, 360.0, 180.0, -180.0,
                                   540.0, -540.0, 541.0])).tolist() == [
        -179.0, 179.0, 0.0, 180.0, -180.0, 180.0, -180.0, -179.0]
