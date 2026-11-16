"""Whole-grid jobs: expand a sweep into circuits, reduce the answers per cell.

WHY THIS IS HERE AND NOT IN THE CALLER
--------------------------------------
A Prop-SL area render is not one calculation, it is a SWEEP: every cell of a
lat/lon grid, at every hour of the time mode, at every frequency of the
frequency mode.  A monthly-median world grid at 2 degrees is 16,471 cells x
24 hours x 9 frequencies -- 395,304 circuits producing 2.8 million
(hour, frequency) samples, of which the map shows 16,471 numbers.

Until now the caller expanded that sweep itself, posted the circuits in
chunks, and reduced the samples back down on its own side.  Every sample
therefore crossed the wire: ~66 MB of JSON for a single-hour world grid, and
the reduction ran only after the last chunk arrived.  Moving both ends here
means the request is a few hundred bytes and the response is one row per
cell -- and, because the reduction consumes each chunk as it lands, it
overlaps the GPU instead of waiting for it.

THE REDUCTION IS A TRANSCRIPTION, NOT A DESIGN
----------------------------------------------
:class:`CellReducer` reproduces PropServer's ``cellAgg`` (models/
voacap_grid.go) exactly, including the parts that are easy to get wrong:

* **It is ORDER-DEPENDENT.**  ``consider`` resets the hourly accumulator every
  time a new best arrives, so what is averaged is the winning sample plus only
  those that come AFTER it.  Feed the samples in a different order and the
  monthly-median numbers change.  The order is the caller's circuit order:
  hours outermost, then frequency-chunk, then cell, then the frequencies
  within a chunk.
* **Ties keep the incumbent.**  The test is ``score > bestScore``, strictly,
  so the EARLIEST sample of a tied maximum wins.
* **Only the winning frequency averages.**  Later samples contribute to the
  mean only when their frequency equals the current best's.
* **A failed circuit contributes nothing at all** -- ``status != 0`` skips the
  whole circuit, every frequency of it, rather than scoring it badly.
  (One extension, not in the Go: a single sample with a non-finite score or
  value -- a not-computed cell, which ``engine.Results`` reports as NaN --
  is dropped the same way.  See :meth:`CellReducer.feed`.)

Rather than replay that scan sample by sample, the same answer falls out of
two vector passes, which is what makes it affordable: the winner is the FIRST
argmax of the score, and after it nothing resets, so the mean is the winner
plus every later sample at the winner's frequency.  :meth:`CellReducer.feed`
does that incrementally, one (hour, frequency-chunk) block at a time -- and
within a block each cell appears exactly once, so the fancy-indexed updates
below have no repeated indices and are well defined.

``tests/test_gridjob.py`` checks this against a literal statement-by-statement
port of the Go, on randomised sweeps, rather than against itself.
"""

import numpy as np

F32 = np.float32
F64 = np.float64

# The engine takes at most 11 frequencies per circuit (the FREL card's
# capacity), which is why a frequency sweep collapses into so few circuits.
FREQS_PER_CIRCUIT = 11

# PropServer's sentinels for "nothing here" (models/voacap_grid.go finalise).
SENTINEL_DB = -200.0
SENTINEL_REL = 0.0

# The metric each rendering mode maximises (models/voacap_grid.go pickGoal).
# Anything unrecognised falls back to SNR, the VOACAP area-mode default.
_GOALS = {"reliability": "reliab", "fieldstrength": "dbu",
          "signalpower": "dbw", "snr": "snr"}


def goal_of(rendering_mode):
    return _GOALS.get(str(rendering_mode or "").lower(), "snr")


def go_round(x):
    """Go's ``math.Round``: half away from zero.

    NOT ``np.round``, which is half-to-even.  This decides the grid's step
    count, so a half-way case landing differently would give the whole render
    a different number of cells than PropServer expects.
    """
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def clamp_lat(lat):
    return np.clip(lat, -90.0, 90.0)


def normalise_lon(lon):
    """Go's ``normaliseLon``: subtract/add 360 UNTIL inside [-180, 180].

    The half-open modulo form ``((lon + 180) % 360) - 180`` is NOT the same
    function: it maps +180 to -180, where the Go loop (``for lon > 180``,
    strictly) leaves +180 alone.  That is the world grid's last column --
    every render the frontend asks for spans -180..180 -- so the two differ on
    a whole meridian of cells, and the float32 the kernel then sees differs
    with it.  Both endpoints are kept as the Go keeps them.
    """
    out = np.array(lon, F64, copy=True)
    hi = out > 180.0
    out[hi] -= 360.0 * np.ceil((out[hi] - 180.0) / 360.0)
    lo = out < -180.0
    out[lo] += 360.0 * np.ceil((-180.0 - out[lo]) / 360.0)
    return out


def grid_cells(min_lat, max_lat, min_lon, max_lon, step):
    """The uniform lat/lon grid, in PropServer's cell order.

    Transcribed from ``gpuGrid`` so the GPU renders on exactly the cells the
    caller expects -- row-major from the south-west corner, ``i / lonSteps``
    integer division for the row and ``i % lonSteps`` for the column, all in
    float64 as the Go is.  Returns ``(lat, lon, lat_steps, lon_steps)``.
    """
    step = F64(step)
    if step <= 0:
        raise ValueError(f"grid_step must be positive, got {float(step)}")
    lat_steps = max(1, int(go_round((F64(max_lat) - F64(min_lat)) / step)) + 1)
    lon_steps = max(1, int(go_round((F64(max_lon) - F64(min_lon)) / step)) + 1)
    i = np.arange(lat_steps * lon_steps, dtype=np.int64)
    lat = clamp_lat(F64(min_lat) + (i // lon_steps) * step)
    lon = normalise_lon(F64(min_lon) + (i % lon_steps) * step)
    return lat, lon, lat_steps, lon_steps


def freq_chunks(freqs):
    """The frequency list split into circuit-sized groups."""
    f = [float(x) for x in freqs]
    if not f:
        raise ValueError("a grid sweep needs at least one frequency")
    return [f[i:i + FREQS_PER_CIRCUIT]
            for i in range(0, len(f), FREQS_PER_CIRCUIT)]


class CellReducer:
    """Per-cell winner-take-all across a (hour, frequency) sweep.

    See the module docstring: this is ``cellAgg`` from PropServer, fed one
    block at a time so it can run while the GPU is still working.
    """

    __slots__ = ("n", "goal", "monthly_median", "default_freq", "has",
                 "best_score", "snr", "dbw", "dbu", "rel", "freq",
                 "count", "s_snr", "s_dbw", "s_dbu", "s_rel")

    def __init__(self, n_cells, rendering_mode="snr", monthly_median=False,
                 default_freq=0.0):
        n = int(n_cells)
        self.n = n
        self.goal = goal_of(rendering_mode)
        self.monthly_median = bool(monthly_median)
        self.default_freq = float(default_freq)
        self.has = np.zeros(n, bool)
        self.best_score = np.full(n, -np.inf, F64)
        self.snr = np.zeros(n, F64)
        self.dbw = np.zeros(n, F64)
        self.dbu = np.zeros(n, F64)
        self.rel = np.zeros(n, F64)
        self.freq = np.zeros(n, F64)
        self.count = np.zeros(n, np.int64)
        self.s_snr = np.zeros(n, F64)
        self.s_dbw = np.zeros(n, F64)
        self.s_dbu = np.zeros(n, F64)
        self.s_rel = np.zeros(n, F64)

    def feed(self, cells, ok, snr, dbw, dbu, rel, f):
        """One (block, frequency) sample per cell, in sweep order.

        ``cells`` are the cell indices this block covers -- each at most once,
        which is what makes the fancy-indexed writes below unambiguous.  ``ok``
        is the per-circuit success mask (a failed circuit contributes nothing).
        ``rel`` is already the 0..100 percentage PropServer aggregates on.

        A sample whose score or any of its four values is non-finite (NaN or
        +-inf) is treated as NOT ok: it contributes nothing, exactly like a
        failed circuit.  This EXTENDS PropServer's semantics -- the Go never
        saw NaN, because its samples never carried one -- and is needed now
        that ``engine.Results`` reports a not-computed (hour, frequency) cell
        as NaN: a NaN would otherwise poison the hourly sums, and a ``-inf``
        or ``+inf`` would win or lose the max for the wrong reason.  For all-
        finite input ``ok`` is unchanged, so the reduction is bit-identical to
        the Go transcription.  A caller feeding raw /SON/ values rather than
        ``Results`` must fold ``Results.valid`` (or ``nhp >= 1``) into ``ok``
        itself: a not-computed slot's raw dbw is a finite 0 dBW preset, which
        would otherwise win every cell.
        """
        cells = np.asarray(cells, np.int64)
        metric = {"snr": snr, "dbw": dbw, "dbu": dbu, "reliab": rel}[self.goal]
        score = np.asarray(metric, F64)
        finite = np.isfinite(score)
        for v in (snr, dbw, dbu, rel):
            finite = finite & np.isfinite(np.asarray(v, F64))
        ok = np.asarray(ok, bool) & finite

        # --- a new best: adopt it and RESTART the hourly accumulator --------
        # `not has or score > bestScore`, strictly greater, so a tie keeps the
        # incumbent and therefore the earliest sample of a tied maximum wins.
        new = ok & (~self.has[cells] | (score > self.best_score[cells]))
        ci = cells[new]
        if ci.size:
            self.has[ci] = True
            self.best_score[ci] = score[new]
            self.snr[ci] = snr[new]
            self.dbw[ci] = dbw[new]
            self.dbu[ci] = dbu[new]
            self.rel[ci] = rel[new]
            self.freq[ci] = f
            if self.monthly_median:
                self.s_snr[ci] = snr[new]
                self.s_dbw[ci] = dbw[new]
                self.s_dbu[ci] = dbu[new]
                self.s_rel[ci] = rel[new]
                self.count[ci] = 1

        # --- otherwise accumulate, but only at the WINNING frequency --------
        if self.monthly_median:
            acc = ok & ~new & (self.freq[cells] == f)
            ci = cells[acc]
            if ci.size:
                self.s_snr[ci] += snr[acc]
                self.s_dbw[ci] += dbw[acc]
                self.s_dbu[ci] += dbu[acc]
                self.s_rel[ci] += rel[acc]
                self.count[ci] += 1

    def finalise(self):
        """``cellAgg.finalise`` for every cell -> five float64 columns.

        Cells that never saw a usable sample get PropServer's sentinels; cells
        whose winning frequency recurred across hours get the mean of those
        hours, at the winning frequency.
        """
        snr = self.snr.copy()
        dbw = self.dbw.copy()
        dbu = self.dbu.copy()
        rel = self.rel.copy()
        freq = self.freq.copy()

        avg = self.count > 1
        if avg.any():
            n = self.count[avg].astype(F64)
            snr[avg] = self.s_snr[avg] / n
            dbw[avg] = self.s_dbw[avg] / n
            dbu[avg] = self.s_dbu[avg] / n
            rel[avg] = self.s_rel[avg] / n

        miss = ~self.has
        snr[miss] = SENTINEL_DB
        dbw[miss] = SENTINEL_DB
        dbu[miss] = SENTINEL_DB
        rel[miss] = SENTINEL_REL
        freq[miss] = self.default_freq
        return snr, dbw, dbu, rel, freq


def block_spans(lo, hi, n_cells):
    """Split a circuit span into the (hour, freq-chunk) blocks it touches.

    The circuit order is hours -> frequency-chunk -> cell, so block ``b`` owns
    circuits ``[b * n_cells, (b + 1) * n_cells)`` and covers every cell once.
    A delivered chunk rarely lines up with that, hence this.  Yields
    ``(block, row_lo, row_hi, cells)`` where the rows are offsets INTO the
    delivered chunk.
    """
    b = lo // n_cells
    while b * n_cells < hi:
        s = max(lo, b * n_cells)
        e = min(hi, (b + 1) * n_cells)
        yield b, s - lo, e - lo, np.arange(s - b * n_cells, e - b * n_cells,
                                           dtype=np.int64)
        b += 1
