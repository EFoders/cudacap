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

ON THE DEVICE
-------------
:class:`DeviceCellReducer` is the same reduction run on the GPU, straight out
of the rig's device state (``cuda/kernels/gridreduce.cuh``): one thread per
cell replays that cell's samples in sweep order, chunk after chunk, so the
order-dependent semantics hold by construction rather than by vector
algebra.  Pass it as ``PipelinedSetupLuffy.run(..., reduce=red)`` -- or use
:func:`run_grid` -- and a sweep's only transfer is the final ``5 x n_cells``
float64 answer.  ``tests/cuda/test_gridreduce.py`` pins it bit for bit to
:class:`CellReducer` (fed as :func:`feed_chunk_host`, the Prop-SL service's
own feed) and to the Go transcription.
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


_GOAL_CODE = {"snr": 0, "dbw": 1, "dbu": 2, "reliab": 3}   # gridreduce.cuh

# The /SON/ columns the reduction reads -- and, with "nhp" and "mode", what the
# Prop-SL service's per-circuit path reads.  Pass either as ``outputs=``.
GRID_OUTPUTS = ("sndb", "dbw", "dbu", "reliab")
SERVICE_OUTPUTS = ("sndb", "dbw", "dbu", "reliab", "nhp", "mode")


class DeviceCellReducer:
    """:class:`CellReducer` on the GPU, fed straight from the device state.

    Same answer, bit for bit, as feeding :class:`CellReducer` the way the
    Prop-SL service does (``feed_chunk_host`` below is that feed): each /SON/
    sample widened to float64, optionally sanitised (``sanitize=True``: a
    non-finite dB value becomes -200 and a non-finite reliability 0, the
    service's ``_finite``), the reliability multiplied by ``rel_scale`` (100,
    PropServer's percentage), then the order-dependent winner-take-all.  With
    ``sanitize=False`` the raw values go in and CellReducer's own non-finite
    rule drops them.  See ``cuda/kernels/gridreduce.cuh`` for why the scan is
    sequential per cell, and ``tests/cuda/test_gridreduce.py`` for the proof.

    The per-cell state stays on the device across chunks AND across runs:
    ``offset`` is the global sweep row of the next row fed, so a sweep may be
    split over several ``rig.run`` calls (e.g. by hour) as long as they are
    made in sweep order.  ``chunks`` is the frequency list per frequency-chunk
    (``freq_chunks``); global row ``g`` is cell ``g % n_cells`` of block
    ``g // n_cells``, which uses ``chunks[block % len(chunks)]`` at /SON/ slots
    ``slot0 .. slot0 + len - 1`` -- the service's circuit order.

    Usage::

        red = DeviceCellReducer(n_cells, chunks, "snr", monthly_median=True)
        rig.run(params, cfgs, ant, ant_idx, reduce=red)
        snr, dbw, dbu, rel, freq = red.finalise()
    """

    def __init__(self, n_cells, chunks, rendering_mode="snr",
                 monthly_median=False, default_freq=0.0, sanitize=True,
                 rel_scale=100.0, slot0=1, block=128):
        import cupy as cp
        from voacap_gpu.cuda import driver as drv

        self._cp = cp
        self.n = int(n_cells)
        if self.n <= 0:
            raise ValueError(f"n_cells must be positive, got {n_cells}")
        chunks = [[float(f) for f in ch] for ch in chunks]
        if not chunks or any(not ch for ch in chunks):
            raise ValueError("every frequency chunk needs a frequency")
        if any(len(ch) > FREQS_PER_CIRCUIT for ch in chunks):
            raise ValueError(f"a frequency chunk holds at most "
                             f"{FREQS_PER_CIRCUIT} frequencies")
        self.chunks = chunks
        self.slot0 = int(slot0)
        if self.slot0 < 0 or self.slot0 + max(map(len, chunks)) > 14:
            raise ValueError("slot0 + frequencies must fit /SON/'s 14 slots")
        self.goal = goal_of(rendering_mode)
        self.monthly_median = bool(monthly_median)
        self.default_freq = float(default_freq)
        self.sanitize = bool(sanitize)
        self.rel_scale = float(rel_scale)
        self.block = int(block)
        self.offset = 0
        self._fn = drv.load_gridreduce_kernel()
        self._reduce = self._fn.get_function("grid_reduce_chunk")
        self._final = self._fn.get_function("grid_finalise")

        fq = np.zeros((len(chunks), FREQS_PER_CIRCUIT), F64)
        for i, ch in enumerate(chunks):
            fq[i, :len(ch)] = ch
        self._fq = cp.asarray(fq.ravel())
        self._nfq = cp.asarray(np.array([len(c) for c in chunks], np.int32))
        n = self.n
        self.has = cp.zeros(n, np.int32)
        self.best = cp.full(n, -np.inf, F64)
        self.v = [cp.zeros(n, F64) for _ in range(5)]   # snr dbw dbu rel freq
        self.count = cp.zeros(n, np.int64)
        self.s = [cp.zeros(n, F64) for _ in range(4)]   # the hourly sums
        self._grid = ((n + self.block - 1) // self.block,)

    def advance(self, rows):
        """``rows`` more sweep rows have been enqueued (the rig calls this)."""
        self.offset += int(rows)

    def enqueue_chunk(self, circ_words, stride, offsets, status, lo, c,
                      stream=None):
        """Reduce rows ``[lo, lo + c)`` (GLOBAL numbering) on ``stream``.

        ``circ_words`` is the chunk's state as int32 words, ``stride`` words a
        row, ``offsets`` the word offsets of slot 0 of sndb/dbw/dbu/reliab --
        ``driver.CIRCUIT_WORDS`` / ``driver.GRID_SON_OFFSETS`` for a rig's
        CircuitState, or a synthetic block's own.  Asynchronous; the caller
        orders successive chunks (the rig does, with events).
        """
        if c <= 0:
            return
        args = (circ_words, np.int32(stride),
                *(np.int32(o) for o in offsets), status,
                np.int64(lo), np.int32(c), np.int32(self.n),
                np.int32(len(self.chunks)), self._fq, self._nfq,
                np.int32(self.slot0), np.int32(_GOAL_CODE[self.goal]),
                np.int32(self.monthly_median), np.int32(self.sanitize),
                np.float64(self.rel_scale), self.has, self.best, *self.v,
                self.count, *self.s)
        if stream is None:
            self._reduce(self._grid, (self.block,), args)
        else:
            with stream:
                self._reduce(self._grid, (self.block,), args)

    def feed_words(self, words, status, offsets, slots=14):
        """Host convenience (tests): reduce a ``(rows, stride)`` int32 block.

        ``offsets`` locate the four columns inside a row exactly as for
        ``enqueue_chunk``; the rows are the next ``rows`` of the sweep.
        """
        cp = self._cp
        words = np.ascontiguousarray(words, np.int32)
        rows, stride = words.shape
        d_w = cp.asarray(words)
        d_s = cp.asarray(np.ascontiguousarray(status, np.int32))
        self.enqueue_chunk(d_w, stride, offsets, d_s, self.offset, rows)
        self.offset += rows
        cp.cuda.Device().synchronize()       # keep d_w/d_s alive till done

    def finalise(self):
        """``cellAgg.finalise`` for every cell -> five float64 numpy columns
        (snr, dbw, dbu, rel, freq), one ``5 * n_cells * 8``-byte transfer."""
        cp = self._cp
        out = cp.empty(5 * self.n, F64)
        self._final(self._grid, (self.block,),
                    (self.has, *self.v, self.count, *self.s,
                     np.float64(self.default_freq), out, np.int32(self.n)))
        host = cp.asnumpy(out).reshape(5, self.n)
        return tuple(np.ascontiguousarray(host[k]) for k in range(5))


def _finite(a, sentinel):
    """The service's sanitiser: float64, non-finite -> ``sentinel``."""
    a = np.asarray(a, F64)
    return np.where(np.isfinite(a), a, sentinel)


def feed_chunk_host(red, chunks, n_cells, lo, hi, res, sanitize=True,
                    rel_scale=100.0, slot0=1):
    """One delivered chunk -> :meth:`CellReducer.feed`, the service's way.

    A transcription of Prop-SL's ``handle_predict_grid.on_chunk`` (with its
    ``_son_columns``): rows ``[lo, hi)`` of the sweep, in GLOBAL numbering,
    are cut into (hour, frequency-chunk) blocks and fed frequency by
    frequency.  ``res`` may be full or slim; only sndb/dbw/dbu/reliab at
    slots ``slot0 ..`` and the status are read.
    """
    mx = max(len(c) for c in chunks)
    sl = slice(slot0, slot0 + mx)
    son = res.son
    if sanitize:
        snr = _finite(son["sndb"][:, sl], -200.0)
        dbw = _finite(son["dbw"][:, sl], -200.0)
        dbu = _finite(son["dbu"][:, sl], -200.0)
        rel = _finite(son["reliab"][:, sl], 0.0)
    else:
        snr, dbw, dbu, rel = (np.asarray(son[k][:, sl], F64)
                              for k in GRID_OUTPUTS)
    ok_all = np.asarray(res.status) == 0
    for blk, r0, r1, cells in block_spans(lo, hi, n_cells):
        ci = blk % len(chunks)
        rows = slice(r0, r1)
        ok = ok_all[rows]
        for j, f in enumerate(chunks[ci]):
            red.feed(cells, ok, snr[rows, j], dbw[rows, j], dbu[rows, j],
                     rel[rows, j] * rel_scale, f)


def sweep_params(tx, lat, lon, hours, n_fchunks, iono=0, npsl=0):
    """The sweep's SETUP_PARAMS block, in the service's circuit order.

    Hours outermost, then frequency-chunk, then cell; ``cfg`` is the
    frequency-chunk index, so ``cfgs[j]`` must carry ``freq_chunks(..)[j]``.
    """
    from voacap_gpu.cuda.driver import SETUP_PARAMS_DTYPE
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    n_cells = lat.size
    n_blocks = len(hours) * int(n_fchunks)
    p = np.zeros(n_blocks * n_cells, SETUP_PARAMS_DTYPE)
    p["tlatd"] = float(tx[0])
    p["tlongd"] = float(tx[1])
    p["rlatd"] = np.tile(lat.astype(np.float32), n_blocks)
    p["rlongd"] = np.tile(lon.astype(np.float32), n_blocks)
    blk_hour = np.repeat(np.asarray(hours, np.int32), int(n_fchunks))
    p["hour"] = np.repeat(blk_hour, n_cells)
    blk_cfg = np.tile(np.arange(int(n_fchunks), dtype=np.int32), len(hours))
    p["cfg"] = np.repeat(blk_cfg, n_cells)
    p["npsl"] = int(npsl)
    p["iono"] = int(iono)
    return p


def run_grid(rig, params, cfgs, ant, ant_idx, n_cells, chunks,
             rendering_mode="snr", monthly_median=False, default_freq=0.0,
             device=True, on_progress=None, sanitize=True, rel_scale=100.0,
             host_outputs=GRID_OUTPUTS):
    """A whole sweep on a ``PipelinedSetupLuffy`` -> one reduced row per cell.

    ``device=True`` reduces on the GPU (:class:`DeviceCellReducer`): nothing
    per-row leaves the card and the answer is one ``5 x n_cells`` float64
    transfer.  ``device=False`` is the host path the service uses today --
    ``on_chunk`` + :class:`CellReducer` -- with the results brought back as
    ``host_outputs`` (slim; pass None for the full /SON/).  Both return
    ``(snr, dbw, dbu, rel, freq)``, bit-identical to each other.

    ``on_progress(done, total)`` is called as each chunk lands.
    """
    n = len(params)
    done = [0]

    if device:
        red = DeviceCellReducer(n_cells, chunks, rendering_mode,
                                monthly_median=monthly_median,
                                default_freq=default_freq, sanitize=sanitize,
                                rel_scale=rel_scale)

        def progress(lo, hi, _res, _m3):
            done[0] += hi - lo
            on_progress(done[0], n)

        rig.run(params, cfgs, ant, ant_idx, reduce=red,
                on_chunk=None if on_progress is None else progress)
        return red.finalise()

    red = CellReducer(n_cells, rendering_mode, monthly_median=monthly_median,
                      default_freq=default_freq)

    def on_chunk(lo, hi, res, _m3):
        feed_chunk_host(red, chunks, n_cells, lo, hi, res, sanitize=sanitize,
                        rel_scale=rel_scale)
        done[0] += hi - lo
        if on_progress is not None:
            on_progress(done[0], n)

    rig.run(params, cfgs, ant, ant_idx, on_chunk=on_chunk,
            outputs=host_outputs)
    return red.finalise()


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
