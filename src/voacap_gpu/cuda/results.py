"""What comes back off the card: full, compact and slim result fetches.

``fetch_luffy`` (every byte, for the replay), ``LuffyResults`` /
``fetch_results`` (the 1,700-byte production answer), the slim-output spec
that gathers only the /SON/ columns a caller reads, and the word offsets a
device-side grid reduction reads straight out of CircuitState.
"""

import numpy as np

from voacap_gpu.cuda.abi import (_CIRC_WORDS, _RESULT_COLS, _RESULT_WORDS,
                                 _span, CIRCUIT_DTYPE, F32, I32)
from voacap_gpu.cuda.kernels import (_RESULT_MAP, _device_map, _rays_from_soa,
                                     _xfer)
from voacap_gpu.cuda.state import SON_DTYPE


def fetch_luffy(state):
    """The staged device state -> ``(out_bufs, out_rays, status)``.

    THE WHOLE 55,652 bytes a thread, both structs.  Right for the replay,
    which compares every cell; wrong for production, which wants
    ``fetch_results``.  After an ``upload="inputs"`` launch the fields the
    host never sent come back as the zeros the device was memset to, not as
    the entry values -- that is not a bug, it is the point.
    """
    import cupy as cp

    d_circ, d_soa, d_status, n = state
    cp.cuda.runtime.deviceSynchronize()
    out = cp.asnumpy(d_circ).view(CIRCUIT_DTYPE)
    out_rays = _rays_from_soa(d_soa, n)
    return out, out_rays, cp.asnumpy(d_status)


class LuffyResults:
    """What a production caller reads back: 1,700 bytes a circuit-hour.

    son:    (N,) SON_DTYPE — the per-frequency results table, /SON/ entire
    xluf:   (N, 25) float32 — /MUFS/ XLUF, the LUF by hour
    jmode:  (N,) int32 — /FRQ/ JMODE, the controlling sample area
    status: (N,) int32 — 0 success, non-zero the refusal a callee reported
    """

    __slots__ = ("son", "xluf", "jmode", "status")

    def __init__(self, son, xluf, jmode, status):
        self.son, self.xluf = son, xluf
        self.jmode, self.status = jmode, status

    def __iter__(self):
        return iter((self.son, self.xluf, self.jmode, self.status))


def _split_results(host, n):
    """One compacted ``(n, _RESULT_WORDS)`` int32 block -> (son, xluf, jmode).

    The result words are three disjoint runs concatenated column-wise; this is
    the decode both ``fetch_results`` and the pipelined runner share, so the
    word bookkeeping lives in exactly one place.
    """
    def col(path):
        c, ln = _RESULT_COLS[path]
        return np.ascontiguousarray(host[:, c:c + ln])

    son = col("son").view(SON_DTYPE).reshape(n)
    xluf = col("muf.xluf").view(F32)
    jmode = col("frq.jmode").reshape(n)
    return son, xluf, jmode


def fetch_results(state):
    """The staged device state -> ``LuffyResults``: /SON/, XLUF, JMODE, status.

    The result words are scattered through CircuitState, so they are gathered
    into one compacted ``(N, m)`` device block first and cross the bus as a
    SINGLE D2H -- 1,700 bytes a circuit-hour against ``fetch_luffy``'s 55,652.
    """
    import cupy as cp

    d_circ, _d_soa, d_status, n = state
    cp.cuda.runtime.deviceSynchronize()
    gath = cp.empty((n, _RESULT_WORDS), np.int32)
    tot = _RESULT_WORDS * n
    _xfer("gather_words")(((tot + 255) // 256,), (256,),
                          (d_circ.view(np.int32), gath,
                           _device_map("result", _RESULT_MAP),
                           I32(_RESULT_WORDS), I32(_CIRC_WORDS),
                           np.int64(tot)))
    host = cp.asnumpy(gath)                          # ONE transfer
    status = cp.asnumpy(d_status)
    son, xluf, jmode = _split_results(host, n)
    return LuffyResults(son, xluf, jmode, status)


# ---------------------------------------------------------------------------
# SLIM OUTPUTS -- gather only the /SON/ columns (and slots) a caller reads
# ---------------------------------------------------------------------------
# ``LuffyResults`` is the whole of /SON/ x 14 slots + XLUF + JMODE: 1,696
# bytes a circuit-hour, ~670 MB for a 2-degree monthly-median world sweep.  A
# map service reads four or six of those 28 columns, at the 9-11 slots that
# carry its frequencies.  ``outputs=`` names the columns; the device gathers
# exactly those words (same ``gather_words`` kernel, a different word map) and
# only they cross the bus.  It is an integer word copy, so a slim column is
# BIT-IDENTICAL to the same column of the full result by construction --
# ``tests/cuda/test_slim_outputs.py`` checks it anyway.
#
# SLOT INDEXING IS PRESERVED: a slim column has ``nslots`` entries and entry k
# IS /SON/ slot k, for k < nslots.  So ``res.son["sndb"][:, 1:1 + nf]`` means
# the same thing on a slim result as on a full one, and a caller's column
# extraction does not change.  ``nslots`` defaults to the slots the configs
# actually use -- slot 0 up to the highest user frequency (``frel[1..11]``);
# pass ``slots=13`` to keep slot 12, LUFFY's MUF slot, as well.

SLIM_EXTRA = ("xluf", "jmode")               # outside /SON/, opt-in by name
SLIM_ALWAYS = ("status", "muf3")             # always returned; accepted, ignored
_SON_SLOTS = SON_DTYPE["sndb"].shape[0]      # 14
_SON_BASE = _span(CIRCUIT_DTYPE, "son")[0]
_USER_FREQ_SLOTS = 11                        # frel[1..11]; [12] is the MUF


def slots_in_use(cfgs):
    """1 + the highest /SON/ slot any config puts a user frequency in.

    ``setup_config`` puts frequency j at ``frel[j + 1]`` and LUFFY skips a
    slot whose ``frel`` is not positive, so slots above this carry nothing a
    caller asked for.  Slot 0 is kept so slot k stays column k.
    """
    frel = np.asarray(cfgs["frel"], F32).reshape(-1, cfgs["frel"].shape[-1])
    used = frel[:, 1:1 + _USER_FREQ_SLOTS] > 0
    hi = 0
    if used.any():
        hi = 1 + int(np.flatnonzero(used.any(axis=0)).max())
    return 1 + hi


class SlimSpec:
    """What a slim run gathers: the word map, and how to decode it.

    ``son_fields`` are /SON/ leaves, ``extra`` a subset of ``SLIM_EXTRA``,
    ``nslots`` the leading slots kept of every 14-slot column.  The gathered
    row is laid out as ``row_dtype`` -- the son fields in the order given,
    then XLUF, then JMODE -- so the host decode is a ``view``, not a copy.
    """

    __slots__ = ("son_fields", "extra", "nslots", "son_dtype", "row_dtype",
                 "map", "words", "key")

    def __init__(self, outputs, nslots):
        nslots = int(nslots)
        if not 1 <= nslots <= _SON_SLOTS:
            raise ValueError(f"slots must be 1..{_SON_SLOTS}, got {nslots}")
        names = [str(o) for o in outputs]
        if len(set(names)) != len(names):
            raise ValueError(f"outputs names a field twice: {names}")
        bad = [o for o in names if o not in SON_DTYPE.names
               and o not in SLIM_EXTRA and o not in SLIM_ALWAYS]
        if bad:
            raise ValueError(
                f"unknown output(s) {bad}: expected /SON/ fields "
                f"{SON_DTYPE.names}, or {SLIM_EXTRA}")
        self.son_fields = tuple(o for o in names if o in SON_DTYPE.names)
        self.extra = tuple(o for o in SLIM_EXTRA if o in names)
        self.nslots = nslots
        sdt, idx = [], []
        for f in self.son_fields:
            sub, off = SON_DTYPE.fields[f][0], SON_DTYPE.fields[f][1]
            w0 = _SON_BASE + off // 4
            if sub.shape:                     # a 14-slot column
                sdt.append((f, sub.base, (nslots,)))
                idx.append(np.arange(w0, w0 + nslots, dtype=I32))
            else:                             # d10r .. d90s: one word
                sdt.append((f, sub))
                idx.append(np.array([w0], I32))
        self.son_dtype = np.dtype(sdt) if sdt else None
        rdt = list(sdt)
        if "xluf" in self.extra:
            s, ln = _span(CIRCUIT_DTYPE, "muf.xluf")
            rdt.append(("xluf", F32, (ln,)))
            idx.append(np.arange(s, s + ln, dtype=I32))
        if "jmode" in self.extra:
            s, _ln = _span(CIRCUIT_DTYPE, "frq.jmode")
            rdt.append(("jmode", I32))
            idx.append(np.array([s], I32))
        self.row_dtype = np.dtype(rdt) if rdt else None
        self.map = (np.concatenate(idx).astype(I32) if idx
                    else np.zeros(0, I32))
        self.words = int(self.map.size)
        assert self.row_dtype is None or self.row_dtype.itemsize == 4 * self.words
        assert self.words <= _RESULT_WORDS
        self.key = ("slim", self.son_fields, self.extra, nslots)

    @property
    def bytes_per_row(self):
        """What crosses the bus per circuit-hour: the words + status + muf3."""
        return 4 * self.words + 4 + 12

    def decode(self, block, status):
        """An OWNED ``(c, words)`` int32 block + status -> ``SlimResults``."""
        c = block.shape[0]
        son = xluf = jmode = None
        if self.row_dtype is not None:
            rows = np.ascontiguousarray(block).view(self.row_dtype).reshape(c)
            if not self.extra:
                son = rows                       # zero-copy: the block IS son
            else:
                if self.son_fields:
                    son = np.empty(c, self.son_dtype)
                    for f in self.son_fields:
                        son[f] = rows[f]
                if "xluf" in self.extra:
                    xluf = np.ascontiguousarray(rows["xluf"])
                if "jmode" in self.extra:
                    jmode = np.ascontiguousarray(rows["jmode"])
        return SlimResults(son, xluf, jmode, status, self)


def slim_spec(outputs, cfgs=None, slots=None):
    """``outputs`` (+ the configs, for the slot count) -> a ``SlimSpec``."""
    if isinstance(outputs, str):
        outputs = (outputs,)
    if slots is None:
        slots = _SON_SLOTS if cfgs is None else slots_in_use(cfgs)
    return SlimSpec(tuple(outputs), slots)


class SlimResults(LuffyResults):
    """A slim ``LuffyResults``: only the columns named in ``outputs=``.

    son:    (N,) structured array of JUST the requested /SON/ fields, each a
            ``(nslots,)`` column whose entry k is /SON/ slot k -- or None
    xluf:   (N, 25) float32 if ``"xluf"`` was requested, else None
    jmode:  (N,) int32 if ``"jmode"`` was requested, else None
    status: (N,) int32, always
    """

    __slots__ = ("spec",)

    def __init__(self, son, xluf, jmode, status, spec):
        super().__init__(son, xluf, jmode, status)
        self.spec = spec

    @property
    def nslots(self):
        return self.spec.nslots

    def take(self, lo, hi):
        """Rows ``[lo, hi)`` as an owned SlimResults (for a padded batch)."""
        def cut(a):
            return None if a is None else a[lo:hi]
        return SlimResults(cut(self.son), cut(self.xluf), cut(self.jmode),
                           self.status[lo:hi], self.spec)


def _gather_slim(d_circ, spec, n, rows=None):
    """Device gather of ``spec``'s words for the first ``rows`` of n states."""
    import cupy as cp
    rows = n if rows is None else rows
    gath = cp.empty((rows, spec.words), np.int32)
    tot = spec.words * rows
    if tot:
        _xfer("gather_words")(((tot + 255) // 256,), (256,),
                              (d_circ.view(np.int32), gath,
                               _device_map(spec.key, spec.map),
                               I32(spec.words), I32(_CIRC_WORDS),
                               np.int64(tot)))
    return gath


def fetch_slim_results(state, spec, rows=None):
    """``fetch_results`` restricted to ``spec`` -> ``SlimResults``.

    ``rows`` limits the transfer to the first ``rows`` circuits (a padded
    batch's real rows); the padding never crosses the bus.
    """
    import cupy as cp
    d_circ, _d_soa, d_status, n = state
    cp.cuda.runtime.deviceSynchronize()
    rows = n if rows is None else int(rows)
    host = cp.asnumpy(_gather_slim(d_circ, spec, n, rows))
    status = cp.asnumpy(d_status[:rows])
    return spec.decode(host, status)


# What a GRID REDUCTION reads straight out of a device CircuitState: the word
# offset of slot 0 of the four /SON/ columns PropServer aggregates, and the
# state stride.  gridjob.DeviceCellReducer takes these (or a synthetic block's
# own) so the reduction never needs a gather at all.
GRID_SON_FIELDS = ("sndb", "dbw", "dbu", "reliab")
GRID_SON_OFFSETS = tuple(_span(CIRCUIT_DTYPE, f"son.{f}")[0]
                         for f in GRID_SON_FIELDS)
CIRCUIT_WORDS = _CIRC_WORDS
