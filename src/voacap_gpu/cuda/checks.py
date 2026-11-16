"""Input validation and stream-ordering helpers shared by every LUFFY rig.

The dtype / shape / range checks run on the host BEFORE anything is
launched, because every index the device takes is a raw pointer offset.
``_drain_default_stream`` and ``_quiesce`` are the two edges of the ordering
contract the rigs keep (construction and the exception path of a run), and
``_PinnedStaging`` is the page-locked staging + async copy trio both
pipelined rigs use.
"""

import numpy as np

from voacap_gpu.cuda.abi import CIRCUIT_DTYPE
from voacap_gpu.cuda.state import RAYS_DTYPE


def _require_dtype(arr, dtype, what):
    if arr.dtype != dtype:
        raise TypeError(f"{what} must have dtype {dtype}, got {arr.dtype}")


def _require_rows(arr, n, what):
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{what} must have shape ({n},), got {arr.shape}")


def _offending(mask, limit=8):
    """'rows [3, 17]' (capped) for a boolean mask with at least one hit."""
    rows = np.flatnonzero(mask)
    head = ", ".join(str(int(r)) for r in rows[:limit])
    more = f", ... ({rows.size} rows)" if rows.size > limit else ""
    return f"rows [{head}{more}]"


def _n_antenna_sets(ant):
    """How many /cantenna/ sets a ``shared_antenna_tables`` dict holds."""
    return int(np.size(ant["iats"])) // (int(ant["numants"]) + 1)


def _check_setup_params(params, ant_idx, n_iono, n_cfg, n_ant):
    """Refuse a SETUP_PARAMS_DTYPE batch the device would mis-read.

    Every index here becomes a raw pointer offset on the device, and an
    out-of-range hour reads past the 25-row VIRTIM table and comes back as
    status 0 with garbage, so these are checked BEFORE launch, with a
    ValueError naming the field and the offending rows.  Vectorised: a few
    comparisons per field, noise next to the ~400k rows/s the path runs at.
    Coordinates use ``~(abs(x) <= lim)``, which is also True for NaN, so
    "finite and in range" is one comparison.
    """
    def refuse(field, mask, rule):
        if mask.any():
            raise ValueError(
                f"SetupParams.{field} out of range ({rule}) at "
                f"{_offending(mask)}")

    hour = params["hour"]
    refuse("hour", (hour < 1) | (hour > 24), "integer hour 1..24")
    for f in ("tlatd", "rlatd"):
        refuse(f, ~(np.abs(params[f]) <= 90.0), "finite, -90..90 degrees")
    # Longitude is circular: refuse only non-finite values, and WRAP an
    # out-of-range one into [-180, 180] (validate.wrap_lon, the PropServer
    # normaliseLon rule) on a COPY, so the caller's array is never mutated
    # and an all-in-range batch is passed through untouched (bit-identical).
    wrapped = None
    for f in ("tlongd", "rlongd"):
        refuse(f, ~np.isfinite(params[f]), "finite degrees")
        out = ~(np.abs(params[f]) <= 180.0)
        if out.any():
            from voacap_gpu.validate import wrap_lon
            if wrapped is None:
                wrapped = params.copy()
            col = wrapped[f]
            col[out] = wrap_lon(col[out].astype(np.float64)).astype(col.dtype)
    if wrapped is not None:
        params = wrapped
    npsl = params["npsl"]
    refuse("npsl", (npsl != 0) & (npsl != 1), "0 or 1")
    iono = params["iono"]
    refuse("iono", (iono < 0) | (iono >= n_iono),
           f"0 <= iono < {n_iono} ionosphere slots")
    cfg = params["cfg"]
    refuse("cfg", (cfg < 0) | (cfg >= n_cfg),
           f"0 <= cfg < {n_cfg} config slots supplied")
    bad = (ant_idx < 0) | (ant_idx >= n_ant)
    if bad.any():
        raise ValueError(
            f"ant_idx out of range (0 <= ant_idx < {n_ant} antenna sets) at "
            f"{_offending(bad)}")
    return params


def _check_luffy_inputs(bufs, rays_bufs, ipflags, iono_idx, ant_idx, n=None):
    """dtype/shape checks for a packed LUFFY batch; returns its length."""
    _require_dtype(bufs, CIRCUIT_DTYPE, "bufs")
    _require_dtype(rays_bufs, RAYS_DTYPE, "rays_bufs")
    if n is None:
        n = bufs.shape[0] if bufs.ndim == 1 else -1
    _require_rows(bufs, n, "bufs")
    _require_rows(rays_bufs, n, "rays_bufs")
    for v, what in ((ipflags, "ipflags"), (iono_idx, "iono_idx"),
                    (ant_idx, "ant_idx")):
        _require_rows(np.asarray(v), n, what)
    return n


def _drain_default_stream():
    """Block until the device has finished everything queued so far.

    THE ORDERING CONTRACT every rig keeps.  Each one allocates and
    fills its buffers (``cp.zeros`` / ``cp.full``) and uploads its tables
    (``cp.asarray``) on whatever stream is current -- CuPy's legacy default
    stream, normally -- and the pipelined rigs then do all their work on
    ``Stream(non_blocking=True)`` streams, which by definition do NOT wait for
    the default stream.  When the pool hands back memory freed by an earlier
    object, a work stream could read it before the fill or upload landed:
    reproduced as wrong rows with status 0, and pinned deterministically by
    ``tests/cuda/test_rig_race.py``.

    A device synchronise is the simplest ordering that is correct whatever
    stream the caller had current at construction and whatever stream a
    later submission uses (an event recorded on "the" default stream would
    have to guess which one the fills went to).  It runs only at the edges --
    end of construction, a table swap, the once-per-run config upload --
    never inside a chunk loop, so it costs the pipelines no overlap.
    """
    import cupy as cp
    cp.cuda.Device().synchronize()


def _quiesce(events, inflight):
    """Wait out every chunk still in flight -- the exception path of a run.

    A pipelined run normally waits on each slot's event before it returns.
    If it raises part-way (a bad ``on_chunk``, a KeyboardInterrupt), chunks
    may still be running on the work streams; letting the rig be freed under
    them would hand their buffers back to the pool while a kernel still
    writes into them -- the mirror image of the construction race.
    """
    for slot in inflight:
        events[slot].synchronize()


class _PinnedStaging:
    """Host staging and async copies, shared by the two pipelined rigs.

    The subclass sets ``self.pinned``, ``self._cp`` (the cupy module) and
    ``self._rt`` (``cupy.cuda.runtime``) before using any of these.
    """

    def _host(self, shape, dtype):
        """A staging array — page-locked when pinned, plain numpy otherwise.

        Page-locked staging is what lets the copies actually overlap compute.
        The plain-numpy path exists only to MEASURE what pinning buys: an
        un-pinned host buffer forces the H2D/D2H to go through a driver bounce
        buffer and cannot overlap with compute.
        """
        n = int(np.prod(shape))
        if not self.pinned:
            return np.empty(shape, dtype)
        mem = self._cp.cuda.alloc_pinned_memory(n * np.dtype(dtype).itemsize)
        return np.frombuffer(mem, dtype, n).reshape(shape)

    def _h2d(self, dst, src, stream):
        self._rt.memcpyAsync(dst.data.ptr, src.ctypes.data, src.nbytes,
                             self._rt.memcpyHostToDevice, stream.ptr)

    def _d2h(self, dst, src, stream):
        self._rt.memcpyAsync(dst.ctypes.data, src.data.ptr, dst.nbytes,
                             self._rt.memcpyDeviceToHost, stream.ptr)
