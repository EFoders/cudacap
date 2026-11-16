"""The device-setup path: raw 32-byte parameters in, results out.

``SetupLuffy`` (one resident batch), ``setup_and_run`` (one-shot) and
``PipelinedSetupLuffy`` (chunked across streams, with slim outputs and the
on-device grid reduction).  No per-circuit host build at all: the setup
kernel builds each CircuitState in place and LUFFY runs on it directly.
"""

import numpy as np

from voacap_gpu.cuda.abi import (_CIRC_WORDS, _RAYS_WORDS, _RESULT_COLS,
                                 _RESULT_WORDS, CIRCUIT_DTYPE, F32, I32,
                                 SETUP_CONFIG_DTYPE, SETUP_PARAMS_DTYPE)
from voacap_gpu.cuda.checks import (_PinnedStaging, _check_setup_params,
                                    _drain_default_stream, _n_antenna_sets,
                                    _quiesce, _require_dtype, _require_rows)
from voacap_gpu.cuda.kernels import (_RESULT_MAP, SMOOTHSTORE_BYTES,
                                     _device_map, _rays_views, _xfer,
                                     load_kernel, load_setup_kernel)
from voacap_gpu.cuda.results import (CIRCUIT_WORDS, GRID_SON_OFFSETS,
                                     LuffyResults, SlimResults, _split_results,
                                     fetch_results, fetch_slim_results,
                                     slim_spec)
from voacap_gpu.cuda.state import SON_DTYPE
from voacap_gpu.cuda.tables import setup_iono_tables


class SetupLuffy:
    """Raw circuit parameters -> results, with the host doing NO per-circuit work.

    Two launches over one set of resident device buffers.  ``setup_batch``
    (cuda/kernels/setup.cuh) builds each CircuitState in place from 32 bytes of
    parameters and the shared tables; ``luffy_batch`` then runs on exactly
    those states without them ever crossing the bus.  What actually travels is

        up    32 bytes a circuit-hour, plus one SETUP_CONFIG per SYSTEM card
        down  1,700 bytes a circuit-hour (``LuffyResults``) + 12 for MUF/FOT/HPF

    against 3,228 up and 1,700 down for ``LuffyArena``, and -- the part that
    matters far more -- against a Python ``build_circuit`` per circuit-hour.

    THE STATE BUFFERS ARE ZEROED ONCE AND NEVER AGAIN, exactly as
    ``LuffyArena`` does, and for the same reason: nothing outside
    ``UPLOAD_FIELDS`` is read before it is written, and ``setup_batch`` writes
    every one of those fields.  Leftovers from the previous submission are
    values this pair of kernels itself produced.

    ``n`` is FIXED for the lifetime of the object, because /RAYS/'s SoA column
    stride is the launch's thread count (raysview.cuh).  Submit a shorter batch
    and the tail lanes simply compute an answer nobody reads -- but pad them
    with a REPEAT of a real row, never zeros: a zeroed ``layers.km`` hangs the
    area loop.  ``run`` does that padding itself.

    Usage::

        slots = [IonoSlot(redmap(dir, "CCIR", 7, 100.0), 7, 100.0)]
        rig = SetupLuffy(n, slots, shared_iono_tables([s.iono for s in slots]),
                         shared_antenna_tables(cants))
        res, muf3 = rig.run(params, cfgs, ant_idx)
    """

    def __init__(self, n, slots, iono, ant=None, block=32, warp=False,
                 warps_per_block=4, fast_math=False):
        import cupy as cp

        self.n = int(n)
        self.block = int(block)
        self._cp = cp
        self._setup_fn = load_setup_kernel(fast_math).get_function(
            "setup_batch")
        self._setup_grid = ((self.n + self.block - 1) // self.block,)

        # per-thread state, allocated and zeroed ONCE
        self._d_circ = cp.zeros(self.n * CIRCUIT_DTYPE.itemsize, np.uint8)
        self._d_soa = cp.zeros((_RAYS_WORDS, self.n), np.int32)
        self._d_status = cp.full(self.n, -1, np.int32)
        self._d_ipflag = cp.zeros(self.n, np.int32)
        self._d_muf3 = cp.zeros((self.n, 3), np.float32)
        self._d_prm = cp.zeros(self.n * SETUP_PARAMS_DTYPE.itemsize, np.uint8)
        self._d_ai = cp.zeros(self.n, np.int32)
        self._d_cfg = None                       # sized on first submission
        self._last_cfgs = None

        # shared read-only tables — one copy for the object's lifetime.  The
        # IONOSPHERE is resident because it is big (REDMAP's maps) and depends
        # only on (coeff, month, SSN); the ANTENNAS are not, because a type-13
        # pattern is AZIMUTH-DEPENDENT and a grid therefore builds one
        # /cantenna/ per cell.  So they are set per submission -- see
        # `set_antennas` -- with an optional table here for the fixed-antenna
        # caller who would rather pay the copy once.
        st = setup_iono_tables(slots)
        d = {k: cp.asarray(v) for k, v in iono.items()}
        s = {k: cp.asarray(v) for k, v in st.items()}
        self._tables = (d, s)                    # keep the copies alive
        self._setup_shared = (s["ab"], s["ikim"], s["anew"], s["bnew"],
                              s["achi"], s["bchi"], d["fakp"], d["fakmap"],
                              d["hmym"], d["fakabp"], d["abmap"], s["f2d"],
                              s["month"], s["ssn"])

        d_nang, d_ang, d_ifob = _rays_views(self._d_soa)
        # IONO_IDX for LUFFY is the SetupParams `iono` column, which the host
        # already has; it is uploaded per submission alongside the parameters.
        self._d_ii = cp.zeros(self.n, np.int32)
        self._luffy_head = (self._d_circ, d_nang, d_ang, d_ifob,
                            self._d_ipflag, self._d_ii, self._d_ai,
                            d["fam"], d["dud"], d["fakp"], d["fakmap"],
                            d["hmym"], d["fakabp"], d["abmap"], d["sys"],
                            d["perr"], d["f2d111"])
        # an iono index has to be valid for BOTH table sets it addresses
        self._n_iono = min(len(slots), int(np.size(iono["f2d111"])))
        self._n_ant = 0
        self._luffy_args = None
        if ant is not None:
            self._install_antennas(ant)
        if warp:
            wpb = int(warps_per_block)
            self._luffy_fn = load_kernel(fast_math=fast_math).get_function(
                "luffy_batch_warp")
            self._luffy_grid = ((self.n + wpb - 1) // wpb,)
            self._luffy_block = (32 * wpb,)
            self._luffy_shared = wpb * SMOOTHSTORE_BYTES
        else:
            self._luffy_fn = load_kernel(fast_math=fast_math).get_function(
                "luffy_batch")
            self._luffy_grid = ((self.n + self.block - 1) // self.block,)
            self._luffy_block = (self.block,)
            self._luffy_shared = 0
        # Submissions run on the current stream, as the fills above did, so
        # a caller who stays on one stream is ordered anyway; this makes the
        # rig safe to drive from ANY stream (see _drain_default_stream).
        _drain_default_stream()

    @property
    def state(self):
        """The ``(d_circ, d_soa, d_status, n)`` tuple the fetchers take."""
        return self._d_circ, self._d_soa, self._d_status, self.n

    def set_antennas(self, ant):
        """Install a ``shared_antenna_tables`` dict for the next submissions.

        Separate from the ionosphere because it changes on a different clock:
        the ionosphere is per (coefficient set, month, SSN) and a whole request
        shares one, while a type-13 pattern is azimuth-dependent and a grid
        builds one /cantenna/ per CELL.  ``numants`` is the last-dimension
        extent gain.cuh indexes with, so it is baked into the argument tuple
        and the tuple is rebuilt here rather than patched.
        """
        self._install_antennas(ant)
        _drain_default_stream()
        return self

    def _install_antennas(self, ant):
        """`set_antennas` without the drain: for ``upload(ant=...)``, whose
        copies go to the same stream as the launch that follows them."""
        cp = self._cp
        a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
        self._ant = a                            # keep the copies alive
        self._n_ant = _n_antenna_sets(ant)
        self._luffy_args = self._luffy_head + (
            a["array"], a["aeff"], a["iats"], a["xfqs"], a["xfqe"],
            a["offazim"], a["pwrdba"], np.int32(ant["numants"]),
            self._d_status, np.int32(self.n))

    def _validate(self, params, cfgs, ant_idx):
        """Input checks shared by `upload` and `run` (which pads after).

        Returns the params to use: the same array, or a copy with
        out-of-range longitudes wrapped (see ``_check_setup_params``)."""
        _require_dtype(params, SETUP_PARAMS_DTYPE, "params")
        _require_dtype(cfgs, SETUP_CONFIG_DTYPE, "cfgs")
        if params.ndim != 1:
            raise ValueError(f"params must be 1-D, got shape {params.shape}")
        _require_rows(ant_idx, params.shape[0], "ant_idx")
        if self._luffy_args is None:
            raise ValueError(
                "no antenna tables: pass ant= to the constructor, to "
                "upload/run, or call set_antennas")
        return _check_setup_params(params, ant_idx, self._n_iono,
                                   cfgs.shape[0], self._n_ant)

    def upload(self, params, cfgs, ant_idx, ant=None):
        """One batch's raw parameters into the resident buffers.  No launch."""
        if ant is not None:
            self._install_antennas(ant)
        params = np.ascontiguousarray(params)
        cfgs = np.ascontiguousarray(cfgs)
        ant_idx = np.ascontiguousarray(ant_idx, I32)
        params = self._validate(params, cfgs, ant_idx)
        if params.shape[0] != self.n:
            raise ValueError(f"upload takes exactly n={self.n} rows, got "
                             f"{params.shape[0]} (run() pads a short batch)")
        return self._upload(params, cfgs, ant_idx)

    def _upload(self, params, cfgs, ant_idx):
        cp = self._cp
        self._d_prm.set(params.view(np.uint8).ravel())
        self._d_cfg = cp.asarray(cfgs.view(np.uint8).ravel())
        self._d_ii.set(np.ascontiguousarray(params["iono"], I32))
        self._d_ai.set(np.ascontiguousarray(ant_idx, I32))
        self._d_status.fill(-1)
        self._last_cfgs = cfgs                   # for a slim fetch's slots
        return self

    def launch_setup(self):
        """Build the CircuitStates in place.  Split out so a test can read the
        built state back BEFORE LUFFY mutates it, and so a benchmark can time
        the two halves apart."""
        self._setup_fn(self._setup_grid, (self.block,), (
            self._d_circ, *_rays_views(self._d_soa)[:2], self._d_ipflag,
            self._d_muf3, self._d_prm, self._d_cfg) + self._setup_shared
            + (np.int32(self.n),))
        return self

    def launch_kernel(self):
        """Run LUFFY over whatever the setup kernel left in the buffers."""
        self._luffy_fn(self._luffy_grid, self._luffy_block, self._luffy_args,
                       shared_mem=self._luffy_shared)
        return self

    def launch(self):
        """setup_batch, then luffy_batch, over the resident buffers."""
        return self.launch_setup().launch_kernel()

    def results(self, outputs=None, slots=None, rows=None):
        """``(LuffyResults, muf3)`` — muf3 is (n, 3): ALLMUF, FOT, HPF at IT.

        ``outputs`` (default None: the whole /SON/, XLUF and JMODE, as ever)
        names the columns to gather instead -- see ``SlimSpec`` -- and returns
        a ``SlimResults``; ``slots`` defaults to the slots the last uploaded
        configs use.  ``rows`` limits a slim fetch to the first ``rows``.
        """
        if outputs is None:
            res = fetch_results(self.state)
            return res, self._cp.asnumpy(self._d_muf3)
        spec = slim_spec(outputs, self._last_cfgs, slots)
        rows = self.n if rows is None else int(rows)
        res = fetch_slim_results(self.state, spec, rows)
        return res, self._cp.asnumpy(self._d_muf3[:rows])

    def run(self, params, cfgs, ant_idx, ant=None, outputs=None, slots=None):
        """A batch of ANY length <= n -> ``(LuffyResults, muf3)``.

        Short batches are padded up to ``n`` by REPEATING the first row (never
        zeros; see the class docstring) and the padding is dropped on the way
        out, so a caller never sees the fixed width.

        ``outputs=("sndb", "dbw", ...)`` gathers and transfers only those
        /SON/ columns (at the slots in use, or ``slots``) and returns a
        ``SlimResults``; status and muf3 always come back.  The default,
        None, is the full ``LuffyResults``, unchanged.
        """
        if ant is not None:
            self._install_antennas(ant)
        params = np.ascontiguousarray(params)
        cfgs = np.ascontiguousarray(cfgs)
        ant_idx = np.ascontiguousarray(ant_idx, I32)
        params = self._validate(params, cfgs, ant_idx)
        c = params.shape[0]
        if not 0 < c <= self.n:
            raise ValueError(f"run takes 1..n={self.n} rows, got {c}")
        if c < self.n:
            pad = self.n - c
            params = np.concatenate([params, np.repeat(params[:1], pad)])
            ant_idx = np.concatenate([ant_idx, np.repeat(ant_idx[:1], pad)])
        self._upload(params, cfgs, ant_idx).launch()
        if outputs is not None:
            return self.results(outputs, slots, rows=c)
        res, muf3 = self.results()
        if c == self.n:
            return res, muf3
        return (LuffyResults(res.son[:c], res.xluf[:c], res.jmode[:c],
                             res.status[:c]), muf3[:c])


def setup_and_run(params, cfgs, slots, iono, ant, ant_idx, block=32,
                  warp=False, warps_per_block=4, fast_math=False):
    """One-shot device-setup round trip -> ``(LuffyResults, muf3)``.

    Allocates for exactly this batch, so for a sweep of several batches keep a
    ``SetupLuffy`` and call ``.run`` repeatedly instead.
    """
    rig = SetupLuffy(len(params), slots, iono, ant, block=block, warp=warp,
                     warps_per_block=warps_per_block, fast_math=fast_math)
    return rig.run(params, cfgs, ant_idx)


class PipelinedSetupLuffy(_PinnedStaging):
    """``SetupLuffy`` across several CUDA streams, so the card never waits.

    ``SetupLuffy`` runs one batch at a time and synchronises to read it back,
    which means the GPU is idle for every host-side stage: the parameter fill,
    the result extraction, and -- in a service -- the JSON encoding of the
    previous answer.  Measured on a world grid that left the card busy about a
    quarter of the time.

    This runner splits a workload into fixed-width CHUNKS and gives each of
    ``nstreams`` slots its OWN device buffers, its own page-locked host staging
    and its own stream.  While chunk N's kernels run, chunk N+1's parameters
    are already crossing the bus and chunk N-1's results are being consumed on
    the host.  Ordering is by per-slot CUDA events; there is no
    ``deviceSynchronize`` in the steady-state loop, which is the whole point --
    a single sync anywhere serialises every stream behind it.

    THE STRIDE CONSTRAINT is the same one ``PipelinedLuffy`` documents:
    ``raysview.cuh`` addresses /RAYS/ element ``o`` of thread ``t`` at
    ``o * n + t``, so the SoA column stride IS the launch width.  Every slot is
    therefore allocated at ``chunk`` and a short final chunk is padded up by
    REPEATING a real parameter row -- never zeros, which would hang the
    sample-area loop.  Only the real rows are handed back.

    THE SHARED TABLES ARE UPLOADED ONCE for the whole run and shared by every
    slot: the ionosphere because it depends only on (coefficients, month, SSN),
    and the ANTENNAS because a job's /cantenna/ set is per-CELL, not per
    circuit -- a grid revisits the same cell at every hour and frequency, so
    one table serves the entire sweep instead of being re-sent per chunk.

    ``on_chunk`` is how a caller keeps up.  It is invoked as
    ``on_chunk(lo, hi, results, muf3)`` the moment a chunk lands, with the
    results for ``params[lo:hi]`` -- so a service can reduce, encode or stream
    that chunk while the next ones are still on the card.  Supply it and
    nothing is accumulated (a monthly-median world sweep is 400,000
    circuit-hours and there is no reason to hold all of it); omit it and
    ``run`` returns the whole ``(LuffyResults, muf3)`` as ``SetupLuffy`` does.

    Usage::

        pipe = PipelinedSetupLuffy(8192, 3, slots, iono)
        pipe.run(params, cfgs, ant, ant_idx, on_chunk=reduce_into_grid)
    """

    def __init__(self, chunk, nstreams, slots=None, iono=None, block=32,
                 pinned=True, fast_math=False):
        import cupy as cp

        self.chunk = int(chunk)
        self.ns = max(1, int(nstreams))
        self.block = int(block)
        self.pinned = bool(pinned)
        self._cp = cp
        self._rt = cp.cuda.runtime
        self._setup_fn = load_setup_kernel(fast_math).get_function(
            "setup_batch")
        self._luffy_fn = load_kernel(fast_math=fast_math).get_function(
            "luffy_batch")
        self._grid = ((self.chunk + self.block - 1) // self.block,)
        self._gather = _xfer("gather_words")
        self._resmap = _device_map("result", _RESULT_MAP)

        # The ionosphere and the antennas are BOTH per-run here, unlike
        # SetupLuffy where the ionosphere is resident.  A long-lived service
        # rig has to serve whatever month/SSN/coefficient set the next request
        # names, and the whole REDMAP + VIRTIM block is under 100 KB, so
        # re-uploading it per job costs nothing worth keeping state for.
        # `_iono_key` skips even that when consecutive jobs share one.
        self._iono_dev = None
        self._setup_shared = None
        self._iono_key = None
        self._ant = None
        self._ant_args = None
        self._n_iono = 0
        self._n_ant = 0
        if slots is not None and iono is not None:
            self._install_ionosphere(slots, iono)

        # per-slot device + pinned staging, zeroed once
        self.streams, self.events = [], []
        self.d_circ, self.d_soa, self.d_status = [], [], []
        self.d_ipflag, self.d_muf3, self.d_prm = [], [], []
        self.d_ai, self.d_cfg, self.d_gath = [], [], []
        self.d_ii = []
        self.p_prm, self.p_ai, self.p_gath = [], [], []
        self.p_status, self.p_muf3, self.p_ii = [], [], []
        self._rays = []
        pw = SETUP_PARAMS_DTYPE.itemsize // 4
        for _ in range(self.ns):
            self.streams.append(cp.cuda.Stream(non_blocking=True))
            self.events.append(cp.cuda.Event(disable_timing=True))
            self.d_circ.append(cp.zeros(self.chunk * CIRCUIT_DTYPE.itemsize,
                                        np.uint8))
            soa = cp.zeros((_RAYS_WORDS, self.chunk), np.int32)
            self.d_soa.append(soa)
            self.d_status.append(cp.full(self.chunk, -1, np.int32))
            self.d_ipflag.append(cp.zeros(self.chunk, np.int32))
            self.d_muf3.append(cp.zeros((self.chunk, 3), np.float32))
            self.d_prm.append(cp.zeros((self.chunk, pw), np.int32))
            self.d_ai.append(cp.zeros(self.chunk, np.int32))
            # LUFFY's iono_idx is the SetupParams `iono` column, but it has to
            # reach the kernel as a CONTIGUOUS int array -- a strided column
            # view of the parameter block is not one.
            self.d_ii.append(cp.zeros(self.chunk, np.int32))
            self.d_cfg.append(None)
            self.d_gath.append(cp.empty((self.chunk, _RESULT_WORDS), np.int32))
            self.p_prm.append(self._host((self.chunk, pw), np.int32))
            self.p_ai.append(self._host(self.chunk, np.int32))
            self.p_ii.append(self._host(self.chunk, np.int32))
            self.p_gath.append(self._host((self.chunk, _RESULT_WORDS),
                                          np.int32))
            self.p_status.append(self._host(self.chunk, np.int32))
            self.p_muf3.append(self._host((self.chunk, 3), np.float32))
            self._rays.append(_rays_views(soa))
        # Every fill and table copy above went to the CURRENT stream, and the
        # slot streams are non-blocking: without this, a first chunk on
        # recycled pool memory can run before the zero-fill lands (reproduced
        # as wrong rows with status 0).  See _drain_default_stream.
        _drain_default_stream()

    def set_ionosphere(self, slots, iono, key=None):
        """Install the REDMAP + VIRTIM tables for the next run.

        ``key`` lets a caller skip a redundant re-upload when the next job
        names the same (coefficients, month, SSN); pass anything hashable that
        identifies the table set, or None to always upload.
        """
        if key is not None and key == self._iono_key:
            return self
        self._install_ionosphere(slots, iono, key)
        _drain_default_stream()       # the copies are read on slot streams
        return self

    def _install_ionosphere(self, slots, iono, key=None):
        cp = self._cp
        st = setup_iono_tables(slots)
        d = {k: cp.asarray(v) for k, v in iono.items()}
        s = {k: cp.asarray(v) for k, v in st.items()}
        self._tables = (d, s)                    # keep the copies alive
        self._iono_dev = d
        self._setup_shared = (s["ab"], s["ikim"], s["anew"], s["bnew"],
                              s["achi"], s["bchi"], d["fakp"], d["fakmap"],
                              d["hmym"], d["fakabp"], d["abmap"], s["f2d"],
                              s["month"], s["ssn"])
        # an iono index has to be valid for BOTH table sets it addresses
        self._n_iono = min(len(slots), int(np.size(iono["f2d111"])))
        self._iono_key = key

    def set_antennas(self, ant):
        self._install_antennas(ant)
        _drain_default_stream()       # the copies are read on slot streams
        return self

    def _install_antennas(self, ant):
        cp = self._cp
        a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
        self._ant = a
        self._n_ant = _n_antenna_sets(ant)
        self._ant_args = (a["array"], a["aeff"], a["iats"], a["xfqs"],
                          a["xfqe"], a["offazim"], a["pwrdba"],
                          np.int32(ant["numants"]))

    def _enqueue(self, slot, c):
        """Every device stage of one chunk, async, on this slot's stream."""
        cp = self._cp
        st = self.streams[slot]
        d = self._iono_dev
        chunk = self.chunk
        d_nang, d_ang, d_ifob = self._rays[slot]
        with st:
            self._h2d(self.d_prm[slot], self.p_prm[slot], st)
            self._h2d(self.d_ai[slot], self.p_ai[slot], st)
            self._h2d(self.d_ii[slot], self.p_ii[slot], st)
            self.d_status[slot].fill(-1)
            self._setup_fn(self._grid, (self.block,), (
                self.d_circ[slot], d_nang, d_ang, self.d_ipflag[slot],
                self.d_muf3[slot], self.d_prm[slot], self.d_cfg[slot])
                + self._setup_shared + (np.int32(chunk),))
            self._luffy_fn(self._grid, (self.block,), (
                self.d_circ[slot], d_nang, d_ang, d_ifob,
                self.d_ipflag[slot], self.d_ii[slot],
                self.d_ai[slot],
                d["fam"], d["dud"], d["fakp"], d["fakmap"], d["hmym"],
                d["fakabp"], d["abmap"], d["sys"], d["perr"], d["f2d111"])
                + self._ant_args + (self.d_status[slot], np.int32(chunk)))
            self._enqueue_results(slot, c, st)
        self.events[slot].record(st)

    # --- the result path: what leaves the card for one chunk ---------------

    def _enqueue_results(self, slot, c, st):
        """Gather + D2H (full or slim), and the optional grid reduction.

        Full (``outputs=None``, no reducer): exactly the historic gather of
        every result word and the three whole-chunk copies.  Slim: only the
        spec's words, and only the ``c`` real rows of each copy.  A reducer
        consumes the chunk ON THE DEVICE, straight out of ``d_circ``; its
        kernels are chained chunk-to-chunk by ``_red_prev`` so the
        order-dependent scan sees the chunks in sweep order even though the
        chunks themselves run on different streams.
        """
        spec, red = self._spec, self._reduce
        if red is not None:
            if self._red_prev is not None:
                st.wait_event(self._red_prev)
            lo = red.offset + self._cur[0]
            red.enqueue_chunk(self.d_circ[slot].view(np.int32), CIRCUIT_WORDS,
                              GRID_SON_OFFSETS, self.d_status[slot], lo, c, st)
            ev = self._red_events[slot]
            ev.record(st)
            self._red_prev = ev
        if spec is None:
            if red is not None:
                return                   # reduce-only: nothing leaves the card
            chunk = self.chunk
            rtot = _RESULT_WORDS * chunk
            self._gather(((rtot + 255) // 256,), (256,),
                         (self.d_circ[slot].view(np.int32), self.d_gath[slot],
                          self._resmap, I32(_RESULT_WORDS), I32(_CIRC_WORDS),
                          np.int64(rtot)))
            self._d2h(self.p_gath[slot], self.d_gath[slot], st)
            self._d2h(self.p_status[slot], self.d_status[slot], st)
            self._d2h(self.p_muf3[slot], self.d_muf3[slot], st)
            return
        m = spec.words
        if m:
            d_g = self.d_gath[slot].ravel()[:c * m].reshape(c, m)
            tot = m * c
            self._gather(((tot + 255) // 256,), (256,),
                         (self.d_circ[slot].view(np.int32), d_g,
                          _device_map(spec.key, spec.map), I32(m),
                          I32(_CIRC_WORDS), np.int64(tot)))
            self._d2h(self.p_gath[slot].reshape(-1)[:c * m].reshape(c, m),
                      d_g, st)
        self._d2h(self.p_status[slot][:c], self.d_status[slot][:c], st)
        self._d2h(self.p_muf3[slot][:c], self.d_muf3[slot][:c], st)

    def _stage(self, slot, params, iono_idx, ant_idx, a, b):
        """Fill this slot's pinned staging; pad a short chunk by repetition."""
        c = b - a
        pw = SETUP_PARAMS_DTYPE.itemsize // 4
        words = params.view(np.int32).reshape(-1, pw)
        self.p_prm[slot][:c] = words[a:b]
        self.p_ai[slot][:c] = ant_idx[a:b]
        self.p_ii[slot][:c] = iono_idx[a:b]
        if c < self.chunk:
            # REPEAT a real row: a zeroed layers.km hangs the area loop, so the
            # padding lanes have to describe a legal circuit even though their
            # answer is discarded.
            self.p_prm[slot][c:] = self.p_prm[slot][0]
            self.p_ai[slot][c:] = self.p_ai[slot][0]
            self.p_ii[slot][c:] = self.p_ii[slot][0]
        return c

    def _harvest(self, slot, a, b):
        """This slot's results OUT of the pinned staging, as owned arrays.

        The staging is reused by the next chunk the moment this returns, so
        nothing handed back may alias it.  The block is therefore COPIED here
        rather than leaning on ``_split_results``'s per-column
        ``ascontiguousarray``: those calls copy a strided slice of a wider
        block *usually*, but a slice of shape ``(1, k)`` is already
        C-contiguous whatever its row stride, so a single-row chunk got a VIEW
        and the next chunk overwrote the answer underneath it.  That is not
        hypothetical -- the assert below caught it on a one-circuit request.
        """
        c = b - a
        spec = self._spec
        if spec is not None:
            # slim: the chunk landed as a (c, words) prefix of the staging
            m = spec.words
            gath = np.array(self.p_gath[slot].reshape(-1)[:c * m]
                            ).reshape(c, m)                # owned, not pinned
            res = spec.decode(gath, np.array(self.p_status[slot][:c], I32))
            for arr in (res.son, res.xluf, res.jmode):
                assert arr is None or not np.shares_memory(
                    arr, self.p_gath[slot]), "slim result aliases staging"
            return res, np.array(self.p_muf3[slot][:c])
        gath = np.array(self.p_gath[slot][:c])          # owned, not pinned
        son, xluf, jmode = _split_results(gath, c)
        for arr, what in ((son, "son"), (xluf, "xluf"), (jmode, "jmode")):
            assert not np.shares_memory(arr, self.p_gath[slot]), (
                f"{what} still aliases the pinned staging buffer")
        res = LuffyResults(son, xluf, jmode,
                           np.array(self.p_status[slot][:c], I32))
        return res, np.array(self.p_muf3[slot][:c])

    def run(self, params, cfgs, ant, ant_idx, on_chunk=None, outputs=None,
            slots=None, reduce=None):
        """A whole workload, chunked and overlapped across the streams.

        Returns ``(LuffyResults, muf3)`` for the whole workload, or ``None``
        when ``on_chunk`` is supplied (the caller consumed each chunk as it
        landed and nothing was kept).

        ``outputs`` (default None: everything, as before) names the /SON/
        columns to bring back -- e.g. ``("sndb", "dbw", "dbu", "reliab",
        "nhp", "mode")``, optionally ``"xluf"`` / ``"jmode"`` -- and only
        those words, at the slots in use (``slots_in_use(cfgs)``, or
        ``slots``), are gathered and copied.  Results, including what
        ``on_chunk`` receives, are then ``SlimResults``; status and muf3
        always come back.  ``outputs=()`` brings back status and muf3 only.

        ``reduce`` is a ``gridjob.DeviceCellReducer``: each chunk is reduced
        into its per-cell state ON THE DEVICE, in sweep order.  With
        ``outputs=None`` nothing per-row crosses the bus at all; ``on_chunk``,
        if given, is then called as ``on_chunk(lo, hi, None, None)`` (for
        progress) and ``run`` returns None -- read the answer with
        ``reduce.finalise()``.  Give ``outputs`` too and ``on_chunk`` gets
        those slim rows as well.
        """
        cp = self._cp
        spec = None if outputs is None else slim_spec(outputs, cfgs, slots)
        if spec is not None:
            _device_map(spec.key, spec.map)     # upload before the drain below
        if reduce is not None:
            if not hasattr(self, "_red_events"):
                self._red_events = [cp.cuda.Event(disable_timing=True)
                                    for _ in range(self.ns)]
        self._spec, self._reduce, self._red_prev = spec, reduce, None
        try:
            out = self._run(params, cfgs, ant, ant_idx, on_chunk, spec, reduce)
        finally:
            self._spec, self._reduce, self._red_prev = None, None, None
        return out

    def _run(self, params, cfgs, ant, ant_idx, on_chunk, spec, reduce):
        cp = self._cp
        params = np.ascontiguousarray(params)
        _require_dtype(params, SETUP_PARAMS_DTYPE, "params")
        cfgs = np.ascontiguousarray(cfgs)
        _require_dtype(cfgs, SETUP_CONFIG_DTYPE, "cfgs")
        ant_idx = np.ascontiguousarray(ant_idx, I32)
        if params.ndim != 1:
            raise ValueError(f"params must be 1-D, got shape {params.shape}")
        n = params.shape[0]
        _require_rows(ant_idx, n, "ant_idx")
        if self._setup_shared is None:
            raise ValueError("no ionosphere: pass slots/iono to the "
                             "constructor or call set_ionosphere")
        # checked against the antenna set being INSTALLED, before anything
        # about the rig changes
        params = _check_setup_params(params, ant_idx, self._n_iono,
                                     cfgs.shape[0], _n_antenna_sets(ant))
        self._install_antennas(ant)
        d_cfg = cp.asarray(cfgs.view(np.uint8).ravel())
        for k in range(self.ns):
            self.d_cfg[k] = d_cfg          # tiny and read-only: shared
        # both uploads above went to the current stream; the slot streams
        # read them (see _drain_default_stream).  Once per run, before the
        # first chunk, so nothing in flight is serialised by it.
        _drain_default_stream()
        iono_idx = np.ascontiguousarray(params["iono"], I32)

        bounds = [(a, min(a + self.chunk, n))
                  for a in range(0, n, self.chunk)]
        # reduce-only: nothing per-row comes back, so there is nothing to keep
        # or harvest -- a chunk is "landed" when its event fires.
        quiet = reduce is not None and spec is None
        harvest = ((lambda slot, a, b: (None, None)) if quiet
                   else self._harvest)
        keep = [] if (on_chunk is None and not quiet) else None
        inflight = {}                       # slot -> (a, b) awaiting harvest
        try:
            for k, (a, b) in enumerate(bounds):
                slot = k % self.ns
                if slot in inflight:        # reuse: drain that slot first
                    pa, pb = inflight.pop(slot)
                    self.events[slot].synchronize()
                    out = harvest(slot, pa, pb)
                    if keep is not None:
                        keep.append((pa, pb) + out)
                    elif on_chunk is not None:
                        on_chunk(pa, pb, out[0], out[1])
                self._stage(slot, params, iono_idx, ant_idx, a, b)
                self._cur = (a, b)
                self._enqueue(slot, b - a)
                inflight[slot] = (a, b)
            for slot in sorted(inflight, key=lambda s: inflight[s][0]):
                a, b = inflight[slot]
                self.events[slot].synchronize()
                out = harvest(slot, a, b)
                del inflight[slot]
                if keep is not None:
                    keep.append((a, b) + out)
                elif on_chunk is not None:
                    on_chunk(a, b, out[0], out[1])
        finally:
            _quiesce(self.events, inflight)
        if reduce is not None:
            reduce.advance(n)
        if keep is None:
            return None

        keep.sort(key=lambda t: t[0])
        if spec is not None:
            return self._assemble_slim(spec, n, keep)
        son = np.empty(n, SON_DTYPE)
        xluf = np.empty((n, _RESULT_COLS["muf.xluf"][1]), F32)
        jmode = np.empty(n, I32)
        status = np.empty(n, I32)
        muf3 = np.empty((n, 3), F32)
        for a, b, res, m3 in keep:
            son[a:b] = res.son
            xluf[a:b] = res.xluf
            jmode[a:b] = res.jmode
            status[a:b] = res.status
            muf3[a:b] = m3
        return LuffyResults(son, xluf, jmode, status), muf3

    @staticmethod
    def _assemble_slim(spec, n, keep):
        """The slim chunks -> one whole-workload ``(SlimResults, muf3)``."""
        son = None if spec.son_dtype is None else np.empty(n, spec.son_dtype)
        xluf = (np.empty((n, _RESULT_COLS["muf.xluf"][1]), F32)
                if "xluf" in spec.extra else None)
        jmode = np.empty(n, I32) if "jmode" in spec.extra else None
        status = np.empty(n, I32)
        muf3 = np.empty((n, 3), F32)
        for a, b, res, m3 in keep:
            if son is not None:
                son[a:b] = res.son
            if xluf is not None:
                xluf[a:b] = res.xluf
            if jmode is not None:
                jmode[a:b] = res.jmode
            status[a:b] = res.status
            muf3[a:b] = m3
        return SlimResults(son, xluf, jmode, status, spec), muf3
