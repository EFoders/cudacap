"""The host-built LUFFY path: packed CircuitStates in, results out.

``stage_luffy`` / ``launch_luffy`` / ``luffy_results`` (one-shot), the
resident ``LuffyArena``, the stream-pipelined ``PipelinedLuffy``, and
``run_luffy``, which packs Stage A ``Circuit`` objects itself.
"""

import numpy as np

from voacap_gpu.cuda.abi import (_CIRC_WORDS, _RAYS_LEN, _RAYS_OFF,
                                 _RAYS_UPLOAD_RUNS, _RAYS_WORDS, _RESULT_COLS,
                                 _RESULT_WORDS, _UPLOAD_IDX, _UPLOAD_RUNS,
                                 CIRCUIT_DTYPE, F32, I32)
from voacap_gpu.cuda.checks import (_PinnedStaging, _check_luffy_inputs,
                                    _drain_default_stream, _quiesce)
from voacap_gpu.cuda.kernels import (_RESULT_MAP, _UPLOAD_MAP,
                                     SMOOTHSTORE_BYTES,
                                     _circuit_inputs_to_device, _device_map,
                                     _rays_inputs_to_soa, _rays_to_soa,
                                     _rays_views, _write_circuit_inputs,
                                     _write_rays_inputs, _xfer, load_kernel)
from voacap_gpu.cuda.results import (LuffyResults, _split_results,
                                     fetch_luffy, fetch_results)
from voacap_gpu.cuda.state import SON_DTYPE
from voacap_gpu.cuda.tables import (pack_circuit, shared_antenna_tables,
                                    shared_iono_tables, unpack_circuit)


def stage_luffy(bufs, rays_bufs, ipflags, iono, iono_idx, ant, ant_idx,
                block=32, warp=False, warps_per_block=4, upload="full",
                groups=1, fast_math=False):
    """Move one batch to the device; return the staged launch, not the result.

    Split out of ``launch_luffy`` so a benchmark can time the KERNEL without
    the bus.  That distinction turned out to matter a great deal: at 32k
    threads the whole ``launch_luffy`` takes ~600 ms of which the kernel is
    ~125 ms, so timing the round trip measures PCIe, not compute.

    ``upload`` picks how much of that state actually crosses the bus:

    * ``"full"`` (the default, and what every existing caller gets) sends all
      55,652 bytes per thread, so ``fetch_luffy`` can hand back every field
      and the replay can compare all of them.
    * ``"inputs"`` sends only ``UPLOAD_FIELDS`` + ``RAYS_UPLOAD_FIELDS`` --
      3,228 bytes per thread -- and zeroes the rest on the device.  The
      kernel computes the same answer; see driver.py's docstring and
      ``tests/cuda/test_input_set.py``.  Fetch it with ``fetch_results``:
      ``fetch_luffy`` would return zeros for the scratch the host never sent.

    ``warp`` selects ``luffy_batch_warp`` (cuda/luffy_warp.cuh) — one WARP per
    circuit-hour, lane L on frequency L+1 — instead of the one-thread-per-
    circuit-hour ``luffy_batch``.  The two are bit-identical; see
    ``tests/cuda/test_luffy_warp.py``.

    Returns ``(run, state)`` — call ``run()`` to launch (as many times as you
    like; the kernel mutates the staged state in place), and pass ``state`` to
    ``fetch_luffy`` to read the answer back.
    """
    import cupy as cp

    if upload not in ("full", "inputs"):
        raise ValueError(f"upload must be 'full' or 'inputs', got {upload!r}")
    bufs = np.ascontiguousarray(bufs)
    rays_bufs = np.ascontiguousarray(rays_bufs)
    n = _check_luffy_inputs(bufs, rays_bufs, ipflags, iono_idx, ant_idx)
    if upload == "full":
        d_circ = cp.asarray(bufs.view(np.uint8))
        d_soa = _rays_to_soa(rays_bufs, n)
    else:
        d_circ = _circuit_inputs_to_device(bufs, n)
        d_soa = _rays_inputs_to_soa(rays_bufs, n)
    d_nang, d_ang, d_ifob = _rays_views(d_soa)
    d = {k: cp.asarray(v) for k, v in iono.items()}
    a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
    d_ipflag = cp.asarray(np.ascontiguousarray(ipflags, I32))
    d_ii = cp.asarray(np.ascontiguousarray(iono_idx, I32))
    d_ai = cp.asarray(np.ascontiguousarray(ant_idx, I32))
    d_status = cp.full(n, -1, np.int32)

    args = (d_circ, d_nang, d_ang, d_ifob, d_ipflag, d_ii, d_ai,
            d["fam"], d["dud"], d["fakp"], d["fakmap"], d["hmym"],
            d["fakabp"], d["abmap"], d["sys"], d["perr"], d["f2d111"],
            a["array"], a["aeff"], a["iats"], a["xfqs"], a["xfqe"],
            a["offazim"], a["pwrdba"], np.int32(ant["numants"]), d_status,
            np.int32(n))

    if warp:
        fn = load_kernel(groups, fast_math).get_function("luffy_batch_warp")
        wpb = int(warps_per_block)
        # one warp carries `groups` circuit-hours, so a block carries
        # wpb * groups of them and needs that many SmoothStore rows
        per_block = wpb * int(groups)
        grid = ((n + per_block - 1) // per_block,)
        blk = (32 * wpb,)
        shared = per_block * SMOOTHSTORE_BYTES

        def run():
            fn(grid, blk, args, shared_mem=shared)
    else:
        fn = load_kernel(fast_math=fast_math).get_function("luffy_batch")
        grid = ((n + block - 1) // block,)

        def run():
            fn(grid, (block,), args)

    return run, (d_circ, d_soa, d_status, n)


def launch_luffy(bufs, rays_bufs, ipflags, iono, iono_idx, ant, ant_idx,
                 block=32, warp=False, warps_per_block=4, upload="full",
                 groups=1, fast_math=False):
    """Run N packed circuit-hours, one thread each.

    bufs:      (N,) array of CIRCUIT_DTYPE — the ENTRY states, not mutated
    rays_bufs: (N,) array of RAYS_DTYPE — /RAYS/, likewise not mutated; it
               goes to the device transposed to structure-of-arrays
    ipflags:   (N,) int
    iono/ant:  the dicts ``shared_iono_tables`` / ``shared_antenna_tables``
               return
    iono_idx / ant_idx: (N,) int, which shared table each thread uses

    Returns ``(out_bufs, out_rays, status)``: the mutated states and the
    per-thread LUFFY status (0 = success, non-zero = the refusal a callee
    reported).
    """
    run, state = stage_luffy(bufs, rays_bufs, ipflags, iono, iono_idx, ant,
                             ant_idx, block=block, warp=warp,
                             warps_per_block=warps_per_block, upload=upload,
                             groups=groups, fast_math=fast_math)
    run()
    return fetch_luffy(state)


def luffy_results(bufs, rays_bufs, ipflags, iono, iono_idx, ant, ant_idx,
                  block=32, warp=False, warps_per_block=4,
                  groups=1, fast_math=False):
    """The production round trip: inputs up, results back, nothing else.

    Same arguments as ``launch_luffy``, same numbers out -- but 3,228 bytes a
    circuit-hour go up instead of 55,652 and 1,700 come back instead of
    55,652.  Returns ``LuffyResults``.
    """
    run, state = stage_luffy(bufs, rays_bufs, ipflags, iono, iono_idx, ant,
                             ant_idx, block=block, warp=warp,
                             warps_per_block=warps_per_block,
                             upload="inputs", groups=groups,
                             fast_math=fast_math)
    run()
    return fetch_results(state)


class LuffyArena:
    """Device state for a FIXED batch size, allocated and zeroed once.

    ``luffy_results`` still pays two costs that have nothing to do with the
    answer: 1.8 GB of ``cudaMalloc`` + memset for the scratch every call (at
    32,256 circuit-hours), and a fresh copy of the shared REDMAP and antenna
    tables.  Neither depends on the batch.  An arena holds both across
    submissions, so a caller sweeping many batches of the same size pays them
    once.

    NOTHING IS RE-ZEROED BETWEEN SUBMISSIONS, and that is the interesting
    part.  Batch k+1 starts with batch k's scratch still in place.  That is
    safe for exactly the reason the whole reduction is safe -- no field
    outside ``UPLOAD_FIELDS`` is read before it is written -- and it is
    *safer* than uninitialised memory would be, because every leftover word
    is a value this kernel itself produced, so no integer is out of its own
    range.  ``tests/cuda/test_input_set.py`` runs the same batch through an
    arena twice and requires the second answer to equal the first.

    Usage::

        arena = LuffyArena(n, iono, ant)
        for batch in batches:
            res = arena.run(bufs, rays_bufs, ipflags, iono_idx, ant_idx)
    """

    def __init__(self, n, iono, ant, block=32, warp=False, warps_per_block=4):
        import cupy as cp

        self.n = int(n)
        self._d_circ = cp.zeros(self.n * CIRCUIT_DTYPE.itemsize, np.uint8)
        self._d_soa = cp.zeros((_RAYS_WORDS, self.n), np.int32)
        self._d_status = cp.full(self.n, -1, np.int32)
        self._d_ipflag = cp.zeros(self.n, np.int32)
        self._d_ii = cp.zeros(self.n, np.int32)
        self._d_ai = cp.zeros(self.n, np.int32)
        d = {k: cp.asarray(v) for k, v in iono.items()}
        a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
        self._tables = (d, a)                    # keep the copies alive
        d_nang, d_ang, d_ifob = _rays_views(self._d_soa)
        self._args = (self._d_circ, d_nang, d_ang, d_ifob, self._d_ipflag,
                      self._d_ii, self._d_ai,
                      d["fam"], d["dud"], d["fakp"], d["fakmap"], d["hmym"],
                      d["fakabp"], d["abmap"], d["sys"], d["perr"],
                      d["f2d111"], a["array"], a["aeff"], a["iats"],
                      a["xfqs"], a["xfqe"], a["offazim"], a["pwrdba"],
                      np.int32(ant["numants"]), self._d_status,
                      np.int32(self.n))
        if warp:
            wpb = int(warps_per_block)
            self._fn = load_kernel().get_function("luffy_batch_warp")
            self._grid = ((self.n + wpb - 1) // wpb,)
            self._block = (32 * wpb,)
            self._shared = wpb * SMOOTHSTORE_BYTES
        else:
            self._fn = load_kernel().get_function("luffy_batch")
            self._grid = ((self.n + block - 1) // block,)
            self._block = (block,)
            self._shared = 0
        # the fills and table copies above went to the current stream; a
        # caller may submit from another one (see _drain_default_stream)
        _drain_default_stream()

    @property
    def state(self):
        """The ``(d_circ, d_soa, d_status, n)`` tuple the fetchers take."""
        return self._d_circ, self._d_soa, self._d_status, self.n

    def upload(self, bufs, rays_bufs, ipflags, iono_idx, ant_idx):
        """One batch's INPUTS into the resident buffers.  No launch."""
        bufs = np.ascontiguousarray(bufs)
        rays_bufs = np.ascontiguousarray(rays_bufs)
        _check_luffy_inputs(bufs, rays_bufs, ipflags, iono_idx, ant_idx,
                            n=self.n)
        _write_circuit_inputs(self._d_circ, bufs, self.n)
        _write_rays_inputs(self._d_soa, rays_bufs, self.n)
        self._d_ipflag.set(np.ascontiguousarray(ipflags, I32))
        self._d_ii.set(np.ascontiguousarray(iono_idx, I32))
        self._d_ai.set(np.ascontiguousarray(ant_idx, I32))
        self._d_status.fill(-1)
        return self

    def launch(self):
        """Run the kernel over whatever is currently in the buffers."""
        self._fn(self._grid, self._block, self._args, shared_mem=self._shared)
        return self

    def submit(self, bufs, rays_bufs, ipflags, iono_idx, ant_idx):
        """Upload one batch's inputs into the resident buffers and launch."""
        return self.upload(bufs, rays_bufs, ipflags, iono_idx,
                           ant_idx).launch()

    def results(self):
        return fetch_results(self.state)

    def run(self, bufs, rays_bufs, ipflags, iono_idx, ant_idx):
        return self.submit(bufs, rays_bufs, ipflags, iono_idx,
                           ant_idx).results()


class PipelinedLuffy(_PinnedStaging):
    """Overlap host prep / H2D / kernel / D2H / unpack across CUDA streams.

    ``LuffyArena`` removed the per-batch allocation and memset, but every stage
    of a submission still runs SERIALLY: the host gather blocks, then the H2D
    blocks, then the kernel, then the D2H, then the unpack, and the GPU sits
    idle for everything that is not the kernel.  On a real multi-batch run that
    shows up as 30-50% utilisation -- the card saturates for the ~124 ms it is
    computing and waits for the host the rest of the time.

    This runner splits a large workload into fixed-size CHUNKS and runs the
    five stages of adjacent chunks CONCURRENTLY on ``nstreams`` CUDA streams,
    each with its own device buffers and its own PAGE-LOCKED host staging
    buffers.  While chunk N computes on the GPU, chunk N+1's host gather and
    H2D are already in flight on another stream and chunk N-1's D2H and unpack
    are finishing.  Ordering is by per-slot CUDA events -- there is no
    ``deviceSynchronize`` anywhere in the steady-state loop, which is what lets
    the stages overlap.

    THE STRIDE CONSTRAINT, and why every chunk is padded to ``chunk``.
    ``cuda/raysview.cuh`` addresses /RAYS/ element ``o`` of thread ``t`` at
    ``o * n + t`` where ``n`` is the launch's thread count.  So the SoA rays
    buffer's column stride IS ``n``; a ragged final chunk launched with a
    smaller ``n`` would read every rays row at the wrong offset.  Rather than
    re-stride per chunk, the runner allocates every buffer at width ``chunk``
    and pads the final chunk up to ``chunk`` by REPEATING real input rows
    (never zeros -- a zeroed ``layers.km`` hangs the area loop, see the input
    set).  The padded lanes compute a valid-but-ignored answer; only the first
    ``c`` rows of each chunk are unpacked.  This is exactly the ragged case the
    correctness test hammers.

    Bit-identity: each stream's device buffers are zeroed ONCE at construction
    and never re-zeroed, the same reuse ``LuffyArena`` relies on and the same
    input-set guarantee makes safe.  The pipelined result is bit-identical to
    the serial arena on every field; ``tests/cuda/test_pipeline.py`` pins
    it across several chunk sizes, stream counts and a ragged tail.

    Usage::

        pipe = PipelinedLuffy(chunk, nstreams, iono, ant)
        res = pipe.run(bufs, rays_bufs, ipflags, iono_idx, ant_idx)
    """

    _RAYS_IN = 42                     # nang(1) + ang(41): the only rays inputs

    def __init__(self, chunk, nstreams, iono, ant, block=32, pinned=True,
                 prep_workers=0, fast_math=False):
        import cupy as cp

        assert _RAYS_UPLOAD_RUNS == ((0, self._RAYS_IN),), _RAYS_UPLOAD_RUNS
        self.chunk = int(chunk)
        self.ns = int(nstreams)
        self.block = int(block)
        self.pinned = bool(pinned)
        self.fast_math = bool(fast_math)
        self._cp = cp
        self._rt = cp.cuda.runtime
        self._m = len(_UPLOAD_IDX)
        # fast_math is the opt-in native-f32 libm (cuda/f32math.cuh); it is NOT
        # bit-exact against Stage A, so a pipelined run inherits exactly the
        # deviation tests/cuda/test_fastmath.py pins and nothing more.
        self._fn = load_kernel(fast_math=fast_math).get_function("luffy_batch")
        self._grid = ((self.chunk + block - 1) // block,)
        self._scatter = _xfer("scatter_words")
        self._gather = _xfer("gather_words")
        self._upmap = _device_map("upload", _UPLOAD_MAP)
        self._resmap = _device_map("result", _RESULT_MAP)

        # shared, read-only tables — one copy for the whole run
        d = {k: cp.asarray(v) for k, v in iono.items()}
        a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
        self._tables = (d, a)
        self._numants = np.int32(ant["numants"])
        self._shared = (d["fam"], d["dud"], d["fakp"], d["fakmap"], d["hmym"],
                        d["fakabp"], d["abmap"], d["sys"], d["perr"],
                        d["f2d111"], a["array"], a["aeff"], a["iats"],
                        a["xfqs"], a["xfqe"], a["offazim"], a["pwrdba"])

        # a worker pool for the host gather, if asked for
        self._pool = None
        if prep_workers > 0:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=int(prep_workers))

        # per-slot device + pinned staging, zeroed once
        rw, rwn = _RAYS_WORDS, _RESULT_WORDS
        self.streams, self.events = [], []
        self.d_circ, self.d_soa, self.d_status = [], [], []
        self.d_src, self.d_gath = [], []
        self.d_ipf, self.d_ii, self.d_ai = [], [], []
        self.p_src, self.p_rays, self.p_gath, self.p_status = [], [], [], []
        self.p_ipf, self.p_ii, self.p_ai = [], [], []
        self._rays_views = []
        for _ in range(self.ns):
            self.streams.append(cp.cuda.Stream(non_blocking=True))
            self.events.append(cp.cuda.Event(disable_timing=True))
            dc = cp.zeros(self.chunk * CIRCUIT_DTYPE.itemsize, np.uint8)
            ds = cp.zeros((rw, self.chunk), np.int32)
            self.d_circ.append(dc)
            self.d_soa.append(ds)
            self.d_status.append(cp.full(self.chunk, -1, np.int32))
            self.d_src.append(cp.empty((self.chunk, self._m), np.int32))
            self.d_gath.append(cp.empty((self.chunk, rwn), np.int32))
            self.d_ipf.append(cp.empty(self.chunk, np.int32))
            self.d_ii.append(cp.empty(self.chunk, np.int32))
            self.d_ai.append(cp.empty(self.chunk, np.int32))
            self.p_src.append(self._host((self.chunk, self._m), np.int32))
            self.p_rays.append(self._host((self._RAYS_IN, self.chunk),
                                          np.int32))
            self.p_gath.append(self._host((self.chunk, rwn), np.int32))
            self.p_status.append(self._host(self.chunk, np.int32))
            self.p_ipf.append(self._host(self.chunk, np.int32))
            self.p_ii.append(self._host(self.chunk, np.int32))
            self.p_ai.append(self._host(self.chunk, np.int32))
            o, ln = _RAYS_OFF, _RAYS_LEN
            d_nang = ds[o["nang"]:o["nang"] + ln["nang"]]
            d_ang = ds[o["ang"]:o["ang"] + ln["ang"]].view(np.float32)
            d_ifob = ds[o["ifob"]:o["ifob"] + ln["ifob"]]
            self._rays_views.append((d_nang, d_ang, d_ifob))
        # Every fill and table copy above went to the CURRENT stream, and the
        # work streams are non-blocking: without this, a first chunk on
        # recycled pool memory can run before the zero-fill lands.  See
        # _drain_default_stream.
        _drain_default_stream()

    def _args(self, slot, n):
        dc = self.d_circ[slot]
        d_nang, d_ang, d_ifob = self._rays_views[slot]
        return (dc, d_nang, d_ang, d_ifob, self.d_ipf[slot], self.d_ii[slot],
                self.d_ai[slot]) + self._shared + (
                self._numants, self.d_status[slot], np.int32(n))

    def _prep(self, slot, words, words_rays, ipf, ii, ai, a, b):
        """Host gather of one chunk into that slot's page-locked buffers.

        Pure CPU numpy: the 19 contiguous input runs concatenated into the
        compacted ``(chunk, m)`` block (the concat form, not a 409-word
        fancy-index -- the fancy path costs 5x on the host), the 42 rays input
        words transposed, and the three index vectors.  The ragged tail is
        padded up to ``chunk`` with a repeat of the chunk's first real row.
        """
        c = b - a
        ps, pr = self.p_src[slot], self.p_rays[slot]
        np.concatenate([words[a:b, s:s + ln] for s, ln in _UPLOAD_RUNS],
                       axis=1, out=ps[:c])
        pr[:, :c] = words_rays[a:b, :self._RAYS_IN].T
        self.p_ipf[slot][:c] = ipf[a:b]
        self.p_ii[slot][:c] = ii[a:b]
        self.p_ai[slot][:c] = ai[a:b]
        if c < self.chunk:                       # pad the ragged tail
            ps[c:] = ps[0]
            pr[:, c:] = pr[:, :1]
            self.p_ipf[slot][c:] = self.p_ipf[slot][0]
            self.p_ii[slot][c:] = self.p_ii[slot][0]
            self.p_ai[slot][c:] = self.p_ai[slot][0]
        return c

    def _enqueue(self, slot):
        """All five device stages of one chunk, async, on this slot's stream."""
        st = self.streams[slot]
        chunk = self.chunk
        with st:
            # H2D: compacted inputs, the 42 rays words, the three index vectors
            self._h2d(self.d_src[slot], self.p_src[slot], st)
            self._h2d(self.d_soa[slot][:self._RAYS_IN], self.p_rays[slot], st)
            self._h2d(self.d_ipf[slot], self.p_ipf[slot], st)
            self._h2d(self.d_ii[slot], self.p_ii[slot], st)
            self._h2d(self.d_ai[slot], self.p_ai[slot], st)
            # scatter the compacted block into CircuitState
            tot = self._m * chunk
            self._scatter(((tot + 255) // 256,), (256,),
                          (self.d_circ[slot].view(np.int32), self.d_src[slot],
                           self._upmap, I32(self._m), I32(_CIRC_WORDS),
                           np.int64(tot)))
            self.d_status[slot].fill(-1)
            # the kernel, one thread per (padded) circuit-hour
            self._fn(self._grid, (self.block,), self._args(slot, chunk))
            # gather the result words and bring them + the status home
            rtot = _RESULT_WORDS * chunk
            self._gather(((rtot + 255) // 256,), (256,),
                         (self.d_circ[slot].view(np.int32), self.d_gath[slot],
                          self._resmap, I32(_RESULT_WORDS), I32(_CIRC_WORDS),
                          np.int64(rtot)))
            self._d2h(self.p_gath[slot], self.d_gath[slot], st)
            self._d2h(self.p_status[slot], self.d_status[slot], st)
        self.events[slot].record(st)

    def _unpack(self, slot, a, b, out):
        """That slot's finished results into the caller's output arrays."""
        c = b - a
        son_o, xluf_o, jmode_o, status_o = out
        son, xluf, jmode = _split_results(self.p_gath[slot][:c], c)
        son_o[a:b] = son
        xluf_o[a:b] = xluf
        jmode_o[a:b] = jmode
        status_o[a:b] = self.p_status[slot][:c]

    def run(self, bufs, rays_bufs, ipflags, iono_idx, ant_idx):
        """A whole workload, chunked and pipelined -> ``LuffyResults``.

        ``bufs`` / ``rays_bufs`` may be ANY length; they are split into
        ``chunk``-sized pieces and the pieces overlap across the streams.  The
        returned tables carry one row per input circuit-hour, in input order,
        bit-identical to the serial path.
        """
        bufs = np.ascontiguousarray(bufs)
        rays_bufs = np.ascontiguousarray(rays_bufs)
        n = _check_luffy_inputs(bufs, rays_bufs, ipflags, iono_idx, ant_idx)
        words = bufs.view(np.int32).reshape(n, _CIRC_WORDS)
        words_rays = rays_bufs.view(np.int32).reshape(n, _RAYS_WORDS)
        ipf = np.ascontiguousarray(ipflags, I32)
        ii = np.ascontiguousarray(iono_idx, I32)
        ai = np.ascontiguousarray(ant_idx, I32)

        son = np.empty(n, SON_DTYPE)
        xluf = np.empty((n, _RESULT_COLS["muf.xluf"][1]), F32)
        jmode = np.empty(n, I32)
        status = np.empty(n, I32)
        out = (son, xluf, jmode, status)

        bounds = [(a, min(a + self.chunk, n))
                  for a in range(0, n, self.chunk)]
        inflight = {}                       # slot -> (a, b) awaiting unpack
        prep_future = {}                    # slot -> Future, if a pool is used
        try:
            for k, (a, b) in enumerate(bounds):
                slot = k % self.ns
                if slot in inflight:        # reuse: wait, then drain that slot
                    pa, pb = inflight.pop(slot)
                    self.events[slot].synchronize()
                    self._unpack(slot, pa, pb, out)
                if self._pool is not None:
                    fut = prep_future.pop(slot, None)
                    if fut is not None:
                        fut.result()
                    self._prep(slot, words, words_rays, ipf, ii, ai, a, b)
                else:
                    self._prep(slot, words, words_rays, ipf, ii, ai, a, b)
                self._enqueue(slot)
                inflight[slot] = (a, b)
            for slot in list(inflight):
                a, b = inflight[slot]
                self.events[slot].synchronize()
                self._unpack(slot, a, b, out)
                del inflight[slot]
        finally:
            _quiesce(self.events, inflight)
        return LuffyResults(son, xluf, jmode, status)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def run_luffy_pipelined(bufs, rays_bufs, ipflags, iono, iono_idx, ant, ant_idx,
                        chunk=32256, nstreams=2, block=32, pinned=True,
                        fast_math=False):
    """Convenience wrapper: build a ``PipelinedLuffy``, run once, return it.

    Allocates per-stream device + pinned buffers for ``chunk`` circuit-hours,
    so for a real sweep of many workloads keep the ``PipelinedLuffy`` object
    and call ``.run`` repeatedly rather than paying the allocation each time.
    """
    pipe = PipelinedLuffy(chunk, nstreams, iono, ant, block=block,
                          pinned=pinned, fast_math=fast_math)
    return pipe.run(bufs, rays_bufs, ipflags, iono_idx, ant_idx)


def run_luffy(circuits, ipflags, block=32, write_back=True):
    """Pack, launch, unpack.  One thread per circuit-hour.

    circuits: a sequence of Stage A ``Circuit`` objects, each carrying its own
    ``iono`` and ``cant``; identical table objects are shared on the device
    (identity-keyed, so the replay's one-IonoState-per-deck costs one copy per
    deck rather than one per hour).

    Returns ``(circuits, status)``.  With ``write_back`` the mutated device
    state is written back into the given Circuit objects in place.
    """
    ipflags = np.asarray(ipflags, I32)
    if len(ipflags) != len(circuits):
        raise ValueError(f"{len(ipflags)} ipflags for {len(circuits)} "
                         "circuits")

    ionos, iono_idx, seen_i = [], [], {}
    cants, ant_idx, seen_a = [], [], {}
    for c in circuits:
        ki = id(c.iono)
        if ki not in seen_i:
            seen_i[ki] = len(ionos)
            ionos.append(c.iono)
        iono_idx.append(seen_i[ki])
        ka = id(c.cant)
        if ka not in seen_a:
            seen_a[ka] = len(cants)
            cants.append(c.cant)
        ant_idx.append(seen_a[ka])

    packed = [pack_circuit(c) for c in circuits]
    bufs = np.stack([p[0] for p in packed])
    rays_bufs = np.stack([p[1] for p in packed])
    out, out_rays, status = launch_luffy(
        bufs, rays_bufs, ipflags, shared_iono_tables(ionos),
        np.array(iono_idx, I32), shared_antenna_tables(cants),
        np.array(ant_idx, I32), block=block)
    if write_back:
        for c, b, r in zip(circuits, out, out_rays):
            unpack_circuit(b, r, c)
    return circuits, status
