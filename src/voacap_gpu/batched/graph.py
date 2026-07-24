"""Capture a STRAIGHT-LINE batched routine into a CUDA graph and replay it.

WHAT THIS IS FOR.  ``tools/bench_stage_b.py`` measured ``regmod_batch`` on an
RTX 5080 as FLAT at ~68 ms from one lane to a million -- 68,000 us/lane at
n=1, 0.076 us/lane at n=1,000,000.  None of that is arithmetic: measured
per-op eager dispatch on this machine is 6.25 us, and REGMOD is ~10,000 op
dispatches, which is the whole 65 ms.  A launch costs the same whether its
kernel covers one lane or a million.  A composed driver calls the back half
once per (hop, frequency), roughly 168 times per simulated hour, so those
launches are ~11 s per simulated hour spent doing nothing.  Recording them
once and replaying them collapses the whole sequence into a single launch,
and it is BIT-FOR-BIT identical because it is the same kernels in the same
order over the same buffers.

MEASURED, on REGMOD's straight-line core (see ``regmod.py`` for the full
table and for why only the core goes in): 31-33 ms eager becomes 4.3-5.2 ms
replayed, a 6-8x cut, flat in the lane count up to 100,000 and only then
starting to grow -- which is the compute-bound regime it should be in.

``lockstep.run_lockstep`` already does this for the iterative GO TO ports.
This is the same trick for the routines that are just a long straight line of
elementwise work, and the two differ in exactly one way worth stating:

  * LOCKSTEP HAS TO RESTORE.  Its step advances a state machine, capture needs
    a warmup, and the warmup genuinely RUNS the step -- so every buffer is
    cloned and put back after both the warmup and the capture.
  * NOTHING HERE NEEDS RESTORING.  A straight-line routine is a pure function
    of its inputs: the warmup writes only to throwaway outputs, so running it
    three extra times leaves the inputs exactly as they were.  That is the
    whole reason this helper is thirty lines shorter than it looks like it
    should be, and it is worth saying out loud so nobody adds a restore dance
    here by analogy.

REPLAY REUSES THE CAPTURED BUFFER ADDRESSES.  This is the rule everything else
follows from.  A replay re-runs the recorded kernels against the exact device
pointers that were live at capture time; it does not know what a Python name
points at now.  So this helper OWNS the buffers:

  * inputs are copied into static tensors it allocated, and the captured
    kernels read only those;
  * outputs live in the graph's private memory pool and are OVERWRITTEN by the
    next replay, so ``run`` clones them before handing them back.  Pass
    ``copy_outputs=False`` if the caller is going to consume them immediately
    and wants the bandwidth back -- at a million lanes REGMOD's twenty (7, N)
    fields are 560 MB, so the clone is ~1 ms of pure memory traffic.

THE CACHE IS KEYED ON (names, shapes, dtypes, device).  A different batch size
is a different graph; there is no re-capture-in-place and no shape
specialisation beyond that.  Each entry pins its private pool for the life of
the process, which at large n is not small -- ``reset()`` drops them all.

TWO THINGS ARE NOT CAPTURABLE, and both bite hard rather than gracefully:

  * A HOST SYNC.  ``bool(t.any())``, ``int(t[0])``, ``t[mask]`` -- anything
    that moves a value to the CPU.  Measured: attempting one inside capture
    does not merely raise, it invalidates the capture stream and every
    subsequent CUDA call in the process fails with
    ``cudaErrorStreamCaptureInvalidated``.  That is why validation must run
    EAGERLY BEFORE capture and stay out of the captured region, and why this
    helper runs ``fn`` once eagerly first (see ``_capture``): a routine that
    is going to raise raises there, on a normal stream, with a normal
    traceback, and no graph is built or cached.
  * A HOST-TO-DEVICE COPY OF A CONSTANT.  ``profile._k`` builds each constant
    with ``torch.tensor(..., device=cuda)``, which is a copy from pageable
    host memory and is refused outright during capture.  Hoisting every ``_k``
    out of the captured region is not available to a caller, because the
    constants are built deep inside ``normal.prbmuf_batch`` and friends, so
    ``_memoized_constants`` MEMOISES ``_k`` in the named modules for the
    duration of the warmup and the capture: the warmup populates the cache
    on the host, the capture then hits it and copies nothing.  The memo is
    installed and removed around capture only and never sees the eager path.
    It is safe precisely because ``_k`` is a pure function of (value, device)
    whose result is only ever read -- and because that is an assumption rather
    than a guarantee, ``_check_constants`` verifies after capture that no
    cached constant was mutated in place.

WHAT WAS DELIBERATELY NOT DONE.  No dynamic-shape support (torch's
``make_graphed_callables`` and re-capture-into-the-same-pool both exist; a
batch size is fixed for a whole area run, so paying one capture per size is
the simpler correct thing).  No attempt to capture routines that sync by
design -- ``gain.antenna_gain_batch`` pays two device syncs per call to refuse
the Fortran's out-of-bounds table read, and its module docstring says so; the
right answer there is to hoist those calls out of the captured region, not to
weaken the check.  And no ``torch.compile``: it needs Triton, which has no
Windows build.
"""

import contextlib
import importlib

import torch

from voacap_gpu.batched.profile import _k as _real_k

WARMUP = 3                      # side-stream runs before capture, as lockstep


def _module(mod):
    return importlib.import_module(mod) if isinstance(mod, str) else mod


@contextlib.contextmanager
def _memoized_constants(modules):
    """Make ``_k`` capture-safe in `modules` by memoising it.

    ``_k(value, like)`` allocates a host scalar and copies it to the device;
    that copy is illegal during capture.  Memoising turns every call after the
    first into a lookup of an already-resident tensor, so the warmup pays the
    copies and the capture pays none.  Yields the cache so the caller can hold
    it alive -- the captured kernels read those tensors by ADDRESS, so if the
    cache were dropped the allocator would hand that memory to something else.
    """
    cache = {}                                  # (value, device) -> tensor
    mods = [_module(name) for name in modules]
    saved = [(m, m._k) for m in mods]

    def memo(value, like):
        # the REAL _k, imported here rather than read off the module, so that
        # a nested or repeated install can never memoise a memo
        key = (float(value), like.device)
        got = cache.get(key)
        if got is None:
            got = cache[key] = _real_k(value, like)
        return got

    for m, _ in saved:
        m._k = memo
    try:
        yield cache
    finally:
        for m, old in saved:
            m._k = old


def _check_constants(cache):
    """A memoised constant that got mutated in place would poison the graph.

    Nothing in Stage B writes to a ``_k`` result -- they are operands, never
    destinations -- but "nothing does" is an assumption about code this helper
    does not own, so it is checked rather than trusted.  One device-to-host
    transfer, once per capture.
    """
    if not cache:
        return
    keys = list(cache)
    got = torch.stack([cache[k] for k in keys]).tolist()
    for (value, _dev), have in zip(keys, got):
        # `_k` rounds to float32 before it allocates, so the expectation has
        # to be rounded the same way -- 1.98 is not 1.98 in single precision
        if have != float(torch.tensor(value, dtype=torch.float32)):
            raise RuntimeError(
                f"a memoised constant changed during capture: {value} became "
                f"{have}.  Something in the captured region writes to a _k() "
                "result in place, which the memo turns into action at a "
                "distance -- give that site its own tensor")


class _Entry:
    """One captured shape: the graph, its static inputs, its live outputs."""

    __slots__ = ("graph", "inputs", "outputs", "constants")

    def __init__(self, graph, inputs, outputs, constants):
        self.graph, self.inputs = graph, inputs
        self.outputs, self.constants = outputs, constants


class StraightLineGraph:
    """Capture-and-replay wrapper around a pure dict-in, dict-out routine.

    `fn` takes one ``{name: tensor}`` dict and returns another.  It must be a
    STRAIGHT LINE: no host syncs, no Python branching on device values, no
    iteration count that depends on the data.  It may allocate freely -- during
    capture those allocations come from the graph's private pool.

    `const_modules` names the modules whose ``_k`` needs memoising for the
    capture to be legal (see the module docstring).  Names are resolved lazily,
    so a module can list itself.
    """

    def __init__(self, fn, name=None, const_modules=()):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "fn")
        self.const_modules = tuple(const_modules)
        self._cache = {}

    def reset(self):
        """Drop every captured graph, releasing their private pools."""
        self._cache.clear()

    def run(self, inputs, use_cuda_graph=True, copy_outputs=True):
        """Run `fn` on `inputs`, through a captured graph when that is possible.

        Falls back to a plain eager call on CPU, when CUDA is unavailable, or
        when `use_cuda_graph` is false -- transparently, and returning the same
        bits, because a replay is the same kernels in the same order.
        """
        if not inputs:
            raise ValueError(f"{self.name}: nothing to capture, no inputs")
        dev = next(iter(inputs.values())).device
        if not (use_cuda_graph and dev.type == "cuda"
                and torch.cuda.is_available()):
            return self.fn(inputs)

        key = self._key(inputs, dev)
        entry = self._cache.get(key)
        if entry is None:
            # NOT cached until capture succeeds: a half-built graph is worse
            # than no graph, and the failure path leaves the stream unusable
            entry = self._capture(inputs)
            self._cache[key] = entry
        else:
            # replay reads the CAPTURED addresses, so the new values have to
            # land in the buffers that were live at capture time
            for name, buf in entry.inputs.items():
                buf.copy_(inputs[name])
        entry.graph.replay()
        if not copy_outputs:
            return dict(entry.outputs)
        return {name: t.clone() for name, t in entry.outputs.items()}

    def _key(self, inputs, dev):
        for name, t in inputs.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{self.name}: input {name!r} is not a tensor;"
                                " non-tensor arguments have to be baked into "
                                "`fn` or they cannot be part of the cache key")
            if t.device != dev:
                raise ValueError(f"{self.name}: input {name!r} is on "
                                 f"{t.device}, the rest on {dev}")
        return (tuple(sorted((n, tuple(t.shape), t.dtype)
                             for n, t in inputs.items())), dev)

    def _capture(self, inputs):
        static = {n: torch.empty(t.shape, dtype=t.dtype, device=t.device)
                  for n, t in inputs.items()}
        for n, t in inputs.items():
            static[n].copy_(t)

        with _memoized_constants(self.const_modules) as constants:
            # ONE EAGER RUN FIRST, on the current stream.  It populates the
            # constant memo where a host-to-device copy is still legal, and it
            # is where a routine that has a host sync or a validation raise in
            # it fails -- cleanly, before any capture state exists.  Inside
            # capture the same failure invalidates the stream for the whole
            # process.
            self.fn(static)
            # warmup on a side stream, as capture requires.  No restore
            # afterwards: `fn` is pure, so these runs touched nothing but
            # throwaway outputs.  See the module docstring.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(WARMUP):
                    self.fn(static)
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    outputs = self.fn(static)
            except Exception as exc:               # pragma: no cover
                raise RuntimeError(
                    f"{self.name}: CUDA graph capture failed.  The eager run "
                    "above it succeeded, so this is capture-specific -- a "
                    "host sync or a host-to-device copy inside the captured "
                    "region.  The capture stream is now invalid; the process "
                    "has to be restarted") from exc
        _check_constants(constants)

        if not isinstance(outputs, dict) or not all(
                isinstance(t, torch.Tensor) for t in outputs.values()):
            raise TypeError(f"{self.name}: `fn` must return a dict of tensors")
        # capture RECORDS the kernels, it does not run them, so the output
        # buffers hold whatever the warmup left.  The caller's replay fills
        # them; nothing here may read them first.
        return _Entry(graph, static, outputs, constants)
