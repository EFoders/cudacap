"""The straight-line CUDA-graph helper: replay is the eager answer, or nothing.

Every claim ``graph.py`` makes is a claim about IDENTITY, not about accuracy —
a replay is the same kernels in the same order over the same buffers, so the
only acceptable result is bit-for-bit.  These tests therefore compare RAW BITS
rather than values: ``torch.equal`` says NaN != NaN, and a routine that
manufactures a NaN on a dead lane (REGMOD does, deliberately, and discards it)
would slip through a value comparison while a bit comparison catches a graph
that replayed against a stale buffer.

WHAT IS NOT TESTED HERE, on purpose: a host sync inside the captured region.
Measured on an RTX 5080, attempting one does not raise a catchable error — it
returns ``cudaErrorStreamCaptureInvalidated`` and every subsequent CUDA call in
the PROCESS fails, so a test for it would take the rest of the session with it.
That asymmetry is exactly why the helper runs `fn` once eagerly before it
captures anything: a routine that is going to fail fails there instead.
"""

import importlib
import sys

import pytest
import torch

from voacap_gpu.batched.graph import StraightLineGraph
from voacap_gpu.batched.profile import _k

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
THIS = __name__                 # resolved lazily by the helper, as a module


def _demo(t):
    """A straight line shaped like Stage B's: libm, `_k` constants, a reduce.

    The reduction matters — REGMOD carries its deferred STOP out of the
    captured region as exactly this kind of per-mode count, so a helper that
    only ever moved (N,) floats would not be exercising the real case.
    """
    x, y = t["x"], t["y"]
    k = lambda v: _k(v, x)                               # noqa: E731
    a = torch.sqrt(torch.clamp(x * x + y * y, min=0.0))
    b = torch.log10(torch.clamp(a, min=k(1.0e-6)))
    c = torch.cos(x) * k(2.0) + torch.sin(y) / k(3.0)
    return {"a": a, "b": b, "c": c, "n": (a > k(1.0)).sum().to(torch.int32)}


def _runner():
    return StraightLineGraph(_demo, "demo", const_modules=(THIS,))


def _inputs(n, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return {"x": torch.randn(n, generator=g).to(device),
            "y": torch.randn(n, generator=g).to(device)}


def _bits(t):
    """The tensor reinterpreted as integers, so NaN == NaN and -0 != +0."""
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _same(a, b):
    return all(torch.equal(_bits(a[n]), _bits(b[n])) for n in a)


def test_cpu_falls_back_to_eager():
    """No CUDA, no graph, same answer — the fallback has to be invisible."""
    r = _runner()
    ins = _inputs(64, "cpu")
    assert _same(r.run(ins, use_cuda_graph=True), _demo(ins))
    assert not r._cache, "a CPU call must not capture anything"


@CUDA
def test_the_graph_path_is_bit_for_bit_the_eager_path():
    r = _runner()
    ins = _inputs(4096, "cuda")
    assert _same(r.run(ins, use_cuda_graph=True), _demo(ins))
    r.reset()


@CUDA
def test_the_replay_tracks_new_input_values():
    """The trap the static buffers exist for.

    A replay re-runs the captured kernels against the CAPTURED ADDRESSES.  If
    the helper handed the caller's tensors straight to `fn` instead of copying
    into buffers it owns, the second call would silently replay the FIRST
    call's numbers — and every field would look plausible.
    """
    r = _runner()
    first, second = _inputs(2048, "cuda", 1), _inputs(2048, "cuda", 2)
    got1 = r.run(first)
    got2 = r.run(second)
    assert _same(got1, _demo(first))
    assert _same(got2, _demo(second))
    assert not _same(got1, got2), "the two input sets are not distinguishable"
    r.reset()


@CUDA
def test_a_new_batch_shape_gets_its_own_graph():
    """The cache key is (names, shapes, dtypes, device); a resize is a recapture."""
    r = _runner()
    small, big = _inputs(128, "cuda", 3), _inputs(5000, "cuda", 4)
    assert _same(r.run(small), _demo(small))
    assert _same(r.run(big), _demo(big))
    assert len(r._cache) == 2
    # ... and going back to the first shape reuses its graph rather than a
    # third one, which is the whole reason the cache is keyed and not a slot
    assert _same(r.run(small), _demo(small))
    assert len(r._cache) == 2
    r.reset()
    assert not r._cache


@CUDA
def test_the_outputs_are_cloned_by_default():
    """The other half of "replay reuses the captured addresses".

    Outputs live in the graph's private pool and the next replay OVERWRITES
    them.  ``copy_outputs=True`` is the default because a caller who keeps a
    result across two calls should not have to know that; ``False`` is the
    documented sharp edge, and it is asserted here so it stays documented.
    """
    r = _runner()
    a, b = _inputs(512, "cuda", 5), _inputs(512, "cuda", 6)
    kept = r.run(a)
    kept_bits = {n: _bits(t).clone() for n, t in kept.items()}
    r.run(b)
    assert all(torch.equal(_bits(kept[n]), kept_bits[n]) for n in kept), \
        "a cloned output was clobbered by the next replay"

    live = r.run(a, copy_outputs=False)
    live_bits = {n: _bits(t).clone() for n, t in live.items()}
    r.run(b)
    assert not all(torch.equal(_bits(live[n]), live_bits[n]) for n in live), \
        "copy_outputs=False is supposed to hand back the live capture buffers"
    r.reset()


@CUDA
def test_the_constant_memo_is_installed_only_around_capture():
    """`_k` is patched to make the capture legal, and must not leak past it."""
    mod = importlib.import_module(THIS)
    before = mod._k
    r = _runner()
    r.run(_inputs(256, "cuda", 7))
    assert mod._k is before, "the memoised _k outlived the capture"
    # the cached constants are held by the entry, because the captured kernels
    # read them by ADDRESS -- dropping them would hand that memory away
    entry = next(iter(r._cache.values()))
    assert entry.constants, "no constants were memoised"
    assert all(t.is_cuda for t in entry.constants.values())
    r.reset()


@CUDA
def test_the_memo_refuses_a_constant_that_was_written_in_place():
    """Memoising makes `_k` results SHARED, so an in-place write becomes action
    at a distance.  Nothing in Stage B does that — but this helper does not own
    the modules it patches, so it checks rather than trusting.
    """
    mod = sys.modules[__name__]

    def mutating(t):
        x = t["x"]
        bad = mod._k(2.0, x)
        bad.add_(mod._k(1.0, x))            # the thing that must not happen
        return {"z": x * bad}

    r = StraightLineGraph(mutating, "mutating", const_modules=(THIS,))
    with pytest.raises(RuntimeError, match="memoised constant"):
        r.run(_inputs(32, "cuda", 8))
    assert mod._k is _k, "the memo was left installed after the failure"
    r.reset()


@CUDA
def test_a_routine_that_raises_fails_before_any_capture_exists():
    """The reason the helper runs `fn` eagerly first.

    A raise DURING capture leaves a half-built graph and, if the raise came
    from a host sync, an unusable stream.  Raising on the eager pre-run costs
    one extra pass and leaves the process healthy, which is asserted here by
    doing more CUDA work afterwards.
    """
    def refuses(t):
        raise ValueError("nope")

    r = StraightLineGraph(refuses, "refuses")
    with pytest.raises(ValueError, match="nope"):
        r.run(_inputs(16, "cuda", 9))
    assert not r._cache, "a failed capture must not be cached"
    assert float(torch.ones(4, device="cuda").sum()) == 4.0


@CUDA
def test_the_inputs_have_to_be_tensors_on_one_device():
    r = _runner()
    ins = _inputs(16, "cuda", 10)
    with pytest.raises(TypeError, match="not a tensor"):
        r.run(dict(ins, scale=2.0))
    with pytest.raises(ValueError, match="is on cpu"):
        r.run(dict(ins, z=torch.zeros(16)))
    with pytest.raises(ValueError, match="no inputs"):
        r.run({})
