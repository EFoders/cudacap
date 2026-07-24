"""The pipelined multi-batch runner returns EXACTLY the serial answer.

``PipelinedLuffy`` overlaps host prep / H2D / kernel / D2H / unpack across
several CUDA streams to keep the card busy between batches.  That is a purely
performance change: every field it hands back must be bit-identical to the
serial ``LuffyArena`` path, which ``test_input_set.py`` in turn pins against
the full replay.

The failure mode a pipeline invites is a stream/event mis-ordering that lets a
chunk read stale or half-written device state -- and it shows up first on a
RAGGED FINAL CHUNK, where the padded-to-``chunk`` lanes and the smaller unpack
slice exercise the boundary the steady-state loop never touches.  So the grid
below deliberately mixes workloads that divide evenly by the chunk size with
ones that do not, and spans 1..4 streams (1 stream is still a real pipeline --
prep of chunk N+1 overlaps the GPU work of chunk N -- and it isolates the
per-slot event bookkeeping from cross-stream effects).

The bar is bit-identity on son (every field), xluf, jmode and the status, as
everywhere else in Stage C.
"""

import numpy as np
import pytest

from voacap_gpu.stage_c.driver import (LuffyArena, PipelinedLuffy,
                                       run_luffy_pipelined,
                                       shared_antenna_tables,
                                       shared_iono_tables)
from voacap_gpu.stage_c.state import SON_DTYPE

from tests.stage_a.test_luffy_replay import DECKS
from tests.stage_c.harness import assert_bitexact, requires_cuda
from tests.stage_c.test_luffy_replay import _replay

pytestmark = requires_cuda


def _tile(v, n):
    """The replay's 504 real circuit-hours tiled/truncated to n, in order."""
    n0 = len(v["ipflags"])
    idx = np.arange(n) % n0
    return (v["entries"][idx], v["entries_rays"][idx], v["ipflags"][idx],
            v["iono_idx"][idx], v["ant_idx"][idx])


def _serial(v, bufs, rays, ipf, ii, ai):
    """The reference: one flat serial arena launch over the whole workload."""
    arena = LuffyArena(len(bufs), shared_iono_tables(v["ionos"]),
                       shared_antenna_tables(v["cants"]))
    return arena.run(bufs, rays, ipf, ii, ai)


def _assert_same(got, exp, what):
    g_son, g_xluf, g_jmode, g_st = got
    e_son, e_xluf, e_jmode, e_st = exp
    assert_bitexact(np.asarray(g_st), np.asarray(e_st), f"{what} status")
    assert_bitexact(np.asarray(g_xluf), np.asarray(e_xluf), f"{what} xluf")
    assert_bitexact(np.asarray(g_jmode), np.asarray(e_jmode), f"{what} jmode")
    for name in SON_DTYPE.names:
        assert_bitexact(np.asarray(g_son[name]), np.asarray(e_son[name]),
                        f"{what} son.{name}")


# (total circuit-hours, chunk size, streams); the ragged ones are the point
_CASES = [
    (504, 504, 1),      # one chunk, one stream — the degenerate pipeline
    (504, 128, 2),      # 504 = 3*128 + 120: a ragged tail
    (1000, 256, 3),     # 1000 = 3*256 + 232: ragged, more streams than needed
    (2016, 512, 4),     # 2016 = 3*512 + 480: ragged tail, four streams
    (2016, 672, 3),     # divides evenly (3 chunks) — the non-ragged control
    (777, 100, 2),      # 777 = 7*100 + 77: many chunks, ragged, laps the slots
]


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
@pytest.mark.parametrize("n,chunk,ns", _CASES)
def test_pipelined_matches_serial(n, chunk, ns):
    v = _replay()
    bufs, rays, ipf, ii, ai = _tile(v, n)
    exp = _serial(v, bufs, rays, ipf, ii, ai)

    pipe = PipelinedLuffy(chunk, ns, shared_iono_tables(v["ionos"]),
                          shared_antenna_tables(v["cants"]))
    got = pipe.run(bufs, rays, ipf, ii, ai)
    _assert_same(tuple(got), tuple(exp), f"pipe n={n} chunk={chunk} ns={ns}")

    # reuse must be safe too: a second run through the SAME object, whose
    # per-stream device buffers are never re-zeroed, has to agree again.
    got2 = pipe.run(bufs, rays, ipf, ii, ai)
    _assert_same(tuple(got2), tuple(exp), f"reuse n={n} chunk={chunk} ns={ns}")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_pipelined_worker_thread_matches_serial():
    """The optional host-gather worker pool must not change the answer."""
    v = _replay()
    bufs, rays, ipf, ii, ai = _tile(v, 1000)
    exp = _serial(v, bufs, rays, ipf, ii, ai)
    pipe = PipelinedLuffy(256, 3, shared_iono_tables(v["ionos"]),
                          shared_antenna_tables(v["cants"]), prep_workers=2)
    got = pipe.run(bufs, rays, ipf, ii, ai)
    pipe.close()
    _assert_same(tuple(got), tuple(exp), "worker pool")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_unpinned_path_matches_serial():
    """The measurement fallback (plain host memory) still computes right."""
    v = _replay()
    bufs, rays, ipf, ii, ai = _tile(v, 800)
    exp = _serial(v, bufs, rays, ipf, ii, ai)
    got = run_luffy_pipelined(bufs, rays, ipf, shared_iono_tables(v["ionos"]),
                              ii, shared_antenna_tables(v["cants"]), ai,
                              chunk=256, nstreams=2, pinned=False)
    _assert_same(tuple(got), tuple(exp), "unpinned")
