"""``PipelinedSetupLuffy`` == ``SetupLuffy``, on every field, at every shape.

The pipelined runner exists to stop the GPU idling while the host stages
parameters and consumes results: each of ``nstreams`` slots has its own device
buffers, its own page-locked staging and its own stream, and there is no
``deviceSynchronize`` in the steady-state loop.  That is exactly the kind of
change that can be subtly wrong in ways a spot check misses -- a chunk
harvested before its event fires, a pinned buffer recycled under a result that
still aliases it, a slot's parameters leaking into its neighbour.

So this compares against the serial ``SetupLuffy`` bit-for-bit, over several
chunk sizes and stream counts, INCLUDING sizes that divide unevenly (the
padding path) and workloads smaller than one chunk.  The cases deliberately
mix ionospheres, config slots, hours and antenna indices so that a slot
confusing any of its per-chunk inputs shows up as a wrong answer rather than
as an identical one.
"""

import numpy as np
import pytest

from voacap_gpu import engine
from voacap_gpu.cuda.driver import (SETUP_CONFIG_DTYPE, SETUP_PARAMS_DTYPE,
                                       IonoSlot, PipelinedSetupLuffy,
                                       SetupLuffy, shared_antenna_tables,
                                       shared_iono_tables)
from voacap_gpu.scalar.redmap import redmap

from tests.cuda.harness import assert_bitexact, requires_cuda

pytestmark = requires_cuda

FREQS_A = (5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 25.0)
FREQS_B = (3.5, 7.1, 14.2)


def _fixture(n):
    """n circuits spanning two ionospheres, two configs, 24 hours, 3 antennas."""
    slots = [
        IonoSlot(redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(120.0)),
                 7, 120.0),
        IonoSlot(redmap(engine._COEFFS_DIR, "URSI", 1, np.float32(40.0)),
                 1, 40.0),
    ]
    sysv = engine.System(fprob=(1.0, 1.0, 1.0, 0.0))
    sysi = engine.System(fprob=(1.0, 1.0, 1.0, 0.70))
    cfgs = np.zeros(2, SETUP_CONFIG_DTYPE)
    cfgs[0] = engine.setup_config(sysv, method=25, freqs=FREQS_A)
    cfgs[1] = engine.setup_config(sysi, method=30, freqs=FREQS_B)

    # three DIFFERENT antenna tables, so ant_idx is load-bearing
    cants = [engine.isotropic_cant(power_kw=1.0, tx_gos=g)
             for g in (0.0, 3.0, -2.5)]
    ant = shared_antenna_tables(cants)

    rng = np.random.default_rng(20260726)
    p = np.zeros(n, SETUP_PARAMS_DTYPE)
    p["tlatd"] = 40.0
    p["tlongd"] = -105.0
    p["rlatd"] = rng.uniform(-80.0, 80.0, n).astype(np.float32)
    p["rlongd"] = rng.uniform(-179.0, 179.0, n).astype(np.float32)
    p["hour"] = rng.integers(1, 25, n)
    p["npsl"] = 0
    p["iono"] = rng.integers(0, 2, n)
    p["cfg"] = p["iono"]                    # config 0 is CCIR, 1 is URSI
    ant_idx = rng.integers(0, 3, n).astype(np.int32)
    return slots, cfgs, ant, ant_idx, p


def _serial(n, slots, iono, cfgs, ant, ant_idx, params):
    rig = SetupLuffy(n, slots, iono, ant)
    return rig.run(params, cfgs, ant_idx)


def _compare(got, exp, tag):
    g_res, g_muf = got
    e_res, e_muf = exp
    for name in e_res.son.dtype.names:
        assert_bitexact(np.ascontiguousarray(g_res.son[name]).ravel(),
                        np.ascontiguousarray(e_res.son[name]).ravel(),
                        f"son.{name} [{tag}]")
    assert_bitexact(g_res.xluf.ravel(), e_res.xluf.ravel(), f"xluf [{tag}]")
    assert np.array_equal(g_res.jmode, e_res.jmode), f"jmode [{tag}]"
    assert np.array_equal(g_res.status, e_res.status), f"status [{tag}]"
    assert_bitexact(g_muf.ravel(), e_muf.ravel(), f"muf3 [{tag}]")


@pytest.mark.parametrize("n,chunk,ns", [
    (2000, 512, 3),         # ragged: 2000 = 3*512 + 464
    (2000, 2000, 2),        # exactly one chunk per stream's first pass
    (1500, 4096, 4),        # workload SMALLER than one chunk
    (4096, 1024, 1),        # one stream: the degenerate, still-must-work case
    (3000, 997, 2),         # nothing divides anything
    (1, 4096, 3),           # ONE circuit: a (1, k) slice is C-contiguous
    (1025, 1024, 2),        # ... and a tail of exactly one row
])
def test_pipelined_matches_serial(n, chunk, ns):
    slots, cfgs, ant, ant_idx, params = _fixture(n)
    iono = shared_iono_tables([s.iono for s in slots])
    exp = _serial(n, slots, iono, cfgs, ant, ant_idx, params)

    pipe = PipelinedSetupLuffy(chunk, ns, slots, iono)
    got = pipe.run(params, cfgs, ant, ant_idx)
    _compare(got, exp, f"n={n} chunk={chunk} ns={ns}")
    assert (exp[0].status == 0).all(), "every circuit succeeded"


def test_on_chunk_sees_every_row_exactly_once_in_order():
    """The streaming callback is the whole point: it must be complete.

    A service reduces each chunk as it lands and never holds the workload, so
    a dropped, doubled or misaligned chunk would silently corrupt the answer
    rather than raise.  This reassembles the callback's pieces and requires
    the result to equal the whole-workload return.
    """
    n, chunk, ns = 2600, 700, 3
    slots, cfgs, ant, ant_idx, params = _fixture(n)
    iono = shared_iono_tables([s.iono for s in slots])
    exp = _serial(n, slots, iono, cfgs, ant, ant_idx, params)

    seen = []
    sndb = np.full((n, 14), np.nan, np.float32)
    muf = np.full((n, 3), np.nan, np.float32)

    def on_chunk(lo, hi, res, m3):
        seen.append((lo, hi))
        assert len(res.son) == hi - lo, "callback row count matches its span"
        sndb[lo:hi] = res.son["sndb"]
        muf[lo:hi] = m3

    out = PipelinedSetupLuffy(chunk, ns, slots, iono).run(
        params, cfgs, ant, ant_idx, on_chunk=on_chunk)
    assert out is None, "streaming mode accumulates nothing"

    covered = sorted(seen)
    assert covered[0][0] == 0 and covered[-1][1] == n
    for (a, b), (c, _d) in zip(covered, covered[1:]):
        assert b == c, f"gap or overlap between {b} and {c}"
    assert not np.isnan(sndb).any(), "every row was delivered"
    assert_bitexact(sndb.ravel(), exp[0].son["sndb"].ravel(), "streamed sndb")
    assert_bitexact(muf.ravel(), exp[1].ravel(), "streamed muf3")


def test_rerunning_a_pipeline_reproduces_itself():
    """Slot buffers are reused across runs and never re-zeroed."""
    n = 1800
    slots, cfgs, ant, ant_idx, params = _fixture(n)
    iono = shared_iono_tables([s.iono for s in slots])
    pipe = PipelinedSetupLuffy(512, 3, slots, iono)
    first = pipe.run(params, cfgs, ant, ant_idx)
    # a different workload in between, to dirty every slot
    slots2, cfgs2, ant2, ai2, p2 = _fixture(999)
    pipe.run(p2, cfgs2, ant2, ai2)
    again = pipe.run(params, cfgs, ant, ant_idx)
    _compare(again, first, "rerun after a foreign workload")
