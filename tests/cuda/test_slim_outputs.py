"""``outputs=`` slim results == the same columns of the full result, bitwise.

The slim path gathers a different word map with the same ``gather_words``
kernel and copies only the real rows of each chunk.  What can go wrong is all
bookkeeping: a field's word offset, the slot count, the row layout the host
decode views the block as, a short chunk's prefix of the pinned staging, a
slim block aliasing the staging that the next chunk overwrites.  So every
requested column is compared bit for bit with the full result on the same
workload, over the pipelined rig's awkward shapes (ragged chunks, one row,
workloads smaller than a chunk) and the serial rig's padded batches -- and
the default (``outputs=None``) is checked to still be the full, unchanged
``LuffyResults``.
"""

import numpy as np
import pytest

from voacap_gpu import gridjob
from voacap_gpu.cuda import driver
from voacap_gpu.cuda.driver import (SON_DTYPE, LuffyResults,
                                       PipelinedSetupLuffy, SetupLuffy,
                                       SlimResults, shared_iono_tables,
                                       slim_spec, slots_in_use)

from tests.cuda.harness import assert_bitexact, requires_cuda
from tests.cuda.test_setup_pipeline import _fixture

pytestmark = requires_cuda

SERVICE = gridjob.SERVICE_OUTPUTS


def _check_slim(slim, full, outputs, nslots, tag, rows=None):
    """Every requested column of ``slim`` == the full result's, bitwise."""
    g_res, g_muf = slim
    e_res, e_muf = full
    assert isinstance(g_res, SlimResults), tag
    rows = slice(None) if rows is None else rows
    names = [o for o in outputs if o in SON_DTYPE.names]
    if names:
        assert g_res.son.dtype.names == tuple(names), tag
    else:
        assert g_res.son is None, tag
    for f in names:
        e = e_res.son[f][rows]
        if e.ndim == 2:
            assert g_res.son[f].shape == (e.shape[0], nslots), (f, tag)
            e = e[:, :nslots]
        assert_bitexact(np.ascontiguousarray(g_res.son[f]).ravel(),
                        np.ascontiguousarray(e).ravel(), f"son.{f} [{tag}]")
    if "xluf" in outputs:
        assert_bitexact(g_res.xluf.ravel(), e_res.xluf[rows].ravel(),
                        f"xluf [{tag}]")
    else:
        assert g_res.xluf is None
    if "jmode" in outputs:
        assert np.array_equal(g_res.jmode, e_res.jmode[rows]), tag
    else:
        assert g_res.jmode is None
    assert np.array_equal(g_res.status, e_res.status[rows]), f"status {tag}"
    assert_bitexact(g_muf.ravel(), e_muf[rows].ravel(), f"muf3 [{tag}]")


def test_slots_in_use_and_bytes():
    slots, cfgs, *_ = _fixture(4)
    # FREQS_A has 7 frequencies (slots 1..7), FREQS_B 3: slot 0 + 7 = 8
    assert slots_in_use(cfgs) == 8
    assert slots_in_use(cfgs[1:]) == 4
    spec = slim_spec(SERVICE, cfgs)
    assert spec.nslots == 8 and spec.words == 6 * 8
    assert spec.bytes_per_row == 4 * 48 + 16
    # the default result path's per-row cost, for comparison
    assert driver.RESULT_BYTES_PER_THREAD + 12 == 1712
    with pytest.raises(ValueError):
        slim_spec(("sndb", "nope"), cfgs)
    with pytest.raises(ValueError):
        slim_spec(("sndb", "sndb"), cfgs)
    with pytest.raises(ValueError):
        slim_spec(("sndb",), cfgs, slots=15)


@pytest.mark.parametrize("n,chunk,ns", [
    (2000, 512, 3),         # ragged: 2000 = 3*512 + 464
    (1500, 4096, 4),        # workload SMALLER than one chunk
    (3000, 997, 2),         # nothing divides anything
    (1, 4096, 3),           # ONE circuit
    (1025, 1024, 2),        # ... and a tail of exactly one row
])
def test_pipelined_slim_equals_full(n, chunk, ns):
    slots, cfgs, ant, ant_idx, params = _fixture(n)
    iono = shared_iono_tables([s.iono for s in slots])
    pipe = PipelinedSetupLuffy(chunk, ns, slots, iono)
    full = pipe.run(params, cfgs, ant, ant_idx)
    assert type(full[0]) is LuffyResults, "the default is the full result"
    assert full[0].son.dtype == SON_DTYPE

    nsl = slots_in_use(cfgs)
    for outputs, slots_kw, k in (
            (SERVICE, None, nsl),
            (("reliab", "xluf", "d50r", "sndb", "jmode"), None, nsl),
            (("mode", "snxx"), 14, 14),              # every slot
            (("dbu",), 13, 13),                      # keep the MUF slot (12)
            ((), None, nsl),                         # status + muf3 only
            (("xluf",), None, nsl)):
        got = pipe.run(params, cfgs, ant, ant_idx, outputs=outputs,
                       slots=slots_kw)
        _check_slim(got, full, outputs, k,
                    f"n={n} chunk={chunk} ns={ns} {outputs}")
    # and the full path is still the full path after slim runs on the rig
    again = pipe.run(params, cfgs, ant, ant_idx)
    for name in SON_DTYPE.names:
        assert_bitexact(np.ascontiguousarray(again[0].son[name]).ravel(),
                        np.ascontiguousarray(full[0].son[name]).ravel(),
                        f"full after slim: {name}")


def test_on_chunk_gets_slim_rows_in_order():
    n, chunk, ns = 2600, 700, 3
    slots, cfgs, ant, ant_idx, params = _fixture(n)
    iono = shared_iono_tables([s.iono for s in slots])
    pipe = PipelinedSetupLuffy(chunk, ns, slots, iono)
    full = pipe.run(params, cfgs, ant, ant_idx)

    seen, pieces = [], []

    def on_chunk(lo, hi, res, m3):
        assert isinstance(res, SlimResults)
        assert len(res.son) == hi - lo == len(res.status) == len(m3)
        seen.append((lo, hi))
        pieces.append((lo, hi, res, m3))

    assert pipe.run(params, cfgs, ant, ant_idx, on_chunk=on_chunk,
                    outputs=SERVICE) is None
    assert seen == sorted(seen), "chunks are delivered in sweep order"
    assert seen[0][0] == 0 and seen[-1][1] == n
    for (a, b), (c, _d) in zip(seen, seen[1:]):
        assert b == c
    # the callback's pieces are owned: none may alias another's memory
    for lo, hi, res, m3 in pieces:
        _check_slim((res, m3), full, SERVICE, slots_in_use(cfgs),
                    f"chunk {lo}:{hi}", rows=slice(lo, hi))
    for (_a, _b, r1, _m1), (_c, _d, r2, _m2) in zip(pieces, pieces[1:]):
        assert not np.shares_memory(r1.son, r2.son)


@pytest.mark.parametrize("n,c", [(512, 512), (512, 300), (512, 1)])
def test_serial_rig_slim_equals_full(n, c):
    slots, cfgs, ant, ant_idx, params = _fixture(c)
    iono = shared_iono_tables([s.iono for s in slots])
    rig = SetupLuffy(n, slots, iono, ant)
    full = rig.run(params, cfgs, ant_idx)
    assert type(full[0]) is LuffyResults
    for outputs in (SERVICE, ("jmode", "xluf", "dbw"), ()):
        got = rig.run(params, cfgs, ant_idx, outputs=outputs)
        assert len(got[0].status) == c and got[1].shape == (c, 3)
        _check_slim(got, full, outputs, slots_in_use(cfgs),
                    f"serial n={n} c={c} {outputs}")
    # results() on the last submission, slim and full
    res, m3 = rig.results(outputs=("sndb",), slots=14, rows=c)
    full_res, full_m3 = rig.results()
    assert_bitexact(res.son["sndb"].ravel(),
                    full_res.son["sndb"][:c].ravel(), "results(outputs=)")
    assert_bitexact(m3.ravel(), full_m3[:c].ravel(), "results muf3")


def test_service_columns_are_unchanged_by_slim():
    """The Prop-SL extraction (/SON/ slot j+1 for freq j) reads the same."""
    n = 900
    slots, cfgs, ant, ant_idx, params = _fixture(n)
    iono = shared_iono_tables([s.iono for s in slots])
    pipe = PipelinedSetupLuffy(256, 2, slots, iono)
    full, _ = pipe.run(params, cfgs, ant, ant_idx)
    slim, _ = pipe.run(params, cfgs, ant, ant_idx, outputs=SERVICE)
    mx = 7
    sl = slice(1, 1 + mx)
    for f in SERVICE:
        a, b = slim.son[f][:, sl], full.son[f][:, sl]
        assert a.shape == b.shape
        assert np.array_equal(a.view(np.int32), b.view(np.int32)), f
