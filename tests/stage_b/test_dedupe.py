"""The dedupe path must be BIT-IDENTICAL to the per-lane path.

That is the entire bar.  ``stage_b.dedupe`` is a work-sharing transformation,
not an approximation: it claims that evaluating ANOIS1 once per distinct
(grid point, hour) and reading the answer back to every lane gives the SAME
BITS as evaluating every lane.  If that claim is not exact it is worthless,
because the rest of Stage B is measured against a Fortran oracle to the ULP
and a dedupe that drifts would show up as noise nobody could attribute.

So the central test does not use a tolerance and does not use ULP counting.
It builds a duplicated batch out of the real ANOIS1 oracle dump -- the same
189 cases ``test_anois1.py`` replays, each repeated to stand in for a run over
many transmitters -- runs it both ways, and asserts ``torch.equal`` on every
returned field including the integer ones.  Anything less would let a future
"optimisation" that touched a value slip through.

THE DUPLICATION IS THE POINT, not padding.  A batch of distinct lanes proves
nothing about sharing: U == N and the compact path IS the per-lane path.  The
fixture therefore repeats each case a fixed number of times and shuffles, so
the compact set is genuinely smaller, the inverse map is genuinely scattered,
and a bug that returned lanes in the wrong order would fail loudly.

Also asserted here, because it is the assumption everything else rests on:
integer-only keys are ENFORCED rather than merely documented, and the
representative lane is the lowest-numbered one in its group rather than
whichever a scatter happened to land on.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.redmap import redmap
from voacap_gpu.stage_b.anois1 import anois1_batch
from voacap_gpu.stage_b.dedupe import (compose, expand, representatives, take,
                                       unique_inverse)

from tests.stage_a.dumputil import COEFFS_DIR, load_dump

DUMP = load_dump("anois1")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# Stand-in for the transmitter count of the planned workload.  Small enough to
# keep the test quick, large enough that a wrong inverse map cannot come out
# looking like the identity.
COPIES = 7


# ---------------------------------------------------------------------------
# the primitives
# ---------------------------------------------------------------------------

def test_float_keys_are_refused():
    """Not a style rule -- the correctness of the whole scheme is "equal key
    implies equal inputs", and a float key is the one way to break it silently.

    Rounding a latitude to make lanes match is the natural thing to reach for
    and it merges genuinely different inputs; the merged answer is
    self-consistent, so nothing downstream can catch it.  Refuse at the door.
    """
    for bad in (torch.zeros(4), torch.zeros(4, dtype=torch.float64)):
        with pytest.raises(TypeError, match="INTEGER"):
            unique_inverse(bad)
        with pytest.raises(TypeError, match="INTEGER"):
            compose(bad)


def test_unique_inverse_round_trips_a_composite_key():
    """``unique_keys[inverse] == keys``, and the rows come back sorted."""
    g = torch.tensor([5, 1, 5, 1, 5, 3], dtype=torch.int64)
    h = torch.tensor([0, 9, 0, 2, 4, 9], dtype=torch.int64)
    keys = compose(g, h)
    uniq, inv = unique_inverse(keys)
    assert torch.equal(uniq[inv], keys)
    assert uniq.shape == (5, 2), f"expected 5 distinct keys, got {uniq.shape}"
    # (1,2) (1,9) (3,9) (5,0) (5,4) -- lexicographic, so the compact batch a
    # driver builds is a function of the key SET, not of lane order
    assert uniq.tolist() == [[1, 2], [1, 9], [3, 9], [5, 0], [5, 4]]
    # the two lanes that really are the same evaluation share a slot
    assert int(inv[0]) == int(inv[2])


def test_composite_key_does_not_alias_across_columns():
    """(1, 24) and (2, 0) are DIFFERENT lanes.

    They would collide under the usual pack-into-one-integer trick if the hour
    bound were taken as 24 and an hour index of 24 ever appeared.  ``compose``
    does not pack, so this is structural rather than lucky -- but it is worth
    a test, because switching to packing later would still pass every other
    test in this file.
    """
    g = torch.tensor([1, 2], dtype=torch.int64)
    h = torch.tensor([24, 0], dtype=torch.int64)
    uniq, inv = unique_inverse(compose(g, h))
    assert uniq.shape[0] == 2 and int(inv[0]) != int(inv[1])


def test_representatives_is_the_lowest_lane_and_is_deterministic():
    """Any lane in a group gives the same answer *if* the premise holds; pick
    the lowest so that if it ever stops holding, the failure reproduces."""
    g = torch.tensor([7, 7, 3, 7, 3], dtype=torch.int64)
    uniq, inv = unique_inverse(g)
    rep = representatives(inv, len(uniq))
    assert torch.equal(inv[rep], torch.arange(len(uniq)))
    assert rep.tolist() == [2, 0], "not the lowest lane in each group"


def test_take_passes_shared_tensors_through():
    """One call has to cope with ANOIS1's two map forms: (29,16,6) shared and
    (29,16,6,N) per-lane.  `n_lanes` is what tells them apart."""
    n = 5
    idx = torch.tensor([0, 3], dtype=torch.int64)
    per_lane = torch.arange(n * 4, dtype=torch.float32).reshape(4, n)
    shared = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    got = take({"a": per_lane, "b": shared, "c": 7}, idx, dim=-1, n_lanes=n)
    assert got["a"].shape == (4, 2)
    assert torch.equal(got["b"], shared), "a shared table was compacted"
    assert got["c"] == 7


def test_expand_performs_no_arithmetic():
    """Denormals, NaN and both zeros survive unchanged, which is the cheap
    way to say "this is an index read" without reading the source."""
    odd = torch.tensor([float("nan"), -0.0, 0.0, 5e-45, -np.inf],
                       dtype=torch.float32)
    inv = torch.tensor([4, 0, 0, 1, 2, 3, 1], dtype=torch.int64)
    out = expand(odd, inv)
    ref = odd[inv]
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32)), (
        "expand changed a bit pattern; it must be a pure gather")
    # inv picks odd[1] == -0.0 at output lanes 3 and 6; == would not notice
    neg0 = int(torch.tensor(-0.0).view(torch.int32))
    assert [int(out.view(torch.int32)[i]) for i in (3, 6)] == [neg0, neg0], (
        "-0.0 became +0.0, which == would not have caught")


# ---------------------------------------------------------------------------
# the real thing: ANOIS1 on oracle inputs, with duplicated lanes
# ---------------------------------------------------------------------------

def _dump_columns():
    """The ANOIS1 dump as numpy columns, with ONE shared REDMAP.

    A real area run has one month and one SSN, so FAKP/FAKABP/F2D are a single
    table shared by the whole grid; the per-lane (29,16,6,N) form exists only
    so ``test_anois1.py`` can replay twelve months as one batch.  Sharing is a
    property of a RUN, so the shared form is the one to test the sharing
    machinery against.  IZF is still honoured per lane so the no-data-base
    early return stays live in the batch.
    """
    _, inputs, _ = DUMP
    st = redmap(COEFFS_DIR, "CCIR", 3, F32(100.0))
    col = lambda i: np.array(  # noqa: E731
        [np.float32(r[i]) for r in inputs], dtype=np.float32)
    per_lane = dict(
        f2d111=np.array([0.0 if int(r[6]) else float(st.f2d[0, 0, 0])
                         for r in inputs], dtype=np.float32),
        gmtr=col(2), rlat=col(3), rlong=col(4), rlongd=col(5))
    return st, per_lane


def _duplicated(per_lane, copies=COPIES, seed=11):
    """Each dump case repeated `copies` times and shuffled.

    This is the area-coverage workload in miniature: one receiver site asked
    the same question once per transmitter.  The shuffle matters -- a
    contiguous tiling would let a broken inverse map still look sorted.
    """
    n0 = len(per_lane["gmtr"])
    src = np.tile(np.arange(n0), copies)
    src = src[np.random.default_rng(seed).permutation(len(src))]
    # as TENSORS: `take` is a torch index_select and leaves anything else
    # alone, which is the right default but would silently no-op on numpy
    return src, {k: torch.as_tensor(v[src]) for k, v in per_lane.items()}


def _run(st, cols, device):
    t = lambda a: torch.as_tensor(  # noqa: E731
        np.ascontiguousarray(a), device=device)
    return anois1_batch(t(st.fakp), t(st.fakabp), t(cols["f2d111"]),
                        t(cols["gmtr"]), t(cols["rlong"]), t(cols["rlongd"]),
                        t(cols["rlat"]))


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing — run "
                    "tools/oracle/dumps/run_anois1_dumps.sh")
@pytest.mark.parametrize("device", DEVICES)
def test_anois1_dedupe_is_bit_identical(device):
    """THE test.  Share the work, gather it back, demand the same bits.

    Every field, including ATNU/ATNY which carry the SIN/COS floor -- the
    floor is irrelevant here because both paths run the same code on the same
    inputs, and that is exactly the point: a dedupe cannot add error, only
    reveal that something else does.
    """
    st, per_lane = _dump_columns()
    src, dup = _duplicated(per_lane)
    n = len(src)

    # the key a real driver would use: the grid point (here the dump case,
    # which IS a distinct receiver site + time) and nothing else
    keys = compose(torch.as_tensor(src, dtype=torch.int64))
    uniq, inv = unique_inverse(keys)
    rep = representatives(inv, len(uniq))
    assert len(uniq) * COPIES == n, "the fixture did not actually duplicate"

    full = _run(st, dup, device)
    compact_cols = take(dup, rep, n_lanes=n)
    shared = _run(st, compact_cols, device)
    got = expand(shared, inv.to(device))

    for name in full:
        assert torch.equal(got[name], full[name]), (
            f"[{device}] {name}: {int((got[name] != full[name]).sum())} of "
            f"{n} lanes differ between the deduped and per-lane paths -- the "
            "dedupe is not a pure work-sharing transformation")


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
def test_measured_redundancy_on_the_area_workload():
    """The saving is REAL BUT MODEST, and the number belongs in the tree.

    ANOIS1 is keyed by (grid point, hour), so an area run's redundancy factor
    is exactly the transmitter count -- there is nothing subtle to measure,
    which is itself the finding.  What this pins is that the utility actually
    delivers that factor on a realistically-shaped key set rather than
    collapsing lanes it should not: U must equal the number of distinct
    (site, hour) pairs, no more and no less.
    """
    n_sites, n_hours, n_tx = 40, 24, 100
    site = torch.arange(n_sites).repeat_interleave(n_hours * n_tx)
    hour = torch.arange(n_hours).repeat_interleave(n_tx).repeat(n_sites)
    tx = torch.arange(n_tx).repeat(n_sites * n_hours)

    uniq, inv = unique_inverse(compose(site, hour))
    assert len(uniq) == n_sites * n_hours
    assert len(inv) / len(uniq) == n_tx, (
        f"redundancy factor {len(inv) / len(uniq)}, expected the TX count")

    # and the transmitter axis must NOT be in the key -- if a future driver
    # adds it "for safety" the sharing silently evaporates, so say so here
    _, inv_tx = unique_inverse(compose(site, hour, tx))
    assert len(torch.unique(inv_tx)) == n_sites * n_hours * n_tx


@pytest.mark.skipif(DUMP is None, reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_a_wrong_key_is_caught_by_the_bit_identity_test(device):
    """The bar has teeth: key on something that is NOT what ANOIS1 reads and
    the round trip stops matching.

    Without this, "bit-identical" could be true for the boring reason that the
    fixture's lanes were all the same anyway.  Keying on the hour ALONE merges
    different receiver sites, and the assertion below is that this genuinely
    produces different answers -- i.e. the dump's lanes really do differ, so
    the passing test above is evidence and not a tautology.
    """
    st, per_lane = _dump_columns()
    src, dup = _duplicated(per_lane, copies=2)
    n = len(src)
    bad = torch.trunc(dup["gmtr"]).to(torch.int64)
    uniq, inv = unique_inverse(compose(bad))
    assert len(uniq) < n // 2, "the deliberately-bad key did not merge lanes"
    rep = representatives(inv, len(uniq))

    full = _run(st, dup, device)
    got = expand(_run(st, take(dup, rep, n_lanes=n), device), inv.to(device))
    assert not torch.equal(got["atnu"], full["atnu"]), (
        "merging different receiver SITES changed nothing -- the fixture's "
        "lanes are degenerate and the bit-identity test proves little")
