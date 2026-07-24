"""Lane bucketing for lockstep: the permutation is exact, the payoff is not.

``plan_buckets`` groups lanes so a lockstep run pays each group's own length
instead of the whole batch's worst case.  Two separate things need saying
about it, and only one of them is good news.

THE PERMUTATION IS EXACT.  Reordering lanes and undoing it is a pure
relabelling -- no arithmetic touches the values -- so a bucketed run must be
BIT-IDENTICAL to an unbucketed one, on every field, including the integer
cursors.  That is asserted here on the real FINDF oracle batch.

THE PAYOFF DEPENDS ENTIRELY ON THE KEY, and FINDF has no good one yet.  Its
step count is determined EXACTLY by how many rays a lane finds -- measured
correlation 1.000 -- and that is not known until the lane has run.  This file
measures both ends of that: what perfect bucketing would save (the ceiling,
using the ray count as an oracle key) against what the best key available
before the run actually delivers.  Recording both is the point; a bucketing
scheme that quotes only its ceiling is quoting a number nobody can have.
"""

import numpy as np
import pytest
import torch

from voacap_gpu.batched.findf import findf_batch
from voacap_gpu.batched.lockstep import plan_buckets, restore_order

from tests.batched.test_findf import DUMP, IFOB_ROWS, _batch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _permute(args, order):
    """Reorder every per-lane input; the antenna-free FINDF args are all
    lane-trailing, so this is one rule."""
    return [a[..., order] for a in args]


@pytest.mark.skipif(DUMP is None or IFOB_ROWS is None,
                    reason="oracle dump missing")
@pytest.mark.parametrize("device", DEVICES)
def test_bucketing_is_bit_identical_to_running_the_batch_whole(device):
    """The whole justification for reordering: it cannot change an answer.

    Run the oracle batch straight, then run it permuted and undo the
    permutation, and require every field to match bit for bit -- floats by
    identity, not by tolerance, because nothing here does arithmetic.
    """
    args, expected, _ = _batch(device)
    rays_a, scal_a = findf_batch(*args)

    key = torch.arange(args[0].shape[-1] - 1, -1, -1, device=device)
    order, edges = plan_buckets(key, 4)
    assert edges[0] == 0 and edges[-1] == len(key)
    rays_b, scal_b = findf_batch(*_permute(args, order))

    for name, v in list(rays_b.items()) + list(scal_b.items()):
        want = rays_a[name] if name in rays_a else scal_a[name]
        got = restore_order(v, order)
        assert torch.equal(got, want), (
            f"[{device}] {name} changed under a pure permutation of lanes")


@pytest.mark.skipif(DUMP is None or IFOB_ROWS is None,
                    reason="oracle dump missing")
def test_restore_order_is_a_true_inverse():
    key = torch.tensor([5, 1, 4, 1, 9, 0, 3])
    order, _ = plan_buckets(key, 3)
    payload = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    assert torch.equal(restore_order(payload[..., order], order), payload)


@pytest.mark.skipif(DUMP is None or IFOB_ROWS is None,
                    reason="oracle dump missing")
def test_the_ceiling_and_the_reachable_saving_are_both_recorded():
    """What perfect bucketing would buy, against what a usable key buys.

    The oracle key is the ray count, which correlates 1.000 with the step
    count and is unknowable in advance.  The realistic key is whether the lane
    has an F1 ledge, which is the strongest thing known BEFORE the run and
    correlates about 0.4.  The gap between the two columns is the reason this
    optimisation is off by default.
    """
    args, expected, fi2 = _batch("cpu")
    rays, scal = findf_batch(*args)
    iaftxr = scal["iaftxr"].to(torch.float64).numpy()
    # steps are a fixed affine function of rays found, so ray count stands in
    steps = iaftxr

    def effective(key, k):
        if k == 1:
            return steps.max()
        q = np.quantile(key, np.linspace(0, 1, k + 1))
        tot = 0.0
        for i in range(k):
            hi = key <= q[i + 1] if i == k - 1 else key < q[i + 1]
            m = (key >= q[i]) & hi
            if m.sum():
                tot += m.sum() * steps[m].max()
        return tot / len(steps)

    flat = steps.max()
    oracle4 = effective(steps, 4)
    real4 = effective((fi2 > 0).to(torch.float64).numpy(), 4)
    assert oracle4 < flat, "perfect bucketing must beat the flat batch"
    # The realistic key is allowed to be barely better than nothing -- that is
    # the measured state of affairs and the test exists to keep it visible
    # rather than to demand it improve.
    assert real4 <= flat + 1e-9
    assert oracle4 < real4, (
        f"the oracle key ({oracle4:.1f} effective steps) should beat the "
        f"a-priori one ({real4:.1f}); if it no longer does, a better key was "
        "found and plan_buckets should probably be turned on")
