"""VC_WARP_GROUPS=2 through the REAL driver, not a test-private kernel.

``test_luffy_warp.py`` proves the packed kernel exact against its own copy of
the launch code.  That is the right place to develop it, but it leaves one
thing unproved: that ``driver.stage_luffy(groups=2)`` computes the same grid,
the same shared-memory size and the same per-warp slot mapping.  Those three
are the whole of the host-side change, and getting any of them wrong produces
a kernel that still runs -- it just silently drops or doubles circuit-hours.

So this drives the shipped path end to end and requires bit-identity with the
thread kernel on all 504 replay circuit-hours.  It also checks a batch that
is NOT a multiple of the per-block capacity, because that is exactly where a
wrong grid formula loses the tail.
"""

import numpy as np
import pytest

from voacap_gpu.cuda import driver

from tests.cuda.harness import requires_cuda
from tests.cuda.test_luffy_replay import _replay

pytestmark = requires_cuda


def _run(n, warp, groups, wpb=4):
    w = _replay()
    n0 = len(w["entries"])
    idx = np.arange(n) % n0
    return driver.launch_luffy(
        w["entries"][idx], w["entries_rays"][idx], w["ipflags"][idx],
        driver.shared_iono_tables(w["ionos"]), w["iono_idx"][idx],
        driver.shared_antenna_tables(w["cants"]), w["ant_idx"][idx],
        warp=warp, warps_per_block=wpb, groups=groups)


@pytest.mark.parametrize("n", [504, 501, 130])
def test_packed_matches_thread_kernel_through_the_driver(n):
    """groups=2 vs the thread kernel, including a ragged tail.

    501 and 130 are deliberately not multiples of warps_per_block*groups: a
    grid computed as ``n/wpb`` instead of ``n/(wpb*groups)`` still launches,
    and still produces correct answers for the circuit-hours it covers, so
    only a size that exercises the tail catches it.
    """
    a, ra, ref_st = _run(n, warp=False, groups=1)
    b, rb, got_st = _run(n, warp=True, groups=2)

    assert np.array_equal(ref_st, got_st), "status differs"
    assert len(a) == n and len(b) == n, "a circuit-hour went missing"

    def raw(arr, f):
        # Bitwise, so the same NaN compares equal and -0.0 does not compare
        # equal to 0.0.  Struct fields are strided, hence the contiguous copy.
        return np.ascontiguousarray(arr[f]).view(np.uint8)

    bad = [f for f in a.dtype.names if not np.array_equal(raw(a, f), raw(b, f))]
    assert not bad, f"packed differs from the thread kernel on {bad[:8]}"
    for f in ra.dtype.names:
        assert np.array_equal(raw(ra, f), raw(rb, f)), \
            f"packed differs on /RAYS/ {f}"


def test_groups_is_validated():
    """12 frequencies need >=12 lanes, so 4 groups of 8 cannot work.

    A silent acceptance here would be the worst outcome: lanes 12..15 of each
    group would never run their frequency and the answer would be short a
    column rather than wrong-looking.
    """
    with pytest.raises(ValueError, match="1 or 2"):
        driver.load_kernel(groups=4)
    with pytest.raises(ValueError):
        driver.load_kernel(groups=0)
