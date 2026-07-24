"""``luffy_batch_warp`` against ``luffy_batch``: same answer, twelve lanes wide.

``luffy_batch`` runs a whole circuit-hour on ONE THREAD and is already proved
bit-identical to Stage A on all 504 replay circuit-hours
(``test_luffy_replay.py``).  ``luffy_batch_warp`` runs the same circuit-hour on
ONE WARP with lane L on frequency L+1, so matching ``luffy_batch`` cell for
cell proves it just as thoroughly, and against a reference this test can run in
the same launch.

WHY THIS IS A REAL TEST AND NOT A FORMALITY.  The twelve frequencies are NOT
independent, and three things cross between them: the /DON/ decile triple that
RELBIL writes conditionally and reads unconditionally; FINDF's IAFTXR/skip
group, which several of its exits never assign, so GMLOSS and SELTXR read the
previous frequency's ray count; and the extent-limited writes to /allMODE/,
/ZON/, /REFLX/, /MODES/, /LOSX/ and TXRGML, which make the state the loop
leaves a MERGE of twelve frequencies' writes rather than the last one's.  The
warp kernel resolves the first two with warp scans and the third with a
poison-seeded, lane-ordered per-cell merge (``cuda/luffy_warp.cuh``).  Break
any of the three and this test fails in the scratch commons while /SON/ still
looks perfect -- which is exactly how each of them was found.

Both halves of the state are compared: CIRCUIT_DTYPE and the separate SoA
/RAYS/ buffer are together the whole Stage A ``Circuit``.
"""

import numpy as np
import pytest

from voacap_gpu.stage_c.driver import (CIRCUIT_DTYPE, SMOOTHSTORE_BYTES,
                                       launch_luffy, shared_antenna_tables,
                                       shared_iono_tables, smoothstore_sizeof)
from voacap_gpu.stage_c.state import RAYS_DTYPE

from tests.stage_a.test_luffy_replay import DECKS
from tests.stage_c.harness import assert_bitexact, requires_cuda
from tests.stage_c.test_luffy_replay import NHOUR, _fields, _get, _replay

pytestmark = requires_cuda


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_smoothstore_shared_memory_size_matches_the_device():
    """The warp kernel's dynamic shared memory is sized from the host."""
    assert smoothstore_sizeof() == SMOOTHSTORE_BYTES, (
        f"sizeof(SmoothStore) {smoothstore_sizeof()} != host "
        f"SMOOTHSTORE_BYTES {SMOOTHSTORE_BYTES}")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
@pytest.mark.parametrize("wpb", [1, 4])
def test_warp_kernel_is_bit_identical_to_the_thread_kernel(wpb):
    v = _replay()
    n = v["entries"].shape[0]
    assert n == len(DECKS) * NHOUR, f"{n} circuit-hours, expected 24 per deck"
    iono = shared_iono_tables(v["ionos"])
    ant = shared_antenna_tables(v["cants"])

    ref, ref_rays, ref_status = launch_luffy(
        v["entries"], v["entries_rays"], v["ipflags"], iono, v["iono_idx"],
        ant, v["ant_idx"])
    got, got_rays, status = launch_luffy(
        v["entries"], v["entries_rays"], v["ipflags"], iono, v["iono_idx"],
        ant, v["ant_idx"], warp=True, warps_per_block=wpb)

    bad = np.flatnonzero(ref_status != 0)
    assert bad.size == 0, "the reference kernel refused; fix that first"
    assert_bitexact(status, ref_status, "status")

    checks = [(CIRCUIT_DTYPE, got, ref, ""),
              (RAYS_DTYPE, got_rays, ref_rays, "rays.")]
    per_struct = {}
    failures = []
    for dt, g_all, e_all, prefix in checks:
        for label, path in _fields(dt):
            label = prefix + label
            g = np.asarray(_get(g_all, path))
            e = np.asarray(_get(e_all, path))
            if g.dtype.kind == "f":
                same = g.view(np.uint32) == e.view(np.uint32)
            else:
                same = g == e
            block = label.split(".")[0]
            hit, tot = per_struct.get(block, (0, 0))
            per_struct[block] = (hit + int(same.sum()), tot + int(same.size))
            if not same.all():
                failures.append((label, int((~same).sum()), int(same.size)))
    hits = sum(h for h, _ in per_struct.values())
    tots = sum(t for _, t in per_struct.values())
    print(f"\nwarp vs thread kernel ({wpb} warps/block): {n} circuit-hours, "
          f"{hits}/{tots} cells bit-exact ({100.0 * hits / tots:.4f}%)")
    for block, (h, t) in per_struct.items():
        print(f"  {block:<10s} {h:>10d}/{t:<10d} {100.0 * h / t:8.4f}%")
    if failures:
        head = "; ".join(f"{lab} {bad_}/{tot}" for lab, bad_, tot
                         in failures[:12])
        raise AssertionError(
            f"{len(failures)} field(s) differ from luffy_batch: {head}")

    for dt, g_all, e_all, prefix in checks:
        for label, path in _fields(dt):
            assert_bitexact(np.asarray(_get(g_all, path)),
                            np.asarray(_get(e_all, path)), prefix + label)


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_the_warp_kernel_spans_all_three_regimes():
    """The three arms are three different programs inside the warp kernel."""
    v = _replay()
    seen = {t: v["tags"].count(t) for t in ("SHORT", "LONG", "SMOOTH")}
    for arm, cnt in seen.items():
        assert cnt, f"no circuit-hour exercises {arm} ({seen})"
