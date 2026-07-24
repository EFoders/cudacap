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

AND THE SAME AGAIN WITH TWO CIRCUIT-HOURS IN ONE WARP.  ``cuda/luffy_warp.cuh``
built with ``-DVC_WARP_GROUPS=2`` splits the warp into two sixteen-lane groups
running two different circuit-hours, which makes every one of those scans and
ballots a GROUP-local operation -- a whole-warp ``__ballot_sync`` would mix one
circuit's LUF cut into the other's, and a whole-warp ``__syncwarp()`` is
undefined the moment the two groups take different branches, which they do
constantly.  That configuration needs a launch this repository's driver does
not yet make, so the packed kernel is written out here and tested here; the
driver change it needs is four lines, and is quoted in the header of
``cuda/luffy_warp.cuh``.
"""

import numpy as np
import pytest

from voacap_gpu.cuda.compile import load
from voacap_gpu.cuda.driver import (CIRCUIT_DTYPE, SMOOTHSTORE_BYTES,
                                       _rays_from_soa, _rays_views,
                                       launch_luffy, shared_antenna_tables,
                                       shared_iono_tables, smoothstore_sizeof,
                                       stage_luffy)
from voacap_gpu.cuda.state import RAYS_DTYPE

from tests.scalar.test_luffy_replay import DECKS
from tests.cuda.harness import assert_bitexact, requires_cuda
from tests.cuda.test_luffy_replay import NHOUR, _fields, _get, _replay

pytestmark = requires_cuda

# TWO CIRCUIT-HOURS PER WARP.  ``cuda/luffy_warp.cuh`` compiled with
# -DVC_WARP_GROUPS=2 cuts the warp into two sixteen-lane groups, each running
# its own circuit-hour on its own twelve frequencies, and every ballot, scan,
# shuffle and syncwarp in the file becomes group-local.  The LAUNCH has to
# change with it -- the circuit index, the SmoothStore slice, the status write
# and the grid all move -- and that lives in ``stage_c/driver.py``, so the
# packed kernel is spelled out HERE rather than shipped: this test is what
# proves the group-local machinery exact, and the driver change is a
# four-line diff against ``luffy_batch_warp`` (see the header of
# cuda/luffy_warp.cuh).
_PACKED_SRC = r'''
#include "luffy.cuh"
#include "luffy_warp.cuh"

extern "C" __global__ void luffy_batch_warp_packed(
        CircuitState* circ,
        int* rays_nang, float* rays_ang, int* rays_ifob,
        const int* ipflag,
        const int* iono_idx, const int* ant_idx,
        const float* fam, const float* dud, const float* fakp,
        const float* fakmap, const float* hmym, const float* fakabp,
        const float* abmap, const float* sys, const float* perr,
        const float* f2d111,
        const float* ant_array, const float* ant_aeff, const int* ant_iats,
        const float* ant_xfqs, const float* ant_xfqe,
        const float* ant_offazim, const float* ant_pwrdba, int numants,
        int* status, int n) {
    extern __shared__ SmoothStore warp_store[];
    int lane = threadIdx.x & 31;
    int wib = threadIdx.x >> 5;                 // warp index in the block
    int grp = lane / VC_WARP_GLANES;            // lane group in the warp
    int slot = wib * VC_WARP_GROUPS + grp;      // circuit-hour in the block
    int i = blockIdx.x * ((blockDim.x >> 5) * VC_WARP_GROUPS) + slot;
    if (i >= n) return;
    LaneState L;
    int st = luffy_warp(&circ[i], rays_nang, rays_ang, rays_ifob, n, i,
                        &L, &warp_store[slot], lane,
                        fam, dud, fakp, fakmap, hmym, fakabp, abmap,
                        sys, perr, f2d111, ant_array, ant_aeff, ant_iats,
                        ant_xfqs, ant_xfqe, ant_offazim, ant_pwrdba, numants,
                        iono_idx[i], ant_idx[i], ipflag[i]);
    if ((lane & (VC_WARP_GLANES - 1)) == 0)
        status[i] = st;
}
'''


def _launch_packed(v, iono, ant, groups, wpb):
    """Run the packed kernel over the whole replay; return it like
    ``launch_luffy`` does."""
    import cupy as cp

    n = v["entries"].shape[0]
    fn = load(_PACKED_SRC, (f"-DVC_WARP_GROUPS={groups}",)) \
        .get_function("luffy_batch_warp_packed")
    # stage_luffy moves the batch and lays /RAYS/ out structure-of-arrays; the
    # `run` it hands back is the SHIPPING kernel's launch, which this test
    # does not use -- only the resident state.
    _, state = stage_luffy(v["entries"], v["entries_rays"], v["ipflags"],
                           iono, v["iono_idx"], ant, v["ant_idx"], warp=True,
                           warps_per_block=wpb)
    d_circ, d_soa, d_status, _ = state
    d_nang, d_ang, d_ifob = _rays_views(d_soa)
    d = {k: cp.asarray(x) for k, x in iono.items()}
    a = {k: cp.asarray(x) for k, x in ant.items() if k != "numants"}
    i32 = np.int32
    args = (d_circ, d_nang, d_ang, d_ifob,
            cp.asarray(np.ascontiguousarray(v["ipflags"], i32)),
            cp.asarray(np.ascontiguousarray(v["iono_idx"], i32)),
            cp.asarray(np.ascontiguousarray(v["ant_idx"], i32)),
            d["fam"], d["dud"], d["fakp"], d["fakmap"], d["hmym"],
            d["fakabp"], d["abmap"], d["sys"], d["perr"], d["f2d111"],
            a["array"], a["aeff"], a["iats"], a["xfqs"], a["xfqe"],
            a["offazim"], a["pwrdba"], i32(ant["numants"]), d_status, i32(n))
    per_block = wpb * groups                    # circuit-hours per block
    fn(((n + per_block - 1) // per_block,), (32 * wpb,), args,
       shared_mem=per_block * SMOOTHSTORE_BYTES)
    cp.cuda.runtime.deviceSynchronize()
    return (cp.asnumpy(d_circ).view(CIRCUIT_DTYPE), _rays_from_soa(d_soa, n),
            cp.asnumpy(d_status))


def _assert_same_everywhere(got, got_rays, status, ref, ref_rays, ref_status,
                            what):
    """Every cell of CircuitState and of the SoA /RAYS/ buffer, bit for bit."""
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
    print(f"\n{what}: {hits}/{tots} cells bit-exact "
          f"({100.0 * hits / tots:.4f}%)")
    for block, (h, t) in per_struct.items():
        print(f"  {block:<10s} {h:>10d}/{t:<10d} {100.0 * h / t:8.4f}%")
    if failures:
        head = "; ".join(f"{lab} {bad_}/{tot}" for lab, bad_, tot
                         in failures[:12])
        raise AssertionError(f"{len(failures)} field(s) differ: {head}")
    for dt, g_all, e_all, prefix in checks:
        for label, path in _fields(dt):
            assert_bitexact(np.asarray(_get(g_all, path)),
                            np.asarray(_get(e_all, path)), prefix + label)


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
@pytest.mark.parametrize("wpb", [1, 4])
def test_two_circuit_hours_per_warp_is_bit_identical_too(wpb):
    """VC_WARP_GROUPS=2: sixteen lanes each, and every collective group-local.

    This is the test that earns the packing.  With two INDEPENDENT
    circuit-hours in one warp, a ``__ballot_sync`` over the whole warp mixes
    one circuit's cut with the other's, a ``__shfl_sync`` with an absolute
    source index reads the wrong circuit's lane, and a ``__syncwarp()`` over
    all thirty-two lanes is undefined the moment the two groups take different
    branches -- which they do constantly, since one may be a short path and
    the other a long one.  Every one of those is masked to its own group; get
    any of them wrong and a handful of the 504 disagree, not all of them.

    The two groups also finish at different times and one of them can be past
    the end of the batch (``i >= n``), so the group masks have to survive
    lanes that have already returned.
    """
    v = _replay()
    iono = shared_iono_tables(v["ionos"])
    ant = shared_antenna_tables(v["cants"])
    ref, ref_rays, ref_status = launch_luffy(
        v["entries"], v["entries_rays"], v["ipflags"], iono, v["iono_idx"],
        ant, v["ant_idx"])
    assert np.flatnonzero(ref_status != 0).size == 0, "reference refused"
    got, got_rays, status = _launch_packed(v, iono, ant, 2, wpb)
    _assert_same_everywhere(got, got_rays, status, ref, ref_rays, ref_status,
                            f"2 circuit-hours/warp ({wpb} warps/block) vs "
                            "the thread kernel")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_the_warp_kernel_spans_all_three_regimes():
    """The three arms are three different programs inside the warp kernel."""
    v = _replay()
    seen = {t: v["tags"].count(t) for t in ("SHORT", "LONG", "SMOOTH")}
    for arm, cnt in seen.items():
        assert cnt, f"no circuit-hour exercises {arm} ({seen})"
