"""WHAT ACTUALLY HAS TO CROSS PCIe — and the proof that nothing else does.

Stage C's round trip used to move all 55,652 bytes of per-thread state up and
all 55,652 back.  At 32,256 circuit-hours that is 1.8 GB each way against a
kernel of a few hundred milliseconds, so the bus, not the arithmetic, set the
throughput.  ``cuda/driver.py`` now declares an INPUT SET -- the fields
LUFFY reads before it writes them -- and ``upload="inputs"`` sends only those,
3,228 bytes a thread, zeroing the rest on the device with a memset.
``fetch_results`` brings back 1,700 instead of 55,652.

THIS FILE IS WHAT MAKES THAT SAFE.  The set was established by experiment --
zero one field in all 504 replay circuit-hours, launch, diff the whole output
against the clean run, and call the field an input only if something else
moved -- and an experiment is worth nothing if it is not repeated.  Without
these tests a future edit that makes one scratch field load-bearing would
change the answers silently, because the driver would still be sending zeros
where that field's value used to be.

Four independent statements:

1. ``test_the_reduced_upload_is_output_neutral`` -- the production path
   (inputs up, results back) returns EXACTLY what the full path returns, on
   all 504 circuit-hours across SHORT, LONG and SMOOTH.  This is the claim
   the driver actually makes.
2. ``test_poisoning_the_declared_scratch_changes_nothing`` -- with the FULL
   upload, filling every float field NOT in the upload set with a NaN
   bit-pattern still leaves the results identical.  Zeros could be a lucky
   value -- they were, for /SON/, whose skipped frequency columns happen to
   hold zeros in every replay deck, which is why /SON/ is carried -- but a
   NaN that is read propagates and cannot hide.  /RAYS/ IFOB, 30,504 bytes
   and 55% of the whole per-thread state, is in that poisoned set, and it is
   the single biggest thing this file protects.
3. ``test_the_arena_does_not_have_to_be_re_zeroed`` -- the same claim under
   the perturbation production actually applies: the previous batch's
   scratch, left in place.
4. ``test_result_fetch_agrees_with_the_full_fetch`` -- the compacted gather
   is picking the right words out of CircuitState in the first place.

The bar everywhere is bit-identity, as everywhere else in Stage C.
"""

import numpy as np
import pytest

from voacap_gpu.cuda.driver import (CARRY_FIELDS, CIRCUIT_DTYPE,
                                       INPUT_FIELDS, LuffyArena,
                                       RAYS_UPLOAD_FIELDS,
                                       RAYS_UPLOAD_LEAVES,
                                       RESULT_BYTES_PER_THREAD, RESULT_FIELDS,
                                       STATE_BYTES_PER_THREAD,
                                       UPLOAD_BYTES_PER_THREAD, UPLOAD_FIELDS,
                                       UPLOAD_LEAVES, expand_fields,
                                       fetch_luffy, fetch_results,
                                       launch_luffy, leaves_of, luffy_results,
                                       shared_antenna_tables,
                                       shared_iono_tables, stage_luffy)
from voacap_gpu.cuda.state import RAYS_DTYPE, SON_DTYPE

from tests.scalar.test_luffy_replay import DECKS
from tests.cuda.harness import assert_bitexact, requires_cuda
from tests.cuda.test_luffy_replay import _replay

pytestmark = requires_cuda

# NaN as float32, and not a quiet default one: if this reaches an output the
# bit pattern says where it came from.
POISON = np.array(np.uint32(0xFFBADBAD), np.uint32).view(np.float32)


def _get(buf, path):
    for p in path.split("."):
        buf = buf[p]
    return buf


def _set(buf, path, value):
    parts = path.split(".")
    v = buf
    for p in parts[:-1]:
        v = v[p]
    v[parts[-1]] = value


def _kind(dtype, path):
    """'f' or 'i' for a leaf — through the SUBARRAY, whose own kind is 'V'."""
    for p in path.split(".")[:-1]:
        dtype = dtype.fields[p][0]
    return dtype.fields[path.split(".")[-1]][0].base.kind


def _scratch_fields():
    """Every leaf the driver declares it does NOT have to send."""
    circ = [f for f in leaves_of(CIRCUIT_DTYPE) if f not in set(UPLOAD_LEAVES)]
    rays = [f for f in leaves_of(RAYS_DTYPE)
            if f not in set(RAYS_UPLOAD_LEAVES)]
    return circ, rays


def _launch(v, entries, entries_rays, **kw):
    return launch_luffy(entries, entries_rays, v["ipflags"],
                        shared_iono_tables(v["ionos"]), v["iono_idx"],
                        shared_antenna_tables(v["cants"]), v["ant_idx"], **kw)


def _results(v, entries, entries_rays):
    return luffy_results(entries, entries_rays, v["ipflags"],
                         shared_iono_tables(v["ionos"]), v["iono_idx"],
                         shared_antenna_tables(v["cants"]), v["ant_idx"])


def _assert_results_match(got, exp, what):
    """Both arguments are (son, xluf, jmode, status) quadruples."""
    g_son, g_xluf, g_jmode, g_st = got
    e_son, e_xluf, e_jmode, e_st = exp
    assert_bitexact(g_st, e_st, f"{what} status")
    assert_bitexact(np.asarray(g_xluf), np.asarray(e_xluf), f"{what} xluf")
    assert_bitexact(np.asarray(g_jmode), np.asarray(e_jmode),
                    f"{what} frq.jmode")
    for name in SON_DTYPE.names:
        assert_bitexact(np.asarray(g_son[name]), np.asarray(e_son[name]),
                        f"{what} son.{name}")


def _as_quad(bufs, status):
    """The result quadruple, pulled out of a FULL fetch."""
    return bufs["son"], bufs["muf"]["xluf"], bufs["frq"]["jmode"], status


# ---------------------------------------------------------------------------
# the declaration itself
# ---------------------------------------------------------------------------

def test_the_declared_sets_are_real_fields_and_add_up():
    """Cheap, but it is what stops a typo silently dropping a field."""
    # expand_fields asserts every path is real and that none is covered twice
    assert set(UPLOAD_LEAVES) == set(expand_fields(CIRCUIT_DTYPE,
                                                   UPLOAD_FIELDS))
    assert set(INPUT_FIELDS).isdisjoint(CARRY_FIELDS)
    expand_fields(RAYS_DTYPE, RAYS_UPLOAD_FIELDS)
    result_leaves = expand_fields(CIRCUIT_DTYPE, RESULT_FIELDS)
    # /SON/ and XLUF are the result table AND a carry: LUFFY leaves the
    # frequency slots it skips exactly as it found them, so they have to go
    # up as well as come back.  JMODE is written unconditionally by SELMOD
    # and is the one result word that does not.
    for f in expand_fields(CIRCUIT_DTYPE, ("son", "muf.xluf")):
        assert f in UPLOAD_LEAVES, f"{f} is a result and must be carried up"
    assert "frq.jmode" in result_leaves

    # THE HEADLINE: /RAYS/ IFOB is 30,504 bytes, 55% of the per-thread state,
    # and it does not travel.  If a future change makes it an input this
    # assert is the one that should be argued with.
    assert "ifob" not in RAYS_UPLOAD_FIELDS

    print(f"\nper circuit-hour: state {STATE_BYTES_PER_THREAD:,} B, "
          f"upload {UPLOAD_BYTES_PER_THREAD:,} B "
          f"({STATE_BYTES_PER_THREAD / UPLOAD_BYTES_PER_THREAD:.1f}x less), "
          f"results {RESULT_BYTES_PER_THREAD:,} B "
          f"({STATE_BYTES_PER_THREAD / RESULT_BYTES_PER_THREAD:.1f}x less)")
    # a tripwire, not a target: if a future edit quietly puts a big block
    # back on the bus, this is where it shows up
    assert UPLOAD_BYTES_PER_THREAD < STATE_BYTES_PER_THREAD // 10
    assert RESULT_BYTES_PER_THREAD < STATE_BYTES_PER_THREAD // 20


# ---------------------------------------------------------------------------
# the three empirical statements
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_result_fetch_agrees_with_the_full_fetch():
    """``fetch_results`` gathers the right words out of CircuitState."""
    v = _replay()
    run, state = stage_luffy(v["entries"], v["entries_rays"], v["ipflags"],
                             shared_iono_tables(v["ionos"]), v["iono_idx"],
                             shared_antenna_tables(v["cants"]), v["ant_idx"])
    run()
    got = fetch_results(state)
    full, _rays, status = fetch_luffy(state)
    _assert_results_match(tuple(got), _as_quad(full, status), "fetch_results")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_the_reduced_upload_is_output_neutral():
    """The production path returns exactly what the full path returns.

    504 circuit-hours, all three regimes, every field of /SON/ plus XLUF,
    JMODE and the status -- with 3,228 bytes a thread having crossed the bus
    instead of 55,652.
    """
    v = _replay()
    full, _rays, status = _launch(v, v["entries"], v["entries_rays"])
    assert (status == 0).all()
    got = _results(v, v["entries"], v["entries_rays"])
    _assert_results_match(tuple(got), _as_quad(full, status),
                          "reduced upload")
    print(f"\nreduced upload bit-exact on {len(status)} circuit-hours "
          f"({UPLOAD_BYTES_PER_THREAD:,} B up, "
          f"{RESULT_BYTES_PER_THREAD:,} B down)")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_poisoning_the_declared_scratch_changes_nothing():
    """Every float the driver declines to send, filled with NaN, full upload.

    This is the statement the byte reduction rests on: those fields are not
    READ.  A field that were read would propagate the NaN into /SON/ within
    one arithmetic operation, and NaN never compares bit-equal to a number.
    Integer scratch cannot be NaN'd -- an out-of-range int index would fault
    or spin rather than fail an assert -- so it is REVERSED along its last
    axis instead: a permutation of values that field already held, in range
    by construction and wrong everywhere.
    """
    v = _replay()
    full, _rays, status = _launch(v, v["entries"], v["entries_rays"])
    assert (status == 0).all()

    circ_scratch, rays_scratch = _scratch_fields()
    ent = v["entries"].copy()
    ent_rays = v["entries_rays"].copy()
    nf = ni = 0
    for buf, dt, fields in ((ent, CIRCUIT_DTYPE, circ_scratch),
                            (ent_rays, RAYS_DTYPE, rays_scratch)):
        for f in fields:
            if _kind(dt, f) == "f":
                _set(buf, f, POISON)
                nf += 1
            else:
                # INT scratch is REVERSED along its last axis instead: that is
                # a permutation of values the field already held, so it cannot
                # fault or spin the way an arbitrary integer index could, and
                # it is still wrong everywhere.  /RAYS/ IFOB -- the 30,504
                # bytes this whole exercise is about -- is the one that
                # matters here.
                cur = np.asarray(_get(buf, f))
                _set(buf, f, cur[..., ::-1] if cur.ndim > 1 else 0)
                ni += 1
    assert nf and ni, "nothing to poison — the declaration is suspect"

    got, _r2, st2 = _launch(v, ent, ent_rays)
    _assert_results_match(_as_quad(got, st2), _as_quad(full, status),
                          "poisoned scratch")
    print(f"\n{len(circ_scratch)} CircuitState + {len(rays_scratch)} /RAYS/ "
          f"scratch fields poisoned ({nf} with NaN, {ni} scrambled): "
          "results identical")


@pytest.mark.skipif(not DECKS, reason="oracle dump missing")
def test_the_arena_does_not_have_to_be_re_zeroed():
    """A reused arena leaves the previous batch's scratch in place.

    ``LuffyArena`` allocates and zeroes once and then never clears again, so
    submission k+1 starts on top of submission k's leftovers.  That is the
    same claim as the poison test, made with the one perturbation a
    production caller will actually hit -- and it is what turns a 1.8 GB
    memset per batch into a 1.8 GB memset per process.
    """
    v = _replay()
    ref = _results(v, v["entries"], v["entries_rays"])
    arena = LuffyArena(len(v["ipflags"]), shared_iono_tables(v["ionos"]),
                       shared_antenna_tables(v["cants"]))
    for k in range(3):
        got = arena.run(v["entries"], v["entries_rays"], v["ipflags"],
                        v["iono_idx"], v["ant_idx"])
        _assert_results_match(tuple(got), tuple(ref), f"arena pass {k}")
