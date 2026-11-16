"""The rigs' construction and uploads are ordered before their work streams.

Every rig allocates and zero-fills its device buffers with ``cp.zeros`` /
``cp.full`` on CuPy's (legacy) default stream, and uploads its shared tables
there with ``cp.asarray``.  The pipelined rigs then run all their work on
``Stream(non_blocking=True)`` streams -- which, by definition, do NOT wait
for the default stream.  So unless the rig orders the two explicitly, a first
chunk can read a buffer BEFORE its zero-fill lands.  On fresh memory that is
invisible (the pool hands out what cudaMalloc gave it); on memory CuPy's pool
recycles from an earlier object it is a wrong answer, usually with status 0.

The reproducer is therefore the pool itself: fill it with a garbage pattern,
free that (the blocks go back to the pool still holding the pattern), build a
rig and run it straight away, and compare bit-for-bit with a serial reference.
The same is done for the everyday shape of the bug -- a new rig built after the
previous one was freed, reusing its still-dirty memory.  Neither needs the
``reference/`` tree: the PipelinedLuffy case builds its circuits with
``engine.build_circuit`` + ``pack_circuit``.

The second half pins the INPUT checks on the device-setup path: they raise
real exceptions (so survive ``python -O``), they name the offending rows, and
the device-side hour guard in setup.cuh refuses an out-of-range hour with a
non-zero status instead of reading past the VIRTIM table.
"""

import gc
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from voacap_gpu import engine
from voacap_gpu.cuda import driver
from voacap_gpu.cuda.driver import (SETUP_CONFIG_DTYPE, SETUP_PARAMS_DTYPE,
                                       IonoSlot, LuffyArena, PipelinedLuffy,
                                       PipelinedSetupLuffy, SetupLuffy,
                                       pack_circuit, shared_antenna_tables,
                                       shared_iono_tables)
from voacap_gpu.scalar.redmap import redmap

from tests.cuda.harness import assert_bitexact, cp, requires_cuda

pytestmark = requires_cuda

ITERATIONS = 20
FREQS_A = (5.0, 7.0, 10.0, 14.0, 18.0, 21.0, 25.0)
FREQS_B = (3.5, 7.1, 14.2)
_POISON_WORD = 0x3F800000           # 1.0f: legal-looking, never what we want


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _setup_fixture(n, seed=20261115):
    """n setup circuits over two ionospheres, two configs, 3 antennas."""
    slots = [
        IonoSlot(redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(120.0)),
                 7, 120.0),
        IonoSlot(redmap(engine._COEFFS_DIR, "URSI", 1, np.float32(40.0)),
                 1, 40.0),
    ]
    cfgs = np.zeros(2, SETUP_CONFIG_DTYPE)
    cfgs[0] = engine.setup_config(engine.System(fprob=(1.0, 1.0, 1.0, 0.0)),
                                  method=25, freqs=FREQS_A)
    cfgs[1] = engine.setup_config(engine.System(fprob=(1.0, 1.0, 1.0, 0.70)),
                                  method=30, freqs=FREQS_B)
    ant = shared_antenna_tables([engine.isotropic_cant(power_kw=1.0, tx_gos=g)
                                 for g in (0.0, 3.0, -2.5)])
    rng = np.random.default_rng(seed)
    p = np.zeros(n, SETUP_PARAMS_DTYPE)
    p["tlatd"] = 40.0
    p["tlongd"] = -105.0
    p["rlatd"] = rng.uniform(-80.0, 80.0, n).astype(np.float32)
    p["rlongd"] = rng.uniform(-179.0, 179.0, n).astype(np.float32)
    p["hour"] = rng.integers(1, 25, n)
    p["npsl"] = 0
    p["iono"] = rng.integers(0, 2, n)
    p["cfg"] = p["iono"]
    ant_idx = rng.integers(0, 3, n).astype(np.int32)
    iono = shared_iono_tables([s.iono for s in slots])
    return slots, iono, cfgs, ant, ant_idx, p


_LUFFY_PATHS = [((51.50, -0.13), (48.85, 2.35)),
                ((40.00, -105.0), (40.00, -70.0)),
                ((40.00, -105.0), (50.00, 0.0)),
                ((-23.00, 30.00), (-35.00, 138.00)),
                ((34.05, -118.24), (35.68, 139.77))]
_LUFFY_HOURS = (1, 6, 12, 18)


def _luffy_fixture(n):
    """Host-built circuits (no reference/ tree), tiled up to n rows."""
    iono = redmap(engine._COEFFS_DIR, "CCIR", 7, np.float32(120.0))
    cant = engine.isotropic_cant(power_kw=1.0)
    bufs, rays, ipf = [], [], []
    for tx, rx in _LUFFY_PATHS:
        for h in _LUFFY_HOURS:
            circ, ip = engine.build_circuit(tx, rx, 7, 120.0, h, FREQS_A,
                                            method=30, cant=cant, iono=iono)
            b, r = pack_circuit(circ)
            bufs.append(b)
            rays.append(r)
            ipf.append(ip)
    k = len(bufs)
    take = np.arange(n) % k
    bufs = np.stack(bufs)[take]
    rays = np.stack(rays)[take]
    ipf = np.array(ipf, np.int32)[take]
    zeros = np.zeros(n, np.int32)
    return (bufs, rays, ipf, zeros, zeros.copy(),
            shared_iono_tables([iono]), shared_antenna_tables([cant]))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _poison(mb=512):
    """Fill CuPy's pool with a garbage pattern and give it back, still dirty.

    The pool serves an allocation from a freed block only if one is big
    enough, so the ordering tests keep every rig buffer under the 16 MB block
    size: a bigger one is a fresh ``cudaMalloc``, which reads as zeros and may
    synchronise the device, hiding the race.  The pool's existing free
    blocks are released first: it prefers an exact-size block a freed rig
    left behind, and that may hold exactly what a fresh rig would write.
    """
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    blocks = [cp.full(16 * 2 ** 20 // 4, _POISON_WORD, np.int32)
              for _ in range(mb // 16)]
    cp.cuda.Device().synchronize()
    del blocks


def _device_arrays(obj, out=None):
    """Every CuPy array a rig owns (its slot buffers and table copies), but
    never the process-wide word maps in ``driver._MAP_CACHE``."""
    out = [] if out is None else out
    if isinstance(obj, cp.ndarray):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _device_arrays(v, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _device_arrays(v, out)
    return out


def _poison_rig(rig):
    """Dirty every buffer of `rig`, then free it.

    CuPy's pool serves an allocation from an exact-size free block first, so
    the NEXT rig built with the same arguments is handed exactly these
    blocks, still holding the pattern -- which makes "what does the work
    stream see before the zero-fill lands" deterministic rather than a matter
    of pool layout.
    """
    shared = {a.data.ptr for a in driver._MAP_CACHE.values()}
    seen = set()
    for a in _device_arrays(list(vars(rig).values())):
        base = a.base if a.base is not None else a
        if base.data.ptr in shared or id(base) in seen:
            continue
        seen.add(id(base))
        base.view(np.uint8).fill(0x3F)
    cp.cuda.Device().synchronize()
    del rig
    gc.collect()


def _rows_wrong(got, exp):
    """Rows differing BITWISE in any result field (son, xluf, jmode, status,
    and muf3 when the results carry it)."""
    g_res, g_muf = got if isinstance(got, tuple) else (got, None)
    e_res, e_muf = exp if isinstance(exp, tuple) else (exp, None)
    n = len(e_res.status)
    bad = np.zeros(n, bool)
    for k in e_res.son.dtype.names:
        g = np.ascontiguousarray(g_res.son[k]).view(np.uint8).reshape(n, -1)
        e = np.ascontiguousarray(e_res.son[k]).view(np.uint8).reshape(n, -1)
        bad |= (g != e).any(axis=1)
    bad |= (np.ascontiguousarray(g_res.xluf).view(np.uint32).reshape(n, -1)
            != np.ascontiguousarray(e_res.xluf).view(np.uint32)
            .reshape(n, -1)).any(axis=1)
    bad |= g_res.jmode != e_res.jmode
    bad |= g_res.status != e_res.status
    if e_muf is not None:
        bad |= (g_muf.view(np.uint32) != e_muf.view(np.uint32)).any(axis=1)
    return int(bad.sum())


def _settle():
    """Drop every reference cycle so freed rigs really go back to the pool."""
    gc.collect()
    cp.cuda.Device().synchronize()


_SPIN_SRC = r'''
extern "C" __global__ void spin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}
'''


def _busy_default_stream(seconds=0.15):
    """Keep the legacy default stream busy for ~`seconds`, WITHOUT waiting.

    Whether the race bites end to end depends on how far behind the default
    stream is when the work streams start -- on an idle card the zero-fills
    usually win, on a card shared with other work they often do not.  A spin
    kernel queued on the default stream makes "behind" deterministic, so the
    ordering tests below fail every time on an unfixed rig rather than
    sometimes.  It is kept well under the Windows TDR (2 s).
    """
    fn = cp.RawKernel(_SPIN_SRC, "spin")
    khz = cp.cuda.Device().attributes["ClockRate"]
    fn((1,), (1,), (np.int64(khz * 1000 * seconds),),
       stream=cp.cuda.Stream.null)


_PIN = {}


@pytest.fixture(scope="module", autouse=True)
def _pinned_scratch():
    """Page-locked read-back space, allocated BEFORE any spin.

    ``cudaHostAlloc`` can synchronise the whole device, which would hide the
    very race these tests look for; so the reader never allocates pinned
    memory itself.
    """
    _PIN["mem"] = cp.cuda.alloc_pinned_memory(96 * 2 ** 20)
    yield
    _PIN.clear()


def _read_on(stream, d_arr):
    """``d_arr`` copied to the host ON `stream` (a non-blocking work stream),
    i.e. exactly what that stream's first kernel would see."""
    rt = cp.cuda.runtime
    mem = _PIN["mem"]
    assert d_arr.nbytes <= mem.size(), (d_arr.nbytes, mem.size())
    h = np.frombuffer(mem, np.uint8, d_arr.nbytes)
    rt.memcpyAsync(h.ctypes.data, d_arr.data.ptr, d_arr.nbytes,
                   rt.memcpyDeviceToHost, stream.ptr)
    stream.synchronize()
    return h.copy().view(d_arr.dtype).reshape(d_arr.shape)


def _stale(stream, pairs, tag):
    """Names of the (name, device, expected-host) pairs that do NOT read as
    expected on `stream`, each prefixed with `tag`."""
    bad = []
    for name, d_arr, want in pairs:
        got = _read_on(stream, d_arr)
        if not np.array_equal(got.view(np.uint8).ravel(),
                              np.ascontiguousarray(want, got.dtype)
                              .view(np.uint8).ravel()):
            bad.append(f"{tag}:{name}")
    return bad


# ---------------------------------------------------------------------------
# the ordering, pinned deterministically
# ---------------------------------------------------------------------------
#
# Each iteration: poison the pool, queue a spin on the default stream, build
# (or re-table) a rig, and read its buffers back ON A WORK STREAM before
# anything else touches the default stream.  An unordered rig reads the
# poison; an ordered one reads what its constructor wrote.  Every rig type is
# built once first so the kernel modules are loaded: ``cuModuleLoad`` can
# synchronise the device and would hide the race on the first iteration.

ORDER_ITERATIONS = 5


def _zeros_like(d):
    return np.zeros(d.shape, d.dtype)


def _table_pairs(dev, host, prefix):
    return [(f"{prefix}.{k}", dev[k], np.asarray(host[k])) for k in dev]


def _warm():
    _busy_default_stream(0.001)
    cp.cuda.Device().synchronize()


def _poison_arrays(arrs):
    for a in arrs:
        (a.base if a.base is not None else a).view(np.uint8).fill(0x3F)
    cp.cuda.Device().synchronize()


def _swap_tables(install, attr):
    """Install a table set, dirty it, install again (releasing the dirty one
    to the pool), then -- with the default stream busy -- install a third
    time, which is handed the dirty blocks back."""
    install()
    _poison_arrays(_device_arrays(attr()))
    install()
    _busy_default_stream()
    install()


def test_pipelined_setup_init_ordered_before_work_streams():
    slots, iono, cfgs, ant, ant_idx, p = _setup_fixture(64)
    make = lambda: PipelinedSetupLuffy(256, 2, slots, iono)
    make()
    _warm()
    stale = []
    for _ in range(ORDER_ITERATIONS):
        _poison()
        _poison_rig(make())
        _busy_default_stream()
        rig = make()
        bad = []
        for s, st in enumerate(rig.streams):
            bad += _stale(st, [
                ("d_circ", rig.d_circ[s], _zeros_like(rig.d_circ[s])),
                ("d_soa", rig.d_soa[s], _zeros_like(rig.d_soa[s])),
                ("d_status", rig.d_status[s],
                 np.full(rig.chunk, -1, np.int32)),
                ("d_prm", rig.d_prm[s], _zeros_like(rig.d_prm[s])),
                ("d_ii", rig.d_ii[s], _zeros_like(rig.d_ii[s]))]
                + _table_pairs(rig._tables[0], iono, "iono"), f"init{s}")
        # later table swaps are uploaded on the default stream too
        _swap_tables(lambda: rig.set_antennas(ant), lambda: rig._ant)
        bad += _stale(rig.streams[0], _table_pairs(rig._ant, ant, "ant"),
                      "set_antennas")
        _swap_tables(lambda: rig.set_ionosphere(slots, iono),
                     lambda: rig._tables)
        bad += _stale(rig.streams[-1],
                      _table_pairs(rig._tables[0], iono, "iono"),
                      "set_ionosphere")
        stale.append(bad)
        del rig
        _settle()
    assert not any(stale), f"stale reads per iteration: {stale}"


def test_pipelined_luffy_init_ordered_before_work_streams(luffy_case):
    _b, _r, _ipf, _ii, _ai, io, an = luffy_case
    make = lambda: PipelinedLuffy(128, 2, io, an)
    make()
    _warm()
    stale = []
    for _ in range(ORDER_ITERATIONS):
        _poison()
        _poison_rig(make())
        _busy_default_stream()
        rig = make()
        bad = []
        for s, st in enumerate(rig.streams):
            bad += _stale(st, [
                ("d_circ", rig.d_circ[s], _zeros_like(rig.d_circ[s])),
                ("d_soa", rig.d_soa[s], _zeros_like(rig.d_soa[s])),
                ("d_status", rig.d_status[s],
                 np.full(rig.chunk, -1, np.int32))]
                + _table_pairs(rig._tables[0], io, "iono"), f"init{s}")
        stale.append(bad)
        del rig
        _settle()
    assert not any(stale), f"stale reads per iteration: {stale}"


def test_serial_rigs_init_complete_on_return(luffy_case):
    """LuffyArena / SetupLuffy run on the caller's current stream, but a
    caller is free to submit from another one; construction must have
    landed by the time it returns."""
    _b, _r, _ipf, _ii, _ai, io, an = luffy_case
    slots, iono, cfgs, ant, ant_idx, p = _setup_fixture(64)
    make_a = lambda: LuffyArena(256, io, an)
    make_s = lambda: SetupLuffy(256, slots, iono, ant)
    make_a()
    make_s()
    _warm()
    other = cp.cuda.Stream(non_blocking=True)
    stale = []
    for _ in range(ORDER_ITERATIONS):
        _poison()
        _poison_rig(make_a())
        _busy_default_stream()
        arena = make_a()
        bad = _stale(other, [
            ("d_circ", arena._d_circ, _zeros_like(arena._d_circ)),
            ("d_soa", arena._d_soa, _zeros_like(arena._d_soa)),
            ("d_status", arena._d_status, np.full(256, -1, np.int32))]
            + _table_pairs(arena._tables[0], io, "iono"), "LuffyArena")
        _poison_rig(make_s())
        _busy_default_stream()
        rig = make_s()
        bad += _stale(other, [
            ("d_circ", rig._d_circ, _zeros_like(rig._d_circ)),
            ("d_soa", rig._d_soa, _zeros_like(rig._d_soa)),
            ("d_status", rig._d_status, np.full(256, -1, np.int32))]
            + _table_pairs(rig._ant, ant, "ant"), "SetupLuffy")
        _swap_tables(lambda: rig.set_antennas(ant), lambda: rig._ant)
        bad += _stale(other, _table_pairs(rig._ant, ant, "ant"),
                      "SetupLuffy.set_antennas")
        stale.append(bad)
        del arena, rig
        _settle()
    assert not any(stale), f"stale reads per iteration: {stale}"



# ---------------------------------------------------------------------------
# the race, end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunk,ns", [(997, 2), (512, 3)])
def test_pipelined_setup_after_pool_poisoning(chunk, ns):
    n = 3000
    slots, iono, cfgs, ant, ant_idx, p = _setup_fixture(n)
    ref = SetupLuffy(n, slots, iono, ant).run(p, cfgs, ant_idx)
    assert (ref[0].status == 0).all()
    wrong = []
    for _ in range(ITERATIONS):
        _poison()
        got = PipelinedSetupLuffy(chunk, ns, slots, iono).run(
            p, cfgs, ant, ant_idx)
        wrong.append(_rows_wrong(got, ref))
        del got
        _settle()
    assert not any(wrong), f"rows wrong per iteration: {wrong}"


def test_new_pipelined_setup_rig_after_previous_freed():
    """The everyday shape: build, run, free, build again on the dirty memory."""
    n = 3000
    slots, iono, cfgs, ant, ant_idx, p = _setup_fixture(n, seed=7)
    ref = SetupLuffy(n, slots, iono, ant).run(p, cfgs, ant_idx)
    # a DIFFERENT workload first, so the freed memory holds real but wrong
    # state rather than zeros or the reference's own leftovers
    _s, _i, cfgs2, ant2, ai2, p2 = _setup_fixture(4096, seed=99)
    wrong = []
    for _ in range(ITERATIONS):
        dirty = PipelinedSetupLuffy(1024, 2, slots, iono)
        dirty.run(p2, cfgs2, ant2, ai2)
        del dirty
        gc.collect()
        pipe = PipelinedSetupLuffy(1024, 2, slots, iono)
        got = pipe.run(p, cfgs, ant, ant_idx)
        wrong.append(_rows_wrong(got, ref))
        del pipe, got
        _settle()
    assert not any(wrong), f"rows wrong per iteration: {wrong}"


def test_setup_luffy_after_pool_poisoning():
    """The serial rig: one stream throughout, but pin it all the same."""
    n = 1500
    slots, iono, cfgs, ant, ant_idx, p = _setup_fixture(n, seed=3)
    ref = SetupLuffy(n, slots, iono, ant).run(p, cfgs, ant_idx)
    wrong = []
    for _ in range(ITERATIONS):
        _poison()
        got = SetupLuffy(n, slots, iono, ant).run(p, cfgs, ant_idx)
        wrong.append(_rows_wrong(got, ref))
        del got
        _settle()
    assert not any(wrong), f"rows wrong per iteration: {wrong}"


@pytest.fixture(scope="module")
def luffy_case():
    return _luffy_fixture(600)


@pytest.mark.parametrize("chunk,ns", [(600, 1), (128, 2)])
def test_pipelined_luffy_after_pool_poisoning(luffy_case, chunk, ns):
    bufs, rays, ipf, ii, ai, io, an = luffy_case
    n = len(bufs)
    ref = LuffyArena(n, io, an).run(bufs, rays, ipf, ii, ai)
    wrong = []
    for _ in range(ITERATIONS):
        _poison()
        got = PipelinedLuffy(chunk, ns, io, an).run(bufs, rays, ipf, ii, ai)
        wrong.append(_rows_wrong(got, ref))
        del got
        _settle()
    assert not any(wrong), f"rows wrong per iteration: {wrong}"


def test_new_pipelined_luffy_after_previous_freed(luffy_case):
    bufs, rays, ipf, ii, ai, io, an = luffy_case
    n = len(bufs)
    ref = LuffyArena(n, io, an).run(bufs, rays, ipf, ii, ai)
    wrong = []
    for _ in range(ITERATIONS):
        dirty = PipelinedLuffy(160, 2, io, an)
        dirty.run(bufs[::-1].copy(), rays[::-1].copy(), ipf[::-1].copy(),
                  ii, ai)
        del dirty
        gc.collect()
        got = PipelinedLuffy(160, 2, io, an).run(bufs, rays, ipf, ii, ai)
        wrong.append(_rows_wrong(got, ref))
        del got
        _settle()
    assert not any(wrong), f"rows wrong per iteration: {wrong}"


# ---------------------------------------------------------------------------
# the input checks
# ---------------------------------------------------------------------------

_BAD = [
    ("hour", 0), ("hour", 25), ("hour", 60), ("hour", -3),
    ("tlatd", 90.5), ("tlatd", np.nan), ("rlatd", -91.0), ("rlatd", np.inf),
    ("tlongd", -np.inf), ("rlongd", np.nan),
    ("npsl", 2), ("npsl", -1),
    ("iono", 2), ("iono", -1),
    ("cfg", 2), ("cfg", -1),
    ("ant_idx", 3), ("ant_idx", -1),
]


@pytest.fixture(scope="module")
def small_setup():
    return _setup_fixture(40, seed=11)


def _corrupt(p, ant_idx, field, value, rows=(3, 17)):
    p = p.copy()
    ant_idx = ant_idx.copy()
    for r in rows:
        if field == "ant_idx":
            ant_idx[r] = value
        else:
            p[field][r] = value
    return p, ant_idx


@pytest.mark.parametrize("rig", ["serial", "pipelined"])
@pytest.mark.parametrize("field,value", _BAD)
def test_setup_input_refused(small_setup, rig, field, value):
    slots, iono, cfgs, ant, ant_idx, p = small_setup
    bp, bai = _corrupt(p, ant_idx, field, value)
    if rig == "serial":
        call = lambda: SetupLuffy(64, slots, iono, ant).run(bp, cfgs, bai)
    else:
        call = lambda: PipelinedSetupLuffy(16, 2, slots, iono).run(
            bp, cfgs, ant, bai)
    with pytest.raises(ValueError) as e:
        call()
    msg = str(e.value)
    assert field in msg, msg
    assert "3" in msg and "17" in msg, f"offending rows not named: {msg}"


def test_setup_boundary_values_accepted(small_setup):
    slots, iono, cfgs, ant, ant_idx, p = small_setup
    p = p.copy()
    p["hour"][:2] = (1, 24)
    p["rlatd"][2:4] = (-90.0, 90.0)
    p["rlongd"][4:6] = (-180.0, 180.0)
    p["npsl"][6] = 1
    res, _m = PipelinedSetupLuffy(16, 2, slots, iono).run(p, cfgs, ant,
                                                          ant_idx)
    assert len(res.status) == len(p)


@pytest.mark.parametrize("rig", ["serial", "pipelined"])
def test_setup_longitude_wrapped_not_refused(small_setup, rig):
    """Out-of-range longitudes are wrapped (on a copy): rows given as 190 /
    -200 compute exactly what -170 / 160 do, and the caller's array is left
    as it was."""
    slots, iono, cfgs, ant, ant_idx, p = small_setup
    canon = p.copy()
    canon["rlongd"][3], canon["tlongd"][17] = -170.0, 160.0
    given = canon.copy()
    given["rlongd"][3], given["tlongd"][17] = 190.0, -200.0
    before = given.copy()
    if rig == "serial":
        run = lambda q: SetupLuffy(64, slots, iono, ant).run(q, cfgs, ant_idx)
    else:
        run = lambda q: PipelinedSetupLuffy(16, 2, slots, iono).run(
            q, cfgs, ant, ant_idx)
    a, _ = run(canon)
    b, _ = run(given)
    assert np.array_equal(given, before), "caller's params were mutated"
    for f in ("sndb", "dbw", "reliab", "nhp"):
        assert_bitexact(np.ascontiguousarray(b.son[f]).ravel(),
                        np.ascontiguousarray(a.son[f]).ravel(), f"son.{f}")
    assert np.array_equal(a.status, b.status)


def test_setup_type_and_shape_errors(small_setup):
    slots, iono, cfgs, ant, ant_idx, p = small_setup
    pipe = PipelinedSetupLuffy(16, 1, slots, iono)
    rig = SetupLuffy(64, slots, iono, ant)
    wrong_dtype = np.zeros(len(p), [("x", np.float32)])
    with pytest.raises(TypeError):
        pipe.run(wrong_dtype, cfgs, ant, ant_idx)
    with pytest.raises(TypeError):
        rig.run(wrong_dtype, cfgs, ant_idx)
    with pytest.raises(TypeError):
        pipe.run(p, np.zeros(2, np.float32), ant, ant_idx)
    with pytest.raises(ValueError):
        pipe.run(p, cfgs, ant, ant_idx[:-1])
    with pytest.raises(ValueError):
        rig.run(p, cfgs, ant_idx[:-1])
    with pytest.raises(ValueError):
        rig.run(np.concatenate([p, p]), cfgs, np.concatenate([ant_idx] * 2))
    with pytest.raises(ValueError):
        PipelinedSetupLuffy(16, 1).run(p, cfgs, ant, ant_idx)   # no iono


def test_device_hour_guard(small_setup, monkeypatch):
    """With the host check bypassed, a bad hour is a STATUS, not a read past
    the 25-row VIRTIM table -- and its neighbours are untouched."""
    slots, iono, cfgs, ant, ant_idx, p = small_setup
    ref_s = SetupLuffy(64, slots, iono, ant).run(p, cfgs, ant_idx)
    ref_p = PipelinedSetupLuffy(16, 2, slots, iono).run(p, cfgs, ant, ant_idx)
    bad_rows = np.array([1, 5, 9, 30])
    q = p.copy()
    q["hour"][bad_rows] = (60, 25, 0, -3)
    monkeypatch.setattr(driver, "_check_setup_params",
                        lambda params, *a, **k: params)   # bypass, pass through
    got_s = SetupLuffy(64, slots, iono, ant).run(q, cfgs, ant_idx)
    got_p = PipelinedSetupLuffy(16, 2, slots, iono).run(q, cfgs, ant, ant_idx)
    good = np.setdiff1d(np.arange(len(p)), bad_rows)
    for (got, ref) in ((got_s, ref_s), (got_p, ref_p)):
        st = got[0].status
        assert (st[bad_rows] == driver.SETUP_STATUS_BAD_HOUR).all(), st
        sub = lambda t: (driver.LuffyResults(t[0].son[good], t[0].xluf[good],
                                             t[0].jmode[good],
                                             t[0].status[good]), t[1][good])
        assert _rows_wrong(sub(got), sub(ref)) == 0


def test_checks_survive_python_O(tmp_path):
    """The refusal is a real exception, not an assert ``-O`` strips."""
    root = pathlib.Path(__file__).resolve().parents[2]
    script = tmp_path / "o_check.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "assert False, 'asserts are live: not running under -O'\n",
        encoding="utf-8")
    # first prove -O really strips asserts in this interpreter
    r = subprocess.run([sys.executable, "-O", str(script)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "import numpy as np\n"
        "from tests.cuda.test_rig_race import _setup_fixture\n"
        "from voacap_gpu.cuda.driver import PipelinedSetupLuffy\n"
        "slots, iono, cfgs, ant, ai, p = _setup_fixture(8)\n"
        "p['hour'][2] = 60\n"
        "try:\n"
        "    PipelinedSetupLuffy(8, 1, slots, iono).run(p, cfgs, ant, ai)\n"
        "except ValueError as e:\n"
        "    print('REFUSED', e)\n"
        "    sys.exit(0)\n"
        "sys.exit(3)\n", encoding="utf-8")
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-O", str(script)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "REFUSED" in r.stdout
