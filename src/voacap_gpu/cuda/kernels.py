"""The NVRTC kernel sources, their loaders, and the device transfer helpers.

Every CUDA translation unit the driver compiles lives here, byte-for-byte:
the LUFFY batch kernels, the word scatter/gather, the device-setup kernel and
the grid reduction, each compiled through ``compile.load`` (the strict
option set).  Alongside them, the host<->device movers the rigs build on:
the SoA /RAYS/ transpose, the compacted input/result word maps (uploaded once
per process into ``_MAP_CACHE``) and the device-side input writers.
"""

import numpy as np

from voacap_gpu.cuda.abi import (_CIRC_WORDS, _RAYS_CHUNK, _RAYS_LEN,
                                 _RAYS_OFF, _RAYS_UPLOAD_RUNS, _RAYS_WORDS,
                                 _RESULT_RUNS, _UPLOAD_IDX, _UPLOAD_RUNS,
                                 CIRCUIT_DTYPE, I32)
from voacap_gpu.cuda.compile import load
from voacap_gpu.cuda.state import RAYS_DTYPE


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------

_KERNEL_SRC = r'''
#include "luffy.cuh"
#include "luffy_warp.cuh"

// setup.cuh's hour guard hands LUFFY this IPFLAG for a circuit it refused to
// build; no legal IPFLAG is negative.  Such a row is answered with status
// VC_STATUS_TABLE_OOB (VC_ST_SETUP_HOUR_OOB in the global status table,
// kernels/status.cuh) and LUFFY never runs on its (unbuilt) state.  The
// values must match setup.cuh's; both are guarded so either may come first.
#ifndef VC_IPFLAG_REFUSED
#define VC_IPFLAG_REFUSED   (-1)
#endif
#ifndef VC_STATUS_TABLE_OOB
#define VC_STATUS_TABLE_OOB VC_ST_SETUP_HOUR_OOB
#endif

extern "C" __global__ void luffy_batch(
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ipflag[i] == VC_IPFLAG_REFUSED) {
        status[i] = VC_STATUS_TABLE_OOB;
        return;
    }
    VC_PROF_BEGIN(VC_PROF_KERNEL);
    // THE STATE RUNS IN LOCAL MEMORY.  circ[] is array-of-structures, so a
    // warp's 32 lanes reading the same field touch 32 lines 24,980 bytes
    // apart; local memory is interleaved per 4-byte word across the warp, so
    // the same access is one or two lines.  GENION's and SETTXR's table scans
    // went 2-8x faster per thread.  The copy in and out is ~8% of a thread's
    // time and moves every byte, so the whole-state semantics (fetch_luffy,
    // the 7M-cell replay) are unchanged.  The cost is device memory: CUDA
    // reserves local memory for EVERY thread slot of the device (sizeof(
    // CircuitState) + the SmoothStore etc. ~= 29 KB x 1536 x SMs, ~3.5 GB on
    // an 84-SM RTX 5080) once per context.  -DVC_LUFFY_GLOBAL_STATE restores
    // the in-place form (~0.46 GB reserved, ~2x slower kernel).
#ifndef VC_LUFFY_GLOBAL_STATE
    CircuitState cl = circ[i];
    CircuitState* cs = &cl;
#else
    CircuitState* cs = &circ[i];
#endif
    status[i] = luffy(cs, rays_nang, rays_ang, rays_ifob, n, i,
                      fam, dud, fakp, fakmap, hmym, fakabp, abmap,
                      sys, perr, f2d111, ant_array, ant_aeff, ant_iats,
                      ant_xfqs, ant_xfqe, ant_offazim, ant_pwrdba, numants,
                      iono_idx[i], ant_idx[i], ipflag[i]);
#ifndef VC_LUFFY_GLOBAL_STATE
    circ[i] = cl;
#endif
    VC_PROF_END(VC_PROF_KERNEL);
}

// ONE WARP PER CIRCUIT-HOUR, ONE LANE PER FREQUENCY (cuda/luffy_warp.cuh).
// blockDim.x must be 32 * warps_per_block, and the dynamic shared memory
// warps_per_block * sizeof(SmoothStore) -- one LUFFY "TEMPORARY STORAGE" per
// warp, because the twelve lanes fill disjoint rows of it and lane 0 reads
// the whole thing back in the blend.
extern "C" __global__ void luffy_batch_warp(
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
    // VC_WARP_GROUPS packs that many circuit-hours into one warp; each gets
    // VC_WARP_GLANES contiguous lanes and every collective in luffy_warp.cuh
    // is group-local.  At the default 1 these fold away: grp is 0, slot is
    // wib, and the lane test is `lane == 0`, i.e. the code below is exactly
    // what it was before packing existed.
    int grp = lane / VC_WARP_GLANES;
    int slot = wib * VC_WARP_GROUPS + grp;
    int i = blockIdx.x * ((blockDim.x >> 5) * VC_WARP_GROUPS) + slot;
    if (i >= n) return;
    // uniform across the group (one circuit-hour), and every collective in
    // luffy_warp.cuh is group-masked, so a whole group may leave here exactly
    // as a tail group leaves at `i >= n`
    if (ipflag[i] == VC_IPFLAG_REFUSED) {
        if ((lane & (VC_WARP_GLANES - 1)) == 0)
            status[i] = VC_STATUS_TABLE_OOB;
        return;
    }
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

extern "C" __global__ void circuit_sizeof(int* out) {
    out[0] = (int)sizeof(CircuitState);
}
'''


def load_kernel(groups: int = 1, fast_math: bool = False):
    """Compile (once, cached per option set) the LUFFY batch kernel module.

    ``groups`` packs that many circuit-hours into ONE warp for
    ``luffy_batch_warp`` -- 12 frequencies need at least 12 lanes, so 1 and 2
    are the legal values.  2 lifts lane utilisation from 12/32 to 24/32 and
    moves the crossover against the thread kernel a long way right; see
    cuda/luffy_warp.cuh.  It changes nothing about the thread kernel.

    ``fast_math`` selects the opt-in native-f32 libm (see cuda/f32math.cuh).
    It is NOT bit-exact against Stage A -- tests/cuda/test_fastmath.py is
    what pins what it costs -- so it is off unless asked for.
    """
    if groups not in (1, 2):
        raise ValueError(
            f"VC_WARP_GROUPS must be 1 or 2 (NFREQ=12 needs >=12 lanes per "
            f"group), got {groups}")
    opts = ()
    if groups != 1:
        opts += (f"-DVC_WARP_GROUPS={groups}",)
    if fast_math:
        opts += ("-DVOACAP_FAST_MATH",)
    return load(_KERNEL_SRC, opts)


def circuit_state_sizeof() -> int:
    """sizeof(CircuitState) as the device compiler computes it."""
    import cupy as cp
    out = cp.zeros(1, np.int32)
    load_kernel().get_function("circuit_sizeof")((1,), (1,), (out,))
    cp.cuda.runtime.deviceSynchronize()
    return int(cp.asnumpy(out)[0])


def _rays_to_soa(rays_bufs, n):
    """(N,) RAYS_DTYPE on the host -> one (words, N) int32 device array.

    Staged in chunks: the transpose needs somewhere to land, and a whole
    second copy of /RAYS/ is 1 GB at 32k threads.
    """
    import cupy as cp

    d_soa = cp.empty((_RAYS_WORDS, n), np.int32)
    for s in range(0, n, _RAYS_CHUNK):
        e = min(n, s + _RAYS_CHUNK)
        blk = cp.asarray(rays_bufs[s:e].view(np.uint8))
        d_soa[:, s:e] = blk.view(np.int32).reshape(e - s, _RAYS_WORDS).T
    return d_soa


# ---------------------------------------------------------------------------
# The compacted-block scatter/gather, one launch each
# ---------------------------------------------------------------------------
# CircuitState's input words are 19 disjoint runs and its result words 3, so
# the obvious implementation is one strided device copy per run.  That is 19
# kernel launches per upload, and on a busy card those launches queue: at 504
# threads the run-at-a-time version cost MORE than shipping the whole 55,652
# bytes.  One launch over a word-index map instead.  This is integer address
# arithmetic only -- no float touches it -- so the fidelity contract in
# compile.py has nothing to bite on.
_XFER_SRC = r'''
extern "C" __global__ void scatter_words(
        int* dst, const int* src, const int* idx, int m, int stride, long tot) {
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= tot) return;
    long i = g / m;                         // which thread's state
    int j = (int)(g - i * m);               // which word of the compact block
    dst[i * (long)stride + idx[j]] = src[g];
}

extern "C" __global__ void gather_words(
        const int* src, int* dst, const int* idx, int m, int stride, long tot) {
    long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= tot) return;
    long i = g / m;
    int j = (int)(g - i * m);
    dst[g] = src[i * (long)stride + idx[j]];
}
'''


def _xfer(name):
    return load(_XFER_SRC).get_function(name)


def _word_map(runs):
    """Contiguous (offset, count) runs -> the flat int32 word-index map."""
    return np.concatenate([np.arange(s, s + ln, dtype=I32)
                           for s, ln in runs])


_UPLOAD_MAP = _word_map(_UPLOAD_RUNS)
_RESULT_MAP = _word_map(_RESULT_RUNS)
_MAP_CACHE = {}


def _device_map(key, host_map):
    """The word-index map — a few hundred int32, uploaded once per process."""
    import cupy as cp
    d = _MAP_CACHE.get(key)
    if d is None:
        d = _MAP_CACHE[key] = cp.asarray(host_map)
    return d


def _write_rays_inputs(d_soa, rays_bufs, n):
    """Send ``RAYS_UPLOAD_FIELDS`` into an existing SoA buffer, nothing else.

    NANG + ANG are 42 words, so the transpose is done on the HOST -- 168 bytes
    a thread is nothing to gather, and it keeps the device side to one
    contiguous copy per run.  IFOB is scratch and is not touched.
    """
    import cupy as cp

    words = rays_bufs.view(np.int32).reshape(n, _RAYS_WORDS)
    for s, ln in _RAYS_UPLOAD_RUNS:
        d_soa[s:s + ln, :] = cp.asarray(
            np.ascontiguousarray(words[:, s:s + ln].T))


def _write_circuit_inputs(d_circ, bufs, n):
    """Send ``UPLOAD_FIELDS`` into an existing device CircuitState array.

    ONE host-side gather into a compacted ``(N, m)`` int32 block, ONE H2D of
    that block, ONE scatter kernel.  Everything outside ``UPLOAD_FIELDS`` is
    left exactly as it was, which costs no PCIe at all.

    The gather is a CONCATENATION OF THE 19 CONTIGUOUS RUNS, not a fancy-index
    over the 409 word offsets, and that is not a style preference: at 32,256
    circuit-hours the fancy-index form takes 163 ms and this one 31 ms.  The
    element-at-a-time path costs more on the host than the bytes it saves on
    the bus.
    """
    import cupy as cp

    m = len(_UPLOAD_IDX)
    words = bufs.view(np.int32).reshape(n, _CIRC_WORDS)
    src = np.concatenate([words[:, s:s + ln] for s, ln in _UPLOAD_RUNS],
                         axis=1)
    d_src = cp.asarray(src)
    tot = m * n
    _xfer("scatter_words")(((tot + 255) // 256,), (256,),
                           (d_circ.view(np.int32), d_src,
                            _device_map("upload", _UPLOAD_MAP), I32(m),
                            I32(_CIRC_WORDS), np.int64(tot)))


def _rays_inputs_to_soa(rays_bufs, n):
    """A fresh zeroed SoA /RAYS/ buffer carrying only the input words."""
    import cupy as cp

    d_soa = cp.zeros((_RAYS_WORDS, n), np.int32)
    _write_rays_inputs(d_soa, rays_bufs, n)
    return d_soa


def _circuit_inputs_to_device(bufs, n):
    """A fresh zeroed device CircuitState array carrying only the inputs."""
    import cupy as cp

    d_circ = cp.zeros(n * CIRCUIT_DTYPE.itemsize, np.uint8)
    _write_circuit_inputs(d_circ, bufs, n)
    return d_circ


def _rays_from_soa(d_soa, n):
    """The inverse: one (words, N) int32 device array -> (N,) RAYS_DTYPE.

    ``get(out=...)`` lands the transposed block straight in the destination
    rows; going through ``cp.asnumpy`` would add a full host-side copy of the
    whole batch, which at 32k threads is a gigabyte for nothing.
    """
    import cupy as cp

    flat = np.empty((n, _RAYS_WORDS), np.int32)
    for s in range(0, n, _RAYS_CHUNK):
        e = min(n, s + _RAYS_CHUNK)
        blk = cp.ascontiguousarray(d_soa[:, s:e].T)
        blk.get(out=flat[s:e])
    return flat.view(RAYS_DTYPE).reshape(n)


def _rays_views(d_soa):
    """The three base pointers the kernel takes, as views into `d_soa`."""
    o, ln = _RAYS_OFF, _RAYS_LEN
    d_nang = d_soa[o["nang"]:o["nang"] + ln["nang"]]
    d_ang = d_soa[o["ang"]:o["ang"] + ln["ang"]].view(np.float32)
    d_ifob = d_soa[o["ifob"]:o["ifob"] + ln["ifob"]]
    return d_nang, d_ang, d_ifob


def smoothstore_sizeof() -> int:
    """sizeof(SmoothStore) — the per-warp dynamic shared memory the warp
    kernel needs (LUFFY's "TEMPORARY STORAGE", cuda/lsmooth.cuh)."""
    import cupy as cp
    out = cp.zeros(1, np.int32)
    load(_KERNEL_SRC + r'''
extern "C" __global__ void smoothstore_size(int* o) {
    o[0] = (int)sizeof(SmoothStore);
}
''').get_function("smoothstore_size")((1,), (1,), (out,))
    cp.cuda.runtime.deviceSynchronize()
    return int(cp.asnumpy(out)[0])


# cuda/lsmooth.cuh: 20 (NFREQ+2, 3) columns + 5 (NFREQ+2,) extras, 4 bytes each
SMOOTHSTORE_BYTES = (20 * 14 * 3 + 5 * 14) * 4


_GRIDREDUCE_SRC = r'''
#include "gridreduce.cuh"
'''


def load_gridreduce_kernel():
    """Compile (once) cuda/kernels/gridreduce.cuh: ``grid_reduce_chunk`` and
    ``grid_finalise``.  Pure binary64 compare/add/divide under the strict
    option set (no FMA contraction), so it is exact against numpy float64."""
    return load(_GRIDREDUCE_SRC)


# ---------------------------------------------------------------------------
# THE DEVICE-SETUP PATH: raw parameters in, results out, no host build at all
# ---------------------------------------------------------------------------

_SETUP_SRC = r'''
#include "setup.cuh"

extern "C" __global__ void setup_batch(
        CircuitState* circ,
        int* rays_nang, float* rays_ang,
        int* ipflag, float* muf3,
        const SetupParams* prm, const SetupConfig* cfg,
        const float* ab, const int* ikim,
        const float* anew, const float* bnew,
        const float* achi, const float* bchi,
        const float* fakp, const float* fakmap, const float* hmym,
        const float* fakabp, const float* abmap,
        const float* f2d, const int* month, const float* ssn,
        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    SetupTables t;
    t.ab = ab; t.ikim = ikim;
    t.anew = anew; t.bnew = bnew; t.achi = achi; t.bchi = bchi;
    t.fakp = fakp; t.fakmap = fakmap; t.hmym = hmym;
    t.fakabp = fakabp; t.abmap = abmap;
    t.f2d = f2d; t.month = month; t.ssn = ssn;
    // IFOB is scratch to the setup chain -- SANG writes NANG and ANG only --
    // so the view's third pointer is never dereferenced here.
    RaysView rays = rays_view_soa(rays_nang, rays_ang, nullptr, n, i);
    SetupParams p = prm[i];
    ipflag[i] = setup_circuit(&circ[i], rays, p, cfg[p.cfg], t,
                              muf3 + (size_t)i * 3);
}

extern "C" __global__ void setup_struct_sizes(int* out) {
    out[0] = (int)sizeof(SetupParams);
    out[1] = (int)sizeof(SetupConfig);
}
'''


def load_setup_kernel(fast_math: bool = False):
    """Compile (once, cached per option set) the device-setup kernel module."""
    opts = ("-DVOACAP_FAST_MATH",) if fast_math else ()
    return load(_SETUP_SRC, opts)


def setup_struct_sizeof():
    """``(sizeof(SetupParams), sizeof(SetupConfig))`` per the device compiler.

    The same offset proof ``circuit_state_sizeof`` carries for CircuitState:
    every member of both structs is four bytes, so numpy's unaligned packing
    equals the C layout iff the totals agree.
    """
    import cupy as cp
    out = cp.zeros(2, np.int32)
    load_setup_kernel().get_function("setup_struct_sizes")((1,), (1,), (out,))
    cp.cuda.runtime.deviceSynchronize()
    return tuple(int(v) for v in cp.asnumpy(out))
