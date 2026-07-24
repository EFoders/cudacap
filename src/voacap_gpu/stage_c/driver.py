"""Host side of Stage C's LUFFY kernel — one thread per circuit-hour.

``cuda/luffy.cuh`` composes the thirty-odd ported leaves into one
``__device__`` function that runs a whole hour of system performance for one
circuit.  This module is the other half: it lays a Stage A ``Circuit`` out in
the byte order ``cuda/circuit.cuh`` declares, moves the SHARED tables (the
REDMAP coefficient maps and the antenna patterns) once, launches N threads,
and reads the mutated states back into Stage A objects.

THE ABI, and how it is proved
-----------------------------
``CIRCUIT_DTYPE`` is a numpy structured dtype built from the per-struct
dtypes in ``stage_c/state.py`` — the same ones ``tests/stage_c/test_state.py``
already pins field-for-field against ``state.cuh`` — followed by the loose
scalars in Stage A ``Circuit`` order.  Every member is 4 bytes, so numpy's
unaligned packing equals the C layout.  ``circuit_state_sizeof()`` compiles a
one-line kernel that writes ``sizeof(CircuitState)``; the test asserts it
equals ``CIRCUIT_DTYPE.itemsize``, which is the same offset proof the state
ABI already carries.

``iono`` and ``cant`` are NOT in the dtype.  They are big, read-only and
shared between threads, so they move once as flat device arrays and each
thread gets an index into them.

/SON/ MDL
---------
``pack_son``/``unpack_son`` encode ``mdl``/``mode``/``moder`` as LAYTYP
indices.  That is right for MODE and MODER, but MDL is not a layer label: it
is LUFFY's one-character model tag, "" / " " / "S" / "L" / "M".  Rather than
duplicate ``state.py``'s logic this module calls the same packers with MDL
blanked and then overwrites that one column with ``MDL_CODE`` — which is what
``cuda/circuit.cuh``'s ``VC_MDL_*`` constants hold.
"""

import numpy as np

from voacap_gpu.stage_c.compile import load
from voacap_gpu.stage_c.state import (ALLMODE_DTYPE, ANOIS_DTYPE, FRQ_DTYPE,
                                      LAYER_DTYPE, LOSX_DTYPE, LPATH_DTYPE,
                                      MODES_DTYPE, MUF_DTYPE, NOISEDIAG_DTYPE,
                                      RAYS_DTYPE, REFLX_DTYPE, SIGD_DTYPE,
                                      SON_DTYPE, TIME_DTYPE, TON_DTYPE,
                                      ZON_DTYPE, pack_allmode, pack_anois,
                                      pack_frq, pack_layer, pack_losx,
                                      pack_lpath, pack_modes, pack_muf,
                                      pack_noisediag, pack_rays, pack_reflx,
                                      pack_sigd, pack_son, pack_time,
                                      pack_ton, pack_zon, unpack_allmode,
                                      unpack_anois, unpack_frq, unpack_layer,
                                      unpack_losx, unpack_lpath, unpack_modes,
                                      unpack_muf, unpack_noisediag,
                                      unpack_rays, unpack_reflx, unpack_sigd,
                                      unpack_son, unpack_time, unpack_ton,
                                      unpack_zon)

F32 = np.float32
I32 = np.int32

# /SON/ MDL, LUFFY's model tag — cuda/circuit.cuh VC_MDL_*.
MDL_CODE = {"": -1, " ": 0, "S": 1, "L": 2, "M": 3}
MDL_LABEL = {v: k for k, v in MDL_CODE.items()}

# (Circuit attribute, per-struct dtype, pack fn, unpack fn), in Circuit order.
_BLOCKS = (
    ("layers", LAYER_DTYPE, pack_layer, unpack_layer),
    ("muf", MUF_DTYPE, pack_muf, unpack_muf),
    ("rays", RAYS_DTYPE, pack_rays, unpack_rays),
    ("reflx", REFLX_DTYPE, pack_reflx, unpack_reflx),
    ("modes", MODES_DTYPE, pack_modes, unpack_modes),
    ("zon", ZON_DTYPE, pack_zon, unpack_zon),
    ("allm", ALLMODE_DTYPE, pack_allmode, unpack_allmode),
    ("son", SON_DTYPE, None, None),                 # MDL needs the shim below
    ("anois", ANOIS_DTYPE, pack_anois, unpack_anois),
    ("ndiag", NOISEDIAG_DTYPE, pack_noisediag, unpack_noisediag),
    ("sigd", SIGD_DTYPE, pack_sigd, unpack_sigd),
    ("ton", TON_DTYPE, pack_ton, unpack_ton),
    ("frq", FRQ_DTYPE, pack_frq, unpack_frq),
    ("time", TIME_DTYPE, pack_time, unpack_time),
    ("losx", LOSX_DTYPE, pack_losx, unpack_losx),
    ("lpath", LPATH_DTYPE, pack_lpath, unpack_lpath),
)

# The loose scalars, in Circuit field order.  `nyear` is a CHARACTER*5 on the
# Stage A side and only ever matters as the test ``nyear == " 2089"``, so it
# is carried as the int flag the ported GENOIS already takes.
_SCALARS = (
    ("gcd", F32), ("gcdkm", F32), ("amind", F32), ("rsn", F32),
    ("dmp", F32), ("pmp", F32), ("rlat", F32), ("rlong", F32),
    ("rlongd", F32), ("tlong", F32),
    ("lufp", I32), ("method", I32), ("noise", I32),
    ("mspec", I32), ("jlong", I32),
    ("nyear_is_2089", I32), ("want_service_prob", I32), ("info", I32),
)

CIRCUIT_DTYPE = np.dtype([(n, dt) for n, dt, _, _ in _BLOCKS]
                         + list(_SCALARS))

# Flat table extents, one circuit's worth of each (redmap.py shapes).
_IONO_TABLES = (("fam", (14, 12)), ("dud", (5, 12, 5)),
                ("fakp", (29, 16, 6)), ("fakmap", (29, 16)),
                ("hmym", (29, 16)), ("fakabp", (2, 6)), ("abmap", (2, 3)),
                ("sys", (9, 16, 6)), ("perr", (9, 4, 6)))


# ---------------------------------------------------------------------------
# /SON/ pack / unpack with the MDL shim
# ---------------------------------------------------------------------------

def _pack_son_mdl(son):
    """pack_son, with MDL re-encoded as a model tag instead of a LAYTYP index."""
    saved = list(son.mdl)
    son.mdl = [""] * len(saved)
    try:
        buf = pack_son(son)
    finally:
        son.mdl = saved
    buf["mdl"] = np.array([MDL_CODE[s] for s in saved], I32)
    return buf


def _unpack_son_mdl(buf, son):
    codes = np.asarray(buf["mdl"]).copy()
    tmp = np.array(buf).copy()
    tmp["mdl"] = -1                              # "" — a legal LAYTYP index
    unpack_son(tmp, son)
    son.mdl = [MDL_LABEL[int(i)] for i in codes]
    return son


# ---------------------------------------------------------------------------
# whole-circuit pack / unpack
# ---------------------------------------------------------------------------

def pack_circuit(circ) -> np.ndarray:
    """One Stage A Circuit -> a shape-() array of CIRCUIT_DTYPE.

    `iono` and `cant` are deliberately absent: they are shared tables, moved
    once by ``shared_iono_tables`` / ``shared_antenna_tables``.
    """
    buf = np.zeros((), dtype=CIRCUIT_DTYPE)
    for name, _dt, pack, _un in _BLOCKS:
        obj = getattr(circ, name)
        buf[name] = _pack_son_mdl(obj) if pack is None else pack(obj)
    for name, _kind in _SCALARS:
        if name == "nyear_is_2089":
            buf[name] = I32(circ.nyear == " 2089")
        elif name == "want_service_prob":
            buf[name] = I32(bool(circ.want_service_prob))
        else:
            buf[name] = getattr(circ, name)
    return buf


def unpack_circuit(buf, circ):
    """Write a (mutated) CIRCUIT_DTYPE buffer back into a Stage A Circuit.

    Array fields are filled in place, so anything holding a reference to a
    sub-state keeps seeing the same object.  ``nyear`` comes back as the
    canonical string for its flag, which is all the flag can distinguish.
    """
    for name, _dt, _pack, unpack in _BLOCKS:
        obj = getattr(circ, name)
        if unpack is None:
            _unpack_son_mdl(buf[name], obj)
        else:
            unpack(buf[name], obj)
    for name, kind in _SCALARS:
        if name == "nyear_is_2089":
            circ.nyear = " 2089" if int(buf[name]) else ""
        elif name == "want_service_prob":
            circ.want_service_prob = bool(int(buf[name]))
        elif kind is I32:
            setattr(circ, name, int(buf[name]))
        else:
            setattr(circ, name, F32(buf[name]))
    return circ


# ---------------------------------------------------------------------------
# the shared tables
# ---------------------------------------------------------------------------

def shared_iono_tables(ionos):
    """A list of REDMAP IonoStates -> the nine flat maps plus F2D(1,1,1)."""
    out = {}
    for name, shape in _IONO_TABLES:
        stack = [np.ascontiguousarray(getattr(s, name), F32).ravel()
                 for s in ionos]
        for a in stack:
            assert a.size == int(np.prod(shape)), (name, a.size, shape)
        out[name] = np.concatenate(stack) if stack else np.zeros(0, F32)
    out["f2d111"] = np.array([F32(s.f2d[0, 0, 0]) for s in ionos], F32)
    return out


def shared_antenna_tables(cants):
    """A list of CantennaStates -> the flat /cantenna/ + /pantenna/ tables.

    ``numants`` is the LAST-DIMENSION EXTENT gain.cuh indexes with, so it has
    to be one number for the whole launch: the maximum over the batch is
    used, and the padding entries carry IATS 0, which GAIN's DO-10 scan
    (``IATS(IAT) == ITR``, ITR being 1 or 2) never matches.  So a circuit with
    fewer antennas behaves exactly as Stage A's own shorter loop does.
    """
    numants = max(int(c.numants) for c in cants)
    na = numants + 1
    for c in cants:
        assert not np.any(np.asarray(c.iats)[int(c.numants) + 1:na]), (
            "padding /cantenna/ entries must carry IATS 0")
    array = np.stack([np.ascontiguousarray(c.array[:, :, :na], F32)
                      for c in cants]).ravel()
    aeff = np.stack([np.ascontiguousarray(c.aeff[:, :na], F32)
                     for c in cants]).ravel()
    iats = np.stack([np.asarray(c.iats[:na], I32) for c in cants]).ravel()
    xfqs = np.stack([np.asarray(c.xfqs[:na], F32) for c in cants]).ravel()
    xfqe = np.stack([np.asarray(c.xfqe[:na], F32) for c in cants]).ravel()
    offazim = np.stack([np.asarray(c.offazim[:na], F32)
                        for c in cants]).ravel()
    pwrdba = np.stack([np.asarray(c.pwrdba[:na], F32) for c in cants]).ravel()
    return dict(array=array, aeff=aeff, iats=iats, xfqs=xfqs, xfqe=xfqe,
                offazim=offazim, pwrdba=pwrdba, numants=numants)


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------

_KERNEL_SRC = r'''
#include "luffy.cuh"

extern "C" __global__ void luffy_batch(
        CircuitState* circ, const int* ipflag,
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
    status[i] = luffy(&circ[i], fam, dud, fakp, fakmap, hmym, fakabp, abmap,
                      sys, perr, f2d111, ant_array, ant_aeff, ant_iats,
                      ant_xfqs, ant_xfqe, ant_offazim, ant_pwrdba, numants,
                      iono_idx[i], ant_idx[i], ipflag[i]);
}

extern "C" __global__ void circuit_sizeof(int* out) {
    out[0] = (int)sizeof(CircuitState);
}
'''


def load_kernel():
    """Compile (once, cached) the LUFFY batch kernel module."""
    return load(_KERNEL_SRC)


def circuit_state_sizeof() -> int:
    """sizeof(CircuitState) as the device compiler computes it."""
    import cupy as cp
    out = cp.zeros(1, np.int32)
    load_kernel().get_function("circuit_sizeof")((1,), (1,), (out,))
    cp.cuda.runtime.deviceSynchronize()
    return int(cp.asnumpy(out)[0])


def launch_luffy(bufs, ipflags, iono, iono_idx, ant, ant_idx, block=32):
    """Run N packed circuit-hours, one thread each.

    bufs:      (N,) array of CIRCUIT_DTYPE — the ENTRY states, not mutated
    ipflags:   (N,) int
    iono/ant:  the dicts ``shared_iono_tables`` / ``shared_antenna_tables``
               return
    iono_idx / ant_idx: (N,) int, which shared table each thread uses

    Returns ``(out_bufs, status)``: the mutated states and the per-thread
    LUFFY status (0 = success, non-zero = the refusal a callee reported).
    """
    import cupy as cp

    bufs = np.ascontiguousarray(bufs)
    assert bufs.dtype == CIRCUIT_DTYPE, bufs.dtype
    n = bufs.shape[0]
    d_circ = cp.asarray(bufs.view(np.uint8))
    d = {k: cp.asarray(v) for k, v in iono.items()}
    a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
    d_ipflag = cp.asarray(np.ascontiguousarray(ipflags, I32))
    d_ii = cp.asarray(np.ascontiguousarray(iono_idx, I32))
    d_ai = cp.asarray(np.ascontiguousarray(ant_idx, I32))
    d_status = cp.full(n, -1, np.int32)

    fn = load_kernel().get_function("luffy_batch")
    grid = ((n + block - 1) // block,)
    fn(grid, (block,),
       (d_circ, d_ipflag, d_ii, d_ai,
        d["fam"], d["dud"], d["fakp"], d["fakmap"], d["hmym"], d["fakabp"],
        d["abmap"], d["sys"], d["perr"], d["f2d111"],
        a["array"], a["aeff"], a["iats"], a["xfqs"], a["xfqe"], a["offazim"],
        a["pwrdba"], np.int32(ant["numants"]), d_status, np.int32(n)))
    cp.cuda.runtime.deviceSynchronize()
    out = cp.asnumpy(d_circ).view(CIRCUIT_DTYPE)
    return out, cp.asnumpy(d_status)


def run_luffy(circuits, ipflags, block=32, write_back=True):
    """Pack, launch, unpack.  One thread per circuit-hour.

    circuits: a sequence of Stage A ``Circuit`` objects, each carrying its own
    ``iono`` and ``cant``; identical table objects are shared on the device
    (identity-keyed, so the replay's one-IonoState-per-deck costs one copy per
    deck rather than one per hour).

    Returns ``(circuits, status)``.  With ``write_back`` the mutated device
    state is written back into the given Circuit objects in place.
    """
    ipflags = np.asarray(ipflags, I32)
    assert len(ipflags) == len(circuits)

    ionos, iono_idx, seen_i = [], [], {}
    cants, ant_idx, seen_a = [], [], {}
    for c in circuits:
        ki = id(c.iono)
        if ki not in seen_i:
            seen_i[ki] = len(ionos)
            ionos.append(c.iono)
        iono_idx.append(seen_i[ki])
        ka = id(c.cant)
        if ka not in seen_a:
            seen_a[ka] = len(cants)
            cants.append(c.cant)
        ant_idx.append(seen_a[ka])

    bufs = np.stack([pack_circuit(c) for c in circuits])
    out, status = launch_luffy(bufs, ipflags, shared_iono_tables(ionos),
                               np.array(iono_idx, I32),
                               shared_antenna_tables(cants),
                               np.array(ant_idx, I32), block=block)
    if write_back:
        for c, b in zip(circuits, out):
            unpack_circuit(b, c)
    return circuits, status
