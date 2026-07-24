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
dtypes in ``cuda/state.py`` — the same ones ``tests/cuda/test_state.py``
already pins field-for-field against ``state.cuh`` — followed by the loose
scalars in Stage A ``Circuit`` order.  Every member is 4 bytes, so numpy's
unaligned packing equals the C layout.  ``circuit_state_sizeof()`` compiles a
one-line kernel that writes ``sizeof(CircuitState)``; the test asserts it
equals ``CIRCUIT_DTYPE.itemsize``, which is the same offset proof the state
ABI already carries.

``iono`` and ``cant`` are NOT in the dtype.  They are big, read-only and
shared between threads, so they move once as flat device arrays and each
thread gets an index into them.

/RAYS/ IS NOT IN THE DTYPE EITHER, and for a completely different reason
--------------------------------------------------------------------------
``RaysState`` is 30,672 of the old 55,652 bytes per thread.  Carried inside
CircuitState it made adjacent lanes of a warp 55,652 bytes apart, so every
warp-wide read of one float cost 32 memory transactions -- measured at that
stride, 34 GB/s of useful bandwidth against 2,892 GB/s for the same data laid
out structure-of-arrays.  So /RAYS/ now lives in its OWN device buffers,
transposed to SoA: element ``o`` of thread ``t`` is at ``o * n + t``.  The
device sees it through ``cuda/raysview.cuh``'s ``RaysView``; the host still
packs and unpacks an ordinary ``(N,) RAYS_DTYPE`` array, and this module does
the transpose on the device (a couple of milliseconds at 32k threads) so the
callers' buffers keep their obvious per-circuit shape.

``pack_circuit`` therefore returns a PAIR -- ``(circuit_buffer, rays_buffer)``
-- and ``unpack_circuit`` takes the same pair back.  Together they are still
the whole Stage A ``Circuit``, field for field.

/SON/ MDL
---------
``pack_son``/``unpack_son`` encode ``mdl``/``mode``/``moder`` as LAYTYP
indices.  That is right for MODE and MODER, but MDL is not a layer label: it
is LUFFY's one-character model tag, "" / " " / "S" / "L" / "M".  Rather than
duplicate ``state.py``'s logic this module calls the same packers with MDL
blanked and then overwrites that one column with ``MDL_CODE`` — which is what
``cuda/circuit.cuh``'s ``VC_MDL_*`` constants hold.

MOST OF THE 55,652 BYTES NEVER NEEDED TO CROSS THE BUS
------------------------------------------------------
Measured at 32,256 circuit-hours, the round trip was ~618 ms of which the
kernel is ~127 ms; the other ~490 ms was 1.8 GB of per-thread state going up
and 1.8 GB coming back.  Almost none of it is INPUT.  ``UPLOAD_FIELDS`` below
is the set that is, established by experiment rather than by reading the
source: zero one field in all 504 replay circuit-hours, launch, and diff the
WHOLE output against the clean run.  A field whose zeroing changes nothing
else is not read.  ``tests/cuda/test_input_set.py`` pins the result, and
``tools/bench_stage_c.py --trip`` measures what it buys.

The three buckets that came out of that sweep, for the whole CircuitState and
the whole of /RAYS/:

* **INPUT** — read before it is written.  ``INPUT_FIELDS``.
* **ECHO** — never read, but not fully overwritten either, so its exit value
  still depends on its entry value (``reflx``, ``modes``, ``zon``, ``allm``,
  most of ``son``, ``losx``, the ``layers`` profile arrays, and /RAYS/ IFOB).
  Scratch as far as the kernel is concerned.
* **DEAD** — never read AND fully rewritten (``anois``, ``ndiag``, ``sigd``,
  ``ton``, ``frq.jmode`` ...).

**/RAYS/ IS NOT AN INPUT except for its first 168 bytes.**  NANG and ANG are
read before FOBBY rewrites them; IFOB — 30,504 of the 30,672 bytes, 55% of
the whole per-thread state — is pure scratch.

THREE THINGS THE EXPERIMENT CANNOT SEE, AND WHY THEY ARE UPLOADED ANYWAY
-----------------------------------------------------------------------
``layers.fs``, ``layers.ielect`` and ``muf.afmuf`` are zero in every one of
the 21 replay decks, so zeroing them is a no-op and the sweep calls them
dead.  They are read (esmod.cuh/regmod.cuh, lecden.cuh, inmuf.cuh/settxr.cuh)
and NOTHING in the port ever writes them, so for a caller who sets them they
are inputs.  132 bytes; they are carried.  The same argument covers the four
``muf`` MUF ladders and ``time.it`` that ``frqcom`` reads on the LUF arms
(IPFLAG 300/400), which the replay does not exercise at all.

``CARRY_FIELDS`` is a different thing again: fields the kernel does NOT read,
but which appear in ``RESULT_FIELDS`` and are only partly written, so the
answer the host reads back still contains their entry values.  /SON/ is the
whole result table and LUFFY skips every frequency slot whose FREL is zero,
leaving the previous hour's numbers in place; /MUFS/ XLUF is untouched
altogether on the system-performance arms, where SETOUT's ``xluf[:] = -1``
preset has to survive.  Carrying them is not about fidelity inside the
kernel; it is about the result table being complete.  Note that a ZERO probe
misses this entirely -- the skipped /SON/ columns happen to hold zeros in the
replay decks -- which is why the pinning test poisons with NaN.

TWO UPLOAD MODES, TWO FETCH PATHS
---------------------------------
``stage_luffy(..., upload="full")`` is the old behaviour and stays the
default: every byte up, so ``fetch_luffy`` can return every byte back and the
replay can compare all 7,012,152 cells.  ``upload="inputs"`` sends 3,228
bytes per thread and ``cp.zeros`` the rest on the device (a memset, not a
copy).  ``fetch_results`` brings back 1,700 bytes -- /SON/, XLUF, JMODE and
the status -- which is what a production caller actually wants.  The two
modes agree bit-for-bit on those results across all 504 replay circuit-hours;
that is the test that makes the reduction safe.

``LuffyArena`` is the third piece: the device buffers and the shared tables
allocated once and reused, so a caller sweeping batches pays neither the
allocation nor the 1.8 GB memset per batch.  It never re-zeroes between
submissions, which is the same claim as the input set made once more --
nothing outside ``UPLOAD_FIELDS`` is read, so the previous batch's leftovers
are as good as zeros.  Measured at 32,256 circuit-hours the two ends together
go from 515 ms to 98 ms, **5.2x**; the round trip from 901 to 472 ms.
"""

import numpy as np

from voacap_gpu.cuda.compile import load
from voacap_gpu.cuda.state import (ALLMODE_DTYPE, ANOIS_DTYPE, FRQ_DTYPE,
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
# `rays` is deliberately absent -- see the module docstring; it travels in its
# own SoA buffers, packed by `pack_rays` / `unpack_rays` straight from here.
_BLOCKS = (
    ("layers", LAYER_DTYPE, pack_layer, unpack_layer),
    ("muf", MUF_DTYPE, pack_muf, unpack_muf),
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

# --- the SoA /RAYS/ buffers ------------------------------------------------
# RAYS_DTYPE is all-4-byte, so one packed RaysState is exactly _RAYS_WORDS
# int32 words and the AoS -> SoA move is a plain (N, W) -> (W, N) transpose of
# the raw words.  `_RAYS_OFF` is each field's word offset inside a row.
_RAYS_WORDS = RAYS_DTYPE.itemsize // 4
_RAYS_OFF = {n: RAYS_DTYPE.fields[n][1] // 4 for n in RAYS_DTYPE.names}
_RAYS_LEN = {n: int(np.prod(RAYS_DTYPE.fields[n][0].shape or (1,)))
             for n in RAYS_DTYPE.names}

# Threads per staging chunk when moving /RAYS/ across the bus.  The transpose
# needs a second copy of whatever it holds, so this bounds the peak overhead
# at ~512 MB rather than a second copy of the whole batch (4 GB at 129k
# threads, which will not fit alongside everything else on a 16 GB card).
_RAYS_CHUNK = max(1, 512_000_000 // RAYS_DTYPE.itemsize)

# One thread's total state, for callers budgeting device memory.
STATE_BYTES_PER_THREAD = CIRCUIT_DTYPE.itemsize + RAYS_DTYPE.itemsize


# ---------------------------------------------------------------------------
# THE INPUT SET — measured, not guessed.  See the module docstring.
# ---------------------------------------------------------------------------

# Read before written.  Every one of these was established by zeroing it in
# all 504 replay circuit-hours and watching the rest of the output move,
# except where marked.
INPUT_FIELDS = (
    # --- /LAYER/ -----------------------------------------------------------
    "layers.km",                # the sample-area count; zeroing it hangs the
    "layers.kfx",               # area-selection loop, which is its own proof
    "layers.iedp",
    "layers.glat", "layers.gyz",
    "layers.fi", "layers.yi", "layers.hi",
    "layers.fs",                # STATIC: read by esmod/regmod, never written
    "layers.ielect",            # STATIC: read by lecden, never written
    "layers.clck", "layers.sigpat", "layers.epspat",
    # --- /MUFS/ ------------------------------------------------------------
    # The four ladders FRQCOM reads at TIME.IT are STATIC additions: they are
    # only read on the LUF arms (IPFLAG 300/400) and the replay runs 100/200.
    "muf.emuf", "muf.f2muf", "muf.allmuf", "muf.hpf",
    "muf.modmuf", "muf.sigl", "muf.sigu",
    "muf.delmuf", "muf.hpmuf", "muf.htmuf", "muf.fvmuf",
    "muf.afmuf",                # STATIC: read by inmuf/settxr, never written
    "muf.nhopmf", "muf.yfot", "muf.yhpf", "muf.ymuf",
    # --- /FRQ/, /TIME/, /LPATH/ --------------------------------------------
    "frq.frel",                 # THE frequency list
    "time.it",                  # STATIC: FRQCOM's index, LUF arms only
    "time.gmtr",
    "lpath.delopt", "lpath.gmin", "lpath.ymin",
)

# ... plus every loose scalar.  Fourteen of the eighteen are measured inputs;
# JLONG is measured dead and TLONG / NYEAR_IS_2089 / INFO are zero in every
# replay deck so the sweep cannot speak for them.  All eighteen are 72 bytes
# together, so the whole block is carried rather than argued about.
INPUT_FIELDS += tuple(n for n, _ in _SCALARS)

# Not read by the kernel, but only PARTLY written, and inside RESULT_FIELDS --
# so the result table the host reads back still carries their entry values.
# This is the Fortran COMMON persistence the replay depends on, and it is the
# reason /SON/ travels UP as well as down: a deck gives nine frequencies, so
# FREL(10), FREL(11) and FREL(13) are zero, LUFFY skips those columns, and
# the previous hour's numbers have to still be sitting in them afterwards.
# /MUFS/ XLUF is the same story one field at a time: nothing on the
# system-performance arms writes it, and SETOUT's ``xluf[:] = -1`` preset has
# to survive.  (Zeroing /SON/ looks harmless on the replay decks -- the
# skipped columns happen to hold zeros -- which is exactly why the pinning
# test poisons with NaN and not with zero.)
CARRY_FIELDS = ("muf.xluf", "son")

UPLOAD_FIELDS = INPUT_FIELDS + CARRY_FIELDS

# /RAYS/: NANG and ANG are read before FOBBY rewrites them.  IFOB -- 30,504
# bytes, 55% of the whole per-thread state -- is not.
RAYS_UPLOAD_FIELDS = ("nang", "ang")

# What a production caller wants back: the per-frequency results table, the
# LUF, and which sample area controlled the path.  (The status is separate --
# it is its own device array.)
RESULT_FIELDS = ("son", "muf.xluf", "frq.jmode")


def leaves_of(dtype, prefix=""):
    """Every scalar-or-subarray leaf path of a (nested) structured dtype."""
    out = []
    for name in dtype.names:
        sub = dtype.fields[name][0]
        if sub.names:
            out.extend(leaves_of(sub, f"{prefix}{name}."))
        else:
            out.append(prefix + name)
    return tuple(out)


def expand_fields(dtype, paths):
    """Field paths -> the leaf paths they cover (some entries are structs)."""
    all_leaves = leaves_of(dtype)
    out = []
    for p in paths:
        hit = [l for l in all_leaves if l == p or l.startswith(p + ".")]
        assert hit, f"{p} is not a field of the dtype"
        out.extend(hit)
    assert len(set(out)) == len(out), "a field is covered twice"
    return tuple(out)


def _span(dtype, path):
    """(word offset, word count) of a (possibly nested) field inside dtype."""
    off, dt = 0, dtype
    for p in path.split("."):
        sub, o = dt.fields[p][0], dt.fields[p][1]
        off += o
        dt = sub
    assert off % 4 == 0 and dt.itemsize % 4 == 0, path
    return off // 4, dt.itemsize // 4


def _merge(spans):
    """Sorted (offset, count) spans -> the same words as maximal runs."""
    out = []
    for s, ln in sorted(spans):
        if out and out[-1][0] + out[-1][1] == s:
            out[-1][1] += ln
        else:
            assert not out or s > out[-1][0] + out[-1][1], f"overlap at {s}"
            out.append([s, ln])
    return tuple((s, ln) for s, ln in out)


def _columns(runs, dtype, paths):
    """path -> (column in the compacted (N, m) buffer, word count)."""
    base, j = {}, 0
    for s, ln in runs:
        base[s] = j
        j += ln
    out = {}
    for p in paths:
        s, ln = _span(dtype, p)
        for rs, rln in runs:
            if rs <= s < rs + rln:
                out[p] = (base[rs] + (s - rs), ln)
                break
        else:                                        # pragma: no cover
            raise AssertionError(p)
    return out


_CIRC_WORDS = CIRCUIT_DTYPE.itemsize // 4
# The leaf-level expansion of the two declarations, for callers that need to
# ask "is THIS field sent?" (tests/cuda/test_input_set.py does).
UPLOAD_LEAVES = expand_fields(CIRCUIT_DTYPE, UPLOAD_FIELDS)
RAYS_UPLOAD_LEAVES = expand_fields(RAYS_DTYPE, RAYS_UPLOAD_FIELDS)
_UPLOAD_RUNS = _merge(_span(CIRCUIT_DTYPE, p) for p in UPLOAD_FIELDS)
_UPLOAD_IDX = np.concatenate([np.arange(s, s + ln, dtype=np.int64)
                              for s, ln in _UPLOAD_RUNS])
_RAYS_UPLOAD_RUNS = _merge(_span(RAYS_DTYPE, p) for p in RAYS_UPLOAD_FIELDS)
_RESULT_RUNS = _merge(_span(CIRCUIT_DTYPE, p) for p in RESULT_FIELDS)
_RESULT_COLS = _columns(_RESULT_RUNS, CIRCUIT_DTYPE, RESULT_FIELDS)
_RESULT_WORDS = sum(ln for _, ln in _RESULT_RUNS)

# What each direction costs per circuit-hour, for callers budgeting PCIe.
UPLOAD_BYTES_PER_THREAD = 4 * (len(_UPLOAD_IDX)
                               + sum(ln for _, ln in _RAYS_UPLOAD_RUNS))
RESULT_BYTES_PER_THREAD = 4 * (_RESULT_WORDS + 1)       # + the status word

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

def pack_circuit(circ):
    """One Stage A Circuit -> ``(CIRCUIT_DTYPE, RAYS_DTYPE)`` shape-() arrays.

    `iono` and `cant` are deliberately absent: they are shared tables, moved
    once by ``shared_iono_tables`` / ``shared_antenna_tables``.  /RAYS/ is
    absent from the first buffer and IS the second: see the module docstring.
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
    return buf, pack_rays(circ.rays)


def unpack_circuit(buf, rays_buf, circ):
    """Write a (mutated) state pair back into a Stage A Circuit.

    Array fields are filled in place, so anything holding a reference to a
    sub-state keeps seeing the same object.  ``nyear`` comes back as the
    canonical string for its flag, which is all the flag can distinguish.
    """
    unpack_rays(rays_buf, circ.rays)
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
#include "luffy_warp.cuh"

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
    status[i] = luffy(&circ[i], rays_nang, rays_ang, rays_ifob, n, i,
                      fam, dud, fakp, fakmap, hmym, fakabp, abmap,
                      sys, perr, f2d111, ant_array, ant_aeff, ant_iats,
                      ant_xfqs, ant_xfqe, ant_offazim, ant_pwrdba, numants,
                      iono_idx[i], ant_idx[i], ipflag[i]);
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


def stage_luffy(bufs, rays_bufs, ipflags, iono, iono_idx, ant, ant_idx,
                block=32, warp=False, warps_per_block=4, upload="full",
                groups=1, fast_math=False):
    """Move one batch to the device; return the staged launch, not the result.

    Split out of ``launch_luffy`` so a benchmark can time the KERNEL without
    the bus.  That distinction turned out to matter a great deal: at 32k
    threads the whole ``launch_luffy`` takes ~600 ms of which the kernel is
    ~125 ms, so timing the round trip measures PCIe, not compute.

    ``upload`` picks how much of that state actually crosses the bus:

    * ``"full"`` (the default, and what every existing caller gets) sends all
      55,652 bytes per thread, so ``fetch_luffy`` can hand back every field
      and the replay can compare all of them.
    * ``"inputs"`` sends only ``UPLOAD_FIELDS`` + ``RAYS_UPLOAD_FIELDS`` --
      3,228 bytes per thread -- and zeroes the rest on the device.  The
      kernel computes the same answer; see the module docstring and
      ``tests/cuda/test_input_set.py``.  Fetch it with ``fetch_results``:
      ``fetch_luffy`` would return zeros for the scratch the host never sent.

    ``warp`` selects ``luffy_batch_warp`` (cuda/luffy_warp.cuh) — one WARP per
    circuit-hour, lane L on frequency L+1 — instead of the one-thread-per-
    circuit-hour ``luffy_batch``.  The two are bit-identical; see
    ``tests/cuda/test_luffy_warp.py``.

    Returns ``(run, state)`` — call ``run()`` to launch (as many times as you
    like; the kernel mutates the staged state in place), and pass ``state`` to
    ``fetch_luffy`` to read the answer back.
    """
    import cupy as cp

    assert upload in ("full", "inputs"), upload
    bufs = np.ascontiguousarray(bufs)
    rays_bufs = np.ascontiguousarray(rays_bufs)
    assert bufs.dtype == CIRCUIT_DTYPE, bufs.dtype
    assert rays_bufs.dtype == RAYS_DTYPE, rays_bufs.dtype
    n = bufs.shape[0]
    assert rays_bufs.shape[0] == n, (rays_bufs.shape, n)
    if upload == "full":
        d_circ = cp.asarray(bufs.view(np.uint8))
        d_soa = _rays_to_soa(rays_bufs, n)
    else:
        d_circ = _circuit_inputs_to_device(bufs, n)
        d_soa = _rays_inputs_to_soa(rays_bufs, n)
    d_nang, d_ang, d_ifob = _rays_views(d_soa)
    d = {k: cp.asarray(v) for k, v in iono.items()}
    a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
    d_ipflag = cp.asarray(np.ascontiguousarray(ipflags, I32))
    d_ii = cp.asarray(np.ascontiguousarray(iono_idx, I32))
    d_ai = cp.asarray(np.ascontiguousarray(ant_idx, I32))
    d_status = cp.full(n, -1, np.int32)

    args = (d_circ, d_nang, d_ang, d_ifob, d_ipflag, d_ii, d_ai,
            d["fam"], d["dud"], d["fakp"], d["fakmap"], d["hmym"],
            d["fakabp"], d["abmap"], d["sys"], d["perr"], d["f2d111"],
            a["array"], a["aeff"], a["iats"], a["xfqs"], a["xfqe"],
            a["offazim"], a["pwrdba"], np.int32(ant["numants"]), d_status,
            np.int32(n))

    if warp:
        fn = load_kernel(groups, fast_math).get_function("luffy_batch_warp")
        wpb = int(warps_per_block)
        # one warp carries `groups` circuit-hours, so a block carries
        # wpb * groups of them and needs that many SmoothStore rows
        per_block = wpb * int(groups)
        grid = ((n + per_block - 1) // per_block,)
        blk = (32 * wpb,)
        shared = per_block * SMOOTHSTORE_BYTES

        def run():
            fn(grid, blk, args, shared_mem=shared)
    else:
        fn = load_kernel(fast_math=fast_math).get_function("luffy_batch")
        grid = ((n + block - 1) // block,)

        def run():
            fn(grid, (block,), args)

    return run, (d_circ, d_soa, d_status, n)


def fetch_luffy(state):
    """The staged device state -> ``(out_bufs, out_rays, status)``.

    THE WHOLE 55,652 bytes a thread, both structs.  Right for the replay,
    which compares every cell; wrong for production, which wants
    ``fetch_results``.  After an ``upload="inputs"`` launch the fields the
    host never sent come back as the zeros the device was memset to, not as
    the entry values -- that is not a bug, it is the point.
    """
    import cupy as cp

    d_circ, d_soa, d_status, n = state
    cp.cuda.runtime.deviceSynchronize()
    out = cp.asnumpy(d_circ).view(CIRCUIT_DTYPE)
    out_rays = _rays_from_soa(d_soa, n)
    return out, out_rays, cp.asnumpy(d_status)


class LuffyResults:
    """What a production caller reads back: 1,700 bytes a circuit-hour.

    son:    (N,) SON_DTYPE — the per-frequency results table, /SON/ entire
    xluf:   (N, 25) float32 — /MUFS/ XLUF, the LUF by hour
    jmode:  (N,) int32 — /FRQ/ JMODE, the controlling sample area
    status: (N,) int32 — 0 success, non-zero the refusal a callee reported
    """

    __slots__ = ("son", "xluf", "jmode", "status")

    def __init__(self, son, xluf, jmode, status):
        self.son, self.xluf = son, xluf
        self.jmode, self.status = jmode, status

    def __iter__(self):
        return iter((self.son, self.xluf, self.jmode, self.status))


def _split_results(host, n):
    """One compacted ``(n, _RESULT_WORDS)`` int32 block -> (son, xluf, jmode).

    The result words are three disjoint runs concatenated column-wise; this is
    the decode both ``fetch_results`` and the pipelined runner share, so the
    word bookkeeping lives in exactly one place.
    """
    def col(path):
        c, ln = _RESULT_COLS[path]
        return np.ascontiguousarray(host[:, c:c + ln])

    son = col("son").view(SON_DTYPE).reshape(n)
    xluf = col("muf.xluf").view(F32)
    jmode = col("frq.jmode").reshape(n)
    return son, xluf, jmode


def fetch_results(state):
    """The staged device state -> ``LuffyResults``: /SON/, XLUF, JMODE, status.

    The result words are scattered through CircuitState, so they are gathered
    into one compacted ``(N, m)`` device block first and cross the bus as a
    SINGLE D2H -- 1,700 bytes a circuit-hour against ``fetch_luffy``'s 55,652.
    """
    import cupy as cp

    d_circ, _d_soa, d_status, n = state
    cp.cuda.runtime.deviceSynchronize()
    gath = cp.empty((n, _RESULT_WORDS), np.int32)
    tot = _RESULT_WORDS * n
    _xfer("gather_words")(((tot + 255) // 256,), (256,),
                          (d_circ.view(np.int32), gath,
                           _device_map("result", _RESULT_MAP),
                           I32(_RESULT_WORDS), I32(_CIRC_WORDS),
                           np.int64(tot)))
    host = cp.asnumpy(gath)                          # ONE transfer
    status = cp.asnumpy(d_status)
    son, xluf, jmode = _split_results(host, n)
    return LuffyResults(son, xluf, jmode, status)


def launch_luffy(bufs, rays_bufs, ipflags, iono, iono_idx, ant, ant_idx,
                 block=32, warp=False, warps_per_block=4, upload="full",
                 groups=1, fast_math=False):
    """Run N packed circuit-hours, one thread each.

    bufs:      (N,) array of CIRCUIT_DTYPE — the ENTRY states, not mutated
    rays_bufs: (N,) array of RAYS_DTYPE — /RAYS/, likewise not mutated; it
               goes to the device transposed to structure-of-arrays
    ipflags:   (N,) int
    iono/ant:  the dicts ``shared_iono_tables`` / ``shared_antenna_tables``
               return
    iono_idx / ant_idx: (N,) int, which shared table each thread uses

    Returns ``(out_bufs, out_rays, status)``: the mutated states and the
    per-thread LUFFY status (0 = success, non-zero = the refusal a callee
    reported).
    """
    run, state = stage_luffy(bufs, rays_bufs, ipflags, iono, iono_idx, ant,
                             ant_idx, block=block, warp=warp,
                             warps_per_block=warps_per_block, upload=upload,
                             groups=groups, fast_math=fast_math)
    run()
    return fetch_luffy(state)


def luffy_results(bufs, rays_bufs, ipflags, iono, iono_idx, ant, ant_idx,
                  block=32, warp=False, warps_per_block=4,
                  groups=1, fast_math=False):
    """The production round trip: inputs up, results back, nothing else.

    Same arguments as ``launch_luffy``, same numbers out -- but 3,228 bytes a
    circuit-hour go up instead of 55,652 and 1,700 come back instead of
    55,652.  Returns ``LuffyResults``.
    """
    run, state = stage_luffy(bufs, rays_bufs, ipflags, iono, iono_idx, ant,
                             ant_idx, block=block, warp=warp,
                             warps_per_block=warps_per_block,
                             upload="inputs", groups=groups,
                             fast_math=fast_math)
    run()
    return fetch_results(state)


class LuffyArena:
    """Device state for a FIXED batch size, allocated and zeroed once.

    ``luffy_results`` still pays two costs that have nothing to do with the
    answer: 1.8 GB of ``cudaMalloc`` + memset for the scratch every call (at
    32,256 circuit-hours), and a fresh copy of the shared REDMAP and antenna
    tables.  Neither depends on the batch.  An arena holds both across
    submissions, so a caller sweeping many batches of the same size pays them
    once.

    NOTHING IS RE-ZEROED BETWEEN SUBMISSIONS, and that is the interesting
    part.  Batch k+1 starts with batch k's scratch still in place.  That is
    safe for exactly the reason the whole reduction is safe -- no field
    outside ``UPLOAD_FIELDS`` is read before it is written -- and it is
    *safer* than uninitialised memory would be, because every leftover word
    is a value this kernel itself produced, so no integer is out of its own
    range.  ``tests/cuda/test_input_set.py`` runs the same batch through an
    arena twice and requires the second answer to equal the first.

    Usage::

        arena = LuffyArena(n, iono, ant)
        for batch in batches:
            res = arena.run(bufs, rays_bufs, ipflags, iono_idx, ant_idx)
    """

    def __init__(self, n, iono, ant, block=32, warp=False, warps_per_block=4):
        import cupy as cp

        self.n = int(n)
        self._d_circ = cp.zeros(self.n * CIRCUIT_DTYPE.itemsize, np.uint8)
        self._d_soa = cp.zeros((_RAYS_WORDS, self.n), np.int32)
        self._d_status = cp.full(self.n, -1, np.int32)
        self._d_ipflag = cp.zeros(self.n, np.int32)
        self._d_ii = cp.zeros(self.n, np.int32)
        self._d_ai = cp.zeros(self.n, np.int32)
        d = {k: cp.asarray(v) for k, v in iono.items()}
        a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
        self._tables = (d, a)                    # keep the copies alive
        d_nang, d_ang, d_ifob = _rays_views(self._d_soa)
        self._args = (self._d_circ, d_nang, d_ang, d_ifob, self._d_ipflag,
                      self._d_ii, self._d_ai,
                      d["fam"], d["dud"], d["fakp"], d["fakmap"], d["hmym"],
                      d["fakabp"], d["abmap"], d["sys"], d["perr"],
                      d["f2d111"], a["array"], a["aeff"], a["iats"],
                      a["xfqs"], a["xfqe"], a["offazim"], a["pwrdba"],
                      np.int32(ant["numants"]), self._d_status,
                      np.int32(self.n))
        if warp:
            wpb = int(warps_per_block)
            self._fn = load_kernel().get_function("luffy_batch_warp")
            self._grid = ((self.n + wpb - 1) // wpb,)
            self._block = (32 * wpb,)
            self._shared = wpb * SMOOTHSTORE_BYTES
        else:
            self._fn = load_kernel().get_function("luffy_batch")
            self._grid = ((self.n + block - 1) // block,)
            self._block = (block,)
            self._shared = 0

    @property
    def state(self):
        """The ``(d_circ, d_soa, d_status, n)`` tuple the fetchers take."""
        return self._d_circ, self._d_soa, self._d_status, self.n

    def upload(self, bufs, rays_bufs, ipflags, iono_idx, ant_idx):
        """One batch's INPUTS into the resident buffers.  No launch."""
        bufs = np.ascontiguousarray(bufs)
        rays_bufs = np.ascontiguousarray(rays_bufs)
        assert bufs.dtype == CIRCUIT_DTYPE and bufs.shape[0] == self.n
        assert rays_bufs.dtype == RAYS_DTYPE and rays_bufs.shape[0] == self.n
        _write_circuit_inputs(self._d_circ, bufs, self.n)
        _write_rays_inputs(self._d_soa, rays_bufs, self.n)
        self._d_ipflag.set(np.ascontiguousarray(ipflags, I32))
        self._d_ii.set(np.ascontiguousarray(iono_idx, I32))
        self._d_ai.set(np.ascontiguousarray(ant_idx, I32))
        self._d_status.fill(-1)
        return self

    def launch(self):
        """Run the kernel over whatever is currently in the buffers."""
        self._fn(self._grid, self._block, self._args, shared_mem=self._shared)
        return self

    def submit(self, bufs, rays_bufs, ipflags, iono_idx, ant_idx):
        """Upload one batch's inputs into the resident buffers and launch."""
        return self.upload(bufs, rays_bufs, ipflags, iono_idx,
                           ant_idx).launch()

    def results(self):
        return fetch_results(self.state)

    def run(self, bufs, rays_bufs, ipflags, iono_idx, ant_idx):
        return self.submit(bufs, rays_bufs, ipflags, iono_idx,
                           ant_idx).results()


class PipelinedLuffy:
    """Overlap host prep / H2D / kernel / D2H / unpack across CUDA streams.

    ``LuffyArena`` removed the per-batch allocation and memset, but every stage
    of a submission still runs SERIALLY: the host gather blocks, then the H2D
    blocks, then the kernel, then the D2H, then the unpack, and the GPU sits
    idle for everything that is not the kernel.  On a real multi-batch run that
    shows up as 30-50% utilisation -- the card saturates for the ~124 ms it is
    computing and waits for the host the rest of the time.

    This runner splits a large workload into fixed-size CHUNKS and runs the
    five stages of adjacent chunks CONCURRENTLY on ``nstreams`` CUDA streams,
    each with its own device buffers and its own PAGE-LOCKED host staging
    buffers.  While chunk N computes on the GPU, chunk N+1's host gather and
    H2D are already in flight on another stream and chunk N-1's D2H and unpack
    are finishing.  Ordering is by per-slot CUDA events -- there is no
    ``deviceSynchronize`` anywhere in the steady-state loop, which is what lets
    the stages overlap.

    THE STRIDE CONSTRAINT, and why every chunk is padded to ``chunk``.
    ``cuda/raysview.cuh`` addresses /RAYS/ element ``o`` of thread ``t`` at
    ``o * n + t`` where ``n`` is the launch's thread count.  So the SoA rays
    buffer's column stride IS ``n``; a ragged final chunk launched with a
    smaller ``n`` would read every rays row at the wrong offset.  Rather than
    re-stride per chunk, the runner allocates every buffer at width ``chunk``
    and pads the final chunk up to ``chunk`` by REPEATING real input rows
    (never zeros -- a zeroed ``layers.km`` hangs the area loop, see the input
    set).  The padded lanes compute a valid-but-ignored answer; only the first
    ``c`` rows of each chunk are unpacked.  This is exactly the ragged case the
    correctness test hammers.

    Bit-identity: each stream's device buffers are zeroed ONCE at construction
    and never re-zeroed, the same reuse ``LuffyArena`` relies on and the same
    input-set guarantee makes safe.  The pipelined result is bit-identical to
    the serial arena on every field; ``tests/cuda/test_pipeline.py`` pins
    it across several chunk sizes, stream counts and a ragged tail.

    Usage::

        pipe = PipelinedLuffy(chunk, nstreams, iono, ant)
        res = pipe.run(bufs, rays_bufs, ipflags, iono_idx, ant_idx)
    """

    _RAYS_IN = 42                     # nang(1) + ang(41): the only rays inputs

    def __init__(self, chunk, nstreams, iono, ant, block=32, pinned=True,
                 prep_workers=0, fast_math=False):
        import cupy as cp

        assert _RAYS_UPLOAD_RUNS == ((0, self._RAYS_IN),), _RAYS_UPLOAD_RUNS
        self.chunk = int(chunk)
        self.ns = int(nstreams)
        self.block = int(block)
        self.pinned = bool(pinned)
        self.fast_math = bool(fast_math)
        self._cp = cp
        self._rt = cp.cuda.runtime
        self._m = len(_UPLOAD_IDX)
        # fast_math is the opt-in native-f32 libm (cuda/f32math.cuh); it is NOT
        # bit-exact against Stage A, so a pipelined run inherits exactly the
        # deviation tests/cuda/test_fastmath.py pins and nothing more.
        self._fn = load_kernel(fast_math=fast_math).get_function("luffy_batch")
        self._grid = ((self.chunk + block - 1) // block,)
        self._scatter = _xfer("scatter_words")
        self._gather = _xfer("gather_words")
        self._upmap = _device_map("upload", _UPLOAD_MAP)
        self._resmap = _device_map("result", _RESULT_MAP)

        # shared, read-only tables — one copy for the whole run
        d = {k: cp.asarray(v) for k, v in iono.items()}
        a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
        self._tables = (d, a)
        self._numants = np.int32(ant["numants"])
        self._shared = (d["fam"], d["dud"], d["fakp"], d["fakmap"], d["hmym"],
                        d["fakabp"], d["abmap"], d["sys"], d["perr"],
                        d["f2d111"], a["array"], a["aeff"], a["iats"],
                        a["xfqs"], a["xfqe"], a["offazim"], a["pwrdba"])

        # a worker pool for the host gather, if asked for
        self._pool = None
        if prep_workers > 0:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=int(prep_workers))

        # per-slot device + pinned staging, zeroed once
        rw, rwn = _RAYS_WORDS, _RESULT_WORDS
        self.streams, self.events = [], []
        self.d_circ, self.d_soa, self.d_status = [], [], []
        self.d_src, self.d_gath = [], []
        self.d_ipf, self.d_ii, self.d_ai = [], [], []
        self.p_src, self.p_rays, self.p_gath, self.p_status = [], [], [], []
        self.p_ipf, self.p_ii, self.p_ai = [], [], []
        self._rays_views = []
        for _ in range(self.ns):
            self.streams.append(cp.cuda.Stream(non_blocking=True))
            self.events.append(cp.cuda.Event(disable_timing=True))
            dc = cp.zeros(self.chunk * CIRCUIT_DTYPE.itemsize, np.uint8)
            ds = cp.zeros((rw, self.chunk), np.int32)
            self.d_circ.append(dc)
            self.d_soa.append(ds)
            self.d_status.append(cp.full(self.chunk, -1, np.int32))
            self.d_src.append(cp.empty((self.chunk, self._m), np.int32))
            self.d_gath.append(cp.empty((self.chunk, rwn), np.int32))
            self.d_ipf.append(cp.empty(self.chunk, np.int32))
            self.d_ii.append(cp.empty(self.chunk, np.int32))
            self.d_ai.append(cp.empty(self.chunk, np.int32))
            self.p_src.append(self._host((self.chunk, self._m), np.int32))
            self.p_rays.append(self._host((self._RAYS_IN, self.chunk),
                                          np.int32))
            self.p_gath.append(self._host((self.chunk, rwn), np.int32))
            self.p_status.append(self._host(self.chunk, np.int32))
            self.p_ipf.append(self._host(self.chunk, np.int32))
            self.p_ii.append(self._host(self.chunk, np.int32))
            self.p_ai.append(self._host(self.chunk, np.int32))
            o, ln = _RAYS_OFF, _RAYS_LEN
            d_nang = ds[o["nang"]:o["nang"] + ln["nang"]]
            d_ang = ds[o["ang"]:o["ang"] + ln["ang"]].view(np.float32)
            d_ifob = ds[o["ifob"]:o["ifob"] + ln["ifob"]]
            self._rays_views.append((d_nang, d_ang, d_ifob))

    def _host(self, shape, dtype):
        """A staging array — page-locked when pinned, plain numpy otherwise.

        The plain-numpy path exists only to MEASURE what pinning buys: an
        un-pinned host buffer forces the H2D/D2H to go through a driver bounce
        buffer and cannot overlap with compute.
        """
        n = int(np.prod(shape))
        if not self.pinned:
            return np.empty(shape, dtype)
        mem = self._cp.cuda.alloc_pinned_memory(n * np.dtype(dtype).itemsize)
        return np.frombuffer(mem, dtype, n).reshape(shape)

    def _args(self, slot, n):
        dc = self.d_circ[slot]
        d_nang, d_ang, d_ifob = self._rays_views[slot]
        return (dc, d_nang, d_ang, d_ifob, self.d_ipf[slot], self.d_ii[slot],
                self.d_ai[slot]) + self._shared + (
                self._numants, self.d_status[slot], np.int32(n))

    def _prep(self, slot, words, words_rays, ipf, ii, ai, a, b):
        """Host gather of one chunk into that slot's page-locked buffers.

        Pure CPU numpy: the 19 contiguous input runs concatenated into the
        compacted ``(chunk, m)`` block (the concat form, not a 409-word
        fancy-index -- the fancy path costs 5x on the host), the 42 rays input
        words transposed, and the three index vectors.  The ragged tail is
        padded up to ``chunk`` with a repeat of the chunk's first real row.
        """
        c = b - a
        ps, pr = self.p_src[slot], self.p_rays[slot]
        np.concatenate([words[a:b, s:s + ln] for s, ln in _UPLOAD_RUNS],
                       axis=1, out=ps[:c])
        pr[:, :c] = words_rays[a:b, :self._RAYS_IN].T
        self.p_ipf[slot][:c] = ipf[a:b]
        self.p_ii[slot][:c] = ii[a:b]
        self.p_ai[slot][:c] = ai[a:b]
        if c < self.chunk:                       # pad the ragged tail
            ps[c:] = ps[0]
            pr[:, c:] = pr[:, :1]
            self.p_ipf[slot][c:] = self.p_ipf[slot][0]
            self.p_ii[slot][c:] = self.p_ii[slot][0]
            self.p_ai[slot][c:] = self.p_ai[slot][0]
        return c

    def _h2d(self, dst, src, stream):
        self._rt.memcpyAsync(dst.data.ptr, src.ctypes.data, src.nbytes,
                             self._rt.memcpyHostToDevice, stream.ptr)

    def _d2h(self, dst, src, stream):
        self._rt.memcpyAsync(dst.ctypes.data, src.data.ptr, dst.nbytes,
                             self._rt.memcpyDeviceToHost, stream.ptr)

    def _enqueue(self, slot):
        """All five device stages of one chunk, async, on this slot's stream."""
        st = self.streams[slot]
        chunk = self.chunk
        with st:
            # H2D: compacted inputs, the 42 rays words, the three index vectors
            self._h2d(self.d_src[slot], self.p_src[slot], st)
            self._h2d(self.d_soa[slot][:self._RAYS_IN], self.p_rays[slot], st)
            self._h2d(self.d_ipf[slot], self.p_ipf[slot], st)
            self._h2d(self.d_ii[slot], self.p_ii[slot], st)
            self._h2d(self.d_ai[slot], self.p_ai[slot], st)
            # scatter the compacted block into CircuitState
            tot = self._m * chunk
            self._scatter(((tot + 255) // 256,), (256,),
                          (self.d_circ[slot].view(np.int32), self.d_src[slot],
                           self._upmap, I32(self._m), I32(_CIRC_WORDS),
                           np.int64(tot)))
            self.d_status[slot].fill(-1)
            # the kernel, one thread per (padded) circuit-hour
            self._fn(self._grid, (self.block,), self._args(slot, chunk))
            # gather the result words and bring them + the status home
            rtot = _RESULT_WORDS * chunk
            self._gather(((rtot + 255) // 256,), (256,),
                         (self.d_circ[slot].view(np.int32), self.d_gath[slot],
                          self._resmap, I32(_RESULT_WORDS), I32(_CIRC_WORDS),
                          np.int64(rtot)))
            self._d2h(self.p_gath[slot], self.d_gath[slot], st)
            self._d2h(self.p_status[slot], self.d_status[slot], st)
        self.events[slot].record(st)

    def _unpack(self, slot, a, b, out):
        """That slot's finished results into the caller's output arrays."""
        c = b - a
        son_o, xluf_o, jmode_o, status_o = out
        son, xluf, jmode = _split_results(self.p_gath[slot][:c], c)
        son_o[a:b] = son
        xluf_o[a:b] = xluf
        jmode_o[a:b] = jmode
        status_o[a:b] = self.p_status[slot][:c]

    def run(self, bufs, rays_bufs, ipflags, iono_idx, ant_idx):
        """A whole workload, chunked and pipelined -> ``LuffyResults``.

        ``bufs`` / ``rays_bufs`` may be ANY length; they are split into
        ``chunk``-sized pieces and the pieces overlap across the streams.  The
        returned tables carry one row per input circuit-hour, in input order,
        bit-identical to the serial path.
        """
        bufs = np.ascontiguousarray(bufs)
        rays_bufs = np.ascontiguousarray(rays_bufs)
        assert bufs.dtype == CIRCUIT_DTYPE, bufs.dtype
        assert rays_bufs.dtype == RAYS_DTYPE, rays_bufs.dtype
        n = bufs.shape[0]
        assert rays_bufs.shape[0] == n
        words = bufs.view(np.int32).reshape(n, _CIRC_WORDS)
        words_rays = rays_bufs.view(np.int32).reshape(n, _RAYS_WORDS)
        ipf = np.ascontiguousarray(ipflags, I32)
        ii = np.ascontiguousarray(iono_idx, I32)
        ai = np.ascontiguousarray(ant_idx, I32)

        son = np.empty(n, SON_DTYPE)
        xluf = np.empty((n, _RESULT_COLS["muf.xluf"][1]), F32)
        jmode = np.empty(n, I32)
        status = np.empty(n, I32)
        out = (son, xluf, jmode, status)

        bounds = [(a, min(a + self.chunk, n))
                  for a in range(0, n, self.chunk)]
        inflight = {}                       # slot -> (a, b) awaiting unpack
        prep_future = {}                    # slot -> Future, if a pool is used
        for k, (a, b) in enumerate(bounds):
            slot = k % self.ns
            if slot in inflight:            # reuse: wait, then drain that slot
                pa, pb = inflight.pop(slot)
                self.events[slot].synchronize()
                self._unpack(slot, pa, pb, out)
            if self._pool is not None:
                fut = prep_future.pop(slot, None)
                if fut is not None:
                    fut.result()
                self._prep(slot, words, words_rays, ipf, ii, ai, a, b)
            else:
                self._prep(slot, words, words_rays, ipf, ii, ai, a, b)
            self._enqueue(slot)
            inflight[slot] = (a, b)
        for slot, (a, b) in inflight.items():
            self.events[slot].synchronize()
            self._unpack(slot, a, b, out)
        return LuffyResults(son, xluf, jmode, status)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def run_luffy_pipelined(bufs, rays_bufs, ipflags, iono, iono_idx, ant, ant_idx,
                        chunk=32256, nstreams=2, block=32, pinned=True,
                        fast_math=False):
    """Convenience wrapper: build a ``PipelinedLuffy``, run once, return it.

    Allocates per-stream device + pinned buffers for ``chunk`` circuit-hours,
    so for a real sweep of many workloads keep the ``PipelinedLuffy`` object
    and call ``.run`` repeatedly rather than paying the allocation each time.
    """
    pipe = PipelinedLuffy(chunk, nstreams, iono, ant, block=block,
                          pinned=pinned, fast_math=fast_math)
    return pipe.run(bufs, rays_bufs, ipflags, iono_idx, ant_idx)


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

    packed = [pack_circuit(c) for c in circuits]
    bufs = np.stack([p[0] for p in packed])
    rays_bufs = np.stack([p[1] for p in packed])
    out, out_rays, status = launch_luffy(
        bufs, rays_bufs, ipflags, shared_iono_tables(ionos),
        np.array(iono_idx, I32), shared_antenna_tables(cants),
        np.array(ant_idx, I32), block=block)
    if write_back:
        for c, b, r in zip(circuits, out, out_rays):
            unpack_circuit(b, r, c)
    return circuits, status
