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
    "layers.hs",                # THE SPORADIC-E BLIND SPOT -- see below
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

# THE SPORADIC-E BLIND SPOT, and why it took the device-setup path to find it.
#
# ``layers.hs`` is the Es virtual height.  ESIND writes it, but ONLY when FPROB
# PSC(4) > 0 -- and every one of the 21 replay decks is a VOACAP deck with
# FPROB (1,1,1,0), Es disabled.  So HS is zero in all 504 replay circuit-hours,
# zeroing it changes nothing, and the sweep filed it under DEAD.  It is not
# dead: esmod/esreg read it on the Es arms, which those decks never take.
#
# Consequence, and it was a live defect rather than a theoretical one: on
# ICEPAC (URSI + FPROB (1,1,1,0.70), the arm Prop-SL runs for its ICEPAC
# model) ``luffy_results`` and the full-upload ``launch_luffy`` DISAGREED --
# measured at 192 of 2,688 /SON/ SNDB cells on a 4-path, 24-hour January
# sweep, because the compact path memset HS to zero on the device and the Es
# modes then propagated at ground level.  Carrying it costs 24 bytes.
#
# The generalisation worth keeping: the input set is measured against the
# decks that exist, so a field that is only live on an arm no deck exercises
# reads as dead.  ``layers.fs``, ``layers.ielect``, ``muf.afmuf`` and the LUF
# ladders were already carried on exactly that argument, made by reading the
# source; HS is the case where the argument was not made and should have been.
# It surfaced because the device-setup path RUNS ESIND and therefore produced
# a non-zero HS where the host path had been sending zeros.

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
# DEVICE-SIDE SETUP — the raw-parameter ABI (cuda/kernels/setup.cuh)
# ---------------------------------------------------------------------------
# ``UPLOAD_FIELDS`` cut what crosses the bus to what LUFFY READS.  It did
# nothing about what it costs to BUILD, and that is where the time actually
# went: measured on the Prop-SL sidecar at a global grid, host prep was 97.6%
# of wall-clock against 0.9% for the kernel and PCIe together.  So the setup
# chain moved to the device too (setup.cuh), and a circuit-hour is now
# described by ``SETUP_PARAMS_DTYPE`` -- 32 BYTES, against 3,228.
#
# Same numpy-vs-C layout argument as CIRCUIT_DTYPE: every member is 4 bytes,
# so unaligned numpy packing equals the C struct, and ``setup_struct_sizeof``
# asserts it against the device compiler's own ``sizeof``.

SETUP_PARAMS_DTYPE = np.dtype([
    ("tlatd", F32), ("tlongd", F32),     # transmitter, degrees (S/W negative)
    ("rlatd", F32), ("rlongd", F32),     # receiver
    ("hour", I32),                       # 1-based UTC hour
    ("npsl", I32),                       # GEOM long-path flag
    ("iono", I32),                       # which ionosphere slot
    ("cfg", I32),                        # which SETUP_CONFIG slot
])

SETUP_CONFIG_DTYPE = np.dtype([
    ("pole_lat", F32), ("pole_lon", F32),
    ("amind", F32), ("rsn", F32), ("dmp", F32), ("pmp", F32),
    ("psc", F32, (5,)),
    ("lufp", I32), ("method", I32), ("noise", I32), ("mspec", I32),
    ("nyear_is_2089", I32), ("want_service_prob", I32), ("info", I32),
    ("itim", I32),
    ("frel", F32, (30,)),
])

SETUP_BYTES_PER_THREAD = SETUP_PARAMS_DTYPE.itemsize

# setup.cuh's device-side hour guard: a row whose hour is outside 1..24 is
# not built (IPFLAG comes back SETUP_IPFLAG_REFUSED) and LUFFY reports it as
# SETUP_STATUS_BAD_HOUR -- 3, the out-of-bounds-table-read code GAIN already
# uses, since Stage A raises IndexError on such an hour.  The host check in
# `_check_setup_params` refuses these rows before launch; the guard is the
# defence in depth for a caller that bypasses it.
SETUP_IPFLAG_REFUSED = -1
SETUP_STATUS_BAD_HOUR = 3


def _require_dtype(arr, dtype, what):
    if arr.dtype != dtype:
        raise TypeError(f"{what} must have dtype {dtype}, got {arr.dtype}")


def _require_rows(arr, n, what):
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{what} must have shape ({n},), got {arr.shape}")


def _offending(mask, limit=8):
    """'rows [3, 17]' (capped) for a boolean mask with at least one hit."""
    rows = np.flatnonzero(mask)
    head = ", ".join(str(int(r)) for r in rows[:limit])
    more = f", ... ({rows.size} rows)" if rows.size > limit else ""
    return f"rows [{head}{more}]"


def _n_antenna_sets(ant):
    """How many /cantenna/ sets a ``shared_antenna_tables`` dict holds."""
    return int(np.size(ant["iats"])) // (int(ant["numants"]) + 1)


def _check_setup_params(params, ant_idx, n_iono, n_cfg, n_ant):
    """Refuse a SETUP_PARAMS_DTYPE batch the device would mis-read.

    Every index here becomes a raw pointer offset on the device, and an
    out-of-range hour reads past the 25-row VIRTIM table and comes back as
    status 0 with garbage, so these are checked BEFORE launch, with a
    ValueError naming the field and the offending rows.  Vectorised: a few
    comparisons per field, noise next to the ~400k rows/s the path runs at.
    Coordinates use ``~(abs(x) <= lim)``, which is also True for NaN, so
    "finite and in range" is one comparison.
    """
    def refuse(field, mask, rule):
        if mask.any():
            raise ValueError(
                f"SetupParams.{field} out of range ({rule}) at "
                f"{_offending(mask)}")

    hour = params["hour"]
    refuse("hour", (hour < 1) | (hour > 24), "integer hour 1..24")
    for f in ("tlatd", "rlatd"):
        refuse(f, ~(np.abs(params[f]) <= 90.0), "finite, -90..90 degrees")
    # Longitude is circular: refuse only non-finite values, and WRAP an
    # out-of-range one into [-180, 180] (validate.wrap_lon, the PropServer
    # normaliseLon rule) on a COPY, so the caller's array is never mutated
    # and an all-in-range batch is passed through untouched (bit-identical).
    wrapped = None
    for f in ("tlongd", "rlongd"):
        refuse(f, ~np.isfinite(params[f]), "finite degrees")
        out = ~(np.abs(params[f]) <= 180.0)
        if out.any():
            from voacap_gpu.validate import wrap_lon
            if wrapped is None:
                wrapped = params.copy()
            col = wrapped[f]
            col[out] = wrap_lon(col[out].astype(np.float64)).astype(col.dtype)
    if wrapped is not None:
        params = wrapped
    npsl = params["npsl"]
    refuse("npsl", (npsl != 0) & (npsl != 1), "0 or 1")
    iono = params["iono"]
    refuse("iono", (iono < 0) | (iono >= n_iono),
           f"0 <= iono < {n_iono} ionosphere slots")
    cfg = params["cfg"]
    refuse("cfg", (cfg < 0) | (cfg >= n_cfg),
           f"0 <= cfg < {n_cfg} config slots supplied")
    bad = (ant_idx < 0) | (ant_idx >= n_ant)
    if bad.any():
        raise ValueError(
            f"ant_idx out of range (0 <= ant_idx < {n_ant} antenna sets) at "
            f"{_offending(bad)}")
    return params


def _check_luffy_inputs(bufs, rays_bufs, ipflags, iono_idx, ant_idx, n=None):
    """dtype/shape checks for a packed LUFFY batch; returns its length."""
    _require_dtype(bufs, CIRCUIT_DTYPE, "bufs")
    _require_dtype(rays_bufs, RAYS_DTYPE, "rays_bufs")
    if n is None:
        n = bufs.shape[0] if bufs.ndim == 1 else -1
    _require_rows(bufs, n, "bufs")
    _require_rows(rays_bufs, n, "rays_bufs")
    for v, what in ((ipflags, "ipflags"), (iono_idx, "iono_idx"),
                    (ant_idx, "ant_idx")):
        _require_rows(np.asarray(v), n, what)
    return n


def _drain_default_stream():
    """Block until the device has finished everything queued so far.

    THE ORDERING CONTRACT every rig below keeps.  Each one allocates and
    fills its buffers (``cp.zeros`` / ``cp.full``) and uploads its tables
    (``cp.asarray``) on whatever stream is current -- CuPy's legacy default
    stream, normally -- and the pipelined rigs then do all their work on
    ``Stream(non_blocking=True)`` streams, which by definition do NOT wait for
    the default stream.  When the pool hands back memory freed by an earlier
    object, a work stream could read it before the fill or upload landed:
    reproduced as wrong rows with status 0, and pinned deterministically by
    ``tests/cuda/test_rig_race.py``.

    A device synchronise is the simplest ordering that is correct whatever
    stream the caller had current at construction and whatever stream a
    later submission uses (an event recorded on "the" default stream would
    have to guess which one the fills went to).  It runs only at the edges --
    end of construction, a table swap, the once-per-run config upload --
    never inside a chunk loop, so it costs the pipelines no overlap.
    """
    import cupy as cp
    cp.cuda.Device().synchronize()


def _quiesce(events, inflight):
    """Wait out every chunk still in flight -- the exception path of a run.

    A pipelined run normally waits on each slot's event before it returns.
    If it raises part-way (a bad ``on_chunk``, a KeyboardInterrupt), chunks
    may still be running on the work streams; letting the rig be freed under
    them would hand their buffers back to the pool while a kernel still
    writes into them -- the mirror image of the construction race.
    """
    for slot in inflight:
        events[slot].synchronize()

# The setup chain's own per-ionosphere tables.  The five REDMAP overlays it
# shares with LUFFY (fakp/fakmap/hmym/fakabp/abmap) come from
# ``shared_iono_tables`` and are NOT duplicated here.
_SETUP_TABLES = (("ikim", (10, 6), I32), ("anew", (3,), F32),
                 ("bnew", (3,), F32), ("achi", (2,), F32),
                 ("bchi", (2,), F32), ("f2d", (16, 6, 6), F32))

# VIRTIM rows precomputed per ionosphere, hours 0..24.  See setup.cuh: the row
# depends only on GMT, and on the ITIM >= 0 arm GMT is the requested hour, so
# 25 rows serve every circuit rather than 1,276 bytes travelling per thread.
SETUP_NHOUR = 25
SETUP_NAB = 319


class IonoSlot:
    """One ionosphere for the device-setup path: REDMAP output + its month/SSN.

    ``redmap`` does not keep the month and SSN it was called with, but the
    setup chain needs both (TIMVAR reads MONTH for the solar-declination
    table, CURMUF reads SSN), so they travel alongside.
    """

    __slots__ = ("iono", "month", "ssn")

    def __init__(self, iono, month, ssn):
        self.iono, self.month, self.ssn = iono, int(month), F32(ssn)


def setup_iono_tables(slots):
    """``IonoSlot`` list -> the flat tables only the device setup chain needs.

    Includes the precomputed VIRTIM rows: 25 hours x 319 floats per ionosphere
    (32 KB), which replaces one 1,276-byte row per circuit-hour.
    """
    from voacap_gpu.scalar.virtim import virtim

    out = {}
    for name, shape, dt in _SETUP_TABLES:
        stack = []
        for s in slots:
            a = np.ascontiguousarray(getattr(s.iono, name), dt).ravel()
            assert a.size == int(np.prod(shape)), (name, a.size, shape)
            stack.append(a)
        out[name] = np.concatenate(stack) if stack else np.zeros(0, dt)
    ab = []
    for s in slots:
        for h in range(SETUP_NHOUR):
            row = np.ascontiguousarray(virtim(s.iono, F32(h)), F32).ravel()
            assert row.size == SETUP_NAB, row.size
            ab.append(row)
    out["ab"] = np.concatenate(ab) if ab else np.zeros(0, F32)
    out["month"] = np.array([s.month for s in slots], I32)
    out["ssn"] = np.array([s.ssn for s in slots], F32)
    return out


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

// setup.cuh's hour guard hands LUFFY this IPFLAG for a circuit it refused to
// build; no legal IPFLAG is negative.  Such a row is answered with status
// VC_STATUS_TABLE_OOB and LUFFY never runs on its (unbuilt) state.  The
// values must match setup.cuh's; both are guarded so either may come first.
#ifndef VC_IPFLAG_REFUSED
#define VC_IPFLAG_REFUSED   (-1)
#endif
#ifndef VC_STATUS_TABLE_OOB
#define VC_STATUS_TABLE_OOB 3
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

    if upload not in ("full", "inputs"):
        raise ValueError(f"upload must be 'full' or 'inputs', got {upload!r}")
    bufs = np.ascontiguousarray(bufs)
    rays_bufs = np.ascontiguousarray(rays_bufs)
    n = _check_luffy_inputs(bufs, rays_bufs, ipflags, iono_idx, ant_idx)
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
        # the fills and table copies above went to the current stream; a
        # caller may submit from another one (see _drain_default_stream)
        _drain_default_stream()

    @property
    def state(self):
        """The ``(d_circ, d_soa, d_status, n)`` tuple the fetchers take."""
        return self._d_circ, self._d_soa, self._d_status, self.n

    def upload(self, bufs, rays_bufs, ipflags, iono_idx, ant_idx):
        """One batch's INPUTS into the resident buffers.  No launch."""
        bufs = np.ascontiguousarray(bufs)
        rays_bufs = np.ascontiguousarray(rays_bufs)
        _check_luffy_inputs(bufs, rays_bufs, ipflags, iono_idx, ant_idx,
                            n=self.n)
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
        # Every fill and table copy above went to the CURRENT stream, and the
        # work streams are non-blocking: without this, a first chunk on
        # recycled pool memory can run before the zero-fill lands.  See
        # _drain_default_stream.
        _drain_default_stream()

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
        n = _check_luffy_inputs(bufs, rays_bufs, ipflags, iono_idx, ant_idx)
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
        try:
            for k, (a, b) in enumerate(bounds):
                slot = k % self.ns
                if slot in inflight:        # reuse: wait, then drain that slot
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
            for slot in list(inflight):
                a, b = inflight[slot]
                self.events[slot].synchronize()
                self._unpack(slot, a, b, out)
                del inflight[slot]
        finally:
            _quiesce(self.events, inflight)
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
    if len(ipflags) != len(circuits):
        raise ValueError(f"{len(ipflags)} ipflags for {len(circuits)} "
                         "circuits")

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


class SetupLuffy:
    """Raw circuit parameters -> results, with the host doing NO per-circuit work.

    Two launches over one set of resident device buffers.  ``setup_batch``
    (cuda/kernels/setup.cuh) builds each CircuitState in place from 32 bytes of
    parameters and the shared tables; ``luffy_batch`` then runs on exactly
    those states without them ever crossing the bus.  What actually travels is

        up    32 bytes a circuit-hour, plus one SETUP_CONFIG per SYSTEM card
        down  1,700 bytes a circuit-hour (``LuffyResults``) + 12 for MUF/FOT/HPF

    against 3,228 up and 1,700 down for ``LuffyArena``, and -- the part that
    matters far more -- against a Python ``build_circuit`` per circuit-hour.

    THE STATE BUFFERS ARE ZEROED ONCE AND NEVER AGAIN, exactly as
    ``LuffyArena`` does, and for the same reason: nothing outside
    ``UPLOAD_FIELDS`` is read before it is written, and ``setup_batch`` writes
    every one of those fields.  Leftovers from the previous submission are
    values this pair of kernels itself produced.

    ``n`` is FIXED for the lifetime of the object, because /RAYS/'s SoA column
    stride is the launch's thread count (raysview.cuh).  Submit a shorter batch
    and the tail lanes simply compute an answer nobody reads -- but pad them
    with a REPEAT of a real row, never zeros: a zeroed ``layers.km`` hangs the
    area loop.  ``run`` does that padding itself.

    Usage::

        slots = [IonoSlot(redmap(dir, "CCIR", 7, 100.0), 7, 100.0)]
        rig = SetupLuffy(n, slots, shared_iono_tables([s.iono for s in slots]),
                         shared_antenna_tables(cants))
        res, muf3 = rig.run(params, cfgs, ant_idx)
    """

    def __init__(self, n, slots, iono, ant=None, block=32, warp=False,
                 warps_per_block=4, fast_math=False):
        import cupy as cp

        self.n = int(n)
        self.block = int(block)
        self._cp = cp
        self._setup_fn = load_setup_kernel(fast_math).get_function(
            "setup_batch")
        self._setup_grid = ((self.n + self.block - 1) // self.block,)

        # per-thread state, allocated and zeroed ONCE
        self._d_circ = cp.zeros(self.n * CIRCUIT_DTYPE.itemsize, np.uint8)
        self._d_soa = cp.zeros((_RAYS_WORDS, self.n), np.int32)
        self._d_status = cp.full(self.n, -1, np.int32)
        self._d_ipflag = cp.zeros(self.n, np.int32)
        self._d_muf3 = cp.zeros((self.n, 3), np.float32)
        self._d_prm = cp.zeros(self.n * SETUP_PARAMS_DTYPE.itemsize, np.uint8)
        self._d_ai = cp.zeros(self.n, np.int32)
        self._d_cfg = None                       # sized on first submission

        # shared read-only tables — one copy for the object's lifetime.  The
        # IONOSPHERE is resident because it is big (REDMAP's maps) and depends
        # only on (coeff, month, SSN); the ANTENNAS are not, because a type-13
        # pattern is AZIMUTH-DEPENDENT and a grid therefore builds one
        # /cantenna/ per cell.  So they are set per submission -- see
        # `set_antennas` -- with an optional table here for the fixed-antenna
        # caller who would rather pay the copy once.
        st = setup_iono_tables(slots)
        d = {k: cp.asarray(v) for k, v in iono.items()}
        s = {k: cp.asarray(v) for k, v in st.items()}
        self._tables = (d, s)                    # keep the copies alive
        self._setup_shared = (s["ab"], s["ikim"], s["anew"], s["bnew"],
                              s["achi"], s["bchi"], d["fakp"], d["fakmap"],
                              d["hmym"], d["fakabp"], d["abmap"], s["f2d"],
                              s["month"], s["ssn"])

        d_nang, d_ang, d_ifob = _rays_views(self._d_soa)
        # IONO_IDX for LUFFY is the SetupParams `iono` column, which the host
        # already has; it is uploaded per submission alongside the parameters.
        self._d_ii = cp.zeros(self.n, np.int32)
        self._luffy_head = (self._d_circ, d_nang, d_ang, d_ifob,
                            self._d_ipflag, self._d_ii, self._d_ai,
                            d["fam"], d["dud"], d["fakp"], d["fakmap"],
                            d["hmym"], d["fakabp"], d["abmap"], d["sys"],
                            d["perr"], d["f2d111"])
        # an iono index has to be valid for BOTH table sets it addresses
        self._n_iono = min(len(slots), int(np.size(iono["f2d111"])))
        self._n_ant = 0
        self._luffy_args = None
        if ant is not None:
            self._install_antennas(ant)
        if warp:
            wpb = int(warps_per_block)
            self._luffy_fn = load_kernel(fast_math=fast_math).get_function(
                "luffy_batch_warp")
            self._luffy_grid = ((self.n + wpb - 1) // wpb,)
            self._luffy_block = (32 * wpb,)
            self._luffy_shared = wpb * SMOOTHSTORE_BYTES
        else:
            self._luffy_fn = load_kernel(fast_math=fast_math).get_function(
                "luffy_batch")
            self._luffy_grid = ((self.n + self.block - 1) // self.block,)
            self._luffy_block = (self.block,)
            self._luffy_shared = 0
        # Submissions run on the current stream, as the fills above did, so
        # a caller who stays on one stream is ordered anyway; this makes the
        # rig safe to drive from ANY stream (see _drain_default_stream).
        _drain_default_stream()

    @property
    def state(self):
        """The ``(d_circ, d_soa, d_status, n)`` tuple the fetchers take."""
        return self._d_circ, self._d_soa, self._d_status, self.n

    def set_antennas(self, ant):
        """Install a ``shared_antenna_tables`` dict for the next submissions.

        Separate from the ionosphere because it changes on a different clock:
        the ionosphere is per (coefficient set, month, SSN) and a whole request
        shares one, while a type-13 pattern is azimuth-dependent and a grid
        builds one /cantenna/ per CELL.  ``numants`` is the last-dimension
        extent gain.cuh indexes with, so it is baked into the argument tuple
        and the tuple is rebuilt here rather than patched.
        """
        self._install_antennas(ant)
        _drain_default_stream()
        return self

    def _install_antennas(self, ant):
        """`set_antennas` without the drain: for ``upload(ant=...)``, whose
        copies go to the same stream as the launch that follows them."""
        cp = self._cp
        a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
        self._ant = a                            # keep the copies alive
        self._n_ant = _n_antenna_sets(ant)
        self._luffy_args = self._luffy_head + (
            a["array"], a["aeff"], a["iats"], a["xfqs"], a["xfqe"],
            a["offazim"], a["pwrdba"], np.int32(ant["numants"]),
            self._d_status, np.int32(self.n))

    def _validate(self, params, cfgs, ant_idx):
        """Input checks shared by `upload` and `run` (which pads after).

        Returns the params to use: the same array, or a copy with
        out-of-range longitudes wrapped (see ``_check_setup_params``)."""
        _require_dtype(params, SETUP_PARAMS_DTYPE, "params")
        _require_dtype(cfgs, SETUP_CONFIG_DTYPE, "cfgs")
        if params.ndim != 1:
            raise ValueError(f"params must be 1-D, got shape {params.shape}")
        _require_rows(ant_idx, params.shape[0], "ant_idx")
        if self._luffy_args is None:
            raise ValueError(
                "no antenna tables: pass ant= to the constructor, to "
                "upload/run, or call set_antennas")
        return _check_setup_params(params, ant_idx, self._n_iono,
                                   cfgs.shape[0], self._n_ant)

    def upload(self, params, cfgs, ant_idx, ant=None):
        """One batch's raw parameters into the resident buffers.  No launch."""
        if ant is not None:
            self._install_antennas(ant)
        params = np.ascontiguousarray(params)
        cfgs = np.ascontiguousarray(cfgs)
        ant_idx = np.ascontiguousarray(ant_idx, I32)
        params = self._validate(params, cfgs, ant_idx)
        if params.shape[0] != self.n:
            raise ValueError(f"upload takes exactly n={self.n} rows, got "
                             f"{params.shape[0]} (run() pads a short batch)")
        return self._upload(params, cfgs, ant_idx)

    def _upload(self, params, cfgs, ant_idx):
        cp = self._cp
        self._d_prm.set(params.view(np.uint8).ravel())
        self._d_cfg = cp.asarray(cfgs.view(np.uint8).ravel())
        self._d_ii.set(np.ascontiguousarray(params["iono"], I32))
        self._d_ai.set(np.ascontiguousarray(ant_idx, I32))
        self._d_status.fill(-1)
        return self

    def launch_setup(self):
        """Build the CircuitStates in place.  Split out so a test can read the
        built state back BEFORE LUFFY mutates it, and so a benchmark can time
        the two halves apart."""
        self._setup_fn(self._setup_grid, (self.block,), (
            self._d_circ, *_rays_views(self._d_soa)[:2], self._d_ipflag,
            self._d_muf3, self._d_prm, self._d_cfg) + self._setup_shared
            + (np.int32(self.n),))
        return self

    def launch_kernel(self):
        """Run LUFFY over whatever the setup kernel left in the buffers."""
        self._luffy_fn(self._luffy_grid, self._luffy_block, self._luffy_args,
                       shared_mem=self._luffy_shared)
        return self

    def launch(self):
        """setup_batch, then luffy_batch, over the resident buffers."""
        return self.launch_setup().launch_kernel()

    def results(self):
        """``(LuffyResults, muf3)`` — muf3 is (n, 3): ALLMUF, FOT, HPF at IT."""
        res = fetch_results(self.state)
        return res, self._cp.asnumpy(self._d_muf3)

    def run(self, params, cfgs, ant_idx, ant=None):
        """A batch of ANY length <= n -> ``(LuffyResults, muf3)``.

        Short batches are padded up to ``n`` by REPEATING the first row (never
        zeros; see the class docstring) and the padding is dropped on the way
        out, so a caller never sees the fixed width.
        """
        if ant is not None:
            self._install_antennas(ant)
        params = np.ascontiguousarray(params)
        cfgs = np.ascontiguousarray(cfgs)
        ant_idx = np.ascontiguousarray(ant_idx, I32)
        params = self._validate(params, cfgs, ant_idx)
        c = params.shape[0]
        if not 0 < c <= self.n:
            raise ValueError(f"run takes 1..n={self.n} rows, got {c}")
        if c < self.n:
            pad = self.n - c
            params = np.concatenate([params, np.repeat(params[:1], pad)])
            ant_idx = np.concatenate([ant_idx, np.repeat(ant_idx[:1], pad)])
        self._upload(params, cfgs, ant_idx).launch()
        res, muf3 = self.results()
        if c == self.n:
            return res, muf3
        return (LuffyResults(res.son[:c], res.xluf[:c], res.jmode[:c],
                             res.status[:c]), muf3[:c])


def setup_and_run(params, cfgs, slots, iono, ant, ant_idx, block=32,
                  warp=False, warps_per_block=4, fast_math=False):
    """One-shot device-setup round trip -> ``(LuffyResults, muf3)``.

    Allocates for exactly this batch, so for a sweep of several batches keep a
    ``SetupLuffy`` and call ``.run`` repeatedly instead.
    """
    rig = SetupLuffy(len(params), slots, iono, ant, block=block, warp=warp,
                     warps_per_block=warps_per_block, fast_math=fast_math)
    return rig.run(params, cfgs, ant_idx)


class PipelinedSetupLuffy:
    """``SetupLuffy`` across several CUDA streams, so the card never waits.

    ``SetupLuffy`` runs one batch at a time and synchronises to read it back,
    which means the GPU is idle for every host-side stage: the parameter fill,
    the result extraction, and -- in a service -- the JSON encoding of the
    previous answer.  Measured on a world grid that left the card busy about a
    quarter of the time.

    This runner splits a workload into fixed-width CHUNKS and gives each of
    ``nstreams`` slots its OWN device buffers, its own page-locked host staging
    and its own stream.  While chunk N's kernels run, chunk N+1's parameters
    are already crossing the bus and chunk N-1's results are being consumed on
    the host.  Ordering is by per-slot CUDA events; there is no
    ``deviceSynchronize`` in the steady-state loop, which is the whole point --
    a single sync anywhere serialises every stream behind it.

    THE STRIDE CONSTRAINT is the same one ``PipelinedLuffy`` documents:
    ``raysview.cuh`` addresses /RAYS/ element ``o`` of thread ``t`` at
    ``o * n + t``, so the SoA column stride IS the launch width.  Every slot is
    therefore allocated at ``chunk`` and a short final chunk is padded up by
    REPEATING a real parameter row -- never zeros, which would hang the
    sample-area loop.  Only the real rows are handed back.

    THE SHARED TABLES ARE UPLOADED ONCE for the whole run and shared by every
    slot: the ionosphere because it depends only on (coefficients, month, SSN),
    and the ANTENNAS because a job's /cantenna/ set is per-CELL, not per
    circuit -- a grid revisits the same cell at every hour and frequency, so
    one table serves the entire sweep instead of being re-sent per chunk.

    ``on_chunk`` is how a caller keeps up.  It is invoked as
    ``on_chunk(lo, hi, results, muf3)`` the moment a chunk lands, with the
    results for ``params[lo:hi]`` -- so a service can reduce, encode or stream
    that chunk while the next ones are still on the card.  Supply it and
    nothing is accumulated (a monthly-median world sweep is 400,000
    circuit-hours and there is no reason to hold all of it); omit it and
    ``run`` returns the whole ``(LuffyResults, muf3)`` as ``SetupLuffy`` does.

    Usage::

        pipe = PipelinedSetupLuffy(8192, 3, slots, iono)
        pipe.run(params, cfgs, ant, ant_idx, on_chunk=reduce_into_grid)
    """

    def __init__(self, chunk, nstreams, slots=None, iono=None, block=32,
                 pinned=True, fast_math=False):
        import cupy as cp

        self.chunk = int(chunk)
        self.ns = max(1, int(nstreams))
        self.block = int(block)
        self.pinned = bool(pinned)
        self._cp = cp
        self._rt = cp.cuda.runtime
        self._setup_fn = load_setup_kernel(fast_math).get_function(
            "setup_batch")
        self._luffy_fn = load_kernel(fast_math=fast_math).get_function(
            "luffy_batch")
        self._grid = ((self.chunk + self.block - 1) // self.block,)
        self._gather = _xfer("gather_words")
        self._resmap = _device_map("result", _RESULT_MAP)

        # The ionosphere and the antennas are BOTH per-run here, unlike
        # SetupLuffy where the ionosphere is resident.  A long-lived service
        # rig has to serve whatever month/SSN/coefficient set the next request
        # names, and the whole REDMAP + VIRTIM block is under 100 KB, so
        # re-uploading it per job costs nothing worth keeping state for.
        # `_iono_key` skips even that when consecutive jobs share one.
        self._iono_dev = None
        self._setup_shared = None
        self._iono_key = None
        self._ant = None
        self._ant_args = None
        self._n_iono = 0
        self._n_ant = 0
        if slots is not None and iono is not None:
            self._install_ionosphere(slots, iono)

        # per-slot device + pinned staging, zeroed once
        self.streams, self.events = [], []
        self.d_circ, self.d_soa, self.d_status = [], [], []
        self.d_ipflag, self.d_muf3, self.d_prm = [], [], []
        self.d_ai, self.d_cfg, self.d_gath = [], [], []
        self.d_ii = []
        self.p_prm, self.p_ai, self.p_gath = [], [], []
        self.p_status, self.p_muf3, self.p_ii = [], [], []
        self._rays = []
        pw = SETUP_PARAMS_DTYPE.itemsize // 4
        for _ in range(self.ns):
            self.streams.append(cp.cuda.Stream(non_blocking=True))
            self.events.append(cp.cuda.Event(disable_timing=True))
            self.d_circ.append(cp.zeros(self.chunk * CIRCUIT_DTYPE.itemsize,
                                        np.uint8))
            soa = cp.zeros((_RAYS_WORDS, self.chunk), np.int32)
            self.d_soa.append(soa)
            self.d_status.append(cp.full(self.chunk, -1, np.int32))
            self.d_ipflag.append(cp.zeros(self.chunk, np.int32))
            self.d_muf3.append(cp.zeros((self.chunk, 3), np.float32))
            self.d_prm.append(cp.zeros((self.chunk, pw), np.int32))
            self.d_ai.append(cp.zeros(self.chunk, np.int32))
            # LUFFY's iono_idx is the SetupParams `iono` column, but it has to
            # reach the kernel as a CONTIGUOUS int array -- a strided column
            # view of the parameter block is not one.
            self.d_ii.append(cp.zeros(self.chunk, np.int32))
            self.d_cfg.append(None)
            self.d_gath.append(cp.empty((self.chunk, _RESULT_WORDS), np.int32))
            self.p_prm.append(self._host((self.chunk, pw), np.int32))
            self.p_ai.append(self._host(self.chunk, np.int32))
            self.p_ii.append(self._host(self.chunk, np.int32))
            self.p_gath.append(self._host((self.chunk, _RESULT_WORDS),
                                          np.int32))
            self.p_status.append(self._host(self.chunk, np.int32))
            self.p_muf3.append(self._host((self.chunk, 3), np.float32))
            self._rays.append(_rays_views(soa))
        # Every fill and table copy above went to the CURRENT stream, and the
        # slot streams are non-blocking: without this, a first chunk on
        # recycled pool memory can run before the zero-fill lands (reproduced
        # as wrong rows with status 0).  See _drain_default_stream.
        _drain_default_stream()

    def _host(self, shape, dtype):
        """Page-locked staging, so the copies can actually overlap compute."""
        n = int(np.prod(shape))
        if not self.pinned:
            return np.empty(shape, dtype)
        mem = self._cp.cuda.alloc_pinned_memory(n * np.dtype(dtype).itemsize)
        return np.frombuffer(mem, dtype, n).reshape(shape)

    def set_ionosphere(self, slots, iono, key=None):
        """Install the REDMAP + VIRTIM tables for the next run.

        ``key`` lets a caller skip a redundant re-upload when the next job
        names the same (coefficients, month, SSN); pass anything hashable that
        identifies the table set, or None to always upload.
        """
        if key is not None and key == self._iono_key:
            return self
        self._install_ionosphere(slots, iono, key)
        _drain_default_stream()       # the copies are read on slot streams
        return self

    def _install_ionosphere(self, slots, iono, key=None):
        cp = self._cp
        st = setup_iono_tables(slots)
        d = {k: cp.asarray(v) for k, v in iono.items()}
        s = {k: cp.asarray(v) for k, v in st.items()}
        self._tables = (d, s)                    # keep the copies alive
        self._iono_dev = d
        self._setup_shared = (s["ab"], s["ikim"], s["anew"], s["bnew"],
                              s["achi"], s["bchi"], d["fakp"], d["fakmap"],
                              d["hmym"], d["fakabp"], d["abmap"], s["f2d"],
                              s["month"], s["ssn"])
        # an iono index has to be valid for BOTH table sets it addresses
        self._n_iono = min(len(slots), int(np.size(iono["f2d111"])))
        self._iono_key = key

    def set_antennas(self, ant):
        self._install_antennas(ant)
        _drain_default_stream()       # the copies are read on slot streams
        return self

    def _install_antennas(self, ant):
        cp = self._cp
        a = {k: cp.asarray(v) for k, v in ant.items() if k != "numants"}
        self._ant = a
        self._n_ant = _n_antenna_sets(ant)
        self._ant_args = (a["array"], a["aeff"], a["iats"], a["xfqs"],
                          a["xfqe"], a["offazim"], a["pwrdba"],
                          np.int32(ant["numants"]))

    def _h2d(self, dst, src, stream):
        self._rt.memcpyAsync(dst.data.ptr, src.ctypes.data, src.nbytes,
                             self._rt.memcpyHostToDevice, stream.ptr)

    def _d2h(self, dst, src, stream):
        self._rt.memcpyAsync(dst.ctypes.data, src.data.ptr, dst.nbytes,
                             self._rt.memcpyDeviceToHost, stream.ptr)

    def _enqueue(self, slot, c):
        """Every device stage of one chunk, async, on this slot's stream."""
        cp = self._cp
        st = self.streams[slot]
        d = self._iono_dev
        chunk = self.chunk
        d_nang, d_ang, d_ifob = self._rays[slot]
        with st:
            self._h2d(self.d_prm[slot], self.p_prm[slot], st)
            self._h2d(self.d_ai[slot], self.p_ai[slot], st)
            self._h2d(self.d_ii[slot], self.p_ii[slot], st)
            self.d_status[slot].fill(-1)
            self._setup_fn(self._grid, (self.block,), (
                self.d_circ[slot], d_nang, d_ang, self.d_ipflag[slot],
                self.d_muf3[slot], self.d_prm[slot], self.d_cfg[slot])
                + self._setup_shared + (np.int32(chunk),))
            self._luffy_fn(self._grid, (self.block,), (
                self.d_circ[slot], d_nang, d_ang, d_ifob,
                self.d_ipflag[slot], self.d_ii[slot],
                self.d_ai[slot],
                d["fam"], d["dud"], d["fakp"], d["fakmap"], d["hmym"],
                d["fakabp"], d["abmap"], d["sys"], d["perr"], d["f2d111"])
                + self._ant_args + (self.d_status[slot], np.int32(chunk)))
            rtot = _RESULT_WORDS * chunk
            self._gather(((rtot + 255) // 256,), (256,),
                         (self.d_circ[slot].view(np.int32), self.d_gath[slot],
                          self._resmap, I32(_RESULT_WORDS), I32(_CIRC_WORDS),
                          np.int64(rtot)))
            self._d2h(self.p_gath[slot], self.d_gath[slot], st)
            self._d2h(self.p_status[slot], self.d_status[slot], st)
            self._d2h(self.p_muf3[slot], self.d_muf3[slot], st)
        self.events[slot].record(st)

    def _stage(self, slot, params, iono_idx, ant_idx, a, b):
        """Fill this slot's pinned staging; pad a short chunk by repetition."""
        c = b - a
        pw = SETUP_PARAMS_DTYPE.itemsize // 4
        words = params.view(np.int32).reshape(-1, pw)
        self.p_prm[slot][:c] = words[a:b]
        self.p_ai[slot][:c] = ant_idx[a:b]
        self.p_ii[slot][:c] = iono_idx[a:b]
        if c < self.chunk:
            # REPEAT a real row: a zeroed layers.km hangs the area loop, so the
            # padding lanes have to describe a legal circuit even though their
            # answer is discarded.
            self.p_prm[slot][c:] = self.p_prm[slot][0]
            self.p_ai[slot][c:] = self.p_ai[slot][0]
            self.p_ii[slot][c:] = self.p_ii[slot][0]
        return c

    def _harvest(self, slot, a, b):
        """This slot's results OUT of the pinned staging, as owned arrays.

        The staging is reused by the next chunk the moment this returns, so
        nothing handed back may alias it.  The block is therefore COPIED here
        rather than leaning on ``_split_results``'s per-column
        ``ascontiguousarray``: those calls copy a strided slice of a wider
        block *usually*, but a slice of shape ``(1, k)`` is already
        C-contiguous whatever its row stride, so a single-row chunk got a VIEW
        and the next chunk overwrote the answer underneath it.  That is not
        hypothetical -- the assert below caught it on a one-circuit request.
        """
        c = b - a
        gath = np.array(self.p_gath[slot][:c])          # owned, not pinned
        son, xluf, jmode = _split_results(gath, c)
        for arr, what in ((son, "son"), (xluf, "xluf"), (jmode, "jmode")):
            assert not np.shares_memory(arr, self.p_gath[slot]), (
                f"{what} still aliases the pinned staging buffer")
        res = LuffyResults(son, xluf, jmode,
                           np.array(self.p_status[slot][:c], I32))
        return res, np.array(self.p_muf3[slot][:c])

    def run(self, params, cfgs, ant, ant_idx, on_chunk=None):
        """A whole workload, chunked and overlapped across the streams.

        Returns ``(LuffyResults, muf3)`` for the whole workload, or ``None``
        when ``on_chunk`` is supplied (the caller consumed each chunk as it
        landed and nothing was kept).
        """
        cp = self._cp
        params = np.ascontiguousarray(params)
        _require_dtype(params, SETUP_PARAMS_DTYPE, "params")
        cfgs = np.ascontiguousarray(cfgs)
        _require_dtype(cfgs, SETUP_CONFIG_DTYPE, "cfgs")
        ant_idx = np.ascontiguousarray(ant_idx, I32)
        if params.ndim != 1:
            raise ValueError(f"params must be 1-D, got shape {params.shape}")
        n = params.shape[0]
        _require_rows(ant_idx, n, "ant_idx")
        if self._setup_shared is None:
            raise ValueError("no ionosphere: pass slots/iono to the "
                             "constructor or call set_ionosphere")
        # checked against the antenna set being INSTALLED, before anything
        # about the rig changes
        params = _check_setup_params(params, ant_idx, self._n_iono,
                                     cfgs.shape[0], _n_antenna_sets(ant))
        self._install_antennas(ant)
        d_cfg = cp.asarray(cfgs.view(np.uint8).ravel())
        for k in range(self.ns):
            self.d_cfg[k] = d_cfg          # tiny and read-only: shared
        # both uploads above went to the current stream; the slot streams
        # read them (see _drain_default_stream).  Once per run, before the
        # first chunk, so nothing in flight is serialised by it.
        _drain_default_stream()
        iono_idx = np.ascontiguousarray(params["iono"], I32)

        bounds = [(a, min(a + self.chunk, n))
                  for a in range(0, n, self.chunk)]
        keep = [] if on_chunk is None else None
        inflight = {}                       # slot -> (a, b) awaiting harvest
        try:
            for k, (a, b) in enumerate(bounds):
                slot = k % self.ns
                if slot in inflight:        # reuse: drain that slot first
                    pa, pb = inflight.pop(slot)
                    self.events[slot].synchronize()
                    out = self._harvest(slot, pa, pb)
                    if on_chunk is None:
                        keep.append((pa, pb) + out)
                    else:
                        on_chunk(pa, pb, out[0], out[1])
                self._stage(slot, params, iono_idx, ant_idx, a, b)
                self._enqueue(slot, b - a)
                inflight[slot] = (a, b)
            for slot in sorted(inflight, key=lambda s: inflight[s][0]):
                a, b = inflight[slot]
                self.events[slot].synchronize()
                out = self._harvest(slot, a, b)
                del inflight[slot]
                if on_chunk is None:
                    keep.append((a, b) + out)
                else:
                    on_chunk(a, b, out[0], out[1])
        finally:
            _quiesce(self.events, inflight)
        if on_chunk is not None:
            return None

        keep.sort(key=lambda t: t[0])
        son = np.empty(n, SON_DTYPE)
        xluf = np.empty((n, _RESULT_COLS["muf.xluf"][1]), F32)
        jmode = np.empty(n, I32)
        status = np.empty(n, I32)
        muf3 = np.empty((n, 3), F32)
        for a, b, res, m3 in keep:
            son[a:b] = res.son
            xluf[a:b] = res.xluf
            jmode[a:b] = res.jmode
            status[a:b] = res.status
            muf3[a:b] = m3
        return LuffyResults(son, xluf, jmode, status), muf3
