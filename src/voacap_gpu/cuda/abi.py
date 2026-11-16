"""The host/device ABI of the LUFFY kernel: layouts, field sets, constants.

Pure numpy, no CuPy: the structured dtypes that mirror ``cuda/circuit.cuh``
and ``cuda/kernels/setup.cuh``, the measured input / carry / result field
sets, the word-run bookkeeping derived from them, and the per-thread byte
budgets.  ``cuda/driver.py``'s module docstring tells the whole story (the
ABI proof, the SoA /RAYS/ buffers, how the input set was measured); this
module is where those declarations live.
"""

import numpy as np

from voacap_gpu.cuda.state import (ALLMODE_DTYPE, ANOIS_DTYPE, FRQ_DTYPE,
                                      LAYER_DTYPE, LOSX_DTYPE, LPATH_DTYPE,
                                      MODES_DTYPE, MUF_DTYPE, NOISEDIAG_DTYPE,
                                      RAYS_DTYPE, REFLX_DTYPE, SIGD_DTYPE,
                                      SON_DTYPE, TIME_DTYPE, TON_DTYPE,
                                      ZON_DTYPE, pack_allmode, pack_anois,
                                      pack_frq, pack_layer, pack_losx,
                                      pack_lpath, pack_modes, pack_muf,
                                      pack_noisediag, pack_reflx, pack_sigd,
                                      pack_time, pack_ton, pack_zon,
                                      unpack_allmode, unpack_anois,
                                      unpack_frq, unpack_layer, unpack_losx,
                                      unpack_lpath, unpack_modes, unpack_muf,
                                      unpack_noisediag, unpack_reflx,
                                      unpack_sigd, unpack_time, unpack_ton,
                                      unpack_zon)

F32 = np.float32
I32 = np.int32

# /SON/ MDL, LUFFY's model tag — cuda/circuit.cuh VC_MDL_*.
MDL_CODE = {"": -1, " ": 0, "S": 1, "L": 2, "M": 3}
MDL_LABEL = {v: k for k, v in MDL_CODE.items()}

# (Circuit attribute, per-struct dtype, pack fn, unpack fn), in Circuit order.
# `rays` is deliberately absent -- see driver.py's docstring; it travels in its
# own SoA buffers, packed by `pack_rays` / `unpack_rays` straight from here.
_BLOCKS = (
    ("layers", LAYER_DTYPE, pack_layer, unpack_layer),
    ("muf", MUF_DTYPE, pack_muf, unpack_muf),
    ("reflx", REFLX_DTYPE, pack_reflx, unpack_reflx),
    ("modes", MODES_DTYPE, pack_modes, unpack_modes),
    ("zon", ZON_DTYPE, pack_zon, unpack_zon),
    ("allm", ALLMODE_DTYPE, pack_allmode, unpack_allmode),
    ("son", SON_DTYPE, None, None),                 # MDL: tables.py's shim
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
# THE INPUT SET — measured, not guessed.  See driver.py's docstring.
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
# SETUP_STATUS_BAD_HOUR -- 1301, VC_ST_SETUP_HOUR_OOB in the global status
# table (cuda/kernels/status.cuh; engine.STATUS_SETUP_HOUR), since Stage A
# raises IndexError on such an hour.  The host check in `_check_setup_params`
# refuses these rows before launch; the guard is the defence in depth for a
# caller that bypasses it.
SETUP_IPFLAG_REFUSED = -1
SETUP_STATUS_BAD_HOUR = 1301

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
