"""Python side of the Stage C per-thread state ABI (see cuda/state.cuh).

For every Stage A state dataclass there is a numpy structured dtype whose
byte layout is IDENTICAL to the corresponding C struct in ``state.cuh``:
every member is 4 bytes (float32/int32) so neither side has padding, fields
are declared in the same order, and multi-dimensional numpy subarrays are
stored row-major exactly as the flattened C arrays.  ``tests/cuda/
test_state.py`` proves the correspondence field-for-field on the device.

``pack_<name>(obj)`` serialises a Stage A instance into a shape-() array of
the struct dtype; ``unpack_<name>(buf, obj)`` writes a (possibly mutated)
buffer back into a Stage A instance, in place for array fields.

Two representational quirks, both documented in state.cuh as well:

* ``SigdState.asm`` — ``asm`` is a C++ keyword, so the C member is named
  ``asm_``.  The numpy field keeps the dataclass name ``asm``; the rename
  applies to the C source only (``C_RENAMES``).
* ``SonState.mdl/mode/moder`` — Stage A holds lists of 2-char LAYTYP labels
  ("" for empty).  Here each becomes an int32 array holding the index into
  ``LAYTYP`` (" E"=0, "F1"=1, "F2"=2, "ES"=3, " N"=4), or -1 for "".
"""

import numpy as np

from voacap_gpu.scalar.sonstate import LAYTYP

F32 = np.float32
I32 = np.int32

# numpy field name -> C member name, where the dataclass name cannot be a C
# identifier.  Test-kernel generators must route through this.
C_RENAMES = {"asm": "asm_"}

# struct name -> fields that are LAYTYP-label lists on the Stage A side
LABEL_FIELDS = {
    "SonState": ("mdl", "mode", "moder"),
}


def _labels_to_idx(labels):
    """List of 2-char LAYTYP labels ('' allowed) -> int32 index array."""
    return np.array([-1 if s == "" else LAYTYP.index(s) for s in labels],
                    dtype=I32)


def _idx_to_labels(idx):
    """int32 index array -> list of LAYTYP labels ('' for -1)."""
    return ["" if i < 0 else LAYTYP[int(i)] for i in np.asarray(idx)]


# ---------------------------------------------------------------------------
# dtypes — field order matches the dataclass (and the C struct) exactly.
# All members are 4-byte scalars or subarrays, so the packed numpy layout
# (no padding, sequential offsets) equals the C layout.
# ---------------------------------------------------------------------------

ALLMODE_DTYPE = np.dtype([
    ("abps", F32, (21,)), ("crel", F32, (21,)), ("fldst", F32, (21,)),
    ("hn", F32, (21,)), ("hp", F32, (21,)), ("prob", F32, (21,)),
    ("rely", F32, (21,)), ("rgain", F32, (21,)), ("sigpow", F32, (21,)),
    ("sn", F32, (21,)), ("spro", F32, (21,)), ("tgain", F32, (21,)),
    ("timed", F32, (21,)), ("tloss", F32, (21,)), ("b", F32, (21,)),
    ("fslos", F32, (21,)), ("grlos", F32, (21,)), ("adv", F32, (21,)),
    ("obf", F32, (21,)), ("tllow", F32, (21,)), ("tlhgh", F32, (21,)),
    ("eff", F32, (21,)), ("nmode", I32, (21,)),
    ("nrel", I32), ("nmmod", I32),
])

FRQ_DTYPE = np.dtype([
    ("frea", F32, (14,)), ("frel", F32, (30,)), ("freq", F32),
    ("jmode", I32), ("itxrcp", I32, (3,)),
])

LAYER_DTYPE = np.dtype([
    ("km", I32), ("kfx", I32), ("iedp", I32), ("month", I32), ("ssn", F32),
    ("clat", F32, (6,)), ("clong", F32, (6,)), ("glat", F32, (6,)),
    ("gmdip", F32, (6,)), ("gyz", F32, (6,)), ("rd", F32, (6,)),
    ("fi", F32, (4, 6)), ("yi", F32, (4, 6)), ("hi", F32, (4, 6)),
    ("fvert", F32, (31, 6)), ("hprim", F32, (31, 6)),
    ("htrue", F32, (31, 6)), ("afac", F32, (31, 6)),
    ("fs", F32, (4, 6)), ("hs", F32, (6,)),
    ("f2m3", F32, (6,)), ("hpf2", F32, (6,)), ("zenang", F32, (6,)),
    ("zenmax", F32, (6,)), ("fsecv", F32, (4,)),
    ("htr", F32, (51, 4)), ("fnsq", F32, (51, 4)), ("ielect", I32, (4,)),
    ("rat", F32, (6,)), ("clck", F32, (6,)), ("abiy", F32, (6,)),
    ("artic", F32, (6,)), ("sigpat", F32, (6,)), ("epspat", F32, (6,)),
    ("psc", F32, (5,)),
])

LOSX_DTYPE = np.dtype([
    ("andvx", F32, (46, 4)), ("advx", F32, (46, 4)), ("aofx", F32, (46, 4)),
    ("arfx", F32, (46, 4)), ("grlosx", F32, (46, 4)),
    ("tgainx", F32, (46, 4)), ("tlskm", F32, (46, 4)),
    ("efflp", F32, (46,)),
])

LPATH_DTYPE = np.dtype([
    ("gcdlng", F32), ("delopt", F32), ("gmin", F32), ("ymin", F32),
    ("txrgml", F32, (46, 3)), ("ltxrgm", I32, (3,)),
])

MODES_DTYPE = np.dtype([
    ("ghop", F32),
    ("delmod", F32, (7, 4)), ("hpmod", F32, (7, 4)), ("htmod", F32, (7, 4)),
    ("fvmod", F32, (7, 4)), ("afmod", F32, (7, 4)), ("itmod", I32, (7, 4)),
])

MUF_DTYPE = np.dtype([
    ("emuf", F32, (25,)), ("f1muf", F32, (25,)), ("f2muf", F32, (25,)),
    ("esmuf", F32, (25,)), ("allmuf", F32, (25,)), ("fot", F32, (25,)),
    ("xluf", F32, (25,)), ("hpf", F32, (25,)), ("angmuf", F32, (25,)),
    ("modmuf", I32),
    ("sigl", F32, (5,)), ("sigu", F32, (5,)), ("delmuf", F32, (5,)),
    ("hpmuf", F32, (5,)), ("htmuf", F32, (5,)), ("fvmuf", F32, (5,)),
    ("afmuf", F32, (5,)), ("nhopmf", I32, (5,)),
    ("yfot", F32, (5,)), ("yhpf", F32, (5,)), ("ymuf", F32, (5,)),
])

ANOIS_DTYPE = np.dtype([
    ("atnu", F32), ("atny", F32), ("cc", F32), ("tm", F32), ("rcnse", F32),
    ("du", F32), ("dl", F32), ("sigm", F32), ("sygu", F32), ("sygl", F32),
    ("kj", I32), ("jk", I32),
])

NOISEDIAG_DTYPE = np.dtype([
    ("mode_n1", I32), ("mode_n2", I32), ("mode_n3", I32),
    ("xn_atnos", F32), ("xn_galactic", F32), ("xn_mmn", F32),
    ("method_used", I32),
])

RAYS_DTYPE = np.dtype([
    ("nang", I32), ("ang", F32, (41,)), ("ifob", I32, (41, 31, 6)),
])

REFLX_DTYPE = np.dtype([
    ("delfx", F32, (46, 4)), ("hpflx", F32, (46, 4)),
    ("htflx", F32, (46, 4)), ("gdflx", F32, (46, 4)),
    ("fvflx", F32, (46, 4)), ("afflx", F32, (46, 4)),
    ("gml", F32, (46, 4)), ("fhp", F32, (46, 4)),
    ("dskpkm", F32, (4,)), ("delskp", F32, (4,)), ("hpskp", F32, (4,)),
    ("htskp", F32, (4,)), ("dmaxkm", F32, (4,)), ("fvskp", F32, (4,)),
    ("iskp", I32, (4,)), ("imode", I32, (46, 4)),
    ("delpen", F32, (4, 6)), ("iaftxr", I32, (4,)),
])

SIGD_DTYPE = np.dtype([
    ("dsl", F32), ("asm", F32), ("dsu", F32), ("aglat", F32),
    ("dslf", F32), ("asmf", F32), ("dsuf", F32), ("acav", F32),
    ("feav", F32), ("afe", F32), ("bfe", F32), ("hnu", F32),
    ("htloss", F32), ("xnuz", F32), ("xve", F32),
])

TON_DTYPE = np.dtype([
    ("adj", F32), ("ads", F32), ("gnos", F32), ("got", F32), ("rel", F32),
    ("sl", F32), ("sls", F32), ("spr", F32), ("su", F32), ("sus", F32),
    ("xnoise", F32), ("znoise", F32), ("nf", I32),
])

SON_DTYPE = np.dtype([
    ("angle", F32, (14,)), ("angler", F32, (14,)), ("cprob", F32, (14,)),
    ("dblos", F32, (14,)), ("dblosl", F32, (14,)), ("dblosu", F32, (14,)),
    ("dbu", F32, (14,)), ("delay", F32, (14,)), ("dbw", F32, (14,)),
    ("nhp", I32, (14,)),
    ("xnynois", F32, (14,)), ("probmp", F32, (14,)), ("reliab", F32, (14,)),
    ("sndb", F32, (14,)), ("snpr", F32, (14,)), ("snrlw", F32, (14,)),
    ("snrup", F32, (14,)), ("sprob", F32, (14,)), ("vhigh", F32, (14,)),
    ("rneff", F32, (14,)),
    ("mdl", I32, (14,)), ("mode", I32, (14,)), ("moder", I32, (14,)),
    ("du_nois", F32, (14,)), ("dl_nois", F32, (14,)),
    ("snxx", F32, (14,)),
    ("gaint", F32, (14,)), ("gainr", F32, (14,)),
    ("d10r", F32), ("d50r", F32), ("d90r", F32),
    ("d10s", F32), ("d50s", F32), ("d90s", F32),
])

TIME_DTYPE = np.dtype([
    ("it", I32), ("gmt", F32), ("utime", F32, (25,)), ("gmtr", F32),
    ("xlmt", F32, (25,)), ("itim", I32), ("jtx", I32),
])

ZON_DTYPE = np.dtype([
    ("abps", F32, (8,)), ("crel", F32, (8,)), ("eff", F32, (8,)),
    ("fldst", F32, (8,)), ("grlos", F32, (8,)), ("hn", F32, (8,)),
    ("hp", F32, (8,)), ("prob", F32, (8,)), ("rely", F32, (8,)),
    ("rgain", F32, (8,)), ("sigpow", F32, (8,)), ("sn", F32, (8,)),
    ("spro", F32, (8,)), ("tgain", F32, (8,)), ("timed", F32, (8,)),
    ("tloss", F32, (8,)), ("b", F32, (8,)), ("fslos", F32, (8,)),
    ("adv", F32, (8,)), ("obf", F32, (8,)), ("tllow", F32, (8,)),
    ("tlhgh", F32, (8,)), ("nmode", I32, (8,)),
])

DTYPES = {
    "AllModeState": ALLMODE_DTYPE,
    "FrqState": FRQ_DTYPE,
    "LayerState": LAYER_DTYPE,
    "LosxState": LOSX_DTYPE,
    "LpathState": LPATH_DTYPE,
    "ModesState": MODES_DTYPE,
    "MufState": MUF_DTYPE,
    "AnoisState": ANOIS_DTYPE,
    "NoiseDiagState": NOISEDIAG_DTYPE,
    "RaysState": RAYS_DTYPE,
    "ReflxState": REFLX_DTYPE,
    "SigdState": SIGD_DTYPE,
    "TonState": TON_DTYPE,
    "SonState": SON_DTYPE,
    "TimeState": TIME_DTYPE,
    "ZonState": ZON_DTYPE,
}


# ---------------------------------------------------------------------------
# generic pack / unpack
# ---------------------------------------------------------------------------

def _pack(obj, dtype, labels=()):
    buf = np.zeros((), dtype=dtype)
    for name in dtype.names:
        if name in labels:
            buf[name] = _labels_to_idx(getattr(obj, name))
        else:
            buf[name] = getattr(obj, name)
    return buf


def _unpack(buf, obj, dtype, labels=()):
    for name in dtype.names:
        sub = dtype.fields[name][0]
        if name in labels:
            setattr(obj, name, _idx_to_labels(buf[name]))
        elif sub.shape:
            getattr(obj, name)[...] = buf[name]     # in place, keeps identity
        elif sub.kind == "i":
            setattr(obj, name, int(buf[name]))
        else:
            setattr(obj, name, F32(buf[name]))
    return obj


def pack_allmode(obj):
    return _pack(obj, ALLMODE_DTYPE)


def unpack_allmode(buf, obj):
    return _unpack(buf, obj, ALLMODE_DTYPE)


def pack_frq(obj):
    return _pack(obj, FRQ_DTYPE)


def unpack_frq(buf, obj):
    return _unpack(buf, obj, FRQ_DTYPE)


def pack_layer(obj):
    return _pack(obj, LAYER_DTYPE)


def unpack_layer(buf, obj):
    return _unpack(buf, obj, LAYER_DTYPE)


def pack_losx(obj):
    return _pack(obj, LOSX_DTYPE)


def unpack_losx(buf, obj):
    return _unpack(buf, obj, LOSX_DTYPE)


def pack_lpath(obj):
    return _pack(obj, LPATH_DTYPE)


def unpack_lpath(buf, obj):
    return _unpack(buf, obj, LPATH_DTYPE)


def pack_modes(obj):
    return _pack(obj, MODES_DTYPE)


def unpack_modes(buf, obj):
    return _unpack(buf, obj, MODES_DTYPE)


def pack_muf(obj):
    return _pack(obj, MUF_DTYPE)


def unpack_muf(buf, obj):
    return _unpack(buf, obj, MUF_DTYPE)


def pack_anois(obj):
    return _pack(obj, ANOIS_DTYPE)


def unpack_anois(buf, obj):
    return _unpack(buf, obj, ANOIS_DTYPE)


def pack_noisediag(obj):
    return _pack(obj, NOISEDIAG_DTYPE)


def unpack_noisediag(buf, obj):
    return _unpack(buf, obj, NOISEDIAG_DTYPE)


def pack_rays(obj):
    return _pack(obj, RAYS_DTYPE)


def unpack_rays(buf, obj):
    return _unpack(buf, obj, RAYS_DTYPE)


def pack_reflx(obj):
    return _pack(obj, REFLX_DTYPE)


def unpack_reflx(buf, obj):
    return _unpack(buf, obj, REFLX_DTYPE)


def pack_sigd(obj):
    return _pack(obj, SIGD_DTYPE)


def unpack_sigd(buf, obj):
    return _unpack(buf, obj, SIGD_DTYPE)


def pack_ton(obj):
    return _pack(obj, TON_DTYPE)


def unpack_ton(buf, obj):
    return _unpack(buf, obj, TON_DTYPE)


def pack_son(obj):
    return _pack(obj, SON_DTYPE, LABEL_FIELDS["SonState"])


def unpack_son(buf, obj):
    return _unpack(buf, obj, SON_DTYPE, LABEL_FIELDS["SonState"])


def pack_time(obj):
    return _pack(obj, TIME_DTYPE)


def unpack_time(buf, obj):
    return _unpack(buf, obj, TIME_DTYPE)


def pack_zon(obj):
    return _pack(obj, ZON_DTYPE)


def unpack_zon(buf, obj):
    return _unpack(buf, obj, ZON_DTYPE)


PACK = {
    "AllModeState": pack_allmode,
    "FrqState": pack_frq,
    "LayerState": pack_layer,
    "LosxState": pack_losx,
    "LpathState": pack_lpath,
    "ModesState": pack_modes,
    "MufState": pack_muf,
    "AnoisState": pack_anois,
    "NoiseDiagState": pack_noisediag,
    "RaysState": pack_rays,
    "ReflxState": pack_reflx,
    "SigdState": pack_sigd,
    "TonState": pack_ton,
    "SonState": pack_son,
    "TimeState": pack_time,
    "ZonState": pack_zon,
}

UNPACK = {
    "AllModeState": unpack_allmode,
    "FrqState": unpack_frq,
    "LayerState": unpack_layer,
    "LosxState": unpack_losx,
    "LpathState": unpack_lpath,
    "ModesState": unpack_modes,
    "MufState": unpack_muf,
    "AnoisState": unpack_anois,
    "NoiseDiagState": unpack_noisediag,
    "RaysState": unpack_rays,
    "ReflxState": unpack_reflx,
    "SigdState": unpack_sigd,
    "TonState": unpack_ton,
    "SonState": unpack_son,
    "TimeState": unpack_time,
    "ZonState": unpack_zon,
}
