"""Host-side tables and whole-circuit pack / unpack.

The SHARED tables (REDMAP coefficient maps, antenna patterns, the setup
chain's own per-ionosphere tables) that move to the device once per launch,
and the Stage A ``Circuit`` <-> ``(CIRCUIT_DTYPE, RAYS_DTYPE)`` packers,
including the /SON/ MDL shim.  Pure numpy, no CuPy.
"""

import numpy as np

from voacap_gpu.cuda.abi import (_BLOCKS, _IONO_TABLES, _SCALARS,
                                 _SETUP_TABLES, CIRCUIT_DTYPE, F32, I32,
                                 MDL_CODE, MDL_LABEL, SETUP_NAB, SETUP_NHOUR)
from voacap_gpu.cuda.state import pack_rays, pack_son, unpack_rays, unpack_son


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
    absent from the first buffer and IS the second: see driver.py's docstring.
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
