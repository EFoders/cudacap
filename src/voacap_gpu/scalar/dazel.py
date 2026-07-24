"""Stage A transliteration of DAZEL (reference/voacapl/src/voacapw/dazel.for).

Great-circle calculations. Two modes:
- mode 0: given end-point coordinates, return distance and azimuths with
  elevations/take-off angles.
- mode 1: given bearing and great-circle distance, return end-point
  coordinates with elevations/take-off angles.

Precision contract (see docs/precision_map.md): the Fortran declares
``IMPLICIT DOUBLE PRECISION(A-L,N-Y)`` — every internal variable is float64,
but ``M`` (mode, integer) and ``Z`` (the COMMON/AZEL/ interface) are excluded,
so all ``z*`` fields are REAL*4/float32. Single-precision literals in the
source (``1.0E-3``, ``-89.9``, ``1.0E-5``) are narrowed to float32 before use
exactly as Fortran evaluates them; constants declared with ``D0`` exponents
(including the deliberately truncated DTOR/RTOD) are float64 verbatim.

Fortran statement labels are cited in comments so the code can be audited
line-against-line with dazel.for.
"""

from dataclasses import dataclass, fields

import numpy as np

F32 = np.float32
F64 = np.float64

# DATA statements (dazel.for:53-54). DTOR/RTOD are the Fortran's own
# truncated constants — deliberately NOT pi/180 to full precision.
PI = F64(3.141592653589793238462643)
RERTH = F64(6370.0)
DTOR = F64(0.01745329252)
RTOD = F64(57.29577951)

# Single-precision literals appearing in mixed expressions (values as REAL*4).
_C1E3 = F32(1.0e-3)
_C1E5 = F64(F32(1.0e-5))
_C89_9 = F64(F32(-89.9))
_C89_9P = F64(F32(89.9))


@dataclass
class Azel:
    """COMMON/AZEL/ — all fields REAL*4 (dazel.for:51-52)."""

    ztlat: np.float32 = F32(0.0)
    ztlon: np.float32 = F32(0.0)
    ztht: np.float32 = F32(0.0)
    zrlat: np.float32 = F32(0.0)
    zrlon: np.float32 = F32(0.0)
    zrht: np.float32 = F32(0.0)
    ztaz: np.float32 = F32(0.0)
    zraz: np.float32 = F32(0.0)
    ztelv: np.float32 = F32(0.0)
    zrelv: np.float32 = F32(0.0)
    zd: np.float32 = F32(0.0)
    zdgc: np.float32 = F32(0.0)
    ztakof: np.float32 = F32(0.0)
    zrakof: np.float32 = F32(0.0)

    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, F32(getattr(self, f.name)))


def dazel(z: Azel, mode: int) -> None:
    if mode == 1:
        _mode1(z)
        return
    _mode0(z)


def _mode0(z: Azel) -> None:
    # 56-65: load interface into doubles; clamp latitudes off the poles
    tlats = F64(z.ztlat)
    tlons = F64(z.ztlon)
    thts = F64(F32(z.ztht) * _C1E3)  # REAL*4 multiply, then widen
    rlats = F64(z.zrlat)
    rlons = F64(z.zrlon)
    rhts = F64(F32(z.zrht) * _C1E3)
    if tlats <= F64(-90.0):
        tlats = _C89_9
    if tlats >= F64(90.0):
        tlats = _C89_9P
    if rlats <= F64(-90.0):
        rlats = _C89_9
    if rlats >= F64(90.0):
        rlats = _C89_9P

    # 66-75: DELON normalized into (-180, 180] via the label 12/18 loops
    delat = rlats - tlats
    adlat = np.abs(delat)
    delon = rlons - tlons
    if delon - F64(-180.0) < 0.0:  # arithmetic IF, label 12 loop
        while delon - F64(-180.0) < 0.0:
            delon = delon + F64(360.0)
    else:  # label 16/18 loop
        while delon - F64(180.0) > 0.0:
            delon = delon - F64(360.0)
    adlon = np.abs(delon)
    delht = rhts - thts

    if adlon - _C1E5 <= 0.0:  # 77
        if adlat - _C1E5 <= 0.0:  # 78
            # 24-35: same coordinates; three delht sub-cases
            z.ztaz = F32(0.0)
            z.zraz = F32(0.0)
            if delht < 0.0:  # 25
                z.ztelv = F32(-90.0)
                z.zrelv = F32(90.0)
                z.zd = F32(-delht)
                z.zdgc = F32(0.0)
                return
            if delht == 0.0:  # 30
                z.ztelv = F32(0.0)
                z.zrelv = F32(0.0)
                z.zd = F32(0.0)
                z.zdgc = F32(0.0)
                return
            z.ztelv = F32(90.0)  # 35
            z.zrelv = F32(-90.0)
            z.zd = F32(delht)
            z.zdgc = F32(0.0)
            return
        # 40-50: same longitude, distinct latitudes
        if delat <= 0.0:  # 42
            z.ztaz = F32(180.0)
            z.zraz = F32(0.0)
        else:  # 45
            z.ztaz = F32(0.0)
            z.zraz = F32(180.0)
        gc = adlat * DTOR  # 50
        sgc = np.sin(F64(0.5) * gc)
        d = np.sqrt(
            delht * delht
            + F64(4.0) * (RERTH + thts) * (RERTH + rhts) * sgc * sgc
        )
        z.zd = F32(d)
    else:
        # 55-60: distinct longitudes; identify west/east points
        if delon <= 0.0:  # 56
            wlat = rlats * DTOR
            elat = tlats * DTOR
        else:  # 60
            wlat = tlats * DTOR
            elat = rlats * DTOR
        # 65: azimuths at W and E
        sdlat = np.sin(F64(0.5) * adlat * DTOR)
        sdlon = np.sin(F64(0.5) * adlon * DTOR)
        sadln = np.sin(adlon * DTOR)
        cwlat = np.cos(wlat)
        celat = np.cos(elat)
        p = F64(2.0) * (sdlat * sdlat + sdlon * sdlon * cwlat * celat)
        sgc = np.sqrt(p * (F64(2.0) - p))
        sdlat = np.sin(elat - wlat)  # reused as sin(delta-lat), per Fortran
        cwaz = (F64(2.0) * celat * np.sin(wlat) * sdlon * sdlon + sdlat) / sgc
        swaz = sadln * celat / sgc
        waz = np.arctan2(swaz, cwaz) * RTOD
        ceaz = (F64(2.0) * cwlat * np.sin(elat) * sdlon * sdlon - sdlat) / sgc
        seaz = sadln * cwlat / sgc
        eaz = np.arctan2(seaz, ceaz) * RTOD
        eaz = F64(360.0) - eaz
        if delon <= 0.0:  # 141-147
            z.ztaz = F32(eaz)
            z.zraz = F32(waz)
        else:
            z.ztaz = F32(waz)
            z.zraz = F32(eaz)
        # 152-155: straight-line distance and great-circle angle
        d = np.sqrt(
            delht * delht + F64(2.0) * (RERTH + thts) * (RERTH + rhts) * p
        )
        z.zd = F32(d)
        cgc = F64(1.0) - p
        gc = np.arctan2(sgc, cgc)

    _elevations(z, gc, d, delht, thts, rhts, set_zdgc=True)


def _elevations(z, gc, d, delht, thts, rhts, set_zdgc):
    # 140: great-circle distance (mode 1 enters at 142, leaving zdgc as-is)
    if set_zdgc:
        z.zdgc = F32(gc * RERTH)
    # 142-150: order end points by height
    if delht >= 0.0:  # 145
        aht = rhts
        bht = thts
    else:
        aht = thts
        bht = rhts
    saelv = (
        F64(0.5)
        * (d * d + np.abs(delht) * (RERTH + aht + RERTH + bht))
        / (d * (RERTH + aht))
    )
    arg = np.maximum(F64(0.0), F64(1.0) - saelv * saelv)
    aelv = np.arctan2(saelv, np.sqrt(arg))
    belv = (aelv - gc) * RTOD
    aelv = -aelv * RTOD
    # 171-183: take-off angles for 4/3 earth radius
    r4thd = RERTH * F64(4.0) / F64(3.0)
    gc = F64(0.75) * gc
    sgc = np.sin(F64(0.5) * gc)
    p = F64(2.0) * sgc * sgc
    aalt = r4thd + aht
    balt = r4thd + bht
    da = np.sqrt(delht * delht + F64(2.0) * aalt * balt * p)
    saelv = F64(0.5) * (da * da + np.abs(delht) * (aalt + balt)) / (da * aalt)
    arg = np.maximum(F64(0.0), F64(1.0) - saelv * saelv)
    atakof = np.arctan(saelv / np.sqrt(arg))
    btakof = (atakof - gc) * RTOD
    atakof = -atakof * RTOD
    # 184-194: assign outputs by delht sign
    if delht < 0.0:  # 151
        z.ztelv = F32(aelv)
        z.zrelv = F32(belv)
        z.ztakof = F32(atakof)
        z.zrakof = F32(btakof)
    else:  # 155
        z.ztelv = F32(belv)
        z.zrelv = F32(aelv)
        z.ztakof = F32(btakof)
        z.zrakof = F32(atakof)


def _mode1(z: Azel) -> None:
    # 198-217: end point from bearing and distance
    tlatr = F64(z.ztlat) * DTOR
    tazr = F64(z.ztaz) * DTOR
    gc = F64(z.zdgc) / RERTH
    colat = PI / F64(2.0) - tlatr
    cosco = np.cos(colat)
    sinco = np.sin(colat)
    cosgc = np.cos(gc)
    singc = np.sin(gc)
    cosb = cosco * cosgc + sinco * singc * np.cos(tazr)
    arg = np.maximum(F64(0.0), F64(1.0) - cosb * cosb)
    b = np.arctan2(np.sqrt(arg), cosb)
    arc = (cosgc - cosco * cosb) / (sinco * np.sin(b))
    arg = np.maximum(F64(0.0), F64(1.0) - arc * arc)
    rdlon = np.arctan2(np.sqrt(arg), arc)
    z.zrlat = F32((PI / F64(2.0) - np.abs(b)) * RTOD)
    drlat = F64(z.zrlat)  # 214-215: sign transfer round-trips through REAL*4
    z.zrlat = F32(np.copysign(drlat, cosb))
    z.zrlon = F32(F64(z.ztlon) + np.abs(rdlon) * RTOD)
    if z.ztaz > F32(180.0):  # 217: REAL*4 comparison
        z.zrlon = F32(F64(z.ztlon) - np.abs(rdlon) * RTOD)
    # 218-223: distance for the elevation block, then join mode 0 at 142.
    # Note: mode 1 does NOT write zd or zdgc (D stays internal).
    thts = F64(F32(z.ztht) * _C1E3)
    rhts = F64(F32(z.zrht) * _C1E3)
    delht = rhts - thts
    sgc = np.sin(F64(0.5) * gc)
    d = np.sqrt(
        delht * delht + F64(4.0) * (RERTH + thts) * (RERTH + rhts) * sgc * sgc
    )
    _elevations(z, gc, d, delht, thts, rhts, set_zdgc=False)
