"""Stage A transliteration of GEOM
(reference/voacapl/src/voacapw/geom.for).

Computes the circuit path geometry: great-circle distance and bearings
(with the NPSL=1 long-path/through-antipode option), the 1/3/5 ionospheric
sample areas along the path, their geographic and geomagnetic coordinates,
magnetic dip/gyrofrequency (via MAGVAR), and per-area ground constants.

The land/sea decision comes from NOISY mode 7 in the Fortran (the
land-mass map, module 7 territory) and is injected here as ``wld_fn(rfltd,
clgd) -> wld`` (degrees in, land iff wld >= 0).  The oracle dump driver
stubs the same seam with an identical rule, so both ground-constant
branches are validated without the map data.

All REAL*4 (float32); transcendentals via f32math.  Fortran statement
labels are cited so the code audits line-against-line with geom.for.
"""

from dataclasses import dataclass, field

import numpy as np

from voacap_gpu.stage_a.constants import D2R, PI, PI2, PIO2, R2D, RZ
from voacap_gpu.stage_a.f32math import F32, acos32, cos32, sin32
from voacap_gpu.stage_a.magfin import magvar

_EPSLON = F32(1.0e-7)
_DDX = F32(0.03)  # closest Tx & Rx can be (degrees)


def _clamp1(qcos):
    if np.abs(qcos) > F32(1.0):
        qcos = np.float32(np.copysign(F32(1.0), qcos))
    return qcos


@dataclass
class GeomResult:
    """The /DON/, /RON/ and /GEOG/ fields GEOM writes (float32; arrays are
    length 5, entries beyond km untouched-zero as in the dump driver)."""

    dlong: np.float32 = F32(0.0)
    gcd: np.float32 = F32(0.0)
    gcdkm: np.float32 = F32(0.0)
    btr: np.float32 = F32(0.0)
    btrd: np.float32 = F32(0.0)
    brtd: np.float32 = F32(0.0)
    alatd: np.float32 = F32(0.0)
    rlatd: np.float32 = F32(0.0)  # GEOM may nudge the input
    km: int = 0
    rd: np.ndarray = field(default_factory=lambda: np.zeros(5, np.float32))
    clat: np.ndarray = field(default_factory=lambda: np.zeros(5, np.float32))
    clong: np.ndarray = field(default_factory=lambda: np.zeros(5, np.float32))
    glat: np.ndarray = field(default_factory=lambda: np.zeros(5, np.float32))
    gyz: np.ndarray = field(default_factory=lambda: np.zeros(5, np.float32))
    gmdip: np.ndarray = field(default_factory=lambda: np.zeros(5, np.float32))
    sigpat: np.ndarray = field(default_factory=lambda: np.zeros(5, np.float32))
    epspat: np.ndarray = field(default_factory=lambda: np.zeros(5, np.float32))


def geom(tlatd, tlongd, rlatd, rlongd, npsl, pole_lat, pole_lon,
         wld_fn) -> GeomResult:
    """Angles in degrees (south/west negative); pole from north_pole.txt."""
    tlatd = F32(tlatd)
    tlongd = F32(tlongd)
    rlatd = F32(rlatd)
    rlongd = F32(rlongd)
    out = GeomResult()

    glt = F32(pole_lat) * D2R  # geom.for:40-41
    glg = F32(pole_lon) * D2R
    # 42-47: coincident endpoints — move Rx away from Tx a little bit
    if np.abs(tlatd - rlatd) <= _DDX and np.abs(tlongd - rlongd) <= _DDX:
        dd = _DDX
        if rlatd < 0.0:
            dd = -_DDX
        rlatd = tlatd - dd
    out.rlatd = rlatd
    tlat = tlatd * D2R  # 48-51
    tlong = tlongd * D2R
    rlat = rlatd * D2R
    rlong = rlongd * D2R
    if np.abs(rlatd) > F32(89.9):  # 52-53: at poles, force long=0
        rlong = F32(0.0)
    if np.abs(tlatd) > F32(89.9):
        tlong = F32(0.0)

    # 55-66: great circle distance
    dlong = tlong - rlong  # label 100
    if np.abs(dlong) > PI:
        dlong = F32(dlong - np.copysign(PI2, dlong))
    if npsl == 1:  # long great circle path
        dlong = F32(dlong - np.copysign(PI2, dlong))
    qcos = sin32(tlat) * sin32(rlat) + cos32(tlat) * cos32(rlat) * cos32(dlong)
    qcos = _clamp1(qcos)
    gcd = acos32(qcos)
    if gcd < F32(0.000001):  # minimum distance is 31.85 meters
        gcd = F32(0.000001)
    if npsl == 1:
        gcd = PI2 - gcd
    gcdkm = gcd * RZ
    out.dlong = dlong
    out.gcd = gcd
    out.gcdkm = gcdkm

    # 68-82: bearing transmitter -> receiver
    if cos32(tlat) - _EPSLON <= 0.0:  # arithmetic IF 101,101,104
        btr = F32(0.0) if tlat <= 0.0 else PI  # 102/103
    else:  # 104
        qcos = F32((sin32(rlat) - sin32(tlat) * cos32(gcd))
                   / (cos32(tlat) * sin32(gcd)))
        qcos = _clamp1(qcos)
        btr = acos32(qcos)
    if dlong > 0.0:  # 82
        btr = PI2 - btr
    # 84-98: bearing receiver -> transmitter
    if cos32(rlat) - _EPSLON <= 0.0:  # 111,111,114
        brt = F32(0.0) if rlat <= 0.0 else PI  # 112/113
    else:  # 114
        qcos = F32((sin32(tlat) - sin32(rlat) * cos32(gcd))
                   / (cos32(rlat) * sin32(gcd)))
        qcos = _clamp1(qcos)
        brt = acos32(qcos)
    if dlong < 0.0:  # 98
        brt = PI2 - brt
    out.btr = btr
    out.btrd = btr * R2D
    out.brtd = brt * R2D

    # 104-120: select sample areas by distance
    if gcdkm <= F32(2000.01):
        km = 1
        out.rd[0] = gcd / F32(2.0)
    elif gcdkm <= F32(4000.0):
        km = 3
        out.rd[0] = F32(1000.0) / RZ
        out.rd[1] = gcd / F32(2.0)
        out.rd[2] = gcd - out.rd[0]
    else:
        km = 5
        out.rd[0] = F32(1000.0) / RZ
        out.rd[1] = out.rd[0] + out.rd[0]
        out.rd[2] = gcd / F32(2.0)
        out.rd[3] = gcd - out.rd[1]
        out.rd[4] = gcd - out.rd[0]
    out.km = km

    # 125-184: reflection area coordinates, geomagnetic latitude, ground
    for i in range(km):  # DO 180
        drf = out.rd[i]
        ctlat = cos32(tlat)
        if ctlat - _EPSLON < 0.0:  # 71,76,76 — STRICT negative is polar
            rflt = F32(tlat - np.copysign(drf, tlat))  # 71
            if np.abs(rflt) > PIO2:
                rflt = F32(PIO2 * np.copysign(F32(1.0), rflt))
            rflg = rlong
        else:  # 76
            qcos = cos32(drf) * sin32(tlat) + sin32(drf) * cos32(tlat) * cos32(btr)
            qcos = _clamp1(qcos)
            rflt = PIO2 - acos32(qcos)
            if cos32(rflt) - _EPSLON <= 0.0:  # 78,78,79: area near a pole
                rflg = tlong  # 78
            else:  # 79
                qcos = F32((cos32(drf) - sin32(rflt) * sin32(tlat))
                           / (cos32(rflt) * cos32(tlat)))
                qcos = _clamp1(qcos)
                rflg = acos32(qcos)
                if drf >= PI:
                    rflg = PI2 - rflg
                rflg = F32(tlong - np.copysign(rflg, dlong))
                if np.abs(rflg) > PI:
                    rflg = F32(rflg - np.copysign(PI2, rflg))
        # 155-163: geomagnetic latitude
        qcos = sin32(glt) * sin32(rflt) + cos32(glt) * cos32(rflt) * cos32(rflg - glg)
        qcos = _clamp1(qcos)
        gat = acos32(qcos)
        out.glat[i] = PIO2 - gat
        out.clat[i] = rflt
        out.clong[i] = rflg
        # 166-167: magnetic field variables
        gyz, gmdip, clg = magvar(rflt, rflg)
        out.gyz[i] = gyz
        out.gmdip[i] = gmdip
        # 171-182: land-mass lookup -> ground constants
        rfltd = rflt * R2D
        clgd = clg * R2D
        if clgd < 0.0:
            clgd = clgd + F32(360.0)
        wld = wld_fn(rfltd, clgd)
        if wld >= 0.0:  # 115: land
            out.sigpat[i] = F32(0.001)
            out.epspat[i] = F32(4.0)
        else:  # sea
            out.sigpat[i] = F32(5.0)
            out.epspat[i] = F32(80.0)

    # 186-187
    alatd = np.abs(out.clat[0]) * R2D
    if km != 1:
        alatd = np.abs((out.clat[0] + out.clat[1] + out.clat[2]) / F32(3.0)) * R2D
    out.alatd = F32(alatd)
    return out
