"""Mutable per-sample-area ionospheric state for the Module 2 layer routines.

TIMVAR/EF1VAR/F2VAR/ESIND/IONSET communicate entirely through COMMON
blocks (/RON/, /ES/, /MFAC/, /GEOG/).  Stage A keeps that coupling explicit
and faithful: this dataclass bundles exactly those arrays as 1-based numpy
arrays (index 0 padded so subscripts match the Fortran) and the routines
read and mutate it in place, in the Fortran's own order.

The geographic inputs (CLAT/CLONG/GLAT/GMDIP/GYZ per sample area) are set
upstream by GEOM/MAGVAR; here the caller (or the tier-B dump) fills them.
KM is the number of active sample areas (1 for a short P2P path, up to 5).
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

# SUN(2,12), COMMON /SSP/, verbatim from blkdat.for DATA SUN (column-major:
# two subsolar-declination samples per month, in degrees).  TIMVAR picks the
# nearer one to bound the zenith angle.
SUN = np.zeros((3, 13), dtype=np.float32)
SUN[1, 1:] = [-23.05, -17.30, -7.88, 4.26, 14.84, 21.93,
              23.15, 18.20, 8.55, -2.90, -14.20, -21.66]
SUN[2, 1:] = [-17.31, -7.89, 4.21, 14.83, 21.93, 23.45,
              18.23, 8.68, -2.86, -14.16, -21.68, -23.45]


def _f2(shape):
    return np.zeros(shape, dtype=np.float32)


@dataclass
class LayerState:
    """/RON/, /ES/, /MFAC/, /GEOG/ + the /PSCA/, /SSP/, SSN scalars.

    (3,5) Fortran arrays are stored (4,6) so ``a[layer, area]`` uses the
    Fortran 1-based subscripts directly; (5,) arrays are stored (6,).
    """

    km: int = 1
    kfx: int = 1
    iedp: int = -1
    month: int = 1
    ssn: F32 = F32(0.0)

    # /GEOG/ + /RON/ geographic inputs (radians), 1-based (6,)
    clat: np.ndarray = field(default_factory=lambda: _f2(6))
    clong: np.ndarray = field(default_factory=lambda: _f2(6))
    glat: np.ndarray = field(default_factory=lambda: _f2(6))
    gmdip: np.ndarray = field(default_factory=lambda: _f2(6))
    gyz: np.ndarray = field(default_factory=lambda: _f2(6))

    # /RON/ layer parameters, 1-based (4,6): [layer 1..3][area 1..5]
    fi: np.ndarray = field(default_factory=lambda: _f2((4, 6)))
    yi: np.ndarray = field(default_factory=lambda: _f2((4, 6)))
    hi: np.ndarray = field(default_factory=lambda: _f2((4, 6)))

    # /ES/ sporadic-E, 1-based
    fs: np.ndarray = field(default_factory=lambda: _f2((4, 6)))
    hs: np.ndarray = field(default_factory=lambda: _f2(6))

    # /MFAC/ F2 factors + zenith angles, 1-based (6,)
    f2m3: np.ndarray = field(default_factory=lambda: _f2(6))
    hpf2: np.ndarray = field(default_factory=lambda: _f2(6))
    zenang: np.ndarray = field(default_factory=lambda: _f2(6))
    zenmax: np.ndarray = field(default_factory=lambda: _f2(6))
    # /MFAC/ FSECV(3) (F1 linear-layer flag frequency), 1-based (4,)
    fsecv: np.ndarray = field(default_factory=lambda: _f2(4))

    # /RON/ electron-density profile HTR(50,3)/FNSQ(50,3), 1-based (51,4);
    # LECDEN fills column 1, GETHP reads it.  IELECT(3) external-EDP flags.
    htr: np.ndarray = field(default_factory=lambda: _f2((51, 4)))
    fnsq: np.ndarray = field(default_factory=lambda: _f2((51, 4)))
    ielect: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.int32))

    # /GEOG/ outputs of TIMVAR, 1-based (6,)
    rat: np.ndarray = field(default_factory=lambda: _f2(6))
    clck: np.ndarray = field(default_factory=lambda: _f2(6))
    abiy: np.ndarray = field(default_factory=lambda: _f2(6))

    # /PSCA/ critical-frequency multipliers, 1-based (5,): E,F1,F2,Es
    psc: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 1.0, 1.0, 0.7],
                                         dtype=np.float32))
