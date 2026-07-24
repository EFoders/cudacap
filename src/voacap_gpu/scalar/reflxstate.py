"""COMMON /REFLX/ -- the per-control-area reflectrix / ray-set table.

FINDF fills this for a control area K: for each ray (index IAF 1..45) the
take-off angle DELFX, virtual/true heights HPFLX/HTFLX, ground distance GDFLX,
vertical frequency FVFLX, deviative-loss AFFLX and layer IMODE, plus the skip
(minimum-distance) ray DSKPKM/DELSKP/HPSKP/HTSKP/FVSKP/ISKP and the maximum
distance DMAXKM.  PENANG fills the penetration angles DELPEN(3,5) first.
GML/FHP are gain/virtual-height columns the later loss routines add.

Stored 1-based: the (45,3) arrays as (46,4) [ray 1..45][area 1..3], the (3,)
arrays as (4,), DELPEN(3,5) as (4,6) [layer 1..3][area 1..5].  Mirrors the
Fortran layout; float32 (IMODE/ISKP integer).
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32


def _r45():
    return np.zeros((46, 4), dtype=np.float32)


def _r3():
    return np.zeros(4, dtype=np.float32)


@dataclass
class ReflxState:
    delfx: np.ndarray = field(default_factory=_r45)
    hpflx: np.ndarray = field(default_factory=_r45)
    htflx: np.ndarray = field(default_factory=_r45)
    gdflx: np.ndarray = field(default_factory=_r45)
    fvflx: np.ndarray = field(default_factory=_r45)
    afflx: np.ndarray = field(default_factory=_r45)
    gml: np.ndarray = field(default_factory=_r45)
    fhp: np.ndarray = field(default_factory=_r45)
    dskpkm: np.ndarray = field(default_factory=_r3)
    delskp: np.ndarray = field(default_factory=_r3)
    hpskp: np.ndarray = field(default_factory=_r3)
    htskp: np.ndarray = field(default_factory=_r3)
    dmaxkm: np.ndarray = field(default_factory=_r3)
    fvskp: np.ndarray = field(default_factory=_r3)
    iskp: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int32))
    imode: np.ndarray = field(
        default_factory=lambda: np.zeros((46, 4), dtype=np.int32))
    # DELPEN(3,5): [layer 1..3][area 1..5]
    delpen: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 6), dtype=np.float32))
    # IAFTXR (really /LOSX/): the ray count FINDF found for each area, 1-based
    # (4,) -- co-located here with the ray table it indexes.
    iaftxr: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.int32))
