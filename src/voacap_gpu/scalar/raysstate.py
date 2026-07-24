"""COMMON /RAYS/ — the ray-set / reflectrix table.

ANG(40) is the take-off-angle table (blkdat.for, the same values SANG uses);
NANG is the usable count SANG computes for the path distance; IFOB(40,30,5)
is the reflectrix — the oblique operating frequency in kHz (integer) at each
[take-off angle][30 sounding freqs][sample area], filled by FOBBY.

Stored 1-based (index 0 padded) so the Fortran subscripts match: ANG(41,),
IFOB(41,31,6).
"""

from dataclasses import dataclass, field

import numpy as np

from voacap_gpu.scalar.sang import ANG as _SANG_ANG


def _default_ang():
    a = np.zeros(41, dtype=np.float32)
    a[1:41] = _SANG_ANG  # SANG's ANG is 0-based length 40 -> ANG(1..40)
    return a


@dataclass
class RaysState:
    nang: int = 0
    ang: np.ndarray = field(default_factory=_default_ang)
    ifob: np.ndarray = field(
        default_factory=lambda: np.zeros((41, 31, 6), dtype=np.int32))
