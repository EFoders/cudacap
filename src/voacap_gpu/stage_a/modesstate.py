"""COMMON /MODES/ -- the up-to-six propagation modes found for one hop.

FDIST scans the /REFLX/ ray table for the current hop distance (GHOP, radians)
and fills, for each mode IL=1..6 at sample area K: the take-off angle DELMOD,
virtual/true heights HPMOD/HTMOD, vertical frequency FVMOD, layer ITMOD and
deviative loss AFMOD.  A mode slot is marked unused by a NEGATIVE HPMOD (the
DO-100 preset is HPMOD=-1, ITMOD=5), which is also how FDIST rejects a mode
whose take-off angle falls below AMIND.

Stored 1-based: the (6,3) arrays as (7,4) [mode 1..6][area 1..3].
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32


def _m6():
    return np.zeros((7, 4), dtype=np.float32)


@dataclass
class ModesState:
    ghop: F32 = F32(0.0)
    delmod: np.ndarray = field(default_factory=_m6)
    hpmod: np.ndarray = field(default_factory=_m6)
    htmod: np.ndarray = field(default_factory=_m6)
    fvmod: np.ndarray = field(default_factory=_m6)
    afmod: np.ndarray = field(default_factory=_m6)
    itmod: np.ndarray = field(
        default_factory=lambda: np.zeros((7, 4), dtype=np.int32))
