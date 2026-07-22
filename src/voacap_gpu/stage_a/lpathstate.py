"""COMMON /LPATH/ -- the long-path branch's own working state.

Six fields, and only two of them are variables:

    GCDLNG  10000 km, the distance past which METHOD 30 abandons the
            short-path hop model.  HFMUFS/HFAREA/HFCIRAF compare GCDKM
            against it and raise IPFLAG to 200 ("use the long path").
    DELOPT  3 degrees, the a-priori optimum take-off angle.  SELTXR uses it
            as its third tie-break and GETTOP as the elevation at which the
            E-layer penetration frequencies are evaluated.
    GMIN    3 dB, the gain-minus-loss window inside which SELTXR stops
            treating two modes as distinguishable.
    YMIN    0.1 hop, the same idea for the fractional-hop tie-break.
    TXRGML  GMLOSS's gain-minus-loss column for each end, (45,2).
    LTXRGM  the ray index SELTXR chose at each end, (2).

The four scalars are blkdat DATA constants that nothing in the program ever
assigns (blkdat.for:163), so they are defaults here rather than inputs; a
caller that wants to sweep them still can.  LTXRGM starts at 1,1 -- also from
blkdat -- which matters, because LNGPAT's guard is ``LTXRGM(n) <= 0`` and a
never-run SELTXR therefore reads as "ray 1 at both ends" rather than as no
answer.

Stored 1-based: TXRGML as (46,3) [ray 1..45][end 1..2], LTXRGM as (3,).
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

NRAY = 45


def _txrgml():
    return np.zeros((NRAY + 1, 3), dtype=np.float32)


@dataclass
class LpathState:
    # blkdat.for:163 -- DATA GCDLNG/10000./,DELOPT/3./,GMIN/3./,YMIN/.1/
    gcdlng: F32 = F32(10000.0)
    delopt: F32 = F32(3.0)
    gmin: F32 = F32(3.0)
    ymin: F32 = F32(0.1)
    txrgml: np.ndarray = field(default_factory=_txrgml)
    # blkdat.for:183 -- DATA LTXRGM/1,1/
    ltxrgm: np.ndarray = field(
        default_factory=lambda: np.array([0, 1, 1], dtype=np.int32))
