"""Stage A transliteration of SANG
(reference/voacapl/src/voacapw/sang.for).

Restricts the take-off-angle scan (NANG entries of the ANG table) by
great-circle distance, then widens for a user minimum angle AMIND.  The
ANG table DATA is from blkdat.for.
"""

import numpy as np

from voacap_gpu.scalar.f32math import F32

ANG = np.array([
    0., 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0,
    14., 16., 18., 20., 22., 24., 26., 28., 30., 32., 34., 36., 38.,
    40., 42., 44., 46., 48., 50., 52., 54., 56., 60., 65., 70., 75.,
    80., 85., 89.99,
], dtype=np.float32)

_NANGX = [40, 34, 29, 24, 19, 14, 12, 9]


def sang(gcdkm, amind) -> int:
    """Returns NANG, the number of usable ANG entries."""
    gcdkm = F32(gcdkm)
    amind = F32(amind)
    # ID = GCDKM/2000. + 1. — REAL*4 expression truncated to INTEGER
    id_ = int(gcdkm / F32(2000.0) + F32(1.0))
    id_ = min(id_, 8)
    iang = _NANGX[id_ - 1]
    if amind > 0.0:  # IF(AMIND) 120,120,100
        if ANG[iang - 1] - F32(10.0) - amind < 0.0:  # 100 -> 105
            for ia in range(1, 41):  # DO 115
                iang = ia
                if ANG[iang - 1] - F32(8.5) - amind >= 0.0:  # 110 -> 120
                    break
    return iang
