"""Stage A transliteration of VIRTIM
(reference/voacapl/src/voacapw/virtim.for).

Collapses the diurnal Fourier dimension of the six blended coefficient
sets at one GMT hour: AB(JB,IZ) = a0 + sum_KA S(KA)*odd + C(KA)*even,
with the harmonics of TIME = (15*GMT - 180) degrees built by the
angle-addition recurrence.  All REAL*4.

Returns AB as a 1-based padded array of length 319 (index 0 unused) so
subscripts match the Fortran; ab[1:] is the /TWO_AB/ AB(318).
"""

import numpy as np

from voacap_gpu.stage_a.constants import D2R
from voacap_gpu.stage_a.f32math import F32, cos32, sin32
from voacap_gpu.stage_a.redmap import IA, IB, IC, IonoState


def virtim(st: IonoState, gmt) -> np.ndarray:
    time = (F32(15.0) * F32(gmt) - F32(180.0)) * D2R
    c = np.zeros(9, dtype=np.float32)
    s = np.zeros(9, dtype=np.float32)
    c[1] = cos32(time)
    s[1] = sin32(time)
    for jb in range(2, 9):  # DO 100
        c[jb] = c[1] * c[jb - 1] - s[1] * s[jb - 1]
        s[jb] = c[1] * s[jb - 1] + s[1] * c[jb - 1]

    cof = st.cofion  # COFION(2604), 0-based storage
    ab = np.zeros(319, dtype=np.float32)
    for iz in range(1, 7):  # DO 110
        j = int(st.ikim[9, iz - 1])       # IKIM(10,IZ): Fourier order
        i = int(st.ikim[8, iz - 1]) + 1   # IKIM(9,IZ)+1: geo functions
        for jb in range(1, i + 1):  # DO 105
            isubb = IA[iz - 1] + (jb - 1) * IB[iz - 1]
            isuba = IC[iz - 1] + jb - 1
            ab[isuba] = cof[isubb - 1]
            for ka in range(1, j + 1):
                isubc = isubb + 2 * ka - 1
                # AB = AB + S(KA)*COFION(ISUBC) + C(KA)*COFION(ISUBC+1)
                ab[isuba] = ((ab[isuba] + s[ka] * cof[isubc - 1])
                             + c[ka] * cof[isubc])
    return ab
