"""Stage A transliteration of VERSY
(reference/voacapl/src/voacapw/versy.for).

Evaluates the geographic variation of the long-term maps at one sample
area: builds the (up to 76) geographic coordinate functions G — powers of
sin(x) plus cos(lat)^m * {cos,sin}(m*lon) blocks, x being the Rawer
magnetic dip except plain latitude for the regular-E set — and dots them
with the AB coefficients into GAMMA(IZ) for IZ = kik..lik
(1..3 Es low/median/up, 4 foF2, 5 M(3000)F2, 6 foE).

Precision: the Fortran declares ``real*8 sx,x,g,cx,t`` (an underflow fix
— cos(lat)^m underflows float32 near the poles), so exactly those are
float64 here; everything else (Y, GOB=COS(CENLAT), FJ, AB, GAMMA) stays
REAL*4, including the FJ*Y product that is computed in single precision
BEFORE widening into T, and the narrowing of GAMMA on every accumulation
statement.
"""

import numpy as np

from voacap_gpu.stage_a.constants import D2R
from voacap_gpu.stage_a.f32math import F32, F64, cos32
from voacap_gpu.stage_a.redmap import IC, IonoState


def versy(st: IonoState, ab: np.ndarray, clat, clong, gmdip,
          kik: int, lik: int) -> np.ndarray:
    """ab is the 1-based (319,) array from virtim; angles in radians
    (float32).  Returns gamma as a 1-based (7,) float32 array."""
    cenlg = F32(clong)
    cenlat = F32(clat)
    gmdip = F32(gmdip)
    c360 = F32(360.0) * D2R
    if cenlg < 0.0:  # 100: change to east longitude
        clg = c360 + cenlg
    else:  # 105
        clg = cenlg
    gob = cos32(cenlat)  # 110 — REAL*4 cosine

    gamma = np.zeros(7, dtype=np.float32)
    for iz in range(kik, lik + 1):  # DO 200
        if iz != 6:  # 115: Rawer magnetic dip angle
            x = F64(gmdip)
        else:  # 120: use latitude for regular E
            x = F64(cenlat)
        y = clg  # 125 — Y stays REAL*4
        i = int(st.ikim[8, iz - 1]) + 1  # IKIM(9,IZ)+1
        k = int(st.ikim[0, iz - 1])      # IKIM(1,IZ)
        sx = np.sin(x)  # DSIN (X is real*8)
        g = np.zeros(77, dtype=np.float64)
        g[1] = 1.0
        g[2] = sx
        if k != 1:  # 130: powers of sin
            if k >= 2:
                for ka in range(2, k + 1):  # DO 135
                    g[ka + 1] = sx * g[ka]
        # 140
        kdif = int(st.ikim[1, iz - 1]) - k  # IKIM(2,IZ) - K
        if kdif != 0:  # 145
            jg = 1
            cx = F64(gob)
            t = F64(y)
            while True:  # label 150
                kk = int(st.ikim[jg - 1, iz - 1]) + 4  # IKIM(JG,IZ)+4
                g[kk - 2] = cx * np.cos(t)  # DCOS
                g[kk - 1] = cx * np.sin(t)
                lo = int(st.ikim[jg, iz - 1])  # IKIM(JG+1,IZ)
                if kdif != 2:  # 155
                    if lo >= kk:  # 160
                        for ka in range(kk, lo + 1, 2):  # DO 170
                            g[ka] = sx * g[ka - 2]       # 165
                            g[ka + 1] = sx * g[ka - 1]   # 170
                # 175
                if jg == 8:  # IF (JG-8) 180,190,180
                    break
                kdif = int(st.ikim[jg + 1, iz - 1]) - lo  # IKIM(JG+2,IZ)
                if kdif == 0:  # 190
                    break
                cx = cx * F64(gob)  # 185
                jg = jg + 1
                fj = F32(jg)
                t = F64(fj * y)  # REAL*4 product, then widened into T
        # 190: final summation of the Fourier expansion
        isuba = IC[iz - 1]
        gamma[iz] = F32(g[1] * F64(ab[isuba]))
        if i >= 2:
            for jb in range(2, i + 1):  # DO 195
                isuba = IC[iz - 1] + jb - 1
                # GAMMA = GAMMA + AB(ISUBA)*G(JB): f64 sum, REAL*4 store
                gamma[iz] = F32(F64(gamma[iz]) + F64(ab[isuba]) * g[jb])
    return gamma
