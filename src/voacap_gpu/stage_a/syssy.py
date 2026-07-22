"""SYSSY -- table lookup for excess-system-loss deciles and prediction errors.

Indexes the CCIR SYS(9,16,6) and PERR(9,4,6) tables (loaded by REDMAP into
/TWO/) by geomagnetic latitude G (radians) and reflection-area local time PT,
with the reflection-area distance index NN (2 short, 5 long).  Returns the
median excess system loss FM, its upper/lower deciles SU/SL, and the
prediction-error median FMP and deciles SUP/SLP.  When no coefficient data is
loaded (F2D(1,1,1) <= 0) it returns the fixed ITSA-1 fallback values.

The KJ (latitude), LOL (local-time block, hemisphere-shifted by KK) and LD/ND
(prediction-error block/offset) indices are integer truncations of REAL*4
expressions; the arithmetic is kept in float32 so boundary cases truncate
exactly as the Fortran does.  Mirrors syssy.for (with SYS/PERR 0-based).
"""

import numpy as np

from voacap_gpu.stage_a.f32math import F32

# SYSSY's own coarse degrees-per-radian (57.29577), NOT the /CON/ R2D.
_RAD2DEG = F32(5.729577e1)


def syssy(st, g, pt, nn):
    """st: IonoState (SYS/PERR/F2D).  Returns (fm, su, sl, fmp, sup, slp)."""
    g = F32(g)
    pt = F32(pt)
    if st.f2d[0, 0, 0] <= 0.0:  # IF(F2D(1,1,1)) 90,90,95 -> no data tape
        fm, dl, du = F32(9.0), F32(4.0), F32(9.0)
        fmp, sup, slp = F32(5.6), F32(0.60), F32(1.63)
        su = du / F32(1.28)
        sl = dl / F32(1.29)
        return fm, su, sl, fmp, sup, slp
    j = int(pt + F32(0.5))
    gg = abs(g) * _RAD2DEG
    kj = int((gg - F32(40.0)) / F32(5.0) + F32(1.5))
    if kj <= 0:        # IF(KJ) 100,100,105
        kj = 1
    elif kj > 9:       # 105  IF(KJ-9) 115,115,110
        kj = 9
    if g < 0.0:        # IF(G) 125,120,120 -> southern hemisphere
        kk, nd = 8, 3
    else:
        kk, nd = 0, 0
    lj = int(F32(j) / F32(3.0) + F32(0.67))
    if lj - 1 < 0:     # IF(LJ-1) 135,140,140
        lj = 8
    lol = lj + kk
    ld = int(F32(j) / F32(6.0) + F32(0.67))
    if ld - 1 < 0:     # IF(LD-1) 145,150,150
        ld = 4
    fm = st.sys[kj - 1, lol - 1, nn - 1]
    du = st.sys[kj - 1, lol - 1, nn]      # SYS(KJ,LOL,NN+1)
    dl = st.sys[kj - 1, lol - 1, nn - 2]  # SYS(KJ,LOL,NN-1)
    fmp = st.perr[kj - 1, ld - 1, nd]     # PERR(KJ,LD,ND+1)
    sup = st.perr[kj - 1, ld - 1, nd + 1]  # PERR(KJ,LD,ND+2)
    slp = st.perr[kj - 1, ld - 1, nd + 2]  # PERR(KJ,LD,ND+3)
    su = du / F32(1.28)
    sl = dl / F32(1.28)
    return fm, su, sl, fmp, sup, slp
