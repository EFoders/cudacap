"""Stage A transliteration of NOISY
(reference/voacapl/src/voacapw/noisy.for).

A general two-variable Fourier-series evaluator over (latitude, longitude).
KJ selects the coefficient map: 1-6 atmospheric noise, 7 land-mass map,
8 the F2 height-of-max / semithickness ratio (RAT) that TIMVAR needs.

The Fortran overlays one COMMON /TWO/ two ways: REDMAP fills it as
FAKP(29,16,6)/FAKMAP(29,16)/HMYM(29,16) and FAKABP(2,6)/ABMAP(2,3); NOISY
reads it as P(29,16,8) and ABP(2,9).  So P(:,:,7)=FAKMAP, P(:,:,8)=HMYM and
ABP(:,7..9)=ABMAP(:,1..3).  ``_two_p``/``_two_abp`` reproduce that overlay
from the IonoState REDMAP already produces.

Module 2 only exercises KJ=8; the atmospheric-noise slices (KJ=1-6) come
online with the Module 7 noise model.  REAL*4 throughout (SIN/COS carry the
[[f32-libm-noise]] floor).
"""

import numpy as np

from voacap_gpu.scalar.f32math import F32, cos32, sin32

_LON_HALF = F32(0.0087266466)   # D2R/2, longitude half-angle
_LAT_D2R = F32(0.01745329252)   # NOISY's own D2R literal (note trailing 2)


def _two_p(st, kj):
    """The KJ-th P(29,16) slice from the REDMAP overlay."""
    if kj <= 6:
        return st.fakp[:, :, kj - 1]
    if kj == 7:
        return st.fakmap
    return st.hmym  # kj == 8


def _two_abp(st, kj):
    """(ALF, BET) = ABP(1,KJ), ABP(2,KJ) from the REDMAP overlay."""
    if kj <= 6:
        return st.fakabp[0, kj - 1], st.fakabp[1, kj - 1]
    return st.abmap[0, kj - 7], st.abmap[1, kj - 7]


def noisy(st, kj: int, xla, ceg) -> np.float32:
    """XLA geographic latitude (deg), CEG east longitude (deg).  Returns the
    Fourier value ANOS.  st is the IonoState from redmap()."""
    p = _two_p(st, kj)          # (29,16), 0-based storage
    alf, bet = _two_abp(st, kj)
    alf = F32(alf)
    bet = F32(bet)

    if kj == 8:  # 100: limits of the Fourier series
        lm, ln = 15, 10
    else:  # 105
        lm, ln = 29, 15

    # 110: longitude half-angle and its sine harmonics
    q = _LON_HALF * F32(ceg)
    c1 = cos32(q)
    s1 = sin32(q)
    sx = np.zeros(16, dtype=np.float32)  # SX(15), 1-based
    sx[1] = s1
    cx = c1
    if ln >= 2:
        for k in range(2, ln + 1):  # DO 115
            tx = sx[k - 1]
            sx[k] = tx * c1 + cx * s1
            cx = cx * c1 - tx * s1  # 115

    zz = np.zeros(30, dtype=np.float32)  # ZZ(29), 1-based
    if lm >= 1:
        for j in range(1, lm + 1):  # DO 125
            r = F32(0.0)
            if ln >= 1:
                for k in range(1, ln + 1):  # DO 120
                    r = r + sx[k] * p[j - 1, k - 1]  # P(J,K,KJ)
            zz[j] = r + p[j - 1, 15]  # 125: + P(J,16,KJ)

    # 110->145: latitude series (angle + 90 deg)
    q = _LAT_D2R * (F32(xla) + F32(90.0))
    s1 = sin32(q)
    c1 = cos32(q)
    sx[1] = s1
    cx = c1
    r = F32(0.0)
    if lm >= 1:
        for k in range(1, lm + 1):  # DO 130 — in-place sin(k*Q) recurrence
            r = r + sx[1] * zz[k]
            ss = sx[1] * c1 + cx * s1
            cx = cx * c1 - sx[1] * s1
            sx[1] = ss  # 130
    # final Fourier value (note linear normalization ALF + BET*Q)
    return r + alf + bet * q
