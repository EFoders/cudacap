"""Stage A transliteration of GENION
(reference/voacapl/src/voacapw/genion.for).

Generates the ionogram for sample area K.  First it lays out 30 vertical-
sounding frequencies FVERT across the D-E tail, the E nose, and the F region
from the E/F1/F2 critical frequencies FI(1/2/3,K) (with a different spacing
depending on whether an F1 layer is present).  Then it fills the virtual
(HPRIM) and true (HTRUE) reflection heights at each frequency, by one of two
models selected by IEDP:

  * IEDP < 0  : full NT-point Gaussian integration of the profile (GETHP) at
                every frequency (label 600);
  * IEDP >= 0 : the fast model (label 705) — a fixed 10-point parabolic-E
                table (HTE/HPE), true heights by table lookup (XLIN) into the
                LECDEN profile, and virtual heights from the BENDY/PEN
                parabolic-F formula when no F1 layer is present, else GETHP.

REAL*4; SQRT is correctly rounded, and the BENDY/PEN/GETHP callees carry the
documented [[f32-libm-noise]] ALOG floor.  Mutates FVERT/HPRIM/HTRUE in /RON/.
"""

import numpy as np

from voacap_gpu.stage_a.bendy import bendy
from voacap_gpu.stage_a.f32math import F32, sqrt32
from voacap_gpu.stage_a.gethp import gethp
from voacap_gpu.stage_a.pen import pen
from voacap_gpu.stage_a.xlin import xlin

# True (HTE) and virtual (HPE) heights for the E layer with HI=110, YI=20 as
# parameters — the fixed 10-point table used by the fast model (DATA HTE/HPE
# in genion.for), 1-based (index 0 padded).
_HTE = np.array([0.0, 70.00, 84.51, 88.52, 93.00, 94.26, 95.71, 97.68,
                 100.27, 104.22, 107.19], dtype=np.float32)
_HPE = np.array([0.0, 70.00, 87.40, 94.00, 98.68, 100.20, 102.65, 107.15,
                 113.74, 125.25, 140.03], dtype=np.float32)


def genion(state, k):
    fi, yi, hi = state.fi, state.yi, state.hi
    fvert, hprim, htrue = state.fvert, state.hprim, state.htrue

    # ---- the 30 vertical-sounding frequencies -----------------------------
    # D-E region tail (XTAIL must agree with LECDEN)
    xtail = F32(0.85)
    fex = fi[1, k] * sqrt32(F32(1.0) - xtail * xtail)
    fvert[1, k] = F32(0.01)
    fvert[4, k] = fex
    fdif = (fvert[4, k] - fvert[1, k]) / F32(3.0)
    fvert[2, k] = fvert[1, k] + fdif
    fvert[3, k] = fvert[2, k] + fdif
    # E region nose (guess for E MUF)
    fvert[9, k] = F32(0.957) * fi[1, k]
    fvert[10, k] = F32(0.99) * fi[1, k]
    fdif = (fvert[9, k] - fvert[4, k]) / F32(5.0)
    fvert[5, k] = fvert[4, k] + fdif
    fvert[6, k] = fvert[5, k] + fdif
    fvert[7, k] = fvert[6, k] + fdif
    fvert[8, k] = fvert[7, k] + fdif
    # E-F cusp
    fvert[11, k] = F32(1.05) * fi[1, k]
    # F region nose
    fvert[30, k] = F32(0.99) * fi[3, k]
    fvert[29, k] = F32(0.98) * fi[3, k]
    fvert[28, k] = F32(0.96) * fi[3, k]
    fvert[27, k] = F32(0.92) * fi[3, k]
    if fi[2, k] <= 0.0:  # IF(FI(2,K)) 385,385,400 -> F2 layer, no F1 layer
        fdif = (fvert[27, k] - fvert[11, k]) / F32(16.0)
        for iff in range(12, 27):  # DO 395 IF=12,26
            fvert[iff, k] = fvert[iff - 1, k] + fdif
    else:  # F1 layer and F2 layer
        fvert[20, k] = F32(0.99) * fi[2, k]
        fdif = (fvert[20, k] - fvert[11, k]) / F32(9.0)
        for iff in range(12, 20):  # DO 405 IF=12,19
            fvert[iff, k] = fvert[iff - 1, k] + fdif
        # F1-F2 cusp
        fvert[21, k] = F32(1.01) * fi[2, k]
        fdif = (fvert[27, k] - fvert[21, k]) / F32(6.0)
        for iff in range(22, 27):  # DO 415 IF=22,26
            fvert[iff, k] = fvert[iff - 1, k] + fdif

    # ---- virtual/true heights ---------------------------------------------
    if state.iedp < 0:  # 600: integrate to get ionogram, NT-point Gaussian
        for iff in range(1, 31):  # DO 640 IF=1,30
            hpx, htx = gethp(state, fvert[iff, k])
            hprim[iff, k] = hpx
            htrue[iff, k] = htx
        return
    # 705: faster version, parabolic E layer with exponential tail
    for iff in range(1, 11):  # DO 710 IF=1,10 (same as GETHP for default E)
        htrue[iff, k] = _HTE[iff]
        hprim[iff, k] = _HPE[iff]
    for iff in range(11, 31):  # DO 715 IF=11,30
        fn = fvert[iff, k] * fvert[iff, k]
        # true height by table lookup
        htrue[iff, k] = xlin(fn, state.fnsq, state.htr, 50, 1)
    if fi[2, k] <= 0.0:  # 720: no F1
        for iff in range(11, 31):  # DO 725 IF=11,30
            fn = fvert[iff, k]
            # bottom of F layer plus bending plus retardation
            hprim[iff, k] = (hi[3, k] - yi[3, k] + bendy(state, 3, k, fn)
                             + (pen(state, 1, k, fn) - F32(2.0) * yi[1, k]))
    else:  # 730: F1 present -- too many possibilities, integrate
        for iff in range(11, 31):  # DO 740 IF=11,30
            hpx, _htx = gethp(state, fvert[iff, k])
            hprim[iff, k] = hpx
