"""Stage A transliteration of FOBBY
(reference/voacapl/src/voacapw/fobby.for).

Reflectrix table for sample area K.  For each restricted take-off angle
ANG(1..NANG) and each of the 30 vertical-sounding frequencies, it applies the
secant law using the TRUE reflection height (so no ionospheric K-factor) to
get the oblique operating frequency, stored as integer kHz in IFOB.  The
per-hop, per-angle table this builds is what FINDF later searches to place
each mode.  REAL*4; COS carries the [[f32-libm-noise]] floor and SQRT is
correctly rounded, so the only divergence from the oracle is the occasional
1-kHz flip when the COS floor nudges 1000*FREQ across an integer boundary.
"""

from voacap_gpu.stage_a.constants import D2R, RZ
from voacap_gpu.stage_a.f32math import F32, cos32, sqrt32


def fobby(state, rays, k):
    fvert, htrue = state.fvert, state.htrue
    ang, ifob = rays.ang, rays.ifob
    for ia in range(1, rays.nang + 1):  # DO 650 IA=1,NANG
        del_ = ang[ia] * D2R
        rcosd = RZ * cos32(del_)
        for ih in range(1, 31):  # DO 645 IH=1,30
            fv = fvert[ih, k]
            # true height so no K factor
            sphe = rcosd / (RZ + htrue[ih, k])
            sqcos = F32(1.0) - sphe * sphe
            sqcos = max(F32(0.000001), sqcos)  # AMAX1(.000001, SQCOS)
            cphe = sqrt32(sqcos)
            freq = fv / cphe
            ifob[ia, ih, k] = int(F32(1000.0) * freq)  # REAL->INT truncates
