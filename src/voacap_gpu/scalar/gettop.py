"""GETTOP -- how much of a long path the ray flies OVER the E layer.

The night-day-night case.  A long path can have a daylit middle whose E layer
reflects a frequency that the dark ends let straight through.  For that
stretch the ray never reaches the F region at all: it is turned by the E
layer, and it never sees the D-region absorption underneath.  GETTOP measures
the stretch; TABS then prices it at a tenth of the usual absorption index.

The three sample areas are read at the E layer only -- HI(1,jf) and FI(1,jf)
-- and the code names areas 1 and 3 "END" and area 2 "MIDDLE", matching
SELMOD's "three samples but check ends only".  Each area's vertical E
critical is projected to the oblique penetration frequency at DELOPT, the
a-priori 3 degree take-off angle: FPE = fv/sqrt(1 - sin^2 phi), with the
secant clamped at sin^2 = 0.9999 so a grazing ray cannot divide by zero.

Then five tests in a row, every one of which can zero the answer:

    FP1 < FP2       the middle must penetrate HIGHER than either end
    FREQ < FP2+.001 ... and the frequency must be below the middle's
    FREQ > FP1-.001 ... and above the higher end's
    GMKM > 1000     the over-the-top stretch must be worth having
    DIKM-GMKM>1000  and enough path must be left underneath it

The last one is commented "MUST BE ABLE TO GET UP THERE FOR M MODE": there is
no point in an over-the-top segment that consumes the whole path, because the
ray still has to climb from the ground at each end.

**The two 0.001 MHz slacks pull in opposite directions, and only one of them
does anything.**  ``FREQ-FP2-.001`` widens the window at the top and
``FREQ-FP1+.001`` widens it at the bottom, so the frequency is admitted from a
kilohertz outside the bracket at either end.  But above FP2 the interpolation's
``(FP2-FREQ)`` factor is NEGATIVE, so everything the top slack admits produces
a negative GMKM that the next test throws away: the top slack cannot change an
answer.  The bottom one can.  Both are transliterated; test_gettop names it.

GMKM is written on every path through the routine, so nothing here reads an
uninitialised output.  The interpolation ``0.5*DIKM*(FP2-FREQ)*(1/(FP2-FP1) +
1/(FP2-FP3))`` cannot divide by zero either: FP2 > FP1 >= FP3 is exactly what
the first test established.

float32; COS and SQRT carry the [[f32-libm-noise]] floor.
"""

from voacap_gpu.scalar.constants import D2R, RZ
from voacap_gpu.scalar.f32math import F32, cos32, sqrt32

_E_LAYER = 1                # HI(1,*)/FI(1,*): the E layer, always
_MIN_TOP = F32(1000.0)      # km, for both the segment and what is left over


def gettop(state, lpath, freq, dikm) -> F32:
    """GETTOP(DIKM,GMKM) -- returns GMKM, the over-the-top distance in km.

    state: LayerState (reads HI(1,1..3) and FI(1,1..3)); lpath: LpathState
    (DELOPT); freq in MHz; dikm the ionospheric distance in km.  Returns 0
    when no over-the-top segment qualifies.
    """
    freq = F32(freq)
    dikm = F32(dikm)
    del_ = lpath.delopt * D2R
    rcosd = RZ * cos32(del_)
    fpe = [F32(0.0)] * 4                    # 1-based, FPE(3)
    #  FIND DISTANCE THAT SUPPORT  E (M)  MODE.
    for jf in (1, 2, 3):
        ht = state.hi[_E_LAYER, jf]
        fv = state.fi[_E_LAYER, jf]
        sphe = rcosd / (RZ + ht)
        sphe = sphe * sphe
        sphe = min(sphe, F32(0.9999))
        # .....E PENETRATION FREQUENCIES
        fpe[jf] = fv / sqrt32(F32(1.0) - sphe)
    fp1 = max(fpe[1], fpe[3])               # .....END
    fp2 = fpe[2]                            # .....MIDDLE
    fp3 = min(fpe[1], fpe[3])               # .....END
    # .....THE MIDDLE IS --
    if fp1 - fp2 >= F32(0.0):               # IF(FP1-FP2) 170,235,235
        return F32(0.0)
    if freq - fp2 - F32(0.001) >= F32(0.0):  # IF(FREQ-FP2-.001) 175,235,235
        return F32(0.0)
    if freq - fp1 + F32(0.001) <= F32(0.0):  # IF(FREQ-FP1+.001) 235,235,185
        return F32(0.0)
    gmkm = F32(0.5) * dikm * (fp2 - freq) * (
        F32(1.0) / (fp2 - fp1) + F32(1.0) / (fp2 - fp3))
    if gmkm - _MIN_TOP <= F32(0.0):         # IF(GMKM-1000.) 235,235,225
        return F32(0.0)
    # .....MUST BE ABLE TO GET UP THERE FOR M MODE
    if dikm - gmkm - _MIN_TOP <= F32(0.0):  # IF(DIKM-GMKM-1000.) 235,235,240
        return F32(0.0)
    return gmkm
