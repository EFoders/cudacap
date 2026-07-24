"""BABS -- absorption loss per kilometre for the long path's absorbing leg.

LNGPAT splits the ionospheric part of a long path in two: the stretch whose
ray passes THROUGH the absorbing D/E region, which is BABS's, and the stretch
that goes over the top of it, which is TABS's.  Both hand back a loss PER
KILOMETRE which LNGPAT multiplies back out by the leg length to get decibels.

The absorption is the model's standard non-deviative term -- 677.2 times an
absorption index over ``(f + fH)**1.98 + 10.2`` -- the same expression REGMOD
and ESMOD use for one short-path hop.  What turns it into a per-kilometre
figure is PORT: XMHOP, the number of hops the leg holds, divided by the leg
length DF*RZ, times SCPHE, the obliquity factor referred to 100 km.

BABS and TABS are the same twenty lines with ONE substitution -- BABS uses the
path-average absorption index ACAV where TABS hardcodes 0.1 -- and they
disagree about the 70 km boundary.  BABS's arithmetic IF sends HP exactly 70
to the LIVE arm; TABS's sends it to the dead arm.  A virtual height of exactly
70 km therefore gives a real absorption here and the 1000 marker there.
Transliterated; test_lngabs names it.

**The gyrofrequency index is hardcoded to sample area 3**, not taken from the
area the ray is in.  Area 3 is the path midpoint in every layout GEOM
produces (RD(3) = GCD/2 for both the three- and the five-area case, and
SETLNG copies area 1 upward when there is only one), so this is the mid-path
gyrofrequency by construction rather than by accident.

1000 is the "no absorption computed" marker and it is a loss per kilometre,
not a flat penalty: LNGPAT forms ``XLF = AF*DF*RZ``, so the DF <= 0 arm
contributes nothing at all while the HP < 70 arm contributes 1000 dB for
every kilometre of the leg.

float32; COS/ATAN/SQRT and the ``**1.98`` power carry the [[f32-libm-noise]]
floor.
"""

from voacap_gpu.scalar.constants import PIO2, RZ
from voacap_gpu.scalar.f32math import F32, atan32, cos32, pow32, sqrt32

_P198 = F32(1.98)           # a default-REAL literal, so powf, not a multiply
_NO_LOSS = F32(1000.0)


def babs(state, sigd, freq, del_, hp, df) -> F32:
    """BABS(DEL,HP,DF,AF) -- returns AF, the loss in dB per kilometre.

    state: LayerState (reads GYZ(3) only); sigd: SigdState (ACAV); freq in
    MHz; del_ the take-off angle in RADIANS; hp the virtual height in km; df
    the absorbing leg's arc in radians.
    """
    del_ = F32(del_)
    hp = F32(hp)
    df = F32(df)
    if df <= F32(0.0):                          # IF(DF) 105,105,110
        return _NO_LOSS
    if hp - F32(70.0) < F32(0.0):               # IF(HP-70.) 105,115,115
        return _NO_LOSS                         # ... so HP == 70 LIVES here
    # .....ABSORPTION INDEX
    af = sigd.acav
    # .....THIS IS THE F2 MODE LOSS FOR A FRACTIONAL NUMBER OF HOPS
    ab = F32(10.2) + pow32(F32(freq) + state.gyz[3], _P198)
    af = F32(677.2) * af / ab
    rcosd = RZ * cos32(del_)
    sphe = rcosd / (RZ + hp)
    phe = atan32(sphe / sqrt32(F32(1.0) - sphe * sphe))
    psi = PIO2 - phe - del_
    xmhop = F32(0.5) * df / psi
    sphe = rcosd / (RZ + F32(100.0))
    scphe = F32(1.0) / sqrt32(F32(1.0) - sphe * sphe)
    port = xmhop * scphe / (df * RZ)
    return af * port
