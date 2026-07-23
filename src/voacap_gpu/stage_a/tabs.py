"""TABS -- absorption loss per kilometre for the leg that misses the D region.

BABS's twin, for the part of a long path whose ray goes OVER the absorbing
region rather than through it.  GETTOP measures that stretch; TABS prices it.

Physically the two differ in one number: the absorption index.  BABS uses the
path average ACAV, TABS a hardcoded ``XI = 0.1`` -- an over-the-top ray still
grazes some ionisation, but a fixed tenth of it rather than whatever the
D region happens to hold.  Everything after that line is character for
character the same geometry as BABS.

Two things about the guard are worth pinning:

* **the arithmetic IF is ordered the other way round.**  BABS tests DF first
  and falls into the dead label from below; TABS tests DM first and falls into
  the dead label from ABOVE, with the live code after it.  Same effect,
  mirrored source.
* **HP exactly 70 km is DEAD here and LIVE in BABS.**  ``IF(HP-70.) 110,110,115``
  puts the zero case with the low branch; BABS's ``IF(HP-70.) 105,115,115``
  puts it with the high one.  Since LNGPAT calls both with the same HPX, a
  path whose averaged virtual height lands on 70.0 gets a real absorbing-leg
  loss and a 1000 dB over-the-top loss.  Not a typo we may fix -- test_lngabs
  asserts it.

Like BABS, the gyrofrequency comes from sample area 3, hardcoded: the path
midpoint in every layout GEOM produces.

float32; COS/ATAN/SQRT and the ``**1.98`` power carry the [[f32-libm-noise]]
floor.
"""

from voacap_gpu.stage_a.constants import PIO2, RZ
from voacap_gpu.stage_a.f32math import F32, atan32, cos32, pow32, sqrt32

_P198 = F32(1.98)           # a default-REAL literal, so powf, not a multiply
_XI = F32(0.1)              # the over-the-top absorption index, fixed
_NO_LOSS = F32(1000.0)


def tabs(state, freq, del_, hp, dm) -> F32:
    """TABS(DEL,HP,DM,AM) -- returns AM, the loss in dB per kilometre.

    state: LayerState (reads GYZ(3) only); freq in MHz; del_ the take-off
    angle in RADIANS; hp the virtual height in km; dm the over-the-top leg's
    arc in radians.  Takes no SigdState: XI replaces ACAV.
    """
    del_ = F32(del_)
    hp = F32(hp)
    dm = F32(dm)
    if dm <= F32(0.0):                          # IF(DM) 110,110,105
        return _NO_LOSS
    if hp - F32(70.0) <= F32(0.0):              # IF(HP-70.) 110,110,115
        return _NO_LOSS                         # ... so HP == 70 DIES here
    # .....ABSORPTION INDEX  (XI, not ACAV -- the only difference from BABS)
    # .....THIS IS THE F2 MODE LOSS FOR A FRACTIONAL NUMBER OF HOPS
    ab = F32(10.2) + pow32(F32(freq) + state.gyz[3], _P198)
    am = F32(677.2) * _XI / ab
    rcosd = RZ * cos32(del_)
    sphe = rcosd / (RZ + hp)
    phe = atan32(sphe / sqrt32(F32(1.0) - sphe * sphe))
    psi = PIO2 - phe - del_
    xmhop = F32(0.5) * dm / psi
    sphe = rcosd / (RZ + F32(100.0))
    scphe = F32(1.0) / sqrt32(F32(1.0) - sphe * sphe)
    port = xmhop * scphe / (dm * RZ)
    return am * port
