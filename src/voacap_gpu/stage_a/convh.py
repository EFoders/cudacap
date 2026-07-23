"""CONVH -- the geometrical convergence factor for a long path.

A great-circle path is a ray bundle on a sphere, and past a quarter turn the
bundle stops spreading and starts CONVERGING again: energy launched in every
azimuth from the transmitter is funnelled back together as it approaches the
antipode.  CONVH is the decibel credit for that focusing, and LNGPAT subtracts
it from the free-space loss.

Two numbers come out.  PTOT is the group path -- two straight ray legs plus
the arc ``(RZ+HP)*GM`` at reflection height -- which sets both the free-space
loss and the time delay.  CH is the convergence, ``10*log10`` of the ratio of
the great-circle arc length to the ground spread ``sin(GD)``, clamped.

The clamps are the interesting part:

* ``DENOM = max(|sin(GD)|, 1e-6)``.  Exactly at the antipode sin(GD) is zero
  and the true convergence is infinite; the floor turns that into a large
  finite number.
* ``CH = min(CH, 15.)``, commented "AT ANTIPODAL" in the source, then caps it
  at 15 dB anyway.  So the 1e-6 floor never actually decides anything -- the
  15 dB ceiling bites first for any path within a few degrees of the antipode.
  Both are transliterated; only the ceiling is observable.
* ``GM = max(GD - 2*PSI, 0.001)``.  GM is the ionospheric arc left after the
  two end legs are removed, and it is floored rather than allowed to go
  negative on a path whose legs overrun it.

**PSI is associated differently here than in BABS and TABS.**  CONVH writes
``PIO2 - DEL - PHE`` where both absorption routines write ``PIO2 - PHE - DEL``.
Mathematically identical, and in float32 not always the same number, so the
order is preserved rather than normalised.

float32; SIN/COS/ALOG10 carry the [[f32-libm-noise]] floor.
"""

from voacap_gpu.stage_a.constants import PIO2, RZ
from voacap_gpu.stage_a.f32math import F32, cos32, log10_32, sin32


def convh(gd, phe, del_, hp, ray):
    """CONVH(GD,PHE,DEL,HP,RAY,CH,PTOT) -- returns ``(ch, ptot)``.

    gd: the great-circle angle in radians; phe: the incidence angle at the
    reflection height, radians; del_: the take-off angle, radians; hp: the
    virtual height in km; ray: the straight-line leg length in km.  CH is the
    convergence in dB, PTOT the group path in km.
    """
    gd = F32(gd)
    phe = F32(phe)
    del_ = F32(del_)
    hp = F32(hp)
    ray = F32(ray)
    denom = max(abs(sin32(gd)), F32(0.000001))
    psi = PIO2 - del_ - phe                     # note the order; see docstring
    gm = gd - F32(2.0) * psi
    gm = max(gm, F32(0.001))
    # .....GROUP PATH
    ptot = F32(2.0) * ray + (RZ + hp) * gm
    smallc = (ptot / RZ) * cos32(del_) / denom
    ch = F32(10.0) * log10_32(smallc)
    # .....AT ANTIPODAL
    ch = min(ch, F32(15.0))
    return ch, ptot
