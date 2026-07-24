"""Stage A transliteration of DISTXY
(reference/voacapl/src/voacapw/distxy.for).

Places the ndistance evenly spaced sample receivers along the great
circle from the original receiver back to the transmitter (used by the
distance-sweep outputs): idistance=1 is the receiver itself,
idistance=ndistance the transmitter, and points between interpolate by
scaling DAZEL's great-circle distance and re-solving mode 1.

Interface REAL*4 (float32); angles arrive in radians and leave in
degrees, using the /CON/ R2D (which differs from DAZEL's internal RTOD).
Mutates the passed Azel (COMMON /AZEL/) exactly like the Fortran; npsl is
in the argument list but unused, kept for signature fidelity.
"""

from voacap_gpu.scalar.constants import R2D
from voacap_gpu.scalar.dazel import Azel, dazel
from voacap_gpu.scalar.f32math import F32


def distxy(z: Azel, idistance: int, ndistance: int,
           tlat, tlon, rlat_dist, rlon_dist, npsl: int = 0):
    """Returns (rlatd, rlongd) in degrees, rlongd normalized to [0, 360]."""
    z.ztlat = F32(F32(tlat) * R2D)
    z.ztlon = F32(F32(tlon) * R2D)
    z.zrlat = F32(F32(rlat_dist) * R2D)
    z.zrlon = F32(F32(rlon_dist) * R2D)
    if idistance == 1:  # Set Rx to Tx
        rlatd = z.zrlat
        rlongd = z.zrlon
    elif idistance == ndistance:  # Set Rx to original Rx
        rlatd = z.ztlat
        rlongd = z.ztlon
    else:  # interpolate in between
        dazel(z, 0)  # distance Tx to Rx
        z.zdgc = F32(z.zdgc * F32(ndistance - idistance)
                     / F32(ndistance - 1))
        dazel(z, 1)  # calc new Rx
        rlatd = z.zrlat
        rlongd = z.zrlon
    # fix up RLONG so it is between 0 & 360
    if rlongd < 0.0:
        rlongd = rlongd + F32(360.0)
    if rlongd > 360.0:
        rlongd = rlongd - F32(360.0)
    return rlatd, rlongd
