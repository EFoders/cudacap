"""Stage A transliteration of TIMVAR
(reference/voacapl/src/voacapw/timvar.for).

Per sample area: local mean time, solar zenith angle, then EF1VAR for the
E/F1 parameters, the F2 height ratio RAT from the NOISY(8) map, and the
E-layer absorption index ABIY.  REAL*4 throughout (SIN/COS/ACOS/EXP carry
the [[f32-libm-noise]] floor).
"""

from voacap_gpu.scalar.constants import D2R, R2D
from voacap_gpu.scalar.ef1var import ef1var
from voacap_gpu.scalar.f32math import F32, acos32, cos32, exp32, sin32
from voacap_gpu.scalar.layerstate import SUN
from voacap_gpu.scalar.noisy import noisy

_HR2RAD = F32(0.261799387)  # 15 deg in radians (longitude -> hour scaling)


def timvar(state, st, ab, gmt):
    """state: LayerState; st: IonoState; ab: virtim output; gmt hour."""
    gmt = F32(gmt)
    month = state.month
    ssl = (F32(180.0) - F32(15.0) * gmt) * D2R
    if state.km < 1:  # 160
        return
    for ii in range(1, state.km + 1):  # DO 155
        cenlg = state.clong[ii]
        cenlat = state.clat[ii]
        cendog = F32(abs(cenlat - D2R * SUN[1, month]))
        ssp = SUN[2, month] * D2R
        cencat = F32(abs(cenlat - ssp))
        if cendog - cencat > 0.0:  # 100 (else 105 keeps SUN(2))
            ssp = SUN[1, month] * D2R
        # 105: local mean time, wrapped into [0,24)
        clock = gmt + cenlg / _HR2RAD
        if F32(24.0) - clock < 0.0:  # 110
            clock = clock - F32(24.0)
        elif clock <= 0.0:  # 115 -> 120
            clock = clock + F32(24.0)
        state.clck[ii] = clock  # 125
        z = cenlg - ssl
        cycen = (sin32(cenlat) * sin32(ssp)
                 + cos32(cenlat) * cos32(ssp) * cos32(z))
        cycen = acos32(cycen)  # zenith angle
        cycen = ef1var(state, st, ab, cycen, ii)  # E and F1 parameters
        gm = F32(abs(state.glat[ii])) * R2D - F32(45.0)
        z = cycen * R2D
        if clock - F32(12.0) < 0.0:  # 135
            z = -z
        z = z + F32(180.0)  # 140
        ym = noisy(st, 8, gm, z)      # RAT map
        if ym - F32(2.0) < 0.0:  # 145
            ym = F32(2.0)
        state.rat[ii] = ym  # 150
        abi = F32(-0.04) + exp32(F32(-2.937) + F32(0.8445) * state.fi[1, ii])
        state.abiy[ii] = abi
