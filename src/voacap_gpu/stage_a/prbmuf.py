"""PRBMUF -- probability the operating frequency exceeds a layer's MUF/FOT/HPF.

The program's standard probability evaluation: PRBMUF = 1 - FNORML(Z), with the
normal tail from a quartic Hastings rational approximation.  Z is the operating
frequency FMHZ's departure from FGO, scaled by the layer's decile spread taken
from the /MUFS/ distributions CURMUF set (SIGL below the median, SIGU above the
median, both relative to FSET).  When no median is placed (FSET<=0) it degrades
to a step at FGO.  IL selects the layer (1=E, 2=F1, 3=F2, 4=Es).  All float32,
mirroring prbmuf.for.
"""

import numpy as np

from voacap_gpu.stage_a.f32math import F32

# Hastings coefficients C(1..4) for the normal CDF tail approximation.
_C = (F32(0.196854), F32(0.115194), F32(0.000344), F32(0.019527))


def prbmuf(muf, fmhz, fgo, fset, il) -> F32:
    """muf: MufState (for SIGL/SIGU).  Returns the exceedance probability."""
    fmhz = F32(fmhz)
    fgo = F32(fgo)
    fset = F32(fset)
    z = fmhz - fgo
    if fset > 0.0:  # 95
        if z <= 0.0:
            sig = fgo * muf.sigl[il] / fset
        else:
            sig = fgo * muf.sigu[il] / fset
        sig = max(sig, F32(0.001))
        z = z / sig
        # PRBMUF = 1 - FNORML(Z); fold to |Z|, clamp for 32-bit range.
        ypx = z
        yp = -z if ypx < 0.0 else z
        yp = min(F32(5.0), yp)
        qx = F32(1.0) + yp * (_C[0] + yp * (_C[1] + yp * (_C[2] + yp * _C[3])))
        qx = qx * qx * qx * qx
        qx = F32(1.0) / qx
        qx = F32(0.5) * qx
        if ypx < 0.0:
            qx = F32(1.0) - qx
    else:  # 145: no median placed -> step at FGO
        qx = F32(1.0) if z <= 0.0 else F32(0.0)
    return qx
