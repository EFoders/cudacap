"""FNORML -- the cumulative normal distribution, AMS 55 26.2.17.

A degree-4 polynomial raised to the fourth power, chosen (per the source
comment) so it stays accurate in 32-bit arithmetic on 16-bit machines.  The
argument is folded to |x| and clamped at 5 sigma before the polynomial, then
the result is reflected back.

The caller almost always wants the upper tail, and writes it as
``1. - FNORML(Z)``.  That is NOT the same as evaluating the tail directly:
for Z >= 0 the Fortran computes ``PX = 1 - QX`` and the caller then computes
``1 - PX``, so the value round-trips through two float32 subtractions and
does not in general return to QX.  Callers here therefore spell out
``F32(1.0) - fnorml(z)`` rather than shortcutting it.

PRBMUF inlines this same polynomial rather than calling it (prbmuf.for says
so in a comment); that copy is validated separately and is left alone.
"""

from voacap_gpu.stage_a.f32math import F32

_C = (F32(0.196854), F32(0.115194), F32(0.000344), F32(0.019527))
_CLAMP = F32(5.0)


def fnorml(ypx) -> F32:
    """P(X <= ypx) for a standard normal X."""
    ypx = F32(ypx)
    yp = ypx
    if ypx < 0.0:               # IF(YPX) 105,110,110
        yp = -ypx
    yp = min(_CLAMP, yp)
    qx = F32(1.0) + yp * (_C[0] + yp * (_C[1] + yp * (_C[2] + yp * _C[3])))
    qx = qx * qx * qx * qx
    qx = F32(1.0) / qx
    qx = F32(0.5) * qx
    if ypx < 0.0:               # IF(YPX) 115,120,120
        px = qx
        # the Fortran then sets QX = 1. - PX, which nothing reads
    else:
        px = F32(1.0) - qx
    return px
