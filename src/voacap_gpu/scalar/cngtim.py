"""CNGTIM — universal time to local mean time at a reference longitude.

Fifteen degrees of longitude is one hour, so the conversion is a division and
a wrap.  What makes it more than that is the sign convention: the offset has
to come out in (-12, +12] rather than [0, 24), or a station just west of the
date line would be reported a day out.  The routine gets there by measuring
the longitude in half-turns and full turns:

    THETA in [0, 180)     offset = THETA/15            east of Greenwich
    THETA in [180, 360)   offset = THETA/15 - 24       ... read as west
    THETA = 360           offset = THETA/15 - 48 + 24  = 0, i.e. Greenwich

ISW picks the direction: +1 converts UT to local, -1 converts local back.

The return value is the day change (-1, 0 or +1) that the conversion implies.
**GEOTIM, its only caller in this path, assigns that to CNGDAY and never
reads it** — so the rollover is computed and discarded, and a prediction hour
does not carry a date.

Two details worth stating rather than leaving to be rediscovered.

**A reference longitude of exactly zero returns immediately**, before the
normalising loop, so the time comes back untouched rather than wrapped into
[0, 24].  A station at Greenwich needs no conversion, which is right, but it
also skips the range fix that every other longitude gets.

**The epsilon in ``(TIME - 23.99999999)/24.0`` does nothing.**  In single
precision that constant IS 24.0 -- the spacing of float32 near 24 is about
1.9e-6, two hundred times larger than the 1e-8 the author was reaching for.
The guard was written in double-precision thinking and compiled away.  The
arithmetic is unaffected either way; transliterated as written.
"""

from .f32math import F32

_ALMOST_24 = F32(23.99999999)   # ... which is exactly 24.0; see the docstring
_DAY = F32(24.0)
_TURN = F32(360.0)
_HALF_TURN = F32(180.0)
_DEG_PER_HOUR = F32(15.0)


def cngtim(time, xtheta, isw: int):
    """Returns ``(converted time, day change)``.

    time: hours; xtheta: reference longitude in degrees east, negative for
    west; isw: +1 universal -> local, -1 local -> universal.  The Fortran
    writes the time back through its dummy argument; here it is returned.
    """
    time = F32(time)
    xtheta = F32(xtheta)
    if xtheta == 0.0:                       # IF(XTHETA) 80, 75, 85
        return time, 0
    theta = _TURN + xtheta if xtheta < 0.0 else xtheta

    int1 = int(theta / _HALF_TURN)          # REAL->INT truncates toward zero
    int2 = int(theta / _TURN)
    fint1 = F32(int1)
    fint2 = F32(int2)
    fisw = F32(isw)
    time = time + fisw * (theta / _DEG_PER_HOUR
                          - fint1 * _DAY + fint2 * _DAY)

    # The day change, before the time is normalised.  INT1/INT2 are reused
    # here, which is why FINT1/FINT2 were captured above.
    day = int((time - _ALMOST_24) / _DAY) + int(time / _DAY)

    if time < 0.0:
        time = time + _DAY
    time = time - fint2 * _DAY
    while True:                             # 110 -- wrap into [0, 24]
        if time < 0.0:
            time = time + _DAY
            continue
        if time > _DAY:
            time = time - _DAY
            continue
        return time, day
