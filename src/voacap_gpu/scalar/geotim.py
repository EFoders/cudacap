"""GEOTIM — put one prediction hour into every time base the model needs.

The ionosphere is driven by LOCAL time (the sun's position over each patch of
it), while the user thinks in universal time and the output tables print
both.  So each hour is resolved three ways plus once per sample area:

    GMT      universal time
    XLMT     local mean time at the transmitter
    GMTR     local mean time at the receiver, which ANOIS1 wants
    CLCK(i)  local mean time at each sample area, which TIMVAR wants

ITIM decides which of the first two the user typed.  Negative means the TIME
card gave local time at the transmitter, so the conversion runs backwards and
GMT is the derived quantity; otherwise the input is universal.

CNGDAY takes CNGTIM's day-change flag four times and is never read -- the
rollover is computed and thrown away.  Noted rather than transcribed.
"""

from .cngtim import cngtim
from .constants import R2D
from .f32math import F32


def geotim(layers, tm, jt: int, tlong, rlong) -> None:
    """Fill /TIME/ and /GEOG/ CLCK for hour `jt`.  Mutates `tm` and `layers`.

    layers: LayerState (reads KM and CLONG, writes CLCK); tm: TimeState
    (reads ITIM, writes the rest); jt: the requested hour; tlong/rlong:
    transmitter and receiver longitude in radians.
    """
    ckc = F32(jt)
    if tm.itim < 0:
        # .....LMT AT TRANSMITTER -- the hour given IS local, so derive UT
        temp = ckc
        ckc, _ = cngtim(ckc, tlong * R2D, -1)
        tm.gmt = ckc
    else:
        # .....GMT AT TRANSMITTER
        tm.gmt = ckc
        ckc, _ = cngtim(ckc, tlong * R2D, 1)
        temp = ckc

    it = int(tm.gmt)                        # REAL->INT truncates toward zero
    if it <= 0:                             # IF(IT) 150, 150, 160
        it = 24                             # midnight belongs to the day ending
    tm.it = it
    tm.utime[it] = tm.gmt
    tm.xlmt[it] = temp

    # .....LMT AT RECEIVER
    tm.gmtr, _ = cngtim(tm.gmt, rlong * R2D, 1)

    # .....LMT AT SAMPLE POINTS
    for ii in range(1, layers.km + 1):      # IF(KM) 190, 190, 170
        layers.clck[ii], _ = cngtim(tm.gmt, layers.clong[ii] * R2D, 1)
