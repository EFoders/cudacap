"""SETLUF — pick the LUF from the frequency complement and the reliabilities.

The lowest usable frequency is the lowest one in the list whose reliability
reaches the required percentage.  The list is ascending, so the first hit
wins and the scan stops.  If nothing reaches it there is no LUF, and the
routine records the best frequency it saw with a NEGATIVE sign -- that is how
the caller and the output tables tell "the LUF is 7.2 MHz" from "nothing
worked, but 7.2 MHz came closest".  FREL(13) gets the magnitude either way.

The header says these values are overwritten by GETLUF when the LUF is
"actually computed"; this routine is the cheap version that reuses the
system-performance pass rather than iterating.

**The search is seeded from slot 1 whether or not slot 1 holds a frequency.**
RF and REL are set from FREL(1)/RELIAB(1) before the loop, and the loop skips
slots whose frequency is non-positive -- but not the seed.  So when slot 1 is
blank, RELIAB(1) (stale, from whatever last wrote /SON/) still sets the bar a
later frequency has to beat to become the fallback, and RF starts as a
non-positive "frequency" that can survive to be negated into XLUF.
"""

from .f32math import F32

_PCT = F32(0.01)
_NFREQ = 11
_I_LUF = 13


def setluf(muf, frq, son, lufp: int, it: int) -> None:
    """Fill /MUFS/ XLUF(it) and /FRQ/ FREL(13).  Mutates both in place.

    muf: MufState; frq: FrqState (reads FREL, writes FREL(13)); son: SonState
    (reads RELIAB); lufp: required reliability in percent, off the input card;
    it: hour index.
    """
    frel = frq.frel
    rf = frel[1]
    rel = son.reliab[1]
    pluf = _PCT * F32(lufp)
    for i in range(1, _NFREQ + 1):          # DO 200 IF = 1,11
        if frel[i] <= 0.0:                  # IF(FREL(IF)) 200, 200, 105
            continue
        if son.reliab[i] >= pluf:           # 105 -> 110: the LUF, and stop
            muf.xluf[it] = frel[i]
            frel[_I_LUF] = frel[i]
            return
        if son.reliab[i] > rel:             # 115 -> 120: a better fallback
            rel = son.reliab[i]
            rf = frel[i]
    # No frequency met the requirement: report the best one, negated.
    muf.xluf[it] = -rf
    frel[_I_LUF] = rf
