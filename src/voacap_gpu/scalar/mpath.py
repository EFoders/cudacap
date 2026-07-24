"""MPATH -- multipath probability: will another mode interfere?

Not ray geometry, despite the name -- that was FDIST's job and ran much
earlier.  This is the last analysis step of the frequency loop: given the most
reliable mode RELBIL chose, how likely is it that some *other* mode arrives
badly enough to matter?

A mode qualifies as interference when all three hold:

* it is not the most reliable mode, and it exists (HP > 0);
* its delay differs from the MRM's by MORE than the operator's tolerable
  delay DMP -- a mode arriving close in time is not multipath, it just adds;
* its signal power is within PMP dB of the MRM's, so it is strong enough to
  matter at all.

The answer is the highest reliability among those, or 0.001 if none qualify.

**The delay test reads backwards from the source's own header comment**,
which says the routine returns the next most probable mode "that is within
the Maximum tolerable time delay (DMP)".  The code skips modes within DMP
and keeps those outside it -- which is what the physics wants, and what the
inline comment on that line ("inside acceptable time delay", on a skip)
agrees with.  The header comment is the thing that is wrong; the code is
transliterated as written.

Both DMP and PMP non-positive means no multipath was requested, and the
routine returns having set only the 0.001 default.
"""

from voacap_gpu.scalar.f32math import F32

_NO_MULTIPATH = F32(0.001)


def mpath(allm, son, ifx: int, dmp, pmp, gcdkm, method: int, mspec: int
          ) -> None:
    """Fill /SON/ PROBMP(ifx).  Mutates `son` in place.

    allm: AllModeState (/allMODE/, including NREL from RELBIL);
    son: SonState, the output.  ifx: frequency index; dmp: maximum tolerable
    delay (ms) and pmp: multipath power tolerance (dB), both from /DON/ and
    both operator-set; gcdkm: path length; method (/ION/) and mspec
    (/CONTRL/) gate the area-coverage exclusion below.
    """
    son.probmp[ifx] = _NO_MULTIPATH
    if dmp <= 0.0 or pmp <= 0.0:
        return                              # no multipath requested
    # METHOD 30's area variant is not valid beyond 7000 km
    if method == 20 and mspec == 121 and gcdkm > F32(7000.0):
        return
    nrel = allm.nrel
    sig_power = allm.sigpow[nrel]
    sig_power_limit = sig_power - F32(pmp)
    ttim = allm.timed[nrel]                 # time delay of the MRM

    for im in range(1, allm.nmmod + 1):     # DO 135
        if im == nrel:
            continue                        # RELBIL already found this one
        if allm.hp[im] <= 0.0:
            continue                        # this mode does not exist
        if abs(allm.timed[im] - ttim) <= F32(dmp):
            continue                        # arrives close enough to just add
        if allm.sigpow[im] < sig_power_limit:
            continue                        # too weak to interfere
        son.probmp[ifx] = max(son.probmp[ifx], allm.rely[im])
