"""SERPRB -- service probability: reliability with the prediction error in it.

RELBIL asks "how often does this circuit work?" given the model.  SERPRB asks
the harder question: "how confident are we in that answer?"  It rebuilds the
same normal-distribution test but widens the spread by everything that is
uncertain about the *prediction* rather than about the hour -- the system-loss
prediction error ADS/SUS/SLS, the noise prediction error SIGM/SYGU/SYGL, and a
flat 2 dB for the required S/N itself.

Which tail is used depends on the required reliability: above 50% the binding
case is a bad hour (noise upper decile, signal lower decile) and XLH is -1, so
the reference level moves DOWN from the median; below 50% it is the other way
round and XLH is +1.

The source's own header notes it "has not been revised to use summed signal",
i.e. it works per mode off /allMODE/ where RELBIL now works off the combined
distribution, and then takes the maximum.  That inconsistency is real and is
preserved.

Two more things preserved as found: ``SPR`` in /TON/ is zeroed and never
touched again, and ``D90S`` is set equal to ``D10S`` rather than to an upper
decile of its own.

float32; SQRT and FNORML's polynomial carry the [[f32-libm-noise]] floor.
"""

from voacap_gpu.stage_a.constants import DCL
from voacap_gpu.stage_a.f32math import F32, sqrt32
from voacap_gpu.stage_a.fnorml import fnorml

# The same standard-normal table RELBIL has, as a separate DATA statement in
# the Fortran; SERPRB uses TME(ITM) directly where RELBIL takes a ratio.
_TME = (F32(0.0), F32(0.1257), F32(0.2533), F32(0.3853), F32(0.5244),
        F32(0.6745), F32(0.8416), F32(1.0364), F32(1.2815), F32(1.6449))
_DR = F32(2.0)              # prediction error in the required S/N, dB
_NO_MODE_HEIGHT = F32(70.0)


class NoModesError(ValueError):
    """SERPRB with no modes reads uninitialised locals in the Fortran."""


def serprb(allm, anois, ton, son, ifx: int, rsn, lufp: int) -> None:
    """Fill /SON/ SPROB(ifx) and /allMODE/ SPRO.  Mutates in place.

    allm: AllModeState (/allMODE/); anois: AnoisState (/ANOIS/);
    ton: TonState (/TON/ ADS/SUS/SLS in, SPR out); son: SonState (SPROB out,
    plus the /DON/ D10S/D50S/D90S triple).  ifx: frequency index;
    rsn: required S/N (/DON/); lufp: required reliability, percent (/ION/).
    """
    if allm.nmmod <= 0:
        # Both loops would skip, leaving IMAX at 1 and D10S/D50S read from
        # uninitialised local arrays.  Refuse rather than reproduce that.
        raise NoModesError(
            "SERPRB needs at least one mode; with NMMOD <= 0 the Fortran "
            "reads uninitialised D10SA/D50SA locals")
    rsn = F32(rsn)
    ton.spr = F32(0.0)
    itm = abs(lufp - 50) // 5 + 1
    itm = min(itm, 10)
    tmx = _TME[itm - 1]

    d10sa = [F32(0.0)] * (allm.nmmod + 1)
    d50sa = [F32(0.0)] * (allm.nmmod + 1)
    for im in range(1, allm.nmmod + 1):                 # DO 145
        if allm.hp[im] <= _NO_MODE_HEIGHT:
            allm.spro[im] = F32(0.001)                  # no mode
            continue                                    # d10sa/d50sa stay 0
        if lufp >= 50:
            dn = tmx * anois.du / DCL                   # noise
            ds = tmx * allm.tllow[im] / DCL             # signal
            xlh = F32(-1.0)
            dso = tmx * ton.sus                         # prediction errors
            dno = tmx * anois.sygu / DCL
        else:
            dn = tmx * anois.dl / DCL
            ds = tmx * allm.tlhgh[im] / DCL
            xlh = F32(1.0)
            dso = tmx * ton.sls
            dno = tmx * anois.sygl / DCL
        d50sa[im] = sqrt32(dn * dn + ds * ds)
        d10sa[im] = d50sa[im] + sqrt32(
            anois.sigm * anois.sigm + ton.ads * ton.ads
            + dno * dno + dso * dso + _DR * _DR)
        d50sa[im] = allm.sn[im] + xlh * d50sa[im]
        z = (rsn - d50sa[im]) / d10sa[im]
        allm.spro[im] = F32(1.0) - fnorml(z)

    imax = 1                                            # DO 150: AMXX is dead
    for i in range(1, allm.nmmod + 1):
        if allm.spro[i] > allm.spro[imax]:
            imax = i
    son.sprob[ifx] = allm.spro[imax]
    son.d10s = d10sa[imax]
    son.d50s = d50sa[imax]
    son.d90s = son.d10s                                 # not a separate decile
