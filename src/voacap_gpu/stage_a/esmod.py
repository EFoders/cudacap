"""ESMOD -- sporadic-E modes, the /ZON/ slots REGMOD does not fill.

REGMOD's job is F- and E-layer modes and, where sporadic E gets in the way of
one, the obscuration loss.  ESMOD's job is the modes that reflect off the Es
layer itself.  The two share a shape -- free-space loss, absorption, ground
loss, antenna gains, an over-the-MUF term, then field strength and SNR -- but
almost none of the detail, because an Es reflection is a thin mirror at a
fixed height rather than a refraction through a profile.  There is no
deviative loss (ADV is 0), the virtual height *is* the true height, and the
"over-the-MUF" term becomes a reflection loss.

At most **two** Es hops are considered.  Slots 4 and 5 are preset to the
no-mode markers first and unconditionally, so a caller can read them whether
or not any Es mode exists; the four early returns below all leave those
presets in place.

Some deliberate transliterations, each of which looks like a slip and is not:

* the sample area K is the one with the **lowest** FS(2,.), not the highest --
  the comment "all modes will be at least this good" says why: the weakest
  link along the path bounds the whole mode.
* ``AC`` indexes GYZ by **K**, the Es sample area, where REGMOD indexes it by
  LX(K); this routine has no LX at all.
* ``ADX``, the correction for an Es layer below the E layer, is added into
  SABPS *and then again* inside XTLOS, so it counts twice in the loss but
  only once in the reported ABPS.
* when the path needs two hops the loop runs IHOP = 2, 2 -- one iteration
  writing slot **4**, leaving slot 5 at its presets.  Slot 4 is not
  "the one-hop mode", it is "the first mode found".

float32; ALOG10/ALOG/SIN/COS/ATAN/ASIN/SQRT and the ``**1.98``/``**-0.7``
powers all carry the [[f32-libm-noise]] floor.
"""

from voacap_gpu.stage_a.constants import PIO2, R2D, RZ, VOFL
from voacap_gpu.stage_a.f32math import (F32, asin32, atan32, cos32, log10_32,
                                        log32, pow32, sin32, sqrt32)
from voacap_gpu.stage_a.gain import antenna_gain, ground_loss
from voacap_gpu.stage_a.prbmuf import prbmuf
from voacap_gpu.stage_a.pwrdb import pwrdb

_ES_SLOTS = (4, 5)          # DO 97 IHT = 4,5 -- only two Es modes
_P198 = F32(1.98)
_PM07 = F32(-0.7)           # the reflection-loss exponent
_REFC = F32(8.9136)
_MAX_DECILE = F32(25.0)     # "maximum observed is 25."


def _powi(x, n: int):
    """``x**n`` with an INTEGER exponent -- gfortran calls powi, which is
    binary exponentiation, not powf.  N is only ever 1 or 2 here."""
    r = F32(1.0)
    b = F32(x)
    while n:
        if n & 1:
            r = r * b
        n >>= 1
        if n:
            b = b * b
    return r


def _preset(zon) -> None:
    """DO 97: the no-mode markers for both Es slots."""
    for iht in _ES_SLOTS:
        zon.obf[iht] = F32(1000.0)
        zon.adv[iht] = F32(1000.0)
        zon.fslos[iht] = F32(1000.0)
        zon.tloss[iht] = F32(1000.0)
        zon.abps[iht] = F32(1000.0)
        zon.eff[iht] = F32(0.0)
        zon.grlos[iht] = F32(1000.0)
        zon.rgain[iht] = F32(0.0)
        zon.tgain[iht] = F32(0.0)
        zon.hn[iht] = F32(-1.0)
        zon.prob[iht] = F32(0.001)
        zon.crel[iht] = F32(-1000.0)     # ESREG presets its slots to +1000
        zon.rely[iht] = F32(0.001)
        zon.spro[iht] = F32(0.001)
        zon.fldst[iht] = F32(-1000.0)
        zon.sigpow[iht] = F32(-1000.0)
        zon.sn[iht] = F32(-1000.0)
        zon.hp[iht] = F32(-1.0)
        zon.b[iht] = F32(-1.0)
        zon.nmode[iht] = 5
        zon.tllow[iht] = F32(10.0)
        zon.timed[iht] = F32(-1.0)
        zon.tlhgh[iht] = F32(10.0)


def esmod(state, muf, sigd, anois, rays, cant, zon,
          freq, jmode, gcd, gcdkm, amind) -> None:
    """Fill /ZON/ slots 4 and 5 with the sporadic-E modes.  Mutates `zon`.

    state: LayerState (/RON/, /ES/, /GEOG/); muf: MufState (/MUFS/);
    sigd: SigdState (/SIGD/); anois: AnoisState (/ANOIS/ RCNSE);
    rays: RaysState (/RAYS/ IFOB); cant: CantennaState; zon: ZonState, the
    output.  freq: MHz; jmode: the ionosphere SELMOD picked; gcd/gcdkm:
    great-circle distance in radians and km (/DON/); amind: the minimum
    usable take-off angle in degrees (/DON/).
    """
    freq = F32(freq)
    gcd = F32(gcd)
    gcdkm = F32(gcdkm)
    amind = F32(amind)

    # DO 95: the sample area with the LOWEST Es critical frequency bounds
    # the mode -- "all modes will be at least this good".
    k = 1
    for i in range(1, state.km + 1):
        if state.fs[2, k] - state.fs[2, i] > 0.0:
            k = i

    # The maps are Es or E, so they are not usable at the low end.
    fsdead = F32(rays.ifob[1, 3, jmode])
    fsdead = fsdead / F32(1000.0)
    fsdead = min(fsdead, F32(3.0))

    _preset(zon)

    if freq <= fsdead:
        return
    if state.fs[2, k] <= 0.0:
        return

    # virtual heights are the true heights: a thin mirror at HS
    sdmax = F32(2.0) * RZ * (
        PIO2 - asin32(F32(1.0) / (F32(1.0) + state.hs[k] / RZ)))
    ihsrt = int(gcdkm / sdmax + F32(1.0))     # REAL->INT truncates
    ihstp = 2
    if ihsrt > 2:                              # only two Es hops are done
        return
    ih = 3

    # absorption loss factor, as for an F2 mode
    ac = F32(677.2) * sigd.acav / (
        pow32(freq + state.gyz[k], _P198) + F32(10.2))

    for ihop in range(ihsrt, ihstp + 1):       # DO 150
        ih = ih + 1
        gp = F32(ihop)
        ghop = gcd / gp
        thet = F32(0.5) * ghop
        tans = sin32(thet) / (F32(1.0) - cos32(thet) + state.hs[k] / RZ)
        psi = atan32(tans)
        secs = F32(1.0) / cos32(psi)
        sfvmod = freq / secs
        esd = state.fs[2, k] * secs            # MUF at this hop distance
        dele = PIO2 - thet - psi
        cdel = cos32(dele)
        adel = dele * R2D
        if adel < amind:
            continue

        # group path, free-space loss
        path = F32(2.0) * sin32(F32(0.5) * ghop) * (RZ + state.hs[k]) / cdel
        hop = F32(ihop)
        path = hop * path
        sflos = F32(32.45) + F32(20.0) * log10_32(path * freq)

        sinp = RZ * cdel / (RZ + F32(100.0))
        cosp = sqrt32(F32(1.0) - sinp * sinp)
        sabps = ac / cosp
        adx = F32(0.0)
        # Es below E: remove the E-layer bending above HS
        if state.fi[1, k] > sfvmod:
            adx = sigd.afe + sigd.bfe * log32(sfvmod / state.fi[1, k])
        sabps = sabps + adx

        # probability of reflection, at the MUF and at both deciles
        pros = prbmuf(muf, freq, esd, muf.ymuf[4], 4)
        pros = min(pros, F32(0.90))            # limit loss
        ref = _REFC * pow32(pros, _PM07)
        esd = state.fs[1, k] * secs            # lower decile: use the FOT
        ps = prbmuf(muf, freq, esd, muf.yfot[4], 4)
        ps = min(ps, F32(0.9))
        refl = _REFC * pow32(ps, _PM07)
        esd = state.fs[3, k] * secs            # upper decile: use the HPF
        ps = prbmuf(muf, freq, esd, muf.yhpf[4], 4)
        ps = min(ps, F32(0.9))
        refu = _REFC * pow32(ps, _PM07)

        zon.tllow[ih] = sigd.dsl + hop * (refl - ref)
        zon.tlhgh[ih] = sigd.dsu + hop * (ref - refu)
        zon.tllow[ih] = min(zon.tllow[ih], _MAX_DECILE)
        zon.tlhgh[ih] = min(zon.tlhgh[ih], _MAX_DECILE)

        # ground loss, averaged over the sample areas
        y = F32(0.0)
        for ig in range(1, state.km + 1):      # DO 130
            y = y + ground_loss(state, ig, dele, freq)
        xkm = F32(state.km)
        sgrlos = y / xkm

        stgain, _steff = antenna_gain(cant, 1, dele, freq)
        srgain, reff = antenna_gain(cant, 2, dele, freq)
        zon.eff[ih] = reff

        # ADX is inside SABPS already and is added a SECOND time here
        xtlos = (sflos + hop * (sabps + ref + adx)
                 + (hop - F32(1.0)) * sgrlos
                 - srgain - stgain + sigd.asm)

        pdb = pwrdb(cant, freq)
        zon.fldst[ih] = (F32(107.2) + pdb + F32(20.0) * log10_32(freq)
                         - xtlos - srgain)
        zon.sigpow[ih] = pdb - xtlos

        # MUFday: the probability the frequency is below the Es MUF
        pros = prbmuf(muf, freq, muf.ymuf[4], muf.ymuf[4], 4)
        iht = ih
        zon.obf[iht] = _REFC * pow32(pros, _PM07)   # F-layer obscuration n/a
        zon.adv[iht] = F32(0.0)                     # no deviative loss
        zon.fslos[iht] = sflos
        zon.tloss[iht] = xtlos
        zon.abps[iht] = ac / cosp + adx             # ADX counted once here
        zon.grlos[iht] = sgrlos
        zon.rgain[iht] = srgain
        zon.tgain[iht] = stgain
        zon.hn[iht] = hop
        zon.sigpow[iht] = pdb - xtlos
        zon.sn[iht] = zon.sigpow[iht] - anois.rcnse - zon.eff[iht]
        zon.hp[iht] = state.hs[k]
        zon.b[iht] = adel
        zon.nmode[iht] = 4
        zon.prob[iht] = _powi(pros, ihop)
        zon.timed[iht] = path / VOFL
