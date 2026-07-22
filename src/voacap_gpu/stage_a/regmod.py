"""REGMOD -- per-mode loss budget for one frequency and hop distance.

Takes the up-to-six regular modes FDIST left in /MODES/ and turns each into a
full link budget in /ZON/: free-space loss, non-deviative absorption, deviative
loss, ground-reflection loss, sporadic-E obscuration, antenna gains, the
over-the-MUF loss and its deciles, and from those the field strength, signal
power and signal-to-noise.  Sporadic-E *modes* are ESMOD's job, not this one --
what appears here is only Es getting in the way of an F mode.

Mode slot 7 is untouched beyond its HN/HP = -1 preset; ESMOD fills it.

Two shapes of loss are computed per mode, then combined at the end:

* ``XTLOS``, the median transmission loss, accumulating free-space + per-hop
  absorption + deviative + ground + obscuration + the system loss ASM, less
  the two antenna gains, plus HOP times the over-the-MUF loss XLS;
* the decile spread TLLOW/TLHGH around it, from the same probability
  evaluation at the FOT and HPF instead of the MUF.

Three details worth naming, since each is invisible in the arithmetic:

* ``**1.98`` is a default-REAL literal, so the exponent is the float32 nearest
  1.98 -- see :func:`~voacap_gpu.stage_a.f32math.pow32`.
* ``DEL`` and ``CDEL`` are *recomputed* partway through (at the 1995 hop-MUF
  change) from the path's own geometry rather than the mode's.  Everything
  before that point -- crucially the three GAIN calls -- uses the mode's
  take-off angle; everything after uses the recomputed one.  The order is
  load-bearing.
* ``ITRY`` is vestigial: it is set to 1 at the top of every iteration, so the
  ``ITRY.lt.1`` test can never fire and the ``ITRY = -1`` near the end is dead.
  Preserved as a comment rather than as code.

float32; ALOG10/ALOG/EXP/COS/SIN/ATAN/SQRT and the 1.98 powers all carry the
[[f32-libm-noise]] floor, and XTLOS sums seven terms of comparable size.
"""

import numpy as np

from voacap_gpu.stage_a.constants import D2R, PIO2, RZ, VOFL
from voacap_gpu.stage_a.f32math import (F32, atan32, cos32, exp32, log10_32,
                                        log32, pow32, sin32, sqrt32)
from voacap_gpu.stage_a.gain import antenna_gain, ground_loss
from voacap_gpu.stage_a.prbmuf import prbmuf
from voacap_gpu.stage_a.pwrdb import pwrdb

_LX = (0, 1, 3, 5)          # DATA LX/1,3,5/ -- ionosphere K to sample area L
_P198 = F32(1.98)
_DELMAX = F32(89.99) * D2R
_PROB_FLOOR = F32(0.000001)


class RegmodError(ValueError):
    """A state REGMOD refuses to run on, where the Fortran STOPs."""


def regmod(state, modes, muf, sigd, anois, rays, cant, zon,
           freq, jmode, gcd) -> None:
    """Fill /ZON/ for the current frequency.  Mutates `zon` in place.

    state: LayerState (/RON/, /ES/, /GEOG/); modes: ModesState (/MODES/);
    muf: MufState (/MUFS/); sigd: SigdState (/SIGD/); anois: AnoisState
    (/ANOIS/); rays: RaysState (/RAYS/); cant: CantennaState (/cantenna/ +
    /pantenna/); zon: ZonState, the output.  freq: MHz; jmode: the ionosphere
    selected by SELMOD; gcd: great-circle distance (radians, /DON/).
    """
    freq = F32(freq)
    gcd = F32(gcd)
    ghop = F32(modes.ghop)
    k = jmode
    ll = _LX[k]
    # average absorption index from SIGDIS
    ac = F32(677.2) * sigd.acav
    bc = pow32(freq + state.gyz[ll], _P198)
    ihop = int(gcd / ghop + F32(0.01))
    hop = F32(ihop)

    for im in range(1, 8):
        # ITRY = 1 here in the Fortran; the ITRY test below it is unreachable
        zon.hn[im] = F32(-1.0)
        zon.hp[im] = F32(-1.0)
        if im >= 7:
            continue
        if modes.hpmod[im, k] <= 0.0:
            continue

        del_ = D2R * modes.delmod[im, k]
        del_ = min(del_, _DELMAX)
        cdel = cos32(del_)
        psi = ghop * F32(0.5)
        phe = PIO2 - psi - del_
        path = (F32(2.0) * (modes.hpmod[im, k] + RZ * (F32(1.0) - cos32(psi)))
                / cos32(phe))
        path = abs(path * hop)
        zon.timed[im] = path / VOFL
        zon.fslos[im] = F32(32.45) + F32(20.0) * log10_32(path * freq)

        if state.fi[1, k] - modes.fvmod[im, k] < 0.0:      # 125: F layer mode
            xnsq = F32(10.2)
            sinp = RZ * cdel / (RZ + F32(100.0))
            secp = F32(1.0) / sqrt32(F32(1.0) - sinp * sinp)
            zon.abps[im] = secp * ac / (bc + xnsq)
            secp = F32(1.0) / sin32(del_ + psi)
            zon.adv[im] = (secp * modes.afmod[im, k]
                           * (pow32(modes.fvmod[im, k] + state.gyz[ll], _P198)
                              + xnsq) / (bc + xnsq))
            zon.obf[im] = F32(0.0)
            obfu = F32(0.0)
            obfl = F32(0.0)
            if state.fs[2, k] > 0.0:
                # Es obscuration.  At low frequencies folding regular E into
                # sporadic E misbehaves, so the sounding floor FSDEAD applies
                # (SIGDIS does the same).
                fsdead = F32(rays.ifob[1, 3, k])           # INTEGER -> REAL
                fsdead = fsdead / F32(1000.0)
                fsdead = min(fsdead, F32(3.0))
                fmhz = max(freq, fsdead)
                sins = RZ * cdel / (RZ + state.hs[k])
                secs = F32(1.0) / sqrt32(F32(1.0) - sins * sins)
                # median MUF for THIS hop, not for the path
                esd = state.fs[2, k] * secs
                pros = prbmuf(muf, fmhz, esd, muf.ymuf[4], 4)
                pros = min(pros, F32(0.90))
                zon.obf[im] = F32(-10.0) * log10_32(F32(1.0) - pros)
                esd = state.fs[1, k] * secs
                # upper decile obscuration at the Es FOT
                pros = prbmuf(muf, fmhz, esd, muf.yfot[4], 4)
                pros = min(pros, F32(0.90))
                obfu = F32(-10.0) * log10_32(F32(1.0) - pros)
                # lower decile obscuration at the Es HPF
                pros = prbmuf(muf, fmhz, esd, muf.yhpf[4], 4)
                pros = min(pros, F32(0.9))
                obfl = F32(-10.0) * log10_32(F32(1.0) - pros)
        else:                                              # 106: D-E mode
            if sigd.htloss - modes.htmod[im, k] <= 0.0:     # 110
                xnsq = F32(10.2)
            else:                                           # 115
                hnux = F32(61.0) + F32(3.0) * (modes.htmod[im, k]
                                               - F32(70.0)) / F32(18.0)
                xnsq = sigd.xnuz * exp32(F32(-2.0) * (hnux - F32(60.0))
                                         / sigd.hnu)
            # secant PHE at the reflection area; 100 km is assumed above that
            heff = min(F32(100.0), modes.htmod[im, k])
            sinp = RZ * cdel / (RZ + heff)
            secp = F32(1.0) / sqrt32(F32(1.0) - sinp * sinp)
            zon.abps[im] = secp * ac / (bc + xnsq)
            # absorption loss with the E-layer bending effect removed
            xv = max(modes.fvmod[im, k] / state.fi[1, k], sigd.xve)
            adx = sigd.afe + sigd.bfe * log32(xv)
            secp = F32(1.0) / sin32(del_ + psi)
            bcx = bc
            zon.adv[im] = (secp * modes.afmod[im, k]
                           * (pow32(modes.fvmod[im, k] + state.gyz[ll], _P198)
                              + xnsq) / (bcx + xnsq) + adx)
            zon.obf[im] = F32(0.0)                # no Es obscuration for D-E
            obfu = F32(0.0)
            obfl = F32(0.0)

        # 160: ground reflection loss, averaged over the sample areas
        y = F32(0.0)
        for ig in range(1, state.km + 1):
            y = y + ground_loss(state, ig, del_, freq)
        zon.grlos[im] = y / F32(state.km)
        zon.tgain[im], _teff = antenna_gain(cant, 1, del_, freq)
        zon.rgain[im], zon.eff[im] = antenna_gain(cant, 2, del_, freq)

        hops = min(hop, F32(2.0))          # only 2 hops for the obscuration
        xtlos = (zon.fslos[im] + hop * (zon.abps[im] + zon.adv[im])
                 + zon.grlos[im] * (hop - F32(1.0)) + hops * zon.obf[im]
                 + sigd.asm - zon.rgain[im] - zon.tgain[im])

        ismod = int(modes.itmod[im, k])
        sphet = RZ * cdel / (RZ + modes.htmod[im, k])
        cphet = F32(1.0) - sphet * sphet
        cphet = max(F32(0.000001), cphet)
        cphet = sqrt32(cphet)

        # 1995 change: base MUFday on the specific HOP muf, not the path muf.
        # The CPHET above is kept as-is for its other uses.
        psi = gcd / F32(2.0)
        psi = psi / hop
        cpsi = cos32(psi)
        spsi = sin32(psi)
        tanp = spsi / (F32(1.0) - cpsi + muf.hpmuf[ismod] / RZ)
        phe = atan32(tanp)
        del_ = PIO2 - phe - psi
        cdel = cos32(del_)
        sphe = RZ * cdel / (RZ + muf.htmuf[ismod])
        xmuf = muf.fvmuf[ismod] / sqrt32(F32(1.0) - sphe * sphe)
        zon.prob[im] = prbmuf(muf, freq, xmuf, muf.ymuf[ismod], ismod)
        if zon.prob[im] < F32(0.0001):
            if zon.prob[im] <= 0.0:
                raise RegmodError(
                    f"MUFday probability {float(zon.prob[im])} for mode {im} "
                    f"at {float(freq)} MHz; the Fortran STOPs here")
            # extra loss as MUFday gets very low, 0 to 24 dB over 1e-4..1e-7
            xfac = log10_32(zon.prob[im])
            if xfac > F32(-4.0):
                xfac = F32(-4.0)
            if xfac < F32(-7.0):
                xfac = F32(-7.0)
            ghlos = (xfac + F32(4.0)) * F32(8.0)
            xtlos = xtlos - ghlos

        xmuf = muf.fvmuf[ismod] / cphet
        p = prbmuf(muf, freq, xmuf, muf.ymuf[ismod], ismod)
        p = max(p, _PROB_FLOOR)
        xls = F32(-10.0) * log10_32(p) / cphet
        xtlos = xtlos + xls * hop                          # median

        cpr = muf.fvmuf[ismod] / muf.ymuf[ismod]
        fvfot = muf.yfot[ismod] * cpr
        xmuf = fvfot / cphet
        pf = prbmuf(muf, freq, xmuf, muf.yfot[ismod], ismod)
        pf = max(pf, _PROB_FLOOR)
        xlsl = F32(-10.0) * log10_32(pf) / cphet
        fvhpf = muf.yhpf[ismod] * cpr
        xmuf = fvhpf / cphet
        pf = prbmuf(muf, freq, xmuf, muf.yhpf[ismod], ismod)
        pf = max(pf, _PROB_FLOOR)
        xlsu = F32(-10.0) * log10_32(pf) / cphet

        # despite the names these are the signal level's lower/upper deciles
        zon.tllow[im] = (sigd.dsl + hops * (obfl - zon.obf[im])
                         + hop * (xlsl - xls))
        zon.tlhgh[im] = (sigd.dsu + hops * (zon.obf[im] - obfu)
                         + hop * (xls - xlsu))
        zon.tllow[im] = min(zon.tllow[im], F32(25.0))
        zon.tlhgh[im] = min(zon.tlhgh[im], F32(25.0))

        zon.tloss[im] = xtlos
        if freq <= 0.0:
            raise RegmodError("frequency must be positive; the Fortran STOPs")
        pdb = pwrdb(cant, freq)
        zon.fldst[im] = (F32(107.2) + pdb + F32(20.0) * log10_32(freq)
                         - xtlos - zon.rgain[im])
        zon.sigpow[im] = pdb - xtlos
        # long-path receiver efficiency correction (1991)
        zon.sn[im] = zon.sigpow[im] - anois.rcnse - zon.eff[im]
        zon.b[im] = modes.delmod[im, k]
        zon.nmode[im] = ismod
        zon.hp[im] = modes.hpmod[im, k]
        zon.hn[im] = F32(ihop)
