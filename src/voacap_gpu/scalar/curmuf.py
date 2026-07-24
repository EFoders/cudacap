"""Stage A transliteration of CURMUF_orig
(reference/voacapl/src/voacapw/curmuf.for).

Circuit MUF and FOT from the full electron-density profile: pick the
controlling sample area, build its profile (LECDEN), and for the E, F2, F1
and sporadic-E layers work out the oblique MUF by the secant law with a
Martyn's-theorem virtual-height correction iterated to 0.1 MHz.  The circuit
MUF is the max over E/F1/F2 (Es excluded); FOT/HPF are the lower/upper
decile envelopes.  REAL*4 throughout; SIN/COS/ASIN/ATAN/EXP carry the
[[f32-libm-noise]] floor, SQRT is clean.

The oracle build selects CURMUF_orig (its version string ends 'W', not
'A'/'a').  The 190/225 GOTO iterations become while-loops with the exact
convergence/try-count exits.  IEDP<0 (the blkdat default) takes the GETHP
profile path; the analytic IEDP>=0 path is preserved too.  INFO's unit-99
debug writes are omitted (no state effect).
"""

from voacap_gpu.scalar.bendy import bendy
from voacap_gpu.scalar.constants import D2R, PIO2, R2D, RZ
from voacap_gpu.scalar.f2dis import f2dis
from voacap_gpu.scalar.f32math import (F32, asin32, atan32, cos32, exp32,
                                        sin32, sqrt32)
from voacap_gpu.scalar.gethp import gethp
from voacap_gpu.scalar.lecden import lecden
from voacap_gpu.scalar.pen import pen

_FQDEL = F32(0.1)
_NFQ = 4
_AE = F32(0.5)
_A1 = F32(0.5)
_A2 = F32(0.5)
_DZF = F32(2000.0)
_BEX = F32(9.5)


def _secant_hops(gcdkm, del_deg, hp, ht, single_div=False):
    """Shared E/F1/F2 geometry: takeoff angle, hop count, and the initial
    secant.  Returns (xhops, psi, cpsi, spsi, del_rad, sphe, secp).

    The F2 block forms PSI as GCDKM/((2*RZ)*XHOPS) (single_div=True) while
    E/F1 form it as GCDKM/(2*RZ) then /XHOPS — same value, but the float32
    rounding differs by up to 1 ULP, so the exact form is preserved."""
    dele = max(F32(0.0), F32(del_deg))
    dl = dele * D2R
    phe = asin32(RZ * cos32(dl) / (RZ + hp))
    nhops = int(F32(0.5) * gcdkm / ((PIO2 - dl - phe) * RZ))
    xhops = F32(nhops + 1)
    if single_div:
        psi = gcdkm / ((F32(2.0) * RZ) * xhops)
    else:
        psi = gcdkm / (F32(2.0) * RZ) / xhops
    cpsi = cos32(psi)
    spsi = sin32(psi)
    tanp = spsi / (F32(1.0) - cpsi + hp / RZ)
    phe = atan32(tanp)
    dl = PIO2 - phe - psi
    cdel = cos32(dl)
    sphe = RZ * cdel / (RZ + ht)
    secp = F32(1.0) / sqrt32(F32(1.0) - sphe * sphe)
    return xhops, psi, cpsi, spsi, dl, sphe, secp


def curmuf(state, muf, st, gcdkm, gcd, amind, ssn, it=1, iedp=-1, info=0):
    """state: LayerState; muf: MufState (outputs); st: IonoState (F2D).
    gcdkm/gcd: great-circle distance km/rad; amind: min takeoff angle deg;
    ssn: sunspot number; it: hour slot.  Mutates muf (and FS(1,KSX))."""
    fi, yi, hi = state.fi, state.yi, state.hi
    gcdkm = F32(gcdkm)
    gcd = F32(gcd)

    kt = 1
    kfx = state.kfx
    kr = 1 if kfx < 2 else (2 if kfx == 2 else 3)
    # select the controlling sample area KS
    if kr < 2:  # 125
        ks = kt
    elif kr == 2:  # 120: FI(1) test
        ks = kr if fi[1, kt] - fi[1, kr] > 0.0 else kt
    else:  # 130
        if abs(fi[3, kt] - fi[3, kr]) - F32(0.01) <= 0.0:  # 120: FI(1) test
            ks = kr if fi[1, kt] - fi[1, kr] > 0.0 else kt
        else:  # 132
            d = fi[3, kt] - fi[3, kr]
            if d < 0.0:  # 125
                ks = kt
            elif d > 0.0:  # 135
                ks = kr
            else:  # ==0 -> 120: FI(1) test
                ks = kr if fi[1, kt] - fi[1, kr] > 0.0 else kt

    lecden(state, ks)

    # tangent (penetration) frequencies for each layer
    xte = F32(1.0) / sqrt32(F32(1.0) + _AE * yi[1, ks] / hi[1, ks])
    fxe = xte * fi[1, ks]
    fx1 = F32(0.0)
    if fi[2, ks] > 0.0:  # 145
        xt1 = F32(1.0) / sqrt32(F32(1.0) + _A1 * yi[2, ks] / hi[2, ks])
        fx1 = xt1 * fi[2, ks]
    xt2 = F32(1.0) / sqrt32(F32(1.0) + _A2 * yi[3, ks] / hi[3, ks])
    fx2 = xt2 * fi[3, ks]
    # force F2MUF to approach MUF(0) at short range
    if gcdkm - _DZF < 0.0:  # 146
        a = F32(-1.0) + F32(1.0) / xt2
        beta = F32(1.0) + a * exp32(-_BEX * gcdkm / _DZF)
        fx2 = beta * fx2
        xt2 = xt2 * beta

    # ---- E layer MUF ----
    if iedp < 0:  # 155
        hpe, hte = gethp(state, fxe)
    else:  # 160 analytic E (110/20 parabola)
        hte, hpe = F32(104.25), F32(125.30)
    xhops, psi, cpsi, spsi, dl, sphe, secp = _secant_hops(
        gcdkm, amind, hpe, hte)
    muf.emuf[it] = fxe * secp
    dele = dl * R2D
    muf.sigl[1] = max(F32(0.01), F32(0.1) * muf.emuf[it])
    muf.sigu[1] = muf.sigl[1]
    muf.delmuf[1] = dele
    muf.hpmuf[1] = hpe
    muf.htmuf[1] = hte
    muf.fvmuf[1] = fxe
    muf.afmuf[1] = F32(0.0)
    muf.nhopmf[1] = int(xhops)
    muf.yfot[1] = muf.emuf[it] - F32(1.28) * muf.sigl[1]
    muf.yhpf[1] = muf.emuf[it] + F32(1.28) * muf.sigu[1]
    muf.ymuf[1] = muf.emuf[it]

    # ---- F2 layer MUF ----
    if iedp < 0:  # 170
        hp2, ht2 = gethp(state, fx2)
    else:  # 175 parabolic E and F2
        ht2 = (hi[3, ks] - yi[3, ks]
               + yi[3, ks] * (F32(1.0) - sqrt32(F32(1.0) - xt2 * xt2)))
        if fi[2, ks] > 0.0:  # 170
            hp2, ht2 = gethp(state, fx2)
        else:  # 176
            hp2 = (hi[3, ks] - yi[3, ks] + bendy(state, 3, ks, fx2)
                   + (pen(state, 1, ks, fx2) - F32(2.0) * yi[1, ks]))
    xhops, psi, cpsi, spsi, dl, sphe, secp = _secant_hops(
        gcdkm, amind, hp2, ht2, single_div=True)
    fob2 = fx2 * secp
    ntry = 0
    xhp = (hp2 - ht2) / RZ
    while True:  # 190 iterate F2MUF with Martyn correction
        fob1 = fob2
        xmut = sphe * sphe
        xfsq = fob1 * fob1 / (fi[3, ks] * fi[3, ks])
        sph = xfsq * xmut * xhp * (ht2 + F32(2.0) * (RZ + ht2) * xhp)
        hpx2 = hp2 + sph
        tanp = spsi / (F32(1.0) - cpsi + hpx2 / RZ)
        phe = atan32(tanp)
        dl = PIO2 - phe - psi
        cdel = cos32(dl)
        sphe = RZ * cdel / (RZ + ht2)
        secp = F32(1.0) / sqrt32(F32(1.0) - sphe * sphe)
        fob2 = fx2 * secp
        ntry += 1
        if abs(fob2 - fob1) - _FQDEL <= 0.0:  # 200 converged
            break
        if _NFQ - ntry < 0:  # 195 out of tries
            break
    muf.f2muf[it] = fob2
    del2 = dl * R2D
    freq = F32(0.9) * muf.f2muf[it]
    sig = f2dis(st, muf.f2muf[it], ssn, state.clat[ks], freq, state.clck[ks])
    muf.sigl[3] = max(F32(0.01), sig)
    freq = F32(1.1) * muf.f2muf[it]
    sig = f2dis(st, muf.f2muf[it], ssn, state.clat[ks], freq, state.clck[ks])
    muf.sigu[3] = max(F32(0.01), sig)
    muf.delmuf[3] = del2
    muf.hpmuf[3] = hpx2
    muf.htmuf[3] = ht2
    muf.fvmuf[3] = fx2
    muf.afmuf[3] = F32(0.0)
    muf.nhopmf[3] = int(xhops)
    muf.yfot[3] = muf.f2muf[it] - F32(1.28) * muf.sigl[3]
    muf.ymuf[3] = muf.f2muf[it]
    muf.yhpf[3] = muf.f2muf[it] + F32(1.28) * muf.sigu[3]

    # ---- F1 layer MUF (or copy E when no F1) ----
    if fi[2, ks] <= 0.0:  # 210 no F1: set F1 = E
        muf.f1muf[it] = muf.emuf[it]
        for a in ("sigl", "sigu", "delmuf", "hpmuf", "htmuf", "fvmuf",
                  "afmuf", "yfot", "ymuf", "yhpf"):
            getattr(muf, a)[2] = getattr(muf, a)[1]
        muf.nhopmf[2] = muf.nhopmf[1]
    else:  # 215
        hp1, ht1 = gethp(state, fx1)
        xhops, psi, cpsi, spsi, dl, sphe, secp = _secant_hops(
            gcdkm, amind, hp1, ht1)
        fob2 = fx1 * secp
        ntry = 0
        xhp = (hp1 - ht1) / RZ
        while True:  # 225 iterate F1MUF
            fob1 = fob2
            xfsq = fob1 * fob1 / (fi[2, ks] * fi[2, ks])
            xmut = sphe * sphe
            sph = xfsq * xmut * xhp * (ht1 + F32(2.0) * (RZ + ht1) * xhp)
            hpy2 = hp1 + sph
            tanp = spsi / (F32(1.0) - cpsi + hpy2 / RZ)
            phe = atan32(tanp)
            dl = PIO2 - phe - psi
            cdel = cos32(dl)
            sphe = RZ * cdel / (RZ + ht1)
            secp = F32(1.0) / sqrt32(F32(1.0) - sphe * sphe)
            fob2 = fx1 * secp
            ntry += 1
            if abs(fob2 - fob1) - _FQDEL <= 0.0:  # 235 converged
                break
            if _NFQ - ntry < 0:  # 230 out of tries
                break
        muf.f1muf[it] = fob2
        del1 = dl * R2D
        muf.sigl[2] = max(F32(0.01), F32(0.1) * muf.f1muf[it])
        muf.sigu[2] = muf.sigl[2]
        muf.delmuf[2] = del1
        muf.hpmuf[2] = hpy2
        muf.htmuf[2] = ht1
        muf.fvmuf[2] = fx1
        muf.afmuf[2] = F32(0.0)
        muf.nhopmf[2] = int(xhops)
        muf.yfot[2] = muf.f1muf[it] - F32(1.28) * muf.sigl[2]
        muf.ymuf[2] = muf.f1muf[it]
        muf.yhpf[2] = muf.f1muf[it] + F32(1.28) * muf.sigu[2]

    # ---- sporadic-E MUF (probability of reflection 0.5) ----
    dels = max(F32(0.0), F32(amind))
    muf.esmuf[it] = F32(1000.0)
    ksx = 0
    tdelx = secpx = F32(0.0)
    hop = F32(0.0)
    for kk in range(1, state.km + 1):  # DO 270
        if state.fs[2, kk] <= 0.0:
            continue
        dl = dels * D2R
        phe = asin32(RZ * cos32(dl) / (RZ + state.hs[kk]))
        nhops = int(F32(0.5) * gcdkm / ((PIO2 - dl - phe) * RZ))
        hop = F32(nhops + 1)
        psi = F32(0.5) * gcd / hop
        tdel = (cos32(psi) - RZ / (RZ + state.hs[kk])) / sin32(psi)
        cdel = F32(1.0) / sqrt32(F32(1.0) + tdel * tdel)
        sphe = RZ * cdel / (RZ + state.hs[kk])
        secp = F32(1.0) / sqrt32(F32(1.0) - sphe * sphe)
        esd = state.fs[2, kk] * secp
        if muf.esmuf[it] - esd > 0.0:  # 265
            muf.esmuf[it] = esd
            ksx = kk
            tdelx = tdel
            secpx = secp
    if muf.esmuf[it] - F32(1000.0) >= 0.0:  # 271 no Es
        for a in ("delmuf", "hpmuf", "htmuf", "fvmuf", "afmuf", "yfot",
                  "ymuf", "yhpf", "sigl", "sigu"):
            getattr(muf, a)[4] = F32(0.0)
        muf.nhopmf[4] = 0
        muf.esmuf[it] = F32(0.0)
    else:  # 272
        muf.delmuf[4] = atan32(tdelx) * R2D
        muf.hpmuf[4] = state.hs[ksx]
        muf.htmuf[4] = muf.hpmuf[4]
        muf.fvmuf[4] = state.fs[2, ksx]
        muf.afmuf[4] = F32(0.0)
        muf.nhopmf[4] = int(hop)
        muf.ymuf[4] = muf.esmuf[it]
        xes = fi[1, ks] - state.fs[1, ksx]
        zdif = F32(0.1)
        fzero = F32(0.1)
        muf.sigu[4] = (state.fs[3, ksx] - state.fs[2, ksx]) / F32(1.28)
        sigb = (state.fs[2, ksx] - state.fs[1, ksx]) / F32(1.28)
        sigz = (state.fs[2, ksx] - fzero) / F32(3.1)
        if xes + zdif < 0.0:  # 315 lower decile is regular E
            muf.sigl[4] = sigb
        elif xes - zdif < 0.0:  # 325 blend
            muf.sigl[4] = sigb + ((xes + zdif) / (F32(2.0) * zdif)) * (
                muf.sigu[4] - sigb)
        else:  # 330
            muf.sigl[4] = muf.sigu[4]
        muf.sigl[4] = min(muf.sigl[4], sigz)  # 335
        state.fs[1, ksx] = state.fs[2, ksx] - F32(1.28) * muf.sigl[4]
        muf.sigl[4] = muf.sigl[4] * secpx  # oblique
        muf.sigu[4] = muf.sigu[4] * secpx
        muf.yfot[4] = muf.esmuf[it] - F32(1.28) * muf.sigl[4]
        muf.yhpf[4] = muf.esmuf[it] + F32(1.28) * muf.sigu[4]

    # ---- circuit MUF (Es excluded) ----
    muf.fot[it] = max(muf.yfot[1], muf.yfot[2], muf.yfot[3])
    muf.allmuf[it] = max(muf.emuf[it], muf.f1muf[it], muf.f2muf[it])
    muf.hpf[it] = max(muf.yhpf[1], muf.yhpf[2], muf.yhpf[3])
    if muf.emuf[it] - muf.allmuf[it] >= 0.0:  # 275
        muf.angmuf[it] = muf.delmuf[1]
        muf.modmuf = 1
    elif muf.f1muf[it] - muf.allmuf[it] >= 0.0:  # 285
        muf.angmuf[it] = muf.delmuf[2]
        muf.modmuf = 2
    else:  # 290
        muf.angmuf[it] = muf.delmuf[3]
        muf.modmuf = 3
