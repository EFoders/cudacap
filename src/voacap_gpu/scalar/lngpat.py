"""LNGPAT -- assemble the long path's single mode.

The short path builds up to twenty candidate modes and lets RELBIL choose.
The long path builds exactly ONE, and this is where it is built.

"Long path" does not mean the long way round the world.  It means a long-haul
great circle -- past 10000 km HFMUFS forces this branch, and between 7000 and
10000 km MSPEC 121 runs it alongside the short path and blends the two.  What
changes is the model, not the direction: instead of N discrete hops the path
is treated as two end legs plus one long ionospheric stretch, priced end to
end.  GCD here is still the short-circle angle.

SELTXR has already chosen one ray at each end, LTXRGM(1) at control area 1 and
LTXRGM(2) at area ITXRCP(2).  LNGPAT averages the two into a single geometry
-- mean take-off angle, mean virtual height -- and prices it:

    CONVH   the antipodal convergence credit and the group path
    GETTOP  how much of the ionospheric stretch flies over the E layer
    TABS    what that over-the-top stretch costs per km
    BABS    what the remainder, through the D region, costs per km

plus half the absorption at each end (the end legs are half hops), the ground
loss for the fractional hops in between, and both antenna gains.

Three things about the result are worth knowing before reading downstream code.

**IT WRITES TWO SLOTS OF ARRAYS THAT HOLD ONE MODE.**  NMMOD is set to 1, but
B(1) and B(2) are the transmission and reception angles OF THE SAME MODE, and
NMODE(1)/NMODE(2) its reflecting layer at each end.  A loop over IM=1..NMMOD
sees only the transmitter's; anything reading slot 2 as "the second mode" is
wrong.  PROB is the same shape: SETTXR leaves the transmitter end's MUFday in
PROB(1) and the receiver end's in PROB(2), and the last line of this routine
collapses them with AMIN1.

**IT CLOBBERS /SIGD/ DSLF AND DSUF.**  They are declared as path signal
distribution parameters and used here as scratch for the per-layer decile
spreads.  RELBIL does the same thing.  Preserved -- they are genuinely
written, and the values that survive are the averaged ones.

**/TON/ IS DECLARED AND NEVER USED.**  Not ported.

The decile spread is computed for NMODE(1)'s layer, and again for NMODE(2)'s
if the two ends reflect off different layers, in which case the two are
averaged.  The layer index feeds YMUF/YFOT/YHPF/FVMUF, which have four
elements, so a mode index outside 1..4 is an out-of-bounds read; see
:class:`ModeIndexError`.

float32; ALOG10, COS, ATAN and SQRT carry the [[f32-libm-noise]] floor.
"""

from voacap_gpu.scalar.babs import babs
from voacap_gpu.scalar.constants import D2R, RZ, VOFL
from voacap_gpu.scalar.convh import convh
from voacap_gpu.scalar.f32math import F32, atan32, cos32, log10_32, sqrt32
from voacap_gpu.scalar.gettop import gettop
from voacap_gpu.scalar.prbmuf import prbmuf
from voacap_gpu.scalar.pwrdb import pwrdb
from voacap_gpu.scalar.tabs import tabs

_MAX_DECILE = F32(25.0)         # the signal deciles are capped here
_TX = 1                         # the transmitter end is always control area 1


class ModeIndexError(IndexError):
    """NMODE at one end is outside 1..4, so YMUF(IS) reads off the end.

    NMODE comes from /REFLX/ IMODE, which FINDF fills with a layer index and
    SETTXR overwrites with MODMUF for the over-the-MUF ray.  A ray SELTXR
    chose but FINDF never filled leaves IMODE at its zero-initialised value,
    and the Fortran then indexes YMUF(0).  Refused here rather than
    reproduced.
    """


def lngpat(state, lpath, losx, reflx, muf, allm, sigd, anois, cant,
           freq, gcd, itxrcp2: int) -> None:
    """Fill /allMODE/ slot 1 (and slot 2's angle and layer) in place.

    state: LayerState (GYZ for BABS/TABS, HI/FI for GETTOP); lpath:
    LpathState (LTXRGM, DELOPT); losx: LosxState; reflx: ReflxState; muf:
    MufState; allm: AllModeState; sigd: SigdState (ASM/DSL/DSU in, DSLF/DSUF
    clobbered); anois: AnoisState (RCNSE); cant: CantennaState (for PWRDB);
    freq in MHz; gcd the great-circle angle in radians; itxrcp2 the receiver
    end's control-area index.

    Returns without touching anything when SELTXR left either end unset.
    """
    freq = F32(freq)
    gcd = F32(gcd)
    l1 = int(lpath.ltxrgm[1])
    l2 = int(lpath.ltxrgm[2])
    # .....LTXRGM(1) .GE. 1 AND LTXRGM(2) .GE. 1 SINCE OVER-THE-MUF ADDED
    if l1 <= 0 or l2 <= 0:
        return
    k2 = int(itxrcp2)

    # .....AVERAGE TAKE-OFF ANGLE
    del_ = F32(0.5) * D2R * (reflx.delfx[l1, _TX] + reflx.delfx[l2, k2])
    # .....AVERAGE VIRTUAL HEIGHT
    hpx = F32(0.5) * (reflx.hpflx[l1, _TX] + reflx.hpflx[l2, k2])
    rcosd = RZ * cos32(del_)
    sphe = rcosd / (RZ + hpx)
    phe = atan32(sphe / sqrt32(F32(1.0) - sphe * sphe))
    dt = F32(0.5) * reflx.gdflx[l1, _TX] / RZ
    dr = F32(0.5) * reflx.gdflx[l2, k2] / RZ
    ray = RZ * cos32(phe + del_) / sphe
    # .....FREE-SPACE CONVERGENCE FACTOR
    ch, ptot = convh(gcd, phe, del_, hpx, ray)
    # .....FREE-SPACE LOSS
    free = F32(36.58) + F32(20.0) * log10_32(
        F32(0.6214) * ptot * freq) - ch
    # .....IONOSPHERIC DISTANCE
    di = gcd - dt - dr
    dikm = di * RZ
    if dikm <= F32(-1.0):
        dm = F32(0.0)
        am = F32(0.0)
        af = F32(0.0)
        df = F32(0.0)
    else:
        # .....OVER-THE-TOP DISTANCE
        gmkm = gettop(state, lpath, freq, dikm)
        dm = gmkm / RZ
        df = di - dm
        df = max(df, F32(0.0))
        # .....LOSS PER KILOMETER
        am = tabs(state, freq, del_, hpx, dm)
        af = babs(state, sigd, freq, del_, hpx, df)
    # .....LOSSES
    xlm = am * dm * RZ
    xlf = af * df * RZ
    #  LOSS AT TRANSMITTER END.
    xlt = (losx.andvx[l1, _TX] + losx.advx[l1, _TX]) / F32(2.0)
    #  LOSS AT RECEIVER END
    xlr = (losx.andvx[l2, k2] + losx.advx[l2, k2]) / F32(2.0)
    # .....AVERAGE GROUND LOSS
    gloss = F32(0.5) * (losx.grlosx[l1, _TX] + losx.grlosx[l2, k2])
    fhop = df / (dr + dt)
    fhop = max(fhop, F32(1.0))
    allm.nmmod = 1
    # .....TRANSMISSION LOSS
    tlosl = (free + xlt + xlm + xlf + xlr + (fhop - F32(1.0)) * gloss
             + sigd.asm - losx.tgainx[l2, k2] - losx.tgainx[l1, _TX])
    allm.grlos[1] = gloss
    allm.hn[1] = F32(0.5) * gcd / (dr + dt)
    # .....NUMBER OF HOPS
    allm.hn[1] = max(F32(1.0), allm.hn[1])
    # .....VIRTUAL HEIGHT
    allm.hp[1] = hpx
    # .....ANTENNA GAIN
    allm.rgain[1] = losx.tgainx[l2, k2]
    allm.tgain[1] = losx.tgainx[l1, _TX]
    allm.eff[1] = losx.efflp[l2]        # ray index only -- always the RX end
    # .....TIME DELAY
    allm.timed[1] = ptot / VOFL
    allm.tloss[1] = tlosl
    # .....TRANSMISSION ANGLE
    allm.b[1] = reflx.delfx[l1, _TX]
    # .....RECEPTION ANGLE  (slot 2 of a per-mode array; see the docstring)
    allm.b[2] = reflx.delfx[l2, k2]
    # .....MODE TYPE
    allm.nmode[1] = reflx.imode[l1, _TX]
    allm.nmode[2] = reflx.imode[l2, k2]
    allm.fslos[1] = free
    allm.adv[1] = (losx.advx[l1, _TX] + losx.advx[l2, k2]) * F32(0.5)
    allm.abps[1] = (losx.andvx[l1, _TX] + losx.andvx[l2, k2]) * F32(0.5)
    allm.obf[1] = (losx.aofx[l1, _TX] + losx.aofx[l2, k2]) * F32(0.5)
    pwr = pwrdb(cant, freq)
    allm.sigpow[1] = pwr - tlosl
    # .....FIELD STRENGTH
    allm.fldst[1] = (F32(107.2) + pwr + F32(20.0) * log10_32(freq)
                     - tlosl - allm.rgain[1])
    # .....SNR
    # ------------    CHANGED 9/24/91  (LONG PATH RCVR EFF CORRECTION)  FJR
    # ------------            SN(1) = SIGPOW(1) - RCNSE
    allm.sn[1] = allm.sigpow[1] - anois.rcnse - allm.eff[1]

    itry = 1
    is_ = int(allm.nmode[1])
    dslff = dsuff = F32(0.0)
    while True:                                             # 200
        if not 1 <= is_ <= 4:
            raise ModeIndexError(
                f"NMODE = {is_} indexes YMUF(1..4) out of range; the ray "
                "SELTXR chose has no layer, so FINDF never filled it")
        p = prbmuf(muf, freq, muf.ymuf[is_], muf.ymuf[is_], is_)
        zlmuf = F32(-10.0) * log10_32(p)
        p = prbmuf(muf, freq, muf.yfot[is_], muf.yfot[is_], is_)
        zlfot = F32(-10.0) * log10_32(p)
        p = prbmuf(muf, freq, muf.yhpf[is_], muf.yhpf[is_], is_)
        zlhpf = F32(-10.0) * log10_32(p)
        cpr = muf.fvmuf[is_] / muf.ymuf[is_]
        sigd.dslf = (zlfot - zlmuf) / cpr
        sigd.dsuf = (zlmuf - zlhpf) / cpr
        if itry - 1 > 0:                        # IF(ITRY-1) 250,250,260
            sigd.dsuf = F32(0.5) * (sigd.dsuf + dsuff)      # 260
            sigd.dslf = F32(0.5) * (sigd.dslf + dslff)
            break
        if is_ - int(allm.nmode[2]) == 0:       # IF(IS-NMODE(2)) 255,265,255
            break                               # both ends on one layer
        itry = 2                                # 255
        is_ = int(allm.nmode[2])
        dslff = sigd.dslf
        dsuff = sigd.dsuf
    allm.tllow[1] = sigd.dsl + allm.hn[1] * sigd.dslf        # 265
    allm.tlhgh[1] = sigd.dsu + allm.hn[1] * sigd.dsuf
    # .....LOWER DECILE
    allm.tllow[1] = min(allm.tllow[1], _MAX_DECILE)
    # .....UPPER DECILE
    allm.tlhgh[1] = min(allm.tlhgh[1], _MAX_DECILE)
    # .....F.DAYS  -- SETTXR left the two ends' MUFdays in slots 1 and 2
    allm.prob[1] = min(allm.prob[1], allm.prob[2])
