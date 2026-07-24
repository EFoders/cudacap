"""SIGDIS -- adjust the signal-distribution tables and set absorption-loss
parameters for the path (SIGDIS_orig, the version.w32 'W'/'H' branch).

Runs in two parts.  First it loops over the KFX sample-area ionospheres,
computing each area's absorption index ABIY (with the Greg Hand / ICEPAC lower
limit of 0.1, applied for the 'W' version) and its system-loss deciles via
SYSSY, then averages ADJ/SU/SL/ADS/SUS/SLS (/TON/) and ACAV/FEAV/AGLAT (/SIGD/)
over the path.  Second, for the circuit mode K=JMODE, it sets the D-E
absorption-loss parameters (HTLOSS/XVE/XNUZ/HNU/AFE/BFE) and the median/decile
signal-level adjustments (ASM/DSU/DSL) by removing the Es-obscuration and
F2-over-the-MUF contributions (via PRBMUF) at the reference frequency FTAB.

The oracle's version string ends 'W', so this is SIGDIS_orig (areas indexed
1..KFX directly, not the 1/3/5 stride of SIGDIS_alex) with the absorption
floor applied.  Mirrors sigdis.for.  float32 throughout; the ALOG10-derived
signal adjustments carry the usual ~1-ULP libm floor.
"""

import numpy as np

from voacap_gpu.scalar.constants import R2D
from voacap_gpu.scalar.f32math import F32, exp32, log10_32
from voacap_gpu.scalar.prbmuf import prbmuf
from voacap_gpu.scalar.syssy import syssy
from voacap_gpu.scalar.xlin import xlin


def sigdis(state, muf, st, sigd, ton, gcdkm, jmode) -> None:
    """state: LayerState; muf: MufState; st: IonoState (SYS/PERR/F2D);
    sigd: SigdState (out); ton: TonState (out); gcdkm: path km; jmode: mode."""
    gcdkm = F32(gcdkm)
    fi = state.fi
    glav = F32(0.0)
    kfxx = state.kfx
    xkf = F32(kfxx)
    xadj = xsu = xsl = xfmp = xsup = xslp = F32(0.0)
    ac = F32(0.0)
    feav = F32(0.0)
    for k in range(1, kfxx + 1):
        # absorption index (Greg Hand fix; 'W' version applies the 0.1 floor)
        abiy = F32(-0.04) + exp32(F32(-2.937) + F32(0.8445) * fi[1, k])
        abiy = max(F32(0.1), abiy)
        state.abiy[k] = abiy
        # system-loss table for this reflection area
        idp = 5 if gcdkm > 2500.0 else 2
        adj, su, sl, fmp, sup, slp = syssy(st, state.glat[k], state.clck[k],
                                           idp)
        xadj = xadj + adj
        xsu = xsu + su
        xsl = xsl + sl
        xfmp = xfmp + fmp
        xsup = xsup + sup
        xslp = xslp + slp
        feav = feav + fi[1, k]
        ac = ac + abiy
        state.artic[k] = adj
        glav = glav + abs(state.glat[k])
    # path averages
    glav = glav / xkf
    sigd.aglat = glav
    sigd.acav = ac / xkf
    feav = feav / xkf
    sigd.feav = feav
    ton.adj = xadj / xkf
    ton.su = xsu / xkf
    ton.sl = xsl / xkf
    ton.ads = xfmp / xkf
    ton.sus = xsup / xkf
    ton.sls = xslp / xkf
    k = jmode
    # D-E region loss adjustment factor
    sigd.htloss = F32(88.0)
    xve = xlin(F32(90.0), state.htrue, state.fvert, 30, k)
    sigd.xve = xve / fi[1, k]
    sigd.xnuz = F32(63.07)
    sigd.hnu = F32(4.39)
    # CCIR 252 (Haydon/Lucas) loss adjustment for E modes
    if feav - F32(2.0) > 0.0:            # IF(FEAV-2.0) 125,125,120
        sigd.afe, sigd.bfe = F32(1.359), F32(8.617)
    elif feav - F32(0.5) <= 0.0:         # 125  IF(FEAV-0.5) 130,130,135
        sigd.afe, sigd.bfe = F32(0.0), F32(0.0)
    else:                                # 135
        sigd.afe = F32(1.359) * (feav - F32(0.5)) / F32(1.5)
        sigd.bfe = F32(8.617) * (feav - F32(0.5)) / F32(1.5)
    # signal-distribution reference frequency FTAB (FOT, F2 layer)
    glav = glav * R2D
    if glav - F32(40.0) <= 0.0:          # IF(GLAV-40.) 145,145,150
        ftab = muf.yfot[3]
    elif glav - F32(50.0) <= 0.0:        # 150  IF(GLAV-50.) 155,155,160
        ftab = muf.yfot[3]
        ftab = ftab - (glav - F32(40.0)) * (ftab - F32(10.0)) / F32(10.0)
    else:                                # 160  near poles
        ftab = F32(10.0)
    # Es obscuration contribution to the table (removed here, replaced per mode)
    eslsm = F32(0.0)
    if muf.ymuf[4] > 0.0:
        dummy = muf.ymuf[4]
        pes = prbmuf(muf, ftab, dummy, dummy, 4)
        pes = min(max(pes, F32(0.1)), F32(0.9))
        eslsm = F32(-10.0) * log10_32(F32(1.0) - pes)
    # F2 over-the-MUF contribution
    dummy = muf.ymuf[3]
    pf2 = prbmuf(muf, ftab, dummy, dummy, 3)
    pf2 = max(pf2, F32(0.1))
    f2lsm = F32(-10.0) * log10_32(pf2)
    # residual (auroral) loss adjustment to the median signal level
    asm = ton.adj - eslsm - f2lsm
    sigd.asm = max(asm, F32(0.0))
    # upper decile
    pes = F32(0.0)
    if muf.yfot[4] > 0.0:
        dummy = muf.yfot[4]
        pes = prbmuf(muf, ftab, dummy, dummy, 4)
        pes = min(max(pes, F32(0.1)), F32(0.9))
    dummy = muf.yhpf[3]
    pf2 = prbmuf(muf, ftab, dummy, dummy, 3)
    pf2 = max(pf2, F32(0.1))
    dsu = (F32(1.28) * ton.sl
           - (F32(10.0) * log10_32(F32(1.0) - pes) + eslsm)
           - (F32(10.0) * log10_32(pf2) + f2lsm))
    sigd.dsu = max(dsu, F32(0.5))
    # lower decile
    pes = F32(0.0)
    if muf.yhpf[4] > 0.0:
        dummy = muf.yhpf[4]
        pes = prbmuf(muf, ftab, dummy, dummy, 4)
        pes = min(max(pes, F32(0.1)), F32(0.9))
    dummy = muf.yfot[3]
    pf2 = prbmuf(muf, ftab, dummy, dummy, 3)
    pf2 = max(pf2, F32(0.1))
    dsl = (F32(1.28) * ton.su
           - (F32(-10.0) * log10_32(F32(1.0) - pes) - eslsm)
           - (F32(-10.0) * log10_32(pf2) - f2lsm))
    sigd.dsl = max(dsl, F32(1.0))
