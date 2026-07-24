"""Stage B (PyTorch, batched) signal-distribution setup — SIGDIS_orig.

Two jobs in one subroutine.  First it walks the KFX sample-area ionospheres
along the path and AVERAGES them: each area's absorption index ABIY (an
exponential in the E critical frequency, floored at 0.1) and its excess-system-
loss deciles from SYSSY, averaged into /TON/ (ADJ/SU/SL/ADS/SUS/SLS) and
/SIGD/ (ACAV/FEAV/AGLAT).  Then, for the circuit's own mode, it sets the D-E
absorption parameters and rebuilds the median and decile signal-level
adjustments ASM/DSU/DSL by SUBTRACTING OFF the Es-obscuration and F2-over-the-
MUF contributions that the tabulated deciles already contain, so the per-mode
code downstream can put back the right ones.

**THE AVERAGE IS OVER A PER-LANE COUNT, AND THAT IS THE ONLY hard part.**
KFX is 1, 2 or 3 in practice (5 slots exist), and in a batch all of them are
live at once, so the sum cannot be a slice reduction.  It is written as a
Python loop over the five area slots with the accumulator updated through
``torch.where(k < kfx, acc + v, acc)``.  That spelling is deliberate: it is
not "sum the masked values", it is "perform exactly the additions the Fortran
performs, in the Fortran's area order".  float32 addition is not associative,
so a reduction that summed all five and subtracted the inactive ones -- or one
that let a library choose the tree -- would differ in the last bit on the very
lanes where it mattered.  Adding a masked ZERO would in fact be safe here
(x + 0 is exact), but the ``where`` on the accumulator needs no such argument.

**A FORTRAN QUIRK THAT LOOKS LIKE A BUG AND IS NOT PORTED AWAY.**  After the
loop the routine sets ``K = JMODE`` and then uses K to index the AREA
dimension: ``XLIN(90., HTRUE, FVERT, 30, K)`` reads area K's ionogram and
``XVE/FI(1,K)`` divides by area K's E critical frequency.  JMODE is a mode
number, not an area number.  It happens to be <= KFX in every deck, so it
always lands on a computed area, but it is a mode index doing an area index's
job and the port reproduces it exactly -- with a per-lane gather along the
area axis, since JMODE varies by lane.

Areas above KFX are still evaluated (that is what batching costs) and their
ABIY/ARTIC are returned.  The Fortran leaves those slots holding the PREVIOUS
circuit's values; there is no previous circuit here, so the returned rows are
freshly computed rather than stale.  **Callers must read only rows 0..KFX-1**;
the two conventions disagree everywhere above that, and neither is meaningful.

This is SIGDIS_orig -- the version.w32 'W' branch, which indexes areas 1..KFX
directly and applies the Greg Hand / ICEPAC absorption floor of 0.1.
SIGDIS_alex (areas 1/3/5) and the 'I'/'a' no-floor variants are deliberately
not ported: the oracle's version string ends 'W'.

float32.  EXP and ALOG10 are the only libm calls; ALOG10 lands in ASM, DSU and
DSL, where it is amplified by the deliberate cancellation of ESLSM and F2LSM
against themselves, so those three carry the [[f32-libm-noise]] floor and the
rest do not.
"""

import torch

from voacap_gpu.scalar.constants import R2D
from voacap_gpu.batched.lockstep import gather_rows
from voacap_gpu.batched.normal import prbmuf_batch
from voacap_gpu.batched.profile import _k
from voacap_gpu.batched.syssy import syssy_batch
from voacap_gpu.batched.xlin import xlin_batch

_F32 = torch.float32

NFV = 30                        # ionogram points XLIN walks
_ABIY_FLOOR = 0.1               # the Greg Hand / ICEPAC lower limit
_PES_LO, _PES_HI = 0.1, 0.9     # AMAX1/AMIN1 around the Es probability
_PF2_LO = 0.1


def sigdis_batch(fi1, glat, clck, kfx, htrue, fvert, jmode, gcdkm,
                 sys, perr, has_data, yfot, yhpf, ymuf, sigl, sigu):
    """Batched SIGDIS_orig.

    fi1: (A, N) the E critical frequency FI(1,K) of each sample area, area in
        its own axis (rows 0..A-1 for Fortran areas 1..A, the ``areas.py``
        convention).  A is normally 5.
    glat/clck: (A, N) geomagnetic latitude (radians) and local time (hours).
    kfx: (N,) how many areas are live on this lane.
    htrue/fvert: (NFV, A, N) each area's ionogram, from ``genion_batch``.
    jmode: (N,) the circuit mode, 1-based -- and, see the module docstring,
        also the area SIGDIS reads the ionogram from.
    gcdkm: (N,) great-circle path length, km; > 2500 picks SYSSY's long-path
        distance code.
    sys/perr/has_data: passed straight to ``syssy_batch``.
    yfot/yhpf/ymuf: (4, N) the FOT/HPF/MUF medians per layer, rows 0..3 for
        Fortran layers 1..4.  Only rows 2 (F2) and 3 (Es) are read.
    sigl/sigu: (4, N) the decile spreads, same row convention; PRBMUF's rows
        are selected here so the caller passes the whole /MUFS/ table.

    Returns ``(sigd, ton, area)``, three dicts of (N,) tensors except
    ``area``'s, which are (A, N):

      sigd: aglat acav feav htloss xve xnuz hnu afe bfe asm dsu dsl
      ton:  adj su sl ads sus sls
      area: abiy artic
    """
    nareas, n = fi1.shape
    dev = fi1.device
    zero = torch.zeros(n, dtype=_F32, device=dev)
    kfx_l = kfx.to(torch.long)
    xkf = kfx.to(_F32)                          # XKF = KFXX

    # IDP=2; IF(GCDKM.GT.2500.)IDP=5 -- a strict >, per lane.
    idp = torch.where(gcdkm > _k(2500.0, gcdkm),
                      torch.full((n,), 5, dtype=torch.long, device=dev),
                      torch.full((n,), 2, dtype=torch.long, device=dev))

    # ---- DO 115 K = 1,KFXX ------------------------------------------------
    glav = zero
    xadj = xsu = xsl = xfmp = xsup = xslp = ac = feav = zero
    abiy_rows, artic_rows = [], []
    for k in range(nareas):
        fik = fi1[k]
        # ABIY = -0.04 + EXP(-2.937 + 0.8445*FI(1,K)), then the 0.1 floor.
        abiy = _k(-0.04, fik) + torch.exp(_k(-2.937, fik)
                                          + _k(0.8445, fik) * fik)
        abiy = torch.maximum(abiy, _k(_ABIY_FLOOR, fik).expand(n))
        adj_k, su_k, sl_k, fmp_k, sup_k, slp_k = syssy_batch(
            sys, perr, has_data, glat[k], clck[k], idp)

        active = kfx_l > k                      # K <= KFXX, 0-based
        # The accumulation ORDER is the Fortran's: area 1 first, and an
        # inactive area performs no addition at all rather than adding zero.
        xadj = torch.where(active, xadj + adj_k, xadj)
        xsu = torch.where(active, xsu + su_k, xsu)
        xsl = torch.where(active, xsl + sl_k, xsl)
        xfmp = torch.where(active, xfmp + fmp_k, xfmp)
        xsup = torch.where(active, xsup + sup_k, xsup)
        xslp = torch.where(active, xslp + slp_k, xslp)
        feav = torch.where(active, feav + fik, feav)
        ac = torch.where(active, ac + abiy, ac)
        glav = torch.where(active, glav + torch.abs(glat[k]), glav)
        # ABIY(K)/ARTIC(K) are written unconditionally here; see the module
        # docstring on why rows at or above KFX are not the Fortran's.
        abiy_rows.append(abiy)
        artic_rows.append(adj_k)

    # ---- path averages ----------------------------------------------------
    glav = glav / xkf
    aglat = glav
    acav = ac / xkf
    feav = feav / xkf
    adj = xadj / xkf
    su = xsu / xkf
    sl = xsl / xkf
    ads = xfmp / xkf
    sus = xsup / xkf
    sls = xslp / xkf

    # ---- K = JMODE: the D-E region loss adjustment ------------------------
    jm = jmode.to(torch.long) - 1
    sel = jm.reshape(1, 1, n).expand(NFV, 1, n)
    ht_k = torch.gather(htrue, 1, sel).squeeze(1)        # (30, N)
    fv_k = torch.gather(fvert, 1, sel).squeeze(1)
    x90 = _k(90.0, fi1).expand(n)
    xve = xlin_batch(x90, ht_k, fv_k, NFV) / gather_rows(fi1, jm)

    # Fixed by DATA in the Fortran; materialised rather than left as stride-0
    # broadcasts so a caller can treat every returned field the same way.
    htloss = _k(88.0, fi1).expand(n).contiguous()
    xnuz = _k(63.07, fi1).expand(n).contiguous()
    hnu = _k(4.39, fi1).expand(n).contiguous()

    # ---- CCIR 252 (Haydon/Lucas) E-mode loss adjustment -------------------
    k1359, k8617 = _k(1.359, feav), _k(8.617, feav)
    # ASSOCIATION MATTERS.  Fortran evaluates `1.359*(FEAV-0.5)/1.5` strictly
    # left to right, i.e. (1.359*(FEAV-0.5))/1.5.  Hoisting the (FEAV-0.5)/1.5
    # ramp out as a shared factor is the obvious tidy-up and it is wrong: it
    # rounds in a different place and moved AFE and BFE by 1 ULP on 5 and 12
    # of the dump's 130 lanes respectively, on CPU as well as CUDA.
    dfe = feav - _k(0.5, feav)
    k15 = _k(1.5, feav)
    hi = feav - _k(2.0, feav) > 0.0                 # IF(FEAV-2.0) 125,125,120
    lo = dfe <= 0.0                                 # IF(FEAV-0.5) 130,130,135
    afe = torch.where(hi, k1359.expand(n),
                      torch.where(lo, zero, k1359 * dfe / k15))
    bfe = torch.where(hi, k8617.expand(n),
                      torch.where(lo, zero, k8617 * dfe / k15))

    # ---- the reference frequency FTAB (FOT, F2 layer) ---------------------
    glav_d = glav * _k(float(R2D), glav)
    yfot3, yfot4 = yfot[2], yfot[3]
    yhpf3, yhpf4 = yhpf[2], yhpf[3]
    ymuf3, ymuf4 = ymuf[2], ymuf[3]
    ten = _k(10.0, glav)
    mid = yfot3 - (glav_d - _k(40.0, glav)) * (yfot3 - ten) / ten
    ftab = torch.where(glav_d - _k(40.0, glav) <= 0.0, yfot3,
                       torch.where(glav_d - _k(50.0, glav) <= 0.0, mid,
                                   ten.expand(n)))

    # ---- remove the Es obscuration and F2 over-the-MUF contributions ------
    sl3, su3 = sigl[2], sigu[2]                     # F2
    sl4, su4 = sigl[3], sigu[3]                     # Es
    one = _k(1.0, ftab)
    kp10, km10 = _k(10.0, ftab), _k(-10.0, ftab)

    def _pes(fset):
        """PRBMUF at FTAB against an Es median, clamped into [0.1, 0.9].

        PRBMUF divides by FSET, so the lanes with no Es (FSET = 0) produce a
        NaN inside it -- which its own ``fset > 0`` select discards, returning
        the step function instead.  The clamp then puts the result inside the
        log's domain on EVERY lane, which is what lets the -10*ALOG10(1-PES)
        below be computed unconditionally and masked afterwards.
        """
        return torch.clamp(prbmuf_batch(ftab, fset, fset, sl4, su4),
                           _PES_LO, _PES_HI)

    def _pf2(fset):
        return torch.clamp(prbmuf_batch(ftab, fset, fset, sl3, su3),
                           min=_PF2_LO)
        # AMIN1(0.9, PF2) is commented out in sigdis.for (GRH, 5/12/2005) --
        # deliberately not applied here either.

    eslsm = torch.where(ymuf4 > 0.0, km10 * torch.log10(one - _pes(ymuf4)),
                        zero)
    f2lsm = km10 * torch.log10(_pf2(ymuf3))

    # residual (auroral) adjustment to the median signal level
    asm = torch.clamp(adj - eslsm - f2lsm, min=0.0)

    # upper decile: PES stays 0 when there is no Es FOT, and 1-0 = 1 makes the
    # ALOG10 term vanish, so the unconditional evaluation is exact there too.
    pes_u = torch.where(yfot4 > 0.0, _pes(yfot4), zero)
    dsu = (_k(1.28, ftab) * sl
           - (kp10 * torch.log10(one - pes_u) + eslsm)
           - (kp10 * torch.log10(_pf2(yhpf3)) + f2lsm))
    dsu = torch.clamp(dsu, min=0.5)

    # lower decile
    pes_l = torch.where(yhpf4 > 0.0, _pes(yhpf4), zero)
    dsl = (_k(1.28, ftab) * su
           - (km10 * torch.log10(one - pes_l) - eslsm)
           - (km10 * torch.log10(_pf2(yfot3)) - f2lsm))
    dsl = torch.clamp(dsl, min=1.0)

    sigd = {"aglat": aglat, "acav": acav, "feav": feav, "htloss": htloss,
            "xve": xve, "xnuz": xnuz, "hnu": hnu, "afe": afe, "bfe": bfe,
            "asm": asm, "dsu": dsu, "dsl": dsl}
    ton = {"adj": adj, "su": su, "sl": sl, "ads": ads, "sus": sus, "sls": sls}
    area = {"abiy": torch.stack(abiy_rows), "artic": torch.stack(artic_rows)}
    return sigd, ton, area
