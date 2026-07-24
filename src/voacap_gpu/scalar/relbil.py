"""RELBIL -- reliability: pick the best mode, then combine all of them.

Everything upstream produced modes.  RELBIL turns them into the numbers a
user reads: how often the circuit works, on which mode, and how much more
power it would take to hit the required grade of service.  It is the first
routine to read /allMODE/ rather than /ZON/ -- by this point the per-hop
tables have been accumulated across the whole frequency and what is left is
a single list of up to twenty candidate modes.

Three separate things happen, and conflating them is the easy mistake:

1. **Per-mode reliability.**  For each mode the required S/N is placed on a
   two-sided normal whose lower spread combines the noise UPPER decile with
   the signal LOWER decile, and vice versa.  That crossing is deliberate --
   a bad hour is high noise *and* low signal -- and the 1994 Lucas/Gibson
   correction that fixed it is still in the source as commented-out lines
   next to the live ones.  A mode with a virtual height at or below 70 km
   is not a mode; it gets reliability 0.001.

2. **Most-reliable-mode selection**, in two passes.  The first finds the
   maximum reliability outright.  The second walks the list again and will
   *replace* that choice with any mode whose reliability is within 0.05,
   preferring fewer hops, and among equal hop counts the better median S/N.
   The comment on the first pass says why it exists: the second pass alone
   "could produce an improper mode", because it compares each candidate
   against a running best that it has already moved.

3. **The combined distribution.**  With more than one mode the signals are
   added as powers with random phase -- four separate sums, since the median,
   the two deciles and the field strength each need their own reference
   level (field strength separately because of the receive antenna).  Each
   sum is taken relative to its own maximum term so the 10**x stays in
   range, and terms more than 100 dB down are dropped.

Deliberate transliterations:

* ``DSLF`` and ``DSUF`` are /SIGD/ variables used here as scratch, so RELBIL
  clobbers them.  Preserved -- they are genuinely written.
* ``IRmethod`` is fixed at 1; the percentage-based comparison (2) and the
  "fall back to max reliability below 1%" rule (3) are commented out in the
  source.  Kept as comments, not code.
* ``CREL`` is set to 1000 and the source says "CREL IS NOT USED NOW".
* ``IS = NMODE(IM)`` inside the per-mode loop and ``MR = HN(IR)`` after it
  are both assigned and never read.  Not ported.
* the transmission loss is written twice: first as the chosen mode's TLOSS,
  then immediately overwritten by the 2014 recalculation from the summed
  power.  Only the second survives, but the first is where DBLOS's meaning
  changed, so the order is kept.

float32; ALOG10 and the ``10.**`` powers carry the [[f32-libm-noise]] floor.
"""

import numpy as np

from voacap_gpu.scalar.f32math import F32, log10_32, pow32, sqrt32
from voacap_gpu.scalar.fnorml import fnorml
from voacap_gpu.scalar.pwrdb import pwrdb
from voacap_gpu.scalar.sonstate import LAYTYP

# DATA TME -- the standard normal at 5% steps from the median, for the
# required-power calculation.  TME(9) = 1.2815 is the 90th percentile, the
# reference the ratio is taken against.
_TME = (F32(0.0), F32(0.1257), F32(0.2533), F32(0.3853), F32(0.5244),
        F32(0.6745), F32(0.8416), F32(1.0364), F32(1.2815), F32(1.6449))
_XEPS = F32(0.05)
_DCL = F32(1.28)            # deciles per sigma
_MIN_REL = F32(0.00000001)
_MIN_SPREAD = F32(0.2)
_MAX_SPREAD = F32(30.0)
_NO_MODE_HEIGHT = F32(70.0)
_FLOOR_DB = F32(-10.0)      # ZEXP floor: terms 100 dB down are dropped


def _in_sigmas(z, spread):
    """Z in units of the decile spread, the Fortran's ``Z/(D??R/1.28)``.

    The division is unguarded there, and a zero spread -- reachable when both
    noise deciles and the signal decile are zero -- yields IEEE infinity.
    That is harmless: FNORML clamps |Z| at 5 before doing anything with it,
    so the reliability saturates exactly as it would for any large Z.  The
    errstate makes the intent explicit rather than letting numpy warn about
    arithmetic the Fortran performs deliberately.  (0/0 would be a different
    matter, but Z is only zero when it takes the other branch.)
    """
    with np.errstate(divide="ignore"):
        return z / (spread / _DCL)


def relbil(allm, anois, sigd, son, cant, ifx: int, freq, rsn, lufp: int
           ) -> None:
    """Fill /SON/ column `ifx`.  Mutates `allm`, `sigd` and `son` in place.

    allm: AllModeState (/allMODE/ -- CREL/RELY/NREL are written back);
    anois: AnoisState (/ANOIS/ RCNSE and the noise deciles);
    sigd: SigdState (/SIGD/ -- DSLF/DSUF are used as scratch);
    son: SonState, the output column; cant: CantennaState (for PWRDB).
    ifx: the frequency index; freq: MHz; rsn: required S/N (/DON/);
    lufp: the required reliability in percent (/ION/).
    """
    inum = allm.nmmod
    if inum <= 0:
        return
    rsn = F32(rsn)
    du2 = anois.du * anois.du
    dl2 = anois.dl * anois.dl

    # ---- 1. per-mode reliability -----------------------------------------
    for im in range(1, inum + 1):                   # DO 310
        if allm.hp[im] <= _NO_MODE_HEIGHT:
            allm.crel[im] = F32(0.001)
            allm.rely[im] = F32(0.001)
            continue
        sigd.dslf = allm.tllow[im]
        sigd.dsuf = allm.tlhgh[im]
        # the crossing is deliberate: a bad hour is high noise AND low signal
        son.d10r = sqrt32(du2 + sigd.dslf * sigd.dslf)
        son.d50r = allm.sn[im]
        son.d90r = sqrt32(dl2 + sigd.dsuf * sigd.dsuf)
        z = rsn - son.d50r
        if z <= 0.0:
            z = _in_sigmas(z, son.d10r)
        else:
            z = _in_sigmas(z, son.d90r)
        allm.rely[im] = F32(1.0) - fnorml(z)
        allm.crel[im] = F32(1000.0)                 # "CREL is not used now"

    # ---- 2. most reliable mode -------------------------------------------
    # First pass: the outright maximum.  The 140 loop below cannot be trusted
    # to find it on its own -- it compares against a best it has already
    # moved -- so it starts from this one instead.
    irmax = 1
    for im in range(2, inum + 1):                   # DO 10
        if allm.rely[im] > allm.rely[irmax]:
            irmax = im
    ir = irmax
    xrel = allm.rely[ir]
    xhn = allm.hn[ir]
    xsn = allm.sn[ir]
    for im in range(1, inum + 1):                   # DO 140
        if im == irmax:
            continue
        if xrel < _MIN_REL:
            xrel = _MIN_REL
        reltest = abs(allm.rely[im] - xrel)
        # IRmethod 2 would divide RELtest by XREL for a percentage test
        if reltest >= _XEPS:
            continue                                # not close to the best
        if abs(xhn - allm.hn[im]) <= _XEPS:
            better = xsn < allm.sn[im]              # equal hops: better S/N
        else:
            better = xhn > allm.hn[im]              # this one has fewer hops
        if better:                                  # 139
            ir = im
            xhn = allm.hn[im]
            xsn = allm.sn[im]
            xrel = allm.rely[im]
    # IRmethod 3 would fall back to IRmax when RELY(IR) < .01
    allm.nrel = ir
    ilayer = int(allm.nmode[ir])

    # ---- 3. the frequency's summary --------------------------------------
    if inum == 1:
        son.reliab[ifx] = allm.rely[ir]
        son.dblosl[ifx] = allm.tllow[ir]
        son.dblosu[ifx] = allm.tlhgh[ir]
        son.dbu[ifx] = allm.fldst[ir]
        son.sndb[ifx] = allm.sn[ir]
        son.dbw[ifx] = allm.sigpow[ir]
    else:
        # add the signals with random phase, i.e. add the powers in watts.
        # Each sum is referred to its own maximum term to keep 10**x in range.
        xdslw = xsigs = xdsup = xfld = F32(0.0)
        dxsigs = dxfld = dxdslw = dxdsup = F32(-1000.0)
        for iv in range(1, inum + 1):               # DO 369
            dxsigs = max(dxsigs, allm.sigpow[iv])
            dxfld = max(dxfld, allm.fldst[iv])
            dxdslw = max(dxdslw, allm.sigpow[iv] - allm.tllow[iv])
            dxdsup = max(dxdsup, allm.sigpow[iv] + allm.tlhgh[iv])
        for im in range(1, inum + 1):               # DO 370
            zexp = F32(0.1) * (allm.sigpow[im] - allm.tllow[im] - dxdslw)
            if zexp > _FLOOR_DB:
                xdslw = xdslw + pow32(F32(10.0), zexp)
            zexp = F32(0.1) * (allm.sigpow[im] - dxsigs)
            if zexp > _FLOOR_DB:
                xsigs = xsigs + pow32(F32(10.0), zexp)
            zexp = F32(0.1) * (allm.sigpow[im] + allm.tlhgh[im] - dxdsup)
            if zexp > _FLOOR_DB:
                xdsup = xdsup + pow32(F32(10.0), zexp)
            # field strength is summed separately: the receive antenna gain
            # is already in it and must not be double-counted
            zexp = F32(0.1) * (allm.fldst[im] - dxfld)
            if zexp > _FLOOR_DB:
                xfld = xfld + pow32(F32(10.0), zexp)
        sigmed = F32(-500.0)
        if xsigs > 0.0:
            sigmed = dxsigs + F32(10.0) * log10_32(xsigs)
        if xdslw > 0.0:
            son.dblosl[ifx] = abs(sigmed - F32(10.0) * log10_32(xdslw)
                                  - dxdslw)
        else:
            son.dblosl[ifx] = F32(0.0)
        son.dblosl[ifx] = min(max(son.dblosl[ifx], _MIN_SPREAD), _MAX_SPREAD)
        if xdsup > 0.0:
            son.dblosu[ifx] = abs(dxdsup + F32(10.0) * log10_32(xdsup)
                                  - sigmed)
        else:
            son.dblosu[ifx] = F32(0.0)
        son.dblosu[ifx] = min(max(son.dblosu[ifx], _MIN_SPREAD), _MAX_SPREAD)
        son.dbw[ifx] = sigmed
        delsig = sigmed - allm.sigpow[ir]
        if xfld > 0.0:
            son.dbu[ifx] = dxfld + F32(10.0) * log10_32(xfld)
        else:
            son.dbu[ifx] = F32(-500.0)
        son.sndb[ifx] = allm.sn[ir] + delsig

        # redo the reliability against the combined distribution
        son.d10r = sqrt32(du2 + son.dblosl[ifx] * son.dblosl[ifx])
        son.d50r = son.sndb[ifx]
        son.d90r = sqrt32(dl2 + son.dblosu[ifx] * son.dblosu[ifx])
        if son.d10r < _MIN_SPREAD:
            son.d10r = _MIN_SPREAD
        if son.d90r > _MAX_SPREAD:
            son.d90r = _MAX_SPREAD
        z = rsn - son.d50r
        if z <= 0.0:
            z = _in_sigmas(z, son.d10r)
        else:
            z = _in_sigmas(z, son.d90r)
        son.reliab[ifx] = F32(1.0) - fnorml(z)

    son.gaint[ifx] = allm.tgain[ir]
    son.gainr[ifx] = allm.rgain[ir]
    son.snrlw[ifx] = son.d10r
    son.snrup[ifx] = son.d90r
    son.angle[ifx] = allm.b[ir]
    son.vhigh[ifx] = allm.hp[ir]
    son.delay[ifx] = allm.timed[ir]
    son.dblos[ifx] = allm.tloss[ir]
    # 2014: the powers have been summed, so the transmission loss has to be
    # recomputed from the summed signal rather than taken from one mode
    son.dblos[ifx] = pwrdb(cant, freq) - son.dbw[ifx]
    son.cprob[ifx] = allm.prob[ir]
    if not 1 <= ilayer <= len(LAYTYP):
        raise IndexError(
            f"NMODE({ir}) = {ilayer} is outside LAYTYP 1..{len(LAYTYP)}; "
            "the Fortran reads out of bounds here")
    son.mode[ifx] = LAYTYP[ilayer - 1]
    son.nhp[ifx] = int(allm.hn[ir])         # REAL->INT truncates toward zero
    son.xnynois[ifx] = anois.rcnse
    son.du_nois[ifx] = anois.du
    son.dl_nois[ifx] = anois.dl
    son.rneff[ifx] = allm.eff[ir]

    # required power gain for the specified reliability
    itm = abs(lufp - 50) // 5 + 1
    itm = min(itm, 10)
    tmx = _TME[itm - 1] / _TME[8]
    if lufp < 50:
        son.snpr[ifx] = -(son.d50r + tmx * son.d90r) + rsn
    else:
        son.snpr[ifx] = -(son.d50r - tmx * son.d10r) + rsn
    son.snxx[ifx] = rsn - son.snpr[ifx]
