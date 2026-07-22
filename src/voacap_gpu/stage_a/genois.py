"""GENOIS -- combine atmospheric, galactic and man-made noise into one
distribution, per ITS Report 87-212 "Updated NOISE Model for use in IONCAP".

Three independent noise sources are each described by a median level and a
pair of decile ratios, and GENOIS has to produce a *single* median RCNSE with
a single pair of deciles for the receiver.  It cannot just add the medians:
adding lognormal variables is a moment problem, so each component is converted
to a linear mean A and variance V, those are summed, and the sum is converted
back on the assumption that the total is itself lognormal (Spaulding's method).

The three components:

* atmospheric -- ATNU/ATNY, the 1 MHz Fam ANOIS1 read out of the CCIR maps for
  the two local-time blocks bracketing the receiver time, carried to frequency
  by :func:`~voacap_gpu.stage_a.genfam.genfam` and then linearly interpolated
  between the blocks on SLOP.  Frequency is capped at 55 MHz for the lookup.
* galactic -- a fixed 52 - 23*log10(f) law, *dropped entirely* (and MODE_N2
  cleared) when the frequency is at or below the F2 critical frequency, since
  galactic noise cannot then penetrate to the receiver.
* man-made -- either an explicit 3 MHz level from the input card (NOISE > 0)
  or one of four CCIR environment categories (NOISE <= 0), each a level at
  1 MHz and a decay exponent.

Two details of the combination are worth naming:

* **Caruana's modification.**  Spaulding's variance sum breaks down when the
  atmospheric deciles are large -- above about 12 dB the implied lognormal
  sigma exceeds what the simple power sum can support and the returned median
  runs away.  When DUA or DLA exceeds 12 dB the sigmas are therefore capped at
  the value implied by the plain power-sum median RNSE, and METHOD_USED
  records that this happened.  This is the April 2007 ITU-R P.372-8 change.
* **Precision.**  A, V and the sigmas are the only REAL*8 quantities in the
  routine (genois.for declares them so): the variance terms span many orders
  of magnitude and the ``log(1 + V/A**2)`` that recovers sigma is a
  cancellation.  Everything else -- including every EXP feeding the sums, and
  the final XRNSE/DU/DL/SIGM/SYGU/SYGL -- is REAL*4.

Also here, incidentally rather than by right: the receiver antenna efficiency
is fetched from GAIN and broadcast to all six /ZON/ mode slots, because this
is the one routine per frequency that is guaranteed to run.

The ``NYEAR = ' 2089'`` escape hatch selects GENOIS_OLD, the pre-2007 model
(no Caruana limit, galactic noise folded in differently, a single decile
exponent for all four man-made categories).  It is a deliberate operator
opt-in to superseded physics and is not ported -- :class:`OldNoiseModel` is
raised instead of silently returning the current model's answer.

float32 except where noted; ALOG10/ALOG/EXP/SQRT and the ``10.**`` powers all
carry the [[f32-libm-noise]] floor.
"""

import numpy as np

from voacap_gpu.stage_a.f32math import (F32, F64, exp32, log10_32, log32,
                                        pow32, sqrt32)
from voacap_gpu.stage_a.gain import antenna_gain
from voacap_gpu.stage_a.genfam import genfam

# DATA DFAC,BFAC,CFAC /7.87384, 30.99872, 5.56765/
#   DFAC = SQRT(2 * 1.282**2 * 4.34294**2)   BFAC = (1.282 * 4.34294)**2
#   CFAC = 4.34294 * 1.282
_DFAC = F32(7.87384)
_BFAC = F32(30.99872)
_CFAC = F32(5.56765)
_DB = F32(4.34294)          # 10/ln(10): nepers <-> dB
_DB8 = F64(4.34294)         # the 4.34294d0 literal in Caruana's test -- NOT
#                             the same constant as _DB promoted, and the
#                             Fortran writes both, so both are kept
_INV_DB = F32(0.23026)      # 1/4.34294
_KTB = F32(204.0)           # dB from kTb (290 K, 1 Hz) to 1 W

# Man-made noise at 1 MHz and its decay exponent, by CCIR environment.
#            #1          #2          #3      #4       #5      #6
#        industrial residential   rural   remote    noisy   quiet
_XNINT = (F32(76.8), F32(72.5), F32(67.2), F32(53.6), F32(83.2), F32(65.2))
_CONN = (F32(27.7), F32(27.7), F32(27.7), F32(28.6), F32(37.5), F32(29.1))
# Categories 5 and 6 are unreachable: MA is clamped with MIN0(4,MA).  Kept
# because the DATA statements carry them and a future NOISE range might.

_TWELVE = F32(12.0)         # the decile above which Spaulding breaks down


class OldNoiseModel(NotImplementedError):
    """NYEAR = ' 2089' selects GENOIS_OLD, the superseded pre-2007 model."""


def _sq(x):
    """``x**2`` with an INTEGER exponent -- gfortran emits a multiply, not
    powf, so this must not go through :func:`pow32`."""
    return x * x


def genois(st, state, anois, ton, zon, diag, cant, freq, noise: int, rlat,
           nyear: str = "") -> None:
    """Fill the receiver noise level and its spread.  Mutates in place.

    st: IonoState (/TWO/ F2D, FAM, DUD); state: LayerState (/RON/ FI, KFX);
    anois: AnoisState (/ANOIS/ -- ATNU/ATNY/CC/TM/KJ/JK in, RCNSE and the five
    spread terms out); ton: TonState (/TON/ GNOS, XNOISE, ZNOISE out);
    zon: ZonState (/ZON/ EFF out); diag: NoiseDiagState (the component
    breakdown); cant: CantennaState (/cantenna/, for the receiver efficiency).
    freq: MHz; noise: the NOISE card value (/ION/); rlat: receiver latitude
    (radians, /DON/); nyear: the year string from /ALPHA/.
    """
    if nyear == " 2089":
        raise OldNoiseModel(
            "NYEAR ' 2089' selects GENOIS_OLD (the pre-2007 noise model), "
            "which is not ported")

    freq = F32(freq)
    dume = min(freq, F32(55.0))                 # AMIN1(FREQ,55.)

    # ---- atmospheric ------------------------------------------------------
    # GENFAM's outputs are the /ANOIS/ DU/DL/SIGM/SYGU/SYGL themselves in the
    # Fortran (aliased actual arguments), but all five are reassigned
    # unconditionally further down before anything reads them, so the
    # intermediate store is dead and they are kept local here.
    if st.f2d[0, 0, 0] < 0.0:       # IF (F2D(1,1,1)) 85,90,90 -- negative only
        # 85: no long-term data base.  ATNOS=204. is immediately overwritten
        # with 0. on the next line, so the atmospheric term contributes
        # 10**(-204/10) W once the -204 dB shift is applied, i.e. nothing.
        atnos = F32(0.0)
        dua, dla = F32(9.0), F32(7.0)
        sma, sua, sla = F32(3.0), F32(1.5), F32(1.5)
    else:
        atnz, du, dl, sigm, sygu, sygl = genfam(
            st, rlat, anois.kj, dume, anois.atnu)
        atnx, dx, dq, sigz, sigx, sigsq = genfam(
            st, rlat, anois.jk, dume, anois.atny)
        # interpolate between the two local-time blocks
        slop = abs(anois.cc - anois.tm) / F32(4.0)
        atnos = atnz + (atnx - atnz) * slop
        dua = du + (dx - du) * slop
        dla = dl + (dq - dl) * slop
        sma = sigm + (sigz - sigm) * slop
        sua = sygu + (sigx - sygu) * slop
        sla = sygl + (sigsq - sygl) * slop

    # (DUA/DFAC)**2 = (DUA/1.282)**2 / (2*4.34294**2), and
    # DUA*DUA/BFAC  = (DUA/1.282)**2 / 4.34294**2 -- the lognormal mean and
    # variance of a component with decile ratio DUA and median ATNOS.
    t = dua / _DFAC
    au1 = F64(exp32(_sq(t) + atnos / _DB))
    vu1 = au1 * au1 * F64(exp32(dua * dua / _BFAC) - F32(1.0))
    t = dla / _DFAC
    al1 = F64(exp32(_sq(t) + atnos / _DB))
    vl1 = al1 * al1 * F64(exp32(dla * dla / _BFAC) - F32(1.0))

    # ---- galactic ---------------------------------------------------------
    gnos = F32(52.0) - F32(23.0) * log10_32(freq)
    dug = dlg = F32(2.0)
    smg = F32(0.5)
    sug = slg = F32(0.2)
    mode_n2 = 1
    t = dug / _DFAC
    au2 = F64(exp32(_sq(t) + gnos / _DB))
    vu2 = au2 * au2 * F64(exp32(dug * dug / _BFAC) - F32(1.0))
    t = dlg / _DFAC
    al2 = F64(exp32(_sq(t) + gnos / _DB))
    vl2 = al2 * al2 * F64(exp32(dlg * dlg / _BFAC) - F32(1.0))
    if freq <= state.fi[3, state.kfx]:   # FI(3,KFX) = F2 critical frequency
        mode_n2 = 0                      # does not penetrate -- drop it
        gnos = F32(0.0)
        au2 = vu2 = al2 = vl2 = F64(0.0)

    # ---- man-made ---------------------------------------------------------
    man = int(noise)
    xnois = F32(man)
    ma = abs(man)
    znoise = xnois
    if man > 0:                          # IF (MAN) 120,114,115 -> 115
        # a 3 MHz level in dB below 1 W: convert to Fam at 1 MHz, then to FREQ
        xnois = _KTB - xnois + F32(13.22)
        xnois = xnois - F32(27.7) * log10_32(freq)
    else:
        if man == 0:                     # 114: -164 dBW on the user card
            ma = 4
        ma = min(4, ma)                  # 120: negative = category index
        xnois = _XNINT[ma - 1] - _CONN[ma - 1] * log10_32(freq)
        znoise = (_KTB - _XNINT[ma - 1]
                  + _CONN[ma - 1] * log10_32(F32(3.0)))
    dum = F32(9.7)
    dlm = F32(6.0)
    sumn = F32(1.5)                      # SUM in the Fortran (a variable)
    smm = F32(5.4)
    slm = F32(1.5)
    t = dum / _DFAC
    au3 = F64(exp32(_sq(t) + xnois / _DB))
    vu3 = au3 * au3 * F64(exp32(dum * dum / _BFAC) - F32(1.0))
    t = dlm / _DFAC
    al3 = F64(exp32(_sq(t) + xnois / _DB))
    vl3 = al3 * al3 * F64(exp32(dlm * dlm / _BFAC) - F32(1.0))

    vu = vu1 + vu2 + vu3
    vl = vl1 + vl2 + vl3
    au = au1 + au2 + au3
    al = al1 + al2 + al3

    # ---- receiver antenna efficiency, for every possible mode -------------
    _gdum, reff = antenna_gain(cant, 2, F32(0.0), freq)
    for im in range(1, 7):               # DO 196 IM=1,6
        zon.eff[im] = reff

    # ---- combine ----------------------------------------------------------
    atnos = atnos - _KTB                 # dB above kTb -> dBW in 1 Hz
    gnos = gnos - _KTB
    xnois = xnois - _KTB
    diag.xn_atnos = atnos
    diag.xn_galactic = gnos
    diag.xn_mmn = xnois

    # the plain power sum, which Caruana's test uses as the sanity bound
    rnse = _DB * log32(pow32(F32(10.0), atnos * F32(0.1))
                       + pow32(F32(10.0), gnos * F32(0.1))
                       + pow32(F32(10.0), xnois * F32(0.1)))

    sigtsqu = np.log(F64(1.0) + vu / (au * au))
    sigtsql = np.log(F64(1.0) + vl / (al * al))
    method_used = 2
    if not (dua <= _TWELVE and dla <= _TWELVE):
        # Spaulding is about to break down: cap each sigma at what the power
        # sum can support.  dxx2/dxx3 non-positive means there is nothing to
        # cap against, so the original value stands.
        dxx = F64(rnse)
        dxx = dxx + F64(204.0)
        dxx2 = F64(2.0) * (np.log(au) - dxx / _DB8)
        dxx3 = F64(2.0) * (np.log(al) - dxx / _DB8)
        if sigtsqu > dxx2 and dxx2 > 0.0:
            method_used = 1
            sigtsqu = dxx2
        if sigtsql > dxx3 and dxx3 > 0.0:
            method_used = 1
            sigtsql = dxx3

    xrnse = F32(F64(_DB) * (np.log(au) - sigtsqu / F64(2.0)) - F64(204.0))
    du = F32(F64(_CFAC) * np.sqrt(sigtsqu))     # upper decile
    dl = F32(F64(_CFAC) * np.sqrt(sigtsql))     # lower decile

    # ---- prediction errors ------------------------------------------------
    # QP* is each component's share of the total power, so the component
    # sigmas combine weighted by how much of the noise they actually are.
    qpa = pow32(F32(10.0), (atnos - xrnse) * F32(0.1))
    qpg = pow32(F32(10.0), (gnos - xrnse) * F32(0.1))
    if mode_n2 == 0:
        qpg = F32(0.0)
    qpm = pow32(F32(10.0), (xnois - xrnse) * F32(0.1))
    sigm = sqrt32(_sq(qpa * sma) + _sq(qpg * smg) + _sq(qpm * smm))

    pv = qpa * exp32((dua - du) * _INV_DB)
    sygu1 = _sq(pv * sua) + _sq((pv - qpa) * sma)
    pv = qpg * exp32((dug - du) * _INV_DB)
    sygu2 = _sq(pv * sug) + _sq((pv - qpg) * smg)
    pv = qpm * exp32((dum - du) * _INV_DB)
    sygu3 = _sq(pv * sumn) + _sq((pv - qpm) * smm)
    sygu = sqrt32(sygu1 + sygu2 + sygu3)

    pv = qpa * exp32((dla - dl) * _INV_DB)
    sygl1 = _sq(pv * sla) + _sq((pv - qpa) * sma)
    pv = qpg * exp32((dlg - dl) * _INV_DB)
    sygl2 = _sq(pv * slg) + _sq((pv - qpg) * smg)   # original used SMA here
    pv = qpm * exp32((dlm - dl) * _INV_DB)
    sygl3 = _sq(pv * slm) + _sq((pv - qpm) * smm)
    sygl = sqrt32(sygl1 + sygl2 + sygl3)

    # The 1991 long-path correction dropped the receiver efficiency from
    # RCNSE (it is added to the signal with the gain instead), so this is
    # XRNSE unmodified.
    anois.rcnse = xrnse
    anois.du = du
    anois.dl = dl
    anois.sigm = sigm
    anois.sygu = sygu
    anois.sygl = sygl
    ton.gnos = gnos
    ton.xnoise = xnois
    ton.znoise = znoise
    diag.mode_n2 = mode_n2
    diag.method_used = method_used
