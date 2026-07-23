"""Stage B (PyTorch, batched) combined receiver noise — GENOIS.

Three independent noise sources -- atmospheric, galactic and man-made -- each
described by a median level and a pair of decile ratios, have to come out as a
SINGLE median with a single pair of deciles.  They cannot simply be added:
summing lognormal variables is a moment problem.  So each component is turned
into a linear mean A and a variance V, those are summed, and the total is
inverted on the assumption that it is itself lognormal (Spaulding's method,
ITS Report 87-212).  Above about 12 dB of atmospheric decile that inversion
runs away, and the sigmas are capped at what the plain power sum can support
instead -- Caruana's April 2007 ITU-R P.372-8 modification, recorded per lane
in ``method_used``.

THIS ROUTINE IS ALL BRANCH AND NO LOOP, which is the opposite of most of Stage
B.  There is no height index, no ray index, nothing to unroll: every lane does
a fixed sequence of about a dozen EXPs.  What varies per lane is WHICH arm of
four separate two-way splits it takes -- no long-term data base, galactic noise
blocked by the F2 critical, the man-made noise card being a level or a category
index, and Caruana's cap firing on either sigma.  All of them are computed on
every lane and selected, which is free here because none of the arms is
expensive and none of them can produce a NaN that would poison the other.

Two of those four are scalar properties of a RUN rather than of a grid point --
the coefficient set either has a long-term data base or it does not, and the
NOISE card is one number for the whole prediction.  They are still taken as
per-lane tensors.  That is not generality for its own sake: the oracle dump
varies both across its cases, and taking them per-lane is what lets the dump be
replayed as one batch with every arm live at once.

PRECISION IS THE POINT OF THIS FILE.  Stage A measured BIT-EXACT on all 11,970
cells of the dump, and that is load-bearing evidence for three rules that are
easy to get wrong and invisible when you do:

* **Only A, V and the sigmas are REAL*8.**  genois.for declares those and
  nothing else.  Every EXP feeding the sums is REAL*4 and only its RESULT
  widens -- ``_exp32(...).to(float64)``, never ``exp`` of a float64 argument.
  Widening the EXP itself changes A in the 8th digit and the cancellation in
  ``log(1 + V/A**2)`` amplifies it.
* **``**2`` is an INTEGER exponent and ``10.**x`` is a REAL one.**  gfortran
  emits a multiply for the first and powf for the second.  ``_sq`` and
  ``_pow32`` are separate functions here for that reason and must not be
  merged.
* **``4.34294`` and ``4.34294d0`` are DIFFERENT constants**, and GENOIS uses
  both -- the first promoted from binary32 in XRNSE, the second a true double
  in Caruana's test.  ``_DB_AS_F64`` and ``_DB8`` below are those two, and
  unifying them is a change to the physics at the 8th digit.

The transcendentals are evaluated in float64 and rounded once to float32,
matching Stage A's ``f32math`` rather than torch's own f32 libm; that is what
carries the bit-exactness across, and it also makes CPU and CUDA agree.

The receiver antenna efficiency is fetched from GAIN here too -- incidentally
rather than by right, because this is the one routine per frequency guaranteed
to run.  It is returned as a single (N,) tensor; the caller broadcasts it into
/ZON/ EFF slots 1..6 and MUST leave slot 7 alone, which belongs to ESMOD.

``NYEAR = ' 2089'`` selects GENOIS_OLD, the pre-2007 model.  It is a deliberate
operator opt-in to superseded physics and is not ported here any more than it
is in Stage A: :class:`~voacap_gpu.stage_a.genois.OldNoiseModel` is raised.
"""

import numpy as np
import torch

from voacap_gpu.stage_a.genois import OldNoiseModel
from voacap_gpu.stage_b.gain import antenna_gain_batch
from voacap_gpu.stage_b.genfam import _log10_32, _pow32, genfam_batch
from voacap_gpu.stage_b.profile import _k

_F32 = torch.float32
_F64 = torch.float64

# DATA DFAC,BFAC,CFAC /7.87384, 30.99872, 5.56765/
#   DFAC = SQRT(2 * 1.282**2 * 4.34294**2)   BFAC = (1.282 * 4.34294)**2
#   CFAC = 4.34294 * 1.282
_DFAC = 7.87384
_BFAC = 30.99872
_CFAC = 5.56765
_DB = 4.34294               # 10/ln(10) as a REAL*4 literal: nepers <-> dB
_INV_DB = 0.23026           # 1/4.34294, also a REAL*4 literal
_KTB = 204.0                # dB from kTb (290 K, 1 Hz) to 1 W
_TWELVE = 12.0              # the decile above which Spaulding breaks down

# The two 4.34294s.  XRNSE promotes the REAL*4 literal; Caruana's test writes
# 4.34294d0.  They differ in the 8th digit and the Fortran uses BOTH.
_DB_AS_F64 = float(np.float64(np.float32(_DB)))
_DB8 = 4.34294
_CFAC_AS_F64 = float(np.float64(np.float32(_CFAC)))

# ALOG10(3.) as gfortran folds it -- the float32 nearest to log10(3).
_LOG10_3 = float(np.float32(np.log10(np.float64(np.float32(3.0)))))

# Man-made noise at 1 MHz and its decay exponent, by CCIR environment.
#      #1          #2          #3      #4      #5     #6
#  industrial residential   rural   remote   noisy  quiet
_XNINT = (76.8, 72.5, 67.2, 53.6, 83.2, 65.2)
_CONN = (27.7, 27.7, 27.7, 28.6, 37.5, 29.1)
# Categories 5 and 6 are unreachable: MA is clamped with MIN0(4,MA).  Kept
# because the DATA statements carry them and a future NOISE range might.


def _sq(x):
    """``x**2`` with an INTEGER exponent -- gfortran emits a multiply, not
    powf, so this must not go through :func:`_pow32`."""
    return x * x


def _exp32(x):
    """REAL*4 EXP: evaluated in float64, rounded ONCE to float32.

    The argument is float32 and the result is float32.  Widening the argument
    to float64 and keeping the answer there is a different routine -- see the
    module docstring on which quantities genois.for actually declares REAL*8.
    """
    return torch.exp(x.to(_F64)).to(_F32)


def _log32(x):
    """REAL*4 ALOG: evaluated in float64, rounded once to float32."""
    return torch.log(x.to(_F64)).to(_F32)


def _k64(value, like):
    """A constant as a float64 SCALAR TENSOR on the right device.

    The float64 counterpart of :func:`~voacap_gpu.stage_b.profile._k`, for the
    same reason: a Python float operand is not reliably the constant you meant
    once CUDA type promotion has had a look at it.
    """
    return torch.tensor(float(value), dtype=_F64, device=like.device)


def _table(values, like):
    """A DATA-statement table as a float32 tensor, for per-lane gathers."""
    return torch.tensor([float(np.float32(v)) for v in values],
                        dtype=_F32, device=like.device)


def genois_batch(fam, dud, nodata, kj, jk, atnu, atny, cc, tm,
                 rlat, freq, noise, fi3, cant, nyear=""):
    """Batched GENOIS.  Returns a dict of tensors, all (N,).

    fam/dud: /TWO/ FAM and DUD, shared (14,12)/(5,12,5) or per-lane with a
        trailing lane axis -- see ``stage_b.genfam``.
    nodata: (N,) bool, ``F2D(1,1,1) < 0`` -- no long-term data base.  A
        property of the coefficient set, hence constant over a real run; taken
        per-lane so the dump's IZF cases replay in one batch.
    kj/jk:  (N,) int64 local-time blocks bracketing the receiver time.
    atnu/atny: (N,) float32 Fam at 1 MHz for those two blocks (ANOIS1).
    cc/tm:  (N,) float32 the two local times; |CC-TM|/4 is the interpolant.
    rlat:   (N,) float32 receiver latitude, radians (GENFAM uses its sign).
    freq:   (N,) float32 MHz.
    noise:  (N,) int64 NOISE card: > 0 a 3 MHz level in dB below 1 W,
            0 the -164 default, < 0 a CCIR environment index.
    fi3:    (N,) float32 FI(3,KFX), the F2 critical frequency.
    cant:   ``BatchedCantenna``; entry ITR=2 is the receiver.
    nyear:  the /ALPHA/ year string; ' 2089' is refused.

    Keys: ``rcnse du dl sigm sygu sygl`` (/ANOIS/), ``gnos xnoise znoise``
    (/TON/), ``eff`` (GAIN's receiver efficiency, for /ZON/ EFF 1..6 -- NOT
    slot 7), ``xn_atnos xn_galactic xn_mmn`` (the component breakdown),
    ``mode_n2`` and ``method_used`` (int64 flags).
    """
    if nyear == " 2089":
        raise OldNoiseModel(
            "NYEAR ' 2089' selects GENOIS_OLD (the pre-2007 noise model), "
            "which is not ported")

    k = {n: _k(v, freq) for n, v in (
        ("0", 0.0), ("1", 1.0), ("0.1", 0.1), ("3", 3.0), ("4", 4.0),
        ("9", 9.0), ("7", 7.0), ("1.5", 1.5), ("10", 10.0), ("13.22", 13.22),
        ("23", 23.0), ("27.7", 27.7), ("30", 30.0), ("52", 52.0),
        ("55", 55.0),
        ("204", _KTB), ("dfac", _DFAC), ("bfac", _BFAC), ("db", _DB),
        ("invdb", _INV_DB), ("log10_3", _LOG10_3),
        ("2", 2.0), ("0.5", 0.5), ("0.2", 0.2),
        ("9.7", 9.7), ("6", 6.0), ("5.4", 5.4))}
    zero64 = _k64(0.0, freq)
    dume = torch.minimum(freq, k["55"])         # AMIN1(FREQ,55.)

    # ---- atmospheric ------------------------------------------------------
    # GENFAM's outputs alias the /ANOIS/ DU/DL/SIGM/SYGU/SYGL in the Fortran,
    # but all five are reassigned unconditionally before anything reads them,
    # so the intermediate store is dead and they stay local.
    atnz, du, dl, sigm, sygu, sygl = genfam_batch(fam, dud, rlat, kj,
                                                  dume, atnu)
    atnx, dx, dq, sigz, sigx, sigsq = genfam_batch(fam, dud, rlat, jk,
                                                   dume, atny)
    slop = torch.abs(cc - tm) / k["4"]          # interpolate between blocks
    # Label 85, the no-data-base arm: ATNOS=204. is immediately overwritten
    # with 0. on the next line, so the atmospheric term contributes 10**(-204/10)
    # W once the -204 dB shift is applied, i.e. nothing.  The deciles are still
    # the fixed 9/7 dB, so the spread does not collapse with the level.
    def _pick(fixed, computed):
        return torch.where(nodata, k[fixed], computed)

    atnos = _pick("0", atnz + (atnx - atnz) * slop)
    dua = _pick("9", du + (dx - du) * slop)
    dla = _pick("7", dl + (dq - dl) * slop)
    sma = _pick("3", sigm + (sigz - sigm) * slop)
    sua = _pick("1.5", sygu + (sigx - sygu) * slop)
    sla = _pick("1.5", sygl + (sigsq - sygl) * slop)

    def _av(d, level):
        """One component's lognormal mean A and variance V, both REAL*8.

        (D/DFAC)**2 = (D/1.282)**2 / (2*4.34294**2) and D*D/BFAC =
        (D/1.282)**2 / 4.34294**2.  BOTH EXPs ARE REAL*4 and only the result
        widens; the ``- 1.`` is a float32 subtract too, before the widening.
        """
        a = _exp32(_sq(d / k["dfac"]) + level / k["db"]).to(_F64)
        v = a * a * (_exp32(d * d / k["bfac"]) - k["1"]).to(_F64)
        return a, v

    au1, vu1 = _av(dua, atnos)
    al1, vl1 = _av(dla, atnos)

    # ---- galactic ---------------------------------------------------------
    gnos = k["52"] - k["23"] * _log10_32(freq)
    dug = dlg = k["2"]
    smg = k["0.5"]
    sug = slg = k["0.2"]
    au2, vu2 = _av(dug, gnos)
    al2, vl2 = _av(dlg, gnos)
    # At or below foF2 galactic noise cannot penetrate to the receiver, so the
    # component is dropped ENTIRELY -- after A and V have been formed from the
    # live GNOS, exactly as the Fortran orders it.
    blocked = freq <= fi3
    mode_n2 = torch.where(blocked, torch.zeros_like(kj), torch.ones_like(kj))
    gnos = torch.where(blocked, torch.zeros_like(gnos), gnos)
    au2, vu2, al2, vl2 = (torch.where(blocked, zero64, x)
                          for x in (au2, vu2, al2, vl2))

    # ---- man-made ---------------------------------------------------------
    man = noise
    xnois0 = man.to(_F32)                       # XNOIS = MAN
    # MA=IABS(MAN); label 114 sets MA=4 for the -164 default; label 120 clamps
    # it with MIN0(4,MA).  On a MAN > 0 lane MA is formed and never read, but
    # the gather below is unconditional so it still has to land in the table --
    # IABS after the MAN=0 substitution is already >= 1, so the lower clamp can
    # never fire and is there to keep a stray negative out of a wrapped index
    # rather than to reproduce anything the Fortran does.
    ma = torch.where(man == 0, torch.full_like(man, 4), man.abs())
    ma = torch.clamp(ma, min=1, max=4)
    xnint = _table(_XNINT, freq)[ma - 1]
    conn = _table(_CONN, freq)[ma - 1]

    lf = _log10_32(freq)
    # 115: a 3 MHz level in dB below 1 W -> Fam at 1 MHz -> Fam at FREQ.  Two
    # statements in the Fortran, so ((204 - X) + 13.22) - 27.7*log10(f).
    xnois_lvl = (k["204"] - xnois0 + k["13.22"]) - k["27.7"] * lf
    # 120: a category index.  ZNOISE is the equivalent 3 MHz level.
    xnois_cat = xnint - conn * lf
    znoise_cat = (k["204"] - xnint) + conn * k["log10_3"]
    is_level = man > 0
    xnois = torch.where(is_level, xnois_lvl, xnois_cat)
    znoise = torch.where(is_level, xnois0, znoise_cat)

    dum = k["9.7"]
    dlm = k["6"]
    sumn = k["1.5"]                             # SUM in the Fortran
    smm = k["5.4"]
    slm = k["1.5"]
    au3, vu3 = _av(dum, xnois)
    al3, vl3 = _av(dlm, xnois)

    vu = vu1 + vu2 + vu3
    vl = vl1 + vl2 + vl3
    au = au1 + au2 + au3
    al = al1 + al2 + al3

    # ---- receiver antenna efficiency, for every possible mode -------------
    # GAIN clamps only IP1 on the frequency axis (MIN0(30,IP1)); I = IFIX(FMC)
    # itself is unclamped, so FMC >= 31 reads ARRAY past the end of the pattern
    # table and picks up whatever COMMON follows it -- the mirror image of the
    # bottom-clamp defect gain.py already refuses.  It only shows up here
    # because GENOIS is the one routine that runs at EVERY frequency, band or
    # no band, and the dump goes to 55 MHz.
    #
    # The pattern tables stop at 30 MHz, so no honest /cantenna/ entry covers
    # 31 and up: the lane matches nothing and GAIN's answer is 0 regardless of
    # what the stray read returned.  Those lanes are therefore given a
    # well-defined 0 and a safe in-table lookup frequency, rather than either
    # reproducing the read or crashing on the batched gather.  A card that
    # declared XFQE above 31 would get 0 here where the Fortran gets garbage.
    oob = freq >= 31.0
    fgain = torch.where(oob, k["30"], freq)
    _gdum, reff = antenna_gain_batch(cant, 2, torch.zeros_like(freq), fgain)
    reff = torch.where(oob, torch.zeros_like(reff), reff)

    # ---- combine ----------------------------------------------------------
    atnos = atnos - k["204"]                    # dB above kTb -> dBW in 1 Hz
    gnos = gnos - k["204"]
    xnois = xnois - k["204"]

    # the plain power sum, which Caruana's test uses as the sanity bound
    rnse = k["db"] * _log32((_pow32(k["10"], atnos * k["0.1"])
                             + _pow32(k["10"], gnos * k["0.1"]))
                            + _pow32(k["10"], xnois * k["0.1"]))

    one64, two64 = _k64(1.0, freq), _k64(2.0, freq)
    sigtsqu = torch.log(one64 + vu / (au * au))
    sigtsql = torch.log(one64 + vl / (al * al))

    # Caruana: above 12 dB of atmospheric decile the implied lognormal sigma
    # exceeds what the power sum can support and the median runs away, so cap
    # each sigma at the power sum's own value.  A non-positive DXX2/DXX3 means
    # there is nothing to cap against and the original stands.  4.34294d0 here
    # is the DOUBLE literal, not the one XRNSE promotes below.
    dxx = rnse.to(_F64) + _k64(_KTB, freq)
    dxx2 = two64 * (torch.log(au) - dxx / _k64(_DB8, freq))
    dxx3 = two64 * (torch.log(al) - dxx / _k64(_DB8, freq))
    caruana = ~((dua <= _TWELVE) & (dla <= _TWELVE))
    capu = caruana & (sigtsqu > dxx2) & (dxx2 > 0.0)
    capl = caruana & (sigtsql > dxx3) & (dxx3 > 0.0)
    sigtsqu = torch.where(capu, dxx2, sigtsqu)
    sigtsql = torch.where(capl, dxx3, sigtsql)
    method_used = torch.where(capu | capl, torch.ones_like(kj),
                              torch.full_like(kj, 2))

    # 4.34294 and 5.56765 promoted from REAL*4 -- see the module docstring
    xrnse = (_k64(_DB_AS_F64, freq) * (torch.log(au) - sigtsqu / two64)
             - _k64(_KTB, freq)).to(_F32)
    cfac64 = _k64(_CFAC_AS_F64, freq)
    du = (cfac64 * torch.sqrt(sigtsqu)).to(_F32)        # upper decile
    dl = (cfac64 * torch.sqrt(sigtsql)).to(_F32)        # lower decile

    # ---- prediction errors ------------------------------------------------
    # QP* is each component's share of the total power, so the component sigmas
    # combine weighted by how much of the noise they actually are.
    qpa = _pow32(k["10"], (atnos - xrnse) * k["0.1"])
    qpg = _pow32(k["10"], (gnos - xrnse) * k["0.1"])
    qpg = torch.where(mode_n2 == 0, torch.zeros_like(qpg), qpg)
    qpm = _pow32(k["10"], (xnois - xrnse) * k["0.1"])
    # SQRT on float32 is correctly rounded in IEEE-754, torch and gfortran
    # alike, so it needs no float64 detour.
    sigm_o = torch.sqrt(_sq(qpa * sma) + _sq(qpg * smg) + _sq(qpm * smm))

    pv = qpa * _exp32((dua - du) * k["invdb"])
    sygu1 = _sq(pv * sua) + _sq((pv - qpa) * sma)
    pv = qpg * _exp32((dug - du) * k["invdb"])
    sygu2 = _sq(pv * sug) + _sq((pv - qpg) * smg)
    pv = qpm * _exp32((dum - du) * k["invdb"])
    sygu3 = _sq(pv * sumn) + _sq((pv - qpm) * smm)
    sygu_o = torch.sqrt(sygu1 + sygu2 + sygu3)

    pv = qpa * _exp32((dla - dl) * k["invdb"])
    sygl1 = _sq(pv * sla) + _sq((pv - qpa) * sma)
    pv = qpg * _exp32((dlg - dl) * k["invdb"])
    sygl2 = _sq(pv * slg) + _sq((pv - qpg) * smg)   # the original used SMA
    pv = qpm * _exp32((dlm - dl) * k["invdb"])
    sygl3 = _sq(pv * slm) + _sq((pv - qpm) * smm)
    sygl_o = torch.sqrt(sygl1 + sygl2 + sygl3)

    # The 1991 long-path correction dropped the receiver efficiency from RCNSE
    # (it is added to the signal with the gain instead), so this is XRNSE as
    # computed, and REFF travels separately.
    return {"rcnse": xrnse, "du": du, "dl": dl, "sigm": sigm_o,
            "sygu": sygu_o, "sygl": sygl_o,
            "gnos": gnos, "xnoise": xnois, "znoise": znoise,
            "eff": reff,
            "xn_atnos": atnos, "xn_galactic": gnos, "xn_mmn": xnois,
            "mode_n2": mode_n2, "method_used": method_used}
