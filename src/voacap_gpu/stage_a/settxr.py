"""SETTXR -- price every ray at the transmitter and receiver ends.

The long path's end tables.  For each of the two control areas ITXRCP holds,
SETTXR walks all 45 rays FINDF laid down and fills /LOSX/: absorption,
deviative loss, sporadic-E obscuration, ground loss and antenna gain.  SELTXR
then picks one ray per end out of that, and LNGPAT averages the two.

The loss model is the short path's, per ray rather than per hop -- the same
677.2*ABIY/((f+fH)**1.98 + 10.2) absorption REGMOD uses, the same deviative
term scaled by the ray's AFFLX, the same PRBMUF obscuration for sporadic E --
with one addition at the end: XLS, the over-the-MUF penalty, is added to the
absorption of EVERY ray.  A long path is expected to run above the MUF, and
this is what it costs.

**THERE ARE TWO SETTXRs.**  settxr.for defines SETTXR_ORIG and SETTXR_ALEX and
a dispatcher that picks between them on the eighth character of the version
string.  The vendored database/version.w32 reads "Version 16.1207W", so the
character is 'W' and **the original is what runs**; the port implements the
original.  What the Alex variant changes is worth recording, because two of
the three are bug fixes:

1. it marks dead rays by setting HPFLX to 0 rather than relying on the
   ray-existence scan finding them, and it stops the scan at IAFTXR;
2. it drops the redundant AMIND test in the pricing loop;
3. **it indexes the gyrofrequency as GYZ(LX(K)) with LX = 1,3,5.**  This one
   is a real correction.  /GEOG/ GYZ and ABIY are per SAMPLE area (1..5, laid
   out along the path with 3 at the midpoint), but K here is a CONTROL area
   (1..3), because IONSET repacks /RON/ FI into control-area slots and
   nothing repacks /GEOG/.  So the original reads the gyrofrequency of
   sample area 2 -- 2000 km from the transmitter -- when it wants the
   receiver end.  The two agree only at K=1.  Transliterated as written.

Findings in the original itself:

* **``IS = IMODE(IA,K)`` is dead**, overwritten by ``IS = MODMUF`` on the very
  next line.  The source brackets both with its own rows of question marks.
  So the over-the-MUF probability uses the circuit's controlling layer for
  every ray, never the ray's own.
* **PROB(JJ) is assigned inside the ray loop**, so it ends up holding the last
  priced ray's value -- which costs nothing, because PROS depends only on
  FREQ, YMUF(MODMUF) and MODMUF, none of which vary with the ray.  Every
  iteration writes the same number.
* **AOFX is the only /LOSX/ column not preset.**  TGAINX, ANDVX, ADVX and
  GRLOSX are all reset at the top of each ray; AOFX is written only inside
  the two layer branches, so a ray that fails the existence tests keeps
  whatever obscuration the previous frequency left there.
* **ILN counts priced rays and is never read** -- its only uses are in the
  commented-out MSPEC 125 block.
* the ray-existence scan runs to 45 regardless of IAFTXR, relying on unfilled
  rays having HPFLX = 0 and thus failing the 70 km test.  That is exactly the
  assumption the Alex variant stopped making.
* the ground-loss GAIN call's efficiency output is discarded, and ``Y`` is
  reused for the ground-loss sum and then the antenna gain.

The over-the-MUF fallback OVERWRITES ray 1 of /REFLX/ -- take-off angle,
heights, vertical frequency, deviative factor, ground distance and layer --
so SETTXR mutates the ray table it is reading.  It fires when no ray passes
all three existence tests, which for an untouched table means immediately,
since ray 1's HPFLX is then 0.

float32; COS/SQRT/EXP/ALOG/ALOG10 and the ``**1.98`` powers all carry the
[[f32-libm-noise]] floor.
"""

from voacap_gpu.stage_a.constants import D2R, RZ
from voacap_gpu.stage_a.f32math import (F32, cos32, exp32, log10_32, log32,
                                        pow32, sqrt32)
from voacap_gpu.stage_a.gain import antenna_gain, ground_loss
from voacap_gpu.stage_a.prbmuf import prbmuf

_P198 = F32(1.98)           # a default-REAL literal, so powf, not a multiply
_NRAY = 45
_MODE_HEIGHT = F32(70.0)    # a ray below this is not a mode
_ES_LAYER = 4               # PRBMUF's sporadic-E index


class ModmufError(IndexError):
    """MODMUF is outside 1..4, so DELMUF/YMUF are read off the end.

    MODMUF is CURMUF's controlling layer and is always 1..4 in a run that got
    this far; a zero means CURMUF never ran.  Refused rather than reproduced.
    """


def settxr(state, muf, sigd, reflx, losx, rays, cant, allm, itxrcp,
           freq, gcdkm, amind) -> None:
    """Fill /LOSX/ for both control areas in place (SETTXR_ORIG).

    state: LayerState (GYZ/ABIY/FI/FS/HS/KM/SIGPAT/EPSPAT); muf: MufState;
    sigd: SigdState (HTLOSS/XNUZ/HNU/AFE/BFE/XVE); reflx: ReflxState (read,
    and ray 1 OVERWRITTEN by the over-the-MUF fallback); losx: LosxState
    (output); rays: RaysState (IFOB, for the Es dead frequency); cant:
    CantennaState; allm: AllModeState (PROB(1) and PROB(2) written); itxrcp:
    the 1-based (3,) control-area indices; freq in MHz; gcdkm in km; amind
    the minimum take-off angle in degrees.
    """
    freq = F32(freq)
    gcdkm = F32(gcdkm)
    amind = F32(amind)
    # .....FOR MAX. NUMBER OF HOPS
    dend = min(F32(4000.0), gcdkm)
    for jj in (1, 2):
        k = int(itxrcp[jj])
        for ia in range(1, _NRAY + 1):
            reflx.gml[ia, k] = F32(-999.0)
            reflx.fhp[ia, k] = F32(999.0)
        # --- DO 80: does ANY ray exist?  first hit jumps to 95 -------------
        found = False
        for ia in range(1, _NRAY + 1):
            # .....DOES THE MODE EXIST (QUESTION MARK)
            if reflx.hpflx[ia, k] - _MODE_HEIGHT < 0.0:   # 90: give up at the
                break                                     # FIRST dead ray
            # .....NUMBER OF HOPS
            xhpm = dend / reflx.gdflx[ia, k]              # 75
            # .....MUST HAVE AT LEAST ONE FOR SHORT PATHS
            if xhpm - F32(0.9) < 0.0:                     # 80: next ray
                continue
            if reflx.delfx[ia, k] - amind >= 0.0:         # 95: a mode exists
                found = True
                break
        if not found:
            #  PENETRATED LAYER, TRY OVER THE MUF MODE.
            # .....SO IF NONE FAILS, INCLUDE OVER-THE-MUF
            im = int(muf.modmuf)
            if not 1 <= im <= 4:
                raise ModmufError(
                    f"MODMUF = {im} indexes DELMUF(1..4) out of range")
            reflx.delfx[1, k] = muf.delmuf[im]
            reflx.hpflx[1, k] = muf.hpmuf[im]
            reflx.htflx[1, k] = muf.htmuf[im]
            reflx.fvflx[1, k] = muf.fvmuf[im]
            reflx.afflx[1, k] = muf.afmuf[im]
            xhop = F32(muf.nhopmf[im])
            reflx.gdflx[1, k] = gcdkm / xhop
            reflx.imode[1, k] = muf.modmuf
        # --- 95: price every ray ------------------------------------------
        bc = pow32(freq + state.gyz[k], _P198)      # GYZ(K): see the docstring
        ac = F32(677.2) * state.abiy[k]
        for ia in range(1, _NRAY + 1):
            # .....PRESET ARRAYS  (AOFX is deliberately not among them)
            losx.tgainx[ia, k] = F32(-10.0)
            losx.andvx[ia, k] = F32(1000.0)
            losx.advx[ia, k] = F32(1000.0)
            losx.grlosx[ia, k] = F32(0.0)
            # .....MODE GOOD (QUESTION MARK)  ONE IS.
            if reflx.hpflx[ia, k] < _MODE_HEIGHT:
                continue
            # .....CHECK FOR MINIMUM
            if reflx.delfx[ia, k] < amind:
                continue
            del_ = reflx.delfx[ia, k] * D2R         # radiation angle
            cdel = cos32(del_)
            # .....E MODE (QUESTION MARK)
            if state.fi[1, k] < reflx.fvflx[ia, k]:
                #  F LAYER MODES                                        (125)
                xnsq = F32(10.2)
                sinp = RZ * cdel / (RZ + F32(100.0))
                secp = F32(1.0) / sqrt32(F32(1.0) - sinp * sinp)
                # .....ABSORPTION LOSS
                losx.andvx[ia, k] = secp * ac / (bc + xnsq)
                sinp = RZ * cdel / (RZ + reflx.hpflx[ia, k])
                secp = F32(1.0) / sqrt32(F32(1.0) - sinp * sinp)
                # .....DEVIATIVE LOSS
                losx.advx[ia, k] = (
                    secp * reflx.afflx[ia, k]
                    * (pow32(reflx.fvflx[ia, k] + state.gyz[k], _P198) + xnsq)
                    / (bc + xnsq))
                #  ES OBSCURTION LOSS
                losx.aofx[ia, k] = F32(0.0)
                # .....SPORADIC E EXISTS (QUESTION MARK)
                if state.fs[2, k] > 0.0:
                    # IFOB is the reflectrix in kHz; slot (1,3) is the Es
                    # "dead" frequency below which the layer blanks the mode
                    fsdead = F32(rays.ifob[1, 3, k])
                    fsdead = fsdead / F32(1000.0)
                    fmhz = max(freq, fsdead)
                    sins = RZ * cdel / (RZ + state.hs[k])
                    secs = F32(1.0) / sqrt32(F32(1.0) - sins * sins)
                    # .....MUF THIS DISTANCE
                    esd = state.fs[2, k] * secs
                    p = prbmuf(muf, fmhz, esd, muf.ymuf[_ES_LAYER], _ES_LAYER)
                    p = max(F32(0.1), p)
                    p = min(F32(0.9), p)
                    # .....OBSCURATION
                    losx.aofx[ia, k] = F32(-10.0) * log10_32(F32(1.0) - p)
            else:
                #  D-E  LAYER MODES.
                # .....VERY LOW REFLECTIONS (QUESTION MARK)
                if sigd.htloss <= reflx.htflx[ia, k]:
                    xnsq = F32(10.2)
                else:
                    hnux = F32(61.0) + F32(3.0) * (
                        reflx.htflx[ia, k] - F32(70.0)) / F32(18.0)
                    # .....COLLISION FREQUENCY
                    xnsq = sigd.xnuz * exp32(
                        F32(-2.0) * (hnux - F32(60.0)) / sigd.hnu)
                heff = min(F32(100.0), reflx.htflx[ia, k])
                sinp = RZ * cdel / (RZ + heff)
                secp = F32(1.0) / sqrt32(F32(1.0) - sinp * sinp)
                # .....ABSORPTION LOSS
                losx.andvx[ia, k] = secp * ac / (bc + xnsq)
                xv = max(reflx.fvflx[ia, k] / state.fi[1, k], sigd.xve)
                # .....CORRECTION FOR E LAYER
                adx = sigd.afe + sigd.bfe * log32(xv)
                sinp = RZ * cdel / (RZ + reflx.hpflx[ia, k])
                secp = F32(1.0) / sqrt32(F32(1.0) - sinp * sinp)
                # .....DEVIATIVE LOSS + CORRECTION
                losx.advx[ia, k] = (
                    secp * reflx.afflx[ia, k]
                    * (pow32(reflx.fvflx[ia, k] + state.gyz[k], _P198) + xnsq)
                    / (bc + xnsq)) + adx
                # .....ES OBSCURATION, NONE
                losx.aofx[ia, k] = F32(0.0)
            #  GROUND LOSS                                              (160)
            y = F32(0.0)
            for ig in range(1, int(state.km) + 1):
                y = y + ground_loss(state, ig, del_, freq)
            xkm = F32(state.km)
            # .....AVERAGE GROUND LOSS
            losx.grlosx[ia, k] = y / xkm
            #  ANTENNA GAINS
            y, teff = antenna_gain(cant, jj, del_, freq)
            losx.tgainx[ia, k] = y
            if jj == 2:
                losx.efflp[ia] = teff       # ray index only -- the RX end
            sphet = RZ * cdel / (RZ + reflx.htflx[ia, k])
            cphet = F32(1.0) - sphet * sphet
            cphet = max(F32(0.00000001), cphet)
            cphet = sqrt32(cphet)
            # C-----?????????????????
            # IS = IMODE(IA,K) is dead: overwritten on the next source line.
            is_ = int(muf.modmuf)
            # C-----?????????????????
            if not 1 <= is_ <= 4:
                raise ModmufError(
                    f"MODMUF = {is_} indexes YMUF(1..4) out of range")
            pros = prbmuf(muf, freq, muf.ymuf[is_], muf.ymuf[is_], is_)
            # .....OVER-THE-MUF, MEDIAN
            xls = F32(-10.0) * log10_32(pros) / cphet
            #  ADD ADJUSTMENT TO ABSORPTION LOSS
            losx.andvx[ia, k] = losx.andvx[ia, k] + xls
            allm.prob[jj] = pros        # same value every ray; see docstring
            # ILN = ILN + 1 -- counted and never read
