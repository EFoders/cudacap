"""SELTXR -- choose one ray at each long-path end.

SETTXR priced all 45 rays; SELTXR picks the one LNGPAT will use, and writes
its index into LTXRGM.  Three tie-breaks in order:

1. **smallest gain minus loss.**  GNOW is the antenna gain less the whole path
   loss times the hop count, so bigger is better; two rays within GMIN (3 dB)
   of each other are treated as indistinguishable and fall through to...
2. **closest to specular reflection.**  YHOP is how far the hop count is from
   a whole number, folded so that 0.7 hops counts as 0.3 away.  Two rays
   within YMIN (0.1 hop) fall through to...
3. **closest to the a-priori optimum take-off angle** DELOPT, 3 degrees.

Findings in the original, and all three are real:

* **XHPM AND IHOP ARE COMPUTED FOR RAY 1 AND THEN USED FOR WHATEVER RAY THE
  SEARCH LANDS ON.**  Label 105 computes them from GDFLX(LTXRGM,K) with
  LTXRGM still 1.  The search at 110/112 then advances LTXRGM looking for a
  ray that exists and clears the minimum angle -- WITHOUT recomputing either.
  So the incumbent's hop count XHPM, its integer part IHOP and therefore its
  fractional hop YHPM all belong to ray 1, while the gain-minus-loss GMAX at
  label 120 uses the found ray's TGAINX/ANDVX/ADVX.  The Alex variant
  restructures exactly this (its label 320 recomputes XHPM for the ray it
  found).  Transliterated as written; test_gmloss names it.
* **TURNING ON THE DIAGNOSTIC CHANGES THE ANSWER.**  ``IAND(INFO,4)`` sets
  IST to 1, which runs the whole selection TWICE -- once with a 0.05 partial
  hop threshold, once with 0.5 -- to print whether the two agree.  The second
  pass resets LTXRGM to IDNXS but NOT XHPM or YHPM, which the first pass has
  already moved at label 160.  So the second pass starts from a different
  incumbent than a single pass would, and can select a different ray.  A
  debugging flag with an observable effect on the prediction.
* **LTXRGM CAN NEVER COME BACK ZERO.**  The give-up path at label 95
  decrements it, but it can only be reached after at least one increment at
  110, so the result is at worst back to 1.  LNGPAT's ``LTXRGM <= 0`` early
  return is therefore dead in the real program -- which is what LNGPAT's own
  comment, "LTXRGM(1) .GE. 1 AND LTXRGM(2) .GE. 1 SINCE OVER-THE-MUF ADDED",
  is saying.

**SELTXR disagrees with SETTXR about exactly 70 km.**  Here the tests are
``HPFLX .LE. 70`` and ``IF(HPFLX-70.) 165,165,125``, both of which kill a ray
at exactly 70; SETTXR's existence scan writes ``IF(HPFLX-70.) 90,75,75``,
which keeps it.  The same disagreement BABS and TABS have, between a different
pair of routines.

Like SETTXR, this is a dispatcher over an original and an Alex variant chosen
on the version string's eighth character; the vendored version.w32 says 'W',
so the ORIGINAL runs and is what is ported here.

GML and FHP are filled as a side effect -- the per-ray gain-minus-loss and
fractional hop -- and **nothing reads them except LNGPAT's diagnostic
listing**.  They exist for the debug output.

float32 throughout; no libm calls at all, so this one is pure arithmetic.
"""

from voacap_gpu.scalar.f32math import F32

_NRAY = 45
_MODE_HEIGHT = F32(70.0)


class RayIndexError(IndexError):
    """The find-first search walked past ray 45.

    Label 110 increments LTXRGM and reads HPFLX at the new index BEFORE the
    IAFTXR bound test, so a table whose rays all exist but all sit under the
    minimum take-off angle drives the index to IAFTXR+1 -- and with IAFTXR at
    45 that reads off the end of the column, into the next control area's
    ray 1 (or, for area 3, out of /REFLX/ entirely).  Refused here.
    """


def seltxr(reflx, losx, lpath, itxrcp, gcdkm, amind, info: int = 0) -> None:
    """Fill LTXRGM (and GML/FHP) in place (SELTXR_ORIG).

    reflx: ReflxState (DELFX/HPFLX/GDFLX/IAFTXR in, GML/FHP out); losx:
    LosxState (TGAINX/ANDVX/ADVX); lpath: LpathState (DELOPT/GMIN/YMIN in,
    LTXRGM out); itxrcp: the 1-based (3,) control-area indices; gcdkm in km;
    amind the minimum take-off angle in degrees; info the /INFORM/ diagnostic
    mask -- bit 4 doubles the selection pass and CHANGES THE RESULT.
    """
    gcdkm = F32(gcdkm)
    amind = F32(amind)
    for jj in (1, 2):
        k = int(itxrcp[jj])
        dend = min(gcdkm, F32(4000.0))
        lpath.ltxrgm[jj] = 1
        # .....ONE MODE GOOD, AS OVER THE MUF FORCED
        # .....FIND FIRST GOOD MODE (START)                              (105)
        xhpm = dend / reflx.gdflx[lpath.ltxrgm[jj], k]
        ihop = int(xhpm)                    # REAL->INT truncates toward zero
        gave_up = False
        while xhpm - F32(0.9) < 0.0:        # IF(XHPM-0.9) 110,115,115
            lpath.ltxrgm[jj] = lpath.ltxrgm[jj] + 1              # 110
            if lpath.ltxrgm[jj] > _NRAY:
                raise RayIndexError(
                    f"SELTXR walked to ray {int(lpath.ltxrgm[jj])} at area "
                    f"{k}: every ray exists and none clears AMIND")
            # .....CHECK MINIMUM ANGLE
            if reflx.hpflx[lpath.ltxrgm[jj], k] <= _MODE_HEIGHT:
                gave_up = True                                   # -> 95
                break
            if lpath.ltxrgm[jj] - reflx.iaftxr[k] > 0:
                gave_up = True                                   # -> 95
                break
            if reflx.delfx[lpath.ltxrgm[jj], k] - amind >= 0.0:  # 112
                break                                            # -> 115
            # else fall round to 110 again; XHPM is NOT recomputed
        if gave_up:
            lpath.ltxrgm[jj] = lpath.ltxrgm[jj] - 1              # 95
            continue
        # .....FIND FIRST GOOD MODE (END)                                (115)
        hopx = F32(ihop)
        yhpm = xhpm - hopx
        ist = 2
        idnxs = int(lpath.ltxrgm[jj])
        if info & 4:
            ist = 1             # ... and the two passes are NOT independent
        for idf in range(ist, 3):
            if idf == 2:
                xfract = F32(0.5)
                lpath.ltxrgm[jj] = idnxs
            else:
                xfract = F32(0.05)
            if yhpm - xfract > 0.0:
                # .....HOP INCREMENT
                yhpm = F32(1.0) - yhpm                           # 116
            # .....GAIN-LOSS                                             (120)
            ia0 = int(lpath.ltxrgm[jj])
            gmax = losx.tgainx[ia0, k] - xhpm * (losx.andvx[ia0, k]
                                                 + losx.advx[ia0, k])
            # .....ANGLE INCREMENT
            delmax = abs(reflx.delfx[ia0, k] - lpath.delopt)
            ls = int(lpath.ltxrgm[jj])
            for ia in range(ls, int(reflx.iaftxr[k]) + 1):
                # .....DOES THE MODE EXIST (QUESTION MARK)
                if reflx.hpflx[ia, k] - _MODE_HEIGHT <= 0.0:
                    continue                                     # 165
                # .....NUMBER OF HOPS                                    (125)
                xhop = dend / reflx.gdflx[ia, k]
                ihop = int(xhop)
                # .....AT LEAST NEAR 1
                if xhop - F32(0.9) < 0.0:
                    continue                                     # 165
                hopx = F32(ihop)                                 # 126
                yhop = xhop - hopx
                if yhop - F32(0.5) > 0.0:
                    # .....HOP INCREMENT
                    yhop = F32(1.0) - yhop                       # 130
                # .....GAIN-LOSS                                         (135)
                gnow = losx.tgainx[ia, k] - xhop * (losx.andvx[ia, k]
                                                    + losx.advx[ia, k])
                reflx.gml[ia, k] = gnow
                reflx.fhp[ia, k] = yhop
                # .....ANGLE INCREMENT
                delnow = abs(reflx.delfx[ia, k] - lpath.delopt)
                #  FIRST CHOICE ON SMALLEST GAIN MINUS LOSS.
                if abs(gnow - gmax) - lpath.gmin > 0.0:          # 140
                    if gnow - gmax <= 0.0:
                        continue
                else:
                    # SECOND CHOICE ON CLOSEST TO SPECULAR REFLECTION
                    # (INTEGER NO. OF HOPS).                             (145)
                    if abs(yhop - yhpm) - lpath.ymin > 0.0:      # 150
                        if yhop - yhpm >= 0.0:
                            continue
                    else:
                        # THIRD CHOICE ON CLOSEST TO APRIORI OPTIMUM
                        # TAKEOFF ANGLE.                                 (155)
                        if delnow - delmax >= 0.0:
                            continue
                lpath.ltxrgm[jj] = ia                            # 160
                xhpm = xhop
                gmax = gnow
                delmax = delnow
                yhpm = yhop
