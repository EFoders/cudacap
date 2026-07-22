"""Stage A transliteration of GETHP
(reference/voacapl/src/voacapw/gethp.for).

Virtual and true reflection heights for vertical frequency FXX, by NT-point
Gaussian integration of 1/mu over the LECDEN profile.  Two monotonicity-
aware table searches (the true-height crossing of FR in FNSQ, and the
plasma-frequency lookup at each integration node) are the arithmetic-IF
GOTO webs 455..490 and 535..615; they are lifted verbatim into helpers.
REAL*4; only SQRT (correctly rounded) appears, so GETHP is bit-clean.
"""

from voacap_gpu.stage_a.datr import NPL, NT, TWDIV, WT, XNPL, XT
from voacap_gpu.stage_a.f32math import F32, sqrt32


def _true_height(state, fr):
    """HT where FR crosses FNSQ (label 445..495)."""
    htr, fnsq = state.htr, state.fnsq
    for ih in range(1, 50):  # DO 490 IH=1,49
        def interp():
            return (htr[ih, 1] + (fr - fnsq[ih, 1])
                    * (htr[ih + 1, 1] - htr[ih, 1])
                    / (fnsq[ih + 1, 1] - fnsq[ih, 1]))
        d = fnsq[ih, 1] - fnsq[ih + 1, 1]
        if d < 0.0:  # 465 increasing
            if fnsq[ih, 1] - fr > 0.0:  # 475
                return interp()
            if fr - fnsq[ih + 1, 1] < 0.0:  # 470 -> 475
                return interp()
        elif d == 0.0:  # 455 flat
            if fr - fnsq[ih, 1] == 0.0:  # 460
                return htr[ih, 1]
        else:  # 480 decreasing
            if fnsq[ih, 1] - fr >= 0.0:  # 485
                if fr - fnsq[ih + 1, 1] > 0.0:  # 475
                    return interp()
    return htr[50, 1]


def _interp_fnsq(state, zi):
    """Plasma frequency squared at height ZI (label 535..625)."""
    htr, fnsq = state.htr, state.fnsq
    if zi - htr[1, 1] <= 0.0:  # 545
        return fnsq[1, 1]
    for ih in range(1, 50):  # DO 615
        def interp():
            return (fnsq[ih, 1] + (zi - htr[ih, 1])
                    * (fnsq[ih + 1, 1] - fnsq[ih, 1])
                    / (htr[ih + 1, 1] - htr[ih, 1]))
        dh = htr[ih, 1] - htr[ih + 1, 1]
        if dh < 0.0:  # 585 increasing height
            if htr[ih, 1] - zi > 0.0:  # 600
                return interp()
            if zi - htr[ih + 1, 1] < 0.0:  # 595 -> 600
                return interp()
        elif dh == 0.0:  # 565 flat
            if zi - htr[ih, 1] == 0.0:  # 575
                return fnsq[ih, 1]
        else:  # 605 decreasing height
            if htr[ih, 1] - zi >= 0.0 and zi - htr[ih + 1, 1] > 0.0:  # 610
                return interp()
    return fnsq[50, 1]


def gethp(state, fxx):
    """Returns (HPX virtual height, HTX true height) in km."""
    fr = F32(fxx) * F32(fxx)
    if fr - state.fnsq[1, 1] <= 0.0:  # 435: below the profile
        htx = state.htr[1, 1]
        return htx, htx
    ht = _true_height(state, fr)
    hp = F32(0.0)
    ht = ht - state.htr[1, 1]
    hrmz = ht * TWDIV * XNPL
    if NT >= 1:
        for ig in range(1, NT + 1):  # DO 635 Gaussian sum
            zg = F32(1.0) - XT[ig]
            zi = ht * (F32(1.0) - TWDIV * (zg ** NPL)) + state.htr[1, 1]
            ysq = _interp_fnsq(state, zi) / fr
            ysq = min(ysq, F32(0.9999))
            ymup = (F32(1.0) / sqrt32(F32(1.0) - ysq)) * (zg ** (NPL - 1))
            zg = F32(1.0) + XT[ig]
            zi = ht * (F32(1.0) - TWDIV * (zg ** NPL)) + state.htr[1, 1]
            ysq = _interp_fnsq(state, zi) / fr
            ysq = min(ysq, F32(0.9999))
            zmup = (F32(1.0) / sqrt32(F32(1.0) - ysq)) * (zg ** (NPL - 1))
            hp = hp + WT[ig] * (ymup + zmup)
    hpx = state.htr[1, 1] + hrmz * hp
    htx = state.htr[1, 1] + ht
    return hpx, htx
