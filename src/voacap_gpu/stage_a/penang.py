"""PENANG -- the penetration take-off angle DELPEN for each layer (E/F1/F2).

For the operating frequency FREQ, DELPEN(IL,K) is the take-off angle above
which the ray penetrates layer IL at control area K -- found by combining the
law of sines, Snell's law and the reflectrix-height increments.  Each layer
uses its cusp height/frequency (HTRUE/FVERT at index 10 for E, 20 for F1);
the F2 test additionally probes indices 28/29/30 to honour the spherical
symmetry of the thick F2 layer, taking the highest refractive height.  A layer
whose vertical frequency exceeds FREQ is fully penetrated (DELPEN forced to
89.9/90); if even the F2 nose penetrates, DSKPKM/DMAXKM are flagged -1.

Writes /REFLX/ DELPEN (and the -1 skip/max flags).  float32; ACOS carries the
usual [[f32-libm-noise]] floor, ill-conditioned as CDEL -> 1.  Mirrors
penang.for.
"""

from voacap_gpu.stage_a.constants import R2D, RZ
from voacap_gpu.stage_a.f32math import F32, acos32, sqrt32

_ONE = F32(1.0)
_CDEL_MAX = F32(0.999999)
_FRAT_MAX = F32(0.9999)


def _cdel_angle(dp, il, k, ht, fv, freq):
    """Layer cusp angle: DELPEN(il,k) from height HT and vertical freq FV.
    Returns True if the layer reflects (caller continues), False if it
    penetrates near-vertically (DELPEN already set to 89.9, caller returns)."""
    frat = fv / freq
    frat = frat * frat
    if frat - _FRAT_MAX >= 0.0:   # penetrates
        return False
    cdel = (RZ + ht) * sqrt32(_ONE - frat) / RZ
    if cdel - _CDEL_MAX <= 0.0:
        dp[il, k] = acos32(cdel) * R2D
    else:
        dp[il, k] = F32(0.0)
    return True


def _f2_penetration(state, reflx, k, freq, fmhz):
    """Label 155: set DELPEN(3,K) for the F2 layer."""
    dp = reflx.delpen
    htrue, fvert = state.htrue, state.fvert
    # IF(FVERT(30,K)-FREQ+0.0001) 156,160,160
    if fvert[30, k] - freq + F32(0.0001) < 0.0:  # F2 penetrated -> probe nose
        t28, t29, t30 = (fvert[28, k] / fmhz, fvert[29, k] / fmhz,
                         fvert[30, k] / fmhz)
        xm28 = (RZ + htrue[28, k]) * sqrt32(_ONE - t28 * t28)
        xm29 = (RZ + htrue[29, k]) * sqrt32(_ONE - t29 * t29)
        xm30 = (RZ + htrue[30, k]) * sqrt32(_ONE - t30 * t30)
        # take the highest refractive height (branch-faithful selection)
        if xm30 - xm29 < 0.0:            # IF(XM30-XM29) 161,157,157
            cdel = xm28 if xm29 - xm28 < 0.0 else xm29  # 161
        else:                            # 157
            cdel = xm28 if xm30 - xm28 < 0.0 else xm30
        cdel = cdel / RZ
        if cdel - _CDEL_MAX <= 0.0:      # 170
            dp[3, k] = acos32(cdel) * R2D
        else:                            # 175  F2 nose penetrates
            dp[3, k] = F32(0.0)
            reflx.dskpkm[k] = F32(-1.0)
            reflx.dmaxkm[k] = F32(-1.0)
    else:                                # 160  F2 reflects
        dp[3, k] = F32(89.9)


def penang(state, reflx, k, freq) -> None:
    """state: LayerState (HTRUE/FVERT/FI); reflx: ReflxState (out); k: area;
    freq: operating frequency (MHz)."""
    freq = F32(freq)
    fmhz = freq
    dp = reflx.delpen
    htrue, fvert, fi = state.htrue, state.fvert, state.fi
    # E LAYER cusp (index 10)
    if not _cdel_angle(dp, 1, k, htrue[10, k], fvert[10, k], freq):
        dp[1, k] = F32(89.9)
        dp[2, k] = F32(90.0)
        dp[3, k] = F32(90.0)
        return
    # IF(FI(2,K)) 145,145,180
    if fi[2, k] <= 0.0:  # no F1: F1 shares the E penetration angle
        dp[2, k] = dp[1, k]
        _f2_penetration(state, reflx, k, freq, fmhz)
        return
    # F1 LAYER present (index 20)
    if not _cdel_angle(dp, 2, k, htrue[20, k], fvert[20, k], freq):
        dp[2, k] = F32(89.9)
        dp[3, k] = F32(90.0)
        return
    _f2_penetration(state, reflx, k, freq, fmhz)
