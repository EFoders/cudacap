// PENANG — the penetration take-off angle DELPEN for each layer (E/F1/F2),
// the CUDA C twin of scalar/penang.py (mirrors penang.for).
//
// For the operating frequency FREQ, DELPEN(IL,K) is the take-off angle above
// which the ray penetrates layer IL at control area K.  Each layer uses its
// cusp height/frequency (HTRUE/FVERT at index 10 for E, 20 for F1); the F2
// test additionally probes indices 28/29/30, taking the highest refractive
// height.  A layer whose vertical frequency exceeds FREQ is fully penetrated
// (DELPEN forced 89.9/90); if even the F2 nose penetrates, DSKPKM/DMAXKM are
// flagged -1.  Writes ReflxState DELPEN (+ the -1 flags).  REAL*4.

#ifndef VOACAP_PENANG_CUH
#define VOACAP_PENANG_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "state.cuh"

#define PENANG_CDEL_MAX 0.999999f
#define PENANG_FRAT_MAX 0.9999f

// Divisions with a COMPILE-TIME-CONSTANT denominator (here /RZ) must go
// through __fdiv_rn: NVRTC strength-reduces `x / 6370.0f` into a multiply by
// the reciprocal even under --prec-div=true (measured on this GPU: 1-ULP
// wrong quotients, amplified ~6 ULP through the ill-conditioned ACOS below).
// __fdiv_rn is the IEEE round-to-nearest f32 division and matches numpy
// bit-for-bit.  Runtime denominators keep the plain `/`.

// Layer cusp angle: DELPEN(il,k) from height HT and vertical freq FV.
// Returns true if the layer reflects (caller continues), false if it
// penetrates near-vertically (caller sets DELPEN = 89.9 and returns).
__device__ inline bool penang_cdel_angle_(float* dp, int il, int k, float ht,
                                          float fv, float freq) {
    float frat = fv / freq;
    frat = frat * frat;
    if (frat - PENANG_FRAT_MAX >= 0.0f)   // penetrates
        return false;
    float cdel = div32((VC_RZ + ht) * sqrt32(1.0f - frat), VC_RZ);
    if (cdel - PENANG_CDEL_MAX <= 0.0f)
        dp[il * 6 + k] = acos32(cdel) * VC_R2D;
    else
        dp[il * 6 + k] = 0.0f;
    return true;
}

// Label 155: set DELPEN(3,K) for the F2 layer.
__device__ inline void penang_f2_penetration_(const LayerState* state,
                                              ReflxState* reflx, int k,
                                              float freq, float fmhz) {
    float* dp = reflx->delpen;
    const float* htrue = state->htrue;
    const float* fvert = state->fvert;
    // IF(FVERT(30,K)-FREQ+0.0001) 156,160,160
    if (fvert[30 * 6 + k] - freq + 0.0001f < 0.0f) {  // F2 penetrated -> nose
        float t28 = fvert[28 * 6 + k] / fmhz;
        float t29 = fvert[29 * 6 + k] / fmhz;
        float t30 = fvert[30 * 6 + k] / fmhz;
        float xm28 = (VC_RZ + htrue[28 * 6 + k]) * sqrt32(1.0f - t28 * t28);
        float xm29 = (VC_RZ + htrue[29 * 6 + k]) * sqrt32(1.0f - t29 * t29);
        float xm30 = (VC_RZ + htrue[30 * 6 + k]) * sqrt32(1.0f - t30 * t30);
        // take the highest refractive height (branch-faithful selection)
        float cdel;
        if (xm30 - xm29 < 0.0f)            // IF(XM30-XM29) 161,157,157
            cdel = (xm29 - xm28 < 0.0f) ? xm28 : xm29;  // 161
        else                               // 157
            cdel = (xm30 - xm28 < 0.0f) ? xm28 : xm30;
        cdel = div32(cdel, VC_RZ);
        if (cdel - PENANG_CDEL_MAX <= 0.0f) {  // 170
            dp[3 * 6 + k] = acos32(cdel) * VC_R2D;
        } else {                               // 175  F2 nose penetrates
            dp[3 * 6 + k] = 0.0f;
            reflx->dskpkm[k] = -1.0f;
            reflx->dmaxkm[k] = -1.0f;
        }
    } else {                               // 160  F2 reflects
        dp[3 * 6 + k] = 89.9f;
    }
}

// state: LayerState (HTRUE/FVERT/FI); reflx: ReflxState (out); k: area;
// freq: operating frequency (MHz).
__device__ inline void penang(const LayerState* state, ReflxState* reflx,
                              int k, float freq) {
    float fmhz = freq;
    float* dp = reflx->delpen;
    const float* htrue = state->htrue;
    const float* fvert = state->fvert;
    const float* fi = state->fi;
    // E LAYER cusp (index 10)
    if (!penang_cdel_angle_(dp, 1, k, htrue[10 * 6 + k], fvert[10 * 6 + k],
                            freq)) {
        dp[1 * 6 + k] = 89.9f;
        dp[2 * 6 + k] = 90.0f;
        dp[3 * 6 + k] = 90.0f;
        return;
    }
    // IF(FI(2,K)) 145,145,180
    if (fi[2 * 6 + k] <= 0.0f) {  // no F1: F1 shares the E penetration angle
        dp[2 * 6 + k] = dp[1 * 6 + k];
        penang_f2_penetration_(state, reflx, k, freq, fmhz);
        return;
    }
    // F1 LAYER present (index 20)
    if (!penang_cdel_angle_(dp, 2, k, htrue[20 * 6 + k], fvert[20 * 6 + k],
                            freq)) {
        dp[2 * 6 + k] = 89.9f;
        dp[3 * 6 + k] = 90.0f;
        return;
    }
    penang_f2_penetration_(state, reflx, k, freq, fmhz);
}

#endif  // VOACAP_PENANG_CUH
