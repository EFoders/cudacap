// GETHP — virtual and true reflection heights for vertical frequency FXX,
// by NT-point Gaussian integration of 1/mu over the LECDEN profile; the
// CUDA C twin of scalar/gethp.py (gethp.for).  REAL*4; only SQRT
// (correctly rounded) appears, so GETHP is bit-clean.
//
// The two monotonicity-aware table searches (the true-height crossing of FR
// in FNSQ, labels 445..495, and the plasma-frequency lookup at each
// integration node, labels 535..625) keep Stage A's branch structure with
// gotos on the original label numbers.  HTR/FNSQ are the (51, 4) row-major
// 1-based arrays inside LayerState: element [ih, 1] is htr[ih * 4 + 1].
//
// The Stage A expressions ``zg ** NPL`` / ``zg ** (NPL - 1)`` with NPL = 1
// are the exact identities zg and 1.0 (verified: np.float32 ** 1 is the
// identity and ** 0 is 1.0 bitwise on the reference numpy), so they are
// folded here with the multiply-by-one kept where Stage A multiplies.

#ifndef VOACAP_GETHP_CUH
#define VOACAP_GETHP_CUH

#include "f32math.cuh"
#include "state.cuh"
#include "datr.cuh"

// HT where FR crosses FNSQ (label 445..495).
__device__ inline float gethp_true_height_(const LayerState* s, float fr) {
    const float* htr = s->htr;
    const float* fnsq = s->fnsq;
    for (int ih = 1; ih <= 49; ++ih) {  // DO 490 IH=1,49
        float d = fnsq[ih * 4 + 1] - fnsq[(ih + 1) * 4 + 1];
        if (d < 0.0f) {                             // 465 increasing
            if (fnsq[ih * 4 + 1] - fr > 0.0f)       // -> 475
                goto L475;
            if (fr - fnsq[(ih + 1) * 4 + 1] < 0.0f) // 470 -> 475
                goto L475;
        } else if (d == 0.0f) {                     // 455 flat
            if (fr - fnsq[ih * 4 + 1] == 0.0f)      // 460
                return htr[ih * 4 + 1];
        } else {                                    // 480 decreasing
            if (fnsq[ih * 4 + 1] - fr >= 0.0f) {    // 485
                if (fr - fnsq[(ih + 1) * 4 + 1] > 0.0f)  // -> 475
                    goto L475;
            }
        }
        continue;
L475:   // interp
        return htr[ih * 4 + 1] + (fr - fnsq[ih * 4 + 1])
               * (htr[(ih + 1) * 4 + 1] - htr[ih * 4 + 1])
               / (fnsq[(ih + 1) * 4 + 1] - fnsq[ih * 4 + 1]);
    }
    return htr[50 * 4 + 1];
}

// Plasma frequency squared at height ZI (label 535..625).
__device__ inline float gethp_interp_fnsq_(const LayerState* s, float zi) {
    const float* htr = s->htr;
    const float* fnsq = s->fnsq;
    if (zi - htr[1 * 4 + 1] <= 0.0f)                // 545
        return fnsq[1 * 4 + 1];
    for (int ih = 1; ih <= 49; ++ih) {  // DO 615
        float dh = htr[ih * 4 + 1] - htr[(ih + 1) * 4 + 1];
        if (dh < 0.0f) {                            // 585 increasing height
            if (htr[ih * 4 + 1] - zi > 0.0f)        // -> 600
                goto L600;
            if (zi - htr[(ih + 1) * 4 + 1] < 0.0f)  // 595 -> 600
                goto L600;
        } else if (dh == 0.0f) {                    // 565 flat
            if (zi - htr[ih * 4 + 1] == 0.0f)       // 575
                return fnsq[ih * 4 + 1];
        } else {                                    // 605 decreasing height
            if (htr[ih * 4 + 1] - zi >= 0.0f
                && zi - htr[(ih + 1) * 4 + 1] > 0.0f)  // 610
                goto L600;
        }
        continue;
L600:   // interp
        return fnsq[ih * 4 + 1] + (zi - htr[ih * 4 + 1])
               * (fnsq[(ih + 1) * 4 + 1] - fnsq[ih * 4 + 1])
               / (htr[(ih + 1) * 4 + 1] - htr[ih * 4 + 1]);
    }
    return fnsq[50 * 4 + 1];
}

// Returns (HPX virtual height, HTX true height) in km through the pointers.
__device__ inline void gethp(const LayerState* s, float fxx,
                             float* hpx_out, float* htx_out) {
    float fr = fxx * fxx;
    if (fr - s->fnsq[1 * 4 + 1] <= 0.0f) {          // 435: below the profile
        float htx = s->htr[1 * 4 + 1];
        *hpx_out = htx;
        *htx_out = htx;
        return;
    }
    float ht = gethp_true_height_(s, fr);
    float hp = 0.0f;
    ht = ht - s->htr[1 * 4 + 1];
    float hrmz = ht * DATR_TWDIV * DATR_XNPL;
    for (int ig = 1; ig <= DATR_NT; ++ig) {         // DO 635 Gaussian sum
        float zg = 1.0f - DATR_XT[ig];
        float zi = ht * (1.0f - DATR_TWDIV * zg) + s->htr[1 * 4 + 1];
        float ysq = gethp_interp_fnsq_(s, zi) / fr;
        ysq = pymin(ysq, 0.9999f);
        float ymup = (1.0f / sqrt32(1.0f - ysq)) * 1.0f;  // * zg**(NPL-1)
        zg = 1.0f + DATR_XT[ig];
        zi = ht * (1.0f - DATR_TWDIV * zg) + s->htr[1 * 4 + 1];
        ysq = gethp_interp_fnsq_(s, zi) / fr;
        ysq = pymin(ysq, 0.9999f);
        float zmup = (1.0f / sqrt32(1.0f - ysq)) * 1.0f;  // * zg**(NPL-1)
        hp = hp + DATR_WT[ig] * (ymup + zmup);
    }
    *hpx_out = s->htr[1 * 4 + 1] + hrmz * hp;
    *htx_out = s->htr[1 * 4 + 1] + ht;
}

#endif  // VOACAP_GETHP_CUH
