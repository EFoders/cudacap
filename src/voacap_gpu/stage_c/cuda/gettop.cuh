// GETTOP — how much of a long path the ray flies OVER the E layer; the CUDA C
// twin of stage_a/gettop.py.  A daylit middle can reflect a frequency the dark
// ends let straight through; that stretch never sees the D-region absorption
// underneath.  The three sample areas are read at the E layer only (HI(1,*)
// and FI(1,*)); each vertical E critical is projected to the oblique
// penetration frequency at DELOPT, the secant clamped at sin^2 = 0.9999.
//
// Five tests in a row, each of which can zero the answer.  GMKM is written on
// every path, so no caller reads an uninitialised output.  The two 0.001 MHz
// slacks pull opposite ways and only the lower one changes an answer.
//
// float32; COS and SQRT carry the [[f32-libm-noise]] floor.  No constant
// denominators — every division is by a runtime value.

#ifndef VOACAP_GETTOP_CUH
#define VOACAP_GETTOP_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "state.cuh"

// GETTOP(DIKM,GMKM) — returns GMKM, the over-the-top distance in km.  state
// reads HI(1,1..3) and FI(1,1..3); lpath reads DELOPT; freq in MHz; dikm the
// ionospheric distance in km.  Returns 0 when no over-the-top segment
// qualifies.  HI/FI are (4,6) row-major: element [1, jf] = arr[1*6 + jf].
__device__ inline float gettop(const LayerState* state,
                               const LpathState* lpath, float freq,
                               float dikm) {
    float del = lpath->delopt * VC_D2R;
    float rcosd = VC_RZ * cos32(del);
    float fpe[4];                       // 1-based, FPE(3)
    // FIND DISTANCE THAT SUPPORT  E (M)  MODE.
    for (int jf = 1; jf <= 3; ++jf) {
        float ht = state->hi[1 * 6 + jf];
        float fv = state->fi[1 * 6 + jf];
        float sphe = rcosd / (VC_RZ + ht);
        sphe = sphe * sphe;
        sphe = pymin(sphe, 0.9999f);
        // .....E PENETRATION FREQUENCIES
        fpe[jf] = fv / sqrt32(1.0f - sphe);
    }
    float fp1 = pymax(fpe[1], fpe[3]);  // .....END
    float fp2 = fpe[2];                 // .....MIDDLE
    float fp3 = pymin(fpe[1], fpe[3]);  // .....END
    // .....THE MIDDLE IS --
    if (fp1 - fp2 >= 0.0f)                      // IF(FP1-FP2) 170,235,235
        return 0.0f;
    if (freq - fp2 - 0.001f >= 0.0f)            // IF(FREQ-FP2-.001) 175,235,235
        return 0.0f;
    if (freq - fp1 + 0.001f <= 0.0f)            // IF(FREQ-FP1+.001) 235,235,185
        return 0.0f;
    float gmkm = 0.5f * dikm * (fp2 - freq)
                 * (1.0f / (fp2 - fp1) + 1.0f / (fp2 - fp3));
    if (gmkm - 1000.0f <= 0.0f)                 // IF(GMKM-1000.) 235,235,225
        return 0.0f;
    // .....MUST BE ABLE TO GET UP THERE FOR M MODE
    if (dikm - gmkm - 1000.0f <= 0.0f)          // IF(DIKM-GMKM-1000.) 235,235,240
        return 0.0f;
    return gmkm;
}

#endif  // VOACAP_GETTOP_CUH
