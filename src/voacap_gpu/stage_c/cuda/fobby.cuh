// FOBBY — the reflectrix table for sample area K.  The CUDA C twin of
// stage_a/fobby.py (mirrors fobby.for).
//
// For each restricted take-off angle ANG(1..NANG) and each of the 30
// vertical-sounding frequencies, applies the secant law using the TRUE
// reflection height (so no ionospheric K-factor) to get the oblique
// operating frequency, stored as integer kHz in IFOB.  The per-hop,
// per-angle table this builds is what FINDF later searches to place each
// mode.  REAL*4; the IFOB store is the Fortran REAL->INT truncation of
// 1000.*FREQ, kept exactly as Stage A wrote it: (int)(1000.0f * freq).

#ifndef VOACAP_FOBBY_CUH
#define VOACAP_FOBBY_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "state.cuh"

__device__ inline void fobby(const LayerState* state, RaysState* rays,
                             int k) {
    const float* fvert = state->fvert;
    const float* htrue = state->htrue;
    const float* ang = rays->ang;
    int* ifob = rays->ifob;
    for (int ia = 1; ia <= rays->nang; ++ia) {  // DO 650 IA=1,NANG
        float del_ = ang[ia] * VC_D2R;
        float rcosd = VC_RZ * cos32(del_);
        for (int ih = 1; ih < 31; ++ih) {       // DO 645 IH=1,30
            float fv = fvert[ih * 6 + k];
            // true height so no K factor
            float sphe = rcosd / (VC_RZ + htrue[ih * 6 + k]);
            float sqcos = 1.0f - sphe * sphe;
            sqcos = pymax(0.000001f, sqcos);    // AMAX1(.000001, SQCOS)
            float cphe = sqrt32(sqcos);
            float freq = fv / cphe;
            // REAL->INT truncates
            ifob[(ia * 31 + ih) * 6 + k] = (int)(1000.0f * freq);
        }
    }
}

#endif  // VOACAP_FOBBY_CUH
