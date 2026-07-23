// BENDY — bending correction for a parabolic layer I at sample area K,
// evaluated at frequency F; the CUDA C twin of stage_a/bendy.py (bendy.for).
// REAL*4; ALOG via log32.
//
// FI/YI are the (4, 6) row-major 1-based arrays inside LayerState:
// element [i, k] is fi[i * 6 + k].

#ifndef VOACAP_BENDY_CUH
#define VOACAP_BENDY_CUH

#include "f32math.cuh"
#include "state.cuh"

__device__ inline float bendy(const LayerState* s, int i, int k, float f) {
    float x = f / s->fi[i * 6 + k];
    x = pymin(x, 0.999f);
    return 0.5f * x * s->yi[i * 6 + k] * log32((1.0f + x) / (1.0f - x));
}

#endif  // VOACAP_BENDY_CUH
