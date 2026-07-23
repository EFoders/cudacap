// PEN — retardation (penetration) term for a parabolic layer I at sample
// area K, evaluated at frequency F; the CUDA C twin of stage_a/pen.py
// (pen.for).  REAL*4; ALOG via log32.
//
// FI/YI are the (4, 6) row-major 1-based arrays inside LayerState:
// element [i, k] is fi[i * 6 + k].

#ifndef VOACAP_PEN_CUH
#define VOACAP_PEN_CUH

#include "f32math.cuh"
#include "state.cuh"

__device__ inline float pen(const LayerState* s, int i, int k, float f) {
    float x = f / s->fi[i * 6 + k];
    x = pymax(x, 1.001f);
    return s->yi[i * 6 + k] * log32((1.0f + x) / (x - 1.0f)) * x;
}

#endif  // VOACAP_PEN_CUH
