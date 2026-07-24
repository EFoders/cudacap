// SELMOD — select the controlling sample area (JMODE); the CUDA C twin of
// scalar/selmod.py.  Pure comparison on KFX and the FI criticals: KFX==1
// has only one area; KFX==2 lets the higher E critical decide between the
// two ends; KFX>=3 compares the ends (areas 1 and 3) on the F2 critical,
// with the E critical breaking a tie within 0.01 MHz.  Returns the 1-based
// area index.

#ifndef VOACAP_SELMOD_CUH
#define VOACAP_SELMOD_CUH

#include "state.cuh"

__device__ inline int selmod(const LayerState* s) {
    const float* fi = s->fi;                // (4, 6): fi[layer * 6 + area]
    if (s->kfx == 1)
        return 1;
    if (s->kfx == 2) {
        // receiver end (2) controls if its E critical is lower
        return (fi[1 * 6 + 1] > fi[1 * 6 + 2]) ? 2 : 1;
    }
    // KFX >= 3: check the ends (areas 1 and 3) only
    float delfi = fi[3 * 6 + 1] - fi[3 * 6 + 3];
    if (fabsf(delfi) < 0.01f) {
        // F2 layers match -> E layer decides
        return (fi[1 * 6 + 1] > fi[1 * 6 + 3]) ? 3 : 1;
    }
    // F2 layer controls
    return (delfi > 0.0f) ? 3 : 1;
}

#endif  // VOACAP_SELMOD_CUH
