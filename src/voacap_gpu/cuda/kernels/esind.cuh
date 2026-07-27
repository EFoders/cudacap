// ESIND — sporadic-E critical-frequency deciles at every sample area, from
// VERSY(1,3), each scaled by PSC(4) (0.70 for median losses; 0.0 disables Es
// entirely, which is the VOACAP-vs-ICEPAC toggle).  Reflection height is a
// fixed 110 km.  The CUDA C twin of scalar/esind.py.
//
// Note the deliberate index reversal: GAMMA(3) is the LOWER decile and
// GAMMA(1) the UPPER, exactly as the Fortran stores them.
//
// REAL*4 throughout; VERSY carries the only transcendentals.

#ifndef VOACAP_ESIND_CUH
#define VOACAP_ESIND_CUH

#include "f32math.cuh"
#include "state.cuh"
#include "versy.cuh"

__device__ inline void esind(LayerState* state, const int* ikim,
                             int ikim_ncols, const float* ab) {
    if (state->km < 1)                          // 105
        return;
    float psc4 = state->psc[4];
    for (int ii = 1; ii <= state->km; ++ii) {   // DO 100
        state->hs[ii] = 110.0f;                 // height of reflection
        float gamma[7];
        versy(ikim, ikim_ncols, ab, state->clat[ii], state->clong[ii],
              state->gmdip[ii], 1, 3, gamma);
        state->fs[1 * 6 + ii] = gamma[3] * psc4;    // lower decile
        state->fs[2 * 6 + ii] = gamma[2] * psc4;    // median
        state->fs[3 * 6 + ii] = gamma[1] * psc4;    // upper decile
    }
}

#endif  // VOACAP_ESIND_CUH
