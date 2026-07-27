// EF1VAR — E and F1 layer parameters for one sample area, from the foE map
// (VERSY(6,6)) and the F1 regression coefficients ANEW/BNEW/ACHI/BCHI that
// REDMAP loaded.  The CUDA C twin of scalar/ef1var.py.
//
// Mutates the E/F1 slots of the LayerState and RETURNS abs(CYCEN), exactly as
// the Fortran passes the zenith angle back through its argument.  The F1
// layer is present only while the zenith angle is below ZMAX; above it the
// three F1 slots are zeroed, which is the branch IONSET later keys on.
//
// The coefficient vectors arrive already narrowed to REAL*4 by the caller,
// matching Stage A's F32(achi[0]) etc.:
//     anew, bnew  (3,)      achi, bchi  (2,)

#ifndef VOACAP_EF1VAR_CUH
#define VOACAP_EF1VAR_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "state.cuh"
#include "versy.cuh"

#define VC_EF1VAR_BETAE  5.5f
#define VC_EF1VAR_BETAF1 4.0f

// cycen: sun zenith angle (radians).  Returns abs(cycen).
__device__ inline float ef1var(LayerState* state, const int* ikim,
                               int ikim_ncols, const float* ab,
                               const float* anew, const float* bnew,
                               const float* achi, const float* bchi,
                               float cycen, int ii) {
    float gamma[7];
    versy(ikim, ikim_ncols, ab, state->clat[ii], state->clong[ii],
          state->gmdip[ii], 6, 6, gamma);
    float ssn = state->ssn;

    float g6 = gamma[6];
    if (g6 - 0.36f < 0.0f)                  // 85: bad map value -> midnight
        g6 = 0.36f * sqrt32(1.0f + 0.0098f * ssn);
    // 90
    state->fi[1 * 6 + ii] = g6 * state->psc[1];     // E critical frequency
    state->hi[1 * 6 + ii] = 110.0f;                 // E height of max
    state->yi[1 * 6 + ii] = div32(state->hi[1 * 6 + ii], VC_EF1VAR_BETAE);

    cycen = fabsf(cycen);
    float z = cycen * VC_R2D;
    float cosz = cos32(cycen);
    float cosdi = cos32(state->gmdip[ii]);          // cos of Rawer dip
    // maximum zenith angle for the F1 layer
    float zmax = (achi[0] + bchi[0] * ssn)
               + (achi[1] + bchi[1] * ssn) * cosdi;
    state->zenmax[ii] = zmax;
    state->zenang[ii] = z;
    if (z <= zmax) {
        float f1 = (anew[0] + bnew[0] * ssn)
                 + (anew[1] + bnew[1] * ssn) * cosz
                 + (anew[2] + bnew[2] * ssn) * cosz * cosz;
        state->fi[2 * 6 + ii] = f1 * state->psc[2]; // F1 critical frequency
        state->hi[2 * 6 + ii] = 165.0f + 0.6428f * z;
        state->yi[2 * 6 + ii] = div32(state->hi[2 * 6 + ii], VC_EF1VAR_BETAF1);
    } else {
        state->fi[2 * 6 + ii] = 0.0f;
        state->hi[2 * 6 + ii] = 0.0f;
        state->yi[2 * 6 + ii] = 0.0f;
    }
    return cycen;
}

#endif  // VOACAP_EF1VAR_CUH
