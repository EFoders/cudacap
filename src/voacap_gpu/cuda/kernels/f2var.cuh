// F2VAR — the F2 layer parameters at every sample area from VERSY(4,5):
// M(3000)F2, the 0.834-critical virtual height, the F2 critical frequency,
// then the E-layer retardation and the (possibly twilight-corrected) F1->F2
// merge that can move HI(2)/YI(2).  The CUDA C twin of scalar/f2var.py.
//
// Consumes the E/F1 slots EF1VAR set and the RAT/ZEN values TIMVAR set, so it
// must run after both.  The Fortran GOTO web (110/210/215/220/225/115/120) is
// preserved here as nested branches, same as Stage A.
//
// REAL*4 throughout; ALOG is the only transcendental.

#ifndef VOACAP_F2VAR_CUH
#define VOACAP_F2VAR_CUH

#include "f32math.cuh"
#include "state.cuh"
#include "versy.cuh"

#define VC_F2VAR_XF1  1.1f
#define VC_F2VAR_DELZ 2.0f

// FAC * ALOG((FAC+1)/(FAC-1)) with FAC already floored — the shared
// retardation kernel; the caller finishes the (.-2.)*semithickness scaling.
__device__ __forceinline__ float f2var_ret_term_(float fac) {
    return fac * log32((fac + 1.0f) / (fac - 1.0f));
}

__device__ inline void f2var(LayerState* state, const int* ikim,
                             int ikim_ncols, const float* ab) {
    for (int ii = 1; ii <= state->km; ++ii) {       // DO 125
        float gamma[7];
        versy(ikim, ikim_ncols, ab, state->clat[ii], state->clong[ii],
              state->gmdip[ii], 4, 5, gamma);
        state->f2m3[ii] = gamma[5];                 // M(3000) factor
        state->hpf2[ii] = 1490.0f / gamma[5] - 176.0f;
        state->fi[3 * 6 + ii] =
            (gamma[4] + 0.5f * state->gyz[ii]) * state->psc[3];
        float zmax = state->zenmax[ii];
        float z = state->zenang[ii];
        // F1 must be less than F2
        state->fi[2 * 6 + ii] =
            pymin(state->fi[2 * 6 + ii], state->fi[3 * 6 + ii] - 0.2f);
        float fc = 0.834f * state->fi[3 * 6 + ii];
        float ec = state->fi[1 * 6 + ii];
        float fcec = fc / ec;
        fcec = pymax(fcec, 1.1f);
        // E layer retardation
        float ret = f2var_ret_term_(fcec);
        ret = (ret - 2.0f) * state->yi[1 * 6 + ii];
        float fc1 = state->fi[2 * 6 + ii];
        if (fc1 > 0.0f) {                           // 110 (else jump to 120)
            float ffec = fc / fc1;
            float rft;
            if (zmax - VC_F2VAR_DELZ - z < 0.0f) {  // 215: near twilight
                ffec = pymax(ffec, VC_F2VAR_XF1);
                float zn = zmax - VC_F2VAR_DELZ;
                float hn = 165.0f + 0.6428f * zn;
                float yn = hn * (state->yi[2 * 6 + ii] / state->hi[2 * 6 + ii]);
                float rfn = f2var_ret_term_(ffec);
                rfn = 0.5f * yn * (rfn - 2.0f);
                float sz = div32(z - zn, VC_F2VAR_DELZ);
                rft = rfn * (1.0f - sz);
                // F2 without F1
                float hm = state->hpf2[ii] - ret;
                float ym = hm / state->rat[ii];
                float dhn = (hm - ym) - (hn - yn);
                if (dhn > 0.0f) {                   // 220: F1 bottom -> F2 bottom
                    float dh = dhn * (1.0f - sz);
                    state->hi[2 * 6 + ii] = (hm - ym) - dh
                                          + state->yi[2 * 6 + ii];
                    if (fc1 - fc > 0.0f) {          // 225: force F1 YM to F2 YM
                        float y1max = yn + (ym - yn)
                            * div32(fc1 / state->fi[3 * 6 + ii] - 0.834f,
                                    0.166f);
                        state->yi[2 * 6 + ii] = yn + (y1max - yn) * sz;
                        state->hi[2 * 6 + ii] = (hm - ym) - dh
                                              + state->yi[2 * 6 + ii];
                    }
                }
            } else {                                // 210: no problem (daytime)
                ffec = pymax(ffec, VC_F2VAR_XF1);
                rft = f2var_ret_term_(ffec);
                rft = 0.5f * state->yi[2 * 6 + ii] * (rft - 2.0f);
            }
            ret = ret + rft;                        // 115
        }
        // 120
        state->hi[3 * 6 + ii] = state->hpf2[ii] - ret;
        state->yi[3 * 6 + ii] = state->hi[3 * 6 + ii] / state->rat[ii];
    }
}

#endif  // VOACAP_F2VAR_CUH
