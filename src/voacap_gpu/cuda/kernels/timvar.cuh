// TIMVAR — per sample area: local mean time, solar zenith angle, then EF1VAR
// for the E/F1 parameters, the F2 height ratio RAT from the NOISY(8) map, and
// the E-layer absorption index ABIY.  The CUDA C twin of scalar/timvar.py.
//
// The SUN table is the blkdat DATA statement (solar declination at the start
// and end of each month), carried here as a device constant exactly as MAGFIN
// carries its G/H/CT tables.  Stored 1-based in both axes so the subscripts
// match the Fortran: SUN(row, month), row 1 = month start, row 2 = month end.
//
// REAL*4 throughout; SIN/COS/ACOS/EXP carry the f32-libm floor.

#ifndef VOACAP_TIMVAR_CUH
#define VOACAP_TIMVAR_CUH

#include "constants.cuh"
#include "ef1var.cuh"
#include "f32math.cuh"
#include "noisy.cuh"
#include "state.cuh"

#define VC_TIMVAR_HR2RAD 0.261799387f   // 15 deg in radians (lon -> hour)

// SUN(3,13), 1-based; row 0 and column 0 are the unused Fortran pad.
__device__ __forceinline__ float timvar_sun_(int row, int month) {
    const float sun[3][13] = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
         0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, -23.05f, -17.30f, -7.88f, 4.26f, 14.84f, 21.93f,
         23.15f, 18.20f, 8.55f, -2.90f, -14.20f, -21.66f},
        {0.0f, -17.31f, -7.89f, 4.21f, 14.83f, 21.93f, 23.45f,
         18.23f, 8.68f, -2.86f, -14.16f, -21.68f, -23.45f},
    };
    return sun[row][month];
}

__device__ inline void timvar(LayerState* state, const int* ikim,
                              int ikim_ncols, const float* ab,
                              const float* anew, const float* bnew,
                              const float* achi, const float* bchi,
                              const float* fakp, const float* fakmap,
                              const float* hmym, const float* fakabp,
                              const float* abmap, float gmt) {
    int month = state->month;
    float ssl = (180.0f - 15.0f * gmt) * VC_D2R;
    if (state->km < 1)                              // 160
        return;
    for (int ii = 1; ii <= state->km; ++ii) {       // DO 155
        float cenlg = state->clong[ii];
        float cenlat = state->clat[ii];
        float cendog = fabsf(cenlat - VC_D2R * timvar_sun_(1, month));
        float ssp = timvar_sun_(2, month) * VC_D2R;
        float cencat = fabsf(cenlat - ssp);
        if (cendog - cencat > 0.0f)                 // 100 (else 105 keeps SUN(2))
            ssp = timvar_sun_(1, month) * VC_D2R;
        // 105: local mean time, wrapped into [0,24)
        float clock = gmt + div32(cenlg, VC_TIMVAR_HR2RAD);
        if (24.0f - clock < 0.0f)                   // 110
            clock = clock - 24.0f;
        else if (clock <= 0.0f)                     // 115 -> 120
            clock = clock + 24.0f;
        state->clck[ii] = clock;                    // 125
        float z = cenlg - ssl;
        float cycen = sin32(cenlat) * sin32(ssp)
                    + cos32(cenlat) * cos32(ssp) * cos32(z);
        cycen = acos32(cycen);                      // zenith angle
        // E and F1 parameters
        cycen = ef1var(state, ikim, ikim_ncols, ab, anew, bnew, achi, bchi,
                       cycen, ii);
        float gm = fabsf(state->glat[ii]) * VC_R2D - 45.0f;
        z = cycen * VC_R2D;
        if (clock - 12.0f < 0.0f)                   // 135
            z = -z;
        z = z + 180.0f;                             // 140
        float ym = noisy(fakp, fakmap, hmym, fakabp, abmap, 8, gm, z);
        if (ym - 2.0f < 0.0f)                       // 145
            ym = 2.0f;
        state->rat[ii] = ym;                        // 150
        float abi = -0.04f + exp32(-2.937f + 0.8445f * state->fi[1 * 6 + ii]);
        state->abiy[ii] = abi;
    }
}

#endif  // VOACAP_TIMVAR_CUH
