// GENION — generates the ionogram for sample area K; the CUDA C twin of
// stage_a/genion.py (genion.for).
//
// Lays out 30 vertical-sounding frequencies FVERT across the D-E tail, the
// E nose, and the F region from the E/F1/F2 critical frequencies FI(1/2/3,K)
// (spacing depends on whether an F1 layer is present), then fills the
// virtual (HPRIM) and true (HTRUE) reflection heights at each frequency:
//
//   * IEDP < 0  : full NT-point Gaussian integration (GETHP) at every
//                 frequency (label 600);
//   * IEDP >= 0 : the fast model (label 705) — a fixed 10-point parabolic-E
//                 table (HTE/HPE), true heights by XLIN lookup into the
//                 LECDEN profile, virtual heights from BENDY/PEN when no F1
//                 layer is present, else GETHP.
//
// Mutates FVERT/HPRIM/HTRUE in LayerState ((31, 6) row-major, [iff*6 + k]).
// REAL*4; SQRT correctly rounded, BENDY/PEN/GETHP as their own ports.

#ifndef VOACAP_GENION_CUH
#define VOACAP_GENION_CUH

#include "f32math.cuh"
#include "state.cuh"
#include "xlin.cuh"
#include "datr.cuh"
#include "gethp.cuh"
#include "bendy.cuh"
#include "pen.cuh"

// True (HTE) and virtual (HPE) heights for the E layer with HI=110, YI=20 —
// the fixed 10-point table used by the fast model (DATA HTE/HPE in
// genion.for), 1-based (index 0 padded).
__device__ __constant__ float GENION_HTE[11] = {
    0.0f, 70.00f, 84.51f, 88.52f, 93.00f, 94.26f, 95.71f, 97.68f,
    100.27f, 104.22f, 107.19f,
};
__device__ __constant__ float GENION_HPE[11] = {
    0.0f, 70.00f, 87.40f, 94.00f, 98.68f, 100.20f, 102.65f, 107.15f,
    113.74f, 125.25f, 140.03f,
};

__device__ inline void genion(LayerState* s, int k) {
    float* fi = s->fi;
    float* yi = s->yi;
    float* hi = s->hi;
    float* fvert = s->fvert;
    float* hprim = s->hprim;
    float* htrue = s->htrue;

    // ---- the 30 vertical-sounding frequencies -----------------------------
    // D-E region tail (XTAIL must agree with LECDEN)
    float xtail = 0.85f;
    float fex = fi[1 * 6 + k] * sqrt32(1.0f - xtail * xtail);
    fvert[1 * 6 + k] = 0.01f;
    fvert[4 * 6 + k] = fex;
    float fdif = div32(fvert[4 * 6 + k] - fvert[1 * 6 + k], 3.0f);
    fvert[2 * 6 + k] = fvert[1 * 6 + k] + fdif;
    fvert[3 * 6 + k] = fvert[2 * 6 + k] + fdif;
    // E region nose (guess for E MUF)
    fvert[9 * 6 + k] = 0.957f * fi[1 * 6 + k];
    fvert[10 * 6 + k] = 0.99f * fi[1 * 6 + k];
    fdif = div32(fvert[9 * 6 + k] - fvert[4 * 6 + k], 5.0f);
    fvert[5 * 6 + k] = fvert[4 * 6 + k] + fdif;
    fvert[6 * 6 + k] = fvert[5 * 6 + k] + fdif;
    fvert[7 * 6 + k] = fvert[6 * 6 + k] + fdif;
    fvert[8 * 6 + k] = fvert[7 * 6 + k] + fdif;
    // E-F cusp
    fvert[11 * 6 + k] = 1.05f * fi[1 * 6 + k];
    // F region nose
    fvert[30 * 6 + k] = 0.99f * fi[3 * 6 + k];
    fvert[29 * 6 + k] = 0.98f * fi[3 * 6 + k];
    fvert[28 * 6 + k] = 0.96f * fi[3 * 6 + k];
    fvert[27 * 6 + k] = 0.92f * fi[3 * 6 + k];
    if (fi[2 * 6 + k] <= 0.0f) {  // IF(FI(2,K)) 385,385,400 -> no F1 layer
        fdif = div32(fvert[27 * 6 + k] - fvert[11 * 6 + k], 16.0f);
        for (int iff = 12; iff <= 26; ++iff)  // DO 395 IF=12,26
            fvert[iff * 6 + k] = fvert[(iff - 1) * 6 + k] + fdif;
    } else {                      // F1 layer and F2 layer
        fvert[20 * 6 + k] = 0.99f * fi[2 * 6 + k];
        fdif = div32(fvert[20 * 6 + k] - fvert[11 * 6 + k], 9.0f);
        for (int iff = 12; iff <= 19; ++iff)  // DO 405 IF=12,19
            fvert[iff * 6 + k] = fvert[(iff - 1) * 6 + k] + fdif;
        // F1-F2 cusp
        fvert[21 * 6 + k] = 1.01f * fi[2 * 6 + k];
        fdif = div32(fvert[27 * 6 + k] - fvert[21 * 6 + k], 6.0f);
        for (int iff = 22; iff <= 26; ++iff)  // DO 415 IF=22,26
            fvert[iff * 6 + k] = fvert[(iff - 1) * 6 + k] + fdif;
    }

    // ---- virtual/true heights ---------------------------------------------
    if (s->iedp < 0) {  // 600: integrate to get ionogram, NT-point Gaussian
        for (int iff = 1; iff <= 30; ++iff) {  // DO 640 IF=1,30
            float hpx, htx;
            gethp(s, fvert[iff * 6 + k], &hpx, &htx);
            hprim[iff * 6 + k] = hpx;
            htrue[iff * 6 + k] = htx;
        }
        return;
    }
    // 705: faster version, parabolic E layer with exponential tail
    for (int iff = 1; iff <= 10; ++iff) {  // DO 710 (same as GETHP default E)
        htrue[iff * 6 + k] = GENION_HTE[iff];
        hprim[iff * 6 + k] = GENION_HPE[iff];
    }
    for (int iff = 11; iff <= 30; ++iff) {  // DO 715 IF=11,30
        float fn = fvert[iff * 6 + k] * fvert[iff * 6 + k];
        // true height by table lookup ((51, 4) arrays -> ncols 4)
        htrue[iff * 6 + k] = xlin(fn, s->fnsq, s->htr, 50, 1, 4);
    }
    if (fi[2 * 6 + k] <= 0.0f) {  // 720: no F1
        for (int iff = 11; iff <= 30; ++iff) {  // DO 725 IF=11,30
            float fn = fvert[iff * 6 + k];
            // bottom of F layer plus bending plus retardation
            hprim[iff * 6 + k] = hi[3 * 6 + k] - yi[3 * 6 + k]
                                 + bendy(s, 3, k, fn)
                                 + (pen(s, 1, k, fn) - 2.0f * yi[1 * 6 + k]);
        }
    } else {  // 730: F1 present -- too many possibilities, integrate
        for (int iff = 11; iff <= 30; ++iff) {  // DO 740 IF=11,30
            float hpx, htx;
            gethp(s, fvert[iff * 6 + k], &hpx, &htx);
            hprim[iff * 6 + k] = hpx;
        }
    }
}

#endif  // VOACAP_GENION_CUH
