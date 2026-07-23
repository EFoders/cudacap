// SYSSY — table lookup for excess-system-loss deciles and prediction errors;
// the CUDA C twin of stage_a/syssy.py.  Indexes the CCIR SYS(9,16,6) and
// PERR(9,4,6) tables (loaded by REDMAP into /TWO/) by geomagnetic latitude G
// (radians) and reflection-area local time PT, with the reflection-area
// distance index NN (2 short, 5 long).  When no coefficient data is loaded
// (F2D(1,1,1) <= 0) it returns the fixed ITSA-1 fallback values.
//
// The tables are SHARED, not per-thread, passed flat row-major 0-based:
//     sys  (9,16,6)  element [a,b,c] = sys[(a*16 + b)*6 + c]
//     perr (9,4,6)   element [a,b,c] = perr[(a*4 + b)*6 + c]
// f2d111 is the F2D(1,1,1) scalar the no-data test reads.
//
// The KJ (latitude), LOL (local-time block, hemisphere-shifted by KK) and
// LD/ND (prediction-error block/offset) indices are integer truncations of
// REAL*4 expressions; the arithmetic stays float32 so boundary cases
// truncate exactly as the Fortran does.  Table reads and two divisions only.
//
// EVERY DIVISION BY A LITERAL GOES THROUGH div32 (CONVENTIONS.md rule 4b).
// NVRTC strength-reduces `x / <constant>` into multiply-by-reciprocal even
// under --prec-div=true (measured on this GPU: 8.832f/1.28f comes out 1 ULP
// low, while the identical division with the divisor loaded from memory is
// exact); div32 is the IEEE round-to-nearest single division and restores
// bit-identity with numpy.

#ifndef VOACAP_SYSSY_CUH
#define VOACAP_SYSSY_CUH

#include "f32math.cuh"

// Returns (fm, su, sl, fmp, sup, slp) through the pointers, in the Stage A
// tuple order.
__device__ inline void syssy(const float* sys, const float* perr,
                             float f2d111, float g, float pt, int nn,
                             float* fm, float* su, float* sl, float* fmp,
                             float* sup, float* slp) {
    if (f2d111 <= 0.0f) {               // IF(F2D(1,1,1)) 90,90,95 -> no tape
        float dl = 4.0f;
        float du = 9.0f;
        *fm = 9.0f;
        *fmp = 5.6f;
        *sup = 0.60f;
        *slp = 1.63f;
        *su = div32(du, 1.28f);
        *sl = div32(dl, 1.29f);     // NOTE: 1.29 on this path alone
        return;
    }
    int j = (int)(pt + 0.5f);
    // SYSSY's own coarse degrees-per-radian (57.29577), NOT the /CON/ R2D.
    float gg = fabsf(g) * 57.29577f;
    int kj = (int)(div32(gg - 40.0f, 5.0f) + 1.5f);
    if (kj <= 0)                        // IF(KJ) 100,100,105
        kj = 1;
    else if (kj > 9)                    // 105  IF(KJ-9) 115,115,110
        kj = 9;
    int kk, nd;
    if (g < 0.0f) {                     // IF(G) 125,120,120 -> southern
        kk = 8;
        nd = 3;
    } else {
        kk = 0;
        nd = 0;
    }
    int lj = (int)(div32((float)j, 3.0f) + 0.67f);
    if (lj - 1 < 0)                     // IF(LJ-1) 135,140,140
        lj = 8;
    int lol = lj + kk;
    int ld = (int)(div32((float)j, 6.0f) + 0.67f);
    if (ld - 1 < 0)                     // IF(LD-1) 145,150,150
        ld = 4;
    *fm = sys[((kj - 1) * 16 + (lol - 1)) * 6 + (nn - 1)];
    float du = sys[((kj - 1) * 16 + (lol - 1)) * 6 + nn];       // SYS(,,NN+1)
    float dl = sys[((kj - 1) * 16 + (lol - 1)) * 6 + (nn - 2)]; // SYS(,,NN-1)
    *fmp = perr[((kj - 1) * 4 + (ld - 1)) * 6 + nd];            // PERR(,,ND+1)
    *sup = perr[((kj - 1) * 4 + (ld - 1)) * 6 + (nd + 1)];      // PERR(,,ND+2)
    *slp = perr[((kj - 1) * 4 + (ld - 1)) * 6 + (nd + 2)];      // PERR(,,ND+3)
    *su = div32(du, 1.28f);
    *sl = div32(dl, 1.28f);
}

#endif  // VOACAP_SYSSY_CUH
