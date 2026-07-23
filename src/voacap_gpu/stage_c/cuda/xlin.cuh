// XLIN — general piecewise-linear table interpolation, the CUDA C twin of
// stage_a/xlin.py (itself a branch-for-branch transliteration of xlin.for's
// arithmetic-IF web).  Tolerant of non-monotonic tables: first bracketing
// segment wins, with increasing / flat / decreasing runs handled separately
// and endpoint clamping outside the table.  REAL*4, no transcendentals.
//
// XN/YN are the Stage A 1-based 2-D arrays, passed FLAT ROW-MAJOR with an
// explicit column count: element [i, k] is xn[i * ncols + k], row 0 and
// column 0 are the usual 1-based padding.  That is the Stage C convention
// for every numpy (rows, cols) array — see CONVENTIONS.md.

#ifndef VOACAP_XLIN_CUH
#define VOACAP_XLIN_CUH

__device__ inline float xlin_interp_(float x, const float* xn, const float* yn,
                                     int i, int k, int ncols) {
    // Label 10: XLIN = A + B*C/D (left-to-right, REAL*4).
    float a = yn[i * ncols + k];
    float b = x - xn[i * ncols + k];
    float c = yn[(i + 1) * ncols + k] - yn[i * ncols + k];
    float d = xn[(i + 1) * ncols + k] - xn[i * ncols + k];
    return a + b * c / d;
}

__device__ inline float xlin(float x, const float* xn, const float* yn,
                             int ntab, int k, int ncols) {
    if (xn[1 * ncols + k] - x > 0.0f)   // IF(XN(1)-X) 3,3,2 : >0 -> YN(1)
        return yn[1 * ncols + k];
    for (int i = 1; i < ntab; ++i) {    // DO 19 J=1,NTAB-1
        float d = xn[i * ncols + k] - xn[(i + 1) * ncols + k];
        if (d < 0.0f) {                 // 8: increasing segment
            if (xn[i * ncols + k] - x <= 0.0f) {        // -> 9
                if (x - xn[(i + 1) * ncols + k] < 0.0f) // -> 10
                    return xlin_interp_(x, xn, yn, i, k, ncols);
                // else -> 19 (continue)
            } else {                                    // -> 10
                return xlin_interp_(x, xn, yn, i, k, ncols);
            }
        } else if (d == 0.0f) {         // 5: flat segment
            if (x - xn[i * ncols + k] == 0.0f)          // -> 7
                return yn[i * ncols + k];
            // else -> 19 (continue)
        } else {                        // 11: decreasing segment
            if (xn[i * ncols + k] - x < 0.0f) {
                // -> 19 (continue)
            } else {                                    // -> 12
                if (x - xn[(i + 1) * ncols + k] > 0.0f) // -> 10
                    return xlin_interp_(x, xn, yn, i, k, ncols);
                // else -> 19 (continue)
            }
        }
    }
    return yn[ntab * ncols + k];        // XLIN = YN(ISMAX)
}

#endif  // VOACAP_XLIN_CUH
