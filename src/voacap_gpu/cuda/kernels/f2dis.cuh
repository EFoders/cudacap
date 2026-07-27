// F2DIS — the F2 MUF distance factor (upper/lower decile spread) CURMUF
// uses; the CUDA C twin of scalar/f2dis.py (Greg Hand's 2009 rewrite).
//
// Picks one cell of the F2D(16,6,6) M(3000) table REDMAP loads — latitude
// band x SSN band/hemisphere x 4-hour time block, with the first eight
// latitude rows holding the upper decile and rows 9..16 the lower — and
// scales the distance of the tabulated ratio from unity into a standard
// deviation.  When no ionospheric data base is loaded (F2D(1,1,1) <= 0) it
// falls back to the classical 15 % spread.
//
// F2D IS SHARED, NOT PER-THREAD, and state.cuh has no IonoState, so the
// table passes as a flat 0-based row-major `const float*` of 16*6*6 = 576
// floats: Stage A's f2d[iz-1, j-1, icc-1] is f2d[(iz-1)*36 + (j-1)*6
// + (icc-1)].  f2d[0] is the F2D(1,1,1) "is there a data base" test — the
// same scalar the rest of Stage C carries as `f2d111`.
//
// Fortran INT() truncation on the table indices is C (int); Python
// max/min on the ICL clamp are on INTEGERS, so they are plain integer
// selects, not pymax/pymin.  Rule 4b: every denominator here (4, 10, 1.28,
// 1.28155) is a compile-time constant and goes through div32.

#ifndef VOACAP_F2DIS_CUH
#define VOACAP_F2DIS_CUH

#include "f32math.cuh"

// f2d: the shared (16,6,6) table, flat row-major 0-based (576 floats);
// fmuf: the F2 MUF; ssn: sunspot number; clat: control-point geographic
// latitude in RADIANS; freq: the frequency under test; abb: local time
// (CLCK) in hours.  Returns SIG, the F2 MUF standard deviation.
__device__ inline float f2dis(const float* f2d, float fmuf, float ssn,
                              float clat, float freq, float abb) {
    if (f2d[0] <= 0.0f) {           // no ionospheric data — classical values
        float sig = div32(0.15f * fmuf, 1.28155f);
        return pymax(sig, 0.001f);
    }

    float cl = clat * 57.29577f;                    // radians -> degrees
    int icc = (int)(div32(abb, 4.0f) + 1.55f);      // 4-hour time blocks
    if (icc > 6)
        icc = 1;

    int i = (freq <= fmuf) ? 8 : 0;                 // lower vs upper decile

    int icl = (int)(9.5f - div32(fabsf(cl), 10.0f));
    icl = (icl < 8) ? icl : 8;                      // min(8, icl)
    icl = (icl > 1) ? icl : 1;                      // max(1, ...)
    int iz = icl + i;

    int j = 2;                                      // 50 < SSN <= 100
    if (ssn <= 50.0f)
        j = 1;
    if (ssn > 100.0f)
        j = 3;
    if (cl <= 0.0f)                                 // Southern Hemisphere
        j = j + 3;

    float sig = div32(fabsf(fmuf - f2d[(iz - 1) * 36 + (j - 1) * 6
                                       + (icc - 1)] * fmuf), 1.28f);
    if (sig <= 0.001f)
        sig = 0.001f;
    return sig;
}

#endif  // VOACAP_F2DIS_CUH
