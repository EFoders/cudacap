// COMMON /CON/ constants, verbatim from blkdat.for:129-131 (REAL*4) — the
// CUDA C twin of stage_a/constants.py.  A C float literal rounds its decimal
// to nearest-even binary32 exactly as np.float32 does, so these are the same
// bit patterns.  Deliberately NOT high-precision: PI is 3.1415926, D2R/R2D
// are truncated, faithful to the Fortran.

#ifndef VOACAP_CONSTANTS_CUH
#define VOACAP_CONSTANTS_CUH

#define VC_D2R  0.01745329251f
#define VC_DCL  1.28f
#define VC_GAMA 0.57721566f
#define VC_PI   3.1415926f
#define VC_PI2  6.283185307f
#define VC_PIO2 1.570796326f
#define VC_R2D  57.295779513f
#define VC_RZ   6370.0f
#define VC_VOFL 299.79246f

#endif  // VOACAP_CONSTANTS_CUH
