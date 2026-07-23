// REAL*4 arithmetic helpers — the CUDA C twin of stage_a/f32math.py.
//
// Evaluate in binary64, round ONCE to binary32.  This is the convention the
// whole project standardised on: it matches Stage A bit-for-bit (verified on
// this GPU: 100k-sample sweeps agree exactly), and sits at the documented
// [[f32-libm-noise]] floor against gfortran's runtime cosf/sinf.
//
// +, -, *, / and sqrtf are correctly rounded in CUDA (with --prec-div/sqrt,
// pinned in compile.py) exactly as in numpy and gfortran, so plain float
// operators are used directly — but ONLY under --fmad=false; see compile.py.

#ifndef VOACAP_F32MATH_CUH
#define VOACAP_F32MATH_CUH

__device__ __forceinline__ float sin32(float x)  { return (float)sin((double)x); }
__device__ __forceinline__ float cos32(float x)  { return (float)cos((double)x); }
__device__ __forceinline__ float acos32(float x) { return (float)acos((double)x); }
__device__ __forceinline__ float atan32(float x) { return (float)atan((double)x); }
__device__ __forceinline__ float asin32(float x) { return (float)asin((double)x); }
__device__ __forceinline__ float log32(float x)  { return (float)log((double)x); }      // ALOG
__device__ __forceinline__ float log10_32(float x) { return (float)log10((double)x); }  // ALOG10
__device__ __forceinline__ float sqrt32(float x) { return (float)sqrt((double)x); }
__device__ __forceinline__ float exp32(float x)  { return (float)exp((double)x); }

// Correctly rounded f32 division THAT SURVIVES CONSTANT FOLDING.  NVRTC
// strength-reduces ``x / 6370.0f`` (any compile-time-constant denominator,
// including via a const-propagated local) into multiply-by-reciprocal even
// under --prec-div=true, a 1-ULP error that conditioning can amplify.
// Rule: every division whose denominator is a literal, a VC_* constant, or
// anything the compiler can see as constant goes through div32.  Divisions
// by genuinely runtime values may use plain ``/``.
__device__ __forceinline__ float div32(float a, float b) {
    return __fdiv_rn(a, b);
}

// The SAME hazard exists in binary64, and GENOIS's REAL*8 accumulators hit
// it (``DXX / 4.34294d0``).  Use this wherever a REAL*8 region divides by a
// constant, exactly as div32 is used in the REAL*4 ones.
__device__ __forceinline__ double div64(double a, double b) {
    return __ddiv_rn(a, b);
}

// REAL*4 x**y with a REAL exponent (gfortran emits powf).  Both operands are
// already float here, mirroring f32math.pow32's round-to-f32-first rule.
__device__ __forceinline__ float pow32(float x, float y) {
    return (float)pow((double)x, (double)y);
}

// Python max()/min() as Stage A uses them, INCLUDING their NaN behaviour
// (comparison false -> first argument wins).  Do not substitute fmaxf/fminf,
// whose NaN semantics differ.
__device__ __forceinline__ float pymax(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float pymin(float a, float b) { return (b < a) ? b : a; }

#endif  // VOACAP_F32MATH_CUH
