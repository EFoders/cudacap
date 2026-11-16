// REAL*4 arithmetic helpers — the CUDA C twin of scalar/f32math.py.
//
// Evaluate in binary64, round ONCE to binary32.  This is the convention the
// whole project standardised on: it matches Stage A bit-for-bit (verified on
// this GPU: 100k-sample sweeps agree exactly), and sits at the documented
// [[f32-libm-noise]] floor against gfortran's runtime cosf/sinf.
//
// +, -, *, / and sqrtf are correctly rounded in CUDA (with --prec-div/sqrt,
// pinned in compile.py) exactly as in numpy and gfortran, so plain float
// operators are used directly — but ONLY under --fmad=false; see compile.py.
//
// There is ONE opt-in escape hatch, -DVOACAP_FAST_MATH, documented at length
// below.  It is off unless a caller passes it through
// ``stage_c.compile.load(..., extra_opts=("-DVOACAP_FAST_MATH",))``, and
// ``tests/cuda/test_fastmath.py`` is what pins what it costs.

// ---------------------------------------------------------------------------
// OPT-IN FAST MATH — -DVOACAP_FAST_MATH.  OFF BY DEFAULT; the default is and
// stays the binary64 path above, which is what makes Stage C bit-identical to
// Stage A.  Read this before turning it on.
//
// WHY there is a knob at all: this is a consumer card (RTX 5080, sm_120) where
// binary64 runs at ~1/64 the binary32 rate, and LUFFY is dense with
// cos/sin/atan/asin/acos/exp/log/log10/pow.  Compiling the SAME kernel against
// CUDA's native f32 libm nearly halves the kernel time (measured: 1.9x) for
// the same register count.
//
// WHAT changes and what does NOT:
//
// * The nine TRANSCENDENTALS switch to the native f32 routines, which are
//   0.5 ulp no longer.  MEASURED on this toolchain over 400k arguments
//   (tests/cuda/test_fastmath.py): 2 ulp worst for expf and log10f, 1 ulp
//   for sinf/cosf/asinf/acosf/atanf/logf/powf, 66-97% of results still
//   bit-identical.  That is the entire accuracy cost, and it is NOT uniform
//   across the program: FOBBY truncates ``1000.*FREQ`` to integer kHz, so a
//   1-ulp cosine there quantises into a whole-kHz step that FINDF then
//   interpolates on (measured: 96 flips in 830,940 cells, 0.0116%, against
//   the 0.251% Stage A already carries vs the Fortran).
//
// * ``sqrt32`` switches to ``sqrtf`` and is STILL BIT-IDENTICAL.  With
//   --prec-sqrt=true ``sqrtf`` is correctly rounded, and rounding a correctly
//   rounded binary64 sqrt of an exactly-representable binary32 argument down
//   to binary32 cannot double-round wrong (53 >= 2*24+2).  So this one is
//   free: it drops a 1/64-rate f64 sqrt for zero error.  The STRICT build
//   now uses it too (with the binary64 path's NaN bits for negative
//   arguments; see sqrt32 below), proved over all 2^32 inputs.
//
// * ``div32``/``div64`` DO NOT CHANGE.  They are not transcendentals, and
//   ``__fdiv_rn`` IS the IEEE f32 divide — there is no cheaper correct one,
//   and the reason they exist (defeating NVRTC's multiply-by-reciprocal
//   strength reduction on constant denominators) is a code-generation hazard
//   that has nothing to do with libm precision.  Relaxing them would buy no
//   speed and reintroduce a 1-ulp error that conditioning amplifies.
//   ``div64`` additionally sits inside GENOIS's genuinely REAL*8
//   accumulators, where dropping to f32 would change the physics rather than
//   the last bit of it.
//
// * ``pymax``/``pymin`` DO NOT CHANGE: comparisons, not arithmetic, and their
//   exact NaN behaviour is load-bearing.
//
// Fast math is a whole-program switch, not a per-call one: mixing conventions
// inside one circuit would make the residual impossible to attribute.
// ---------------------------------------------------------------------------

#ifndef VOACAP_F32MATH_CUH
#define VOACAP_F32MATH_CUH

#ifdef VOACAP_FAST_MATH

__device__ __forceinline__ float sin32(float x)  { return sinf(x); }
__device__ __forceinline__ float cos32(float x)  { return cosf(x); }
__device__ __forceinline__ float acos32(float x) { return acosf(x); }
__device__ __forceinline__ float atan32(float x) { return atanf(x); }
__device__ __forceinline__ float asin32(float x) { return asinf(x); }
__device__ __forceinline__ float log32(float x)  { return logf(x); }      // ALOG
__device__ __forceinline__ float log10_32(float x) { return log10f(x); }  // ALOG10
__device__ __forceinline__ float sqrt32(float x) { return sqrtf(x); }
__device__ __forceinline__ float exp32(float x)  { return expf(x); }

#else

__device__ __forceinline__ float sin32(float x)  { return (float)sin((double)x); }
__device__ __forceinline__ float cos32(float x)  { return (float)cos((double)x); }
__device__ __forceinline__ float acos32(float x) { return (float)acos((double)x); }
__device__ __forceinline__ float atan32(float x) { return (float)atan((double)x); }
__device__ __forceinline__ float asin32(float x) { return (float)asin((double)x); }
__device__ __forceinline__ float log32(float x)  { return (float)log((double)x); }      // ALOG
__device__ __forceinline__ float log10_32(float x) { return (float)log10((double)x); }  // ALOG10
// SQRT is the one function evaluated in binary32 on the strict path, because
// it is the one where that is PROVABLY the same answer.  ``(float)sqrt(
// (double)x)`` is a correctly rounded binary64 square root rounded again to
// binary32, and for a binary32 argument that double rounding is innocuous
// (53 >= 2*24 + 2), so it equals the correctly rounded binary32 ``sqrtf``
// (--prec-sqrt=true) -- except in the bits of the NaN a negative argument
// produces: the binary64 path yields 0xffc00000, sqrtf 0x7fffffff.  That one
// case is patched below, and tests/cuda/test_exact_shortcuts.py compares
// this function with the binary64 expression over ALL 2^32 binary32 inputs
// (0 mismatches).  The binary64 sqrt runs at 1/64 rate on this card and was
// the inner operation of GETHP's 40-node Gaussian integral.
__device__ __forceinline__ float sqrt32(float x) {
    float r = sqrtf(x);
    return (x < 0.0f && r != r) ? __uint_as_float(0xffc00000u) : r;
}
__device__ __forceinline__ float exp32(float x)  { return (float)exp((double)x); }

#endif  // VOACAP_FAST_MATH

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
#ifdef VOACAP_FAST_MATH
    return powf(x, y);              // ~8 ulp; see the fast-math note above
#else
    return (float)pow((double)x, (double)y);
#endif
}

// Python max()/min() as Stage A uses them, INCLUDING their NaN behaviour
// (comparison false -> first argument wins).  Do not substitute fmaxf/fminf,
// whose NaN semantics differ.
__device__ __forceinline__ float pymax(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float pymin(float a, float b) { return (b < a) ? b : a; }

#endif  // VOACAP_F32MATH_CUH
