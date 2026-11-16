// cos32_deg — ``cos32(deg * VC_D2R)`` with the forty /RAYS/ take-off angles
// answered from a table instead of a binary64 cosine.
//
// WHY.  On this card binary64 runs at 1/64 of the binary32 rate, and the
// strict build evaluates every COS in binary64 (f32math.cuh).  FOBBY, FINDF,
// FDIST and SETTXR take the cosine of a ray's take-off angle, and almost every
// such angle IS one of the forty fixed ANG entries (blkdat DATA, sang.cuh's
// SANG_ANG): FINDF copies ``DELFX = ANG(IA)`` on its exact and interpolated
// arms, and everything downstream reads DELFX.  So the same forty cosines were
// recomputed, in binary64, every frequency of every hour.
//
// WHAT, EXACTLY.  `VC_SANG_COS[i]` holds the bits of ``cos32(SANG_ANG[i] *
// VC_D2R)`` as this device computes them (the strict binary64 path), and
// tests/cuda/test_angcos.py re-derives every entry on the device and checks
// `cos32_deg` against ``cos32(x * VC_D2R)`` over all 2^32 binary32 inputs.
// A lookup HITS only when `deg` is BITWISE one of the table's angles, in which
// case ``deg * VC_D2R`` is bitwise the table's argument and the stored value
// is the cosine the binary64 path would return; anything else -- cusp angles,
// interpolated DELMOD, -0, NaN -- falls through to the original expression.
// Same bits for every input; no arithmetic in the program changed.
//
// Under -DVOACAP_FAST_MATH the table would hold the wrong function, so the
// helper is the plain native expression there.

#ifndef VOACAP_ANGCOS_CUH
#define VOACAP_ANGCOS_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "sang.cuh"

__device__ __constant__ unsigned VC_SANG_COS[40] = {
    0x3f800000u, 0x3f7ffd81u, 0x3f7ff605u, 0x3f7fd814u, 0x3f7fa62fu,
    0x3f7f605cu, 0x3f7f069eu, 0x3f7e98fdu, 0x3f7d8235u, 0x3f7c1c5cu,
    0x3f7a67e2u, 0x3f78654du, 0x3f76153fu, 0x3f737871u, 0x3f708fb2u,
    0x3f6d5becu, 0x3f69de1du, 0x3f66175eu, 0x3f6208dau, 0x3f5db3d7u,
    0x3f5919aeu, 0x3f543bcfu, 0x3f4f1bbdu, 0x3f49bb13u, 0x3f441b7du,
    0x3f3e3ebdu, 0x3f3826a7u, 0x3f31d522u, 0x3f2b4c24u, 0x3f248dbbu,
    0x3f1d9bfeu, 0x3f167919u, 0x3f0f2744u, 0x3effffffu, 0x3ed8616bu,
    0x3eaf1d44u, 0x3e8483efu, 0x3e31d0d7u, 0x3db27ebeu, 0x39371444u,
};

// The table slot `deg` would occupy if it were a ray angle (SANG_ANG is
// 0, .5, 1..6, 8..56 step 2, 60..85 step 5, 89.99); the caller still checks
// the bits, so a wrong guess only costs the fallback.
__device__ __forceinline__ int vc_sang_slot_(float deg) {
    if (!(deg >= 0.0f && deg <= 90.0f))
        return -1;                      // also NaN
    if (deg < 1.0f)
        return (deg == 0.5f) ? 1 : 0;
    if (deg <= 6.0f)
        return (int)deg + 1;
    if (deg <= 56.0f)
        return (int)deg / 2 + 4;
    if (deg < 89.0f)
        return ((int)deg - 60) / 5 + 33;
    return 39;
}

// cos32(rad) where the caller guarantees rad == deg * VC_D2R (it has just
// computed exactly that).
__device__ __forceinline__ float cos32_deg(float deg, float rad) {
#ifndef VOACAP_FAST_MATH
    int i = vc_sang_slot_(deg);
    if (i >= 0 && i < 40 && __float_as_uint(SANG_ANG[i]) == __float_as_uint(deg))
        return __uint_as_float(VC_SANG_COS[i]);
#endif
    return cos32(rad);
}

#endif  // VOACAP_ANGCOS_CUH
