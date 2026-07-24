// BABS — absorption loss per kilometre for the long path's absorbing leg;
// the CUDA C twin of scalar/babs.py.  The non-deviative term
// 677.2*ACAV/((f+fH)**1.98 + 10.2) turned into a per-kilometre figure by
// PORT (hops per leg length, referred to 100 km).  1000 dB/km is the "no
// absorption computed" marker.
//
// BABS's arithmetic IF sends HP exactly 70 km to the LIVE arm; TABS's sends
// the same number to the dead arm — the disagreement test_lngabs pins.  The
// gyrofrequency is hardcoded to sample area 3 (the path midpoint).
//
// float32; COS/ATAN/SQRT and the **1.98 power carry the [[f32-libm-noise]]
// floor.  One div32 site: RCOSD / (RZ + 100.0f) is a constant denominator.

#ifndef VOACAP_BABS_CUH
#define VOACAP_BABS_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "state.cuh"

// BABS(DEL,HP,DF,AF) — returns AF, the loss in dB per kilometre.  state reads
// GYZ(3) only; sigd reads ACAV; freq in MHz; del the take-off angle in
// radians; hp the virtual height in km; df the absorbing leg's arc in radians.
__device__ inline float babs(const LayerState* state, const SigdState* sigd,
                             float freq, float del, float hp, float df) {
    if (df <= 0.0f)                     // IF(DF) 105,105,110
        return 1000.0f;
    if (hp - 70.0f < 0.0f)              // IF(HP-70.) 105,115,115 -> 70 LIVES
        return 1000.0f;
    // .....ABSORPTION INDEX
    float af = sigd->acav;
    // .....THIS IS THE F2 MODE LOSS FOR A FRACTIONAL NUMBER OF HOPS
    float ab = 10.2f + pow32(freq + state->gyz[3], 1.98f);
    af = 677.2f * af / ab;
    float rcosd = VC_RZ * cos32(del);
    float sphe = rcosd / (VC_RZ + hp);
    float phe = atan32(sphe / sqrt32(1.0f - sphe * sphe));
    float psi = VC_PIO2 - phe - del;
    float xmhop = 0.5f * df / psi;
    sphe = div32(rcosd, VC_RZ + 100.0f);
    float scphe = 1.0f / sqrt32(1.0f - sphe * sphe);
    float port = xmhop * scphe / (df * VC_RZ);
    return af * port;
}

#endif  // VOACAP_BABS_CUH
