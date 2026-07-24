// TABS — absorption loss per kilometre for the leg that misses the D region;
// the CUDA C twin of scalar/tabs.py.  BABS's twin, character for character
// the same geometry, differing in one number: the absorption index is the
// hardcoded XI = 0.1 rather than the path average ACAV.
//
// The guard is mirrored: TABS tests DM first (falls into the dead label from
// ABOVE), and HP exactly 70 km is DEAD here where BABS keeps it LIVE
// (IF(HP-70.) 110,110,115).  Gyrofrequency from sample area 3, hardcoded.
//
// float32; COS/ATAN/SQRT and the **1.98 power carry the [[f32-libm-noise]]
// floor.  One div32 site: RCOSD / (RZ + 100.0f) is a constant denominator.

#ifndef VOACAP_TABS_CUH
#define VOACAP_TABS_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "state.cuh"

// TABS(DEL,HP,DM,AM) — returns AM, the loss in dB per kilometre.  state reads
// GYZ(3) only; freq in MHz; del the take-off angle in radians; hp the virtual
// height in km; dm the over-the-top leg's arc in radians.  No SigdState: XI
// replaces ACAV.
__device__ inline float tabs(const LayerState* state, float freq, float del,
                             float hp, float dm) {
    if (dm <= 0.0f)                     // IF(DM) 110,110,105
        return 1000.0f;
    if (hp - 70.0f <= 0.0f)             // IF(HP-70.) 110,110,115 -> 70 DIES
        return 1000.0f;
    // .....ABSORPTION INDEX  (XI = 0.1, not ACAV -- the only difference)
    // .....THIS IS THE F2 MODE LOSS FOR A FRACTIONAL NUMBER OF HOPS
    float ab = 10.2f + pow32(freq + state->gyz[3], 1.98f);
    float am = 677.2f * 0.1f / ab;
    float rcosd = VC_RZ * cos32(del);
    float sphe = rcosd / (VC_RZ + hp);
    float phe = atan32(sphe / sqrt32(1.0f - sphe * sphe));
    float psi = VC_PIO2 - phe - del;
    float xmhop = 0.5f * dm / psi;
    sphe = div32(rcosd, VC_RZ + 100.0f);
    float scphe = 1.0f / sqrt32(1.0f - sphe * sphe);
    float port = xmhop * scphe / (dm * VC_RZ);
    return am * port;
}

#endif  // VOACAP_TABS_CUH
