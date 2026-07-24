// CONVH — the geometrical convergence factor for a long path; the CUDA C twin
// of scalar/convh.py.  Past a quarter turn a great-circle ray bundle stops
// spreading and converges toward the antipode; CH is the decibel credit for
// that focusing and PTOT the group path (two straight legs plus the reflection
// arc).  Both are returned through out-pointers.
//
// Three clamps, all transliterated: DENOM floored at 1e-6, CH ceilinged at
// 15 dB (which bites first, so the floor never decides anything), and GM
// floored at 0.001.  PSI is associated PIO2 - DEL - PHE here, NOT the
// PIO2 - PHE - DEL of BABS/TABS — kept as written, distinct in float32.
//
// float32; SIN/COS/ALOG10 carry the [[f32-libm-noise]] floor.  One div32
// site: PTOT / RZ is a constant denominator.

#ifndef VOACAP_CONVH_CUH
#define VOACAP_CONVH_CUH

#include "constants.cuh"
#include "f32math.cuh"

// CONVH(GD,PHE,DEL,HP,RAY,CH,PTOT).  gd the great-circle angle (radians); phe
// the incidence angle at reflection height (radians); del the take-off angle
// (radians); hp the virtual height (km); ray the straight-line leg (km).
// Writes ch (dB) and ptot (km).
__device__ inline void convh(float gd, float phe, float del, float hp,
                             float ray, float* ch, float* ptot) {
    float denom = pymax(fabsf(sin32(gd)), 0.000001f);
    float psi = VC_PIO2 - del - phe;    // note the order; see header
    float gm = gd - 2.0f * psi;
    gm = pymax(gm, 0.001f);
    // .....GROUP PATH
    float pt = 2.0f * ray + (VC_RZ + hp) * gm;
    float smallc = div32(pt, VC_RZ) * cos32(del) / denom;
    float c = 10.0f * log10_32(smallc);
    // .....AT ANTIPODAL
    c = pymin(c, 15.0f);
    *ch = c;
    *ptot = pt;
}

#endif  // VOACAP_CONVH_CUH
