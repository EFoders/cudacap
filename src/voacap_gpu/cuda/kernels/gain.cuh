// GAIN — antenna power gain and ground reflection loss (Module 5); the CUDA C
// twin of stage_a/gain.py.  Two independent routines behind one Fortran entry
// point, selected by ITR:
//
//   * ITR < 0  -> ground_loss(): Fresnel reflection loss for a sample area
//     from SIGPAT/EPSPAT in the LayerState.  Self-contained maths, no tables.
//   * ITR >= 0 -> antenna_gain(): bilinear /cantenna/ lookup over
//     [frequency 1..30 MHz][elevation 0..90 deg], plus AEFF over frequency.
//
// THE ANTENNA TABLES ARE SHARED, NOT PER-THREAD.  A circuit has one transmit
// and one receive antenna, so ARRAY/AEFF stay a single device-resident table
// and only the indices into it are per-thread (see stage_b/gain.py's
// BatchedCantenna for the same convention).  They pass as flat row-major
// pointers with A = numants:
//     array   (31, 92, A+1)  element [i, j, iat] = array[(i*92 + j)*(A+1) + iat]
//     aeff    (31, A+1)      element [i, iat]    = aeff[i*(A+1) + iat]
//     iats / xfqs / xfqe / offazim : (A+1,), 1-based like Stage A.
//
// THE MISSING BOTTOM CLAMP IS FLAGGED, NOT REPRODUCED.  GAIN clamps only the
// TOP of each axis (MIN0(30,..) / MIN0(91,..)); FMC < 1 or DELTA < 0 reads
// ARRAY out of bounds in the Fortran and picks up a neighbouring antenna's
// pattern — a real defect INMUF is known to trigger.  Stage A raises there;
// a device function cannot, so this mirrors stage_b's policy: the indices
// are clamped so nothing ever reads out of bounds, and `*oob_flag` records
// the refusal so the HOST can reject the answer.  oob_flag contract (the
// function always writes it, 0 on the clean path):
//     1  — the lookup index left the table (I < 1, I > 30 or J < 1);
//          Stage A raises IndexError on exactly these lanes.
//     2  — the selected entry is an area-coverage antenna (OFFAZIM == -999,
//          /Careaant/ IARRAY360); Stage A raises NotImplementedError.
// On flag 2 the outputs keep their preset 0; on flag 1 they hold the
// clamped-index interpolation, which the host must refuse.
//
// float32; COS/SIN/ATAN/ASIN/ALOG all carry the [[f32-libm-noise]] floor.

#ifndef VOACAP_GAIN_CUH
#define VOACAP_GAIN_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "state.cuh"

// OFFAZIM sentinel for an area-coverage antenna (stage_a/antennastate.py).
#define VC_AREA_COVERAGE (-999.0f)

// GAIN's ITR<0 branch: ground reflection loss in dB for sample area IG.
// st: LayerState (SIGPAT/EPSPAT); delta: elevation (radians); fmc: MHz.
// Returns RAIN, the loss magnitude; the companion EFF output is 0 on this
// path (GAIN presets it and never touches it here).  A non-positive
// conductivity means "no ground data" and returns 0.
__device__ inline float ground_loss(const LayerState* st, int ig, float delta,
                                    float fmc) {
    float sigma = st->sigpat[ig];
    float er = st->epspat[ig];
    if (sigma <= 0.0f)
        return 0.0f;                    // RAIN keeps its preset 0
    float relta = delta;
    float x = 18000.0f * sigma / fmc;
    float t = cos32(relta);
    float q = sin32(relta);
    float r = q * q;
    float s = r * r;
    float ert = er - t * t;
    float rho = sqrt32(ert * ert + x * x);
    float rho12 = sqrt32(rho);
    float alpha = -atan32(x / ert);
    float u = er * er + x * x;
    float v = sqrt32(u);
    float asxv = asin32(x / v);
    // vertical-polarisation reflection coefficient
    float cv = sqrt32(rho * rho + u * u * s
                      - 2.0f * rho * u * r * cos32(alpha + 2.0f * asxv))
               / (rho + u * r
                  + 2.0f * rho12 * v * q * cos32(alpha * 0.5f + asxv));
    // horizontal-polarisation reflection coefficient
    float ch = sqrt32(rho * rho + s - 2.0f * rho * r * cos32(alpha))
               / (rho + r + 2.0f * rho12 * q * cos32(alpha * 0.5f));
    float rain = 4.3429f * log32(0.5f * (ch * ch + cv * cv));
    rain = fabsf(rain);
    if (delta <= 0.00000001f)
        rain = 6.0f;
    return rain;
}

// GAIN's DO-10 scan: first /cantenna/ entry for terminal `itr` covering fmc.
// Returns the 1-based index, or 0 for "no match" — which GAIN treats as a
// 0 dB isotrope rather than an error.
__device__ inline int gain_select_(const int* iats, const float* xfqs,
                                   const float* xfqe, int numants, int itr,
                                   float fmc) {
    for (int iat = 1; iat <= numants; ++iat) {
        if (iats[iat] != itr)
            continue;
        if (xfqs[iat] <= fmc && fmc <= xfqe[iat])
            return iat;
    }
    return 0;
}

// GAIN's ITR>=0 branch: bilinear gain lookup.  itr: 1 = transmitter,
// 2 = receiver; delta: elevation (radians); fmc: MHz.  Writes RAIN (a gain
// in dB here, not a loss), EFF, and oob_flag (see header comment).
__device__ inline void antenna_gain(const float* array, const float* aeff,
                                    const int* iats, const float* xfqs,
                                    const float* xfqe, const float* offazim,
                                    int numants, int itr, float delta,
                                    float fmc, float* rain, float* eff,
                                    int* oob_flag) {
    int na = numants + 1;               // last-dim extent of array/aeff
    *rain = 0.0f;                       // RAIN/EFF presets
    *eff = 0.0f;
    *oob_flag = 0;
    int iat = gain_select_(iats, xfqs, xfqe, numants, itr, fmc);
    if (iat == 0)
        return;                         // 0 dB isotrope on a miss
    if (offazim[iat] == VC_AREA_COVERAGE) {
        *oob_flag = 2;                  // Stage A raises NotImplementedError
        return;
    }
    int i = (int)fmc;                   // REAL->INT truncates toward 0
    float xfmc = fmc - (float)i;
    float deltd = delta * VC_R2D;
    int j = (int)(deltd + 1.0f);
    j = (j < 91) ? j : 91;
    float xdelta = deltd - (float)(j - 1);
    if (i < 1 || j < 1 || i > 30)
        *oob_flag = 1;                  // Stage A raises IndexError here
    // Clamped indices: identical to Stage A's I/J/IP1/JP1 whenever the flag
    // is clean, and safely in-bounds when it is not.
    int iz = (i < 1) ? 1 : ((i > 30) ? 30 : i);
    int jz = (j < 1) ? 1 : j;           // top already clamped above
    int ip1 = (iz + 1 <= 30) ? iz + 1 : 30;   // MIN0(30, I+1)
    int jp1 = (jz + 1 <= 91) ? jz + 1 : 91;   // MIN0(91, J+1)
    float xx = array[(iz * 92 + jz) * na + iat];
    float yx = array[(ip1 * 92 + jz) * na + iat];
    float xy = array[(iz * 92 + jp1) * na + iat];
    float yy = array[(ip1 * 92 + jp1) * na + iat];
    float rx = xx + xfmc * (yx - xx);
    float ry = xy + xfmc * (yy - xy);
    *rain = rx + xdelta * (ry - rx);
    *eff = aeff[iz * na + iat]
           + xfmc * (aeff[ip1 * na + iat] - aeff[iz * na + iat]);
}

#endif  // VOACAP_GAIN_CUH
