// GENFAM — frequency dependence of atmospheric noise, plus its decile
// tables; the CUDA C twin of scalar/genfam.py.
//
//   * FA        — Fam at FREQ from the FAM(14,12) coefficients: two nested
//                 degree-6 Horner evaluations (PZ from FAM 1..7, PX from
//                 FAM 8..14), first at the fixed abscissa -0.75 to invert
//                 the 1 MHz value into the curve's own variable CZ, then at
//                 U = (8*2**log10(FREQ) - 11)/4.
//   * DUA/DLA   — upper and lower decile ratios,
//   * DUS/DLS   — their standard deviations,
//   * DMS       — sigma-Fam,
// the last five from quartic polynomials in log10(FREQ) held in DUD(5,12,5).
//
// Two frequency clamps sit in that polynomial loop and both matter: the
// decile curves in Report 322 stop at 20 MHz and sigma-Fam stops at 10 MHz,
// so X is pinned at log10(20) above 20 MHz for all five, and again at 1.0
// for the fifth alone above 10 MHz.
//
// IBK selects the table column: local-time block 1..6 in the northern
// hemisphere, 7..12 in the southern, on the sign of the latitude alone.
//
// The tables come from REDMAP and are SHARED, not per-thread, passed flat
// row-major 0-based:
//     fam (14,12)   element [a,b]   = fam[a*12 + b]
//     dud (5,12,5)  element [a,b,c] = dud[(a*12 + b)*5 + c]
//
// float32; ALOG10 and the 2.**X power carry the [[f32-libm-noise]] floor.

#ifndef VOACAP_GENFAM_CUH
#define VOACAP_GENFAM_CUH

#include "f32math.cuh"

// The paired Horner evaluations at abscissa U1 (label 110); j is the
// 0-based FAM/DUD column.
__device__ inline void genfam_pzpx_(const float* fam, int j, float u1,
                                    float* pz, float* px) {
    float p = u1 * fam[0 * 12 + j] + fam[1 * 12 + j];
    float q = u1 * fam[7 * 12 + j] + fam[8 * 12 + j];
    for (int i = 3; i <= 7; ++i) {      // DO 115 I=3,7
        p = u1 * p + fam[(i - 1) * 12 + j];
        q = u1 * q + fam[(i + 6) * 12 + j];
    }
    *pz = p;
    *px = q;
}

// y2: latitude (radians — only its sign is used); iblk: local-time block
// 1..6; freq: MHz; z: Fam at 1 MHz.  Writes (fa, dua, dla, dms, dus, dls),
// in the Fortran's argument order.
__device__ inline void genfam(const float* fam, const float* dud, float y2,
                              int iblk, float freq, float z, float* fa,
                              float* dua, float* dla, float* dms, float* dus,
                              float* dls) {
    int ibk = iblk;
    if (y2 < 0.0f)                      // southern hemisphere -> blocks 7..12
        ibk = ibk + 6;
    int j = ibk - 1;

    float x = log10_32(freq);
    // rule 4b: constant denominator -> div32 (here /4.0f is an exact
    // reciprocal either way, but the rule is categorical)
    float u = div32(8.0f * pow32(2.0f, x) - 11.0f, 4.0f);
    float pz, px;
    genfam_pzpx_(fam, j, -0.75f, &pz, &px);  // KOP=1: recover CZ
    float cz = z * pz + px;
    cz = z + z - cz;
    genfam_pzpx_(fam, j, u, &pz, &px);       // KOP=2: this frequency
    *fa = cz * pz + px;

    // Report 322's decile curves end at 20 MHz, sigma-Fam's at 10 MHz.
    if (freq > 20.0f)
        x = log10_32(20.0f);
    float v[5];
    for (int i = 1; i <= 5; ++i) {      // DO 145
        if (i == 5 && freq > 10.0f)
            x = 1.0f;
        float y = dud[(0 * 12 + j) * 5 + (i - 1)];
        for (int jj = 2; jj <= 5; ++jj) // DO 140
            y = y * x + dud[((jj - 1) * 12 + j) * 5 + (i - 1)];
        v[i - 1] = y;
    }
    *dua = v[0];
    *dla = v[1];
    *dus = v[2];
    *dls = v[3];
    *dms = v[4];
}

#endif  // VOACAP_GENFAM_CUH
