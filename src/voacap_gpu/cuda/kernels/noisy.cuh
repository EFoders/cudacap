// NOISY — general two-variable Fourier-series evaluator over (latitude,
// longitude); the CUDA C twin of scalar/noisy.py.  KJ selects the
// coefficient map: 1-6 atmospheric noise, 7 the land-mass map, 8 the F2
// height-of-max / semithickness ratio (RAT) that TIMVAR needs.
//
// The Fortran overlays one COMMON /TWO/ two ways: REDMAP fills it as
// FAKP(29,16,6)/FAKMAP(29,16)/HMYM(29,16) and FAKABP(2,6)/ABMAP(2,3); NOISY
// reads it as P(29,16,8) and ABP(2,9), so P(:,:,7)=FAKMAP, P(:,:,8)=HMYM and
// ABP(:,7..9)=ABMAP(:,1..3).  NOISY owns no tables itself — the maps come
// from REDMAP — so they pass as SHARED flat row-major 0-based pointers and
// the overlay selection happens here, exactly as Stage A's _two_p/_two_abp:
//     fakp   (29,16,6)  element [a,b,c] = fakp[(a*16 + b)*6 + c]
//     fakmap (29,16)    element [a,b]   = fakmap[a*16 + b]
//     hmym   (29,16)    element [a,b]   = hmym[a*16 + b]
//     fakabp (2,6)      element [a,b]   = fakabp[a*6 + b]
//     abmap  (2,3)      element [a,b]   = abmap[a*3 + b]
//
// REAL*4 throughout (SIN/COS carry the [[f32-libm-noise]] floor).

#ifndef VOACAP_NOISY_CUH
#define VOACAP_NOISY_CUH

#include "f32math.cuh"

// The (j, k) element of the KJ-th P(29,16) slice from the REDMAP overlay
// (1-based j, k as the Fortran writes them).
__device__ inline float noisy_p_(const float* fakp, const float* fakmap,
                                 const float* hmym, int kj, int j, int k) {
    if (kj <= 6)
        return fakp[((j - 1) * 16 + (k - 1)) * 6 + (kj - 1)];
    if (kj == 7)
        return fakmap[(j - 1) * 16 + (k - 1)];
    return hmym[(j - 1) * 16 + (k - 1)];    // kj == 8
}

// XLA geographic latitude (deg), CEG east longitude (deg).  Returns the
// Fourier value ANOS.
__device__ inline float noisy(const float* fakp, const float* fakmap,
                              const float* hmym, const float* fakabp,
                              const float* abmap, int kj, float xla,
                              float ceg) {
    // (ALF, BET) = ABP(1,KJ), ABP(2,KJ) from the REDMAP overlay
    float alf, bet;
    if (kj <= 6) {
        alf = fakabp[0 * 6 + (kj - 1)];
        bet = fakabp[1 * 6 + (kj - 1)];
    } else {
        alf = abmap[0 * 3 + (kj - 7)];
        bet = abmap[1 * 3 + (kj - 7)];
    }

    int lm, ln;
    if (kj == 8) {                      // 100: limits of the Fourier series
        lm = 15;
        ln = 10;
    } else {                            // 105
        lm = 29;
        ln = 15;
    }

    // 110: longitude half-angle and its sine harmonics
    float q = 0.0087266466f * ceg;      // D2R/2, longitude half-angle
    float c1 = cos32(q);
    float s1 = sin32(q);
    float sx[16];                       // SX(15), 1-based
    sx[1] = s1;
    float cx = c1;
    for (int k = 2; k <= ln; ++k) {     // DO 115
        float tx = sx[k - 1];
        sx[k] = tx * c1 + cx * s1;
        cx = cx * c1 - tx * s1;         // 115
    }

    float zz[30];                       // ZZ(29), 1-based
    for (int j = 1; j <= lm; ++j) {     // DO 125
        float r = 0.0f;
        for (int k = 1; k <= ln; ++k)   // DO 120
            r = r + sx[k] * noisy_p_(fakp, fakmap, hmym, kj, j, k);
        zz[j] = r + noisy_p_(fakp, fakmap, hmym, kj, j, 16);  // + P(J,16,KJ)
    }

    // 110->145: latitude series (angle + 90 deg); NOISY's own D2R literal
    // (note the trailing 2)
    q = 0.01745329252f * (xla + 90.0f);
    s1 = sin32(q);
    c1 = cos32(q);
    sx[1] = s1;
    cx = c1;
    float r = 0.0f;
    for (int k = 1; k <= lm; ++k) {     // DO 130 — in-place sin(k*Q) recurrence
        r = r + sx[1] * zz[k];
        float ss = sx[1] * c1 + cx * s1;
        cx = cx * c1 - sx[1] * s1;
        sx[1] = ss;                     // 130
    }
    // final Fourier value (note linear normalization ALF + BET*Q)
    return r + alf + bet * q;
}

#endif  // VOACAP_NOISY_CUH
