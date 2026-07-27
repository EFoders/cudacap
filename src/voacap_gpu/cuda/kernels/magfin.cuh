// MAGFIN / MAGVAR — the Jensen & Cain (1960 epoch) spherical-harmonic
// geomagnetic field, degree/order 6, and the wrapper that turns it into the
// gyrofrequency and Rawer magnetic dip the ionospheric chain consumes.  The
// CUDA C twin of scalar/magfin.py (reference/voacapl/src/voacapw/magfin.for,
// magvar.for).
//
// The whole routine is data-INDEPENDENT in its control flow except for one
// branch: the |latitude| clamp to RD (~89.897 deg), which also zeroes the
// longitude.  The Legendre P/DP recurrences are indexed by the loop counters
// (n, m) only, so they stay plain C loops with fully unrollable trip counts —
// nothing here needs predication.
//
// The Fortran P/DP/SP/CP work arrays are statics initialised by DATA, but
// every entry a call reads is either a DATA constant that is never
// overwritten (P(1,1)=1, DP(1,1)=0, CP(1)=1, SP(1)=0) or written earlier in
// the same call, so fresh zeroed locals per call are equivalent — and the
// ZEROES ARE LOAD-BEARING: the n>=3, m=n-1 step of the DO-140 recurrence
// reads P(n-2,n-1)/DP(n-2,n-1), which no iteration ever writes.
//
// Arrays are 1-based (slot 0 unused) to keep subscripts identical to the
// Fortran; the G/H/CT DATA tables are column-major in the Fortran source and
// are transposed here into row-major [n][m] literals (verified elementwise
// against Stage A's _CT/_G/_H).
//
// REAL*4 throughout; SIN/COS/SQRT/ATAN go through f32math (binary64 evaluate,
// single rounding).  One div32 site: AR = HC/(HC+HEIGHT) — MAGVAR always
// passes the literal 300000.0, so after inlining the denominator is
// compile-time constant and NVRTC would strength-reduce the divide.

#ifndef VOACAP_MAGFIN_CUH
#define VOACAP_MAGFIN_CUH

#include "constants.cuh"
#include "f32math.cuh"

#define MAGFIN_RD 1.56905124f   // latitude clamp, radians (~89.897 deg)
#define MAGFIN_HC 6371200.0f    // earth radius, metres

// Schmidt recurrence coefficients CT(n,m), 1-based [n][m]; row/col 0 padded.
__device__ __constant__ float MAGFIN_CT[8][8] = {
    {0.0f, 0.0f,       0.0f,       0.0f,       0.0f,       0.0f, 0.0f},
    {0.0f, 0.0f,       0.0f,       0.0f,       0.0f,       0.0f, 0.0f},
    {0.0f, 0.0f,       0.0f,       0.0f,       0.0f,       0.0f, 0.0f},
    {0.0f, .33333333f, 0.0f,       0.0f,       0.0f,       0.0f, 0.0f},
    {0.0f, .26666667f, .2f,        0.0f,       0.0f,       0.0f, 0.0f},
    {0.0f, .25714286f, .22857142f, .14285714f, 0.0f,       0.0f, 0.0f},
    {0.0f, .25396825f, .23809523f, .19047619f, .11111111f, 0.0f, 0.0f},
    {0.0f, .25252525f, .24242424f, .21212121f, .16161616f, .09090909f, 0.0f},
};

// Gauss coefficients G(n,m), 1-based [n][m].
__device__ __constant__ float MAGFIN_G[8][8] = {
    {0.0f, 0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f},
    {0.0f, 0.0f,      0.0f,      0.0f,      0.0f,      0.0f,      0.0f},
    {0.0f, .304112f,  .021474f,  0.0f,      0.0f,      0.0f,      0.0f},
    {0.0f, .024035f,  -.051253f, -.013381f, 0.0f,      0.0f,      0.0f},
    {0.0f, -.031518f, .062130f,  -.024898f, -.006496f, 0.0f,      0.0f},
    {0.0f, -.041794f, -.045298f, -.021795f, .007008f,  -.002044f, 0.0f},
    {0.0f, .016256f,  -.034407f, -.019447f, -.000608f, .002775f,  .000697f},
    {0.0f, -.019523f, -.004853f, .003212f,  .021413f,  .001051f,  .000227f},
};

// The m=7 column of G, split out so the initialiser above stays 7 wide.
// (G(7,7) is the only non-zero entry in it.)
__device__ __constant__ float MAGFIN_G77 = .001115f;

// Gauss coefficients H(n,m), 1-based [n][m].
__device__ __constant__ float MAGFIN_H[8][8] = {
    {0.0f, 0.0f, 0.0f,      0.0f,      0.0f,      0.0f,      0.0f},
    {0.0f, 0.0f, 0.0f,      0.0f,      0.0f,      0.0f,      0.0f},
    {0.0f, 0.0f, -.057989f, 0.0f,      0.0f,      0.0f,      0.0f},
    {0.0f, 0.0f, .033124f,  -.001579f, 0.0f,      0.0f,      0.0f},
    {0.0f, 0.0f, .014870f,  -.004075f, .00021f,   0.0f,      0.0f},
    {0.0f, 0.0f, -.011825f, .010006f,  .00043f,   .001385f,  0.0f},
    {0.0f, 0.0f, -.000796f, -.002f,    .004597f,  .002421f,  -.001218f},
    {0.0f, 0.0f, -.005758f, -.008735f, -.003406f, -.000118f, -.001116f},
};

// The m=7 column of H (H(7,7) only).
__device__ __constant__ float MAGFIN_H77 = -.000325f;

// UNE — the three field components MAGFIN returns, in gauss.
struct MagfinUne {
    float up;       // UNE(1) = -BV
    float north;    // UNE(2) =  BN
    float east;     // UNE(3) = -BPHI/S
};

// MAGFIN(P1,P2,HEIGHT,UNE): geographic latitude and EAST longitude in
// radians, height in METRES.  Returns the (up, north, east) field in gauss.
__device__ inline MagfinUne magfin(float lat_rad, float east_lon_rad,
                                   float height_m) {
    float p1 = lat_rad;
    float p2 = east_lon_rad;
    // magfin.for:44-50 (arithmetic IFs): clamp |lat| to RD, zeroing lon.
    // Exactly p1 == RD falls through UNCLAMPED, as in Stage A.
    if (p1 - MAGFIN_RD > 0.0f) {
        p1 = MAGFIN_RD;
        p2 = 0.0f;
    } else if (p1 - MAGFIN_RD < 0.0f) {
        if (p1 + MAGFIN_RD < 0.0f) {
            p1 = -MAGFIN_RD;
            p2 = 0.0f;
        }
    }

    float phi = p2;
    float ar = div32(MAGFIN_HC, MAGFIN_HC + height_m);
    float c = sin32(p1);
    float s = sqrt32(1.0f - c * c);          // CP(1) - C*C
    float sp[8], cp[8], aor[8];
    for (int i = 0; i < 8; ++i) {
        sp[i] = 0.0f;
        cp[i] = 0.0f;
        aor[i] = 0.0f;
    }
    cp[1] = 1.0f;
    sp[2] = sin32(phi);
    cp[2] = cos32(phi);
    aor[1] = ar * ar;
    aor[2] = aor[1] * ar;
    for (int m = 3; m <= 7; ++m) {           // DO 120
        sp[m] = sp[2] * cp[m - 1] + cp[2] * sp[m - 1];
        cp[m] = cp[2] * cp[m - 1] - sp[2] * sp[m - 1];
        aor[m] = ar * aor[m - 1];
    }

    float p[8][8], dp[8][8];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            p[i][j] = 0.0f;
            dp[i][j] = 0.0f;
        }
    p[1][1] = 1.0f;
    float bv = 0.0f;
    float bn = 0.0f;
    float bphi = 0.0f;
    for (int n = 2; n <= 7; ++n) {           // DO 145
        float fn = (float)n;
        float sumr = 0.0f;
        float sumt = 0.0f;
        float sump = 0.0f;
        for (int m = 1; m <= n; ++m) {       // DO 140
            if (n == m) {                    // 125
                p[n][n] = s * p[n - 1][n - 1];
                dp[n][n] = s * dp[n - 1][n - 1] + c * p[n - 1][n - 1];
            } else if (n == 2) {             // 132 (m=1)
                p[2][1] = c;
                dp[2][1] = -s;
            } else {                         // 134
                float ct = (m == 7) ? 0.0f : MAGFIN_CT[n][m];
                p[n][m] = c * p[n - 1][m] - ct * p[n - 2][m];
                dp[n][m] = (c * dp[n - 1][m] - s * p[n - 1][m]
                            - ct * dp[n - 2][m]);
            }
            float fm = (float)(m - 1);
            float g = (m == 7) ? MAGFIN_G77 : MAGFIN_G[n][m];
            float h = (m == 7) ? MAGFIN_H77 : MAGFIN_H[n][m];
            float ts = g * cp[m] + h * sp[m];
            sumr = sumr + p[n][m] * ts;
            sumt = sumt + dp[n][m] * ts;
            sump = sump + fm * p[n][m] * (-g * sp[m] + h * cp[m]);
        }
        bv = bv + aor[n] * fn * sumr;
        bn = bn - aor[n] * sumt;
        bphi = bphi - aor[n] * sump;
    }
    MagfinUne une;
    une.up = -bv;
    une.north = bn;
    une.east = -bphi / s;                    // s is a runtime denominator
    return une;
}

// MAGVAR(CENLAT,CENLG,GYZ,GMDIP,CLG): position in radians.  Writes the
// gyrofrequency GYZ (MHz), the Rawer magnetic dip GMDIP (radians) and the
// east longitude CLG folded into 0..2pi (radians).
__device__ inline void magvar(float cenlat, float cenlg, float* gyz,
                              float* gmdip, float* clg) {
    float c360 = 360.0f * VC_D2R;
    float lg;
    if (cenlg < 0.0f)                        // magvar.for:21-25
        lg = c360 + cenlg;
    else
        lg = cenlg;
    MagfinUne une = magfin(cenlat, lg, 300000.0f);
    float hhm = sqrt32(une.up * une.up + une.north * une.north
                       + une.east * une.east);
    float tmp = une.north * une.north + une.east * une.east;
    float gob = cos32(cenlat);
    if (gob <= 0.000001f)
        gob = 0.000001f;
    float x = atan32(atan32(-une.up / sqrt32(tmp)) / sqrt32(gob));
    *gyz = 2.8f * hhm;
    *gmdip = x;
    *clg = lg;
}

#endif  // VOACAP_MAGFIN_CUH
