// VERSY — geographic variation of the long-term ionospheric maps at one
// sample area; the CUDA C twin of scalar/versy.py.
//
// Builds the (up to 76) geographic coordinate functions G — powers of sin(x)
// plus cos(lat)^m * {cos,sin}(m*lon) blocks, X being the Rawer magnetic dip
// except plain latitude for the regular-E set — and dots them with the AB
// coefficients from VIRTIM into GAMMA(IZ) for IZ = KIK..LIK
// (1..3 Es low/median/up, 4 foF2, 5 M(3000)F2, 6 foE).
//
// PRECISION.  The Fortran declares ``real*8 sx,x,g,cx,t`` (an underflow fix:
// cos(lat)^m underflows binary32 near the poles), so exactly those five are
// ``double`` here, DSIN/DCOS are the binary64 sin/cos, and everything else
// stays REAL*4 — Y, GOB=COS(CENLAT) (a REAL*4 cosine through cos32), FJ, AB
// and GAMMA.  Two narrowings are load-bearing and must not be "simplified":
//   * T = FJ*Y is a REAL*4 product that is only then widened into the
//     binary64 T, and
//   * GAMMA is narrowed back to binary32 on EVERY accumulation statement,
//     while the accumulation itself is done in binary64.
//
// VERSY owns no tables.  Like NOISY it takes the REDMAP state it needs as a
// flat row-major device pointer plus its column count, so the two compose in
// one kernel:
//     ikim (10,6) int32   element [a,b] = ikim[a * ikim_ncols + b]
// AB is the 1-based padded (319,) array VIRTIM returns (index 0 unused), and
// GAMMA is written as a 1-based (7,) array (index 0 unused); entries outside
// KIK..LIK are zeroed, exactly as Stage A's ``np.zeros(7)`` leaves them.

#ifndef VOACAP_VERSY_CUH
#define VOACAP_VERSY_CUH

#include "f32math.cuh"
#include "constants.cuh"

// The /ONE/ AB(318) start offsets per set (redmap.py's IC, from the
// versy.for / virtim.for DATA statements).
__device__ __forceinline__ int versy_ic_(int iz) {
    const int ic[6] = {1, 56, 117, 172, 248, 297};
    return ic[iz - 1];
}

// CLAT/CLONG/GMDIP in radians (float32); KIK..LIK the layer indices to fill.
// Writes GAMMA(1..6) into gamma[1..6] (gamma[0] is the unused 1-based pad).
__device__ inline void versy(const int* ikim, int ikim_ncols, const float* ab,
                             float clat, float clong, float gmdip,
                             int kik, int lik, float* gamma) {
    float cenlg = clong;
    float cenlat = clat;
    float c360 = 360.0f * VC_D2R;
    float clg;
    if (cenlg < 0.0f)               // 100: change to east longitude
        clg = c360 + cenlg;
    else                            // 105
        clg = cenlg;
    float gob = cos32(cenlat);      // 110 — REAL*4 cosine

    for (int j = 0; j < 7; ++j)
        gamma[j] = 0.0f;

    double g[77];                   // G(76), 1-based (real*8)

    for (int iz = kik; iz <= lik; ++iz) {        // DO 200
        double x;
        if (iz != 6)                // 115: Rawer magnetic dip angle
            x = (double)gmdip;
        else                        // 120: use latitude for regular E
            x = (double)cenlat;
        float y = clg;              // 125 — Y stays REAL*4
        int i = ikim[8 * ikim_ncols + (iz - 1)] + 1;   // IKIM(9,IZ)+1
        int k = ikim[0 * ikim_ncols + (iz - 1)];       // IKIM(1,IZ)
        double sx = sin(x);         // DSIN (X is real*8)

        for (int jj = 0; jj < 77; ++jj)
            g[jj] = 0.0;
        g[1] = 1.0;
        g[2] = sx;
        if (k != 1) {               // 130: powers of sin
            if (k >= 2) {
                for (int ka = 2; ka <= k; ++ka)        // DO 135
                    g[ka + 1] = sx * g[ka];
            }
        }
        // 140
        int kdif = ikim[1 * ikim_ncols + (iz - 1)] - k;   // IKIM(2,IZ) - K
        if (kdif != 0) {            // 145
            int jg = 1;
            double cx = (double)gob;
            double t = (double)y;
            while (true) {          // label 150
                // IKIM(JG,IZ)+4
                int kk = ikim[(jg - 1) * ikim_ncols + (iz - 1)] + 4;
                g[kk - 2] = cx * cos(t);               // DCOS
                g[kk - 1] = cx * sin(t);
                int lo = ikim[jg * ikim_ncols + (iz - 1)];   // IKIM(JG+1,IZ)
                if (kdif != 2) {    // 155
                    if (lo >= kk) { // 160
                        for (int ka = kk; ka <= lo; ka += 2) {   // DO 170
                            g[ka] = sx * g[ka - 2];    // 165
                            g[ka + 1] = sx * g[ka - 1];    // 170
                        }
                    }
                }
                // 175
                if (jg == 8)        // IF (JG-8) 180,190,180
                    break;
                // IKIM(JG+2,IZ) - LO
                kdif = ikim[(jg + 1) * ikim_ncols + (iz - 1)] - lo;
                if (kdif == 0)      // 190
                    break;
                cx = cx * (double)gob;                 // 185
                jg = jg + 1;
                float fj = (float)jg;
                // REAL*4 product, then widened into T
                t = (double)(fj * y);
            }
        }
        // 190: final summation of the Fourier expansion
        int isuba = versy_ic_(iz);
        gamma[iz] = (float)(g[1] * (double)ab[isuba]);
        if (i >= 2) {
            for (int jb = 2; jb <= i; ++jb) {          // DO 195
                isuba = versy_ic_(iz) + jb - 1;
                // GAMMA = GAMMA + AB(ISUBA)*G(JB): f64 sum, REAL*4 store
                gamma[iz] = (float)((double)gamma[iz]
                                    + (double)ab[isuba] * g[jb]);
            }
        }
    }
}

#endif  // VOACAP_VERSY_CUH
