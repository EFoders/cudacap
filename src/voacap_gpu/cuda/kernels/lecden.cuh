// LECDEN — builds the electron-density profile HTR(50,1)/FNSQ(50,1) (true
// height km, plasma frequency squared MHz^2) from the E/F1/F2 layer
// parameters for sample area K; the CUDA C twin of scalar/lecden.py
// (lecden.for).  Takes the pointwise maximum of the D-region exponential
// tail, the parabolic E, a linear valley, the parabolic-or-linear F1, and
// the parabolic F2.
//
// MUTATES HI(2,K)/YI(2,K)/FSECV(K) during the F1 fit — the adjusted YI(2,K)
// is read downstream (ALOSFV, absorption), so the in/out contract is exactly
// Stage A's: pass the same LayerState on and the adjusted values are in it.
// REAL*4; only SQRT (correctly rounded) and EXP appear.
//
// The one arithmetic note: Stage A writes ``(fi[1,k]/fi[2,k]) ** 2`` (numpy
// scalar power).  gfortran emits a multiply for an INTEGER exponent and so
// does every other port in this project (see batched/profile.py); numpy's
// f32 powf deviates from the correctly rounded square by 1 ULP on ~0.03% of
// arguments.  Here it is the multiply t*t; the Stage C test counts any
// resulting divergence and asserts it rare.
//
// FI/YI/HI live in LayerState as (4, 6) row-major arrays indexed
// [i * 6 + k]; FSECV is (4,); HTR/FNSQ are (51, 4) indexed [ih * 4 + 1].

#ifndef VOACAP_LECDEN_CUH
#define VOACAP_LECDEN_CUH

#include "f32math.cuh"
#include "state.cuh"

__device__ inline void lecden(LayerState* s, int k) {
    float* fi = s->fi;
    float* yi = s->yi;
    float* hi = s->hi;
    float* htr = s->htr;
    float* fnsq = s->fnsq;
    float* fsecv = s->fsecv;

    // external electron-density profile in use -> nothing to build
    for (int i = 1; i <= 3; ++i)        // DO 90
        if (s->ielect[i] > 0)           // 85
            return;

    float xlow = 0.8516f;
    float hz = hi[1 * 6 + k] - yi[1 * 6 + k];
    float xup = 0.98f * fi[1 * 6 + k] / fi[3 * 6 + k];
    float hlow = hz + yi[1 * 6 + k] * (1.0f + sqrt32(1.0f - xlow * xlow));
    float hte = hi[1 * 6 + k] + yi[1 * 6 + k];
    float hb2 = hi[3 * 6 + k] - yi[3 * 6 + k];
    float fce = fi[1 * 6 + k] * fi[1 * 6 + k];
    float fc2 = fi[3 * 6 + k] * fi[3 * 6 + k];
    float hup = hb2 + yi[3 * 6 + k] * (1.0f - sqrt32(1.0f - xup * xup));
    float fup = xup * xup * fc2;
    float flow = xlow * xlow * fce;
    float asp = 0.0f;
    if (hup - hlow > 0.0f)              // 165
        asp = (fup - flow) / (hup - hlow);

    // ---- F1-layer fit (may move HI(2)/YI(2)) ----
    int lin = 0;
    float s1 = 0.0f;
    float fc1 = 0.0f;
    float hb1 = hi[2 * 6 + k] - yi[2 * 6 + k];   // line 42 (original hi/yi)
    float ht1 = hi[2 * 6 + k] + yi[2 * 6 + k];   // line 43 (original hi/yi)
    if (fi[2 * 6 + k] > 0.0f) {         // 185
        fc1 = fi[2 * 6 + k] * fi[2 * 6 + k];
        float htw = hb2 + yi[3 * 6 + k] * (1.0f - sqrt32(1.0f - fc1 / fc2));
        if (htw > hi[2 * 6 + k] + 0.001f) {      // -> 205
            lin = 0;
            yi[2 * 6 + k] = pymin(yi[2 * 6 + k],
                                  hi[2 * 6 + k] - hi[1 * 6 + k] + 1.0f);
            fsecv[k] = -1.0f;
        } else {                        // 190
            float ys = htw - hb1;
            ys = pymax(1.0f, ys);
            s1 = fc1 / ys;
            hi[2 * 6 + k] = htw;
            yi[2 * 6 + k] = ys;
            if (hb2 - hb1 < 0.0f) {     // 194 -> 205
                yi[2 * 6 + k] = hi[2 * 6 + k] - hb2;
                lin = 0;
                yi[2 * 6 + k] = pymin(yi[2 * 6 + k],
                                      hi[2 * 6 + k] - hi[1 * 6 + k] + 1.0f);
                fsecv[k] = -1.0f;
            } else {                    // 195
                lin = 1;
                fsecv[k] = sqrt32(flow);
                float t = fi[1 * 6 + k] / fi[2 * 6 + k];
                float denom = 1.0f - t * t;      // Stage A: 1 - (...) ** 2
                denom = pymax(0.17f, denom);
                float yb = (hi[2 * 6 + k] - hi[1 * 6 + k]) / denom;
                if (ys - yb >= 0.0f) {  // 401 (else 402 falls through)
                    ys = yb;
                    yi[2 * 6 + k] = ys;
                    s1 = fc1 / ys;
                    hb1 = hi[1 * 6 + k];
                }
                ht1 = htw;              // 415
            }
        }
    }

    // ---- D-region tail + the HTR height grid ----
    float hd = 70.0f;
    float xtail = 0.85f;
    float hex = hi[1 * 6 + k] - xtail * yi[1 * 6 + k];
    float fnx = 1.0f - xtail * xtail;
    float alp = 2.0f * (hi[1 * 6 + k] - hex)
                / (fnx * yi[1 * 6 + k] * yi[1 * 6 + k]);
    float fsq = fnx * exp32(-alp * (hex - hd));
    htr[1 * 4 + 1] = hd;
    htr[5 * 4 + 1] = hex;
    float hdif = (htr[5 * 4 + 1] - htr[1 * 4 + 1]) * 0.25f;
    hdif = pymax(0.0f, hdif);
    htr[4 * 4 + 1] = htr[5 * 4 + 1] - pymin(1.0f, hdif);
    htr[2 * 4 + 1] = htr[1 * 4 + 1] + hdif;
    htr[3 * 4 + 1] = (htr[2 * 4 + 1] + htr[4 * 4 + 1]) * 0.5f;
    htr[11 * 4 + 1] = hi[1 * 6 + k];    // E below nose
    hdif = div32(htr[11 * 4 + 1] - htr[5 * 4 + 1], 6.0f);
    for (int ih = 6; ih <= 10; ++ih)    // DO 100
        htr[ih * 4 + 1] = htr[(ih - 1) * 4 + 1] + hdif;
    htr[17 * 4 + 1] = hi[1 * 6 + k] + yi[1 * 6 + k];  // E above nose
    hdif = div32(htr[17 * 4 + 1] - htr[11 * 4 + 1], 6.0f);
    for (int ih = 12; ih <= 16; ++ih)   // DO 105
        htr[ih * 4 + 1] = htr[(ih - 1) * 4 + 1] + hdif;
    htr[11 * 4 + 1] = 0.5f * (htr[10 * 4 + 1] + htr[12 * 4 + 1]);
    htr[50 * 4 + 1] = hi[3 * 6 + k];
    bool two_layers = fi[2 * 6 + k] > 0.0f;
    if (two_layers) {                   // 125
        hb2 = hi[3 * 6 + k] - yi[3 * 6 + k];
        float hb1b = hi[2 * 6 + k] - yi[2 * 6 + k] + 0.00001f;
        if (hb2 - hb1b <= 0.0f)         // 110 (treat as no F1)
            two_layers = false;
    }
    if (!two_layers) {                  // 110: F2 layer, no F1 layer
        htr[18 * 4 + 1] = hi[3 * 6 + k] - yi[3 * 6 + k];
        hdif = div32(htr[50 * 4 + 1] - htr[18 * 4 + 1], 32.0f);
        for (int ih = 19; ih <= 49; ++ih)  // DO 115
            htr[ih * 4 + 1] = htr[(ih - 1) * 4 + 1] + hdif;
    } else {                            // 126: F1 and F2 layers
        htr[18 * 4 + 1] = hi[2 * 6 + k] - yi[2 * 6 + k];
        htr[18 * 4 + 1] = pymax(htr[18 * 4 + 1], htr[17 * 4 + 1] + 1.0f);
        htr[28 * 4 + 1] = hi[2 * 6 + k];
        hdif = div32(htr[28 * 4 + 1] - htr[18 * 4 + 1], 10.0f);
        for (int ih = 19; ih <= 27; ++ih)  // DO 135
            htr[ih * 4 + 1] = htr[(ih - 1) * 4 + 1] + hdif;
        hdif = div32(htr[50 * 4 + 1] - htr[28 * 4 + 1], 22.0f);
        for (int ih = 29; ih <= 49; ++ih)  // DO 145
            htr[ih * 4 + 1] = htr[(ih - 1) * 4 + 1] + hdif;
    }

    // ---- per-height density (max of the layer contributions) ----
    hb1 = pymax(hi[1 * 6 + k], hi[2 * 6 + k] - yi[2 * 6 + k]);  // line 143
    float hi1k = hi[1 * 6 + k], yi1k = yi[1 * 6 + k];
    float hi2k = hi[2 * 6 + k], yi2k = yi[2 * 6 + k];
    float hi3k = hi[3 * 6 + k], yi3k = yi[3 * 6 + k];
    bool has_f1 = fi[2 * 6 + k] > 0.0f;
    for (int ih = 1; ih <= 50; ++ih) {  // DO 375
        float fnd = 0.0f, fne = 0.0f, fn1 = 0.0f, fn2 = 0.0f, fnval = 0.0f;
        float h = htr[ih * 4 + 1];
        if (h - hlow > 0.0f) {          // 225
            if (h - hup < 0.0f)         // 235 linear valley
                fnval = fup + asp * (h - hup);
        }
        // 245
        if (h - hex < 0.0f) {           // 275 exponential D-E
            fnd = fce * fsq * exp32(alp * (h - hd));
        } else if (h - hte <= 0.0f) {   // 260 parabolic E
            float z = (h - hi1k) / yi1k;
            fne = fce * (1.0f - z * z);
        }
        // 285
        if (has_f1) {                   // 290
            if (hb1 - h <= 0.0f && h - ht1 <= 0.0f) {  // 295 -> 300
                if (lin <= 0) {         // 305 parabolic F1
                    float z = (h - hi2k) / yi2k;
                    fn1 = fc1 * (1.0f - z * z);
                } else {                // 335 linear F1
                    fn1 = s1 * (h - (hi2k - yi2k));
                }
            }
        }
        // 345
        if (hb2 - h <= 0.0f) {          // 355 parabolic F2
            float z = (h - hi3k) / yi3k;
            fn2 = fc2 * (1.0f - z * z);
        }
        // 365: Python max(fnd, fne, fnval, fn1, fn2) folds left
        fnsq[ih * 4 + 1] =
            pymax(pymax(pymax(pymax(fnd, fne), fnval), fn1), fn2);
    }
}

#endif  // VOACAP_LECDEN_CUH
