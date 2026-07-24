// ALOSFV — the deviative-loss factor AFAC(30,5) for sample area K; the
// CUDA C twin of scalar/alosfv.py (alosfv.for).  Zero except for
// high-angle modes reflecting above the height of the MUF, where the loss
// is the average collision frequency (an exponential CF in height) times
// the group-minus-phase path excess (HPRIM - HTRUE - HM), normalised to the
// loss equation and forced continuous across the E->F1->F2 cusps.
//
// Reads the ionogram (HTRUE/HPRIM from GENION) and the per-layer MUF
// heights (HPMUF/HTMUF from CURMUF, in MufState); its YI(2,K) input is
// LECDEN's ADJUSTED value — pass the same LayerState LECDEN mutated,
// exactly as Stage A's signature does.  Mutates AFAC ((31, 6) row-major,
// [ih * 6 + k]).  REAL*4; only EXP appears.
//
// The Fortran leaves CF undefined if no E-layer point reflects above
// HTMUF(1) (then A(2)=CF at the E->F cusp is garbage); Stage A initialises
// cf = 0.0 and so does this port.

#ifndef VOACAP_ALOSFV_CUH
#define VOACAP_ALOSFV_CUH

#include "f32math.cuh"
#include "state.cuh"

__device__ inline void alosfv(LayerState* s, const MufState* muf, int k) {
    float* fi = s->fi;
    float* yi = s->yi;
    float* hi = s->hi;
    float* htrue = s->htrue;
    float* hprim = s->hprim;
    float* afac = s->afac;
    const float* hpmuf = muf->hpmuf;
    const float* htmuf = muf->htmuf;

    float a1 = 0.2f, a2 = 0.1f, a3 = 0.1f;  // A(1..3), DATA 0.2,0.1,0.1
    float hm1 = 0.0f, hm2 = 0.0f;           // HM(1..2)
    float cf = 0.0f;

    // ---- E layer (IH 1..10) ----
    hm1 = hpmuf[1] - htmuf[1];
    float hz = hi[1 * 6 + k] - yi[1 * 6 + k];
    for (int ih = 1; ih <= 10; ++ih) {  // DO 105
        if (htrue[ih * 6 + k] - htmuf[1] <= 0.0f) {  // 100
            afac[ih * 6 + k] = 0.0f;
        } else {  // 102: collision-frequency adjustment factor
            float zexp = -2.0f * (htrue[ih * 6 + k] - hz) / yi[1 * 6 + k];
            zexp = pymax(zexp, -10.0f);
            cf = a1 * exp32(zexp);
            afac[ih * 6 + k] = cf * (hprim[ih * 6 + k] - htrue[ih * 6 + k]
                                     - hm1);
        }
    }

    if (fi[2 * 6 + k] <= 0.0f) {  // 115: F2 layer with no F1 ledge
        a2 = cf;                                 // continuity at E->F2 cusp
        hz = htrue[11 * 6 + k];
        hm2 = hprim[13 * 6 + k] - htrue[13 * 6 + k];
        for (int ih = 11; ih <= 12; ++ih) {  // DO 120 IH=11,12
            float zexp = -2.0f * (htrue[ih * 6 + k] - hz) / yi[3 * 6 + k];
            zexp = pymax(zexp, -10.0f);
            cf = a2 * exp32(zexp);
            cf = pymax(cf, 0.05f);
            afac[ih * 6 + k] = cf * (hprim[ih * 6 + k] - htrue[ih * 6 + k]
                                     - hm2);
        }
        afac[13 * 6 + k] = 0.0f;
        a3 = 0.1f;
        hm2 = hpmuf[3] - htmuf[3];
        for (int ih = 14; ih <= 30; ++ih) {  // DO 135 IH=14,30
            if (htrue[ih * 6 + k] - htmuf[3] <= 0.0f) {  // 125
                afac[ih * 6 + k] = 0.0f;
            } else {  // 130
                float zexp = -2.0f * (htrue[ih * 6 + k] - hz) / yi[3 * 6 + k];
                zexp = pymax(zexp, -10.0f);
                cf = a3 * exp32(zexp);
                cf = pymax(cf, 0.5f);
                afac[ih * 6 + k] = cf * (hprim[ih * 6 + k] - htrue[ih * 6 + k]
                                         - hm2);
            }
        }
    } else {  // 140 -> 145: F1 layer present
        a2 = cf;                                 // continuity at E->F1 cusp
        hz = htrue[11 * 6 + k];
        hm2 = hprim[13 * 6 + k] - htrue[13 * 6 + k];
        for (int ih = 11; ih <= 12; ++ih) {  // DO 150 IH=11,12
            float zexp = -2.0f * (htrue[ih * 6 + k] - hz) / yi[2 * 6 + k];
            zexp = pymax(zexp, -10.0f);
            cf = a2 * exp32(zexp);
            cf = pymax(cf, 0.05f);
            afac[ih * 6 + k] = cf * (hprim[ih * 6 + k] - htrue[ih * 6 + k]
                                     - hm2);
        }
        afac[13 * 6 + k] = 0.0f;
        a2 = 0.1f;
        float htm = htmuf[2];
        float htp = hpmuf[2];
        hm2 = htp - htm;
        for (int ih = 14; ih <= 20; ++ih) {  // DO 165 IH=14,20
            if (htrue[ih * 6 + k] - htm <= 0.0f) {  // 155
                afac[ih * 6 + k] = 0.0f;
            } else {  // 160
                float zexp = -2.0f * (htrue[ih * 6 + k] - hz) / yi[2 * 6 + k];
                zexp = pymax(zexp, -10.0f);
                cf = a3 * exp32(zexp);  // A(3), faithful to Stage A / Fortran
                cf = pymax(cf, 0.05f);
                afac[ih * 6 + k] = cf * (hprim[ih * 6 + k] - htrue[ih * 6 + k]
                                         - hm2);
            }
        }
        // F2 layer with F1 ledge -- continuity at F1->F2
        a3 = cf;
        hz = htrue[21 * 6 + k];
        hm2 = hprim[23 * 6 + k] - htrue[23 * 6 + k];
        for (int ih = 21; ih <= 22; ++ih) {  // DO 170 IH=21,22
            float zexp = -2.0f * (htrue[ih * 6 + k] - hz) / yi[3 * 6 + k];
            zexp = pymax(zexp, -10.0f);
            cf = a3 * exp32(zexp);
            cf = pymax(0.05f, cf);
            afac[ih * 6 + k] = cf * (hprim[ih * 6 + k] - htrue[ih * 6 + k]
                                     - hm2);
        }
        afac[23 * 6 + k] = 0.0f;
        a3 = 0.1f;
        htm = htmuf[3];
        htp = hpmuf[3];
        hm2 = htp - htm;
        for (int ih = 24; ih <= 30; ++ih) {  // DO 185 IH=24,30
            if (htrue[ih * 6 + k] - htm <= 0.0f) {  // 175
                afac[ih * 6 + k] = 0.0f;
            } else {  // 180
                float zexp = -2.0f * (htrue[ih * 6 + k] - hz) / yi[3 * 6 + k];
                zexp = pymax(zexp, -10.0f);
                cf = a3 * exp32(zexp);
                cf = pymax(cf, 0.05f);
                afac[ih * 6 + k] = cf * (hprim[ih * 6 + k] - htrue[ih * 6 + k]
                                         - hm2);
            }
        }
    }

    for (int ih = 1; ih <= 30; ++ih)  // DO 205: floor at zero
        afac[ih * 6 + k] = pymax(0.0f, afac[ih * 6 + k]);

    (void)a2;  // a2 is dead after its last read, as in Stage A
}

#endif  // VOACAP_ALOSFV_CUH
