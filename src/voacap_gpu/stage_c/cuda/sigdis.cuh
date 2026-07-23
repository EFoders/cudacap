// SIGDIS — signal-distribution tables and the D-E absorption-loss parameters
// for the path (SIGDIS_orig, the version.w32 'W'/'H' branch); the CUDA C twin
// of stage_a/sigdis.py.
//
// Part one loops over the KFX sample-area ionospheres, computing each area's
// absorption index ABIY (with the Greg Hand / ICEPAC lower limit of 0.1 that
// the 'W' version applies) and its system-loss deciles via SYSSY, then
// averages ADJ/SU/SL/ADS/SUS/SLS (/TON/) and ACAV/FEAV/AGLAT (/SIGD/) over
// the path.  Part two, for the circuit mode K=JMODE, sets the D-E loss
// parameters (HTLOSS/XVE/XNUZ/HNU/AFE/BFE) and the median/decile signal-level
// adjustments (ASM/DSU/DSL) by removing the Es-obscuration and F2-over-the-MUF
// contributions (via PRBMUF) at the reference frequency FTAB.
//
// The SYS/PERR tables are SHARED, not per-thread, and pass flat exactly as
// syssy.cuh documents; f2d111 is the F2D(1,1,1) scalar SYSSY's no-data test
// reads.  HTRUE/FVERT are the LayerState (31,6) arrays, so XLIN gets ncols=6.
//
// float32 throughout; the ALOG10-derived signal adjustments carry the usual
// ~1-ULP libm floor.  Rule 4b: /1.5 and /10 go through div32; the /XKF path
// averages divide by a genuinely runtime KFX and use plain `/`.

#ifndef VOACAP_SIGDIS_CUH
#define VOACAP_SIGDIS_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "prbmuf.cuh"
#include "state.cuh"
#include "syssy.cuh"
#include "xlin.cuh"

// state: LayerState (ABIY/ARTIC out); muf: MufState; sys/perr/f2d111: the
// shared /TWO/ tables; sigd: SigdState (out); ton: TonState (out);
// gcdkm: path km; jmode: mode.
__device__ inline void sigdis(LayerState* state, const MufState* muf,
                              const float* sys, const float* perr,
                              float f2d111, SigdState* sigd, TonState* ton,
                              float gcdkm, int jmode) {
    const float* fi = state->fi;
    float glav = 0.0f;
    int kfxx = state->kfx;
    float xkf = (float)kfxx;
    float xadj = 0.0f, xsu = 0.0f, xsl = 0.0f;
    float xfmp = 0.0f, xsup = 0.0f, xslp = 0.0f;
    float ac = 0.0f;
    float feav = 0.0f;
    for (int k = 1; k <= kfxx; ++k) {
        // absorption index (Greg Hand fix; the 'W' version applies the floor)
        float abiy = -0.04f + exp32(-2.937f + 0.8445f * fi[1 * 6 + k]);
        abiy = pymax(0.1f, abiy);
        state->abiy[k] = abiy;
        // system-loss table for this reflection area
        int idp = (gcdkm > 2500.0f) ? 5 : 2;
        float adj, su, sl, fmp, sup, slp;
        syssy(sys, perr, f2d111, state->glat[k], state->clck[k], idp,
              &adj, &su, &sl, &fmp, &sup, &slp);
        xadj = xadj + adj;
        xsu = xsu + su;
        xsl = xsl + sl;
        xfmp = xfmp + fmp;
        xsup = xsup + sup;
        xslp = xslp + slp;
        feav = feav + fi[1 * 6 + k];
        ac = ac + abiy;
        state->artic[k] = adj;
        glav = glav + fabsf(state->glat[k]);
    }
    // path averages (XKF is a runtime KFX, so plain `/`)
    glav = glav / xkf;
    sigd->aglat = glav;
    sigd->acav = ac / xkf;
    feav = feav / xkf;
    sigd->feav = feav;
    ton->adj = xadj / xkf;
    ton->su = xsu / xkf;
    ton->sl = xsl / xkf;
    ton->ads = xfmp / xkf;
    ton->sus = xsup / xkf;
    ton->sls = xslp / xkf;
    int k = jmode;
    // D-E region loss adjustment factor
    sigd->htloss = 88.0f;
    float xve = xlin(90.0f, state->htrue, state->fvert, 30, k, 6);
    sigd->xve = xve / fi[1 * 6 + k];
    sigd->xnuz = 63.07f;
    sigd->hnu = 4.39f;
    // CCIR 252 (Haydon/Lucas) loss adjustment for E modes
    if (feav - 2.0f > 0.0f) {           // IF(FEAV-2.0) 125,125,120
        sigd->afe = 1.359f;
        sigd->bfe = 8.617f;
    } else if (feav - 0.5f <= 0.0f) {   // 125  IF(FEAV-0.5) 130,130,135
        sigd->afe = 0.0f;
        sigd->bfe = 0.0f;
    } else {                            // 135
        sigd->afe = div32(1.359f * (feav - 0.5f), 1.5f);
        sigd->bfe = div32(8.617f * (feav - 0.5f), 1.5f);
    }
    // signal-distribution reference frequency FTAB (FOT, F2 layer)
    glav = glav * VC_R2D;
    float ftab;
    if (glav - 40.0f <= 0.0f) {         // IF(GLAV-40.) 145,145,150
        ftab = muf->yfot[3];
    } else if (glav - 50.0f <= 0.0f) {  // 150  IF(GLAV-50.) 155,155,160
        ftab = muf->yfot[3];
        ftab = ftab - div32((glav - 40.0f) * (ftab - 10.0f), 10.0f);
    } else {                            // 160  near poles
        ftab = 10.0f;
    }
    // Es obscuration contribution (removed here, replaced per mode)
    float eslsm = 0.0f;
    float pes, pf2, dummy;
    if (muf->ymuf[4] > 0.0f) {
        dummy = muf->ymuf[4];
        pes = prbmuf(muf, ftab, dummy, dummy, 4);
        pes = pymin(pymax(pes, 0.1f), 0.9f);
        eslsm = -10.0f * log10_32(1.0f - pes);
    }
    // F2 over-the-MUF contribution
    dummy = muf->ymuf[3];
    pf2 = prbmuf(muf, ftab, dummy, dummy, 3);
    pf2 = pymax(pf2, 0.1f);
    float f2lsm = -10.0f * log10_32(pf2);
    // residual (auroral) loss adjustment to the median signal level
    float asm_v = ton->adj - eslsm - f2lsm;
    sigd->asm_ = pymax(asm_v, 0.0f);
    // upper decile
    pes = 0.0f;
    if (muf->yfot[4] > 0.0f) {
        dummy = muf->yfot[4];
        pes = prbmuf(muf, ftab, dummy, dummy, 4);
        pes = pymin(pymax(pes, 0.1f), 0.9f);
    }
    dummy = muf->yhpf[3];
    pf2 = prbmuf(muf, ftab, dummy, dummy, 3);
    pf2 = pymax(pf2, 0.1f);
    float dsu = 1.28f * ton->sl
                - (10.0f * log10_32(1.0f - pes) + eslsm)
                - (10.0f * log10_32(pf2) + f2lsm);
    sigd->dsu = pymax(dsu, 0.5f);
    // lower decile
    pes = 0.0f;
    if (muf->yhpf[4] > 0.0f) {
        dummy = muf->yhpf[4];
        pes = prbmuf(muf, ftab, dummy, dummy, 4);
        pes = pymin(pymax(pes, 0.1f), 0.9f);
    }
    dummy = muf->yfot[3];
    pf2 = prbmuf(muf, ftab, dummy, dummy, 3);
    pf2 = pymax(pf2, 0.1f);
    float dsl = 1.28f * ton->su
                - (-10.0f * log10_32(1.0f - pes) - eslsm)
                - (-10.0f * log10_32(pf2) - f2lsm);
    sigd->dsl = pymax(dsl, 1.0f);
}

#endif  // VOACAP_SIGDIS_CUH
