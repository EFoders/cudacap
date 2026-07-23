// FINDF — area coverage: find all propagation modes at the operating
// frequency.  The CUDA C twin of stage_a/findf.py (mirrors findf.for, with
// SANG/PENANG already run, /REFLX/ 1-based, IAFTXR stored on ReflxState).
//
// For control area K and the operating frequency FREQ, FINDF walks the
// reflectrix table IFOB layer by layer (E, then F1/F2) and angle by angle,
// inserting the penetration-angle cusps (from PENANG, which it calls) into
// the take-off-angle scan, and for each ray records into ReflxState the
// take-off angle DELFX, the true/virtual heights HTFLX/HPFLX, the vertical
// frequency FVFLX, the deviative loss AFFLX and the layer IMODE.  Martyn's
// theorem is corrected for a spherical ionosphere (the SPH term) to give the
// ground distance GDFLX; DSKPKM/DMAXKM track the skip/max distances, the
// latter clamped to take-off angles at or above AMIND.
//
// The Fortran is a dense arithmetic-IF / GO TO state machine; Stage A kept
// it faithful by dispatching on the statement labels (`pc`), and this port
// transliterates the same web as C labels L225..L400 — one thread runs the
// whole thing scalar, so every branch matches by construction.  REAL*4.

#ifndef VOACAP_FINDF_CUH
#define VOACAP_FINDF_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "penang.cuh"
#include "state.cuh"

// state: LayerState (ionogram); reflx: ReflxState (out, needs DELPEN from
// PENANG which FINDF calls); rays: RaysState (ANG/IFOB/NANG, IFOB from
// FOBBY, NANG from SANG); k: area; freq: MHz; amind: min take-off angle
// (deg).  _ITYPE(IL) == IL, so layer indices are stored directly.
__device__ inline void findf(const LayerState* state, ReflxState* reflx,
                             const RaysState* rays, int k, float freq,
                             float amind) {
    float* delfx = reflx->delfx;
    float* hpflx = reflx->hpflx;
    float* htflx = reflx->htflx;
    float* gdflx = reflx->gdflx;
    float* fvflx = reflx->fvflx;
    float* afflx = reflx->afflx;
    int* imode = reflx->imode;
    float* delpen = reflx->delpen;
    const float* htrue = state->htrue;
    const float* hprim = state->hprim;
    const float* fvert = state->fvert;
    const float* afac = state->afac;
    const float* fi = state->fi;
    const float* ang = rays->ang;
    const int* ifob = rays->ifob;
    int nang = rays->nang;

    int jfhz = (int)(1000.0f * freq);
    reflx->dmaxkm[k] = 0.0f;
    reflx->dskpkm[k] = 10000.0f;
    for (int ia0 = 1; ia0 < 46; ++ia0) {  // DO 100
        hpflx[ia0 * 4 + k] = 0.0f;
        delfx[ia0 * 4 + k] = 0.0f;
        gdflx[ia0 * 4 + k] = 0.0f;
    }
    float fc2 = fi[3 * 6 + k] * fi[3 * 6 + k];
    penang(state, reflx, k, freq);

    int ia = 0;
    int iaf = 1;
    int icusp = -1;
    int il = 1;
    int ih = 1;
    int ilow = 1;
    int ihigh = 10;
    float fv = 0.0f, hp = 0.0f;
    int d;
    int guard = 0;  // translation-bug tripwire; never hit in practice.  Every
                    // cycle in the label web passes L275 or L305 — a runaway
                    // exits instead of hanging the device, and the resulting
                    // state mismatch fails the bit-exact compare loudly.
    goto L275;

L225:                                   // SET FOR F LAYER
    ih = 11;
    ilow = 11;
    if (fi[2 * 6 + k] <= 0.0f) goto L235;
    goto L245;
L235:                                   // F2 layer only
    il = 3;
    icusp = -1;
    ihigh = 30;
    goto L275;
L245:                                   // F1 layer  (245 -> 236 -> 275)
    il = 2;
    icusp = -1;
    ihigh = 20;
    goto L275;
L255:                                   // F2 layer after F1
    il = 3;
    icusp = -1;
    ilow = ihigh + 1;
    ihigh = 30;
    ih = 21;
    goto L275;
L265:                                   // GO TO (225,255,400),IL
    if (il == 1) goto L225;
    if (il == 2) goto L255;
    goto L400;
L275:                                   // START OF SEARCH
    if (++guard > 200000) goto L400;
    if (delpen[il * 6 + k] <= 0.0f) goto L265;
    // L285
    if (delpen[il * 6 + k] - 89.99f <= 0.0f) goto L295;
    goto L400;
L295:                                   // INCREMENT ANGLE
    ia = ia + 1;
    if (ia - nang <= 0) goto L300;
    goto L400;
L300:
    if (delpen[il * 6 + k] - ang[ia] <= 0.0f) goto L345;
    goto L305;
L305:                                   // SEARCH FOR FREQUENCY
    if (++guard > 200000) goto L400;
    if (ifob[(ia * 31 + ilow) * 6 + k] - jfhz < 0) goto L306;
    goto L325;
L306:
    if (ih - ihigh < 0) goto L315;
    goto L275;
L315:
    d = ifob[(ia * 31 + ih) * 6 + k] - jfhz;
    if (d < 0) goto L335;
    if (d == 0) goto L325;
    goto L330;
L325:                                   // EXACT FREQUENCY
    delfx[iaf * 4 + k] = ang[ia];
    htflx[iaf * 4 + k] = htrue[ih * 6 + k];
    afflx[iaf * 4 + k] = afac[ih * 6 + k];
    fv = fvert[ih * 6 + k];
    hp = hprim[ih * 6 + k];
    imode[iaf * 4 + k] = il;            // _ITYPE(IL) == IL
    goto L375;
L330:                                   // INCREMENT HEIGHT INDEX
    ih = ih + 1;
    goto L305;
L335:
    if (ifob[(ia * 31 + ih + 1) * 6 + k] - jfhz < 0) goto L330;
    goto L340;
L340: {                                 // INTERPOLATION
    float slopd = (float)(ifob[(ia * 31 + ih + 1) * 6 + k]
                          - ifob[(ia * 31 + ih) * 6 + k]);
    slopd = pymax(1.0f, slopd);
    float slope = (float)(jfhz - ifob[(ia * 31 + ih) * 6 + k]) / slopd;
    htflx[iaf * 4 + k] = htrue[ih * 6 + k]
        + slope * (htrue[(ih + 1) * 6 + k] - htrue[ih * 6 + k]);
    fv = fvert[ih * 6 + k]
        + slope * (fvert[(ih + 1) * 6 + k] - fvert[ih * 6 + k]);
    delfx[iaf * 4 + k] = ang[ia];
    hp = hprim[ih * 6 + k]
        + slope * (hprim[(ih + 1) * 6 + k] - hprim[ih * 6 + k]);
    afflx[iaf * 4 + k] = afac[ih * 6 + k]
        + slope * (afac[(ih + 1) * 6 + k] - afac[ih * 6 + k]);
    imode[iaf * 4 + k] = il;            // _ITYPE(IL) == IL
    goto L375;
}
L345:                                   // INSERT CUSP
    delfx[iaf * 4 + k] = delpen[il * 6 + k];
    htflx[iaf * 4 + k] = htrue[ihigh * 6 + k];
    afflx[iaf * 4 + k] = afac[ihigh * 6 + k];
    fv = fvert[ihigh * 6 + k];
    hp = hprim[ihigh * 6 + k];
    ia = ia - 1;
    icusp = 0;
    imode[iaf * 4 + k] = il;            // _ITYPE(IL) == IL
    goto L375;
L350:                                   // F2 is the last layer?
    if (il - 3 < 0) goto L355;
    goto L400;
L355:
    if (delpen[il * 6 + k] - 89.9f < 0.0f) goto L360;
    goto L400;
L360:                                   // INSERT CUSP FOR NEXT LAYER
    delfx[iaf * 4 + k] = delfx[(iaf - 1) * 4 + k] + 0.001f;
    htflx[iaf * 4 + k] = htrue[(ihigh + 1) * 6 + k];
    afflx[iaf * 4 + k] = afac[(ihigh + 1) * 6 + k];
    fv = fvert[(ihigh + 1) * 6 + k];
    hp = hprim[(ihigh + 1) * 6 + k];
    icusp = 1;
    imode[iaf * 4 + k] = (fi[2 * 6 + k] <= 0.0f) ? 3 : il + 1;  // _ITYPE
    goto L375;
L375: {                                 // MARTYN correction -> ground dist
    float del_ = delfx[iaf * 4 + k] * VC_D2R;
    float rcosd = VC_RZ * cos32(del_);
    float xfsq = freq * freq / fc2;
    float ht = htflx[iaf * 4 + k];
    float xmut = 1.0f - fv * fv / (freq * freq);
    // __fdiv_rn: /RZ is a constant-denominator division, which NVRTC would
    // otherwise strength-reduce to a 1-ULP-wrong reciprocal multiply even
    // under --prec-div=true (see penang.cuh).
    float xhp = div32(hp - ht, VC_RZ);
    float sph = xfsq * xmut * xhp * (ht + 2.0f * (VC_RZ + ht) * xhp);
    hp = hp + sph;
    float phe = asin32(rcosd / (VC_RZ + hp));
    float gdr = 2.0f * VC_RZ * (VC_PIO2 - del_ - phe);
    gdflx[iaf * 4 + k] = gdr;
    hpflx[iaf * 4 + k] = hp;
    fvflx[iaf * 4 + k] = fv;
    // skip distance (minimum)
    if (reflx->dskpkm[k] - gdr > 0.0f) {
        reflx->dskpkm[k] = gdr;
        reflx->delskp[k] = delfx[iaf * 4 + k];
        reflx->htskp[k] = ht;
        reflx->hpskp[k] = hp;
        reflx->fvskp[k] = fv;
        reflx->iskp[k] = il;            // _ITYPE(IL) == IL
    }
    // maximum distance (only for take-off angles >= AMIND)
    if (reflx->dmaxkm[k] - gdr <= 0.0f) {
        if (delfx[iaf * 4 + k] - amind >= 0.0f)
            reflx->dmaxkm[k] = gdr;
    }
    reflx->iaftxr[k] = iaf;
    iaf = iaf + 1;
    if (iaf - 45 <= 0) goto L396;
    goto L400;
}
L396:                                   // IF(ICUSP) 275,350,265
    if (icusp < 0) goto L275;
    if (icusp == 0) goto L350;
    goto L265;
L400:                                   // done
    return;
}

#endif  // VOACAP_FINDF_CUH
