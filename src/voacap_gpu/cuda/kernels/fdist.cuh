// FDIST — find up to six modes (raysets) for the current hop distance.  The
// CUDA C twin of scalar/fdist.py (mirrors fdist.for).
//
// For the hop distance DHOPKM = GHOP*RZ, FDIST walks the ReflxState ray
// table looking for the low- and high-angle rays of each layer whose ground
// distance brackets the hop.  Where the table brackets the hop distance it
// interpolates the virtual/true height and deviative loss, then re-derives
// the take-off angle from the exact geometry and back-solves the vertical
// frequency by Snell's law.  Modes whose take-off angle falls below AMIND
// are rejected by negating HPMOD.  Returns NUMMOD, the count of usable
// modes.
//
// Like FINDF this is a dense arithmetic-IF / GO TO machine; Stage A
// dispatched on the Fortran statement labels, and this port transliterates
// the same web as C labels L140..L230.  REAL*4.

#ifndef VOACAP_FDIST_CUH
#define VOACAP_FDIST_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "state.cuh"

// reflx: ReflxState (the ray table FINDF filled); modes: ModesState (out,
// reads GHOP); k: area; freq: MHz; amind: min take-off angle (deg).
__device__ inline int fdist(const ReflxState* reflx, ModesState* modes,
                            int k, float freq, float amind) {
    const float* delfx = reflx->delfx;
    const float* hpflx = reflx->hpflx;
    const float* htflx = reflx->htflx;
    const float* gdflx = reflx->gdflx;
    const float* fvflx = reflx->fvflx;
    const float* afflx = reflx->afflx;
    const int* imode = reflx->imode;
    float* delmod = modes->delmod;
    float* hpmod = modes->hpmod;
    float* htmod = modes->htmod;
    float* fvmod = modes->fvmod;
    float* afmod = modes->afmod;
    int* itmod = modes->itmod;

    float dhopkm = modes->ghop * VC_RZ;
    for (int im = 1; im < 7; ++im) {    // DO 100 -- "no mode"
        hpmod[im * 4 + k] = -1.0f;
        itmod[im * 4 + k] = 5;
    }

    int ih = 0, il = 0;
    float dth2 = 0.0f;
    float d;
    int guard = 0;  // translation-bug tripwire; never hit in practice.
                    // Every cycle in the label web passes L145, so a runaway
                    // exits to L230 instead of hanging the device (and the
                    // state mismatch fails the bit-exact compare loudly).
    if (dhopkm >= reflx->dmaxkm[k]) goto L230;
    goto L140;

L140:
    il = il + 1;
    // fall through to 145
L145:
    if (++guard > 200000) goto L230;
    ih = ih + 1;
    if (ih > 44 || hpflx[(ih + 1) * 4 + k] <= 0.0f) goto L230;
    d = gdflx[ih * 4 + k] - gdflx[(ih + 1) * 4 + k];
    if (d < 0.0f) goto L170;
    if (d == 0.0f) goto L160;
    goto L215;
L160:
    if (fabsf(dhopkm - gdflx[ih * 4 + k]) <= 0.001f) goto L145;
    goto L165;
L165:                                   // take the table entry as-is
    delmod[il * 4 + k] = delfx[ih * 4 + k];
    hpmod[il * 4 + k] = hpflx[ih * 4 + k];
    htmod[il * 4 + k] = htflx[ih * 4 + k];
    itmod[il * 4 + k] = imode[ih * 4 + k];
    afmod[il * 4 + k] = afflx[ih * 4 + k];
    fvmod[il * 4 + k] = fvflx[ih * 4 + k];
    goto L305;
L170:
    d = gdflx[ih * 4 + k] - dhopkm;
    if (d < 0.0f) goto L175;
    if (d == 0.0f) goto L165;
    goto L145;
L175:
    d = dhopkm - gdflx[(ih + 1) * 4 + k];
    if (d < 0.0f) goto L180;
    if (d == 0.0f) goto L176;
    goto L145;
L176:
    ih = ih + 1;
    goto L165;
L180:
    dth2 = gdflx[(ih + 1) * 4 + k] - gdflx[ih * 4 + k];
    if (fabsf(dth2) - 0.001f <= 0.0f) goto L165;
    goto L185;
L185: {                                 // linear interpolation in the table
    // __fdiv_rn: /RZ is a constant-denominator division, which NVRTC would
    // otherwise strength-reduce to a 1-ULP-wrong reciprocal multiply even
    // under --prec-div=true (see penang.cuh).
    float dth = dhopkm - gdflx[ih * 4 + k];
    float thet = div32(0.5f * dhopkm, VC_RZ);
    float hp1 = hpflx[ih * 4 + k], hp2 = hpflx[(ih + 1) * 4 + k];
    float ht1 = htflx[ih * 4 + k], ht2 = htflx[(ih + 1) * 4 + k];
    float hp = hp1 + (hp2 - hp1) * dth / dth2;
    float ht = ht1 + (ht2 - ht1) * dth / dth2;
    afmod[il * 4 + k] = afflx[ih * 4 + k]
        + (afflx[(ih + 1) * 4 + k] - afflx[ih * 4 + k]) * dth / dth2;
    hpmod[il * 4 + k] = hp;
    float st = sin32(thet);
    float tanp = st / (1.0f - cos32(thet) + div32(hp, VC_RZ));
    float phe = atan32(tanp);
    // force correct geometry: radiation angle, then FV by Snell's law
    delmod[il * 4 + k] = (VC_PIO2 - phe - thet) * VC_R2D;
    float sphi = VC_RZ * cos32(delmod[il * 4 + k] * VC_D2R) / (VC_RZ + ht);
    sphi = 1.0f - sphi * sphi;
    sphi = pymax(sphi, 0.000001f);
    fvmod[il * 4 + k] = freq * sqrt32(sphi);
    htmod[il * 4 + k] = ht;
    itmod[il * 4 + k] = imode[ih * 4 + k];
    goto L305;
}
L305:                                   // minimum-angle check
    if (delmod[il * 4 + k] >= amind) goto L315;
    hpmod[il * 4 + k] = -fabsf(hpmod[il * 4 + k]);
    goto L145;
L315:
    if (il - 6 < 0) goto L140;
    goto L230;
L215:
    d = gdflx[ih * 4 + k] - dhopkm;
    if (d < 0.0f) goto L145;
    if (d == 0.0f) goto L165;
    goto L220;
L220:
    d = dhopkm - gdflx[(ih + 1) * 4 + k];
    if (d < 0.0f) goto L145;
    if (d == 0.0f) goto L176;
    goto L180;

L230: {
    int nummod = 0;
    for (int ilm = 1; ilm < 7; ++ilm) {  // DO 236
        if (hpmod[ilm * 4 + k] < 0.0f)
            continue;
        nummod = nummod + 1;
    }
    return nummod;
}
}

#endif  // VOACAP_FDIST_CUH
