// ESMOD — sporadic-E modes, the /ZON/ slots REGMOD does not fill; the CUDA C
// twin of scalar/esmod.py.  REGMOD handles F- and E-layer modes and, where
// sporadic E gets in the way of one, the obscuration loss.  ESMOD handles the
// modes that reflect off the Es layer itself: a thin mirror at a fixed height
// rather than a refraction through a profile, so there is no deviative loss
// (ADV is 0), the virtual height IS the true height, and the "over-the-MUF"
// term becomes a reflection loss.
//
// At most TWO Es hops are considered.  Slots 4 and 5 are preset to the no-mode
// markers first and unconditionally, so a caller can read them whether or not
// an Es mode exists; all four early returns leave those presets in place.
//
// Deliberate transliterations, each of which looks like a slip and is not:
//
//   * the sample area K is the one with the LOWEST FS(2,.), not the highest —
//     the weakest link along the path bounds the whole mode.
//   * AC indexes GYZ by K, the Es sample area, where REGMOD indexes it by
//     LX(K); this routine has no LX at all.
//   * ADX, the correction for an Es layer below the E layer, is added into
//     SABPS *and then again* inside XTLOS, so it counts twice in the loss but
//     only once in the reported ABPS.
//   * when the path needs two hops the loop runs IHOP = 2,2 — one iteration
//     writing slot 4, leaving slot 5 at its presets.  Slot 4 is not "the
//     one-hop mode", it is "the first mode found".
//   * PROB uses an INTEGER exponent (PROS**IHOP), so it is gfortran's powi
//     binary exponentiation, never pow32.  The **1.98 and **-0.7 exponents are
//     default-REAL literals and do go through pow32.
//
// Where Stage A RAISES, this returns a status with /ZON/ exactly as Stage A
// leaves it at the raise; the codes match regmod.cuh so a caller can test them
// once:  0 = ok, 3 = GAIN out-of-bounds (Stage A: IndexError),
// 4 = GAIN's unported area-coverage antenna (Stage A: NotImplementedError).
// ESMOD has no STOP of its own, so codes 1 and 2 never occur here.
//
// The /cantenna/ + /pantenna/ tables are SHARED, not per-thread, and pass flat
// exactly as gain.cuh / pwrdb.cuh document.  Rule 4b: 1000, RZ and RZ+100 are
// compile-time denominators and go through div32.  No REAL*8 anywhere.
//
// float32; ALOG10/ALOG/SIN/COS/ATAN/ASIN/SQRT and the **1.98 / **-0.7 powers
// all carry the [[f32-libm-noise]] floor.

#ifndef VOACAP_ESMOD_CUH
#define VOACAP_ESMOD_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "gain.cuh"
#include "prbmuf.cuh"
#include "pwrdb.cuh"
#include "raysview.cuh"
#include "state.cuh"

// x**n with an INTEGER exponent — gfortran calls powi, which is binary
// exponentiation, not powf.  N is only ever 1 or 2 here.
__device__ inline float esmod_powi_(float x, int n) {
    float r = 1.0f;
    float b = x;
    while (n) {
        if (n & 1)
            r = r * b;
        n >>= 1;
        if (n)
            b = b * b;
    }
    return r;
}

// DO 97: the no-mode markers for both Es slots.
__device__ inline void esmod_preset_(ZonState* zon) {
    for (int iht = 4; iht <= 5; ++iht) {
        zon->obf[iht]    = 1000.0f;
        zon->adv[iht]    = 1000.0f;
        zon->fslos[iht]  = 1000.0f;
        zon->tloss[iht]  = 1000.0f;
        zon->abps[iht]   = 1000.0f;
        zon->eff[iht]    = 0.0f;
        zon->grlos[iht]  = 1000.0f;
        zon->rgain[iht]  = 0.0f;
        zon->tgain[iht]  = 0.0f;
        zon->hn[iht]     = -1.0f;
        zon->prob[iht]   = 0.001f;
        zon->crel[iht]   = -1000.0f;    // ESREG presets its slots to +1000
        zon->rely[iht]   = 0.001f;
        zon->spro[iht]   = 0.001f;
        zon->fldst[iht]  = -1000.0f;
        zon->sigpow[iht] = -1000.0f;
        zon->sn[iht]     = -1000.0f;
        zon->hp[iht]     = -1.0f;
        zon->b[iht]      = -1.0f;
        zon->nmode[iht]  = 5;
        zon->tllow[iht]  = 10.0f;
        zon->timed[iht]  = -1.0f;
        zon->tlhgh[iht]  = 10.0f;
    }
}

// Fill /ZON/ slots 4 and 5 with the sporadic-E modes.  state: LayerState
// (/RON/, /ES/, /GEOG/); muf: MufState; sigd: SigdState; anois: AnoisState
// (RCNSE only); rays: RaysView over /RAYS/ (IFOB only); zon: ZonState, the
// output.
// freq: MHz; jmode: the ionosphere SELMOD picked; gcd/gcdkm: great-circle
// distance in radians and km (/DON/); amind: the minimum usable take-off
// angle in degrees (/DON/).  Returns the status documented above.
__device__ inline int esmod(const LayerState* state, const MufState* muf,
                            const SigdState* sigd, const AnoisState* anois,
                            const RaysView& rays, const float* cant_array,
                            const float* cant_aeff, const int* cant_iats,
                            const float* cant_xfqs, const float* cant_xfqe,
                            const float* cant_offazim,
                            const float* cant_pwrdba, int numants,
                            ZonState* zon, float freq, int jmode, float gcd,
                            float gcdkm, float amind) {
    const float refc = 8.9136f;         // the reflection-loss coefficient
    const float pm07 = -0.7f;           // the reflection-loss exponent

    // DO 95: the sample area with the LOWEST Es critical frequency bounds the
    // mode — "all modes will be at least this good".
    int k = 1;
    for (int i = 1; i <= state->km; ++i) {
        if (state->fs[2 * 6 + k] - state->fs[2 * 6 + i] > 0.0f)
            k = i;
    }

    // The maps are Es or E, so they are not usable at the low end.
    float fsdead = (float)rv_ifob(rays, 1, 3, jmode);
    fsdead = div32(fsdead, 1000.0f);
    fsdead = pymin(fsdead, 3.0f);

    esmod_preset_(zon);

    if (freq <= fsdead)
        return 0;
    if (state->fs[2 * 6 + k] <= 0.0f)
        return 0;

    // virtual heights are the true heights: a thin mirror at HS
    float sdmax = 2.0f * VC_RZ
                  * (VC_PIO2 - asin32(1.0f / (1.0f
                                              + div32(state->hs[k], VC_RZ))));
    int ihsrt = (int)(gcdkm / sdmax + 1.0f);        // REAL->INT truncates
    int ihstp = 2;
    if (ihsrt > 2)                                  // only two Es hops are done
        return 0;
    int ih = 3;

    // absorption loss factor, as for an F2 mode
    float ac = 677.2f * sigd->acav
               / (pow32(freq + state->gyz[k], 1.98f) + 10.2f);

    GroundMemo gmemo;                   // ground_loss_sum's, per call
    ground_memo_init(&gmemo);
    for (int ihop = ihsrt; ihop <= ihstp; ++ihop) {         // DO 150
        ih = ih + 1;
        float gp = (float)ihop;
        float ghop = gcd / gp;
        float thet = 0.5f * ghop;
        float tans = sin32(thet)
                     / (1.0f - cos32(thet) + div32(state->hs[k], VC_RZ));
        float psi = atan32(tans);
        float secs = 1.0f / cos32(psi);
        float sfvmod = freq / secs;
        float esd = state->fs[2 * 6 + k] * secs;    // MUF at this hop distance
        float dele = VC_PIO2 - thet - psi;
        float cdel = cos32(dele);
        float adel = dele * VC_R2D;
        if (adel < amind)
            continue;

        // group path, free-space loss
        float path = 2.0f * sin32(0.5f * ghop) * (VC_RZ + state->hs[k]) / cdel;
        float hop = (float)ihop;
        path = hop * path;
        float sflos = 32.45f + 20.0f * log10_32(path * freq);

        float sinp = div32(VC_RZ * cdel, VC_RZ + 100.0f);
        float cosp = sqrt32(1.0f - sinp * sinp);
        float sabps = ac / cosp;
        float adx = 0.0f;
        // Es below E: remove the E-layer bending above HS
        if (state->fi[1 * 6 + k] > sfvmod)
            adx = sigd->afe
                  + sigd->bfe * log32(sfvmod / state->fi[1 * 6 + k]);
        sabps = sabps + adx;

        // probability of reflection, at the MUF and at both deciles
        float pros = prbmuf(muf, freq, esd, muf->ymuf[4], 4);
        pros = pymin(pros, 0.90f);                  // limit loss
        float ref = refc * pow32(pros, pm07);
        esd = state->fs[1 * 6 + k] * secs;          // lower decile: the FOT
        float ps = prbmuf(muf, freq, esd, muf->yfot[4], 4);
        ps = pymin(ps, 0.9f);
        float refl = refc * pow32(ps, pm07);
        esd = state->fs[3 * 6 + k] * secs;          // upper decile: the HPF
        ps = prbmuf(muf, freq, esd, muf->yhpf[4], 4);
        ps = pymin(ps, 0.9f);
        float refu = refc * pow32(ps, pm07);

        zon->tllow[ih] = sigd->dsl + hop * (refl - ref);
        zon->tlhgh[ih] = sigd->dsu + hop * (ref - refu);
        zon->tllow[ih] = pymin(zon->tllow[ih], 25.0f);
        zon->tlhgh[ih] = pymin(zon->tlhgh[ih], 25.0f);

        // ground loss, averaged over the sample areas
        float y = ground_loss_sum(state, dele, cdel, freq,
                                  &gmemo);                  // DO 130
        float xkm = (float)state->km;
        float sgrlos = y / xkm;

        int oob = 0;
        float stgain, steff, srgain, reff;
        antenna_gain(cant_array, cant_aeff, cant_iats, cant_xfqs, cant_xfqe,
                     cant_offazim, numants, 1, dele, freq, &stgain, &steff,
                     &oob);
        if (oob != 0)
            return (oob == 1) ? 3 : 4;      // EFF is not yet assigned
        antenna_gain(cant_array, cant_aeff, cant_iats, cant_xfqs, cant_xfqe,
                     cant_offazim, numants, 2, dele, freq, &srgain, &reff,
                     &oob);
        if (oob != 0)
            return (oob == 1) ? 3 : 4;      // nor here
        (void)steff;                        // the TX efficiency is discarded
        zon->eff[ih] = reff;

        // ADX is inside SABPS already and is added a SECOND time here
        float xtlos = sflos + hop * (sabps + ref + adx)
                      + (hop - 1.0f) * sgrlos - srgain - stgain + sigd->asm_;

        float pdb = pwrdb(cant_iats, cant_xfqs, cant_xfqe, cant_pwrdba,
                          numants, freq);
        zon->fldst[ih] = 107.2f + pdb + 20.0f * log10_32(freq) - xtlos - srgain;
        zon->sigpow[ih] = pdb - xtlos;

        // MUFday: the probability the frequency is below the Es MUF
        pros = prbmuf(muf, freq, muf->ymuf[4], muf->ymuf[4], 4);
        int iht = ih;
        zon->obf[iht] = refc * pow32(pros, pm07);   // F obscuration n/a
        zon->adv[iht] = 0.0f;                       // no deviative loss
        zon->fslos[iht] = sflos;
        zon->tloss[iht] = xtlos;
        zon->abps[iht] = ac / cosp + adx;           // ADX counted once here
        zon->grlos[iht] = sgrlos;
        zon->rgain[iht] = srgain;
        zon->tgain[iht] = stgain;
        zon->hn[iht] = hop;
        zon->sigpow[iht] = pdb - xtlos;
        zon->sn[iht] = zon->sigpow[iht] - anois->rcnse - zon->eff[iht];
        zon->hp[iht] = state->hs[k];
        zon->b[iht] = adel;
        zon->nmode[iht] = 4;
        zon->prob[iht] = esmod_powi_(pros, ihop);
        zon->timed[iht] = div32(path, VC_VOFL);
    }
    return 0;
}

#endif  // VOACAP_ESMOD_CUH
