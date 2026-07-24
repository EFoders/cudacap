// SERPRB — service probability: reliability with the prediction error folded
// in; the CUDA C twin of stage_a/serprb.py.  Rebuilds RELBIL's normal test
// per mode off /allMODE/ but widens the spread by everything uncertain about
// the PREDICTION (system-loss error ADS/SUS/SLS, noise error SIGM/SYGU/SYGL,
// and a flat 2 dB for the required S/N), then takes the maximum over modes.
//
// Preserved exactly as Stage A found them: /TON/ SPR is zeroed and never
// touched again, and D90S is set equal to D10S (not an upper decile of its
// own).  The Fortran with NMMOD <= 0 reads uninitialised D10SA/D50SA locals;
// Stage A refuses, and the test never drives this device path with NMMOD <= 0.
//
// DCL (1.28) is a compile-time constant, so every "/ DCL" goes through div32
// (CONVENTIONS.md rule 4b); the per-mode "/ D10SA[im]" denominator is a
// genuine runtime value and uses plain '/'.  SQRT and FNORML's polynomial
// carry the documented [[f32-libm-noise]] floor.

#ifndef VOACAP_SERPRB_CUH
#define VOACAP_SERPRB_CUH

#include "f32math.cuh"
#include "constants.cuh"
#include "state.cuh"
#include "fnorml.cuh"

// The standard-normal table (RELBIL's TME); SERPRB uses TME(ITM) directly.
__device__ __constant__ float SERPRB_TME[10] = {
    0.0f, 0.1257f, 0.2533f, 0.3853f, 0.5244f,
    0.6745f, 0.8416f, 1.0364f, 1.2815f, 1.6449f};

// allm: AllModeState (/allMODE/, SPRO out); anois: AnoisState; ton: TonState
// (ADS/SUS/SLS in, SPR out); son: SonState (SPROB + the /DON/ D10S/D50S/D90S
// triple out).  ifx: frequency index; rsn: required S/N; lufp: required
// reliability, percent.  Assumes allm->nmmod >= 1 (Stage A raises otherwise).
// D10S/D50S/D90S are /SON/ SCALARS, not per-frequency columns, so the
// warp-per-circuit-hour kernel (cuda/luffy_warp.cuh) cannot let twelve lanes
// write them concurrently.  This variant takes them as out-pointers; `serprb`
// below is the original, aimed straight at /SON/.  Nothing else differs.
__device__ inline void serprb_ex(AllModeState* allm, const AnoisState* anois,
                                 TonState* ton, SonState* son, int ifx,
                                 float rsn, int lufp, float* d10s_out,
                                 float* d50s_out, float* d90s_out) {
    ton->spr = 0.0f;
    int itm = abs(lufp - 50) / 5 + 1;
    itm = (itm < 10) ? itm : 10;                        // min(itm, 10)
    float tmx = SERPRB_TME[itm - 1];

    // d10sa/d50sa: local per-mode scratch, slots 0..nmmod, zeroed (dead modes
    // leave them 0, exactly as the Python list comprehension does).
    float d10sa[21];
    float d50sa[21];
    for (int i = 0; i <= allm->nmmod; ++i) {
        d10sa[i] = 0.0f;
        d50sa[i] = 0.0f;
    }

    for (int im = 1; im <= allm->nmmod; ++im) {         // DO 145
        if (allm->hp[im] <= 70.0f) {                    // no mode
            allm->spro[im] = 0.001f;
            continue;                                   // d10sa/d50sa stay 0
        }
        float dn, ds, xlh, dso, dno;
        if (lufp >= 50) {
            dn = div32(tmx * anois->du, VC_DCL);        // noise
            ds = div32(tmx * allm->tllow[im], VC_DCL);  // signal
            xlh = -1.0f;
            dso = tmx * ton->sus;                       // prediction errors
            dno = div32(tmx * anois->sygu, VC_DCL);
        } else {
            dn = div32(tmx * anois->dl, VC_DCL);
            ds = div32(tmx * allm->tlhgh[im], VC_DCL);
            xlh = 1.0f;
            dso = tmx * ton->sls;
            dno = div32(tmx * anois->sygl, VC_DCL);
        }
        d50sa[im] = sqrt32(dn * dn + ds * ds);
        d10sa[im] = d50sa[im] + sqrt32(
            anois->sigm * anois->sigm + ton->ads * ton->ads
            + dno * dno + dso * dso + 2.0f * 2.0f);     // _DR * _DR
        d50sa[im] = allm->sn[im] + xlh * d50sa[im];
        float z = (rsn - d50sa[im]) / d10sa[im];        // runtime denom
        allm->spro[im] = 1.0f - fnorml(z);
    }

    int imax = 1;                                       // DO 150
    for (int i = 1; i <= allm->nmmod; ++i) {
        if (allm->spro[i] > allm->spro[imax])
            imax = i;
    }
    son->sprob[ifx] = allm->spro[imax];
    *d10s_out = d10sa[imax];
    *d50s_out = d50sa[imax];
    *d90s_out = *d10s_out;                              // not a separate decile
}

__device__ inline void serprb(AllModeState* allm, const AnoisState* anois,
                              TonState* ton, SonState* son, int ifx,
                              float rsn, int lufp) {
    serprb_ex(allm, anois, ton, son, ifx, rsn, lufp, &son->d10s, &son->d50s,
              &son->d90s);
}

#endif  // VOACAP_SERPRB_CUH
