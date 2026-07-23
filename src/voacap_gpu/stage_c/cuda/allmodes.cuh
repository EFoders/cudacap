// ALLMODES — accumulate /ZON/ modes into /allMODE/ across the hop sweep;
// the CUDA C twin of stage_a/allmodes.py.  Pure bookkeeping: no arithmetic
// at all, only assignment and IFLG-driven index selection.
//
//   IFLG 0      reset: clear the 20 slots to the no-mode markers, FREQ = FVAL
//   IFLG 1      slot 1 only, plus the staged NMMOD+2 marker slot
//   IFLG 2      slots 4..5, the sporadic-E pair
//   otherwise   slots 1..6, the regular modes
//
// The IFLG=1 arm's oddities are preserved verbatim: HP(2)=HP(1) is written
// BACK into /ZON/, TLOSS(2) gets the 32000 sentinel, and NMODE/B land in
// slot NMMOD+2 rather than at the cursor.
//
// Stage A raises AllModesOverflow where the Fortran would write past slot
// 20; here the same condition returns 1 with the state exactly as Stage A
// leaves it at the raise (NMMOD already advanced, nothing else written).
// Return 0 is success.  *freq is written ONLY on the IFLG=0 reset call,
// mirroring Stage A returning the new FREQ there and None otherwise.

#ifndef VOACAP_ALLMODES_CUH
#define VOACAP_ALLMODES_CUH

#include "state.cuh"

#define VOACAP_ALLMODES_NSLOT 20

__device__ inline int allmodes(ZonState* zon, AllModeState* allm, int iflg,
                               float fval, float* freq) {
    if (iflg == 0) {
        allm->nmmod = 0;
        for (int i = 1; i <= VOACAP_ALLMODES_NSLOT; ++i) {
            allm->tloss[i] = 99999.0f;
            allm->tllow[i] = 999.0f;
            allm->tlhgh[i] = 999.0f;
            allm->hp[i] = -1.0f;
        }
        *freq = fval;                       // FREQ = FVAL
        return 0;
    }

    int ist, lst;
    if (iflg == 1) {
        ist = 1; lst = 1;
        zon->hp[2] = zon->hp[1];
        allm->tloss[2] = 32000.0f;
        // NB slot NMMOD+2, not the cursor: next-but-one is reserved
        if (allm->nmmod + 2 > VOACAP_ALLMODES_NSLOT)
            return 1;                       // AllModesOverflow
        allm->nmode[allm->nmmod + 2] = zon->nmode[2];
        allm->b[allm->nmmod + 2] = zon->b[2];
    } else if (iflg == 2) {
        ist = 4; lst = 5;                   // the sporadic-E pair
    } else {
        ist = 1; lst = 6;                   // the regular modes
    }

    for (int i = ist; i <= lst; ++i) {
        if (zon->hp[i] > 0.0f) {
            allm->nmmod += 1;
            if (allm->nmmod > VOACAP_ALLMODES_NSLOT)
                return 1;                   // AllModesOverflow
            int n = allm->nmmod;
            // allmodestate.FIELDS, in order
            allm->abps[n]   = zon->abps[i];
            allm->crel[n]   = zon->crel[i];
            allm->fldst[n]  = zon->fldst[i];
            allm->hn[n]     = zon->hn[i];
            allm->hp[n]     = zon->hp[i];
            allm->prob[n]   = zon->prob[i];
            allm->rely[n]   = zon->rely[i];
            allm->rgain[n]  = zon->rgain[i];
            allm->sigpow[n] = zon->sigpow[i];
            allm->sn[n]     = zon->sn[i];
            allm->spro[n]   = zon->spro[i];
            allm->tgain[n]  = zon->tgain[i];
            allm->timed[n]  = zon->timed[i];
            allm->tloss[n]  = zon->tloss[i];
            allm->b[n]      = zon->b[i];
            allm->fslos[n]  = zon->fslos[i];
            allm->grlos[n]  = zon->grlos[i];
            allm->adv[n]    = zon->adv[i];
            allm->obf[n]    = zon->obf[i];
            allm->tllow[n]  = zon->tllow[i];
            allm->tlhgh[n]  = zon->tlhgh[i];
            allm->eff[n]    = zon->eff[i];
            allm->nmode[n]  = zon->nmode[i];
        }
    }
    return 0;
}

#endif  // VOACAP_ALLMODES_CUH
