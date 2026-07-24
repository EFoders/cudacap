// LSMOOTH — the long-path/short-path blend (VOA memo, 15 January 1991); the
// CUDA C twin of scalar/lsmooth.py.
//
// Between 7000 and 10000 km neither propagation model is right, so LUFFY runs
// its whole frequency loop TWICE -- once with IPFLAG 100 and once with 200 --
// saving each pass's /SON/ column into a (13,2) temporary, and then combines
// them here.  `save_pass` is the saving half, called from inside the
// frequency loop; `blend` is the combining half, called after the second
// pass; `smooth_` is the per-frequency interpolation `blend` delegates to.
//
// The blend is on POWER, not on decibels: DELPOW is the long path's signal
// lower decile minus the short path's, in dB, and if the long path is weaker
// the short path is taken outright with no interpolation at all.  Otherwise
//
//     SMOOTH = SSPLD + 10*log10( DISINT*(10^(DELPOW/10) - 1) + 1 )
//
// which is SSPLD at DISINT 0 and SLPLD at DISINT 1, so the ends of the band
// join the pure models continuously.  The DELPOW > 375 guard clamps 10^37.5
// to keep the exponential inside float32.
//
// STAGE A'S SHAPE, PRESERVED
//   * exactly 10000 km takes the long path with no smoothing (the outer band
//     test admits <= 10000, the smoothing test is < 10000);
//   * MDL(13) is 'M' for the whole hour but each frequency's own MDL is 'S'
//     or 'L' unless that frequency actually smoothed;
//   * only the smoothed arm re-runs the physics -- GENOIS, RELBIL and SERPRB
//     again, from the blended power.
//
// `SmoothStore` is the Fortran's "TEMPORARY STORAGE": one column per
// frequency per pass, index [ifx, idx] with idx 1 for the short-path pass and
// 2 for the long, which is what (IPFLAG+1)/100 evaluates to.  It lives in the
// thread's own local memory (LUFFY makes one per hour), so it is flattened
// row-major with a stride of VC_SMOOTH_NPASS exactly as numpy's (14, 3).
//
// Stage A's ``store.son[name][ifx, idx]`` would raise IndexError for idx
// outside 0..2, which is what a LUF-arm IPFLAG of 300 produces.  A device
// function cannot raise, so `save_pass` returns VC_SMOOTH_BAD_IDX there and
// leaves the store untouched, mirroring numpy refusing the assignment.
//
// Rule 4b: DISINT's /3000 and DELPOW's /10 are constant denominators and go
// through div32.

#ifndef VOACAP_LSMOOTH_CUH
#define VOACAP_LSMOOTH_CUH

#include "circuit.cuh"
#include "f32math.cuh"
#include "genois.cuh"
#include "pwrdb.cuh"
#include "relbil.cuh"
#include "serprb.cuh"
#include "state.cuh"

#define VC_SMOOTH_NFREQ  12
#define VC_SMOOTH_NROW   14         // NFREQ + 2, the numpy first dimension
#define VC_SMOOTH_NPASS  3          // ... and its second: idx 0 unused, 1, 2
#define VC_SMOOTH_BAD_IDX 21        // Stage A's IndexError on store[ifx, idx]

// LUFFY's "TEMPORARY STORAGE".  The sixteen _SON_SAVED columns, then the
// three that are not plain /SON/ floats, then the long pass's own extras.
struct SmoothStore {
    float angle[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float cprob[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float dblos[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float dblosl[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float dblosu[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float delay[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float dbw[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float vhigh[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float dbu[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float reliab[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float sndb[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float snpr[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float snrlw[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float snrup[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float sprob[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float probmp[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    int   nhp[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    int   smode[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];   // LAYTYP index, -1 = ""
    float tgain[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    float rgain[VC_SMOOTH_NROW * VC_SMOOTH_NPASS];
    // the long pass alone also keeps these, for the arm that selects it
    float zdu[VC_SMOOTH_NROW];
    float zdl[VC_SMOOTH_NROW];
    int   inmode[VC_SMOOTH_NROW];
    float zangler[VC_SMOOTH_NROW];
    int   zmoder[VC_SMOOTH_NROW];                    // LAYTYP index, -1 = ""
};

// Stage A's SmoothStore() -- numpy zeros, and "" for the label columns.
__device__ inline void smoothstore_init(SmoothStore* s) {
    for (int i = 0; i < VC_SMOOTH_NROW * VC_SMOOTH_NPASS; ++i) {
        s->angle[i] = 0.0f;  s->cprob[i] = 0.0f;  s->dblos[i] = 0.0f;
        s->dblosl[i] = 0.0f; s->dblosu[i] = 0.0f; s->delay[i] = 0.0f;
        s->dbw[i] = 0.0f;    s->vhigh[i] = 0.0f;  s->dbu[i] = 0.0f;
        s->reliab[i] = 0.0f; s->sndb[i] = 0.0f;   s->snpr[i] = 0.0f;
        s->snrlw[i] = 0.0f;  s->snrup[i] = 0.0f;  s->sprob[i] = 0.0f;
        s->probmp[i] = 0.0f; s->tgain[i] = 0.0f;  s->rgain[i] = 0.0f;
        s->nhp[i] = 0;
        s->smode[i] = -1;                           // ""
    }
    for (int i = 0; i < VC_SMOOTH_NROW; ++i) {
        s->zdu[i] = 0.0f; s->zdl[i] = 0.0f; s->zangler[i] = 0.0f;
        s->inmode[i] = 0;
        s->zmoder[i] = -1;                          // ""
    }
}

// Stash this frequency's answer for the pass that just ran (lines 330-369).
// Called once per frequency per pass, only under MSPEC 121.
// The part-wise form.  /allMODE/ and /ANOIS/ are PER-LANE in the
// warp-per-circuit-hour kernel (cuda/luffy_warp.cuh) while /SON/ stays shared,
// so they cannot all be reached through one CircuitState there.  `save_pass`
// below is the original call, and passes the circuit's own three.
__device__ inline int save_pass_parts(const SonState* son,
                                      const AllModeState* allm,
                                      const AnoisState* anois, SmoothStore* s,
                                      int ifx, int ipflag) {
    int idx = (ipflag + 1) / 100;               // 100 -> 1, 200 -> 2
    if (idx < 0 || idx >= VC_SMOOTH_NPASS)
        return VC_SMOOTH_BAD_IDX;               // numpy would raise IndexError
    int p = ifx * VC_SMOOTH_NPASS + idx;
    s->angle[p]  = son->angle[ifx];
    s->cprob[p]  = son->cprob[ifx];
    s->dblos[p]  = son->dblos[ifx];
    s->dblosl[p] = son->dblosl[ifx];
    s->dblosu[p] = son->dblosu[ifx];
    s->delay[p]  = son->delay[ifx];
    s->dbw[p]    = son->dbw[ifx];
    s->vhigh[p]  = son->vhigh[ifx];
    s->dbu[p]    = son->dbu[ifx];
    s->reliab[p] = son->reliab[ifx];
    s->sndb[p]   = son->sndb[ifx];
    s->snpr[p]   = son->snpr[ifx];
    s->snrlw[p]  = son->snrlw[ifx];
    s->snrup[p]  = son->snrup[ifx];
    s->sprob[p]  = son->sprob[ifx];
    s->probmp[p] = son->probmp[ifx];
    s->nhp[p]    = son->nhp[ifx];
    s->smode[p]  = son->mode[ifx];
    // NB slot 1 of /allMODE/, not slot IFX: these are the CHOSEN mode's gains.
    s->tgain[p]  = allm->tgain[1];
    s->rgain[p]  = allm->rgain[1];

    if (ipflag == 200) {
        // .....SAVE LONG PATH PARAMETERS
        s->zdu[ifx]     = anois->du;
        s->zdl[ifx]     = anois->dl;
        s->inmode[ifx]  = allm->nmode[1];
        s->zangler[ifx] = son->angler[ifx];
        s->zmoder[ifx]  = son->moder[ifx];
    }
    return 0;
}

__device__ inline int save_pass(const CircuitState* c, SmoothStore* s,
                                int ifx, int ipflag) {
    return save_pass_parts(&c->son, &c->allm, &c->anois, s, ifx, ipflag);
}

// Interpolate the two models on power and redo the physics from there.
__device__ inline int smooth_(CircuitState* c, const CircuitTables* t,
                              int ifx, float freq, float sspld, float delpow) {
    c->son.mdl[ifx] = VC_MDL_M;
    float disint = div32(c->gcdkm - 7000.0f, 3000.0f);
    float delx;
    if (delpow > 375.0f)
        delx = pow32(10.0f, 37.5f);             // ... avoid the overflow
    else
        delx = pow32(10.0f, div32(delpow, 10.0f));
    float smooth = sspld + 10.0f * log10_32(disint * (delx - 1.0f) + 1.0f);
    c->allm.sigpow[1] = smooth + c->allm.tllow[1];
    float pdb = pwrdb(t->iats, t->xfqs, t->xfqe, t->pwrdba, t->numants, freq);
    c->allm.tloss[1] = pdb - c->allm.sigpow[1];
    c->allm.sn[1] = c->allm.sigpow[1] - c->anois.rcnse - c->allm.eff[1];
    c->allm.fldst[1] = 107.2f + pdb + 20.0f * log10_32(freq)
                       - c->allm.tloss[1] - c->allm.rgain[1];
    c->allm.nmmod = 1;
    int st = genois(t->fam, t->dud, t->f2d111, &c->layers, &c->anois, &c->ton,
                    &c->zon, &c->ndiag, t->array, t->aeff, t->iats, t->xfqs,
                    t->xfqe, t->offazim, t->numants, freq, c->noise, c->rlat,
                    c->nyear_is_2089);
    if (st) return st;
    st = relbil(&c->allm, &c->anois, &c->sigd, &c->son, t->iats, t->xfqs,
                t->xfqe, t->pwrdba, t->numants, ifx, freq, c->rsn, c->lufp);
    if (st) return st;
    if (c->want_service_prob)
        serprb(&c->allm, &c->anois, &c->ton, &c->son, ifx, c->rsn, c->lufp);
    return 0;
}

// Combine the two passes into one answer per frequency (lines 410-522).
__device__ inline int blend(CircuitState* c, const CircuitTables* t,
                            const SmoothStore* s) {
    c->son.mdl[13] = VC_MDL_M;
    for (int ifx = 1; ifx <= VC_SMOOTH_NFREQ; ++ifx) {      // DO 290 IF=1,12
        c->son.mdl[ifx] = VC_MDL_BLANK;
        float freq = c->frq.frel[ifx];
        if (freq <= 0.0f)
            continue;
        c->frq.freq = freq;
        int p1 = ifx * VC_SMOOTH_NPASS + 1;
        int p2 = ifx * VC_SMOOTH_NPASS + 2;
        // The signal's lower decile, per pass.  DBLOSL's sign is not trusted:
        // the ABS was added 9/14/94 "per VOA".
        float slpld = s->dbw[p2] - fabsf(s->dblosl[p2]);
        float sspld = s->dbw[p1] - fabsf(s->dblosl[p1]);
        float delpow = slpld - sspld;

        int idx;
        if (delpow < 0.0f) {
            // .....USE SHORT PATH -- the higher signal level wins outright
            idx = 1;
            c->son.mdl[ifx] = VC_MDL_S;
            c->son.moder[ifx] = s->smode[p1];
            c->son.angler[ifx] = s->angle[p1];
            // TGAIN and RGAIN are /allMODE/ arrays indexed by MODE everywhere
            // else; here the original indexes them by FREQUENCY.  Both run to
            // 20 so it is legal, and the long arm below goes back to slot 1.
            c->allm.tgain[ifx] = s->tgain[p1];
            c->allm.rgain[ifx] = s->rgain[p1];
            // ... and these two are what actually gets printed
            c->son.gaint[ifx] = c->allm.tgain[ifx];
            c->son.gainr[ifx] = c->allm.rgain[ifx];
            c->son.snxx[ifx] = c->rsn - s->snpr[p1];
        } else {
            // .....USE LONG PATH, OR SMOOTH IF UNDER 10000 KM
            idx = 2;
            c->son.mdl[ifx] = VC_MDL_L;
            c->son.moder[ifx] = s->zmoder[ifx];
            c->son.angler[ifx] = s->zangler[ifx];
            c->anois.du = s->zdu[ifx];
            c->anois.dl = s->zdl[ifx];
            c->allm.b[1] = s->angle[p2];
            c->allm.tllow[1] = s->dblosl[p2];
            c->allm.tlhgh[1] = s->dblosu[p2];
            c->allm.timed[1] = s->delay[p2];
            c->allm.hn[1] = (float)s->nhp[p2];
            c->allm.hp[1] = s->vhigh[p2];
            c->allm.prob[1] = s->cprob[p2];
            c->allm.nmode[1] = s->inmode[ifx];
            c->allm.tgain[1] = s->tgain[p2];
            c->allm.rgain[1] = s->rgain[p2];
            c->allm.eff[1] = c->son.rneff[ifx];
        }

        // .....COPY THE WINNING PASS'S /SON/ COLUMN
        int p = ifx * VC_SMOOTH_NPASS + idx;
        c->son.mode[ifx]   = s->smode[p];
        c->son.angle[ifx]  = s->angle[p];
        c->son.cprob[ifx]  = s->cprob[p];
        c->son.dblos[ifx]  = s->dblos[p];
        c->son.dblosl[ifx] = s->dblosl[p];
        c->son.dblosu[ifx] = s->dblosu[p];
        c->son.delay[ifx]  = s->delay[p];
        c->son.dbw[ifx]    = s->dbw[p];
        c->son.vhigh[ifx]  = s->vhigh[p];
        c->son.dbu[ifx]    = s->dbu[p];
        c->son.reliab[ifx] = s->reliab[p];
        c->son.sndb[ifx]   = s->sndb[p];
        c->son.snpr[ifx]   = s->snpr[p];
        c->son.snrlw[ifx]  = s->snrlw[p];
        c->son.snrup[ifx]  = s->snrup[p];
        c->son.sprob[ifx]  = s->sprob[p];
        c->son.probmp[ifx] = s->probmp[p];
        c->son.nhp[ifx]    = s->nhp[p];
        c->anois.rcnse = c->son.xnynois[ifx];

        if (delpow >= 0.0f && c->gcdkm < 10000.0f) {
            int st = smooth_(c, t, ifx, freq, sspld, delpow);
            if (st) return st;
        }
    }
    return 0;
}

// Whether this circuit runs both models and blends them.  The band is
// INCLUSIVE at both ends here; the smoothing itself excludes 10000 km.
// (Carried for parity with scalar/lsmooth.py; LUFFY inlines the same test.)
__device__ inline bool wants_smoothing(const CircuitState* c) {
    return c->mspec == 121 && 7000.0f <= c->gcdkm && c->gcdkm <= 10000.0f;
}

#endif  // VOACAP_LSMOOTH_CUH
