// LUFFY — one hour of system performance over the whole frequency list; the
// CUDA C twin of stage_a/luffy.py.  ONE THREAD RUNS ONE WHOLE CIRCUIT-HOUR.
//
// This is the driver every Stage C leaf hangs off.  It runs once per hour and
// does three things in order: pick the controlling sample area and build its
// ionosphere; loop over the eleven frequencies; and for each one, sweep the
// plausible hop counts, collect every mode that exists, and hand the
// collection to RELBIL for a single answer.
//
//     SELMOD                        which sample area controls the path
//     ANOIS1                        noise at 1 MHz at the receiver
//     LECDEN/GENION/FOBBY/ALOSFV    electron profile -> ionogram -> reflectrix
//     SETLNG, SIGDIS                fill the other areas, set the loss spread
//     for each frequency:
//         ALLMODES(0)               clear the accumulator
//         GENOIS                    noise at this frequency
//         FINDF                     ray set for this frequency
//         for each hop count:
//             FDIST, INMUF          up to six modes, plus the over-the-MUF one
//             ALLMODES              accumulate them
//         ESMOD, ESREG              the sporadic-E slots
//         ALLMODES(2)               accumulate those
//         GENOIS, RELBIL            noise again, then the combined distribution
//         SERPRB, MPATH             service and multipath probability
//
// IPFLAG selects the model: 100 short-path system performance, 200 long-path,
// 300 short-path LUF, 400 long-path LUF.  All four run.
//
// The long path is a different shape rather than a longer version of the same
// thing.  Over 10,000 km the two ends of the circuit are in different places,
// different local times and often different hemispheres, so there is no single
// controlling ionosphere to sweep hop counts through.  Each end gets its own
// ionosphere and ray table, SETTXR prices all 45 rays at each, SELTXR picks
// one per end, and LNGPAT joins the pair into a single mode written straight
// into /allMODE/ slot 1.  No hop loop, no /ZON/ accumulation.
//
// FOUR QUIRKS OF THE ORIGINAL, PRESERVED VERBATIM
//
// **IPFG is re-read from IPFLAG at the top of every frequency iteration.**
// The entry code may have raised it to 200 to select the long-path *setup*
// for a path beyond the smoothing distance, but that assignment does not
// survive into the loop, so the per-frequency *body* reverts to whatever the
// caller asked for.  A path can therefore run the long-path area selection and
// the short-path mode calculation.
//
// **Label 300's search for the best reliability does not update its running
// maximum.**  REL is set from RELIAB(1) and never reassigned inside the loop,
// so IG ends up being the LAST frequency that beats slot 1 rather than the
// best one.  Transliterated as written; see no_luf_.
//
// **Two locals are dead**: IHMX (an upper hop bound computed from NHOPMF and
// never read) and IXHP (a hop counter incremented and never read).  Noted
// where they would go rather than transcribed.
//
// **The 7000-10000 km recycle** runs the short path, jumps back to label 83
// with IPFLAG forced to 200, runs the long path, and blends the two per
// frequency (lsmooth.cuh).  Label 83 is BELOW SELMOD and ANOIS1, which run
// once for both models.
//
// STATUS, NOT EXCEPTIONS.  Stage A raises where the Fortran misbehaves; a
// device function cannot, so every callee's non-zero status is propagated
// unchanged and the state is left exactly where Stage A leaves it at the
// raise.  LUFFY's own refusal -- /allMODE/ NMODE(2) outside LAYTYP on the
// long-path receiver-end assignment, where Stage A raises IndexError -- is
// VC_LUFFY_BAD_NMODE2.  Return 0 is success.
//
// The mspec save/restore around the body is Stage A's try/finally: MSPEC is
// zeroed past the smoothing distance and restored at label 999 whatever
// happens, including on a refusal.

#ifndef VOACAP_LUFFY_CUH
#define VOACAP_LUFFY_CUH

#include "allmodes.cuh"
#include "alosfv.cuh"
#include "anois1.cuh"
#include "circuit.cuh"
#include "esmod.cuh"
#include "esreg.cuh"
#include "f32math.cuh"
#include "fdist.cuh"
#include "findf.cuh"
#include "fobby.cuh"
#include "frqcom.cuh"
#include "genion.cuh"
#include "genois.cuh"
#include "gmloss.cuh"
#include "inmuf.cuh"
#include "lecden.cuh"
#include "lngpat.cuh"
#include "lsmooth.cuh"
#include "mpath.cuh"
#include "raysview.cuh"
#include "relbil.cuh"
#include "selmod.cuh"
#include "seltxr.cuh"
#include "serprb.cuh"
#include "setlng.cuh"
#include "sigdis.cuh"
#include "state.cuh"

#define VC_LUFFY_NFREQ  12          // DO 265 IFX=1,12 -- 11 freqs plus the MUF
#define VC_LUFFY_ILUF   13          // FREA(13)/MDL(13)/NHP(13), the summary
#define VC_LUFFY_MAX_EXTRA_HOPS 2   // IHSTP = MIN0(IHMAX, IHSRT+2)
#define VC_LUFFY_ES_FVAL 999.0f     // ALLMODES(2,FVAL): the sentinel frequency
#define VC_LUFFY_NLAYTYP 5          // len(LAYTYP)

// LUFFY's own refusal: Stage A's IndexError on /allMODE/ NMODE(2).
#define VC_LUFFY_BAD_NMODE2 22

// ---------------------------------------------------------------------------
// GENOIS, with the circuit's tables -- Stage A's _genois.
// ---------------------------------------------------------------------------
__device__ inline int luffy_genois_(CircuitState* c, const CircuitTables* t,
                                    float freq) {
    return genois(t->fam, t->dud, t->f2d111, &c->layers, &c->anois, &c->ton,
                  &c->zon, &c->ndiag, t->array, t->aeff, t->iats, t->xfqs,
                  t->xfqe, t->offazim, t->numants, freq, c->noise, c->rlat,
                  c->nyear_is_2089);
}

// ---------------------------------------------------------------------------
// Labels 83-85: choose the control area and build its ionosphere.
//
// Writes *k (the area the frequency loop will work in) and *xmdl (the
// one-letter model tag /SON/ MDL(13) carries).
//
// The short path has ONE controlling area, the one SELMOD picked.  The long
// path has TWO -- the transmitter end and the receiver end each get their own
// ionosphere, because over 10,000 km the two ends are in different places,
// different local times and often different hemispheres.  That is the whole
// reason the long-path model exists.
// ---------------------------------------------------------------------------
__device__ inline void luffy_select_area_(CircuitState* c,
                                          const RaysView& rays, int ipfg,
                                          int* k_out, int* xmdl_out) {
    int k;
    int xmdl;
    bool goto_85;
    if (ipfg == 100 || ipfg == 300) {
        // .....SET FLAG TO INDICATE SHORT PATH MODEL
        c->jlong = -1;
        xmdl = VC_MDL_S;
        k = c->frq.jmode;
        goto_85 = false;
    } else {
        // .....SET FLAG TO INDICATE LONG PATH MODEL
        c->jlong = 1;
        xmdl = VC_MDL_L;
        // .....SELECT LONG PATH TRANSMITTER AREA
        k = 1;
        c->frq.itxrcp[1] = 1;
        goto_85 = (c->mspec == 121 && c->frq.jmode == k);
    }

    while (true) {
        if (!goto_85) {
            // .....ELECTRON DENSITY PROFILE -> IONOGRAM -> REFLECTRIX ->
            // .....DEVIATION LOSS FACTOR
            lecden(&c->layers, k);
            genion(&c->layers, k);
            fobby(&c->layers, rays, k);
            alosfv(&c->layers, &c->muf, k);
            if (ipfg == 100 || k > 1) {
                *k_out = k; *xmdl_out = xmdl; return;
            }
        }
        goto_85 = false;
        // Label 85.  The long path arrives here to build its RECEIVER end;
        // short-path LUF with JMODE == 1 also falls in, and rebuilds the
        // ionosphere at the receiver index, leaving THAT one in place.
        k = c->layers.kfx;
        c->frq.itxrcp[2] = c->layers.kfx;
        if (c->mspec == 121 && c->frq.jmode == k) {
            *k_out = k; *xmdl_out = xmdl; return;
        }
        if (c->layers.kfx == 1) {
            c->frq.itxrcp[2] = 2;
            *k_out = k; *xmdl_out = xmdl; return;
        }
    }
}

// IHMIN over the three layer MUFs, skipping F1 when it carries no modes.
__device__ inline int luffy_min_hops_(const int* nhopmf) {
    int ihmin = nhopmf[1];
    if (0 < nhopmf[2] && nhopmf[2] < ihmin)
        ihmin = nhopmf[2];
    if (nhopmf[3] < ihmin)
        ihmin = nhopmf[3];
    return ihmin;
}

// ---------------------------------------------------------------------------
// The hop sweep: every mode at every plausible hop count, accumulated.
//
// *skip comes back non-zero when the frequency should be abandoned outright
// (label 122's ``go to 265``), which skips RELBIL and everything after it.
// ---------------------------------------------------------------------------
__device__ inline int luffy_short_path_(CircuitState* c,
                                        const CircuitTables* t,
                                        const RaysView& rays, float freq,
                                        int k, int ihmin, int ipfg,
                                        int* skip) {
    *skip = 0;
    // .....RAY SET TABLE
    findf(&c->layers, &c->reflx, rays, k, freq, c->amind);
    int ihsrt, ihstp;
    if (c->reflx.dmaxkm[k] <= 0.0f) {
        if (ipfg == 300) {
            *skip = 1;
            return 0;
        }
        // .....ONLY ONE OVER THE MUF MODE
        ihsrt = ihmin;
        ihstp = ihmin;
    } else {
        // .....UP TO THREE HOPS  (DMAXKM/DSKPKM are runtime denominators)
        ihsrt = (int)(c->gcdkm / c->reflx.dmaxkm[k] + 1.0f);
        int ihmax = (int)(c->gcdkm / c->reflx.dskpkm[k]);
        if (ihsrt < ihmin)
            ihsrt = ihmin;
        if (ihmax < ihsrt)
            ihmax = ihsrt;
        int cap = ihsrt + VC_LUFFY_MAX_EXTRA_HOPS;
        ihstp = (ihmax < cap) ? ihmax : cap;        // min()
        if (ihsrt > ihmin) {
            int lo = ihstp - VC_LUFFY_MAX_EXTRA_HOPS;
            ihsrt = (ihmin > lo) ? ihmin : lo;      // max()
        }
    }

    // .....HOP LOOP (IXHP counts the trips here and is never read)
    for (int ihop = ihsrt; ihop <= ihstp; ++ihop) {
        float hop = (float)ihop;
        c->modes.ghop = c->gcd / hop;               // .....HOP DISTANCE
        // .....FIND UP TO SIX MODES.  FDIST returns NUMMOD, not a status --
        // Stage A discards it here, and so does this.
        (void)fdist(&c->reflx, &c->modes, k, freq, c->amind);
        int nummod = 0;
        int st = inmuf(&c->layers, &c->modes, &c->muf, &c->sigd, &c->anois,
                   rays, t->array, t->aeff, t->iats, t->xfqs, t->xfqe,
                   t->offazim, t->pwrdba, t->numants, &c->zon, freq,
                   c->frq.jmode, c->gcd, ihop, &nummod);
        if (st) return st;
        // .....SAVE ALL MODES AVAILABLE
        float dummy = 0.0f;
        st = allmodes(&c->zon, &c->allm, ipfg, hop, &dummy);
        if (st) return st;
    }

    // .....UP TO 2 ES MODES, THEN THE (DEAD) ES-F MODE
    int st = esmod(&c->layers, &c->muf, &c->sigd, &c->anois, rays,
                   t->array, t->aeff, t->iats, t->xfqs, t->xfqe, t->offazim,
                   t->pwrdba, t->numants, &c->zon, freq, c->frq.jmode, c->gcd,
                   c->gcdkm, c->amind);
    if (st) return st;
    esreg(&c->zon);
    float dummy = 0.0f;
    return allmodes(&c->zon, &c->allm, 2, VC_LUFFY_ES_FVAL, &dummy);
}

// ---------------------------------------------------------------------------
// The two-ended model: price every ray at each end, pick one, join them.
// ---------------------------------------------------------------------------
__device__ inline int luffy_long_path_(CircuitState* c,
                                       const CircuitTables* t,
                                       const RaysView& rays, float freq) {
    // .....AREA COVERAGE, TRANSMITTER END, THEN RECEIVER END
    findf(&c->layers, &c->reflx, rays, c->frq.itxrcp[1], freq, c->amind);
    findf(&c->layers, &c->reflx, rays, c->frq.itxrcp[2], freq, c->amind);
    // .....GAIN MINUS LOSS (which runs SETTXR)
    int st = gmloss(&c->layers, &c->muf, &c->sigd, &c->reflx, &c->losx,
                    rays, t->array, t->aeff, t->iats, t->xfqs, t->xfqe,
                    t->offazim, t->numants, &c->allm, &c->zon, &c->lpath,
                    c->frq.itxrcp, freq, c->gcdkm, c->amind);
    if (st) return st;
    // .....SELECT OPTIMUM AT TRANSMITTER/RECEIVER ENDS
    st = seltxr(&c->reflx, &c->losx, &c->lpath, c->frq.itxrcp, c->gcdkm,
                c->amind, c->info);
    if (st) return st;
    // .....DO MODE CALCULATIONS
    return lngpat(&c->layers, &c->lpath, &c->losx, &c->reflx, &c->muf,
                  &c->allm, &c->sigd, &c->anois, t->iats, t->xfqs, t->xfqe,
                  t->pwrdba, t->numants, freq, c->gcd, c->frq.itxrcp[2]);
}

// ---------------------------------------------------------------------------
// One trip through DO 265.  *found comes back non-zero when the LUF has been
// found (label 165).
// ---------------------------------------------------------------------------
__device__ inline int luffy_frequency_(CircuitState* c, const CircuitTables* t,
                                       const RaysView& rays,
                                       int ifx, int ipflag, int k, int ihmin,
                                       float pluf, SmoothStore* store,
                                       int* found) {
    *found = 0;
    int ipfg = ipflag;                  // line 179; see the header comment
    float freq;
    if (ipfg < 300) {
        // .....PRESET IN COMMON/SON/
        c->son.nhp[ifx] = -1;
        c->son.sprob[ifx] = 0.0f;
        c->son.probmp[ifx] = 0.0f;
        if (c->frq.frel[ifx] <= 0.0f)
            return 0;                   // no frequency in this slot
        freq = c->frq.frel[ifx];
    } else {
        freq = c->frq.frea[ifx];
        c->son.reliab[ifx] = 0.0f;
    }

    float newfreq = 0.0f;
    int st = allmodes(&c->zon, &c->allm, 0, freq, &newfreq);  // clear
    if (st) return st;
    freq = newfreq;
    c->frq.freq = freq;
    // .....NOISE DISTRIBUTION
    st = luffy_genois_(c, t, freq);
    if (st) return st;

    if (!(ipfg == 100 || ipfg == 300)) {
        st = luffy_long_path_(c, t, rays, freq);
        if (st) return st;
    } else {
        int skip = 0;
        st = luffy_short_path_(c, t, rays, freq, k, ihmin, ipfg, &skip);
        if (st) return st;
        if (skip)
            return 0;                   // label 122 -> 265, next frequency
    }

    // .....COMBINED RELIABILITY
    st = luffy_genois_(c, t, freq);
    if (st) return st;
    st = relbil(&c->allm, &c->anois, &c->sigd, &c->son, t->iats, t->xfqs,
                t->xfqe, t->pwrdba, t->numants, ifx, freq, c->rsn, c->lufp);
    if (st) return st;

    if (ipfg < 300) {
        if (c->want_service_prob)       // IF( LINBOT(14) ) 285,285,280
            serprb(&c->allm, &c->anois, &c->ton, &c->son, ifx, c->rsn,
                   c->lufp);
        if (ipfg == 100) {
            // .....CALCULATE MULTIPATH
            mpath(&c->allm, &c->son, ifx, c->dmp, c->pmp, c->gcdkm, c->method,
                  c->mspec);
        } else {
            // .....SET RECEIVER END.  The long path has no hop loop to fill
            // /SON/ ANGLER, so it comes from the mode LNGPAT left in slot 2.
            int is_ = c->allm.nmode[2];
            if (!(1 <= is_ && is_ <= VC_LUFFY_NLAYTYP))
                return VC_LUFFY_BAD_NMODE2;     // Stage A raises IndexError
            c->son.moder[ifx] = is_ - 1;        // LAYTYP[is_ - 1]
            c->son.angler[ifx] = c->allm.b[2];
        }
        // OUTALL(IFX) is presentation only under METHOD 25 -- out of scope.
    } else {
        // .....TEST RELIABILITY
        if (c->son.reliab[ifx] >= pluf) {
            *found = 1;
            return 0;                   // label 165, and no save
        }
    }
    if (c->mspec == 121)
        return save_pass(c, store, ifx, ipflag);
    return 0;
}

// ---------------------------------------------------------------------------
// Label 165.  The first frequency to make the grade IS the LUF; below it the
// answer lies between this slot and the one under it, so interpolate.
//
// The divisor cannot vanish: every slot 1..12 is tested on this arm (the "no
// frequency here" skip belongs to the system-performance arm), so
// RELIAB(IFX-1) is strictly below PLUF and RELIAB(IFX) at or above it.
// ---------------------------------------------------------------------------
__device__ inline void luffy_luf_found_(CircuitState* c, int ifx, float pluf) {
    float* frea = c->frq.frea;
    if (ifx == 1) {                     // 1st frequency is good
        c->muf.xluf[c->time.it] = c->frq.freq;
    } else {
        // .....NO ITERATION, SELECT THE FREQUENCY COMPLEMENT FIRST
        float flow = frea[ifx - 1];
        float fhigh = frea[ifx];
        float rlow = c->son.reliab[ifx - 1];
        float rhigh = c->son.reliab[ifx];
        // .....USE LINEAR INTERPOLATION (a runtime denominator)
        c->muf.xluf[c->time.it] = flow + (fhigh - flow) * (pluf - rlow)
                                  / (rhigh - rlow);
    }
    frea[VC_LUFFY_ILUF] = c->muf.xluf[c->time.it];
}

// ---------------------------------------------------------------------------
// Label 300.  Nothing reached the required reliability, so report the best
// frequency negated -- except that the comparison never updates REL, so
// "best" means "the last one that beat slot 1".  See the header comment.
// ---------------------------------------------------------------------------
__device__ inline void luffy_no_luf_(CircuitState* c) {
    int ig = 1;
    float rel = c->son.reliab[1];       // ... and never reassigned
    for (int i = 2; i <= VC_LUFFY_NFREQ; ++i)    // DO 160 IF = 2,12
        if (c->son.reliab[i] > rel)
            ig = i;
    c->muf.xluf[c->time.it] = -c->frq.frea[ig];
    c->frq.frea[VC_LUFFY_ILUF] = c->frq.frea[ig];
}

// ---------------------------------------------------------------------------
// What happens once every frequency has been done.  *again comes back
// non-zero to say "go back to label 83 and run the other model", which only
// the first pass of the smoothing recycle does.
// ---------------------------------------------------------------------------
__device__ inline int luffy_after_loop_(CircuitState* c,
                                        const CircuitTables* t, int ipflag,
                                        const SmoothStore* store, int* again) {
    *again = 0;
    if (c->method == 25)
        return 0;
    if (ipflag >= 300) {
        luffy_no_luf_(c);               // label 300
        return 0;
    }
    // .....SMOOTHING ONLY FOR DISTANCES 7000 - 10000 KM.  MSPEC was zeroed on
    // entry past 10000 km, so that case leaves here too.
    if (c->mspec != 121 || !(7000.0f <= c->gcdkm && c->gcdkm <= 10000.0f))
        return 0;
    if (ipflag == 100) {
        *again = 1;                     // run the long path, then blend
        return 0;
    }
    return blend(c, t, store);
}

// ---------------------------------------------------------------------------
// The body: Stage A's _luffy.
// ---------------------------------------------------------------------------
__device__ inline int luffy_body_(CircuitState* c, const CircuitTables* t,
                                  const RaysView& rays, int ipflag) {
    int ipfg = ipflag;
    // LUFFY's "TEMPORARY STORAGE" -- only the smoothing recycle reads it, and
    // it has to outlive the first pass, so it is made once per hour here.
    SmoothStore store;
    smoothstore_init(&store);
    if (c->mspec == 121 && c->gcdkm > 10000.0f) {
        ipfg = 200;                     // long-path SETUP only; see the header
        c->mspec = 0;                   // beyond the smoothing distance
    }

    float pluf = 0.0f;
    if (ipfg > 200) {
        // .....FREQUENCY COMPLEMENT, and the required reliability
        int st = frqcom(&c->muf, c->frq.frea, 0, c->time.it);
        if (st) return st;
        pluf = 0.01f * (float)c->lufp;
    }

    // .....SELECT CONTROLLING SAMPLE AREA
    c->frq.jmode = selmod(&c->layers);
    // .....GET NOISE VALUE AT 1 MHZ FOR RCVR SITE
    anois1(&c->anois, t->fakp, t->fakmap, t->hmym, t->fakabp, t->abmap,
           t->f2d111, c->time.gmtr, c->rlong, c->rlongd, c->rlat);

    // Label 83.  The smoothing recycle comes back here, NOT to the top:
    // SELMOD and ANOIS1 are above the label and run once for both models.
    while (true) {
        int k, xmdl;
        luffy_select_area_(c, rays, ipfg, &k, &xmdl);  // labels 83, 84, 85
        // .....CAN RUN FOR SHORT PATHS ALSO, SO SET ALL SAMPLE AREAS
        setlng(&c->layers, rays);
        // .....SET ABSORPTION LOSS PARAMETERS, ADJUST SIGNAL DISTRIBUTION
        // .....TABLES
        sigdis(&c->layers, &c->muf, t->sys, t->perr, t->f2d111, &c->sigd,
               &c->ton, c->gcdkm, c->frq.jmode);

        int ihmin = luffy_min_hops_(c->muf.nhopmf);  // IHMX here, and dead

        // .....START FREQUENCY LOOP
        c->son.nhp[VC_LUFFY_ILUF] = -1;
        c->son.mdl[VC_LUFFY_ILUF] = xmdl;   // XMDL, chosen with K at label 83
        for (int ifx = 1; ifx <= VC_LUFFY_NFREQ; ++ifx)
            c->son.mdl[ifx] = VC_MDL_BLANK;

        for (int ifx = 1; ifx <= VC_LUFFY_NFREQ; ++ifx) {   // DO 265 IFX=1,12
            int found = 0;
            int st = luffy_frequency_(c, t, rays, ifx, ipflag, k, ihmin, pluf,
                                      &store, &found);
            if (st) return st;
            if (found) {
                luffy_luf_found_(c, ifx, pluf);          // label 165
                return 0;
            }
        }
        // .....END OF FREQUENCY LOOP
        int again = 0;
        int st = luffy_after_loop_(c, t, ipflag, &store, &again);
        if (st) return st;
        if (!again)
            return 0;
        // .....DO LONG AND SHORT PATH MODEL: round two, the other model.
        ipflag = 200;
        ipfg = 200;
    }
}

// ---------------------------------------------------------------------------
// LUFFY.  Run one hour over the frequency complement, mutating *c throughout.
//
// The shared tables pass as the flat device pointers each leaf header
// documents; iono_idx selects which circuit's REDMAP maps to use and ant_idx
// which circuit's antenna patterns, exactly as the per-thread state selects
// itself.  numants is the /cantenna/ extent shared by the whole launch
// (padding entries carry IATS 0, which GAIN's DO-10 scan never matches).
//
// COMMON /RAYS/ does NOT ride inside CircuitState: it is 30,672 of the 55,652
// bytes and its warp-strided reads dominated the kernel's memory cost, so it
// arrives as three STRUCTURE-OF-ARRAYS buffers laid out for `rays_n` threads,
// this one at lane `rays_t`.  Everything else about it is unchanged.
//
// ipflag: 100 short-path system performance, 200 long-path, 300 short-path
// LUF, 400 long-path LUF.  /SON/ is the result -- one column per frequency --
// plus /MUFS/ XLUF and /FRQ/ FREA(13) on the LUF arm.
// ---------------------------------------------------------------------------
__device__ inline int luffy(CircuitState* c,
                            int* rays_nang, float* rays_ang, int* rays_ifob,
                            int rays_n, int rays_t,
                            const float* fam, const float* dud,
                            const float* fakp, const float* fakmap,
                            const float* hmym, const float* fakabp,
                            const float* abmap, const float* sys,
                            const float* perr, const float* f2d111,
                            const float* ant_array, const float* ant_aeff,
                            const int* ant_iats, const float* ant_xfqs,
                            const float* ant_xfqe, const float* ant_offazim,
                            const float* ant_pwrdba, int numants,
                            int iono_idx, int ant_idx, int ipflag) {
    int na = numants + 1;
    CircuitTables t;
    t.fam     = fam     + (long)iono_idx * VC_FAM_STRIDE;
    t.dud     = dud     + (long)iono_idx * VC_DUD_STRIDE;
    t.fakp    = fakp    + (long)iono_idx * VC_FAKP_STRIDE;
    t.fakmap  = fakmap  + (long)iono_idx * VC_FAKMAP_STRIDE;
    t.hmym    = hmym    + (long)iono_idx * VC_HMYM_STRIDE;
    t.fakabp  = fakabp  + (long)iono_idx * VC_FAKABP_STRIDE;
    t.abmap   = abmap   + (long)iono_idx * VC_ABMAP_STRIDE;
    t.sys     = sys     + (long)iono_idx * VC_SYS_STRIDE;
    t.perr    = perr    + (long)iono_idx * VC_PERR_STRIDE;
    t.f2d111  = f2d111[iono_idx];
    t.array   = ant_array   + (long)ant_idx * VC_ANT_NFREQ * VC_ANT_NELEV * na;
    t.aeff    = ant_aeff    + (long)ant_idx * VC_ANT_NFREQ * na;
    t.iats    = ant_iats    + (long)ant_idx * na;
    t.xfqs    = ant_xfqs    + (long)ant_idx * na;
    t.xfqe    = ant_xfqe    + (long)ant_idx * na;
    t.offazim = ant_offazim + (long)ant_idx * na;
    t.pwrdba  = ant_pwrdba  + (long)ant_idx * na;
    t.numants = numants;

    RaysView rays = rays_view_soa(rays_nang, rays_ang, rays_ifob,
                                  rays_n, rays_t);

    int mspec_temp = c->mspec;              // restored at label 999
    int st = luffy_body_(c, &t, rays, ipflag);
    c->mspec = mspec_temp;                  // Stage A's try/finally
    return st;
}

#endif  // VOACAP_LUFFY_CUH
