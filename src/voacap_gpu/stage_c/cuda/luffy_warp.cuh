// LUFFY, ONE WARP PER CIRCUIT-HOUR, ONE LANE PER FREQUENCY.
//
// `luffy.cuh` runs a whole circuit-hour on ONE thread.  Measured at 32,256
// threads that costs 0.395 us per (circuit, hour, frequency), and the kernel
// is not bandwidth-bound: it is latency-bound on a long serial per-thread
// dependency chain and it loses to fine-grained warp divergence (giving every
// warp 32 copies of the SAME circuit-hour, so all 32 lanes take identical
// branches, runs at 0.235 us/eval -- 1.68x).  This header spends both of
// those: one warp takes one circuit-hour, lane L takes frequency L+1, so the
// twelve DO-265 trips run side by side over ONE ionosphere and ONE reflectrix
// and the serial chain shortens by up to twelve.
//
// The setup phase (SELMOD, ANOIS1, LECDEN/GENION/FOBBY/ALOSFV, SETLNG,
// SIGDIS) does not parallelise and runs on lane 0.  MEASURED, by running the
// kernel with the DO-265 trip count as a runtime parameter so the register
// allocation is held identical: setup is 11.2% of the kernel at 8,064
// circuit-hours and 12.1% at 32,256.  With twelve lanes on the other 88% the
// Amdahl ceiling is 5.4x / 5.2x.
//
// WHAT IT IS WORTH, AND WHERE.  A/B against `luffy_batch` on the same staged
// batch, kernel only, 4 warps/block (tools/bench_stage_c.py --warp):
//
//     circuit-hours     504    2,016    4,032    8,064   32,256
//     thread  us/eval  12.87    3.52     1.89     1.01     0.40
//     warp    us/eval   2.22    1.34     1.16     1.04     0.98
//     speedup          5.81x   2.63x    1.63x    0.97x    0.42x
//
// It essentially HITS THE CEILING (5.8x against a 5.4x prediction) while the
// GPU is empty, and loses once the GPU is full.  Both halves of that are the
// same fact: `luffy_batch` is latency-bound, and the way it hides latency is
// by having thousands of INDEPENDENT circuit-hours in flight.  Below ~4,000
// circuit-hours there are not enough to fill the machine and shortening one
// circuit-hour's chain is the only lever there is; above ~8,000 there are,
// and this kernel gives up 20 of every 32 lanes and needs 254 registers
// against 168 (so 258 resident threads/SM against 390) to do it.  The
// crossover sits at about 8,000 circuit-hours -- a 21-deck, 24-hour run of
// 504 is well inside the region where it wins by 5.8x, a continental grid is
// well outside it.  Neither kernel replaces the other; pick by batch size.
//
// WHAT MAKES THIS EXACT, WHICH IS THE WHOLE PROBLEM
// -------------------------------------------------
// The twelve frequencies are not independent.  Three separate things cross
// between them, and each is handled exactly -- no approximation anywhere.
//
// 1. THE /DON/ TRIPLE.  RELBIL writes D10R/D50R/D90R only inside two
//    conditional blocks but READS them unconditionally at the end (SNRLW,
//    SNRUP, SNPR, SNXX).  A frequency whose modes are all dead therefore
//    answers from the PREVIOUS frequency's triple.  `relbil` is split
//    (relbil.cuh) into `relbil_head`, which reports whether it assigned the
//    triple, and `relbil_tail`, which consumes it; between them this file
//    runs a WARP SCAN so lane L receives the triple from the highest lane
//    j <= L that wrote one, or the pre-loop value if none did.
//
// 2. FINDF'S SKIP GROUP.  FINDF is a goto machine and several of its exits
//    never reach the assignments to IAFTXR / ISKP / DELSKP / HPSKP / HTSKP /
//    FVSKP, so GMLOSS, SELTXR and FDIST read the PREVIOUS frequency's ray
//    count.  Same treatment: a warp scan right after FINDF, resolving each
//    element from the highest lane at or below that wrote one.  (Found the
//    hard way -- without it, 217 of the 504 replay circuit-hours disagree in
//    /LPATH/ TXRGML.)
//
// 3. EVERYTHING ELSE THE LOOP LEAVES BEHIND.  /allMODE/, /ZON/, /REFLX/,
//    /MODES/, /LOSX/ and TXRGML are written only up to a per-frequency extent
//    (NMMOD modes, IAFTXR rays), so the state the loop leaves is a MERGE of
//    twelve frequencies' writes, cell by cell -- not the last frequency's
//    state.  Committing whole structs from "the last lane that got that far"
//    is wrong, and so is committing the cells that moved off the entry value.
//    What IS right: seed those structs with POISON, because none of their
//    loop-carried cells is ever read (ALLMODES(0) clears /allMODE/, the
//    presets clear /ZON/, FINDF rebuilds the reflectrix, FDIST rebuilds
//    /MODES/, GMLOSS rebuilds /LOSX/), so a cell still holding poison at the
//    end is EXACTLY a cell that lane did not write.  The merge then applies
//    the lanes' writes in lane order, cell by cell.  The poison is
//    0xFFBADBAD: a NaN as a float and a negative number as an int, so a read
//    of one would blow up loudly rather than pass for data.
//
//    The structs that DO carry real values in -- /MUFS/, /SIGD/, /ANOIS/,
//    /TON/, /c_noise/ and /LPATH/'s blkdat constants -- cannot be poisoned
//    and do not need to be: every lane writes the same cells of them, so they
//    commit whole, from the last lane that reached the point at which the
//    struct is first written.
//
// The design was proved in Python against stage_a before this file existed:
// twelve independent copies, this merge, this scan -- 504 of 504 replay
// circuit-hours bit-identical on every field of the whole Circuit.
//
// THE LUF ARM AND THE ABORT.  The LUF arm (IPFLAG >= 300) stops at the FIRST
// frequency reaching the required reliability, and any callee's non-zero
// status stops the loop where Stage A raises.  Both are resolved with
// __ballot_sync after the fact: the CUT is the lowest lane that stopped the
// loop, lanes above it roll back the /SON/ column and the smoothing-store row
// they wrote optimistically, and no lane above it is allowed into the merge.
//
// WHAT STAYS SERIAL.  Label 300 (_no_luf), label 165 (_luf_found) and the
// 7000-10000 km blend all run after the loop on lane 0 -- `blend` is itself a
// twelve-frequency loop with a genuine carry through /ANOIS/ and /allMODE/,
// and it is a few percent of the work.  The setup phase likewise.

#ifndef VOACAP_LUFFY_WARP_CUH
#define VOACAP_LUFFY_WARP_CUH

#include "luffy.cuh"

#define VC_WARP_FULL   0xffffffffu
#define VC_WARP_POISON 0xFFBADBADu      // NaN as a float, negative as an int

// The reach levels: the point in one DO-265 trip at which each of the
// real-valued structs is first written.  A struct commits from the highest
// lane at or below the cut whose reach reaches its level.
#define VC_WR_ALLM   1                  // ALLMODES(0), and /FRQ/ FREQ
#define VC_WR_NOISE  2                  // GENOIS -> /ANOIS/ /TON/ /c_noise/
#define VC_WR_REFLX  3                  // FINDF
#define VC_WR_MODEL  4                  // the hop loop / the long path
#define VC_WR_RELBIL 6
#define VC_WR_SERPRB 7

// ---------------------------------------------------------------------------
// One lane's private state.
//
// The first five members are the POISONED scratch and are merged cell by
// cell; `lpath` straddles (TXRGML is scratch, the four scalars and LTXRGM are
// blkdat constants read by SELTXR); the rest carry real values in and commit
// whole.  Order matters: the poison fill and the merge both walk
// [reflx .. losx] as one flat run of words.
// ---------------------------------------------------------------------------
struct LaneState {
    ReflxState     reflx;               // poisoned, bar the skip group below
    ModesState     modes;               // poisoned
    ZonState       zon;                 // poisoned
    AllModeState   allm;                // poisoned
    LosxState      losx;                // poisoned
    LpathState     lpath;               // TXRGML poisoned, the rest carried
    MufState       muf;
    SigdState      sigd;
    AnoisState     anois;
    TonState       ton;
    NoiseDiagState ndiag;
    float freq;                         // /FRQ/ FREQ
    float d10s, d50s, d90s;             // the /DON/ service triple (SERPRB)
};

#define VC_WR_TXRGML_WORDS  ((int)(sizeof(((LpathState*)0)->txrgml) / 4))

// ---------------------------------------------------------------------------
// Warp primitives
// ---------------------------------------------------------------------------

// The bits at or below this lane.
__device__ __forceinline__ unsigned wr_below_(unsigned m, int lane) {
    return m & (VC_WARP_FULL >> (31 - lane));
}

// The value from the highest lane j <= lane that HAS one, else `fallback`
// (which must be warp-uniform).  Carried as a bit pattern so one routine
// serves floats and ints alike.
__device__ __forceinline__ int wr_scan_(int v, int have, int fallback,
                                        int lane) {
    unsigned m = __ballot_sync(VC_WARP_FULL, have);
    unsigned b = wr_below_(m, lane);
    int src = b ? (31 - __clz((int)b)) : -1;
    int got = __shfl_sync(VC_WARP_FULL, v, src < 0 ? 0 : src);
    return (src < 0) ? fallback : got;
}

// The highest lane satisfying `p`, or -1.
__device__ __forceinline__ int wr_last_(int p) {
    unsigned m = __ballot_sync(VC_WARP_FULL, p);
    return m ? (31 - __clz((int)m)) : -1;
}

// The whole ordered per-cell merge: `n` words of each lane's private copy,
// applied in lane order, poison meaning "this lane did not write it".
// Written 32 words at a time so the global stores coalesce.
__device__ inline void wr_merge_(int* dst, const int* src, int n, int keep,
                                 int lane) {
    for (int base = 0; base < n; base += 32) {
        int myval = 0;
        int myhave = 0;
        for (int j = 0; j < 32; ++j) {
            int idx = base + j;
            int v = (idx < n) ? src[idx] : 0;
            int wrote = keep && (idx < n)
                        && ((unsigned)v != VC_WARP_POISON);
            unsigned m = __ballot_sync(VC_WARP_FULL, wrote);
            int hi = m ? (31 - __clz((int)m)) : 0;
            int val = __shfl_sync(VC_WARP_FULL, v, hi);
            if (lane == j) { myval = val; myhave = (m != 0); }
        }
        int idx = base + lane;
        if (idx < n && myhave)
            dst[idx] = myval;
    }
}

__device__ inline void wr_fill_(int* p, int n, int v) {
    for (int i = 0; i < n; ++i)
        p[i] = v;
}

// ---------------------------------------------------------------------------
// The /SON/ column this lane owns, snapshotted so it can be rolled back if
// the lane turns out to be past the cut.
// ---------------------------------------------------------------------------
struct SonCol {
    float angle, angler, cprob, dblos, dblosl, dblosu, dbu, delay, dbw;
    int nhp;
    float xnynois, probmp, reliab, sndb, snpr, snrlw, snrup, sprob, vhigh;
    float rneff;
    int mdl, mode, moder;
    float du_nois, dl_nois, snxx, gaint, gainr;
};

#define VC_SONCOL(OP)                                                         \
    OP(angle) OP(angler) OP(cprob) OP(dblos) OP(dblosl) OP(dblosu) OP(dbu)    \
    OP(delay) OP(dbw) OP(nhp) OP(xnynois) OP(probmp) OP(reliab) OP(sndb)      \
    OP(snpr) OP(snrlw) OP(snrup) OP(sprob) OP(vhigh) OP(rneff) OP(mdl)        \
    OP(mode) OP(moder) OP(du_nois) OP(dl_nois) OP(snxx) OP(gaint) OP(gainr)

__device__ inline void soncol_save_(SonCol* s, const SonState* son, int ifx) {
#define VC_OP(f) s->f = son->f[ifx];
    VC_SONCOL(VC_OP)
#undef VC_OP
}

__device__ inline void soncol_restore_(const SonCol* s, SonState* son,
                                       int ifx) {
#define VC_OP(f) son->f[ifx] = s->f;
    VC_SONCOL(VC_OP)
#undef VC_OP
}

// One frequency's row of the smoothing store, likewise snapshotted.
struct StoreRow {
    int w[2 * 20 + 5];
};

#define VC_STOREROW(OP)                                                       \
    OP(angle) OP(cprob) OP(dblos) OP(dblosl) OP(dblosu) OP(delay) OP(dbw)     \
    OP(vhigh) OP(dbu) OP(reliab) OP(sndb) OP(snpr) OP(snrlw) OP(snrup)        \
    OP(sprob) OP(probmp) OP(nhp) OP(smode) OP(tgain) OP(rgain)

// save != 0 copies the store's row out, save == 0 puts it back.  One routine
// for both so the field list cannot drift between them.
__device__ inline void storerow_(StoreRow* r, SmoothStore* s, int ifx,
                                 int save) {
    int p1 = ifx * VC_SMOOTH_NPASS + 1;
    int p2 = ifx * VC_SMOOTH_NPASS + 2;
    int n = 0;
#define VC_OP(F) { int* a = (int*)&s->F[p1]; int* b = (int*)&s->F[p2];        \
                   if (save) { r->w[n] = *a; r->w[n + 1] = *b; }              \
                   else { *a = r->w[n]; *b = r->w[n + 1]; }                   \
                   n += 2; }
    VC_STOREROW(VC_OP)
#undef VC_OP
#define VC_OP1(P) { int* a = (int*)(P); if (save) r->w[n] = *a;               \
                    else *a = r->w[n]; n += 1; }
    VC_OP1(&s->zdu[ifx])
    VC_OP1(&s->zdl[ifx])
    VC_OP1(&s->inmode[ifx])
    VC_OP1(&s->zangler[ifx])
    VC_OP1(&s->zmoder[ifx])
#undef VC_OP1
}

// SMOOTHSTORE_INIT, split across the warp.
__device__ inline void smoothstore_init_warp_(SmoothStore* s, int lane) {
    for (int i = lane; i < VC_SMOOTH_NROW * VC_SMOOTH_NPASS; i += 32) {
        s->angle[i] = 0.0f;  s->cprob[i] = 0.0f;  s->dblos[i] = 0.0f;
        s->dblosl[i] = 0.0f; s->dblosu[i] = 0.0f; s->delay[i] = 0.0f;
        s->dbw[i] = 0.0f;    s->vhigh[i] = 0.0f;  s->dbu[i] = 0.0f;
        s->reliab[i] = 0.0f; s->sndb[i] = 0.0f;   s->snpr[i] = 0.0f;
        s->snrlw[i] = 0.0f;  s->snrup[i] = 0.0f;  s->sprob[i] = 0.0f;
        s->probmp[i] = 0.0f; s->tgain[i] = 0.0f;  s->rgain[i] = 0.0f;
        s->nhp[i] = 0;
        s->smode[i] = -1;                           // ""
    }
    for (int i = lane; i < VC_SMOOTH_NROW; i += 32) {
        s->zdu[i] = 0.0f; s->zdl[i] = 0.0f; s->zangler[i] = 0.0f;
        s->inmode[i] = 0;
        s->zmoder[i] = -1;                          // ""
    }
    __syncwarp();
}

// ---------------------------------------------------------------------------
// The two model bodies, on the lane's private state.  Line for line the same
// as luffy.cuh's, with the struct pointers redirected and `reach` reported;
// they are NOT shared with the one-thread kernel, which stays untouched as
// the correctness reference.
// ---------------------------------------------------------------------------
__device__ inline int luffy_short_path_lane_(const CircuitState* c,
                                             const CircuitTables* t,
                                             const RaysView& rays,
                                             LaneState* L, float freq, int k,
                                             int ihmin, int ipfg, int* skip,
                                             int* reach) {
    *skip = 0;
    int ihsrt, ihstp;
    if (L->reflx.dmaxkm[k] <= 0.0f) {
        if (ipfg == 300) {
            *skip = 1;
            return 0;
        }
        // .....ONLY ONE OVER THE MUF MODE
        ihsrt = ihmin;
        ihstp = ihmin;
    } else {
        // .....UP TO THREE HOPS
        ihsrt = (int)(c->gcdkm / L->reflx.dmaxkm[k] + 1.0f);
        int ihmax = (int)(c->gcdkm / L->reflx.dskpkm[k]);
        if (ihsrt < ihmin)
            ihsrt = ihmin;
        if (ihmax < ihsrt)
            ihmax = ihsrt;
        int cap = ihsrt + VC_LUFFY_MAX_EXTRA_HOPS;
        ihstp = (ihmax < cap) ? ihmax : cap;
        if (ihsrt > ihmin) {
            int lo = ihstp - VC_LUFFY_MAX_EXTRA_HOPS;
            ihsrt = (ihmin > lo) ? ihmin : lo;
        }
    }

    *reach = VC_WR_MODEL;
    for (int ihop = ihsrt; ihop <= ihstp; ++ihop) {
        float hop = (float)ihop;
        L->modes.ghop = c->gcd / hop;               // .....HOP DISTANCE
        (void)fdist(&L->reflx, &L->modes, k, freq, c->amind);
        int nummod = 0;
        int st = inmuf(&c->layers, &L->modes, &L->muf, &L->sigd, &L->anois,
                       rays, t->array, t->aeff, t->iats, t->xfqs, t->xfqe,
                       t->offazim, t->pwrdba, t->numants, &L->zon, freq,
                       c->frq.jmode, c->gcd, ihop, &nummod);
        if (st) return st;
        float dummy = 0.0f;
        st = allmodes(&L->zon, &L->allm, ipfg, hop, &dummy);
        if (st) return st;
    }

    int st = esmod(&c->layers, &L->muf, &L->sigd, &L->anois, rays, t->array,
                   t->aeff, t->iats, t->xfqs, t->xfqe, t->offazim, t->pwrdba,
                   t->numants, &L->zon, freq, c->frq.jmode, c->gcd, c->gcdkm,
                   c->amind);
    if (st) return st;
    esreg(&L->zon);
    float dummy = 0.0f;
    return allmodes(&L->zon, &L->allm, 2, VC_LUFFY_ES_FVAL, &dummy);
}

__device__ inline int luffy_long_path_lane_(const CircuitState* c,
                                            const CircuitTables* t,
                                            const RaysView& rays,
                                            LaneState* L, float freq) {
    int st = gmloss(&c->layers, &L->muf, &L->sigd, &L->reflx, &L->losx, rays,
                    t->array, t->aeff, t->iats, t->xfqs, t->xfqe, t->offazim,
                    t->numants, &L->allm, &L->zon, &L->lpath, c->frq.itxrcp,
                    freq, c->gcdkm, c->amind);
    if (st) return st;
    st = seltxr(&L->reflx, &L->losx, &L->lpath, c->frq.itxrcp, c->gcdkm,
                c->amind, c->info);
    if (st) return st;
    return lngpat(&c->layers, &L->lpath, &L->losx, &L->reflx, &L->muf,
                  &L->allm, &L->sigd, &L->anois, t->iats, t->xfqs, t->xfqe,
                  t->pwrdba, t->numants, freq, c->gcd, c->frq.itxrcp[2]);
}

// ---------------------------------------------------------------------------
// The whole DO-265 loop, twelve frequencies at once.
//
// EVERY LANE OF THE WARP MUST CALL THIS, lanes 12..31 included: the two warp
// scans and the ballots are collective, so a lane that has nothing to do
// still has to arrive at them.  Hence `alive` instead of `return`.
//
// Returns the loop's status; *found_ifx comes back as the LUF frequency, or 0.
// ---------------------------------------------------------------------------
__device__ inline int luffy_freq_loop_warp_(CircuitState* c,
                                            const CircuitTables* t,
                                            const RaysView& rays,
                                            LaneState* L, SmoothStore* store,
                                            int lane, int ipflag, int k,
                                            int ihmin, float pluf,
                                            int* found_ifx) {
    const int ifx = lane + 1;
    const int act = (lane < VC_LUFFY_NFREQ);
    const int ipfg = ipflag;
    const int is_long = !(ipfg == 100 || ipfg == 300);

    // ---- seed the lane -----------------------------------------------------
    L->muf = c->muf;
    L->sigd = c->sigd;
    L->anois = c->anois;
    L->ton = c->ton;
    L->ndiag = c->ndiag;
    L->lpath = c->lpath;                    // TXRGML poisoned just below
    L->freq = c->frq.freq;
    L->d10s = c->son.d10s;
    L->d50s = c->son.d50s;
    L->d90s = c->son.d90s;
    // The scratch: poison in, so what survives is the lane's write mask.
    // (Skipping /LOSX/ on the short path and /MODES/ on the long -- neither
    // path touches the other's -- saves 6 KB of the 16 and MEASURES 6-9%
    // SLOWER at every size, so it is not done: the fill and the merge are not
    // what this kernel is short of.  See the A/B below.)
#define VC_WR_POISON_ONE(S) wr_fill_((int*)&L->S, (int)(sizeof(L->S) / 4),    \
                                     (int)VC_WARP_POISON);
    VC_WR_POISON_ONE(reflx)
    VC_WR_POISON_ONE(modes)
    VC_WR_POISON_ONE(zon)
    VC_WR_POISON_ONE(allm)
    VC_WR_POISON_ONE(losx)
#undef VC_WR_POISON_ONE
    wr_fill_((int*)L->lpath.txrgml, VC_WR_TXRGML_WORDS, (int)VC_WARP_POISON);

    SonCol saved;
    StoreRow srow;
    if (act) {
        soncol_save_(&saved, &c->son, ifx);
        storerow_(&srow, store, ifx, 1);
    }

    int st = 0;
    int reach = 0;
    int found = 0;
    int alive = act;
    float freq = 0.0f;

    // ---- the preamble, ALLMODES(0), GENOIS --------------------------------
    if (alive) {
        if (ipfg < 300) {
            // .....PRESET IN COMMON/SON/
            c->son.nhp[ifx] = -1;
            c->son.sprob[ifx] = 0.0f;
            c->son.probmp[ifx] = 0.0f;
            if (c->frq.frel[ifx] <= 0.0f)
                alive = 0;                  // no frequency in this slot
            else
                freq = c->frq.frel[ifx];
        } else {
            freq = c->frq.frea[ifx];
            c->son.reliab[ifx] = 0.0f;
        }
    }
    if (alive) {
        reach = VC_WR_ALLM;
        float newfreq = 0.0f;
        st = allmodes(&L->zon, &L->allm, 0, freq, &newfreq);
        if (st) alive = 0;
        else { freq = newfreq; L->freq = freq; }
    }
    if (alive) {
        reach = VC_WR_NOISE;
        st = genois(t->fam, t->dud, t->f2d111, &c->layers, &L->anois, &L->ton,
                    &L->zon, &L->ndiag, t->array, t->aeff, t->iats, t->xfqs,
                    t->xfqe, t->offazim, t->numants, freq, c->noise, c->rlat,
                    c->nyear_is_2089);
        if (st) alive = 0;
    }

    // ---- FINDF, then SCAN 1: the skip group -------------------------------
    if (alive) {
        reach = VC_WR_REFLX;
        if (is_long) {
            // .....AREA COVERAGE, TRANSMITTER END, THEN RECEIVER END
            findf(&c->layers, &L->reflx, rays, c->frq.itxrcp[1], freq,
                  c->amind);
            findf(&c->layers, &L->reflx, rays, c->frq.itxrcp[2], freq,
                  c->amind);
        } else {
            // .....RAY SET TABLE
            findf(&c->layers, &L->reflx, rays, k, freq, c->amind);
        }
    }
    // Collective: all 32 lanes.  Element by element, the value FINDF wrote,
    // or the highest lower lane's, or the pre-loop one.
    {
        int* ia = c->reflx.iaftxr;
        int* ip = L->reflx.iaftxr;
        for (int i = 0; i < 4; ++i) {
            int v = ip[i];
            ip[i] = wr_scan_(v, (unsigned)v != VC_WARP_POISON, ia[i], lane);
        }
#define VC_WR_CARRY(F)                                                        \
        for (int i = 0; i < 4; ++i) {                                         \
            int v = *(int*)&L->reflx.F[i];                                    \
            *(int*)&L->reflx.F[i] = wr_scan_(v,                               \
                    (unsigned)v != VC_WARP_POISON, *(int*)&c->reflx.F[i],     \
                    lane);                                                    \
        }
        VC_WR_CARRY(iskp)
        VC_WR_CARRY(delskp)
        VC_WR_CARRY(hpskp)
        VC_WR_CARRY(htskp)
        VC_WR_CARRY(fvskp)
        VC_WR_CARRY(dskpkm)
        VC_WR_CARRY(dmaxkm)
#undef VC_WR_CARRY
    }

    // ---- the model body ---------------------------------------------------
    if (alive) {
        if (is_long) {
            reach = VC_WR_MODEL;
            st = luffy_long_path_lane_(c, t, rays, L, freq);
            if (st) alive = 0;
        } else {
            int skip = 0;
            st = luffy_short_path_lane_(c, t, rays, L, freq, k, ihmin, ipfg,
                                        &skip, &reach);
            if (st) alive = 0;
            else if (skip) alive = 0;       // label 122 -> 265
        }
    }
    if (alive) {
        // .....COMBINED RELIABILITY
        st = genois(t->fam, t->dud, t->f2d111, &c->layers, &L->anois, &L->ton,
                    &L->zon, &L->ndiag, t->array, t->aeff, t->iats, t->xfqs,
                    t->xfqe, t->offazim, t->numants, freq, c->noise, c->rlat,
                    c->nyear_is_2089);
        if (st) alive = 0;
    }

    // ---- RELBIL, in halves, with SCAN 2 between them ----------------------
    RelbilDon don;
    don.d10r = c->son.d10r;
    don.d50r = c->son.d50r;
    don.d90r = c->son.d90r;
    int ran = 0, wrote = 0, ir = 0;
    if (alive) {
        reach = VC_WR_RELBIL;
        relbil_head(&L->allm, &L->anois, &L->sigd, &c->son, ifx, c->rsn, &don,
                    &ran, &wrote, &ir);
    }
    {   // collective
        int e10 = __float_as_int(c->son.d10r);
        int e50 = __float_as_int(c->son.d50r);
        int e90 = __float_as_int(c->son.d90r);
        int have = alive && ran && wrote;
        don.d10r = __int_as_float(wr_scan_(__float_as_int(don.d10r), have,
                                           e10, lane));
        don.d50r = __int_as_float(wr_scan_(__float_as_int(don.d50r), have,
                                           e50, lane));
        don.d90r = __int_as_float(wr_scan_(__float_as_int(don.d90r), have,
                                           e90, lane));
    }
    if (alive && ran) {
        st = relbil_tail(&L->allm, &L->anois, &c->son, t->iats, t->xfqs,
                         t->xfqe, t->pwrdba, t->numants, ifx, freq, c->rsn,
                         c->lufp, &don, ir);
        if (st) alive = 0;
    }

    // ---- SERPRB / MPATH / the receiver end, or the LUF test ---------------
    if (alive) {
        reach = VC_WR_SERPRB;
        if (ipfg < 300) {
            if (c->want_service_prob)       // IF( LINBOT(14) ) 285,285,280
                serprb_ex(&L->allm, &L->anois, &L->ton, &c->son, ifx, c->rsn,
                          c->lufp, &L->d10s, &L->d50s, &L->d90s);
            if (ipfg == 100) {
                mpath(&L->allm, &c->son, ifx, c->dmp, c->pmp, c->gcdkm,
                      c->method, c->mspec);
            } else {
                int is_ = L->allm.nmode[2];
                if (!(1 <= is_ && is_ <= VC_LUFFY_NLAYTYP)) {
                    st = VC_LUFFY_BAD_NMODE2;
                    alive = 0;
                } else {
                    c->son.moder[ifx] = is_ - 1;
                    c->son.angler[ifx] = L->allm.b[2];
                }
            }
        } else {
            // .....TEST RELIABILITY
            if (c->son.reliab[ifx] >= pluf) {
                found = 1;
                alive = 0;                  // label 165, and no save
            }
        }
    }
    if (alive && c->mspec == 121) {
        st = save_pass_parts(&c->son, &L->allm, &L->anois, store, ifx,
                             ipflag);
        if (st) alive = 0;
    }

    // ---- where the serial loop would have stopped -------------------------
    unsigned badm = __ballot_sync(VC_WARP_FULL, act && st != 0);
    unsigned fndm = __ballot_sync(VC_WARP_FULL, act && found);
    int firstbad = badm ? (__ffs((int)badm) - 1) : VC_LUFFY_NFREQ;
    int firstfnd = fndm ? (__ffs((int)fndm) - 1) : VC_LUFFY_NFREQ;
    int cut = (firstbad < firstfnd) ? firstbad : firstfnd;
    if (cut > VC_LUFFY_NFREQ - 1)
        cut = VC_LUFFY_NFREQ - 1;

    // roll back the lanes the serial loop never reached
    if (act && lane > cut) {
        soncol_restore_(&saved, &c->son, ifx);
        storerow_(&srow, store, ifx, 0);
    }
    __syncwarp();

    // ---- commit -----------------------------------------------------------
    int keep = act && (lane <= cut);
#define VC_WR_MERGE_ONE(S) wr_merge_((int*)&c->S, (const int*)&L->S,          \
                                     (int)(sizeof(L->S) / 4), keep, lane);
    VC_WR_MERGE_ONE(reflx)
    VC_WR_MERGE_ONE(modes)
    VC_WR_MERGE_ONE(zon)
    VC_WR_MERGE_ONE(allm)
    VC_WR_MERGE_ONE(losx)
#undef VC_WR_MERGE_ONE
    wr_merge_((int*)c->lpath.txrgml, (const int*)L->lpath.txrgml,
              VC_WR_TXRGML_WORDS, keep, lane);

    int h1 = wr_last_(keep && reach >= VC_WR_ALLM);
    int h2 = wr_last_(keep && reach >= VC_WR_NOISE);
    int h4 = wr_last_(keep && reach >= VC_WR_MODEL);
    int h6 = wr_last_(keep && reach >= VC_WR_RELBIL);
    int h7 = wr_last_(keep && reach >= VC_WR_SERPRB);
    if (lane == h1)
        c->frq.freq = L->freq;
    if (lane == h2) {
        c->anois = L->anois;
        c->ton = L->ton;
        c->ndiag = L->ndiag;
    }
    if (lane == h4) {
        c->muf = L->muf;
        if (is_long) {
            c->sigd = L->sigd;
            c->lpath.gcdlng = L->lpath.gcdlng;
            c->lpath.delopt = L->lpath.delopt;
            c->lpath.gmin = L->lpath.gmin;
            c->lpath.ymin = L->lpath.ymin;
            for (int i = 0; i < 3; ++i)
                c->lpath.ltxrgm[i] = L->lpath.ltxrgm[i];
        }
    }
    if (!is_long && lane == h6)
        c->sigd = L->sigd;
    if (lane == h7) {
        c->son.d10s = L->d10s;
        c->son.d50s = L->d50s;
        c->son.d90s = L->d90s;
    }
    // the /DON/ triple the loop leaves is the cut lane's resolved one
    if (lane == cut) {
        c->son.d10r = don.d10r;
        c->son.d50r = don.d50r;
        c->son.d90r = don.d90r;
    }
    __syncwarp();

    // ---- the loop's own answer -------------------------------------------
    int out_st = 0;
    if (firstbad <= cut)
        out_st = __shfl_sync(VC_WARP_FULL, st, firstbad);
    *found_ifx = (firstbad > firstfnd && firstfnd <= cut) ? firstfnd + 1 : 0;
    return out_st;
}

// ---------------------------------------------------------------------------
// LUFFY's body, one warp.  Setup on lane 0, the frequency loop warp-wide,
// label 165 / label 300 / the blend back on lane 0.
// ---------------------------------------------------------------------------
__device__ inline int luffy_body_warp_(CircuitState* c,
                                       const CircuitTables* t,
                                       const RaysView& rays, LaneState* L,
                                       SmoothStore* store, int lane,
                                       int ipflag) {
    smoothstore_init_warp_(store, lane);

    int ipfg = ipflag;
    float pluf = 0.0f;
    int st = 0;
    if (lane == 0) {
        if (c->mspec == 121 && c->gcdkm > 10000.0f) {
            ipfg = 200;                     // long-path SETUP only
            c->mspec = 0;                   // beyond the smoothing distance
        }
        if (ipfg > 200) {
            // .....FREQUENCY COMPLEMENT, and the required reliability
            st = frqcom(&c->muf, c->frq.frea, 0, c->time.it);
            pluf = 0.01f * (float)c->lufp;
        }
        if (!st) {
            // .....SELECT CONTROLLING SAMPLE AREA
            c->frq.jmode = selmod(&c->layers);
            // .....GET NOISE VALUE AT 1 MHZ FOR RCVR SITE
            anois1(&c->anois, t->fakp, t->fakmap, t->hmym, t->fakabp,
                   t->abmap, t->f2d111, c->time.gmtr, c->rlong, c->rlongd,
                   c->rlat);
        }
    }
    __threadfence_block();
    __syncwarp();
    ipfg = __shfl_sync(VC_WARP_FULL, ipfg, 0);
    pluf = __shfl_sync(VC_WARP_FULL, pluf, 0);
    st = __shfl_sync(VC_WARP_FULL, st, 0);
    if (st)
        return st;

    // Label 83.  The smoothing recycle comes back here, NOT to the top.
    while (true) {
        int k = 0, xmdl = 0;
        if (lane == 0) {
            luffy_select_area_(c, rays, ipfg, &k, &xmdl);   // labels 83-85
            // .....CAN RUN FOR SHORT PATHS ALSO, SO SET ALL SAMPLE AREAS
            setlng(&c->layers, rays);
            // .....SET ABSORPTION LOSS PARAMETERS, ADJUST SIGNAL
            // .....DISTRIBUTION TABLES
            sigdis(&c->layers, &c->muf, t->sys, t->perr, t->f2d111, &c->sigd,
                   &c->ton, c->gcdkm, c->frq.jmode);
            // .....START FREQUENCY LOOP
            c->son.nhp[VC_LUFFY_ILUF] = -1;
            c->son.mdl[VC_LUFFY_ILUF] = xmdl;
            for (int ifx = 1; ifx <= VC_LUFFY_NFREQ; ++ifx)
                c->son.mdl[ifx] = VC_MDL_BLANK;
        }
        __threadfence_block();
        __syncwarp();
        k = __shfl_sync(VC_WARP_FULL, k, 0);
        int ihmin = luffy_min_hops_(c->muf.nhopmf);     // IHMX here, and dead

        int found_ifx = 0;
        st = luffy_freq_loop_warp_(c, t, rays, L, store, lane, ipflag, k,
                                   ihmin, pluf, &found_ifx);
        if (st)
            return st;

        int again = 0;
        if (lane == 0) {
            if (found_ifx)
                luffy_luf_found_(c, found_ifx, pluf);    // label 165
            else
                st = luffy_after_loop_(c, t, ipflag, store, &again);
        }
        __threadfence_block();
        __syncwarp();
        st = __shfl_sync(VC_WARP_FULL, st, 0);
        again = __shfl_sync(VC_WARP_FULL, again, 0);
        if (st || found_ifx || !again)
            return st;
        // .....DO LONG AND SHORT PATH MODEL: round two, the other model.
        ipflag = 200;
        ipfg = 200;
    }
}

// ---------------------------------------------------------------------------
// LUFFY, one warp.  Same contract as `luffy`, same tables, same status codes;
// `lane` is threadIdx.x & 31 and `store` is this warp's slice of dynamic
// shared memory.
// ---------------------------------------------------------------------------
__device__ inline int luffy_warp(CircuitState* c,
                                 int* rays_nang, float* rays_ang,
                                 int* rays_ifob, int rays_n, int rays_t,
                                 LaneState* L, SmoothStore* store, int lane,
                                 const float* fam, const float* dud,
                                 const float* fakp, const float* fakmap,
                                 const float* hmym, const float* fakabp,
                                 const float* abmap, const float* sys,
                                 const float* perr, const float* f2d111,
                                 const float* ant_array, const float* ant_aeff,
                                 const int* ant_iats, const float* ant_xfqs,
                                 const float* ant_xfqe,
                                 const float* ant_offazim,
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

    RaysView rays = rays_view_soa(rays_nang, rays_ang, rays_ifob, rays_n,
                                  rays_t);

    int mspec_temp = c->mspec;              // restored at label 999
    int st = luffy_body_warp_(c, &t, rays, L, store, lane, ipflag);
    __threadfence_block();
    __syncwarp();
    if (lane == 0)
        c->mspec = mspec_temp;              // Stage A's try/finally
    return st;
}

#endif  // VOACAP_LUFFY_WARP_CUH
