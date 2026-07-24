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
// THREE THINGS CAPPED THIS DESIGN, AND ALL THREE HAVE NOW MOVED.  What the
// file used to say, and what it says now:
//
//   1. THE SETUP PHASE RAN ON LANE 0.  SELMOD, ANOIS1, LECDEN/GENION/FOBBY/
//      ALOSFV, SETLNG and SIGDIS are a per-circuit-hour prologue with no
//      frequency in them, and thirty-one lanes waited through all of it.
//      They now run on the whole warp -- see "THE SETUP PHASE, ACROSS THE
//      LANES" below.  MEASURED, one binary apart on one machine state, by
//      compiling with -DVC_WARP_SETUP_LANES=1 (which puts the setup back
//      whole in every lane, i.e. back on one lane's worth of work) and by
//      giving the kernel the DO-265 trip count as a runtime argument so the
//      register allocation is held identical:
//
//                              504    4,032    8,064   32,256
//          setup share BEFORE  0.39     0.52     0.54     0.51
//          setup share AFTER   0.20     0.33     0.33     0.31
//          Amdahl ceiling      7.8x     6.3x     6.1x     6.4x   before
//                              9.8x     8.4x     8.4x     8.6x   after
//
//      (the ceiling is (S + 12(W-S))/W: the serial program is the setup plus
//      twelve of one lane's DO-265 trips.  S here is an OVER-estimate -- the
//      trip-count-zero kernel still pays the poison fill, the merge and the
//      blend -- so the true setup share is smaller than the table says and
//      the true ceiling higher.)
//
//   2. IT WANTED 255 REGISTERS.  It now wants 144: the rollback snapshot went
//      out of line.  See "THE ROLLBACK SNAPSHOT" below.  256 -> 448 resident
//      threads/SM, and faster with it.
//
//   3. IT WASTED 20 OF EVERY 32 LANES.  VC_WARP_GROUPS=2 puts TWO
//      circuit-hours in one warp, sixteen lanes each, and every ballot, scan
//      and syncwarp in this file is now group-local so that is exact -- it is
//      proved bit-identical on all 504 replay circuit-hours by
//      tests/stage_c/test_luffy_warp.py.  It is NOT the shipping default,
//      because the launch has to change with it and the launch is not in this
//      file; see "PACKING" below.
//
// WHAT IT IS WORTH, AND WHERE.  All five columns timed in ONE interleaved
// run, kernel only, 4 warps/block, same staged batch.  (The GPU was shared
// while this was measured, so read the RATIOS, not the milliseconds -- the
// per-eval figures are perhaps 3x above what a quiet machine gives.)
//
//     circuit-hours       504   2,016   4,032   8,064  16,128  32,256
//     thread   us/eval  36.71   10.34    5.35    2.95    1.64    1.23
//     BEFORE   us/eval   7.36    3.99    3.52    3.19    3.15    3.06
//     AFTER    us/eval   4.88    2.44    2.20    1.69    1.83    1.70
//     packed   us/eval   5.57    2.15    1.69    1.46    1.32    1.10
//
//     AFTER  / BEFORE    1.51x   1.64x   1.60x   1.89x   1.72x   1.80x
//     packed / BEFORE    1.32x   1.85x   2.09x   2.18x   2.40x   2.78x
//     thread / AFTER     7.52x   4.24x   2.44x   1.75x   0.90x   0.72x
//     thread / packed    6.59x   4.80x   3.17x   2.02x   1.25x   1.12x
//
// THE CROSSOVER AGAINST `luffy_batch` HAS MOVED FROM ~8,000 CIRCUIT-HOURS TO
// ~14,000, and to ~40,000 packed (measured: packed is 1.06x the thread kernel
// at 32,256 and 0.90x at 64,512).  Why there is a crossover at all is
// unchanged: `luffy_batch` is latency-bound and hides its latency by having
// thousands of INDEPENDENT circuit-hours in flight, so on a full machine it
// does not need a shorter chain, while this kernel spends 32 threads to get
// one.  Everything above raises the number of circuit-hours this kernel can
// keep in flight (448 resident threads/SM instead of 256; two circuit-hours
// per warp instead of one) or shortens what they have to do.
//
// Packed loses at 504 (5.57 against 4.88 us/eval) and that is expected: with
// two groups the setup splits over sixteen lanes instead of thirty-two, and
// on an empty machine there is nothing to spend the freed warps on.  Neither
// kernel replaces the other; pick by batch size.
//
// PACKING: WHAT THE DRIVER HAS TO DO
// ----------------------------------
// Building this header with -DVC_WARP_GROUPS=2 makes every collective
// group-local, but the LAUNCH still hands one circuit-hour to one warp.  The
// change is entirely in `stage_c/driver.py` -- four lines in the kernel and
// three in `stage_luffy` -- and is spelled out in full, and proved exact, in
// tests/stage_c/test_luffy_warp.py's `_PACKED_SRC`:
//
//     device, in luffy_batch_warp:
//         int grp  = lane / VC_WARP_GLANES;          // NEW
//         int slot = wib * VC_WARP_GROUPS + grp;     // NEW
//         int i = blockIdx.x * ((blockDim.x >> 5) * VC_WARP_GROUPS) + slot;
//         ... &warp_store[slot] ...                  // was warp_store[wib]
//         if ((lane & (VC_WARP_GLANES - 1)) == 0)    // was lane == 0
//             status[i] = st;
//
//     host, in stage_luffy(warp=True):
//         per_block = wpb * VC_WARP_GROUPS
//         grid   = ((n + per_block - 1) // per_block,)
//         shared = per_block * SMOOTHSTORE_BYTES
//
// VC_WARP_GROUPS must leave at least NFREQ = 12 lanes per group, so 2 is the
// only value above 1 that works on a 32-lane warp.
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
// Every one of those is now GROUP-local rather than warp-local, so that two
// circuit-hours can share a warp without their cuts, their scans or their
// merges touching.  `wr_ballot_`/`wr_scan_`/`wr_last_`/`wr_merge_` all take a
// WarpGroup; at VC_WARP_GROUPS = 1 the group IS the warp and the code is what
// it always was.
//
// WHAT STAYS SERIAL.  Label 300 (_no_luf), label 165 (_luf_found) and the
// 7000-10000 km blend all run after the loop on the group's lane 0 -- `blend`
// is itself a twelve-frequency loop with a genuine carry through /ANOIS/ and
// /allMODE/, and it is a few percent of the work.  ALOSFV likewise, and the
// scalar spine of each setup routine -- but "serial" there means run
// REDUNDANTLY IN EVERY LANE, which in lockstep costs the same as running it
// in one and needs no broadcast afterwards.

#ifndef VOACAP_LUFFY_WARP_CUH
#define VOACAP_LUFFY_WARP_CUH

#include "luffy.cuh"

#define VC_WARP_FULL   0xffffffffu
#define VC_WARP_POISON 0xFFBADBADu      // NaN as a float, negative as an int

// ---------------------------------------------------------------------------
// HOW MANY CIRCUIT-HOURS SHARE ONE WARP
//
// NFREQ is 12, so one circuit-hour per warp leaves twenty of thirty-two lanes
// idle.  VC_WARP_GROUPS cuts the warp into that many equal LANE GROUPS, each
// taking its own circuit-hour: 2 gives sixteen lanes each (twelve on
// frequencies, four idle), which doubles lane occupancy and halves the number
// of warps -- and, because the two groups are independent circuit-hours,
// EVERY collective in this file has to become group-local.  That is what the
// WarpGroup handle below is for: `w.mask` names the group's lanes for the
// ballots and the syncs, `w.g` is the lane's index WITHIN its group, and
// `w.base` is the absolute lane the group starts at, which is what
// __shfl_sync needs (its source index is absolute, not masked).
//
// 1 IS WHAT SHIPS, and at 1 every one of those reduces to the whole warp:
// mask 0xffffffff, w.g == lane, w.base == 0, so the generated code is
// identical to the one-circuit-hour-per-warp kernel.  Building with
// -DVC_WARP_GROUPS=2 ALSO NEEDS THE LAUNCH TO CHANGE -- the circuit index,
// the SmoothStore slice, the status write and the grid size all live in
// stage_c/driver.py's kernel, which this header does not own.  See "PACKING"
// in the header comment for the exact change.
// ---------------------------------------------------------------------------
#ifndef VC_WARP_GROUPS
#define VC_WARP_GROUPS 1
#endif
#define VC_WARP_GLANES (32 / VC_WARP_GROUPS)
// GLANES low bits set.  Written as a right shift, not (1u << GLANES) - 1,
// because that shift is undefined at GLANES == 32.
#define VC_WARP_GMASK  (VC_WARP_FULL >> (32 - VC_WARP_GLANES))

// One lane's place in its group.  Four ints, passed by value.
struct WarpGroup {
    int      lane;                      // absolute lane, 0..31
    int      g;                         // lane within the group
    int      base;                      // absolute lane of the group's lane 0
    unsigned mask;                      // the group's ballot / sync mask
};

__device__ __forceinline__ WarpGroup wr_group_(int lane) {
    WarpGroup w;
    w.lane = lane;
    w.g = lane & (VC_WARP_GLANES - 1);
    w.base = lane - w.g;
    w.mask = VC_WARP_GMASK << w.base;
    return w;
}

// The group-relative ballot: `p` polled over the group's lanes, shifted down
// so bit i is the group's lane i.
__device__ __forceinline__ unsigned wr_ballot_(const WarpGroup& w, int p) {
    return (__ballot_sync(w.mask, p) >> w.base) & VC_WARP_GMASK;
}

// How many lanes the four big setup loops are spread over -- the group's
// lanes, capped by VC_WARP_SETUP_LANES.  32 is what ships.  Compiling with
// -DVC_WARP_SETUP_LANES=1 collapses the offset to 0 and the stride to 1, so
// every lane runs every iteration -- which IS the old lane-0 setup, since
// redundancy is free in lockstep.  That is the A/B the setup fraction below
// was measured with, on one machine state, one binary apart.
#ifndef VC_WARP_SETUP_LANES
#define VC_WARP_SETUP_LANES 32
#endif
#define VC_WSTRIDE ((VC_WARP_SETUP_LANES < VC_WARP_GLANES)                    \
                    ? VC_WARP_SETUP_LANES : VC_WARP_GLANES)
#define VC_WSL(w) ((w).g & (VC_WSTRIDE - 1))

// THE ROLLBACK SNAPSHOT GOES OUT OF LINE, AND THAT IS THE OCCUPANCY FIX.
//
// SonCol is 28 scalars and StoreRow is 45 ints, and both are live across the
// WHOLE of one lane's DO-265 body -- taken before it starts, read only if the
// lane turns out to be past the cut.  Fully inlined they sit in 73 registers
// for the duration and were most of why the kernel wanted 255 (the
// architectural cap, so the real demand was higher still).  __noinline__ on
// the three routines that touch them forces the structs to be addressable,
// which puts them in local memory and hands the register allocator its live
// range back: 255 -> 144 registers, 256 -> 448 resident threads per SM, and
// it is FASTER as well as smaller -- 1.15x at 8,064 circuit-hours and 1.12x
// at 32,256 (the snapshot is written once and read at most once, so paying
// two local accesses for 73 registers is a trade with no downside).
//
// -DVC_WARP_SNAP_NI= (empty) restores the register-resident form and is how
// that A/B is reproduced.  __noinline__ on the SETUP routines or on the two
// MODEL bodies, by contrast, does not move the count at all: they are not
// what the allocator is short of.
#ifndef VC_WARP_SNAP_NI
#define VC_WARP_SNAP_NI __noinline__
#endif

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

// The group-relative bits at or below this lane.
__device__ __forceinline__ unsigned wr_below_(unsigned m, int g) {
    return m & (VC_WARP_FULL >> (31 - g));
}

// The value from the highest lane j <= this one IN THE GROUP that HAS one,
// else `fallback` (which must be group-uniform).  Carried as a bit pattern so
// one routine serves floats and ints alike.
__device__ __forceinline__ int wr_scan_(int v, int have, int fallback,
                                        const WarpGroup& w) {
    unsigned m = wr_ballot_(w, have);
    unsigned b = wr_below_(m, w.g);
    int src = b ? (31 - __clz((int)b)) : -1;
    int got = __shfl_sync(w.mask, v, w.base + (src < 0 ? 0 : src));
    return (src < 0) ? fallback : got;
}

// The highest lane of the group satisfying `p`, group-relative, or -1.
__device__ __forceinline__ int wr_last_(int p, const WarpGroup& w) {
    unsigned m = wr_ballot_(w, p);
    return m ? (31 - __clz((int)m)) : -1;
}

// The whole ordered per-cell merge: `n` words of each lane's private copy,
// applied in lane order within the group, poison meaning "this lane did not
// write it".  Written a group's width at a time so the global stores
// coalesce.
__device__ inline void wr_merge_(int* dst, const int* src, int n, int keep,
                                 const WarpGroup& w) {
    for (int base = 0; base < n; base += VC_WARP_GLANES) {
        int myval = 0;
        int myhave = 0;
        for (int j = 0; j < VC_WARP_GLANES; ++j) {
            int idx = base + j;
            int v = (idx < n) ? src[idx] : 0;
            int wrote = keep && (idx < n)
                        && ((unsigned)v != VC_WARP_POISON);
            unsigned m = wr_ballot_(w, wrote);
            int hi = m ? (31 - __clz((int)m)) : 0;
            int val = __shfl_sync(w.mask, v, w.base + hi);
            if (w.g == j) { myval = val; myhave = (m != 0); }
        }
        int idx = base + w.g;
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

__device__ VC_WARP_SNAP_NI inline void soncol_save_(SonCol* s,
                                                    const SonState* son,
                                                    int ifx) {
#define VC_OP(f) s->f = son->f[ifx];
    VC_SONCOL(VC_OP)
#undef VC_OP
}

__device__ VC_WARP_SNAP_NI inline void soncol_restore_(const SonCol* s,
                                                       SonState* son,
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
__device__ VC_WARP_SNAP_NI inline void storerow_(StoreRow* r, SmoothStore* s,
                                                 int ifx, int save) {
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

// SMOOTHSTORE_INIT, split across the group.
__device__ inline void smoothstore_init_warp_(SmoothStore* s,
                                              const WarpGroup& w) {
    int lane = w.g;
    for (int i = lane; i < VC_SMOOTH_NROW * VC_SMOOTH_NPASS;
         i += VC_WARP_GLANES) {
        s->angle[i] = 0.0f;  s->cprob[i] = 0.0f;  s->dblos[i] = 0.0f;
        s->dblosl[i] = 0.0f; s->dblosu[i] = 0.0f; s->delay[i] = 0.0f;
        s->dbw[i] = 0.0f;    s->vhigh[i] = 0.0f;  s->dbu[i] = 0.0f;
        s->reliab[i] = 0.0f; s->sndb[i] = 0.0f;   s->snpr[i] = 0.0f;
        s->snrlw[i] = 0.0f;  s->snrup[i] = 0.0f;  s->sprob[i] = 0.0f;
        s->probmp[i] = 0.0f; s->tgain[i] = 0.0f;  s->rgain[i] = 0.0f;
        s->nhp[i] = 0;
        s->smode[i] = -1;                           // ""
    }
    for (int i = lane; i < VC_SMOOTH_NROW; i += VC_WARP_GLANES) {
        s->zdu[i] = 0.0f; s->zdl[i] = 0.0f; s->zangler[i] = 0.0f;
        s->inmode[i] = 0;
        s->zmoder[i] = -1;                          // ""
    }
    __syncwarp(w.mask);
}

// ---------------------------------------------------------------------------
// THE SETUP PHASE, ACROSS THE LANES
//
// SELMOD, ANOIS1, LECDEN/GENION/FOBBY/ALOSFV, SETLNG and SIGDIS used to run on
// lane 0 while the other thirty-one waited, which is precisely the serial
// fraction that caps the design.  They run here instead on the whole warp,
// under ONE rule that makes the transcription safe:
//
//     EVERY LANE RUNS THE SCALAR PARTS REDUNDANTLY, AND ONLY THE BIG
//     INDEPENDENT LOOPS ARE SPLIT.
//
// Redundant is free.  A warp issues one instruction for all thirty-two lanes,
// so running a serial chain in every lane costs exactly what running it in
// lane 0 costs; the stores all land on the same address and coalesce into one
// transaction.  What it BUYS is that the answer ends up in every lane's
// registers and every lane's view of memory, so nothing has to be broadcast
// and no fence is needed -- and because every lane computes the same values,
// every branch is warp-uniform, so the lanes never diverge and the read-modify
// -write chains (FREA's ladder, LECDEN's height grid) stay in lockstep.
//
// Split needs a fence.  A lane that owns only some indices of an array has
// not written the others, so a `wr_sync_()` (a block fence, which is also a
// compiler barrier, plus `__syncwarp()` to reconverge) closes every split
// routine before the next one reads what it produced.
//
// What is split, and why it is exactly the same arithmetic:
//
//     LECDEN  DO 375, fifty independent heights          50 steps -> 2
//     GENION  the thirty ionogram heights (GETHP is a
//             twenty-node Gaussian with two table
//             searches per node -- the single most
//             expensive thing in the setup)             30 steps -> 1
//     FOBBY   the thirty sounding frequencies at each
//             of NANG angles                          1230 steps -> 41
//     SETLNG  the 40x30 IFOB block per copied area     1200 steps -> 40
//     SIGDIS  the KFX sample areas (ABIY + SYSSY),
//             with the six path sums re-added IN AREA
//             ORDER through __shfl_sync so the
//             floating-point summation order is the
//             Fortran's, not a tree reduction's        KFX steps -> 1
//     ANOIS1  the two NOISY evaluations, as ONE call
//             with a per-lane KJ -- both maps take the
//             identical code path, so the "second"
//             call is free                                2 -> 1
//
// ALOSFV stays whole: its collision frequency CF carries from loop to loop
// and from the last iteration that took the else-branch, so splitting it
// would need a last-writer scan for a routine that is ~30 EXPs.
// ---------------------------------------------------------------------------

// Make one lane's writes visible to the rest, and reconverge.  The fence is
// also a compiler barrier, so values cached in registers across it reload.
__device__ __forceinline__ void wr_sync_(const WarpGroup& w) {
    __threadfence_block();
    __syncwarp(w.mask);
}

// LECDEN, split at DO 375.  Everything above it is the serial scalar fit and
// the height grid, run in every lane.
__device__ inline void lecden_warp_(LayerState* s, int k,
                                    const WarpGroup& w) {
    float* fi = s->fi;
    float* yi = s->yi;
    float* hi = s->hi;
    float* htr = s->htr;
    float* fnsq = s->fnsq;
    float* fsecv = s->fsecv;

    for (int i = 1; i <= 3; ++i)        // DO 90
        if (s->ielect[i] > 0)           // 85
            return;

    float xlow = 0.8516f;
    float hz = hi[1 * 6 + k] - yi[1 * 6 + k];
    float xup = 0.98f * fi[1 * 6 + k] / fi[3 * 6 + k];
    float hlow = hz + yi[1 * 6 + k] * (1.0f + sqrt32(1.0f - xlow * xlow));
    float hte = hi[1 * 6 + k] + yi[1 * 6 + k];
    float hb2 = hi[3 * 6 + k] - yi[3 * 6 + k];
    float fce = fi[1 * 6 + k] * fi[1 * 6 + k];
    float fc2 = fi[3 * 6 + k] * fi[3 * 6 + k];
    float hup = hb2 + yi[3 * 6 + k] * (1.0f - sqrt32(1.0f - xup * xup));
    float fup = xup * xup * fc2;
    float flow = xlow * xlow * fce;
    float asp = 0.0f;
    if (hup - hlow > 0.0f)              // 165
        asp = (fup - flow) / (hup - hlow);

    int lin = 0;
    float s1 = 0.0f;
    float fc1 = 0.0f;
    float hb1 = hi[2 * 6 + k] - yi[2 * 6 + k];
    float ht1 = hi[2 * 6 + k] + yi[2 * 6 + k];
    if (fi[2 * 6 + k] > 0.0f) {         // 185
        fc1 = fi[2 * 6 + k] * fi[2 * 6 + k];
        float htw = hb2 + yi[3 * 6 + k] * (1.0f - sqrt32(1.0f - fc1 / fc2));
        if (htw > hi[2 * 6 + k] + 0.001f) {      // -> 205
            lin = 0;
            yi[2 * 6 + k] = pymin(yi[2 * 6 + k],
                                  hi[2 * 6 + k] - hi[1 * 6 + k] + 1.0f);
            fsecv[k] = -1.0f;
        } else {                        // 190
            float ys = htw - hb1;
            ys = pymax(1.0f, ys);
            s1 = fc1 / ys;
            hi[2 * 6 + k] = htw;
            yi[2 * 6 + k] = ys;
            if (hb2 - hb1 < 0.0f) {     // 194 -> 205
                yi[2 * 6 + k] = hi[2 * 6 + k] - hb2;
                lin = 0;
                yi[2 * 6 + k] = pymin(yi[2 * 6 + k],
                                      hi[2 * 6 + k] - hi[1 * 6 + k] + 1.0f);
                fsecv[k] = -1.0f;
            } else {                    // 195
                lin = 1;
                fsecv[k] = sqrt32(flow);
                float t = fi[1 * 6 + k] / fi[2 * 6 + k];
                float denom = 1.0f - t * t;
                denom = pymax(0.17f, denom);
                float yb = (hi[2 * 6 + k] - hi[1 * 6 + k]) / denom;
                if (ys - yb >= 0.0f) {  // 401
                    ys = yb;
                    yi[2 * 6 + k] = ys;
                    s1 = fc1 / ys;
                    hb1 = hi[1 * 6 + k];
                }
                ht1 = htw;              // 415
            }
        }
    }

    float hd = 70.0f;
    float xtail = 0.85f;
    float hex = hi[1 * 6 + k] - xtail * yi[1 * 6 + k];
    float fnx = 1.0f - xtail * xtail;
    float alp = 2.0f * (hi[1 * 6 + k] - hex)
                / (fnx * yi[1 * 6 + k] * yi[1 * 6 + k]);
    float fsq = fnx * exp32(-alp * (hex - hd));
    htr[1 * 4 + 1] = hd;
    htr[5 * 4 + 1] = hex;
    float hdif = (htr[5 * 4 + 1] - htr[1 * 4 + 1]) * 0.25f;
    hdif = pymax(0.0f, hdif);
    htr[4 * 4 + 1] = htr[5 * 4 + 1] - pymin(1.0f, hdif);
    htr[2 * 4 + 1] = htr[1 * 4 + 1] + hdif;
    htr[3 * 4 + 1] = (htr[2 * 4 + 1] + htr[4 * 4 + 1]) * 0.5f;
    htr[11 * 4 + 1] = hi[1 * 6 + k];
    hdif = div32(htr[11 * 4 + 1] - htr[5 * 4 + 1], 6.0f);
    for (int ih = 6; ih <= 10; ++ih)    // DO 100
        htr[ih * 4 + 1] = htr[(ih - 1) * 4 + 1] + hdif;
    htr[17 * 4 + 1] = hi[1 * 6 + k] + yi[1 * 6 + k];
    hdif = div32(htr[17 * 4 + 1] - htr[11 * 4 + 1], 6.0f);
    for (int ih = 12; ih <= 16; ++ih)   // DO 105
        htr[ih * 4 + 1] = htr[(ih - 1) * 4 + 1] + hdif;
    htr[11 * 4 + 1] = 0.5f * (htr[10 * 4 + 1] + htr[12 * 4 + 1]);
    htr[50 * 4 + 1] = hi[3 * 6 + k];
    bool two_layers = fi[2 * 6 + k] > 0.0f;
    if (two_layers) {                   // 125
        hb2 = hi[3 * 6 + k] - yi[3 * 6 + k];
        float hb1b = hi[2 * 6 + k] - yi[2 * 6 + k] + 0.00001f;
        if (hb2 - hb1b <= 0.0f)         // 110
            two_layers = false;
    }
    if (!two_layers) {                  // 110
        htr[18 * 4 + 1] = hi[3 * 6 + k] - yi[3 * 6 + k];
        hdif = div32(htr[50 * 4 + 1] - htr[18 * 4 + 1], 32.0f);
        for (int ih = 19; ih <= 49; ++ih)  // DO 115
            htr[ih * 4 + 1] = htr[(ih - 1) * 4 + 1] + hdif;
    } else {                            // 126
        htr[18 * 4 + 1] = hi[2 * 6 + k] - yi[2 * 6 + k];
        htr[18 * 4 + 1] = pymax(htr[18 * 4 + 1], htr[17 * 4 + 1] + 1.0f);
        htr[28 * 4 + 1] = hi[2 * 6 + k];
        hdif = div32(htr[28 * 4 + 1] - htr[18 * 4 + 1], 10.0f);
        for (int ih = 19; ih <= 27; ++ih)  // DO 135
            htr[ih * 4 + 1] = htr[(ih - 1) * 4 + 1] + hdif;
        hdif = div32(htr[50 * 4 + 1] - htr[28 * 4 + 1], 22.0f);
        for (int ih = 29; ih <= 49; ++ih)  // DO 145
            htr[ih * 4 + 1] = htr[(ih - 1) * 4 + 1] + hdif;
    }

    hb1 = pymax(hi[1 * 6 + k], hi[2 * 6 + k] - yi[2 * 6 + k]);
    float hi1k = hi[1 * 6 + k], yi1k = yi[1 * 6 + k];
    float hi2k = hi[2 * 6 + k], yi2k = yi[2 * 6 + k];
    float hi3k = hi[3 * 6 + k], yi3k = yi[3 * 6 + k];
    bool has_f1 = fi[2 * 6 + k] > 0.0f;
    // DO 375 -- fifty independent heights, one per lane.  Every lane wrote
    // the whole HTR grid above, so each reads back its own store.
    for (int ih = 1 + VC_WSL(w); ih <= 50; ih += VC_WSTRIDE) {
        float fnd = 0.0f, fne = 0.0f, fn1 = 0.0f, fn2 = 0.0f, fnval = 0.0f;
        float h = htr[ih * 4 + 1];
        if (h - hlow > 0.0f) {          // 225
            if (h - hup < 0.0f)         // 235
                fnval = fup + asp * (h - hup);
        }
        if (h - hex < 0.0f) {           // 275
            fnd = fce * fsq * exp32(alp * (h - hd));
        } else if (h - hte <= 0.0f) {   // 260
            float z = (h - hi1k) / yi1k;
            fne = fce * (1.0f - z * z);
        }
        if (has_f1) {                   // 290
            if (hb1 - h <= 0.0f && h - ht1 <= 0.0f) {  // 295 -> 300
                if (lin <= 0) {         // 305
                    float z = (h - hi2k) / yi2k;
                    fn1 = fc1 * (1.0f - z * z);
                } else {                // 335
                    fn1 = s1 * (h - (hi2k - yi2k));
                }
            }
        }
        if (hb2 - h <= 0.0f) {          // 355
            float z = (h - hi3k) / yi3k;
            fn2 = fc2 * (1.0f - z * z);
        }
        fnsq[ih * 4 + 1] =
            pymax(pymax(pymax(pymax(fnd, fne), fnval), fn1), fn2);
    }
    wr_sync_(w);                         // FNSQ is lane-partitioned
}

// GENION, split at the ionogram heights.  The thirty sounding FREQUENCIES are
// a chain of running sums and stay whole; the thirty HEIGHTS are independent,
// and DO 715's true height and DO 725/740's virtual height are fused into one
// pass because neither reads the other's output.
__device__ inline void genion_warp_(LayerState* s, int k,
                                    const WarpGroup& w) {
    float* fi = s->fi;
    float* yi = s->yi;
    float* hi = s->hi;
    float* fvert = s->fvert;
    float* hprim = s->hprim;
    float* htrue = s->htrue;

    float xtail = 0.85f;
    float fex = fi[1 * 6 + k] * sqrt32(1.0f - xtail * xtail);
    fvert[1 * 6 + k] = 0.01f;
    fvert[4 * 6 + k] = fex;
    float fdif = div32(fvert[4 * 6 + k] - fvert[1 * 6 + k], 3.0f);
    fvert[2 * 6 + k] = fvert[1 * 6 + k] + fdif;
    fvert[3 * 6 + k] = fvert[2 * 6 + k] + fdif;
    fvert[9 * 6 + k] = 0.957f * fi[1 * 6 + k];
    fvert[10 * 6 + k] = 0.99f * fi[1 * 6 + k];
    fdif = div32(fvert[9 * 6 + k] - fvert[4 * 6 + k], 5.0f);
    fvert[5 * 6 + k] = fvert[4 * 6 + k] + fdif;
    fvert[6 * 6 + k] = fvert[5 * 6 + k] + fdif;
    fvert[7 * 6 + k] = fvert[6 * 6 + k] + fdif;
    fvert[8 * 6 + k] = fvert[7 * 6 + k] + fdif;
    fvert[11 * 6 + k] = 1.05f * fi[1 * 6 + k];
    fvert[30 * 6 + k] = 0.99f * fi[3 * 6 + k];
    fvert[29 * 6 + k] = 0.98f * fi[3 * 6 + k];
    fvert[28 * 6 + k] = 0.96f * fi[3 * 6 + k];
    fvert[27 * 6 + k] = 0.92f * fi[3 * 6 + k];
    if (fi[2 * 6 + k] <= 0.0f) {        // 385: no F1 layer
        fdif = div32(fvert[27 * 6 + k] - fvert[11 * 6 + k], 16.0f);
        for (int iff = 12; iff <= 26; ++iff)    // DO 395
            fvert[iff * 6 + k] = fvert[(iff - 1) * 6 + k] + fdif;
    } else {                            // 400
        fvert[20 * 6 + k] = 0.99f * fi[2 * 6 + k];
        fdif = div32(fvert[20 * 6 + k] - fvert[11 * 6 + k], 9.0f);
        for (int iff = 12; iff <= 19; ++iff)    // DO 405
            fvert[iff * 6 + k] = fvert[(iff - 1) * 6 + k] + fdif;
        fvert[21 * 6 + k] = 1.01f * fi[2 * 6 + k];
        fdif = div32(fvert[27 * 6 + k] - fvert[21 * 6 + k], 6.0f);
        for (int iff = 22; iff <= 26; ++iff)    // DO 415
            fvert[iff * 6 + k] = fvert[(iff - 1) * 6 + k] + fdif;
    }

    if (s->iedp < 0) {                  // 600: integrate at every frequency
        for (int iff = 1 + VC_WSL(w); iff <= 30;
             iff += VC_WSTRIDE) {                   // DO 640
            float hpx, htx;
            gethp(s, fvert[iff * 6 + k], &hpx, &htx);
            hprim[iff * 6 + k] = hpx;
            htrue[iff * 6 + k] = htx;
        }
        wr_sync_(w);
        return;
    }
    // 705: the fast model.  DO 710, DO 715 and DO 725/740 in one pass.
    bool no_f1 = fi[2 * 6 + k] <= 0.0f;
    for (int iff = 1 + VC_WSL(w); iff <= 30;
         iff += VC_WSTRIDE) {
        if (iff <= 10) {                                     // DO 710
            htrue[iff * 6 + k] = GENION_HTE[iff];
            hprim[iff * 6 + k] = GENION_HPE[iff];
            continue;
        }
        float fv = fvert[iff * 6 + k];
        float fn = fv * fv;                                  // DO 715
        htrue[iff * 6 + k] = xlin(fn, s->fnsq, s->htr, 50, 1, 4);
        if (no_f1) {                                         // DO 725
            hprim[iff * 6 + k] = hi[3 * 6 + k] - yi[3 * 6 + k]
                                 + bendy(s, 3, k, fv)
                                 + (pen(s, 1, k, fv) - 2.0f * yi[1 * 6 + k]);
        } else {                                             // DO 740
            float hpx, htx;
            gethp(s, fv, &hpx, &htx);
            hprim[iff * 6 + k] = hpx;
        }
    }
    wr_sync_(w);
}

// FOBBY, split at the thirty sounding frequencies.  The angle loop stays
// serial and its RCOSD is recomputed in every lane, which costs nothing in
// lockstep and saves shuffling it.
__device__ inline void fobby_warp_(const LayerState* state,
                                   const RaysView& rays, int k,
                                   const WarpGroup& w) {
    const float* fvert = state->fvert;
    const float* htrue = state->htrue;
    int nang = rv_nang(rays);
    for (int ia = 1; ia <= nang; ++ia) {        // DO 650 IA=1,NANG
        float del_ = rv_ang(rays, ia) * VC_D2R;
        float rcosd = VC_RZ * cos32(del_);
        for (int ih = 1 + VC_WSL(w); ih < 31;
             ih += VC_WSTRIDE) {               // DO 645 IH=1,30
            float fv = fvert[ih * 6 + k];
            float sphe = rcosd / (VC_RZ + htrue[ih * 6 + k]);
            float sqcos = 1.0f - sphe * sphe;
            sqcos = pymax(0.000001f, sqcos);
            float cphe = sqrt32(sqcos);
            float freq = fv / cphe;
            rv_ifob(rays, ia, ih, k) = (int)(1000.0f * freq);
        }
    }
    wr_sync_(w);
}

// SETLNG: the scalar replication runs in every lane, the 40x30 IFOB block and
// the thirty ionogram rows split.
__device__ inline void setlng_iono_warp_(LayerState* s, const RaysView& rays,
                                         int a, int src,
                                         const WarpGroup& w) {
    for (int il = 1 + VC_WSL(w); il <= 30; il += VC_WSTRIDE) {
        s->hprim[il * 6 + a] = s->hprim[il * 6 + src];
        s->htrue[il * 6 + a] = s->htrue[il * 6 + src];
        s->fvert[il * 6 + a] = s->fvert[il * 6 + src];
        s->afac[il * 6 + a]  = s->afac[il * 6 + src];
    }
    for (int ia = 1; ia <= 40; ++ia)
        for (int il = 1 + VC_WSL(w); il <= 30; il += VC_WSTRIDE)
            rv_ifob(rays, ia, il, a) = rv_ifob(rays, ia, il, src);
}

__device__ inline void setlng_warp_(LayerState* s, const RaysView& rays,
                                    const WarpGroup& w) {
    if (s->kfx >= 3)
        return;
    if (s->kfx <= 1) {
        for (int a = 2; a <= 5; ++a)            // 110
            setlng_area_(s, a, 1);
        for (int a = 2; a <= 3; ++a)
            setlng_iono_warp_(s, rays, a, 1, w);
        wr_sync_(w);
        return;
    }
    for (int a = 4; a <= 5; ++a)                // 135
        setlng_area_(s, a, 3);
    setlng_iono_warp_(s, rays, 3, 2, w);
    wr_sync_(w);
}

// SIGDIS, split over the KFX sample areas.
//
// The six path sums are the reason this needs care: a warp reduction would
// re-associate them and stop being bit-exact.  Lane k-1 computes area k's
// ABIY and SYSSY row in parallel, and the sums are then re-added IN AREA
// ORDER out of __shfl_sync -- the Fortran's own left-to-right accumulation,
// just with the expensive part already done.  ABIY/ARTIC are broadcast the
// same way and stored by every lane, so no fence is needed.
__device__ inline void sigdis_warp_(LayerState* state, const MufState* muf,
                                    const float* sys, const float* perr,
                                    float f2d111, SigdState* sigd,
                                    TonState* ton, float gcdkm, int jmode,
                                    const WarpGroup& w) {
    const float* fi = state->fi;
    float glav = 0.0f;
    int kfxx = state->kfx;
    float xkf = (float)kfxx;
    float xadj = 0.0f, xsu = 0.0f, xsl = 0.0f;
    float xfmp = 0.0f, xsup = 0.0f, xslp = 0.0f;
    float ac = 0.0f;
    float feav = 0.0f;

    float l_abiy = 0.0f, l_adj = 0.0f, l_su = 0.0f, l_sl = 0.0f;
    float l_fmp = 0.0f, l_sup = 0.0f, l_slp = 0.0f;
    if (w.g < kfxx) {
        int k = w.g + 1;
        float abiy = -0.04f + exp32(-2.937f + 0.8445f * fi[1 * 6 + k]);
        l_abiy = pymax(0.1f, abiy);
        int idp = (gcdkm > 2500.0f) ? 5 : 2;
        syssy(sys, perr, f2d111, state->glat[k], state->clck[k], idp,
              &l_adj, &l_su, &l_sl, &l_fmp, &l_sup, &l_slp);
    }
    __syncwarp(w.mask);
    for (int k = 1; k <= kfxx; ++k) {
        int src = w.base + k - 1;
        float abiy = __shfl_sync(w.mask, l_abiy, src);
        float adj = __shfl_sync(w.mask, l_adj, src);
        state->abiy[k] = abiy;
        state->artic[k] = adj;
        xadj = xadj + adj;
        xsu = xsu + __shfl_sync(w.mask, l_su, src);
        xsl = xsl + __shfl_sync(w.mask, l_sl, src);
        xfmp = xfmp + __shfl_sync(w.mask, l_fmp, src);
        xsup = xsup + __shfl_sync(w.mask, l_sup, src);
        xslp = xslp + __shfl_sync(w.mask, l_slp, src);
        feav = feav + fi[1 * 6 + k];
        ac = ac + abiy;
        glav = glav + fabsf(state->glat[k]);
    }
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
    sigd->htloss = 88.0f;
    float xve = xlin(90.0f, state->htrue, state->fvert, 30, k, 6);
    sigd->xve = xve / fi[1 * 6 + k];
    sigd->xnuz = 63.07f;
    sigd->hnu = 4.39f;
    if (feav - 2.0f > 0.0f) {           // 120
        sigd->afe = 1.359f;
        sigd->bfe = 8.617f;
    } else if (feav - 0.5f <= 0.0f) {   // 130
        sigd->afe = 0.0f;
        sigd->bfe = 0.0f;
    } else {                            // 135
        sigd->afe = div32(1.359f * (feav - 0.5f), 1.5f);
        sigd->bfe = div32(8.617f * (feav - 0.5f), 1.5f);
    }
    glav = glav * VC_R2D;
    float ftab;
    if (glav - 40.0f <= 0.0f) {         // 145
        ftab = muf->yfot[3];
    } else if (glav - 50.0f <= 0.0f) {  // 155
        ftab = muf->yfot[3];
        ftab = ftab - div32((glav - 40.0f) * (ftab - 10.0f), 10.0f);
    } else {                            // 160
        ftab = 10.0f;
    }
    float eslsm = 0.0f;
    float pes, pf2, dummy;
    if (muf->ymuf[4] > 0.0f) {
        dummy = muf->ymuf[4];
        pes = prbmuf(muf, ftab, dummy, dummy, 4);
        pes = pymin(pymax(pes, 0.1f), 0.9f);
        eslsm = -10.0f * log10_32(1.0f - pes);
    }
    dummy = muf->ymuf[3];
    pf2 = prbmuf(muf, ftab, dummy, dummy, 3);
    pf2 = pymax(pf2, 0.1f);
    float f2lsm = -10.0f * log10_32(pf2);
    float asm_v = ton->adj - eslsm - f2lsm;
    sigd->asm_ = pymax(asm_v, 0.0f);
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

// ANOIS1's two NOISY calls as ONE.  Both KJ and JK land in 1..6, so the two
// evaluations take the identical code path through NOISY and differ only in
// which coefficient column they read: give lane 1 the JK column and the warp
// computes both in the time of one, with no divergence at all.
__device__ inline void anois1_warp_(AnoisState* anois, const float* fakp,
                                    const float* fakmap, const float* hmym,
                                    const float* fakabp, const float* abmap,
                                    float f2d111, float gmtr, float rlong,
                                    float rlongd, float rlat,
                                    const WarpGroup& w) {
    if (f2d111 <= 0.0f) {               // 90: no data base
        anois->atnu = 0.0f;
        anois->atny = 0.0f;
        return;
    }
    float cc = gmtr;
    anois->cc = cc;
    int kj = 6;
    if (cc - 22.0f < 0.0f)              // 105
        kj = (int)(div32(cc, 4.0f) + 1.0f);
    float tm = (float)(4 * kj - 2);
    anois->tm = tm;
    int jk;
    if (cc - tm < 0.0f)                 // 115
        jk = kj - 1;
    else if (cc - tm == 0.0f)           // 120
        jk = kj;
    else                                // 125
        jk = kj + 1;
    if (jk <= 0)                        // 135
        jk = 6;
    else if (jk - 6 > 0)                // 145
        jk = 1;
    anois->kj = kj;
    anois->jk = jk;
    float ceg;
    if (rlong < 0.0f)                   // 155
        ceg = 360.0f + rlong * VC_R2D;
    else
        ceg = rlongd;
    float xla = rlat * VC_R2D;
    float v = noisy(fakp, fakmap, hmym, fakabp, abmap,
                    (w.g == 1) ? jk : kj, xla, ceg);
    anois->atnu = __shfl_sync(w.mask, v, w.base);
    anois->atny = __shfl_sync(w.mask, v, w.base + 1);
}

// Labels 83-85, on the whole warp.  The control flow is identical to
// luffy.cuh's and warp-uniform, so K and XMDL come back in every lane and
// nothing has to be broadcast.
__device__ inline void luffy_select_area_warp_(CircuitState* c,
                                               const RaysView& rays, int ipfg,
                                               int* k_out, int* xmdl_out,
                                               const WarpGroup& w) {
    int k;
    int xmdl;
    bool goto_85;
    if (ipfg == 100 || ipfg == 300) {
        c->jlong = -1;
        xmdl = VC_MDL_S;
        k = c->frq.jmode;
        goto_85 = false;
    } else {
        c->jlong = 1;
        xmdl = VC_MDL_L;
        k = 1;
        c->frq.itxrcp[1] = 1;
        goto_85 = (c->mspec == 121 && c->frq.jmode == k);
    }

    while (true) {
        if (!goto_85) {
            lecden_warp_(&c->layers, k, w);
            genion_warp_(&c->layers, k, w);
            fobby_warp_(&c->layers, rays, k, w);
            alosfv(&c->layers, &c->muf, k);
            if (ipfg == 100 || k > 1) {
                *k_out = k; *xmdl_out = xmdl; return;
            }
        }
        goto_85 = false;
        k = c->layers.kfx;              // label 85
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
                                            const WarpGroup& w, int ipflag,
                                            int k, int ihmin, float pluf,
                                            int nfx, int* found_ifx) {
    const int lane = w.g;                   // the lane WITHIN the group
    const int ifx = lane + 1;
    const int act = (lane < nfx);
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
            ip[i] = wr_scan_(v, (unsigned)v != VC_WARP_POISON, ia[i], w);
        }
#define VC_WR_CARRY(F)                                                        \
        for (int i = 0; i < 4; ++i) {                                         \
            int v = *(int*)&L->reflx.F[i];                                    \
            *(int*)&L->reflx.F[i] = wr_scan_(v,                               \
                    (unsigned)v != VC_WARP_POISON, *(int*)&c->reflx.F[i],     \
                    w);                                                       \
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
                                           e10, w));
        don.d50r = __int_as_float(wr_scan_(__float_as_int(don.d50r), have,
                                           e50, w));
        don.d90r = __int_as_float(wr_scan_(__float_as_int(don.d90r), have,
                                           e90, w));
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
    unsigned badm = wr_ballot_(w, act && st != 0);
    unsigned fndm = wr_ballot_(w, act && found);
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
    __syncwarp(w.mask);

    // ---- commit -----------------------------------------------------------
    int keep = act && (lane <= cut);
#define VC_WR_MERGE_ONE(S) wr_merge_((int*)&c->S, (const int*)&L->S,          \
                                     (int)(sizeof(L->S) / 4), keep, w);
    VC_WR_MERGE_ONE(reflx)
    VC_WR_MERGE_ONE(modes)
    VC_WR_MERGE_ONE(zon)
    VC_WR_MERGE_ONE(allm)
    VC_WR_MERGE_ONE(losx)
#undef VC_WR_MERGE_ONE
    wr_merge_((int*)c->lpath.txrgml, (const int*)L->lpath.txrgml,
              VC_WR_TXRGML_WORDS, keep, w);

    int h1 = wr_last_(keep && reach >= VC_WR_ALLM, w);
    int h2 = wr_last_(keep && reach >= VC_WR_NOISE, w);
    int h4 = wr_last_(keep && reach >= VC_WR_MODEL, w);
    int h6 = wr_last_(keep && reach >= VC_WR_RELBIL, w);
    int h7 = wr_last_(keep && reach >= VC_WR_SERPRB, w);
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
    __syncwarp(w.mask);

    // ---- the loop's own answer -------------------------------------------
    int out_st = 0;
    if (firstbad <= cut)
        out_st = __shfl_sync(w.mask, st, w.base + firstbad);
    *found_ifx = (firstbad > firstfnd && firstfnd <= cut) ? firstfnd + 1 : 0;
    return out_st;
}

// ---------------------------------------------------------------------------
// LUFFY's body, one lane group.  Setup across the group, the frequency loop
// one lane per frequency, label 165 / label 300 / the blend back on the
// group's lane 0.
// ---------------------------------------------------------------------------
__device__ inline int luffy_body_warp_(CircuitState* c,
                                       const CircuitTables* t,
                                       const RaysView& rays, LaneState* L,
                                       SmoothStore* store,
                                       const WarpGroup& w, int ipflag,
                                       int nfx) {
    smoothstore_init_warp_(store, w);

    int ipfg = ipflag;
    float pluf = 0.0f;
    int st = 0;
    // Every lane, redundantly: identical values, identical stores, so IPFG,
    // PLUF and the status are already in every lane and nothing is broadcast.
    if (c->mspec == 121 && c->gcdkm > 10000.0f) {
        ipfg = 200;                     // long-path SETUP only
        c->mspec = 0;                   // beyond the smoothing distance
    }
    if (ipfg > 200) {
        // .....FREQUENCY COMPLEMENT, and the required reliability
        st = frqcom(&c->muf, c->frq.frea, 0, c->time.it);
        pluf = 0.01f * (float)c->lufp;
    }
    if (st)
        return st;
    // .....SELECT CONTROLLING SAMPLE AREA
    c->frq.jmode = selmod(&c->layers);
    // .....GET NOISE VALUE AT 1 MHZ FOR RCVR SITE
    anois1_warp_(&c->anois, t->fakp, t->fakmap, t->hmym, t->fakabp, t->abmap,
                 t->f2d111, c->time.gmtr, c->rlong, c->rlongd, c->rlat, w);

    // Label 83.  The smoothing recycle comes back here, NOT to the top.
    while (true) {
        int k = 0, xmdl = 0;
        luffy_select_area_warp_(c, rays, ipfg, &k, &xmdl, w);    // 83-85
        // .....CAN RUN FOR SHORT PATHS ALSO, SO SET ALL SAMPLE AREAS
        setlng_warp_(&c->layers, rays, w);
        // .....SET ABSORPTION LOSS PARAMETERS, ADJUST SIGNAL DISTRIBUTION
        // .....TABLES
        sigdis_warp_(&c->layers, &c->muf, t->sys, t->perr, t->f2d111,
                     &c->sigd, &c->ton, c->gcdkm, c->frq.jmode, w);
        // .....START FREQUENCY LOOP
        c->son.nhp[VC_LUFFY_ILUF] = -1;
        c->son.mdl[VC_LUFFY_ILUF] = xmdl;
        for (int ifx = 1; ifx <= VC_LUFFY_NFREQ; ++ifx)
            c->son.mdl[ifx] = VC_MDL_BLANK;
        int ihmin = luffy_min_hops_(c->muf.nhopmf);     // IHMX here, and dead

        int found_ifx = 0;
        st = luffy_freq_loop_warp_(c, t, rays, L, store, w, ipflag, k,
                                   ihmin, pluf, nfx, &found_ifx);
        if (st)
            return st;
        // the twelve lanes' /SON/ columns and smoothing-store rows, before
        // lane 0 reads all of them
        wr_sync_(w);

        int again = 0;
        if (w.g == 0) {
            if (found_ifx)
                luffy_luf_found_(c, found_ifx, pluf);    // label 165
            else
                st = luffy_after_loop_(c, t, ipflag, store, &again);
        }
        __threadfence_block();
        __syncwarp(w.mask);
        st = __shfl_sync(w.mask, st, w.base);
        again = __shfl_sync(w.mask, again, w.base);
        if (st || found_ifx || !again)
            return st;
        // .....DO LONG AND SHORT PATH MODEL: round two, the other model.
        ipflag = 200;
        ipfg = 200;
    }
}

// ---------------------------------------------------------------------------
// LUFFY, one warp.  Same contract as `luffy`, same tables, same status codes;
// `lane` is threadIdx.x & 31 and `store` is this LANE GROUP's slice of
// dynamic shared memory (at VC_WARP_GROUPS = 1, the warp's).
//
// `nfx` is the DO-265 trip count and exists only to be measured with: it
// defaults to the twelve frequencies, and setting it to 0 leaves the setup
// phase and the collective scaffolding running with the register allocation
// held identical, which is how the setup share in the header was taken.
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
                                 int iono_idx, int ant_idx, int ipflag,
                                 int nfx = VC_LUFFY_NFREQ) {
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
    WarpGroup w = wr_group_(lane);
    int st = luffy_body_warp_(c, &t, rays, L, store, w, ipflag, nfx);
    __threadfence_block();
    __syncwarp(w.mask);
    if (w.g == 0)
        c->mspec = mspec_temp;              // Stage A's try/finally
    return st;
}

#endif  // VOACAP_LUFFY_WARP_CUH
