// Optional per-phase clock64() instrumentation of the thread LUFFY kernel.
//
// OFF BY DEFAULT.  Without -DVOACAP_LUFFY_PROFILE every macro below expands
// to nothing, so the production kernel's preprocessed source -- and therefore
// its binary and its answers -- is exactly what it was without this header.
// tools/kernel_perf/phase_profile.py compiles the profiled build, points
// `vc_prof_buf` at a device buffer and reads the counters back.
//
// What is counted, per THREAD (so the buffer is laid out [counter][thread]
// and every update is a plain, coalesced, non-atomic read-modify-write):
//
//   [ph]            clock64() cycles between VC_PROF_BEGIN(ph) and
//                   VC_PROF_END(ph).  This is wall time on the SM as the
//                   thread sees it: stalls, and the other lanes' divergent
//                   work executed while this lane waits, are included.  That
//                   is the cost the phase actually imposes on its warp.
//   [NPH + ph]      __popc(__activemask()) at each entry -- the lanes that
//                   arrive at the phase together
//   [2 * NPH + ph]  the number of entries
//
// so sum([NPH+ph]) / (32 * sum([2*NPH+ph])) is the mean active-lane fraction
// at entry, as seen by the threads entering.  Phases nest (TOTAL contains
// everything, HOPS contains FDIST/INMUF/ALLM), so shares are computed against
// TOTAL, never by summing.
//
// Only the strict thread kernel (luffy.cuh) is instrumented; luffy_warp.cuh
// includes luffy.cuh, but the profiled build is never used for it.

#ifndef VOACAP_LUFFYPROF_CUH
#define VOACAP_LUFFYPROF_CUH

#define VC_PROF_AREA    0   // labels 83-85 + SETLNG + SIGDIS (+SELMOD/ANOIS1)
#define VC_PROF_NOISE1  1   // ALLMODES(0) + the first GENOIS
#define VC_PROF_FINDF   2   // short path: FINDF (ray set table)
#define VC_PROF_FDIST   3   // short path: FDIST, all hops
#define VC_PROF_INMUF   4   // short path: INMUF (REGMOD: loss), all hops
#define VC_PROF_ALLM    5   // short path: ALLMODES accumulate, all hops
#define VC_PROF_ES      6   // short path: ESMOD + ESREG + ALLMODES(2)
#define VC_PROF_LONG    7   // long path: FINDF x2, GMLOSS/SETTXR, SELTXR, LNGPAT
#define VC_PROF_NOISE2  8   // the second GENOIS
#define VC_PROF_RELBIL  9   // RELBIL
#define VC_PROF_POST    10  // SERPRB + MPATH / receiver end + smoothing save
#define VC_PROF_AFTER   11  // after the loop: LUF / blend (smoothing)
#define VC_PROF_HOPS    12  // short path: the whole hop loop (3+4+5)
#define VC_PROF_TOTAL   13  // the whole body
// sub-phases of AREA and LONG
#define VC_PROF_LECDEN  14
#define VC_PROF_GENION  15
#define VC_PROF_FOBBY   16
#define VC_PROF_ALOSFV  17
#define VC_PROF_SETLNG  18
#define VC_PROF_SIGDIS  19
#define VC_PROF_LFINDF  20  // long path: both FINDFs
#define VC_PROF_GMLOSS  21  // long path: GMLOSS (SETTXR prices 45 rays/end)
#define VC_PROF_SELTXR  22
#define VC_PROF_LNGPAT  23
#define VC_PROF_KERNEL  24  // luffy_batch's whole thread, state copies included
#define VC_PROF_NPH     25

#ifdef VOACAP_LUFFY_PROFILE

__device__ unsigned long long* vc_prof_buf = nullptr;
__device__ int vc_prof_n = 0;

__device__ __forceinline__ long long vc_prof_begin_(int ph) {
    unsigned m = __activemask();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long* b = vc_prof_buf;
    int n = vc_prof_n;
    if (b && tid < n) {
        b[(long)(VC_PROF_NPH + ph) * n + tid] += (unsigned long long)__popc(m);
        b[(long)(2 * VC_PROF_NPH + ph) * n + tid] += 1ull;
    }
    return clock64();
}

__device__ __forceinline__ void vc_prof_end_(int ph, long long t0) {
    long long dt = clock64() - t0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long* b = vc_prof_buf;
    int n = vc_prof_n;
    if (b && tid < n)
        b[(long)ph * n + tid] += (unsigned long long)dt;
}

#define VC_PROF_BEGIN(ph) long long vc_prof_t0_##ph = vc_prof_begin_(ph)
#define VC_PROF_END(ph)   vc_prof_end_(ph, vc_prof_t0_##ph)

#else

#define VC_PROF_BEGIN(ph)
#define VC_PROF_END(ph)

#endif  // VOACAP_LUFFY_PROFILE

#endif  // VOACAP_LUFFYPROF_CUH
