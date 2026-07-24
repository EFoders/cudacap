// RaysView — COMMON /RAYS/ addressed through a stride, so the biggest block
// of per-thread state can live STRUCTURE-OF-ARRAYS instead of inside
// CircuitState.
//
// WHY.  Stage C keeps one whole circuit-hour per thread, and CircuitState was
// carried array-of-structures: thread i's state is `circuits[i]`, so adjacent
// lanes of a warp are sizeof(CircuitState) bytes apart.  Every warp-wide read
// of one float then costs 32 separate memory transactions.  Measured at that
// stride, AoS sustains 34 GB/s of useful bandwidth against 2,892 GB/s for the
// same access laid out SoA.  RaysState is 30,672 of the 55,652 bytes, so it
// goes first.
//
// WHAT.  The view is a HANDLE, not storage: three base pointers, the stride
// `n` (the number of threads the buffers were laid out for) and this thread's
// index `t`.  A logical element at flat offset `o` lives at `o * n + t`:
//
//     nang        offset 0                       -> nang[t]
//     ang[i]      offset i,          0..40       -> ang[i * n + t]
//     ifob[i,j,k] offset (i*31+j)*6+k            -> ifob[off * n + t]
//
// `rays_view_of(RaysState*)` returns the n=1, t=0 view, which is EXACTLY the
// old packed-struct layout — that is what lets every per-routine test keep its
// existing packed RaysState and its existing bit-exact expectations, and so
// what proves the refactor is inert.
//
// CONSTNESS.  The accessors return references and take `const RaysView&`,
// because const on the handle says nothing about the pointee.  Read-only
// callees still declare `const RaysView&` as documentation, exactly as they
// declared `const RaysState*` before; the compiler no longer enforces it.
//
// This header deliberately contains NO arithmetic.  It is an indexing change
// and nothing else: every value a routine computes must be bit-identical
// before and after.  It is, and the 504-circuit-hour replay proves it at
// 7,012,152 of 7,012,152 cells.
//
// WHAT IT WAS WORTH — 1.03x, and that is a RESULT, not a disappointment.
// Because the view carries (n, t), the two layouts are one A/B away from each
// other with the kernel code held byte-identical: n = nthreads, t = tid is
// SoA, and n = 1, t = 0 over a per-thread base pointer is exactly the old
// AoS.  Measured that way at 32,256 threads, 129.1 ms -> 124.9 ms, and the
// same 1.03x at 504, 2,016 and 8,064 threads.  The reason is that the kernel
// is not bandwidth-bound: kernel time goes 63 ms -> 127 ms for a 64x increase
// in threads, so it is hiding a long per-thread dependency chain, not waiting
// on DRAM.  Coalescing was the wrong lever.  See tools/bench_stage_c.py.
// Do NOT convert the remaining structs expecting a different answer.

#ifndef VOACAP_RAYSVIEW_CUH
#define VOACAP_RAYSVIEW_CUH

#include "state.cuh"

// IFOB is (41, 31, 6) row-major, as state.cuh documents.
#define VC_RAYS_NIH   31
#define VC_RAYS_NAREA 6
#define VC_RAYS_NANG  41
#define VC_RAYS_IFOB_SIZE (VC_RAYS_NANG * VC_RAYS_NIH * VC_RAYS_NAREA)

struct RaysView {
    int*   nang;                // [t]
    float* ang;                 // [i * n + t]
    int*   ifob;                // [((i * 31 + j) * 6 + k) * n + t]
    int    n;                   // stride: threads the buffers were laid for
    int    t;                   // this thread's lane
};

__device__ __forceinline__ int& rv_nang(const RaysView& v) {
    return v.nang[v.t];
}

__device__ __forceinline__ float& rv_ang(const RaysView& v, int i) {
    return v.ang[i * v.n + v.t];
}

__device__ __forceinline__ int& rv_ifob(const RaysView& v, int i, int j,
                                        int k) {
    return v.ifob[((i * VC_RAYS_NIH + j) * VC_RAYS_NAREA + k) * v.n + v.t];
}

// IFOB by the flat (i * 31 + j) row index the callers already carry, for the
// sites that had precomputed it.
__device__ __forceinline__ int& rv_ifob_row(const RaysView& v, int row,
                                            int k) {
    return v.ifob[(row * VC_RAYS_NAREA + k) * v.n + v.t];
}

// A view over ONE packed RaysState: stride 1, lane 0.  Byte-for-byte the
// layout every RaysState test already builds and every RaysState assertion
// already checks.
__device__ __forceinline__ RaysView rays_view_of(RaysState* s) {
    RaysView v;
    v.nang = &s->nang;
    v.ang  = s->ang;
    v.ifob = s->ifob;
    v.n = 1;
    v.t = 0;
    return v;
}

// Same, for a read-only struct.  The view's accessors hand back references,
// so the const is dropped here; callees taking `const RaysView&` do not write.
__device__ __forceinline__ RaysView rays_view_of(const RaysState* s) {
    return rays_view_of(const_cast<RaysState*>(s));
}

// A view over the SoA batch buffers: `n` threads' worth, this thread at `t`.
__device__ __forceinline__ RaysView rays_view_soa(int* nang, float* ang,
                                                  int* ifob, int n, int t) {
    RaysView v;
    v.nang = nang;
    v.ang  = ang;
    v.ifob = ifob;
    v.n = n;
    v.t = t;
    return v;
}

#endif  // VOACAP_RAYSVIEW_CUH
