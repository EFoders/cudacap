// SANG — restrict the take-off-angle scan to the entries a path of this
// length can actually use; the CUDA C twin of scalar/sang.py
// (reference/voacapl/src/voacapw/sang.for).
//
// Two steps and no arithmetic worth the name.  A distance bucket
// (ID = GCDKM/2000. + 1., clamped at 8) selects NANG from the NANGX DATA
// table: the longer the path, the shallower the steepest ray that still
// closes, so the scan is cut from the top.  Then, if the user asked for a
// minimum take-off angle, the scan is WIDENED again -- SANG walks ANG from
// the bottom until it finds the first entry that clears AMIND + 8.5 degrees,
// and takes that index as the new count.  The two margins are different
// numbers (10.0 on the test, 8.5 on the walk) and both are load-bearing:
// the test asks "is the distance-limited top already a comfortable 10
// degrees above the floor?", the walk then places the top 8.5 degrees above
// it.  Transcribed as written.
//
// THE RESULT IS AN INDEX, NOT AN ANGLE.  NANG counts usable ANG entries,
// 1-based in the Fortran sense: ANG(1..NANG) with ANG 0-based here, so the
// tests read SANG_ANG[NANG - 1].  The 40-entry ANG table is blkdat DATA and
// is reproduced below; a C float literal rounds its decimal to nearest-even
// binary32 exactly as np.float32 does, so 89.99f is Stage A's 89.99.
//
// ONE div32 SITE: GCDKM / 2000. has a compile-time-constant denominator, and
// NVRTC would strength-reduce it to a multiply by the reciprocal.  That is a
// 1-ULP error feeding a TRUNCATION, so it does not stay 1 ULP: a case sitting
// exactly on a bucket edge (the oracle grid has 2000.0, 4000.0, 16000.0)
// lands in the wrong bucket and NANG jumps by five.
//
// SANG is a pure function of (GCDKM, AMIND).  Stage A returns NANG and its
// caller (engine.py:382) does `circ.rays.nang = sang(...)`; `sang_rays` below
// is that assignment, plus the blkdat ANG initialisation that Stage A carries
// as RaysState's dataclass default (scalar/raysstate.py) rather than in SANG.

#ifndef VOACAP_SANG_CUH
#define VOACAP_SANG_CUH

#include "f32math.cuh"
#include "raysview.cuh"

// ANG, blkdat.for — 0-BASED length 40, exactly as scalar/sang.py's ANG.
// (COMMON /RAYS/ stores the same numbers 1-based in ANG(1..40); see
// sang_rays below, and state.cuh's `float ang[41]`.)
__device__ __constant__ float SANG_ANG[40] = {
     0.0f,  0.5f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  8.0f, 10.0f,
    12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 34.0f, 36.0f, 38.0f, 40.0f, 42.0f, 44.0f, 46.0f, 48.0f, 50.0f,
    52.0f, 54.0f, 56.0f, 60.0f, 65.0f, 70.0f, 75.0f, 80.0f, 85.0f, 89.99f,
};

// NANGX, blkdat.for — the distance-bucket ray counts, 0-based (bucket ID
// reads NANGX[ID - 1], the Fortran NANGX(ID)).
__device__ __constant__ int SANG_NANGX[8] = {40, 34, 29, 24, 19, 14, 12, 9};

// SANG(GCDKM, AMIND) -> NANG, the number of usable ANG entries.
// gcdkm: great-circle distance, km; amind: user minimum take-off angle,
// degrees (<= 0 means "no minimum", the 120 arm).
__device__ inline int sang(float gcdkm, float amind) {
    // ID = GCDKM/2000. + 1. — a REAL*4 expression truncated to INTEGER.
    // div32: constant denominator, and the truncation amplifies 1 ULP into a
    // whole bucket.
    int id_ = (int)(div32(gcdkm, 2000.0f) + 1.0f);
    id_ = (8 < id_) ? 8 : id_;                  // Python min(id_, 8)
    int iang = SANG_NANGX[id_ - 1];
    if (amind > 0.0f) {                         // IF(AMIND) 120,120,100
        if (SANG_ANG[iang - 1] - 10.0f - amind < 0.0f) {   // 100 -> 105
            for (int ia = 1; ia <= 40; ++ia) {  // DO 115
                iang = ia;
                // 110 -> 120: first entry 8.5 deg clear of the floor
                if (SANG_ANG[iang - 1] - 8.5f - amind >= 0.0f)
                    break;
            }
        }
    }
    return iang;
}

// SANG as its caller uses it: NANG into COMMON /RAYS/, and ANG(1..40) filled
// from the blkdat table.  Stage A does the first in engine.py and the second
// in RaysState's field default; both land in the same struct, so both are
// here.  ANG(0) is the 1-based padding slot and stays zero, exactly as
// scalar/raysstate.py's np.zeros(41) leaves it.
__device__ inline int sang_rays(const RaysView& rays, float gcdkm,
                                float amind) {
    rv_ang(rays, 0) = 0.0f;
    for (int i = 1; i <= 40; ++i)
        rv_ang(rays, i) = SANG_ANG[i - 1];
    int nang = sang(gcdkm, amind);
    rv_nang(rays) = nang;
    return nang;
}

#endif  // VOACAP_SANG_CUH
