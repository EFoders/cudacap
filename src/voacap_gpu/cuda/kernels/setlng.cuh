// SETLNG — replicate the computed sample area across the array slots; the
// CUDA C twin of scalar/setlng.py.  Pure array replication, no arithmetic:
//
//   KFX >= 3   all slots already populated; do nothing
//   KFX <= 1   copy area 1's per-area scalars + FS/FI/YI/HI into areas 2..5,
//              and its ionogram (HPRIM/HTRUE/FVERT/AFAC/IFOB) into 2..3
//   KFX == 2   copy area 3 into areas 4..5, and area 2's ionogram into 3
//
// The deliberately asymmetric index ranges are the whole content of the
// routine and are preserved exactly.

#ifndef VOACAP_SETLNG_CUH
#define VOACAP_SETLNG_CUH

#include "raysview.cuh"
#include "state.cuh"

// One destination area's scalar + layer-parameter copy (areas src -> a).
__device__ inline void setlng_area_(LayerState* s, int a, int src) {
    s->clat[a]   = s->clat[src];
    s->clong[a]  = s->clong[src];
    s->glat[a]   = s->glat[src];
    s->rd[a]     = s->rd[src];
    s->gyz[a]    = s->gyz[src];
    s->rat[a]    = s->rat[src];
    s->gmdip[a]  = s->gmdip[src];
    s->clck[a]   = s->clck[src];
    s->abiy[a]   = s->abiy[src];
    s->artic[a]  = s->artic[src];
    s->sigpat[a] = s->sigpat[src];
    s->epspat[a] = s->epspat[src];
    s->hs[a]     = s->hs[src];
    for (int il = 1; il <= 3; ++il) {       // FS/FI/YI/HI rows 1..3, (4, 6)
        s->fs[il * 6 + a] = s->fs[il * 6 + src];
        s->fi[il * 6 + a] = s->fi[il * 6 + src];
        s->yi[il * 6 + a] = s->yi[il * 6 + src];
        s->hi[il * 6 + a] = s->hi[il * 6 + src];
    }
}

// One destination area's ionogram copy (rows 1..30 of the (31, 6) arrays,
// and IFOB's (41, 31, 6) block rows 1..40 x 1..30).
__device__ inline void setlng_iono_(LayerState* s, const RaysView& rays,
                                    int a, int src) {
    for (int il = 1; il <= 30; ++il) {
        s->hprim[il * 6 + a] = s->hprim[il * 6 + src];
        s->htrue[il * 6 + a] = s->htrue[il * 6 + src];
        s->fvert[il * 6 + a] = s->fvert[il * 6 + src];
        s->afac[il * 6 + a]  = s->afac[il * 6 + src];
    }
    for (int ia = 1; ia <= 40; ++ia)
        for (int il = 1; il <= 30; ++il)
            rv_ifob(rays, ia, il, a) = rv_ifob(rays, ia, il, src);
}

__device__ inline void setlng(LayerState* s, const RaysView& rays) {
    // IF(KFX-3) 105,100,100  — KFX>=3: everything already filled.
    if (s->kfx >= 3)
        return;
    // 105  IF(KFX-1) 110,110,135
    if (s->kfx <= 1) {
        // 110  ONE SAMPLE AREA: area 1 -> areas 2..5 (scalars + FS/FI/YI/HI)
        for (int a = 2; a <= 5; ++a)
            setlng_area_(s, a, 1);
        // ionogram: area 1 -> areas 2..3 only
        for (int a = 2; a <= 3; ++a)
            setlng_iono_(s, rays, a, 1);
        return;
    }
    // 135  THREE SAMPLE AREAS: area 3 -> areas 4..5 (scalars + FS/FI/YI/HI)
    for (int a = 4; a <= 5; ++a)
        setlng_area_(s, a, 3);
    // ionogram: area 2 -> area 3
    setlng_iono_(s, rays, 3, 2);
}

#endif  // VOACAP_SETLNG_CUH
