// MPATH — multipath probability; the CUDA C twin of stage_a/mpath.py.
// Given the most reliable mode (NREL, chosen by RELBIL), find the highest
// reliability among the OTHER modes that (a) exist (HP > 0), (b) arrive
// MORE than DMP milliseconds away — the delay test reads backwards from the
// Fortran header comment, and the code is what is ported — and (c) sit
// within PMP dB of the MRM's signal power.  0.001 if none qualify, or if
// multipath was not requested (DMP or PMP non-positive), or if METHOD 20 /
// MSPEC 121 beyond 7000 km.  Writes /SON/ PROBMP(ifx) only.

#ifndef VOACAP_MPATH_CUH
#define VOACAP_MPATH_CUH

#include "f32math.cuh"
#include "state.cuh"

__device__ inline void mpath(const AllModeState* allm, SonState* son,
                             int ifx, float dmp, float pmp, float gcdkm,
                             int method, int mspec) {
    son->probmp[ifx] = 0.001f;
    if (dmp <= 0.0f || pmp <= 0.0f)
        return;                             // no multipath requested
    // METHOD 30's area variant is not valid beyond 7000 km
    if (method == 20 && mspec == 121 && gcdkm > 7000.0f)
        return;
    int nrel = allm->nrel;
    float sig_power = allm->sigpow[nrel];
    float sig_power_limit = sig_power - pmp;
    float ttim = allm->timed[nrel];         // time delay of the MRM

    for (int im = 1; im <= allm->nmmod; ++im) {     // DO 135
        if (im == nrel)
            continue;                       // RELBIL already found this one
        if (allm->hp[im] <= 0.0f)
            continue;                       // this mode does not exist
        if (fabsf(allm->timed[im] - ttim) <= dmp)
            continue;                       // arrives close enough to just add
        if (allm->sigpow[im] < sig_power_limit)
            continue;                       // too weak to interfere
        son->probmp[ifx] = pymax(son->probmp[ifx], allm->rely[im]);
    }
}

#endif  // VOACAP_MPATH_CUH
