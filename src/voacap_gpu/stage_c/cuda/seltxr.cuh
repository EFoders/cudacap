// SELTXR — choose one ray at each long-path end; the CUDA C twin of
// stage_a/seltxr.py (SELTXR_ORIG; the vendored version string dispatches to
// the original, not the Alex variant).  Three tie-breaks: smallest gain
// minus loss within GMIN, then closest to specular reflection within YMIN,
// then closest to the a-priori optimum take-off angle DELOPT.
//
// The Stage A findings are all preserved verbatim:
//   * XHPM and IHOP are computed for ray 1 at label 105 and NOT recomputed
//     by the find-first search, so whatever ray it lands on is priced with
//     ray 1's hop count;
//   * INFO bit 4 runs the selection twice, and the second pass resets
//     LTXRGM to IDNXS but not XHPM/YHPM — the diagnostic changes the answer;
//   * the give-up path decrements after at least one increment, so LTXRGM
//     can never come back zero;
//   * a ray at exactly 70 km is dead here (.LE.) though SETTXR kept it.
//
// GML and FHP are filled as a side effect; nothing reads them except
// LNGPAT's diagnostic listing.
//
// Stage A raises RayIndexError where the find-first search would walk past
// ray 45 (the increment precedes the IAFTXR bound test); here the same
// condition returns 1 with the state exactly as Stage A leaves it at the
// raise (LTXRGM(jj) = 46, nothing since).  Return 0 is success.
//
// float32 throughout; no libm calls at all — pure comparison arithmetic.

#ifndef VOACAP_SELTXR_CUH
#define VOACAP_SELTXR_CUH

#include "f32math.cuh"
#include "state.cuh"

#define VOACAP_SELTXR_NRAY 45

__device__ inline int seltxr(ReflxState* reflx, const LosxState* losx,
                             LpathState* lpath, const int* itxrcp,
                             float gcdkm, float amind, int info) {
    for (int jj = 1; jj <= 2; ++jj) {
        int k = itxrcp[jj];
        float dend = pymin(gcdkm, 4000.0f);
        lpath->ltxrgm[jj] = 1;
        // .....ONE MODE GOOD, AS OVER THE MUF FORCED
        // .....FIND FIRST GOOD MODE (START)                            (105)
        float xhpm = dend / reflx->gdflx[lpath->ltxrgm[jj] * 4 + k];
        int ihop = (int)xhpm;               // REAL->INT truncates toward zero
        int gave_up = 0;
        while (xhpm - 0.9f < 0.0f) {        // IF(XHPM-0.9) 110,115,115
            lpath->ltxrgm[jj] = lpath->ltxrgm[jj] + 1;              // 110
            if (lpath->ltxrgm[jj] > VOACAP_SELTXR_NRAY)
                return 1;                   // RayIndexError
            // .....CHECK MINIMUM ANGLE
            if (reflx->hpflx[lpath->ltxrgm[jj] * 4 + k] <= 70.0f) {
                gave_up = 1;                                        // -> 95
                break;
            }
            if (lpath->ltxrgm[jj] - reflx->iaftxr[k] > 0) {
                gave_up = 1;                                        // -> 95
                break;
            }
            if (reflx->delfx[lpath->ltxrgm[jj] * 4 + k] - amind >= 0.0f)
                break;                      // 112 -> 115
            // else fall round to 110 again; XHPM is NOT recomputed
        }
        if (gave_up) {
            lpath->ltxrgm[jj] = lpath->ltxrgm[jj] - 1;              // 95
            continue;
        }
        // .....FIND FIRST GOOD MODE (END)                              (115)
        float hopx = (float)ihop;
        float yhpm = xhpm - hopx;
        int ist = 2;
        int idnxs = lpath->ltxrgm[jj];
        if (info & 4)
            ist = 1;            // ... and the two passes are NOT independent
        for (int idf = ist; idf <= 2; ++idf) {
            float xfract;
            if (idf == 2) {
                xfract = 0.5f;
                lpath->ltxrgm[jj] = idnxs;
            } else {
                xfract = 0.05f;
            }
            if (yhpm - xfract > 0.0f) {
                // .....HOP INCREMENT
                yhpm = 1.0f - yhpm;                                 // 116
            }
            // .....GAIN-LOSS                                          (120)
            int ia0 = lpath->ltxrgm[jj];
            float gmax = losx->tgainx[ia0 * 4 + k]
                - xhpm * (losx->andvx[ia0 * 4 + k] + losx->advx[ia0 * 4 + k]);
            // .....ANGLE INCREMENT
            float delmax = fabsf(reflx->delfx[ia0 * 4 + k] - lpath->delopt);
            int ls = lpath->ltxrgm[jj];
            for (int ia = ls; ia <= reflx->iaftxr[k]; ++ia) {
                // .....DOES THE MODE EXIST (QUESTION MARK)
                if (reflx->hpflx[ia * 4 + k] - 70.0f <= 0.0f)
                    continue;                                       // 165
                // .....NUMBER OF HOPS                                  (125)
                float xhop = dend / reflx->gdflx[ia * 4 + k];
                ihop = (int)xhop;
                // .....AT LEAST NEAR 1
                if (xhop - 0.9f < 0.0f)
                    continue;                                       // 165
                hopx = (float)ihop;                                 // 126
                float yhop = xhop - hopx;
                if (yhop - 0.5f > 0.0f) {
                    // .....HOP INCREMENT
                    yhop = 1.0f - yhop;                             // 130
                }
                // .....GAIN-LOSS                                       (135)
                float gnow = losx->tgainx[ia * 4 + k]
                    - xhop * (losx->andvx[ia * 4 + k]
                              + losx->advx[ia * 4 + k]);
                reflx->gml[ia * 4 + k] = gnow;
                reflx->fhp[ia * 4 + k] = yhop;
                // .....ANGLE INCREMENT
                float delnow = fabsf(reflx->delfx[ia * 4 + k]
                                     - lpath->delopt);
                //  FIRST CHOICE ON SMALLEST GAIN MINUS LOSS.
                if (fabsf(gnow - gmax) - lpath->gmin > 0.0f) {      // 140
                    if (gnow - gmax <= 0.0f)
                        continue;
                } else {
                    // SECOND CHOICE ON CLOSEST TO SPECULAR REFLECTION
                    // (INTEGER NO. OF HOPS).                           (145)
                    if (fabsf(yhop - yhpm) - lpath->ymin > 0.0f) {  // 150
                        if (yhop - yhpm >= 0.0f)
                            continue;
                    } else {
                        // THIRD CHOICE ON CLOSEST TO APRIORI OPTIMUM
                        // TAKEOFF ANGLE.                               (155)
                        if (delnow - delmax >= 0.0f)
                            continue;
                    }
                }
                lpath->ltxrgm[jj] = ia;                             // 160
                xhpm = xhop;
                gmax = gnow;
                delmax = delnow;
                yhpm = yhop;
            }
        }
    }
    return 0;
}

#endif  // VOACAP_SELTXR_CUH
