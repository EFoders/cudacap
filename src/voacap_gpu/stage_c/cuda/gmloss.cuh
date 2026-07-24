// GMLOSS — "the gain minus loss function", and mostly a wrapper; the CUDA C
// twin of stage_a/gmloss.py.  Sixty lines, of which the last twelve compute a
// quantity NOTHING EVER READS: TXRGML is assigned here and nowhere else in
// voacapw, and no routine reads it.  So GMLOSS has exactly two surviving
// effects — it presets all seven /ZON/ slots to the model's "no mode" markers
// (GENOIS runs afterwards and reads the block), and it calls SETTXR.
//
// The TXRGML loop is transliterated anyway, because "write-only" is a claim
// about the whole program and the dump is what makes it checkable.  Note it
// does NOT preset TXRGML: a ray that fails either test keeps whatever the
// previous frequency left there.
//
// GMLOSS agrees with SETTXR about exactly 70 km and disagrees with SELTXR:
// ``IF(HPFLX(IA,K)-70.) 110,105,105`` keeps a ray at exactly 70, where
// SELTXR's ``HPFLX .LE. 70`` kills it.  Likewise ``IF(DELFX-AMIND) 110,106,106``
// keeps a ray exactly on the minimum angle.
//
// Returns SETTXR's status unchanged (0 = ok; see settxr.cuh for 1..4).  On a
// refusal the /ZON/ preset has already happened and the TXRGML loop has not,
// which is exactly where Stage A's exception leaves it.
//
// float32; no libm calls of its own -- everything transcendental is inside
// SETTXR.  No divisions at all, so no div32 site.

#ifndef VOACAP_GMLOSS_CUH
#define VOACAP_GMLOSS_CUH

#include "f32math.cuh"
#include "settxr.cuh"
#include "state.cuh"

#define VOACAP_GMLOSS_NZON 7
#define VOACAP_GMLOSS_MODE_HEIGHT 70.0f

// DO 240: every /ZON/ slot back to its "no mode" marker.  The usual mixed
// vocabulary: 1000 dB for a loss that was not computed, 0.001 for a
// probability, -1000 for a level, -1 for a height or a delay, and NMODE 5,
// the ' N' entry in LAYTYP.
__device__ inline void gmloss_preset_(ZonState* zon) {
    for (int im = 1; im <= VOACAP_GMLOSS_NZON; ++im) {
        zon->obf[im] = 1000.0f;
        zon->adv[im] = 1000.0f;
        zon->fslos[im] = 1000.0f;
        zon->tloss[im] = 1000.0f;
        zon->abps[im] = 1000.0f;
        zon->eff[im] = 0.0f;
        zon->grlos[im] = 1000.0f;
        zon->rgain[im] = 0.0f;
        zon->tgain[im] = 0.0f;
        zon->hn[im] = -1.0f;
        zon->prob[im] = 0.001f;
        zon->crel[im] = 0.001f;
        zon->rely[im] = 0.001f;
        zon->spro[im] = 0.001f;
        zon->fldst[im] = -1000.0f;
        zon->sigpow[im] = -1000.0f;
        zon->sn[im] = -1000.0f;
        zon->timed[im] = -1.0f;
        zon->hp[im] = -1.0f;
        zon->b[im] = -1.0f;
        zon->nmode[im] = 5;         // LAYTYP(5) = ' N', i.e. no layer
    }
}

// Arguments are SETTXR's plus `zon` (ZonState, preset here) and `lpath`
// (LpathState, TXRGML written here).
__device__ inline int gmloss(const LayerState* state, const MufState* muf,
                             const SigdState* sigd, ReflxState* reflx,
                             LosxState* losx, const RaysState* rays,
                             const float* cant_array, const float* cant_aeff,
                             const int* cant_iats, const float* cant_xfqs,
                             const float* cant_xfqe,
                             const float* cant_offazim, int numants,
                             AllModeState* allm, ZonState* zon,
                             LpathState* lpath, const int* itxrcp,
                             float freq, float gcdkm, float amind) {
    // .....PRESET ALL ARRAYS
    gmloss_preset_(zon);
    // .....TRANSMITTER/RECEIVER ENDS
    int status = settxr(state, muf, sigd, reflx, losx, rays, cant_array,
                        cant_aeff, cant_iats, cant_xfqs, cant_xfqe,
                        cant_offazim, numants, allm, itxrcp, freq, gcdkm,
                        amind);
    if (status != 0)
        return status;              // Stage A's exception propagates here
    for (int jj = 1; jj <= 2; ++jj) {
        int k = itxrcp[jj];
        for (int ia = 1; ia <= reflx->iaftxr[k]; ++ia) {
            // .....AT LEAST AN OVER THE MUF
            if (reflx->hpflx[ia * 4 + k] - VOACAP_GMLOSS_MODE_HEIGHT < 0.0f)
                continue;           // ... so exactly 70 LIVES
            if (reflx->delfx[ia * 4 + k] - amind < 0.0f)
                continue;           // ... and so does exactly AMIND
            // .....GAIN-LOSS, TRANSMITTER /RECEIVER ENDS
            lpath->txrgml[ia * 3 + jj] =
                losx->tgainx[ia * 4 + k]
                - (losx->andvx[ia * 4 + k] + losx->advx[ia * 4 + k]);
        }
    }
    return 0;
}

#endif  // VOACAP_GMLOSS_CUH
