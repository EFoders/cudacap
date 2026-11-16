// SETTXR — price every ray at the transmitter and receiver ends; the CUDA C
// twin of scalar/settxr.py (SETTXR_ORIG, which is what the vendored
// "Version 16.1207W" string dispatches to).  For each of the two control areas
// ITXRCP holds, SETTXR walks all 45 rays FINDF laid down and fills /LOSX/:
// absorption, deviative loss, sporadic-E obscuration, ground loss and antenna
// gain, then adds the over-the-MUF penalty XLS to EVERY ray's absorption.
//
// Every Stage A finding is preserved verbatim:
//
//   * the gyrofrequency is read as GYZ(K) with K a CONTROL area, though /GEOG/
//     GYZ is indexed by SAMPLE area — the Alex variant's GYZ(LX(K)) is the
//     correction, and it is NOT what runs.  Transliterated as written.
//   * ``IS = IMODE(IA,K)`` is dead, overwritten by ``IS = MODMUF`` on the next
//     source line, so the over-the-MUF probability uses the circuit's
//     controlling layer for every ray.
//   * PROB(JJ) is assigned inside the ray loop (last ray wins — and every
//     iteration writes the same number).
//   * AOFX is the one /LOSX/ column NOT preset per ray: TGAINX, ANDVX, ADVX
//     and GRLOSX are reset at the top of every ray, AOFX only inside the two
//     layer branches, so a dead ray keeps the previous frequency's value.
//   * the ray-existence scan runs to 45 regardless of IAFTXR and gives up at
//     the FIRST ray under 70 km; the over-the-MUF fallback then OVERWRITES
//     ray 1 of /REFLX/, so SETTXR mutates the table it is reading.
//   * ILN counts priced rays and is never read — dropped, as Stage A did.
//
// Where Stage A RAISES, this returns a status with the state exactly as
// Stage A leaves it at the raise:
//   0 = ok,
//   1 = MODMUF outside 1..4 at the over-the-MUF fallback (Stage A:
//       ModmufError, "indexes DELMUF(1..4)"),
//   2 = MODMUF outside 1..4 at the PROS call in the pricing loop (Stage A:
//       ModmufError, "indexes YMUF(1..4)"),
//   3 = GAIN's out-of-bounds table read (Stage A: IndexError),
//   4 = GAIN's unported area-coverage antenna (Stage A: NotImplementedError).
// The GAIN codes are 3/4 in REGMOD and ESMOD too, so a caller can test them
// once.  On 3 or 4 the ray's TGAINX keeps its -10 preset and EFFLP is not
// written — exactly where Stage A's tuple assignment stops.
//
// The /cantenna/ tables are SHARED, not per-thread, and pass flat exactly as
// gain.cuh documents.
//
// Rule 4b: div32 on every compile-time-constant denominator — RZ+100 in the F
// branch, 1000 for the Es dead frequency, 18 in the collision-frequency
// profile.  Every other division has a genuinely runtime denominator.
//
// float32; COS/SQRT/EXP/ALOG/ALOG10 and the ``**1.98`` powers all carry the
// [[f32-libm-noise]] floor.

#ifndef VOACAP_SETTXR_CUH
#define VOACAP_SETTXR_CUH

#include "angcos.cuh"
#include "constants.cuh"
#include "f32math.cuh"
#include "gain.cuh"
#include "prbmuf.cuh"
#include "raysview.cuh"
#include "state.cuh"

#define VOACAP_SETTXR_NRAY 45
#define VOACAP_SETTXR_MODE_HEIGHT 70.0f     // a ray below this is not a mode
#define VOACAP_SETTXR_ES_LAYER 4            // PRBMUF's sporadic-E index

// state: LayerState (GYZ/ABIY/FI/FS/HS/KM/SIGPAT/EPSPAT); muf: MufState;
// sigd: SigdState (HTLOSS/XNUZ/HNU/AFE/BFE/XVE); reflx: ReflxState (read, and
// ray 1 OVERWRITTEN by the over-the-MUF fallback); losx: LosxState (output);
// rays: RaysView over /RAYS/ (IFOB, for the Es dead frequency); allm: AllModeState
// (PROB(1) and PROB(2) written); itxrcp: the 1-based (3,) control-area
// indices; freq in MHz; gcdkm in km; amind the minimum take-off angle in
// degrees.  Returns the status documented above.
__device__ inline int settxr(const LayerState* state, const MufState* muf,
                             const SigdState* sigd, ReflxState* reflx,
                             LosxState* losx, const RaysView& rays,
                             const float* cant_array, const float* cant_aeff,
                             const int* cant_iats, const float* cant_xfqs,
                             const float* cant_xfqe,
                             const float* cant_offazim, int numants,
                             AllModeState* allm, const int* itxrcp,
                             float freq, float gcdkm, float amind) {
    // .....FOR MAX. NUMBER OF HOPS
    float dend = pymin(4000.0f, gcdkm);
    // Pure-function memos across the ray loop (bit-exact: a hit returns what
    // a fresh evaluation of the same argument returns).  PROS does not depend
    // on the ray at all -- MODMUF and YMUF are fixed for the call -- so its
    // ALOG10 is evaluated once instead of once per priced ray.
    GroundMemo gmemo;
    ground_memo_init(&gmemo);
    bool lp_have = false;
    unsigned lp_key = 0u;
    float lp_val = 0.0f;
    for (int jj = 1; jj <= 2; ++jj) {
        int k = itxrcp[jj];
        for (int ia = 1; ia <= VOACAP_SETTXR_NRAY; ++ia) {
            reflx->gml[ia * 4 + k] = -999.0f;
            reflx->fhp[ia * 4 + k] = 999.0f;
        }
        // --- DO 80: does ANY ray exist?  first hit jumps to 95 -------------
        int found = 0;
        for (int ia = 1; ia <= VOACAP_SETTXR_NRAY; ++ia) {
            // .....DOES THE MODE EXIST (QUESTION MARK)
            if (reflx->hpflx[ia * 4 + k] - VOACAP_SETTXR_MODE_HEIGHT < 0.0f)
                break;                      // 90: give up at the FIRST dead ray
            // .....NUMBER OF HOPS
            float xhpm = dend / reflx->gdflx[ia * 4 + k];            // 75
            // .....MUST HAVE AT LEAST ONE FOR SHORT PATHS
            if (xhpm - 0.9f < 0.0f)                                  // 80
                continue;
            if (reflx->delfx[ia * 4 + k] - amind >= 0.0f) {          // 95
                found = 1;
                break;
            }
        }
        if (!found) {
            //  PENETRATED LAYER, TRY OVER THE MUF MODE.
            // .....SO IF NONE FAILS, INCLUDE OVER-THE-MUF
            int im = muf->modmuf;
            if (im < 1 || im > 4)
                return 1;                   // ModmufError: DELMUF(1..4)
            reflx->delfx[1 * 4 + k] = muf->delmuf[im];
            reflx->hpflx[1 * 4 + k] = muf->hpmuf[im];
            reflx->htflx[1 * 4 + k] = muf->htmuf[im];
            reflx->fvflx[1 * 4 + k] = muf->fvmuf[im];
            reflx->afflx[1 * 4 + k] = muf->afmuf[im];
            float xhop = (float)muf->nhopmf[im];
            reflx->gdflx[1 * 4 + k] = gcdkm / xhop;
            reflx->imode[1 * 4 + k] = muf->modmuf;
        }
        // --- 95: price every ray ------------------------------------------
        float bc = pow32(freq + state->gyz[k], 1.98f);  // GYZ(K): see header
        float ac = 677.2f * state->abiy[k];
        for (int ia = 1; ia <= VOACAP_SETTXR_NRAY; ++ia) {
            // .....PRESET ARRAYS  (AOFX is deliberately not among them)
            losx->tgainx[ia * 4 + k] = -10.0f;
            losx->andvx[ia * 4 + k] = 1000.0f;
            losx->advx[ia * 4 + k] = 1000.0f;
            losx->grlosx[ia * 4 + k] = 0.0f;
            // .....MODE GOOD (QUESTION MARK)  ONE IS.
            if (reflx->hpflx[ia * 4 + k] < VOACAP_SETTXR_MODE_HEIGHT)
                continue;
            // .....CHECK FOR MINIMUM
            if (reflx->delfx[ia * 4 + k] < amind)
                continue;
            float deld = reflx->delfx[ia * 4 + k];
            float del_ = deld * VC_D2R;                     // radiation angle
            float cdel = cos32_deg(deld, del_);             // cos32(del_)
            // .....E MODE (QUESTION MARK)
            if (state->fi[1 * 6 + k] < reflx->fvflx[ia * 4 + k]) {
                //  F LAYER MODES                                        (125)
                float xnsq = 10.2f;
                float sinp = div32(VC_RZ * cdel, VC_RZ + 100.0f);
                float secp = 1.0f / sqrt32(1.0f - sinp * sinp);
                // .....ABSORPTION LOSS
                losx->andvx[ia * 4 + k] = secp * ac / (bc + xnsq);
                sinp = VC_RZ * cdel / (VC_RZ + reflx->hpflx[ia * 4 + k]);
                secp = 1.0f / sqrt32(1.0f - sinp * sinp);
                // .....DEVIATIVE LOSS
                losx->advx[ia * 4 + k] =
                    secp * reflx->afflx[ia * 4 + k]
                    * (pow32(reflx->fvflx[ia * 4 + k] + state->gyz[k], 1.98f)
                       + xnsq) / (bc + xnsq);
                //  ES OBSCURTION LOSS
                losx->aofx[ia * 4 + k] = 0.0f;
                // .....SPORADIC E EXISTS (QUESTION MARK)
                if (state->fs[2 * 6 + k] > 0.0f) {
                    // IFOB is the reflectrix in kHz; slot (1,3) is the Es
                    // "dead" frequency below which the layer blanks the mode
                    float fsdead = (float)rv_ifob(rays, 1, 3, k);
                    fsdead = div32(fsdead, 1000.0f);
                    float fmhz = pymax(freq, fsdead);
                    float sins = VC_RZ * cdel / (VC_RZ + state->hs[k]);
                    float secs = 1.0f / sqrt32(1.0f - sins * sins);
                    // .....MUF THIS DISTANCE
                    float esd = state->fs[2 * 6 + k] * secs;
                    float p = prbmuf(muf, fmhz, esd,
                                     muf->ymuf[VOACAP_SETTXR_ES_LAYER],
                                     VOACAP_SETTXR_ES_LAYER);
                    p = pymax(0.1f, p);
                    p = pymin(0.9f, p);
                    // .....OBSCURATION
                    losx->aofx[ia * 4 + k] = -10.0f * log10_32(1.0f - p);
                }
            } else {
                //  D-E  LAYER MODES.
                // .....VERY LOW REFLECTIONS (QUESTION MARK)
                float xnsq;
                if (sigd->htloss <= reflx->htflx[ia * 4 + k]) {
                    xnsq = 10.2f;
                } else {
                    float hnux = 61.0f
                        + div32(3.0f * (reflx->htflx[ia * 4 + k] - 70.0f),
                                18.0f);
                    // .....COLLISION FREQUENCY
                    xnsq = sigd->xnuz
                           * exp32((-2.0f) * (hnux - 60.0f) / sigd->hnu);
                }
                float heff = pymin(100.0f, reflx->htflx[ia * 4 + k]);
                float sinp = VC_RZ * cdel / (VC_RZ + heff);
                float secp = 1.0f / sqrt32(1.0f - sinp * sinp);
                // .....ABSORPTION LOSS
                losx->andvx[ia * 4 + k] = secp * ac / (bc + xnsq);
                float xv = pymax(reflx->fvflx[ia * 4 + k]
                                 / state->fi[1 * 6 + k], sigd->xve);
                // .....CORRECTION FOR E LAYER
                float adx = sigd->afe + sigd->bfe * log32(xv);
                sinp = VC_RZ * cdel / (VC_RZ + reflx->hpflx[ia * 4 + k]);
                secp = 1.0f / sqrt32(1.0f - sinp * sinp);
                // .....DEVIATIVE LOSS + CORRECTION
                losx->advx[ia * 4 + k] =
                    (secp * reflx->afflx[ia * 4 + k]
                     * (pow32(reflx->fvflx[ia * 4 + k] + state->gyz[k], 1.98f)
                        + xnsq) / (bc + xnsq)) + adx;
                // .....ES OBSCURATION, NONE
                losx->aofx[ia * 4 + k] = 0.0f;
            }
            //  GROUND LOSS                                              (160)
            float y = ground_loss_sum(state, del_, cdel, freq,
                                      &gmemo);              // DO IG
            float xkm = (float)state->km;
            // .....AVERAGE GROUND LOSS
            losx->grlosx[ia * 4 + k] = y / xkm;
            //  ANTENNA GAINS
            float teff;
            int oob = 0;
            antenna_gain(cant_array, cant_aeff, cant_iats, cant_xfqs,
                         cant_xfqe, cant_offazim, numants, jj, del_, freq,
                         &y, &teff, &oob);
            if (oob != 0)
                return (oob == 1) ? 3 : 4;  // TGAINX/EFFLP are not assigned
            losx->tgainx[ia * 4 + k] = y;
            if (jj == 2)
                losx->efflp[ia] = teff;     // ray index only -- the RX end
            float sphet = VC_RZ * cdel / (VC_RZ + reflx->htflx[ia * 4 + k]);
            float cphet = 1.0f - sphet * sphet;
            cphet = pymax(0.00000001f, cphet);
            cphet = sqrt32(cphet);
            // C-----?????????????????
            // IS = IMODE(IA,K) is dead: overwritten on the next source line.
            int is_ = muf->modmuf;
            // C-----?????????????????
            if (is_ < 1 || is_ > 4)
                return 2;                   // ModmufError: YMUF(1..4)
            float pros = prbmuf(muf, freq, muf->ymuf[is_], muf->ymuf[is_],
                                is_);
            // .....OVER-THE-MUF, MEDIAN
            float lpros;
            if (lp_have && __float_as_uint(pros) == lp_key) {
                lpros = lp_val;
            } else {
                lpros = log10_32(pros);
                lp_have = true;
                lp_key = __float_as_uint(pros);
                lp_val = lpros;
            }
            float xls = -10.0f * lpros / cphet;
            //  ADD ADJUSTMENT TO ABSORPTION LOSS
            losx->andvx[ia * 4 + k] = losx->andvx[ia * 4 + k] + xls;
            allm->prob[jj] = pros;      // same value every ray; see header
            // ILN = ILN + 1 -- counted and never read
        }
    }
    return 0;
}

#endif  // VOACAP_SETTXR_CUH
