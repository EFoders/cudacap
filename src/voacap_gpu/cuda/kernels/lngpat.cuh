// LNGPAT — assemble the long path's single mode; the CUDA C twin of
// scalar/lngpat.py.  SELTXR has already chosen one ray at each end,
// LTXRGM(1) at control area 1 and LTXRGM(2) at area ITXRCP(2); LNGPAT averages
// the two into a single geometry — mean take-off angle, mean virtual height —
// and prices it with CONVH (convergence credit and group path), GETTOP (how
// much of the ionospheric stretch flies over the E layer), TABS (what that
// costs per km) and BABS (what the rest, through the D region, costs per km),
// plus half the absorption at each end, the ground loss for the fractional
// hops between, and both antenna gains.
//
// BABS and TABS DISAGREE about a virtual height of exactly 70 km — BABS keeps
// it live, TABS lets it die — and LNGPAT calls both with the same HPX.  That
// asymmetry lives in babs.cuh/tabs.cuh; here they are simply both called.
//
// Three things about the result, all preserved:
//
//   * IT WRITES TWO SLOTS OF ARRAYS THAT HOLD ONE MODE.  NMMOD is 1, but B(1)
//     and B(2) are the transmission and reception angles OF THE SAME MODE, and
//     NMODE(1)/NMODE(2) its reflecting layer at each end.  PROB is the same
//     shape: SETTXR leaves the transmitter end's MUFday in PROB(1) and the
//     receiver end's in PROB(2), and the last line collapses them with AMIN1.
//   * IT CLOBBERS /SIGD/ DSLF AND DSUF, using them as scratch for the
//     per-layer decile spreads (RELBIL does the same).
//   * /TON/ is declared and never used; not ported.
//
// Returns without touching anything when SELTXR left either end unset
// (LTXRGM <= 0) — dead in the real program, since SELTXR's give-up path
// decrements only after an increment, but transliterated.
//
// Where Stage A RAISES ModeIndexError — NMODE at one end outside 1..4, so
// YMUF(IS) reads off the end of a four-element array — this returns 1 with
// the state exactly as Stage A leaves it at the raise (everything through
// SN(1) written; on the second pass DSLF/DSUF still hold pass one's values).
// 0 is success.
//
// Rule 4b: div32 on every compile-time-constant denominator — RZ for DT, DR
// and DM, 2 for the two end losses, VOFL for the time delay.  FHOP, HN and
// the decile CPR divisions all have runtime denominators.
//
// float32; ALOG10, COS, ATAN and SQRT carry the [[f32-libm-noise]] floor.

#ifndef VOACAP_LNGPAT_CUH
#define VOACAP_LNGPAT_CUH

#include "babs.cuh"
#include "constants.cuh"
#include "convh.cuh"
#include "f32math.cuh"
#include "gettop.cuh"
#include "prbmuf.cuh"
#include "pwrdb.cuh"
#include "state.cuh"
#include "tabs.cuh"

#define VOACAP_LNGPAT_MAX_DECILE 25.0f  // the signal deciles are capped here
#define VOACAP_LNGPAT_TX 1              // the transmitter end is control area 1

// state: LayerState (GYZ for BABS/TABS, HI/FI for GETTOP); lpath: LpathState
// (LTXRGM, DELOPT); losx: LosxState; reflx: ReflxState; muf: MufState; allm:
// AllModeState (output, /allMODE/ slot 1 and slot 2's angle and layer); sigd:
// SigdState (ASM/DSL/DSU in, DSLF/DSUF clobbered); anois: AnoisState (RCNSE);
// the /cantenna/ + /pantenna/ scan tables for PWRDB; freq in MHz; gcd the
// great-circle angle in radians; itxrcp2 the receiver end's control area.
__device__ inline int lngpat(const LayerState* state, const LpathState* lpath,
                             const LosxState* losx, const ReflxState* reflx,
                             const MufState* muf, AllModeState* allm,
                             SigdState* sigd, const AnoisState* anois,
                             const int* cant_iats, const float* cant_xfqs,
                             const float* cant_xfqe, const float* cant_pwrdba,
                             int numants, float freq, float gcd,
                             int itxrcp2) {
    int l1 = lpath->ltxrgm[1];
    int l2 = lpath->ltxrgm[2];
    // .....LTXRGM(1) .GE. 1 AND LTXRGM(2) .GE. 1 SINCE OVER-THE-MUF ADDED
    if (l1 <= 0 || l2 <= 0)
        return 0;
    int k2 = itxrcp2;
    const int tx = VOACAP_LNGPAT_TX;

    // .....AVERAGE TAKE-OFF ANGLE
    float del_ = 0.5f * VC_D2R * (reflx->delfx[l1 * 4 + tx]
                                  + reflx->delfx[l2 * 4 + k2]);
    // .....AVERAGE VIRTUAL HEIGHT
    float hpx = 0.5f * (reflx->hpflx[l1 * 4 + tx]
                        + reflx->hpflx[l2 * 4 + k2]);
    float rcosd = VC_RZ * cos32(del_);
    float sphe = rcosd / (VC_RZ + hpx);
    float phe = atan32(sphe / sqrt32(1.0f - sphe * sphe));
    float dt = div32(0.5f * reflx->gdflx[l1 * 4 + tx], VC_RZ);
    float dr = div32(0.5f * reflx->gdflx[l2 * 4 + k2], VC_RZ);
    float ray = VC_RZ * cos32(phe + del_) / sphe;
    // .....FREE-SPACE CONVERGENCE FACTOR
    float ch, ptot;
    convh(gcd, phe, del_, hpx, ray, &ch, &ptot);
    // .....FREE-SPACE LOSS
    float free_ = 36.58f + 20.0f * log10_32(0.6214f * ptot * freq) - ch;
    // .....IONOSPHERIC DISTANCE
    float di = gcd - dt - dr;
    float dikm = di * VC_RZ;
    float dm, am, af, df;
    if (dikm <= -1.0f) {
        dm = 0.0f;
        am = 0.0f;
        af = 0.0f;
        df = 0.0f;
    } else {
        // .....OVER-THE-TOP DISTANCE
        float gmkm = gettop(state, lpath, freq, dikm);
        dm = div32(gmkm, VC_RZ);
        df = di - dm;
        df = pymax(df, 0.0f);
        // .....LOSS PER KILOMETER
        am = tabs(state, freq, del_, hpx, dm);
        af = babs(state, sigd, freq, del_, hpx, df);
    }
    // .....LOSSES
    float xlm = am * dm * VC_RZ;
    float xlf = af * df * VC_RZ;
    //  LOSS AT TRANSMITTER END.
    float xlt = div32(losx->andvx[l1 * 4 + tx] + losx->advx[l1 * 4 + tx],
                      2.0f);
    //  LOSS AT RECEIVER END
    float xlr = div32(losx->andvx[l2 * 4 + k2] + losx->advx[l2 * 4 + k2],
                      2.0f);
    // .....AVERAGE GROUND LOSS
    float gloss = 0.5f * (losx->grlosx[l1 * 4 + tx]
                          + losx->grlosx[l2 * 4 + k2]);
    float fhop = df / (dr + dt);
    fhop = pymax(fhop, 1.0f);
    allm->nmmod = 1;
    // .....TRANSMISSION LOSS
    float tlosl = free_ + xlt + xlm + xlf + xlr + (fhop - 1.0f) * gloss
                  + sigd->asm_ - losx->tgainx[l2 * 4 + k2]
                  - losx->tgainx[l1 * 4 + tx];
    allm->grlos[1] = gloss;
    allm->hn[1] = 0.5f * gcd / (dr + dt);
    // .....NUMBER OF HOPS
    allm->hn[1] = pymax(1.0f, allm->hn[1]);
    // .....VIRTUAL HEIGHT
    allm->hp[1] = hpx;
    // .....ANTENNA GAIN
    allm->rgain[1] = losx->tgainx[l2 * 4 + k2];
    allm->tgain[1] = losx->tgainx[l1 * 4 + tx];
    allm->eff[1] = losx->efflp[l2];     // ray index only -- always the RX end
    // .....TIME DELAY
    allm->timed[1] = div32(ptot, VC_VOFL);
    allm->tloss[1] = tlosl;
    // .....TRANSMISSION ANGLE
    allm->b[1] = reflx->delfx[l1 * 4 + tx];
    // .....RECEPTION ANGLE  (slot 2 of a per-mode array; see the header)
    allm->b[2] = reflx->delfx[l2 * 4 + k2];
    // .....MODE TYPE
    allm->nmode[1] = reflx->imode[l1 * 4 + tx];
    allm->nmode[2] = reflx->imode[l2 * 4 + k2];
    allm->fslos[1] = free_;
    allm->adv[1] = (losx->advx[l1 * 4 + tx] + losx->advx[l2 * 4 + k2]) * 0.5f;
    allm->abps[1] = (losx->andvx[l1 * 4 + tx]
                     + losx->andvx[l2 * 4 + k2]) * 0.5f;
    allm->obf[1] = (losx->aofx[l1 * 4 + tx] + losx->aofx[l2 * 4 + k2]) * 0.5f;
    float pwr = pwrdb(cant_iats, cant_xfqs, cant_xfqe, cant_pwrdba, numants,
                      freq);
    allm->sigpow[1] = pwr - tlosl;
    // .....FIELD STRENGTH
    allm->fldst[1] = 107.2f + pwr + 20.0f * log10_32(freq) - tlosl
                     - allm->rgain[1];
    // .....SNR
    // ------------    CHANGED 9/24/91  (LONG PATH RCVR EFF CORRECTION)  FJR
    // ------------            SN(1) = SIGPOW(1) - RCNSE
    allm->sn[1] = allm->sigpow[1] - anois->rcnse - allm->eff[1];

    int itry = 1;
    int is_ = allm->nmode[1];
    float dslff = 0.0f;
    float dsuff = 0.0f;
    while (true) {                                          // 200
        if (is_ < 1 || is_ > 4)
            return 1;               // ModeIndexError: YMUF(1..4) off the end
        float p = prbmuf(muf, freq, muf->ymuf[is_], muf->ymuf[is_], is_);
        float zlmuf = -10.0f * log10_32(p);
        p = prbmuf(muf, freq, muf->yfot[is_], muf->yfot[is_], is_);
        float zlfot = -10.0f * log10_32(p);
        p = prbmuf(muf, freq, muf->yhpf[is_], muf->yhpf[is_], is_);
        float zlhpf = -10.0f * log10_32(p);
        float cpr = muf->fvmuf[is_] / muf->ymuf[is_];
        sigd->dslf = (zlfot - zlmuf) / cpr;
        sigd->dsuf = (zlmuf - zlhpf) / cpr;
        if (itry - 1 > 0) {                 // IF(ITRY-1) 250,250,260
            sigd->dsuf = 0.5f * (sigd->dsuf + dsuff);       // 260
            sigd->dslf = 0.5f * (sigd->dslf + dslff);
            break;
        }
        if (is_ - allm->nmode[2] == 0)      // IF(IS-NMODE(2)) 255,265,255
            break;                          // both ends on one layer
        itry = 2;                                           // 255
        is_ = allm->nmode[2];
        dslff = sigd->dslf;
        dsuff = sigd->dsuf;
    }
    allm->tllow[1] = sigd->dsl + allm->hn[1] * sigd->dslf;   // 265
    allm->tlhgh[1] = sigd->dsu + allm->hn[1] * sigd->dsuf;
    // .....LOWER DECILE
    allm->tllow[1] = pymin(allm->tllow[1], VOACAP_LNGPAT_MAX_DECILE);
    // .....UPPER DECILE
    allm->tlhgh[1] = pymin(allm->tlhgh[1], VOACAP_LNGPAT_MAX_DECILE);
    // .....F.DAYS  -- SETTXR left the two ends' MUFdays in slots 1 and 2
    allm->prob[1] = pymin(allm->prob[1], allm->prob[2]);
    return 0;
}

#endif  // VOACAP_LNGPAT_CUH
