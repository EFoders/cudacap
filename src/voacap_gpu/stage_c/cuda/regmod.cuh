// REGMOD — per-mode loss budget for one frequency and hop distance; the CUDA C
// twin of stage_a/regmod.py.  Takes the up-to-six regular modes FDIST left in
// /MODES/ and turns each into a full link budget in /ZON/: free-space loss,
// non-deviative absorption, deviative loss, ground-reflection loss, sporadic-E
// obscuration, antenna gains, the over-the-MUF loss and its deciles, and from
// those the field strength, signal power and signal-to-noise.  Sporadic-E
// *modes* are ESMOD's job; what appears here is only Es getting in the way of
// an F mode.  Mode slot 7 is untouched beyond its HN/HP = -1 preset.
//
// Three details invisible in the arithmetic, kept exactly as Stage A has them:
//
//   * ``**1.98`` is a default-REAL literal, so the exponent is the binary32
//     nearest 1.98 and the power goes through pow32, never a multiply.
//   * DEL and CDEL are RECOMPUTED partway through (the 1995 hop-MUF change)
//     from the path's own geometry rather than the mode's.  Everything before
//     that point — crucially the three GAIN calls — uses the mode's take-off
//     angle; everything after uses the recomputed one.  The order is
//     load-bearing.
//   * ITRY is vestigial: set to 1 at the top of every iteration, so the
//     ``ITRY.lt.1`` test can never fire and the ``ITRY = -1`` near the end is
//     dead.  Preserved as this comment rather than as code.
//
// Where Stage A RAISES, this returns a status with /ZON/ exactly as Stage A
// leaves it at the raise:
//   0 = ok,
//   1 = MUFday probability <= 0 (Stage A: RegmodError; the Fortran STOPs),
//   2 = FREQ <= 0 (Stage A: RegmodError; the Fortran STOPs),
//   3 = GAIN's out-of-bounds table read (Stage A: IndexError),
//   4 = GAIN's unported area-coverage antenna (Stage A: NotImplementedError).
// The GAIN codes are 3/4 in ESMOD too, so a caller can test them once.
// On 3 from the TRANSMIT gain nothing of that mode's TGAIN/RGAIN/EFF is
// written; on 3 from the RECEIVE gain TGAIN is written and the other two are
// not — exactly where Stage A's tuple assignment stops.
//
// The /cantenna/ + /pantenna/ tables are SHARED, not per-thread, and pass flat
// exactly as gain.cuh / pwrdb.cuh document.
//
// Rule 4b: every division by a compile-time-constant denominator goes through
// div32 (VOFL, RZ+100, 1000, 18, 2, RZ).  No REAL*8 anywhere — regmod.for is
// REAL*4 throughout.
//
// float32; ALOG10/ALOG/EXP/COS/SIN/ATAN/SQRT and the 1.98 powers all carry the
// [[f32-libm-noise]] floor, and XTLOS sums seven terms of comparable size.

#ifndef VOACAP_REGMOD_CUH
#define VOACAP_REGMOD_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "gain.cuh"
#include "prbmuf.cuh"
#include "pwrdb.cuh"
#include "state.cuh"

// state: LayerState (/RON/, /ES/, /GEOG/); modes: ModesState (/MODES/);
// muf: MufState (/MUFS/); sigd: SigdState (/SIGD/); anois: AnoisState
// (/ANOIS/); rays: RaysState (/RAYS/); zon: ZonState, the output.
// freq: MHz; jmode: the ionosphere SELMOD picked; gcd: great-circle distance
// in radians (/DON/).  Returns the status documented above.
__device__ inline int regmod(const LayerState* state, const ModesState* modes,
                             const MufState* muf, const SigdState* sigd,
                             const AnoisState* anois, const RaysState* rays,
                             const float* cant_array, const float* cant_aeff,
                             const int* cant_iats, const float* cant_xfqs,
                             const float* cant_xfqe, const float* cant_offazim,
                             const float* cant_pwrdba, int numants,
                             ZonState* zon, float freq, int jmode, float gcd) {
    const float delmax = 89.99f * VC_D2R;
    const float prob_floor = 0.000001f;
    const int lx[4] = {0, 1, 3, 5};     // DATA LX/1,3,5/ — K to sample area L
    float ghop = modes->ghop;
    int k = jmode;
    int ll = lx[k];
    // average absorption index from SIGDIS
    float ac = 677.2f * sigd->acav;
    float bc = pow32(freq + state->gyz[ll], 1.98f);
    int ihop = (int)(gcd / ghop + 0.01f);
    float hop = (float)ihop;

    for (int im = 1; im <= 7; ++im) {
        // ITRY = 1 here in the Fortran; the ITRY test below it is unreachable
        zon->hn[im] = -1.0f;
        zon->hp[im] = -1.0f;
        if (im >= 7)
            continue;
        if (modes->hpmod[im * 4 + k] <= 0.0f)
            continue;

        float del_ = VC_D2R * modes->delmod[im * 4 + k];
        del_ = pymin(del_, delmax);
        float cdel = cos32(del_);
        float psi = ghop * 0.5f;
        float phe = VC_PIO2 - psi - del_;
        float path = (2.0f * (modes->hpmod[im * 4 + k]
                              + VC_RZ * (1.0f - cos32(psi)))) / cos32(phe);
        path = fabsf(path * hop);
        zon->timed[im] = div32(path, VC_VOFL);
        zon->fslos[im] = 32.45f + 20.0f * log10_32(path * freq);

        float obfu, obfl;
        if (state->fi[1 * 6 + k] - modes->fvmod[im * 4 + k] < 0.0f) {
            // 125: F layer mode
            float xnsq = 10.2f;
            float sinp = div32(VC_RZ * cdel, VC_RZ + 100.0f);
            float secp = 1.0f / sqrt32(1.0f - sinp * sinp);
            zon->abps[im] = secp * ac / (bc + xnsq);
            secp = 1.0f / sin32(del_ + psi);
            zon->adv[im] = secp * modes->afmod[im * 4 + k]
                           * (pow32(modes->fvmod[im * 4 + k] + state->gyz[ll],
                                    1.98f) + xnsq) / (bc + xnsq);
            zon->obf[im] = 0.0f;
            obfu = 0.0f;
            obfl = 0.0f;
            if (state->fs[2 * 6 + k] > 0.0f) {
                // Es obscuration.  At low frequencies folding regular E into
                // sporadic E misbehaves, so the sounding floor FSDEAD applies
                // (SIGDIS does the same).
                float fsdead = (float)rays->ifob[(1 * 31 + 3) * 6 + k];
                fsdead = div32(fsdead, 1000.0f);
                fsdead = pymin(fsdead, 3.0f);
                float fmhz = pymax(freq, fsdead);
                float sins = VC_RZ * cdel / (VC_RZ + state->hs[k]);
                float secs = 1.0f / sqrt32(1.0f - sins * sins);
                // median MUF for THIS hop, not for the path
                float esd = state->fs[2 * 6 + k] * secs;
                float pros = prbmuf(muf, fmhz, esd, muf->ymuf[4], 4);
                pros = pymin(pros, 0.90f);
                zon->obf[im] = -10.0f * log10_32(1.0f - pros);
                esd = state->fs[1 * 6 + k] * secs;
                // upper decile obscuration at the Es FOT
                pros = prbmuf(muf, fmhz, esd, muf->yfot[4], 4);
                pros = pymin(pros, 0.90f);
                obfu = -10.0f * log10_32(1.0f - pros);
                // lower decile obscuration at the Es HPF
                pros = prbmuf(muf, fmhz, esd, muf->yhpf[4], 4);
                pros = pymin(pros, 0.9f);
                obfl = -10.0f * log10_32(1.0f - pros);
            }
        } else {
            // 106: D-E mode
            float xnsq;
            if (sigd->htloss - modes->htmod[im * 4 + k] <= 0.0f) {    // 110
                xnsq = 10.2f;
            } else {                                                  // 115
                float hnux = 61.0f + div32(3.0f * (modes->htmod[im * 4 + k]
                                                   - 70.0f), 18.0f);
                xnsq = sigd->xnuz * exp32((-2.0f) * (hnux - 60.0f) / sigd->hnu);
            }
            // secant PHE at the reflection area; 100 km is assumed above that
            float heff = pymin(100.0f, modes->htmod[im * 4 + k]);
            float sinp = VC_RZ * cdel / (VC_RZ + heff);
            float secp = 1.0f / sqrt32(1.0f - sinp * sinp);
            zon->abps[im] = secp * ac / (bc + xnsq);
            // absorption loss with the E-layer bending effect removed
            float xv = pymax(modes->fvmod[im * 4 + k] / state->fi[1 * 6 + k],
                             sigd->xve);
            float adx = sigd->afe + sigd->bfe * log32(xv);
            secp = 1.0f / sin32(del_ + psi);
            float bcx = bc;
            zon->adv[im] = secp * modes->afmod[im * 4 + k]
                           * (pow32(modes->fvmod[im * 4 + k] + state->gyz[ll],
                                    1.98f) + xnsq) / (bcx + xnsq) + adx;
            zon->obf[im] = 0.0f;                // no Es obscuration for D-E
            obfu = 0.0f;
            obfl = 0.0f;
        }

        // 160: ground reflection loss, averaged over the sample areas
        float y = 0.0f;
        for (int ig = 1; ig <= state->km; ++ig)
            y = y + ground_loss(state, ig, del_, freq);
        zon->grlos[im] = y / (float)state->km;

        int oob = 0;
        float g, e;
        antenna_gain(cant_array, cant_aeff, cant_iats, cant_xfqs, cant_xfqe,
                     cant_offazim, numants, 1, del_, freq, &g, &e, &oob);
        if (oob != 0)
            return (oob == 1) ? 3 : 4;          // TGAIN is not yet assigned
        zon->tgain[im] = g;
        antenna_gain(cant_array, cant_aeff, cant_iats, cant_xfqs, cant_xfqe,
                     cant_offazim, numants, 2, del_, freq, &g, &e, &oob);
        if (oob != 0)
            return (oob == 1) ? 3 : 4;          // RGAIN/EFF are not
        zon->rgain[im] = g;
        zon->eff[im] = e;

        float hops = pymin(hop, 2.0f);          // only 2 hops for obscuration
        float xtlos = zon->fslos[im] + hop * (zon->abps[im] + zon->adv[im])
                      + zon->grlos[im] * (hop - 1.0f) + hops * zon->obf[im]
                      + sigd->asm_ - zon->rgain[im] - zon->tgain[im];

        int ismod = modes->itmod[im * 4 + k];
        float sphet = VC_RZ * cdel / (VC_RZ + modes->htmod[im * 4 + k]);
        float cphet = 1.0f - sphet * sphet;
        cphet = pymax(0.000001f, cphet);
        cphet = sqrt32(cphet);

        // 1995 change: base MUFday on the specific HOP muf, not the path muf.
        // The CPHET above is kept as-is for its other uses.
        psi = div32(gcd, 2.0f);
        psi = psi / hop;
        float cpsi = cos32(psi);
        float spsi = sin32(psi);
        float tanp = spsi / (1.0f - cpsi + div32(muf->hpmuf[ismod], VC_RZ));
        phe = atan32(tanp);
        del_ = VC_PIO2 - phe - psi;
        cdel = cos32(del_);
        float sphe = VC_RZ * cdel / (VC_RZ + muf->htmuf[ismod]);
        float xmuf = muf->fvmuf[ismod] / sqrt32(1.0f - sphe * sphe);
        zon->prob[im] = prbmuf(muf, freq, xmuf, muf->ymuf[ismod], ismod);
        if (zon->prob[im] < 0.0001f) {
            if (zon->prob[im] <= 0.0f)
                return 1;                       // the Fortran STOPs here
            // extra loss as MUFday gets very low, 0 to 24 dB over 1e-4..1e-7
            float xfac = log10_32(zon->prob[im]);
            if (xfac > -4.0f)
                xfac = -4.0f;
            if (xfac < -7.0f)
                xfac = -7.0f;
            float ghlos = (xfac + 4.0f) * 8.0f;
            xtlos = xtlos - ghlos;
        }

        xmuf = muf->fvmuf[ismod] / cphet;
        float p = prbmuf(muf, freq, xmuf, muf->ymuf[ismod], ismod);
        p = pymax(p, prob_floor);
        float xls = -10.0f * log10_32(p) / cphet;
        xtlos = xtlos + xls * hop;                          // median

        float cpr = muf->fvmuf[ismod] / muf->ymuf[ismod];
        float fvfot = muf->yfot[ismod] * cpr;
        xmuf = fvfot / cphet;
        float pf = prbmuf(muf, freq, xmuf, muf->yfot[ismod], ismod);
        pf = pymax(pf, prob_floor);
        float xlsl = -10.0f * log10_32(pf) / cphet;
        float fvhpf = muf->yhpf[ismod] * cpr;
        xmuf = fvhpf / cphet;
        pf = prbmuf(muf, freq, xmuf, muf->yhpf[ismod], ismod);
        pf = pymax(pf, prob_floor);
        float xlsu = -10.0f * log10_32(pf) / cphet;

        // despite the names these are the signal level's lower/upper deciles
        zon->tllow[im] = sigd->dsl + hops * (obfl - zon->obf[im])
                         + hop * (xlsl - xls);
        zon->tlhgh[im] = sigd->dsu + hops * (zon->obf[im] - obfu)
                         + hop * (xls - xlsu);
        zon->tllow[im] = pymin(zon->tllow[im], 25.0f);
        zon->tlhgh[im] = pymin(zon->tlhgh[im], 25.0f);

        zon->tloss[im] = xtlos;
        if (freq <= 0.0f)
            return 2;                           // the Fortran STOPs here
        float pdb = pwrdb(cant_iats, cant_xfqs, cant_xfqe, cant_pwrdba,
                          numants, freq);
        zon->fldst[im] = 107.2f + pdb + 20.0f * log10_32(freq) - xtlos
                         - zon->rgain[im];
        zon->sigpow[im] = pdb - xtlos;
        // long-path receiver efficiency correction (1991)
        zon->sn[im] = zon->sigpow[im] - anois->rcnse - zon->eff[im];
        zon->b[im] = modes->delmod[im * 4 + k];
        zon->nmode[im] = ismod;
        zon->hp[im] = modes->hpmod[im * 4 + k];
        zon->hn[im] = (float)ihop;
    }
    return 0;
}

#endif  // VOACAP_REGMOD_CUH
