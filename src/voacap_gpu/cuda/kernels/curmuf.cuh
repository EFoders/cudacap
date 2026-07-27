// CURMUF — circuit MUF and FOT from the full electron-density profile.
// The CUDA C twin of scalar/curmuf.py (which is CURMUF_orig, the version the
// oracle build selects).
//
// Picks the controlling sample area, builds its profile (LECDEN), and for the
// E, F2, F1 and sporadic-E layers works out the oblique MUF by the secant law
// with a Martyn's-theorem virtual-height correction iterated to 0.1 MHz.  The
// circuit MUF is the max over E/F1/F2 (Es excluded); FOT/HPF are the
// lower/upper decile envelopes.
//
// The 190/225 GOTO iterations are while-loops with the exact convergence and
// try-count exits.  IEDP<0 (the blkdat default) takes the GETHP profile path;
// the analytic IEDP>=0 path is preserved too.  INFO's unit-99 debug writes are
// omitted (no state effect).
//
// TWO FAITHFUL QUIRKS, do not "fix":
//   * the E/F1 blocks form PSI as GCDKM/(2*RZ) then /XHOPS while the F2 block
//     forms it as GCDKM/((2*RZ)*XHOPS) — the same value, but the float32
//     rounding differs by up to 1 ULP, so both spellings are kept;
//   * HOP, used for NHOPMF(4), keeps the value from the LAST sample area the
//     Es loop examined, not the winning one.
//
// REAL*4 throughout; SIN/COS/ASIN/ATAN/EXP carry the f32-libm floor, SQRT is
// clean.

#ifndef VOACAP_CURMUF_CUH
#define VOACAP_CURMUF_CUH

#include "bendy.cuh"
#include "constants.cuh"
#include "f2dis.cuh"
#include "f32math.cuh"
#include "gethp.cuh"
#include "lecden.cuh"
#include "pen.cuh"
#include "state.cuh"

#define VC_CURMUF_FQDEL 0.1f
#define VC_CURMUF_NFQ   4
#define VC_CURMUF_AE    0.5f
#define VC_CURMUF_A1    0.5f
#define VC_CURMUF_A2    0.5f
#define VC_CURMUF_DZF   2000.0f
#define VC_CURMUF_BEX   9.5f

struct CurmufHops {
    float xhops, psi, cpsi, spsi, del_, sphe, secp;
};

// Shared E/F1/F2 geometry: takeoff angle, hop count and the initial secant.
__device__ inline CurmufHops curmuf_secant_hops_(float gcdkm, float del_deg,
                                                 float hp, float ht,
                                                 bool single_div) {
    CurmufHops o;
    float dele = pymax(0.0f, del_deg);
    float dl = dele * VC_D2R;
    float phe = asin32(VC_RZ * cos32(dl) / (VC_RZ + hp));
    int nhops = (int)(0.5f * gcdkm / ((VC_PIO2 - dl - phe) * VC_RZ));
    o.xhops = (float)(nhops + 1);
    if (single_div)
        o.psi = gcdkm / ((2.0f * VC_RZ) * o.xhops);
    else
        o.psi = div32(gcdkm, 2.0f * VC_RZ) / o.xhops;
    o.cpsi = cos32(o.psi);
    o.spsi = sin32(o.psi);
    float tanp = o.spsi / (1.0f - o.cpsi + div32(hp, VC_RZ));
    phe = atan32(tanp);
    dl = VC_PIO2 - phe - o.psi;
    float cdel = cos32(dl);
    o.sphe = VC_RZ * cdel / (VC_RZ + ht);
    o.secp = 1.0f / sqrt32(1.0f - o.sphe * o.sphe);
    o.del_ = dl;
    return o;
}

// state: LayerState; muf: MufState (outputs); f2d: the (16,6,6) F2D table.
// gcdkm/gcd: great-circle distance km/rad; amind: min takeoff angle (deg);
// ssn: sunspot number; it: hour slot.  Mutates muf (and FS(1,KSX)).
__device__ inline void curmuf(LayerState* state, MufState* muf,
                              const float* f2d, float gcdkm, float gcd,
                              float amind, float ssn, int it, int iedp) {
    float* fi = state->fi;
    float* yi = state->yi;
    float* hi = state->hi;

    int kt = 1;
    int kfx = state->kfx;
    int kr = (kfx < 2) ? 1 : ((kfx == 2) ? 2 : 3);
    // select the controlling sample area KS
    int ks;
    if (kr < 2) {                                       // 125
        ks = kt;
    } else if (kr == 2) {                               // 120: FI(1) test
        ks = (fi[1 * 6 + kt] - fi[1 * 6 + kr] > 0.0f) ? kr : kt;
    } else {                                            // 130
        if (fabsf(fi[3 * 6 + kt] - fi[3 * 6 + kr]) - 0.01f <= 0.0f) {
            ks = (fi[1 * 6 + kt] - fi[1 * 6 + kr] > 0.0f) ? kr : kt;
        } else {                                        // 132
            float d = fi[3 * 6 + kt] - fi[3 * 6 + kr];
            if (d < 0.0f)                               // 125
                ks = kt;
            else if (d > 0.0f)                          // 135
                ks = kr;
            else                                        // ==0 -> 120
                ks = (fi[1 * 6 + kt] - fi[1 * 6 + kr] > 0.0f) ? kr : kt;
        }
    }

    lecden(state, ks);

    // tangent (penetration) frequencies for each layer
    float xte = 1.0f / sqrt32(1.0f + VC_CURMUF_AE * yi[1 * 6 + ks]
                                   / hi[1 * 6 + ks]);
    float fxe = xte * fi[1 * 6 + ks];
    float fx1 = 0.0f;
    float xt1;
    if (fi[2 * 6 + ks] > 0.0f) {                        // 145
        xt1 = 1.0f / sqrt32(1.0f + VC_CURMUF_A1 * yi[2 * 6 + ks]
                                 / hi[2 * 6 + ks]);
        fx1 = xt1 * fi[2 * 6 + ks];
    }
    float xt2 = 1.0f / sqrt32(1.0f + VC_CURMUF_A2 * yi[3 * 6 + ks]
                                   / hi[3 * 6 + ks]);
    float fx2 = xt2 * fi[3 * 6 + ks];
    // force F2MUF to approach MUF(0) at short range
    if (gcdkm - VC_CURMUF_DZF < 0.0f) {                 // 146
        float a = -1.0f + 1.0f / xt2;
        float beta = 1.0f + a * exp32(div32(-VC_CURMUF_BEX * gcdkm,
                                            VC_CURMUF_DZF));
        fx2 = beta * fx2;
        xt2 = xt2 * beta;
    }

    // ---- E layer MUF ----
    float hpe, hte;
    if (iedp < 0) {                                     // 155
        gethp(state, fxe, &hpe, &hte);
    } else {                                            // 160 analytic E
        hte = 104.25f;
        hpe = 125.30f;
    }
    CurmufHops h = curmuf_secant_hops_(gcdkm, amind, hpe, hte, false);
    muf->emuf[it] = fxe * h.secp;
    float dele = h.del_ * VC_R2D;
    muf->sigl[1] = pymax(0.01f, 0.1f * muf->emuf[it]);
    muf->sigu[1] = muf->sigl[1];
    muf->delmuf[1] = dele;
    muf->hpmuf[1] = hpe;
    muf->htmuf[1] = hte;
    muf->fvmuf[1] = fxe;
    muf->afmuf[1] = 0.0f;
    muf->nhopmf[1] = (int)h.xhops;
    muf->yfot[1] = muf->emuf[it] - 1.28f * muf->sigl[1];
    muf->yhpf[1] = muf->emuf[it] + 1.28f * muf->sigu[1];
    muf->ymuf[1] = muf->emuf[it];

    // ---- F2 layer MUF ----
    float hp2, ht2;
    if (iedp < 0) {                                     // 170
        gethp(state, fx2, &hp2, &ht2);
    } else {                                            // 175 parabolic E/F2
        ht2 = hi[3 * 6 + ks] - yi[3 * 6 + ks]
            + yi[3 * 6 + ks] * (1.0f - sqrt32(1.0f - xt2 * xt2));
        if (fi[2 * 6 + ks] > 0.0f) {                    // 170
            gethp(state, fx2, &hp2, &ht2);
        } else {                                        // 176
            hp2 = hi[3 * 6 + ks] - yi[3 * 6 + ks] + bendy(state, 3, ks, fx2)
                + (pen(state, 1, ks, fx2) - 2.0f * yi[1 * 6 + ks]);
        }
    }
    h = curmuf_secant_hops_(gcdkm, amind, hp2, ht2, true);
    float fob2 = fx2 * h.secp;
    int ntry = 0;
    float xhp = div32(hp2 - ht2, VC_RZ);
    float hpx2 = hp2;
    while (true) {                          // 190 Martyn-corrected iteration
        float fob1 = fob2;
        float xmut = h.sphe * h.sphe;
        float xfsq = fob1 * fob1 / (fi[3 * 6 + ks] * fi[3 * 6 + ks]);
        float sph = xfsq * xmut * xhp * (ht2 + 2.0f * (VC_RZ + ht2) * xhp);
        hpx2 = hp2 + sph;
        float tanp = h.spsi / (1.0f - h.cpsi + div32(hpx2, VC_RZ));
        float phe = atan32(tanp);
        h.del_ = VC_PIO2 - phe - h.psi;
        float cdel = cos32(h.del_);
        h.sphe = VC_RZ * cdel / (VC_RZ + ht2);
        h.secp = 1.0f / sqrt32(1.0f - h.sphe * h.sphe);
        fob2 = fx2 * h.secp;
        ntry += 1;
        if (fabsf(fob2 - fob1) - VC_CURMUF_FQDEL <= 0.0f)   // 200 converged
            break;
        if (VC_CURMUF_NFQ - ntry < 0)                       // 195 out of tries
            break;
    }
    muf->f2muf[it] = fob2;
    float del2 = h.del_ * VC_R2D;
    float freq = 0.9f * muf->f2muf[it];
    float sig = f2dis(f2d, muf->f2muf[it], ssn, state->clat[ks], freq,
                      state->clck[ks]);
    muf->sigl[3] = pymax(0.01f, sig);
    freq = 1.1f * muf->f2muf[it];
    sig = f2dis(f2d, muf->f2muf[it], ssn, state->clat[ks], freq,
                state->clck[ks]);
    muf->sigu[3] = pymax(0.01f, sig);
    muf->delmuf[3] = del2;
    muf->hpmuf[3] = hpx2;
    muf->htmuf[3] = ht2;
    muf->fvmuf[3] = fx2;
    muf->afmuf[3] = 0.0f;
    muf->nhopmf[3] = (int)h.xhops;
    muf->yfot[3] = muf->f2muf[it] - 1.28f * muf->sigl[3];
    muf->ymuf[3] = muf->f2muf[it];
    muf->yhpf[3] = muf->f2muf[it] + 1.28f * muf->sigu[3];

    // ---- F1 layer MUF (or copy E when no F1) ----
    if (fi[2 * 6 + ks] <= 0.0f) {                       // 210 no F1: F1 = E
        muf->f1muf[it] = muf->emuf[it];
        muf->sigl[2] = muf->sigl[1];
        muf->sigu[2] = muf->sigu[1];
        muf->delmuf[2] = muf->delmuf[1];
        muf->hpmuf[2] = muf->hpmuf[1];
        muf->htmuf[2] = muf->htmuf[1];
        muf->fvmuf[2] = muf->fvmuf[1];
        muf->afmuf[2] = muf->afmuf[1];
        muf->yfot[2] = muf->yfot[1];
        muf->ymuf[2] = muf->ymuf[1];
        muf->yhpf[2] = muf->yhpf[1];
        muf->nhopmf[2] = muf->nhopmf[1];
    } else {                                            // 215
        float hp1, ht1;
        gethp(state, fx1, &hp1, &ht1);
        h = curmuf_secant_hops_(gcdkm, amind, hp1, ht1, false);
        fob2 = fx1 * h.secp;
        ntry = 0;
        xhp = div32(hp1 - ht1, VC_RZ);
        float hpy2 = hp1;
        while (true) {                                  // 225 iterate F1MUF
            float fob1 = fob2;
            float xfsq = fob1 * fob1 / (fi[2 * 6 + ks] * fi[2 * 6 + ks]);
            float xmut = h.sphe * h.sphe;
            float sph = xfsq * xmut * xhp * (ht1 + 2.0f * (VC_RZ + ht1) * xhp);
            hpy2 = hp1 + sph;
            float tanp = h.spsi / (1.0f - h.cpsi + div32(hpy2, VC_RZ));
            float phe = atan32(tanp);
            h.del_ = VC_PIO2 - phe - h.psi;
            float cdel = cos32(h.del_);
            h.sphe = VC_RZ * cdel / (VC_RZ + ht1);
            h.secp = 1.0f / sqrt32(1.0f - h.sphe * h.sphe);
            fob2 = fx1 * h.secp;
            ntry += 1;
            if (fabsf(fob2 - fob1) - VC_CURMUF_FQDEL <= 0.0f)   // 235
                break;
            if (VC_CURMUF_NFQ - ntry < 0)                       // 230
                break;
        }
        muf->f1muf[it] = fob2;
        float del1 = h.del_ * VC_R2D;
        muf->sigl[2] = pymax(0.01f, 0.1f * muf->f1muf[it]);
        muf->sigu[2] = muf->sigl[2];
        muf->delmuf[2] = del1;
        muf->hpmuf[2] = hpy2;
        muf->htmuf[2] = ht1;
        muf->fvmuf[2] = fx1;
        muf->afmuf[2] = 0.0f;
        muf->nhopmf[2] = (int)h.xhops;
        muf->yfot[2] = muf->f1muf[it] - 1.28f * muf->sigl[2];
        muf->ymuf[2] = muf->f1muf[it];
        muf->yhpf[2] = muf->f1muf[it] + 1.28f * muf->sigu[2];
    }

    // ---- sporadic-E MUF (probability of reflection 0.5) ----
    float dels = pymax(0.0f, amind);
    muf->esmuf[it] = 1000.0f;
    int ksx = 0;
    float tdelx = 0.0f, secpx = 0.0f;
    float hop = 0.0f;
    for (int kk = 1; kk <= state->km; ++kk) {           // DO 270
        if (state->fs[2 * 6 + kk] <= 0.0f)
            continue;
        float dl = dels * VC_D2R;
        float phe = asin32(VC_RZ * cos32(dl) / (VC_RZ + state->hs[kk]));
        int nhops = (int)(0.5f * gcdkm / ((VC_PIO2 - dl - phe) * VC_RZ));
        hop = (float)(nhops + 1);
        float psi = 0.5f * gcd / hop;
        float tdel = (cos32(psi) - VC_RZ / (VC_RZ + state->hs[kk]))
                   / sin32(psi);
        float cdel = 1.0f / sqrt32(1.0f + tdel * tdel);
        float sphe = VC_RZ * cdel / (VC_RZ + state->hs[kk]);
        float secp = 1.0f / sqrt32(1.0f - sphe * sphe);
        float esd = state->fs[2 * 6 + kk] * secp;
        if (muf->esmuf[it] - esd > 0.0f) {              // 265
            muf->esmuf[it] = esd;
            ksx = kk;
            tdelx = tdel;
            secpx = secp;
        }
    }
    if (muf->esmuf[it] - 1000.0f >= 0.0f) {             // 271 no Es
        muf->delmuf[4] = 0.0f;
        muf->hpmuf[4] = 0.0f;
        muf->htmuf[4] = 0.0f;
        muf->fvmuf[4] = 0.0f;
        muf->afmuf[4] = 0.0f;
        muf->yfot[4] = 0.0f;
        muf->ymuf[4] = 0.0f;
        muf->yhpf[4] = 0.0f;
        muf->sigl[4] = 0.0f;
        muf->sigu[4] = 0.0f;
        muf->nhopmf[4] = 0;
        muf->esmuf[it] = 0.0f;
    } else {                                            // 272
        muf->delmuf[4] = atan32(tdelx) * VC_R2D;
        muf->hpmuf[4] = state->hs[ksx];
        muf->htmuf[4] = muf->hpmuf[4];
        muf->fvmuf[4] = state->fs[2 * 6 + ksx];
        muf->afmuf[4] = 0.0f;
        muf->nhopmf[4] = (int)hop;
        muf->ymuf[4] = muf->esmuf[it];
        float xes = fi[1 * 6 + ks] - state->fs[1 * 6 + ksx];
        float zdif = 0.1f;
        float fzero = 0.1f;
        muf->sigu[4] = div32(state->fs[3 * 6 + ksx] - state->fs[2 * 6 + ksx],
                             1.28f);
        float sigb = div32(state->fs[2 * 6 + ksx] - state->fs[1 * 6 + ksx],
                           1.28f);
        float sigz = div32(state->fs[2 * 6 + ksx] - fzero, 3.1f);
        if (xes + zdif < 0.0f)                          // 315 lower decile = E
            muf->sigl[4] = sigb;
        else if (xes - zdif < 0.0f)                     // 325 blend
            muf->sigl[4] = sigb + div32(xes + zdif, 2.0f * zdif)
                                * (muf->sigu[4] - sigb);
        else                                            // 330
            muf->sigl[4] = muf->sigu[4];
        muf->sigl[4] = pymin(muf->sigl[4], sigz);       // 335
        state->fs[1 * 6 + ksx] = state->fs[2 * 6 + ksx]
                               - 1.28f * muf->sigl[4];
        muf->sigl[4] = muf->sigl[4] * secpx;            // oblique
        muf->sigu[4] = muf->sigu[4] * secpx;
        muf->yfot[4] = muf->esmuf[it] - 1.28f * muf->sigl[4];
        muf->yhpf[4] = muf->esmuf[it] + 1.28f * muf->sigu[4];
    }

    // ---- circuit MUF (Es excluded) ----
    muf->fot[it] = pymax(pymax(muf->yfot[1], muf->yfot[2]), muf->yfot[3]);
    muf->allmuf[it] = pymax(pymax(muf->emuf[it], muf->f1muf[it]),
                            muf->f2muf[it]);
    muf->hpf[it] = pymax(pymax(muf->yhpf[1], muf->yhpf[2]), muf->yhpf[3]);
    if (muf->emuf[it] - muf->allmuf[it] >= 0.0f) {      // 275
        muf->angmuf[it] = muf->delmuf[1];
        muf->modmuf = 1;
    } else if (muf->f1muf[it] - muf->allmuf[it] >= 0.0f) {  // 285
        muf->angmuf[it] = muf->delmuf[2];
        muf->modmuf = 2;
    } else {                                            // 290
        muf->angmuf[it] = muf->delmuf[3];
        muf->modmuf = 3;
    }
}

#endif  // VOACAP_CURMUF_CUH
