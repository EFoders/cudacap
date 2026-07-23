// RELBIL — reliability: pick the best mode, then combine all of them; the
// CUDA C twin of stage_a/relbil.py.  Three separate things happen:
//
// 1. Per-mode reliability.  The required S/N is placed on a two-sided normal
//    whose LOWER spread combines the noise UPPER decile with the signal LOWER
//    decile and vice versa — the crossing is deliberate (a bad hour is high
//    noise AND low signal).  A virtual height at or below 70 km is not a
//    mode; it gets reliability 0.001 and D10R/D50R/D90R are NOT assigned, so
//    a dead-only mode leaves the persisted /DON/ triple from the previous
//    frequency standing.  That stale read is part of the answer.
// 2. Most-reliable-mode selection, in two passes.  The first finds the
//    outright maximum; the second walks the list again and replaces it with
//    any mode within 0.05 that has fewer hops (or, at equal hops, a better
//    median S/N).  The first pass exists because the second compares against
//    a running best it has already moved.
// 3. The combined distribution.  With more than one mode the signals add as
//    powers with random phase — four separate sums (median, both deciles,
//    field strength), each referred to its own maximum term, terms more than
//    100 dB down dropped.
//
// MODE(IFX) is a LAYTYP label in Stage A; here it is the LAYTYP INDEX
// (state.py LABEL_FIELDS), i.e. NMODE(IR) - 1.  Where Stage A RAISES on an
// out-of-range NMODE (the Fortran reads LAYTYP out of bounds), this returns
// status 1 with the state exactly as Stage A leaves it at the raise —
// GAINT..CPROB written, MODE/NHP/XNYNOIS onward not.
//
// The /cantenna/ + /pantenna/ tables are SHARED and pass flat as pwrdb.cuh
// documents.  float32; ALOG10 and the 10.** powers carry the libm floor.

#ifndef VOACAP_RELBIL_CUH
#define VOACAP_RELBIL_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "fnorml.cuh"
#include "pwrdb.cuh"
#include "state.cuh"

#define VR_XEPS        0.05f
#define VR_MIN_REL     0.00000001f
#define VR_MIN_SPREAD  0.2f
#define VR_MAX_SPREAD  30.0f
#define VR_NO_MODE_HT  70.0f
#define VR_FLOOR_DB    -10.0f       // ZEXP floor: terms 100 dB down dropped

// Z in units of the decile spread, the Fortran's ``Z/(D??R/1.28)``.  The
// division is unguarded there and a zero spread yields IEEE infinity, which
// FNORML's 5-sigma clamp absorbs exactly as it would any large Z.
__device__ inline float relbil_in_sigmas_(float z, float spread) {
    return z / div32(spread, VC_DCL);
}

// allm: AllModeState (CREL/RELY/NREL written back); anois: AnoisState;
// sigd: SigdState (DSLF/DSUF used as scratch); son: SonState, the output
// column; the cant_* pointers are /cantenna/ + /pantenna/ for PWRDB.
// ifx: frequency index; freq: MHz; rsn: required S/N; lufp: the required
// reliability in percent.  Returns 0, or 1 where Stage A raises IndexError.
__device__ inline int relbil(AllModeState* allm, const AnoisState* anois,
                             SigdState* sigd, SonState* son,
                             const int* cant_iats, const float* cant_xfqs,
                             const float* cant_xfqe, const float* cant_pwrdba,
                             int numants, int ifx, float freq, float rsn,
                             int lufp) {
    int inum = allm->nmmod;
    if (inum <= 0)
        return 0;
    float du2 = anois->du * anois->du;
    float dl2 = anois->dl * anois->dl;

    // ---- 1. per-mode reliability ------------------------------------------
    for (int im = 1; im <= inum; ++im) {            // DO 310
        if (allm->hp[im] <= VR_NO_MODE_HT) {
            allm->crel[im] = 0.001f;
            allm->rely[im] = 0.001f;
            continue;
        }
        sigd->dslf = allm->tllow[im];
        sigd->dsuf = allm->tlhgh[im];
        // the crossing is deliberate: a bad hour is high noise AND low signal
        son->d10r = sqrt32(du2 + sigd->dslf * sigd->dslf);
        son->d50r = allm->sn[im];
        son->d90r = sqrt32(dl2 + sigd->dsuf * sigd->dsuf);
        float z = rsn - son->d50r;
        if (z <= 0.0f)
            z = relbil_in_sigmas_(z, son->d10r);
        else
            z = relbil_in_sigmas_(z, son->d90r);
        allm->rely[im] = 1.0f - fnorml(z);
        allm->crel[im] = 1000.0f;                   // "CREL is not used now"
    }

    // ---- 2. most reliable mode --------------------------------------------
    // First pass: the outright maximum.  The 140 loop below cannot be trusted
    // to find it on its own — it compares against a best it has already
    // moved — so it starts from this one instead.
    int irmax = 1;
    for (int im = 2; im <= inum; ++im)              // DO 10
        if (allm->rely[im] > allm->rely[irmax])
            irmax = im;
    int ir = irmax;
    float xrel = allm->rely[ir];
    float xhn = allm->hn[ir];
    float xsn = allm->sn[ir];
    for (int im = 1; im <= inum; ++im) {            // DO 140
        if (im == irmax)
            continue;
        if (xrel < VR_MIN_REL)
            xrel = VR_MIN_REL;
        float reltest = fabsf(allm->rely[im] - xrel);
        // IRmethod 2 would divide RELtest by XREL for a percentage test
        if (reltest >= VR_XEPS)
            continue;                               // not close to the best
        bool better;
        if (fabsf(xhn - allm->hn[im]) <= VR_XEPS)
            better = xsn < allm->sn[im];            // equal hops: better S/N
        else
            better = xhn > allm->hn[im];            // this one has fewer hops
        if (better) {                               // 139
            ir = im;
            xhn = allm->hn[im];
            xsn = allm->sn[im];
            xrel = allm->rely[im];
        }
    }
    // IRmethod 3 would fall back to IRmax when RELY(IR) < .01
    allm->nrel = ir;
    int ilayer = allm->nmode[ir];

    // ---- 3. the frequency's summary ---------------------------------------
    if (inum == 1) {
        son->reliab[ifx] = allm->rely[ir];
        son->dblosl[ifx] = allm->tllow[ir];
        son->dblosu[ifx] = allm->tlhgh[ir];
        son->dbu[ifx] = allm->fldst[ir];
        son->sndb[ifx] = allm->sn[ir];
        son->dbw[ifx] = allm->sigpow[ir];
    } else {
        // add the signals with random phase, i.e. add the powers in watts.
        // Each sum is referred to its own maximum term to keep 10**x in range.
        float xdslw = 0.0f, xsigs = 0.0f, xdsup = 0.0f, xfld = 0.0f;
        float dxsigs = -1000.0f, dxfld = -1000.0f;
        float dxdslw = -1000.0f, dxdsup = -1000.0f;
        for (int iv = 1; iv <= inum; ++iv) {        // DO 369
            dxsigs = pymax(dxsigs, allm->sigpow[iv]);
            dxfld = pymax(dxfld, allm->fldst[iv]);
            dxdslw = pymax(dxdslw, allm->sigpow[iv] - allm->tllow[iv]);
            dxdsup = pymax(dxdsup, allm->sigpow[iv] + allm->tlhgh[iv]);
        }
        for (int im = 1; im <= inum; ++im) {        // DO 370
            float zexp = 0.1f * (allm->sigpow[im] - allm->tllow[im] - dxdslw);
            if (zexp > VR_FLOOR_DB)
                xdslw = xdslw + pow32(10.0f, zexp);
            zexp = 0.1f * (allm->sigpow[im] - dxsigs);
            if (zexp > VR_FLOOR_DB)
                xsigs = xsigs + pow32(10.0f, zexp);
            zexp = 0.1f * (allm->sigpow[im] + allm->tlhgh[im] - dxdsup);
            if (zexp > VR_FLOOR_DB)
                xdsup = xdsup + pow32(10.0f, zexp);
            // field strength is summed separately: the receive antenna gain
            // is already in it and must not be double-counted
            zexp = 0.1f * (allm->fldst[im] - dxfld);
            if (zexp > VR_FLOOR_DB)
                xfld = xfld + pow32(10.0f, zexp);
        }
        float sigmed = -500.0f;
        if (xsigs > 0.0f)
            sigmed = dxsigs + 10.0f * log10_32(xsigs);
        if (xdslw > 0.0f)
            son->dblosl[ifx] = fabsf(sigmed - 10.0f * log10_32(xdslw)
                                     - dxdslw);
        else
            son->dblosl[ifx] = 0.0f;
        son->dblosl[ifx] = pymin(pymax(son->dblosl[ifx], VR_MIN_SPREAD),
                                 VR_MAX_SPREAD);
        if (xdsup > 0.0f)
            son->dblosu[ifx] = fabsf(dxdsup + 10.0f * log10_32(xdsup)
                                     - sigmed);
        else
            son->dblosu[ifx] = 0.0f;
        son->dblosu[ifx] = pymin(pymax(son->dblosu[ifx], VR_MIN_SPREAD),
                                 VR_MAX_SPREAD);
        son->dbw[ifx] = sigmed;
        float delsig = sigmed - allm->sigpow[ir];
        if (xfld > 0.0f)
            son->dbu[ifx] = dxfld + 10.0f * log10_32(xfld);
        else
            son->dbu[ifx] = -500.0f;
        son->sndb[ifx] = allm->sn[ir] + delsig;

        // redo the reliability against the combined distribution
        son->d10r = sqrt32(du2 + son->dblosl[ifx] * son->dblosl[ifx]);
        son->d50r = son->sndb[ifx];
        son->d90r = sqrt32(dl2 + son->dblosu[ifx] * son->dblosu[ifx]);
        if (son->d10r < VR_MIN_SPREAD)
            son->d10r = VR_MIN_SPREAD;
        if (son->d90r > VR_MAX_SPREAD)
            son->d90r = VR_MAX_SPREAD;
        float z = rsn - son->d50r;
        if (z <= 0.0f)
            z = relbil_in_sigmas_(z, son->d10r);
        else
            z = relbil_in_sigmas_(z, son->d90r);
        son->reliab[ifx] = 1.0f - fnorml(z);
    }

    son->gaint[ifx] = allm->tgain[ir];
    son->gainr[ifx] = allm->rgain[ir];
    son->snrlw[ifx] = son->d10r;
    son->snrup[ifx] = son->d90r;
    son->angle[ifx] = allm->b[ir];
    son->vhigh[ifx] = allm->hp[ir];
    son->delay[ifx] = allm->timed[ir];
    son->dblos[ifx] = allm->tloss[ir];
    // 2014: the powers have been summed, so the transmission loss has to be
    // recomputed from the summed signal rather than taken from one mode
    son->dblos[ifx] = pwrdb(cant_iats, cant_xfqs, cant_xfqe, cant_pwrdba,
                            numants, freq) - son->dbw[ifx];
    son->cprob[ifx] = allm->prob[ir];
    if (!(ilayer >= 1 && ilayer <= 5))
        return 1;               // Stage A raises IndexError; LAYTYP has 5
    son->mode[ifx] = ilayer - 1;            // LAYTYP INDEX, not the label
    son->nhp[ifx] = (int)allm->hn[ir];      // REAL->INT truncates toward zero
    son->xnynois[ifx] = anois->rcnse;
    son->du_nois[ifx] = anois->du;
    son->dl_nois[ifx] = anois->dl;
    son->rneff[ifx] = allm->eff[ir];

    // required power gain for the specified reliability.  DATA TME — the
    // standard normal at 5% steps from the median; TME(9) = 1.2815 is the
    // 90th percentile, the reference the ratio is taken against.
    const float tme[10] = {0.0f, 0.1257f, 0.2533f, 0.3853f, 0.5244f,
                           0.6745f, 0.8416f, 1.0364f, 1.2815f, 1.6449f};
    int d = lufp - 50;
    int itm = ((d < 0) ? -d : d) / 5 + 1;
    itm = (itm < 10) ? itm : 10;
    float tmx = div32(tme[itm - 1], tme[8]);
    if (lufp < 50)
        son->snpr[ifx] = -(son->d50r + tmx * son->d90r) + rsn;
    else
        son->snpr[ifx] = -(son->d50r - tmx * son->d10r) + rsn;
    son->snxx[ifx] = rsn - son->snpr[ifx];
    return 0;
}

#endif  // VOACAP_RELBIL_CUH
