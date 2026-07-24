// GENOIS — atmospheric + galactic + man-made noise combined into one
// distribution (ITS Report 87-212); the CUDA C twin of stage_a/genois.py.
//
// Each component is converted from (median, decile pair) to a linear mean A
// and variance V, those are summed, and the sum is converted back assuming
// the total is itself lognormal (Spaulding).  Caruana's 2007 modification
// caps the recovered sigmas at what the plain power sum RNSE can support once
// either atmospheric decile exceeds 12 dB, and records that in METHOD_USED.
//
// REAL*8 REGIONS, MIRRORED EXACTLY AND NOWHERE ELSE (genois.for declares them
// so, and Stage A's F64 marks them): the twelve A/V accumulators AU?/VU?/AL?/
// VL?, their sums AU/VU/AL/VL, SIGTSQU/SIGTSQL, and Caruana's DXX/DXX2/DXX3.
// Every EXP feeding those sums is REAL*4 and only its RESULT widens; XRNSE,
// DU, DL and everything downstream are REAL*4 again.  Two constants that look
// identical are not: 4.34294 (binary32, promoted in XRNSE and DU/DL) and
// 4.34294d0 (a true double, used only in Caruana's test) — both are written.
// `**2` is an INTEGER exponent, so it is a multiply, never pow32.
//
// Where Stage A RAISES, this returns a status with the state exactly as Stage
// A leaves it at the raise (nothing has been written at either point):
//   0 = ok, 1 = NYEAR ' 2089' selects the unported GENOIS_OLD,
//   2 = GAIN's out-of-bounds table read (Stage A: IndexError),
//   3 = GAIN's unported area-coverage antenna (Stage A: NotImplementedError).
// `nyear_is_2089` is the caller's already-made string test.
//
// The /TWO/ FAM/DUD tables and the /cantenna/ tables are SHARED, not
// per-thread, and pass flat exactly as genfam.cuh / gain.cuh document.
//
// Rule 4b: every division by a literal goes through div32; the two constant
// DOUBLE divisions go through __ddiv_rn for the same reason.

#ifndef VOACAP_GENOIS_CUH
#define VOACAP_GENOIS_CUH

#include "f32math.cuh"
#include "gain.cuh"
#include "genfam.cuh"
#include "state.cuh"

// DATA DFAC,BFAC,CFAC /7.87384, 30.99872, 5.56765/
#define VG_DFAC   7.87384f
#define VG_BFAC   30.99872f
#define VG_CFAC   5.56765f
#define VG_DB     4.34294f      // 10/ln(10): nepers <-> dB (REAL*4)
#define VG_DB8    4.34294       // the 4.34294d0 literal — a DIFFERENT constant
#define VG_INV_DB 0.23026f      // 1/4.34294
#define VG_KTB    204.0f        // dB from kTb (290 K, 1 Hz) to 1 W
#define VG_TWELVE 12.0f         // the decile above which Spaulding breaks down

// ``x**2`` with an INTEGER exponent — gfortran emits a multiply, not powf.
__device__ inline float genois_sq_(float x) { return x * x; }

// st_*: the shared /TWO/ tables and the F2D(1,1,1) scalar; state: LayerState
// (/RON/ FI, KFX); anois: AnoisState (ATNU/ATNY/CC/TM/KJ/JK in, RCNSE and the
// five spread terms out); ton: TonState; zon: ZonState (EFF out); diag:
// NoiseDiagState; the cant_* pointers are /cantenna/; freq: MHz; noise: the
// NOISE card value; rlat: receiver latitude (radians).
__device__ inline int genois(const float* fam, const float* dud, float f2d111,
                             const LayerState* state, AnoisState* anois,
                             TonState* ton, ZonState* zon,
                             NoiseDiagState* diag, const float* cant_array,
                             const float* cant_aeff, const int* cant_iats,
                             const float* cant_xfqs, const float* cant_xfqe,
                             const float* cant_offazim, int numants,
                             float freq, int noise, float rlat,
                             int nyear_is_2089) {
    if (nyear_is_2089)
        return 1;                               // GENOIS_OLD is not ported

    float dume = pymin(freq, 55.0f);            // AMIN1(FREQ,55.)

    // ---- atmospheric ----------------------------------------------------
    // GENFAM's outputs alias the /ANOIS/ DU/DL/SIGM/SYGU/SYGL in the Fortran,
    // but all five are reassigned before anything reads them, so the
    // intermediate store is dead and they stay local here.
    float atnos, dua, dla, sma, sua, sla;
    if (f2d111 < 0.0f) {            // IF (F2D(1,1,1)) 85,90,90 — negative only
        // 85: no long-term data base.  ATNOS=204. is immediately overwritten
        // with 0. on the next line, so the atmospheric term contributes
        // nothing once the -204 dB shift is applied.
        atnos = 0.0f;
        dua = 9.0f;
        dla = 7.0f;
        sma = 3.0f;
        sua = 1.5f;
        sla = 1.5f;
    } else {
        float atnz, du_, dl_, sigm_, sygu_, sygl_;
        genfam(fam, dud, rlat, anois->kj, dume, anois->atnu,
               &atnz, &du_, &dl_, &sigm_, &sygu_, &sygl_);
        float atnx, dx, dq, sigz, sigx, sigsq;
        genfam(fam, dud, rlat, anois->jk, dume, anois->atny,
               &atnx, &dx, &dq, &sigz, &sigx, &sigsq);
        // interpolate between the two local-time blocks
        float slop = div32(fabsf(anois->cc - anois->tm), 4.0f);
        atnos = atnz + (atnx - atnz) * slop;
        dua = du_ + (dx - du_) * slop;
        dla = dl_ + (dq - dl_) * slop;
        sma = sigm_ + (sigz - sigm_) * slop;
        sua = sygu_ + (sigx - sygu_) * slop;
        sla = sygl_ + (sigsq - sygl_) * slop;
    }

    // (DUA/DFAC)**2 = (DUA/1.282)**2 / (2*4.34294**2), and DUA*DUA/BFAC =
    // (DUA/1.282)**2 / 4.34294**2 — the lognormal mean and variance of a
    // component with decile ratio DUA and median ATNOS.
    float t = div32(dua, VG_DFAC);
    double au1 = (double)exp32(genois_sq_(t) + div32(atnos, VG_DB));
    double vu1 = au1 * au1
               * (double)(exp32(div32(dua * dua, VG_BFAC)) - 1.0f);
    t = div32(dla, VG_DFAC);
    double al1 = (double)exp32(genois_sq_(t) + div32(atnos, VG_DB));
    double vl1 = al1 * al1
               * (double)(exp32(div32(dla * dla, VG_BFAC)) - 1.0f);

    // ---- galactic -------------------------------------------------------
    float gnos = 52.0f - 23.0f * log10_32(freq);
    float dug = 2.0f, dlg = 2.0f;
    float smg = 0.5f;
    float sug = 0.2f, slg = 0.2f;
    int mode_n2 = 1;
    t = div32(dug, VG_DFAC);
    double au2 = (double)exp32(genois_sq_(t) + div32(gnos, VG_DB));
    double vu2 = au2 * au2
               * (double)(exp32(div32(dug * dug, VG_BFAC)) - 1.0f);
    t = div32(dlg, VG_DFAC);
    double al2 = (double)exp32(genois_sq_(t) + div32(gnos, VG_DB));
    double vl2 = al2 * al2
               * (double)(exp32(div32(dlg * dlg, VG_BFAC)) - 1.0f);
    if (freq <= state->fi[3 * 6 + state->kfx]) {   // FI(3,KFX) = foF2
        mode_n2 = 0;                               // cannot penetrate — drop
        gnos = 0.0f;
        au2 = 0.0;
        vu2 = 0.0;
        al2 = 0.0;
        vl2 = 0.0;
    }

    // ---- man-made -------------------------------------------------------
    // Man-made noise at 1 MHz and its decay exponent, by CCIR environment:
    //   #1 industrial #2 residential #3 rural #4 remote #5 noisy #6 quiet.
    // 5 and 6 are unreachable (MA is clamped with MIN0(4,MA)); kept because
    // the DATA statements carry them.
    const float xnint[6] = {76.8f, 72.5f, 67.2f, 53.6f, 83.2f, 65.2f};
    const float conn[6] = {27.7f, 27.7f, 27.7f, 28.6f, 37.5f, 29.1f};
    int man = noise;
    float xnois = (float)man;
    int ma = (man < 0) ? -man : man;
    float znoise = xnois;
    if (man > 0) {                  // IF (MAN) 120,114,115 -> 115
        // a 3 MHz level in dB below 1 W: to Fam at 1 MHz, then to FREQ
        xnois = VG_KTB - xnois + 13.22f;
        xnois = xnois - 27.7f * log10_32(freq);
    } else {
        if (man == 0)               // 114: -164 dBW on the user card
            ma = 4;
        ma = (ma < 4) ? ma : 4;     // 120: negative = category index
        xnois = xnint[ma - 1] - conn[ma - 1] * log10_32(freq);
        znoise = VG_KTB - xnint[ma - 1] + conn[ma - 1] * log10_32(3.0f);
    }
    float dum = 9.7f;
    float dlm = 6.0f;
    float sumn = 1.5f;              // SUM in the Fortran (a variable)
    float smm = 5.4f;
    float slm = 1.5f;
    t = div32(dum, VG_DFAC);
    double au3 = (double)exp32(genois_sq_(t) + div32(xnois, VG_DB));
    double vu3 = au3 * au3
               * (double)(exp32(div32(dum * dum, VG_BFAC)) - 1.0f);
    t = div32(dlm, VG_DFAC);
    double al3 = (double)exp32(genois_sq_(t) + div32(xnois, VG_DB));
    double vl3 = al3 * al3
               * (double)(exp32(div32(dlm * dlm, VG_BFAC)) - 1.0f);

    double vu = vu1 + vu2 + vu3;
    double vl = vl1 + vl2 + vl3;
    double au = au1 + au2 + au3;
    double al = al1 + al2 + al3;

    // ---- receiver antenna efficiency, for every possible mode ------------
    float gdum, reff;
    int oob = 0;
    antenna_gain(cant_array, cant_aeff, cant_iats, cant_xfqs, cant_xfqe,
                 cant_offazim, numants, 2, 0.0f, freq, &gdum, &reff, &oob);
    if (oob != 0)
        return (oob == 1) ? 2 : 3;  // Stage A raises here; nothing written yet
    for (int im = 1; im <= 6; ++im)             // DO 196 IM=1,6
        zon->eff[im] = reff;

    // ---- combine ---------------------------------------------------------
    atnos = atnos - VG_KTB;         // dB above kTb -> dBW in 1 Hz
    gnos = gnos - VG_KTB;
    xnois = xnois - VG_KTB;
    diag->xn_atnos = atnos;
    diag->xn_galactic = gnos;
    diag->xn_mmn = xnois;

    // the plain power sum, which Caruana's test uses as the sanity bound
    float rnse = VG_DB * log32(pow32(10.0f, atnos * 0.1f)
                               + pow32(10.0f, gnos * 0.1f)
                               + pow32(10.0f, xnois * 0.1f));

    double sigtsqu = log(1.0 + vu / (au * au));
    double sigtsql = log(1.0 + vl / (al * al));
    int method_used = 2;
    if (!(dua <= VG_TWELVE && dla <= VG_TWELVE)) {
        // Spaulding is about to break down: cap each sigma at what the power
        // sum can support.  A non-positive DXX2/DXX3 means there is nothing
        // to cap against, so the original value stands.
        double dxx = (double)rnse;
        dxx = dxx + 204.0;
        double dxx2 = 2.0 * (log(au) - __ddiv_rn(dxx, VG_DB8));
        double dxx3 = 2.0 * (log(al) - __ddiv_rn(dxx, VG_DB8));
        if (sigtsqu > dxx2 && dxx2 > 0.0) {
            method_used = 1;
            sigtsqu = dxx2;
        }
        if (sigtsql > dxx3 && dxx3 > 0.0) {
            method_used = 1;
            sigtsql = dxx3;
        }
    }

    float xrnse = (float)((double)VG_DB * (log(au) - __ddiv_rn(sigtsqu, 2.0))
                          - 204.0);
    float du = (float)((double)VG_CFAC * sqrt(sigtsqu));    // upper decile
    float dl = (float)((double)VG_CFAC * sqrt(sigtsql));    // lower decile

    // ---- prediction errors ------------------------------------------------
    // QP* is each component's share of the total power, so the component
    // sigmas combine weighted by how much of the noise they actually are.
    float qpa = pow32(10.0f, (atnos - xrnse) * 0.1f);
    float qpg = pow32(10.0f, (gnos - xrnse) * 0.1f);
    if (mode_n2 == 0)
        qpg = 0.0f;
    float qpm = pow32(10.0f, (xnois - xrnse) * 0.1f);
    float sigm = sqrt32(genois_sq_(qpa * sma) + genois_sq_(qpg * smg)
                        + genois_sq_(qpm * smm));

    float pv = qpa * exp32((dua - du) * VG_INV_DB);
    float sygu1 = genois_sq_(pv * sua) + genois_sq_((pv - qpa) * sma);
    pv = qpg * exp32((dug - du) * VG_INV_DB);
    float sygu2 = genois_sq_(pv * sug) + genois_sq_((pv - qpg) * smg);
    pv = qpm * exp32((dum - du) * VG_INV_DB);
    float sygu3 = genois_sq_(pv * sumn) + genois_sq_((pv - qpm) * smm);
    float sygu = sqrt32(sygu1 + sygu2 + sygu3);

    pv = qpa * exp32((dla - dl) * VG_INV_DB);
    float sygl1 = genois_sq_(pv * sla) + genois_sq_((pv - qpa) * sma);
    pv = qpg * exp32((dlg - dl) * VG_INV_DB);
    float sygl2 = genois_sq_(pv * slg) + genois_sq_((pv - qpg) * smg);
    pv = qpm * exp32((dlm - dl) * VG_INV_DB);
    float sygl3 = genois_sq_(pv * slm) + genois_sq_((pv - qpm) * smm);
    float sygl = sqrt32(sygl1 + sygl2 + sygl3);

    // The 1991 long-path correction dropped the receiver efficiency from
    // RCNSE (it is added to the signal with the gain instead), so this is
    // XRNSE unmodified.
    anois->rcnse = xrnse;
    anois->du = du;
    anois->dl = dl;
    anois->sigm = sigm;
    anois->sygu = sygu;
    anois->sygl = sygl;
    ton->gnos = gnos;
    ton->xnoise = xnois;
    ton->znoise = znoise;
    diag->mode_n2 = mode_n2;
    diag->method_used = method_used;
    return 0;
}

#endif  // VOACAP_GENOIS_CUH
