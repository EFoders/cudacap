// Per-thread state structs — the CUDA C twins of the Stage A COMMON-block
// dataclasses (allmodestate.py, frqstate.py, ... zonstate.py).
//
// ABI CONTRACT (load-bearing, see cuda/state.py + tests/cuda/
// test_state.py for the offset proof):
//   * fields appear in EXACTLY the dataclass order, same lower-case names;
//   * every member is 4 bytes (float or int) so the struct has no internal
//     padding and Python byte offsets are trivially sequential;
//   * numpy arrays keep their total shape flattened row-major, 1-based
//     padding (unused row/column 0) intact — a (r, c) array is
//     `float x[r * c]` indexed `x[i * c + k]`;
//   * Stage A string label fields (2-char LAYTYP labels) become int arrays
//     holding an index into LAYTYP (" E","F1","F2","ES"," N") = 0..4,
//     or -1 for the empty label "".
//
// Do not reorder fields, do not introduce any non-4-byte member.

#ifndef VOACAP_STATE_CUH
#define VOACAP_STATE_CUH

// COMMON /allMODE/ — every mode found across the whole hop sweep (scalar/
// allmodestate.py).  22 accumulator arrays, slot 1..20 used, 1-based.
struct AllModeState {
    float abps[21];
    float crel[21];
    float fldst[21];
    float hn[21];
    float hp[21];
    float prob[21];
    float rely[21];
    float rgain[21];
    float sigpow[21];
    float sn[21];
    float spro[21];
    float tgain[21];
    float timed[21];
    float tloss[21];
    float b[21];
    float fslos[21];
    float grlos[21];
    float adv[21];
    float obf[21];
    float tllow[21];
    float tlhgh[21];
    float eff[21];
    int nmode[21];
    int nrel;                   // blkdat DATA NREL/1/
    int nmmod;                  // blkdat DATA NMMOD/1/
};

// COMMON /FRQ/ — the frequency complement and the frequency under test
// (scalar/frqstate.py).
struct FrqState {
    float frea[14];             // 1..11 freqs, 12 MUF, 13 LUF
    float frel[30];             // + slot 14 = DECRED's IFREQ flag
    float freq;
    int jmode;
    int itxrcp[3];              // 1-based (3,): [1]=tx area, [2]=rx area
};

// /RON/, /ES/, /MFAC/, /GEOG/, /PSCA/ + SSN scalars — per-sample-area
// ionospheric state (scalar/layerstate.py).  1-based everywhere.
struct LayerState {
    int km;
    int kfx;
    int iedp;
    int month;
    float ssn;
    float clat[6];
    float clong[6];
    float glat[6];
    float gmdip[6];
    float gyz[6];
    float rd[6];
    float fi[4 * 6];            // (4, 6) row-major: fi[layer * 6 + area]
    float yi[4 * 6];            // (4, 6) row-major
    float hi[4 * 6];            // (4, 6) row-major
    float fvert[31 * 6];        // (31, 6) row-major: fvert[freq * 6 + area]
    float hprim[31 * 6];        // (31, 6) row-major
    float htrue[31 * 6];        // (31, 6) row-major
    float afac[31 * 6];         // (31, 6) row-major
    float fs[4 * 6];            // (4, 6) row-major
    float hs[6];
    float f2m3[6];
    float hpf2[6];
    float zenang[6];
    float zenmax[6];
    float fsecv[4];
    float htr[51 * 4];          // (51, 4) row-major: htr[pt * 4 + col]
    float fnsq[51 * 4];         // (51, 4) row-major
    int ielect[4];
    float rat[6];
    float clck[6];
    float abiy[6];
    float artic[6];
    float sigpat[6];
    float epspat[6];
    float psc[5];
};

// COMMON /LOSX/ — the long path's per-ray loss and gain table (scalar/
// losxstate.py).  ARFX and TLSKM are dead in the Fortran; carried for layout.
struct LosxState {
    float andvx[46 * 4];        // (46, 4) row-major: andvx[ray * 4 + area]
    float advx[46 * 4];         // (46, 4) row-major
    float aofx[46 * 4];         // (46, 4) row-major
    float arfx[46 * 4];         // (46, 4) row-major (dead)
    float grlosx[46 * 4];       // (46, 4) row-major
    float tgainx[46 * 4];       // (46, 4) row-major
    float tlskm[46 * 4];        // (46, 4) row-major (dead)
    float efflp[46];            // indexed by ray alone
};

// COMMON /LPATH/ — the long-path branch's own working state (scalar/
// lpathstate.py).  The four scalars are blkdat DATA constants.
struct LpathState {
    float gcdlng;               // DATA 10000.
    float delopt;               // DATA 3.
    float gmin;                 // DATA 3.
    float ymin;                 // DATA .1
    float txrgml[46 * 3];       // (46, 3) row-major: txrgml[ray * 3 + end]
    int ltxrgm[3];              // blkdat DATA LTXRGM/1,1/ in slots 1..2
};

// COMMON /MODES/ — the up-to-six propagation modes for one hop (scalar/
// modesstate.py).
struct ModesState {
    float ghop;
    float delmod[7 * 4];        // (7, 4) row-major: delmod[mode * 4 + area]
    float hpmod[7 * 4];         // (7, 4) row-major
    float htmod[7 * 4];         // (7, 4) row-major
    float fvmod[7 * 4];         // (7, 4) row-major
    float afmod[7 * 4];         // (7, 4) row-major
    int itmod[7 * 4];           // (7, 4) row-major
};

// COMMON /MUFS/ — the circuit MUF/FOT outputs CURMUF writes (scalar/
// mufstate.py).  Hour arrays 1-based (25,), layer arrays 1-based (5,).
struct MufState {
    float emuf[25];
    float f1muf[25];
    float f2muf[25];
    float esmuf[25];
    float allmuf[25];
    float fot[25];
    float xluf[25];
    float hpf[25];
    float angmuf[25];
    int modmuf;
    float sigl[5];
    float sigu[5];
    float delmuf[5];
    float hpmuf[5];
    float htmuf[5];
    float fvmuf[5];
    float afmuf[5];
    int nhopmf[5];
    float yfot[5];
    float yhpf[5];
    float ymuf[5];
};

// COMMON /ANOIS/ — the atmospheric-noise working set (scalar/noisestate.py).
struct AnoisState {
    float atnu;
    float atny;
    float cc;
    float tm;
    float rcnse;
    float du;
    float dl;
    float sigm;
    float sygu;
    float sygl;
    int kj;
    int jk;
};

// COMMON /c_noise/, /c_noises/, /c_method_used/ — GENOIS's breakdown
// (scalar/noisestate.py, NoiseDiagState).
struct NoiseDiagState {
    int mode_n1;
    int mode_n2;
    int mode_n3;
    float xn_atnos;
    float xn_galactic;
    float xn_mmn;
    int method_used;
};

// COMMON /RAYS/ — the ray-set / reflectrix table (scalar/raysstate.py).
struct RaysState {
    int nang;
    float ang[41];
    int ifob[41 * 31 * 6];      // (41, 31, 6) row-major:
                                // ifob[i * 31 * 6 + j * 6 + k]
};

// COMMON /REFLX/ — the per-control-area reflectrix table (scalar/
// reflxstate.py).  IAFTXR is nominally /LOSX/ but lives here in Stage A.
struct ReflxState {
    float delfx[46 * 4];        // (46, 4) row-major: delfx[ray * 4 + area]
    float hpflx[46 * 4];        // (46, 4) row-major
    float htflx[46 * 4];        // (46, 4) row-major
    float gdflx[46 * 4];        // (46, 4) row-major
    float fvflx[46 * 4];        // (46, 4) row-major
    float afflx[46 * 4];        // (46, 4) row-major
    float gml[46 * 4];          // (46, 4) row-major
    float fhp[46 * 4];          // (46, 4) row-major
    float dskpkm[4];
    float delskp[4];
    float hpskp[4];
    float htskp[4];
    float dmaxkm[4];
    float fvskp[4];
    int iskp[4];
    int imode[46 * 4];          // (46, 4) row-major
    float delpen[4 * 6];        // (4, 6) row-major: delpen[layer * 6 + area]
    int iaftxr[4];
};

// COMMON /SIGD/ — signal-distribution and D-E absorption-loss parameters
// (scalar/signalstate.py, SigdState).
struct SigdState {
    float dsl;
    float asm_;                 // Stage A / Fortran name "asm"; renamed only
                                // because `asm` is a C++ keyword.  Same slot.
    float dsu;
    float aglat;
    float dslf;
    float asmf;
    float dsuf;
    float acav;
    float feav;
    float afe;
    float bfe;
    float hnu;
    float htloss;
    float xnuz;
    float xve;
};

// COMMON /TON/ — path-average system loss + noise/reliability accumulators
// (scalar/signalstate.py, TonState).
struct TonState {
    float adj;
    float ads;
    float gnos;
    float got;
    float rel;
    float sl;
    float sls;
    float spr;
    float su;
    float sus;
    float xnoise;
    float znoise;
    int nf;
};

// COMMON /SON/ + /DUDL_NOIS/ + /sncom/ + /cgains/ + /ALPHA/ + the /DON/
// decile triples — the per-FREQUENCY results table (scalar/sonstate.py).
// mdl/mode/moder are Stage A lists of 2-char labels; here each slot is an
// int index into LAYTYP (" E"=0, "F1"=1, "F2"=2, "ES"=3, " N"=4), or -1
// for the empty label "".
struct SonState {
    float angle[14];
    float angler[14];
    float cprob[14];
    float dblos[14];
    float dblosl[14];
    float dblosu[14];
    float dbu[14];
    float delay[14];
    float dbw[14];
    int nhp[14];
    float xnynois[14];
    float probmp[14];
    float reliab[14];
    float sndb[14];
    float snpr[14];
    float snrlw[14];
    float snrup[14];
    float sprob[14];
    float vhigh[14];
    float rneff[14];
    int mdl[14];                // LAYTYP index, -1 = ""
    int mode[14];               // LAYTYP index, -1 = ""
    int moder[14];              // LAYTYP index, -1 = ""
    float du_nois[14];
    float dl_nois[14];
    float snxx[14];
    float gaint[14];
    float gainr[14];
    float d10r;
    float d50r;
    float d90r;
    float d10s;
    float d50s;
    float d90s;
};

// COMMON /TIME/ — the hour under test in every time base (scalar/
// timestate.py).
struct TimeState {
    int it;
    float gmt;
    float utime[25];
    float gmtr;
    float xlmt[25];
    int itim;
    int jtx;
};

// COMMON /ZON/ — the per-mode results table for one frequency and hop
// (scalar/zonstate.py).  1-based (8,), slots 1..7 used.
struct ZonState {
    float abps[8];
    float crel[8];
    float eff[8];
    float fldst[8];
    float grlos[8];
    float hn[8];
    float hp[8];
    float prob[8];
    float rely[8];
    float rgain[8];
    float sigpow[8];
    float sn[8];
    float spro[8];
    float tgain[8];
    float timed[8];
    float tloss[8];
    float b[8];
    float fslos[8];
    float adv[8];
    float obf[8];
    float tllow[8];
    float tlhgh[8];
    int nmode[8];
};

#endif  // VOACAP_STATE_CUH
