// CircuitState — one circuit-hour's worth of per-thread state, the CUDA C
// twin of stage_a/circuit.py's `Circuit`.
//
// ABI CONTRACT (same rules as state.cuh, proved by driver.py's CIRCUIT_DTYPE
// and tests/stage_c/test_luffy_replay.py's sizeof kernel):
//   * the sixteen per-thread state structs appear BY VALUE, in exactly the
//     Stage A `Circuit` field order, with the same names;
//   * `iono` and `cant` are NOT here.  Stage A carries them on the Circuit,
//     but they are big, read-only and SHARED between threads: the coefficient
//     maps and the antenna patterns.  They pass to `luffy` as flat device
//     pointers plus a per-thread index, exactly as every leaf header already
//     takes them (see gain.cuh, noisy.cuh, syssy.cuh, genfam.cuh).
//   * then the loose scalars, again in Circuit order, every one 4 bytes so
//     the struct stays padding-free.
//
// Two representational notes:
//
//   * `nyear` is a Fortran CHARACTER*5 on the Stage A side and only ever
//     matters as the single test ``nyear == " 2089"`` (GENOIS's pre-1989
//     noise model).  It is carried here as the int flag `nyear_is_2089`,
//     which is exactly what genois.cuh already takes.
//   * `want_service_prob` is a Python bool (LINBOT(14)); an int here,
//     non-zero being true.
//
// /SON/ MDL is a one-character model tag ("S" short, "L" long, "M" smoothed,
// " " cleared, "" never set) rather than a LAYTYP layer label, so it does NOT
// use state.cuh's LAYTYP encoding.  The codes below are its own; MODE and
// MODER keep the LAYTYP encoding state.cuh documents.

#ifndef VOACAP_CIRCUIT_CUH
#define VOACAP_CIRCUIT_CUH

#include "state.cuh"

// /SON/ MDL(1..13) encoding — the model tag, not a layer label.
#define VC_MDL_EMPTY (-1)       // ""   (a fresh SonState)
#define VC_MDL_BLANK 0          // " "  (cleared at the top of the freq loop)
#define VC_MDL_S     1          // "S"  short path
#define VC_MDL_L     2          // "L"  long path
#define VC_MDL_M     3          // "M"  smoothed

struct CircuitState {
    // --- the commons, one member per block, in Circuit's field order -------
    LayerState     layers;
    MufState       muf;
    RaysState      rays;
    ReflxState     reflx;
    ModesState     modes;
    ZonState       zon;
    AllModeState   allm;
    SonState       son;
    AnoisState     anois;
    NoiseDiagState ndiag;
    SigdState      sigd;
    TonState       ton;
    FrqState       frq;
    TimeState      time;
    LosxState      losx;
    LpathState     lpath;

    // --- /DON/ path geometry and circuit requirements ----------------------
    float gcd;                  // great-circle distance, radians
    float gcdkm;                // ... and km
    float amind;                // SANG's reduced minimum take-off angle, deg
    float rsn;                  // required signal-to-noise, dB
    float dmp;                  // max tolerable multipath delay, ms
    float pmp;                  // max tolerable multipath power spread, dB
    float rlat;                 // receiver latitude, radians
    float rlong;                // receiver longitude, radians
    float rlongd;               // ... and degrees
    float tlong;                // transmitter longitude, radians

    // --- /ION/ -------------------------------------------------------------
    int lufp;                   // required reliability, percent
    int method;
    int noise;                  // man-made noise category

    // --- /CONTRL/ ----------------------------------------------------------
    int mspec;                  // 121 selects long/short path smoothing
    int jlong;                  // output-only; -1 short path, +1 long

    // --- /ALPHA/ NYEAR, as the one flag that matters -----------------------
    int nyear_is_2089;          // NYEAR == " 2089" -> the pre-1989 noise model

    // --- /OUTPRT/ LINBOT(14) ----------------------------------------------
    int want_service_prob;

    // --- /INFORM/ INFO -----------------------------------------------------
    int info;
};

// The SHARED, read-only tables a circuit-hour needs, with the per-circuit
// offsets already applied.  This is not a Stage A object: it exists only so
// the driver's parameter list stays legible, and holds exactly the pointers
// the leaf headers already document (noisy.cuh, syssy.cuh, genfam.cuh,
// gain.cuh).  Every member is a pointer INTO a device-resident table, never
// a copy.
struct CircuitTables {
    // REDMAP's /TWO/ maps for this circuit's month + SSN
    const float* fam;           // (14, 12)
    const float* dud;           // (5, 12, 5)
    const float* fakp;          // (29, 16, 6)
    const float* fakmap;        // (29, 16)
    const float* hmym;          // (29, 16)
    const float* fakabp;        // (2, 6)
    const float* abmap;         // (2, 3)
    const float* sys;           // (9, 16, 6)
    const float* perr;          // (9, 4, 6)
    float f2d111;               // F2D(1,1,1), the "is there a data base" test
    // /cantenna/ + /pantenna/ for this circuit, with A = numants
    const float* array;         // (31, 92, A+1)
    const float* aeff;          // (31, A+1)
    const int*   iats;          // (A+1,)
    const float* xfqs;          // (A+1,)
    const float* xfqe;          // (A+1,)
    const float* offazim;       // (A+1,)
    const float* pwrdba;        // (A+1,)
    int numants;
};

// Flat-table strides, one circuit's worth of each (see redmap.py shapes).
#define VC_FAM_STRIDE    (14 * 12)
#define VC_DUD_STRIDE    (5 * 12 * 5)
#define VC_FAKP_STRIDE   (29 * 16 * 6)
#define VC_FAKMAP_STRIDE (29 * 16)
#define VC_HMYM_STRIDE   (29 * 16)
#define VC_FAKABP_STRIDE (2 * 6)
#define VC_ABMAP_STRIDE  (2 * 3)
#define VC_SYS_STRIDE    (9 * 16 * 6)
#define VC_PERR_STRIDE   (9 * 4 * 6)
#define VC_ANT_NFREQ     31
#define VC_ANT_NELEV     92

#endif  // VOACAP_CIRCUIT_CUH
