// THE STATUS TABLE — every refusal code the LUFFY kernel can return, globally
// unique, in one place.
//
// A device function cannot raise, so where Stage A raises (because the
// Fortran would read out of bounds, STOP, or reach an unported feature) each
// callee returns a non-zero status and LUFFY propagates the FIRST one.  The
// callees' own codes are LOCAL -- 1, 2, 3 ... per routine, and pinned by each
// routine's own test (tests/cuda/test_<routine>.py) -- so before this table
// the same number meant different things: 3 was both "area-coverage receive
// antenna in GENOIS" and "GAIN table miss in REGMOD/ESMOD/SETTXR", and the
// setup kernel's hour guard reused 3 as well.
//
// So LUFFY (luffy.cuh, luffy_warp.cuh, lsmooth.cuh's blend) and the batch
// kernels map every callee status onto ONE global code at the call site:
//
//     global = VC_ST_<ROUTINE> + local          (local in 1..99)
//
// which makes ``status / 100`` the routine and ``status % 100`` that
// routine's own documented code.  0 is success, unchanged; the drivers' -1
// preset ("the kernel never wrote this row") is unchanged; nothing here is
// 1..99, so the CPU backend's own STATUS_CPU_UNMAPPED (99, engine.py) cannot
// collide with a kernel code.
//
//   code  routine  meaning                                    Stage A raise
//   ----  -------  -----------------------------------------  -------------
//      0           success
//     -1           driver preset: the kernel never wrote it
//    101  ALLMODES more than 20 modes                          AllModesOverflow
//    201  FRQCOM   frequency-complement index out of range     FrequencyIndexError
//    301  GENOIS   NYEAR ' 2089' selects unported GENOIS_OLD   NotImplementedError
//    302  GENOIS   receive-antenna GAIN lookup left the table  IndexError
//                  (e.g. a frequency below 1 MHz)
//    303  GENOIS   receive antenna is area-coverage            NotImplementedError
//                  (OFFAZIM == -999, unported)
//    401  REGMOD   MUFday probability <= 0 (Fortran STOP)      RegmodError
//    402  REGMOD   frequency <= 0 (Fortran STOP)               RegmodError
//    403  REGMOD   antenna GAIN lookup left the table          IndexError
//    404  REGMOD   area-coverage antenna                       NotImplementedError
//    503  ESMOD    antenna GAIN lookup left the table          IndexError
//    504  ESMOD    area-coverage antenna                       NotImplementedError
//    601  SETTXR   MODMUF outside 1..4 (DELMUF)                ModmufError
//    602  SETTXR   MODMUF outside 1..4 (YMUF)                  ModmufError
//    603  SETTXR   antenna GAIN lookup left the table          IndexError
//    604  SETTXR   area-coverage antenna                       NotImplementedError
//    705  INMUF    /MODES/ ITMOD outside 1..3                  IndexError
//    706  INMUF    zero-distance stage writes /MODES/ row 7    IndexError
//    801  LNGPAT   YMUF(1..4) index off the end                ModeIndexError
//    901  RELBIL   mode label outside LAYTYP                   IndexError
//   1001  SELTXR   ray index out of range                      RayIndexError
//   1101  LSMOOTH  smoothing-store index (LUF-arm IPFLAG)      IndexError
//   1201  LUFFY    /allMODE/ NMODE(2) outside LAYTYP (long     IndexError
//                  path receiver end)
//   1301  SETUP    hour outside 1..24: no VIRTIM row, the      IndexError
//                  circuit is not built (setup.cuh hour guard)
//
// REGMOD and ESMOD codes reach LUFFY through INMUF and the Es slots; SETTXR's
// through GMLOSS.  Local codes 5/6 are INMUF's own, 1..4 there are REGMOD's
// propagated verbatim -- vc_status_inmuf() splits them.  ESMOD has no local
// 1 or 2.  engine._scalar_status maps the CPU backend's exceptions onto this
// same table, and tests/cuda/test_status_codes.py holds the Python constants
// (engine.STATUS_*) to the values below.

#ifndef VOACAP_STATUS_CUH
#define VOACAP_STATUS_CUH

#define VC_ST_OK        0

// per-routine bases
#define VC_ST_ALLMODES  100
#define VC_ST_FRQCOM    200
#define VC_ST_GENOIS    300
#define VC_ST_REGMOD    400
#define VC_ST_ESMOD     500
#define VC_ST_SETTXR    600
#define VC_ST_INMUF     700
#define VC_ST_LNGPAT    800
#define VC_ST_RELBIL    900
#define VC_ST_SELTXR    1000
#define VC_ST_LSMOOTH   1100
#define VC_ST_LUFFY     1200
#define VC_ST_SETUP     1300

// the codes LUFFY and the batch kernels raise themselves
#define VC_ST_LSMOOTH_BAD_IDX   (VC_ST_LSMOOTH + 1)
#define VC_ST_LUFFY_BAD_NMODE2  (VC_ST_LUFFY + 1)
#define VC_ST_SETUP_HOUR_OOB    (VC_ST_SETUP + 1)

// A callee's local status -> the global code.  0 stays 0.
__device__ __forceinline__ int vc_status(int base, int local) {
    return local ? base + local : 0;
}

// INMUF returns REGMOD's 1..4 verbatim and its own 5/6.
__device__ __forceinline__ int vc_status_inmuf(int local) {
    if (!local)
        return 0;
    return (local <= 4) ? VC_ST_REGMOD + local : VC_ST_INMUF + local;
}

#endif  // VOACAP_STATUS_CUH
