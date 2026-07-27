// GEOTIM — put one prediction hour into every time base the model needs;
// the CUDA C twin of scalar/geotim.py, with CNGTIM (scalar/cngtim.py)
// alongside it because GEOTIM is its only caller in this path and Stage C
// has no cngtim.cuh chunk.  The CNGTIM block below carries the guard
// VOACAP_CNGTIM_CUH, so if that header is ever split out and included first,
// this copy compiles away instead of colliding.
//
// The ionosphere is driven by LOCAL time, the user thinks in universal time,
// and the output tables print both, so each hour is resolved three ways plus
// once per sample area:
//
//     GMT      universal time                    -> /TIME/ GMT, UTIME(IT)
//     XLMT     local mean time at the transmitter-> /TIME/ XLMT(IT)
//     GMTR     local mean time at the receiver   -> /TIME/ GMTR (ANOIS1)
//     CLCK(i)  local mean time at sample area i  -> /GEOG/ CLCK (TIMVAR)
//
// ITIM decides which of the first two the user typed.  NEGATIVE means the
// TIME card gave local time at the transmitter, so the conversion runs
// BACKWARDS (ISW = -1) and GMT is the derived quantity; otherwise the input
// is universal and GMT is just the requested hour.
//
// Three details that are easy to lose and are pinned by the test:
//
// * **IT is a TRUNCATION, then a rewrite.**  IT = INT(GMT) truncates toward
//   zero, and IF(IT) 150,150,160 sends IT <= 0 to 24 -- midnight belongs to
//   the day that is ENDING.  That rewrite is only reachable on the ITIM < 0
//   arm: on the universal arm GMT is assigned the requested hour before any
//   conversion touches it, so IT is JT and the rewrite would need JT itself
//   to be 0.
// * **CNGDAY is written four-plus times and never read.**  CNGTIM computes a
//   day change; /TIME/ has nowhere to put a date, so GEOTIM throws every one
//   away.  Here that is the `day` out-parameter pointed at a scratch int.
// * **The KM loop is 1-based and may not run at all** (IF(KM) 190,190,170),
//   in which case CLCK keeps whatever it had.
//
// div32 SITES: every CNGTIM division is by a literal (180., 360., 15., 24.),
// so all six go through div32.  THETA/180. and THETA/360. are truncated to
// integers immediately afterwards, which is where a strength-reduced 1 ULP
// would turn into a whole 24-hour offset.

#ifndef VOACAP_GEOTIM_CUH
#define VOACAP_GEOTIM_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "state.cuh"

#ifndef VOACAP_CNGTIM_CUH
#define VOACAP_CNGTIM_CUH

// CNGTIM — universal time <-> local mean time at a reference longitude.
//
// Fifteen degrees of longitude is one hour, so the conversion is a division
// and a wrap.  What makes it more than that is the sign convention: the
// offset has to come out in (-12, +12] rather than [0, 24), or a station just
// west of the date line would be reported a day out.  The routine gets there
// by measuring the longitude in half-turns (INT1) and full turns (INT2):
//
//     THETA in [0, 180)    offset = THETA/15                east of Greenwich
//     THETA in [180, 360)  offset = THETA/15 - 24           ... read as west
//     THETA = 360          offset = THETA/15 - 48 + 24 = 0, i.e. Greenwich
//
// ISW picks the direction: +1 converts UT to local, -1 converts local back.
//
// A REFERENCE LONGITUDE OF EXACTLY ZERO RETURNS IMMEDIATELY, before the
// normalising loop, so the time comes back untouched rather than wrapped into
// [0, 24].  Greenwich needs no conversion, which is right, but it also skips
// the range fix every other longitude gets.
//
// THE EPSILON IN (TIME - 23.99999999)/24.0 DOES NOTHING.  In single precision
// that constant IS 24.0f -- the spacing of binary32 near 24 is about 1.9e-6,
// two hundred times larger than the 1e-8 the author was reaching for.  Kept
// as written; it costs nothing and removing it would be a silent edit.
//
// Returns the converted time; the day change (-1, 0, +1) is written through
// `day`, which GEOTIM points at a scratch int and never reads.
__device__ inline float cngtim(float time, float xtheta, int isw, int* day) {
    if (xtheta == 0.0f) {               // IF(XTHETA) 80, 75, 85
        *day = 0;
        return time;
    }
    float theta = (xtheta < 0.0f) ? (360.0f + xtheta) : xtheta;

    int int1 = (int)div32(theta, 180.0f);   // REAL->INT truncates toward zero
    int int2 = (int)div32(theta, 360.0f);
    float fint1 = (float)int1;
    float fint2 = (float)int2;
    float fisw = (float)isw;
    time = time + fisw * (div32(theta, 15.0f)
                          - fint1 * 24.0f + fint2 * 24.0f);

    // The day change, BEFORE the time is normalised.  INT1/INT2 are reused
    // below, which is why FINT1/FINT2 were captured above.
    *day = (int)div32(time - 23.99999999f, 24.0f) + (int)div32(time, 24.0f);

    if (time < 0.0f)
        time = time + 24.0f;
    time = time - fint2 * 24.0f;
    while (true) {                      // 110 -- wrap into [0, 24]
        if (time < 0.0f) {
            time = time + 24.0f;
            continue;
        }
        if (time > 24.0f) {
            time = time - 24.0f;
            continue;
        }
        return time;
    }
}

#endif  // VOACAP_CNGTIM_CUH

// GEOTIM — fill /TIME/ and /GEOG/ CLCK for hour `jt`.
//
// layers: reads KM and CLONG, writes CLCK; tm: reads ITIM, writes IT, GMT,
// GMTR, UTIME(IT), XLMT(IT); jt: the requested hour; tlong/rlong: transmitter
// and receiver longitude in RADIANS (GEOTIM does the R2D itself).
__device__ inline void geotim(LayerState* layers, TimeState* tm, int jt,
                              float tlong, float rlong) {
    int cngday;                         // written, never read -- see the note
    float ckc = (float)jt;
    float temp;
    if (tm->itim < 0) {
        // .....LMT AT TRANSMITTER -- the hour given IS local, so derive UT
        temp = ckc;
        ckc = cngtim(ckc, tlong * VC_R2D, -1, &cngday);
        tm->gmt = ckc;
    } else {
        // .....GMT AT TRANSMITTER
        tm->gmt = ckc;
        ckc = cngtim(ckc, tlong * VC_R2D, 1, &cngday);
        temp = ckc;
    }

    int it = (int)tm->gmt;              // REAL->INT truncates toward zero
    if (it <= 0)                        // IF(IT) 150, 150, 160
        it = 24;                        // midnight belongs to the day ending
    tm->it = it;
    tm->utime[it] = tm->gmt;
    tm->xlmt[it] = temp;

    // .....LMT AT RECEIVER
    tm->gmtr = cngtim(tm->gmt, rlong * VC_R2D, 1, &cngday);

    // .....LMT AT SAMPLE POINTS
    for (int ii = 1; ii <= layers->km; ++ii)    // IF(KM) 190, 190, 170
        layers->clck[ii] = cngtim(tm->gmt, layers->clong[ii] * VC_R2D, 1,
                                  &cngday);
}

#endif  // VOACAP_GEOTIM_CUH
