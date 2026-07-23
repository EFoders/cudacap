// ANOIS1 — the 1 MHz atmospheric noise at the receiver site; the CUDA C twin
// of stage_a/anois1.py.
//
// Picks the two local-time noise blocks nearest the receiver mean time GMTR —
// KJ (the block CC falls in) and JK (the adjacent block toward CC, wrapped
// cyclically over the six 4-hour blocks) — then evaluates the atmospheric
// noise Fourier map (NOISY) at the receiver latitude/longitude for each,
// giving the primary ATNU and secondary ATNY.  GENOIS interpolates between
// them.  With no long-term ionospheric data base (F2D(1,1,1) <= 0) the noise
// is zeroed here and GENOIS resets it to -204 dB — and, exactly as in the
// Fortran, CC/TM/KJ/JK are then left at whatever /ANOIS/ already held.
//
// The NOISY maps are SHARED, not per-thread, and pass as the flat row-major
// 0-based pointers noisy.cuh documents.  f2d111 is the F2D(1,1,1) scalar the
// no-data test reads (IonoState is not a per-thread struct).
//
// REAL*4 throughout; the NOISY series carries the [[f32-libm-noise]] SIN/COS
// floor.  Rule 4b: the one constant division (CC/4) goes through div32.

#ifndef VOACAP_ANOIS1_CUH
#define VOACAP_ANOIS1_CUH

#include "constants.cuh"
#include "f32math.cuh"
#include "noisy.cuh"
#include "state.cuh"

// anois: AnoisState (out); the five map pointers + f2d111 stand in for Stage
// A's IonoState; gmtr: receiver mean time (h); rlong/rlongd: receiver
// longitude (rad/deg); rlat: receiver latitude (rad).
__device__ inline void anois1(AnoisState* anois, const float* fakp,
                              const float* fakmap, const float* hmym,
                              const float* fakabp, const float* abmap,
                              float f2d111, float gmtr, float rlong,
                              float rlongd, float rlat) {
    if (f2d111 <= 0.0f) {           // IF(F2D(1,1,1)) 90,90,100 -> no data base
        anois->atnu = 0.0f;
        anois->atny = 0.0f;
        return;
    }
    float cc = gmtr;
    anois->cc = cc;
    int kj = 6;
    if (cc - 22.0f < 0.0f)          // IF(CC-22.) 105,110,110
        kj = (int)(div32(cc, 4.0f) + 1.0f);
    float tm = (float)(4 * kj - 2);
    anois->tm = tm;
    int jk;
    if (cc - tm < 0.0f)             // IF(CC-TM) 115,120,125
        jk = kj - 1;
    else if (cc - tm == 0.0f)       // 120
        jk = kj;
    else                            // 125
        jk = kj + 1;
    if (jk <= 0)                    // IF(JK) 135,135,140
        jk = 6;
    else if (jk - 6 > 0)            // 140  IF(JK-6) 150,150,145
        jk = 1;
    anois->kj = kj;
    anois->jk = jk;
    float ceg;
    if (rlong < 0.0f)               // IF(RLONG) 155,160,160  -> west
        ceg = 360.0f + rlong * VC_R2D;
    else                            // east longitude (degrees)
        ceg = rlongd;
    float xla = rlat * VC_R2D;      // latitude (degrees), "+" north
    anois->atnu = noisy(fakp, fakmap, hmym, fakabp, abmap, kj, xla, ceg);
    anois->atny = noisy(fakp, fakmap, hmym, fakabp, abmap, jk, xla, ceg);
}

#endif  // VOACAP_ANOIS1_CUH
