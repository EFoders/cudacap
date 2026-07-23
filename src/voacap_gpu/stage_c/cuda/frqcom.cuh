// FRQCOM — synthesise a frequency complement from 2 to 40 MHz; the CUDA C
// twin of stage_a/frqcom.py.  Slot 12 always gets the circuit MUF; slot 13
// is zeroed ("no LUF yet") by every arm EXCEPT case 1, whose omission is
// load-bearing and preserved.  Case 6 is dead (its guard is the constant
// 36) exactly as in the Fortran, so case 5 is the final arm.
//
// `frea` is the caller's 1-based frequency array — LUFFY passes /FRQ/ FREA,
// the hour driver passes FREL; only slots 1..13 are touched either way.
//
// Stage A raises FrequencyIndexError where case 3's NE would write past
// FREA's 13 elements; here the same condition returns 1 with the state
// exactly as Stage A leaves it at the raise (slot 12 written, the ladder
// untouched).  Return 0 is success.

#ifndef VOACAP_FRQCOM_CUH
#define VOACAP_FRQCOM_CUH

#include "f32math.cuh"
#include "state.cuh"

__device__ inline int frqcom(const MufState* muf, float* frea, int ifreq,
                             int it) {
    // .....INSERT MUF INTO FREQUENCY SLOT 12
    frea[12] = muf->allmuf[it];
    if (ifreq == -10) {                     // IF(IFREQ + 10) 95, 90, 95
        // .....INSERT FOT INTO FREQUENCY SLOT 1
        frea[1] = muf->fot[it];
        return 0;
    }

    float xfl = pymin(muf->emuf[it], muf->f2muf[it]);
    float xfh = muf->hpf[it];
    xfl = pymax(2.0f, xfl);
    xfh = pymin(xfh, 40.0f);

    if (xfh <= 2.0f) {
        // .....CASE 1.  NOT LIKELY TO OCCUR -- and slot 13 is left alone
        frea[1] = 2.0f;
        for (int i = 2; i <= 11; ++i)
            frea[i] = frea[i - 1] + 2.0f;
        return 0;
    }

    if (xfl <= 2.0f) {
        // .....CASE 2.  NIGHTTIME -- the whole band is above the lower MUF
        float delf = div32(xfh - 2.0f, 11.0f);
        frea[1] = 2.0f;
        for (int i = 2; i <= 11; ++i)
            frea[i] = frea[i - 1] + delf;
        frea[13] = 0.0f;
        return 0;
    }

    if (xfh <= 2.0f + 20.0f) {
        // .....CASE 3.  INSERT SMALLER MUF, WITHIN EQUAL INCREMENTS
        // .....RANGE .LE. 20 MHZ
        float delf = div32(xfh - 2.0f, 9.0f);
        int ne = (int)((xfl - 2.0f) / delf);    // REAL->INT truncates
        ne = ne + 2;
        if (ne > 12)
            return 1;                       // FrequencyIndexError
        frea[1] = 2.0f;
        for (int i = 2; i <= ne; ++i)
            frea[i] = frea[i - 1] + delf;
        frea[ne] = xfl;                     // the knee, planted on slot NE
        frea[ne + 1] = frea[ne - 1] + delf; // resume the ladder, ignoring XFL
        ne = ne + 2;
        if (ne > 11) ne = 11;               // NE = MIN0(NE, 11)
        for (int i = ne; i <= 11; ++i)
            frea[i] = frea[i - 1] + delf;
        frea[13] = 0.0f;
        return 0;
    }

    if (xfl <= 2.0f + 2.0f) {
        // .....CASE 4.  INSERT LOWER MUF BETWEEN 2 AND 4 MHZ
        // .....THEN EQUAL INCREMENTS FROM 4 MHZ TO 40 MHZ
        float delf = div32(xfh - (2.0f + 2.0f), 9.0f);
        frea[1] = 2.0f;
        frea[2] = xfl;
        frea[3] = 2.0f + 2.0f;
        for (int i = 4; i <= 11; ++i)
            frea[i] = frea[i - 1] + delf;
        frea[13] = 0.0f;
        return 0;
    }

    // IF( (FHIGH-2.) -FLOW) 185,185,166 -- constant 36, so case 6 is dead.
    // .....CASE 5.  EQUAL INCREMENTS TO LOWER MUF, THEN EQUAL INCREMENTS TO
    // .....UPPER LIMIT
    int ne = (int)div32(xfl - 2.0f, 2.0f);  // >= 1 here: case 5 needs XFL > 4
    if (ne > 7) ne = 7;                     // NE = MIN0(NE, 7)
    float xne = (float)ne;
    float delf = (xfl - 2.0f) / xne;
    frea[1] = 2.0f;
    for (int i = 2; i <= ne; ++i)
        frea[i] = frea[i - 1] + delf;
    xne = (float)ne;
    delf = (xfh - frea[ne]) / (12.0f - xne - 1.0f);
    ne = ne + 1;
    for (int i = ne; i <= 11; ++i)
        frea[i] = frea[i - 1] + delf;
    frea[13] = 0.0f;
    return 0;
}

#endif  // VOACAP_FRQCOM_CUH
