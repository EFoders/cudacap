// PRBMUF — probability the operating frequency exceeds a layer's MUF/FOT/HPF;
// the CUDA C twin of stage_a/prbmuf.py.  PRBMUF = 1 - FNORML(Z) with the
// normal tail from the quartic Hastings rational approximation, Z the
// operating frequency's departure from FGO scaled by the layer's decile
// spread (/MUFS/ SIGL below the median, SIGU above, both relative to FSET).
// When no median is placed (FSET <= 0) it degrades to a step at FGO.
// IL selects the layer (1=E, 2=F1, 3=F2, 4=Es).  Pure float32 arithmetic —
// no transcendentals, so no libm floor.

#ifndef VOACAP_PRBMUF_CUH
#define VOACAP_PRBMUF_CUH

#include "f32math.cuh"
#include "state.cuh"

// muf: MufState (for SIGL/SIGU).  Returns the exceedance probability.
__device__ inline float prbmuf(const MufState* muf, float fmhz, float fgo,
                               float fset, int il) {
    float z = fmhz - fgo;
    float qx;
    if (fset > 0.0f) {                  // 95
        float sig;
        if (z <= 0.0f)
            sig = fgo * muf->sigl[il] / fset;
        else
            sig = fgo * muf->sigu[il] / fset;
        sig = pymax(sig, 0.001f);
        z = z / sig;
        // PRBMUF = 1 - FNORML(Z); fold to |Z|, clamp for 32-bit range.
        float ypx = z;
        float yp = (ypx < 0.0f) ? -z : z;
        yp = pymin(5.0f, yp);
        qx = 1.0f + yp * (0.196854f + yp * (0.115194f
                 + yp * (0.000344f + yp * 0.019527f)));
        qx = qx * qx * qx * qx;
        qx = 1.0f / qx;
        qx = 0.5f * qx;
        if (ypx < 0.0f)
            qx = 1.0f - qx;
    } else {                            // 145: no median placed -> step at FGO
        qx = (z <= 0.0f) ? 1.0f : 0.0f;
    }
    return qx;
}

#endif  // VOACAP_PRBMUF_CUH
