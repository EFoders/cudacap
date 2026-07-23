// FNORML — cumulative normal, AMS 55 26.2.17; CUDA C twin of
// stage_a/fnorml.py.  Degree-4 polynomial raised to the fourth power,
// argument folded to |x| and clamped at 5 sigma, result reflected back.
// Callers that want the upper tail spell out ``1.0f - fnorml(z)`` — the two
// float32 subtractions do NOT cancel and are part of the Fortran's answer.

#ifndef VOACAP_FNORML_CUH
#define VOACAP_FNORML_CUH

#include "f32math.cuh"

__device__ inline float fnorml(float ypx) {
    float yp = ypx;
    if (ypx < 0.0f)                 // IF(YPX) 105,110,110
        yp = -ypx;
    yp = pymin(5.0f, yp);
    float qx = 1.0f + yp * (0.196854f + yp * (0.115194f
                     + yp * (0.000344f + yp * 0.019527f)));
    qx = qx * qx * qx * qx;
    qx = 1.0f / qx;
    qx = 0.5f * qx;
    float px;
    if (ypx < 0.0f)                 // IF(YPX) 115,120,120
        px = qx;
    else
        px = 1.0f - qx;
    return px;
}

#endif  // VOACAP_FNORML_CUH
