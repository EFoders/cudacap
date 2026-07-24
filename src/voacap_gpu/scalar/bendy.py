"""Stage A transliteration of the BENDY function
(reference/voacapl/src/voacapw/bendy.for).

Bending correction for a parabolic layer I at sample area K, evaluated at
frequency F.  REAL*4; ALOG carries the mild [[f32-libm-noise]] floor.
"""

from voacap_gpu.scalar.f32math import F32, log32


def bendy(state, i, k, f):
    x = F32(f) / state.fi[i, k]
    x = min(x, F32(0.999))
    return F32(0.5) * x * state.yi[i, k] * log32(
        (F32(1.0) + x) / (F32(1.0) - x))
