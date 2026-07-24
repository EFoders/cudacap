"""Stage A transliteration of the PEN function
(reference/voacapl/src/voacapw/pen.for).

Retardation (penetration) term for a parabolic layer I at sample area K,
evaluated at frequency F.  REAL*4; ALOG carries the mild
[[f32-libm-noise]] floor.
"""

from voacap_gpu.scalar.f32math import F32, log32


def pen(state, i, k, f):
    x = F32(f) / state.fi[i, k]
    x = max(x, F32(1.001))
    return state.yi[i, k] * log32((F32(1.0) + x) / (x - F32(1.0))) * x
