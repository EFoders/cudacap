"""Stage A transliteration of ESIND
(reference/voacapl/src/voacapw/esind.for).

Evaluates the sporadic-E critical-frequency deciles for every sample area
from VERSY(1,3): FS(1)=lower decile, FS(2)=median, FS(3)=upper decile, each
scaled by PSC(4) (default 0.70, for median losses).  Reflection height is a
fixed 110 km.  REAL*4 throughout.
"""

from voacap_gpu.scalar.f32math import F32
from voacap_gpu.scalar.versy import versy


def esind(state, st, ab):
    """state: LayerState; st: IonoState; ab: virtim output."""
    if state.km < 1:  # 105
        return
    psc4 = state.psc[4]
    for ii in range(1, state.km + 1):  # DO 100
        state.hs[ii] = F32(110.0)      # height of reflection
        gamma = versy(st, ab, state.clat[ii], state.clong[ii],
                      state.gmdip[ii], 1, 3)
        state.fs[1, ii] = gamma[3] * psc4  # lower decile
        state.fs[2, ii] = gamma[2] * psc4  # median
        state.fs[3, ii] = gamma[1] * psc4  # upper decile
