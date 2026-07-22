"""Stage A transliteration of EF1VAR
(reference/voacapl/src/voacapw/ef1var.for).

Evaluates the E and F1 layer parameters for one sample area from the foE
map (VERSY(6,6)) and the F1 regression coefficients ANEW/BNEW/ACHI/BCHI
that REDMAP loaded.  Mutates the E/F1 slots of the LayerState and returns
the (abs) zenith angle CYCEN, exactly as the Fortran passes it back through
its argument.  REAL*4 throughout.
"""

import numpy as np

from voacap_gpu.stage_a.constants import R2D
from voacap_gpu.stage_a.f32math import F32, cos32
from voacap_gpu.stage_a.versy import versy

_BETAE = F32(5.5)
_BETAF1 = F32(4.0)


def ef1var(state, st, ab, cycen, ii):
    """state: LayerState; st: IonoState (coeffs); ab: virtim output.
    cycen: sun zenith angle (radians).  Returns abs(cycen)."""
    gamma = versy(st, ab, state.clat[ii], state.clong[ii],
                  state.gmdip[ii], 6, 6)
    ssn = F32(state.ssn)
    anew, bnew = st.anew, st.bnew
    achi, bchi = st.achi, st.bchi

    g6 = gamma[6]
    if g6 - F32(0.36) < 0.0:  # 85: bad map value, use midnight value
        g6 = F32(0.36) * np.sqrt(F32(1.0) + F32(0.0098) * ssn)
    # 90
    state.fi[1, ii] = g6 * state.psc[1]        # E layer critical frequency
    state.hi[1, ii] = F32(110.0)               # E height of max ionization
    state.yi[1, ii] = state.hi[1, ii] / _BETAE  # E semithickness

    cycen = F32(abs(cycen))
    z = cycen * R2D
    cosz = cos32(cycen)
    cosdi = cos32(state.gmdip[ii])             # cos of Rawer magnetic dip
    # maximum zenith angle for the F1 layer
    zmax = (F32(achi[0]) + F32(bchi[0]) * ssn
            + (F32(achi[1]) + F32(bchi[1]) * ssn) * cosdi)
    state.zenmax[ii] = zmax
    state.zenang[ii] = z
    if z <= zmax:
        f1 = ((F32(anew[0]) + F32(bnew[0]) * ssn)
              + (F32(anew[1]) + F32(bnew[1]) * ssn) * cosz
              + (F32(anew[2]) + F32(bnew[2]) * ssn) * cosz * cosz)
        state.fi[2, ii] = f1 * state.psc[2]    # F1 critical frequency
        state.hi[2, ii] = F32(165.0) + F32(0.6428) * z  # F1 height of max
        state.yi[2, ii] = state.hi[2, ii] / _BETAF1     # F1 semithickness
    else:
        state.fi[2, ii] = F32(0.0)
        state.hi[2, ii] = F32(0.0)
        state.yi[2, ii] = F32(0.0)
    return cycen
