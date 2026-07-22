"""SELMOD -- select the controlling sample area (JMODE) for ray-path calcs.

For KFX==1 there is only one area.  For KFX==2 the two ends share the same F
layer (IONSET forced them equal), so the higher E critical frequency decides.
For KFX>=3 only the two ends are compared: the F2 critical controls unless the
ends match within 0.01 MHz, in which case the E critical breaks the tie.
Returns the 1-based area index (the Fortran stores it in /FRQ/ JMODE).
Mirrors selmod.for.
"""

from voacap_gpu.stage_a.f32math import F32


def selmod(state) -> int:
    """state: LayerState (reads KFX and FI).  Returns JMODE."""
    fi = state.fi
    if state.kfx == 1:
        return 1
    if state.kfx == 2:
        # receiver end (2) controls if its E critical is lower
        return 2 if fi[1, 1] > fi[1, 2] else 1
    # KFX >= 3: check the ends (areas 1 and 3) only
    delfi = fi[3, 1] - fi[3, 3]
    if abs(delfi) < F32(0.01):
        # F2 layers match -> E layer decides
        return 3 if fi[1, 1] > fi[1, 3] else 1
    # F2 layer controls
    return 3 if delfi > 0.0 else 1
