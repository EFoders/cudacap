"""ALLMODES -- accumulate /ZON/ modes into /allMODE/ across the hop sweep.

Pure bookkeeping: no arithmetic at all, only assignment.  What it does have is
fiddly index selection, driven by IFLG:

======  ==============================================================
IFLG    behaviour
======  ==============================================================
0       reset -- clear the 20 slots to their "no mode" markers and set
        the current frequency from FVAL
1       take slot 1 only, and separately stage slot 2 as a marker
2       take slots 4-5 (the sporadic-E pair ESMOD fills)
other   take slots 1-6, the regular modes
======  ==============================================================

A /ZON/ slot is copied only if its virtual height HP is positive, the same
"is this mode real" test used everywhere else.

The IFLG=1 arm is the odd one.  Before copying it writes ``HP(2) = HP(1)`` back
into /ZON/, parks a 32000 sentinel in ``zTLOSS(2)``, and writes NMODE/B into
slot ``NMMOD+2`` rather than the cursor -- reserving the next-but-one slot for
a partner mode that a later call fills in.  All of that is preserved verbatim.

The MSPEC=125 blocks throughout allmodes.for are commented out in the source
(they belong to an area-coverage variant that is out of scope), so nothing
from them is ported.
"""

import numpy as np

from voacap_gpu.stage_a.allmodestate import FIELDS, NSLOT
from voacap_gpu.stage_a.f32math import F32

_NO_TLOSS = F32(99999.0)
_NO_DECILE = F32(999.0)
_MARKER_TLOSS = F32(32000.0)


class AllModesOverflow(IndexError):
    """More than 20 modes accumulated -- the Fortran would write past the end.

    NMMOD is never bound-checked and accumulates across every call between
    resets, so a long enough hop sweep walks off the end of the /allMODE/
    arrays and into whatever follows them in the common block.
    """


def allmodes(zon, allm, iflg: int, fval=0.0):
    """Append /ZON/'s live modes to /allMODE/.  Mutates `allm` (and `zon`).

    Returns the new FREQ on the reset call (IFLG=0), where the Fortran assigns
    ``FREQ = FVAL`` into /FRQ/ -- the caller has to adopt it, since frequency
    is threaded as a value here rather than through a shared common.  Returns
    None otherwise.
    """
    if iflg == 0:
        allm.nmmod = 0
        for i in range(1, NSLOT + 1):
            allm.tloss[i] = _NO_TLOSS
            allm.tllow[i] = _NO_DECILE
            allm.tlhgh[i] = _NO_DECILE
            allm.hp[i] = F32(-1.0)
        return F32(fval)                    # FREQ = FVAL

    if iflg == 1:
        ist, lst = 1, 1
        zon.hp[2] = zon.hp[1]
        allm.tloss[2] = _MARKER_TLOSS
        # NB slot NMMOD+2, not the cursor: the next-but-one slot is reserved
        _check(allm.nmmod + 2)
        allm.nmode[allm.nmmod + 2] = zon.nmode[2]
        allm.b[allm.nmmod + 2] = zon.b[2]
    elif iflg == 2:
        ist, lst = 4, 5                     # the sporadic-E pair
    else:
        ist, lst = 1, 6                     # the regular modes

    for i in range(ist, lst + 1):
        if zon.hp[i] > 0.0:
            allm.nmmod += 1
            _check(allm.nmmod)
            n = allm.nmmod
            for name in FIELDS:
                getattr(allm, name)[n] = getattr(zon, name)[i]
            allm.nmode[n] = zon.nmode[i]
    return None


def _check(n: int) -> None:
    if n > NSLOT:
        raise AllModesOverflow(
            f"/allMODE/ slot {n} is past the end of {NSLOT}; the Fortran "
            f"writes into whatever follows the common block")
