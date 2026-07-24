"""Stage B (PyTorch, batched) multipath probability — MPATH.

Not ray geometry, despite the name -- that was FDIST's job and ran much
earlier.  This is the last analysis step of the frequency loop: given the most
reliable mode RELBIL chose, how likely is it that some *other* mode arrives
badly enough to matter?

A mode qualifies as interference when all three hold:

* it is not the most reliable mode, and it exists (HP > 0);
* its delay differs from the MRM's by MORE than the operator's tolerable delay
  DMP -- a mode arriving close in time is not multipath, it just adds;
* its signal power is within PMP dB of the MRM's, so it is strong enough to
  matter at all.

The answer is the highest reliability among those, or 0.001 if none qualify.

**THE DELAY TEST READS BACKWARDS FROM THE SOURCE'S OWN HEADER COMMENT**, which
says the routine returns the next most probable mode "that is within the
Maximum tolerable time delay (DMP)".  The code skips modes WITHIN DMP and keeps
those OUTSIDE it -- which is what the physics wants, and what the inline
comment on that line ("inside acceptable time delay", sitting on a skip)
agrees with.  The header comment is the thing that is wrong; the code is
transliterated as written.

A MODE IS DEAD AT HP <= 0 HERE but at HP <= 70 km in SERPRB, which runs
immediately before this on the same table.  A slot between the two thresholds
is live for one routine and dead for the other.  Both are correct and both are
preserved.

Batching is one masked maximum per mode slot over a fixed twenty-iteration
Python loop; MAX is associative, so this is the one reduction in the cluster
whose order genuinely does not matter.  The two early RETURNs become lane masks
rather than exits, because METHOD/MSPEC/GCDKM and DMP/PMP can all differ
between grid points in an area run.

No libm at all -- comparisons, one subtraction and a maximum.
"""

import numpy as np
import torch

from voacap_gpu.batched.lockstep import gather_rows as _g
from voacap_gpu.batched.profile import _k

_F32 = torch.float32
NM = 20

_NO_MULTIPATH = float(np.float32(0.001))
_AREA_LIMIT_KM = float(np.float32(7000.0))


def mpath_batch(modes, nmmod, nrel, dmp, pmp, gcdkm, method, mspec):
    """Batched MPATH.  Returns PROBMP as (N,) float32.

    modes: dict of (NM+1, N) tensors 1-based in the mode axis -- ``hp``,
        ``timed``, ``sigpow`` and ``rely`` are read.  RELY is RELBIL's output,
        so RELBIL must have run on the same batch first.  A /allMODE/ state
        dict as ``stage_b.allmodes`` builds one can be handed straight in.
    nmmod: (N,) int mode count.  nrel: (N,) int, RELBIL's most reliable mode.
    dmp: (N,) maximum tolerable delay, ms; pmp: (N,) multipath power
        tolerance, dB.  Both are operator-set /DON/ values and either being
        non-positive means no multipath was requested.
    gcdkm: (N,) path length.  method/mspec: (N,) ints from /ION/ and /CONTRL/;
        a 0-dim tensor broadcasts, for the common case where the whole batch
        is one run.
    """
    hp, timed = modes["hp"], modes["timed"]
    sigpow, rely = modes["sigpow"], modes["rely"]
    n = gcdkm.shape[0]
    dev = gcdkm.device
    nmmod = nmmod.to(torch.long)
    nrel = nrel.to(torch.long)

    probmp = torch.full((n,), _NO_MULTIPATH, dtype=_F32, device=dev)
    requested = (dmp > 0.0) & (pmp > 0.0)
    # METHOD 30's area variant is not valid beyond 7000 km
    area_cut = ((method == 20) & (mspec == 121)
                & (gcdkm > _k(_AREA_LIMIT_KM, gcdkm)))
    run = requested & ~area_cut

    sig_power_limit = _g(sigpow, nrel) - pmp
    ttim = _g(timed, nrel)                      # time delay of the MRM

    for im in range(1, NM + 1):                 # DO 135
        qualifies = (
            run
            & (im <= nmmod)
            & (nrel != im)                      # RELBIL already found this one
            & (hp[im] > 0.0)                    # this mode does exist
            # OUTSIDE the tolerable delay is what counts -- see the docstring
            & (torch.abs(timed[im] - ttim) > dmp)
            & (sigpow[im] >= sig_power_limit))  # strong enough to interfere
        probmp = torch.where(qualifies, torch.maximum(probmp, rely[im]),
                             probmp)
    return probmp
