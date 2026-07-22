"""COMMON /SIGD/ and /TON/ — the path signal-distribution and system-loss
parameters SIGDIS sets and the frequency x hop loop consumes.

/SIGD/ carries the median/decile signal-level adjustments (ASM/DSU/DSL) and
the D-E absorption-loss parameters (AFE/BFE/HNU/HTLOSS/XNUZ/XVE), plus the
path-average geomagnetic latitude (AGLAT), absorption index (ACAV) and E
critical (FEAV).  The *F-suffixed trio (DSLF/ASMF/DSUF) is a second frequency's
distribution set elsewhere; carried for layout completeness.

/TON/ carries the path-average system loss (ADJ) and its deciles/prediction
errors (SL/SU/SLS/SUS/ADS) plus the noise/reliability accumulators the later
routines fill (GNOS/GOT/REL/SPR/XNOISE/ZNOISE/NF); SIGDIS writes only the
system-loss averages.  Kept as plain float32 scalars mirroring the commons.
"""

from dataclasses import dataclass

import numpy as np

F32 = np.float32


@dataclass
class SigdState:
    dsl: F32 = F32(0.0)
    asm: F32 = F32(0.0)
    dsu: F32 = F32(0.0)
    aglat: F32 = F32(0.0)
    dslf: F32 = F32(0.0)
    asmf: F32 = F32(0.0)
    dsuf: F32 = F32(0.0)
    acav: F32 = F32(0.0)
    feav: F32 = F32(0.0)
    afe: F32 = F32(0.0)
    bfe: F32 = F32(0.0)
    hnu: F32 = F32(0.0)
    htloss: F32 = F32(0.0)
    xnuz: F32 = F32(0.0)
    xve: F32 = F32(0.0)


@dataclass
class TonState:
    adj: F32 = F32(0.0)
    ads: F32 = F32(0.0)
    gnos: F32 = F32(0.0)
    got: F32 = F32(0.0)
    rel: F32 = F32(0.0)
    sl: F32 = F32(0.0)
    sls: F32 = F32(0.0)
    spr: F32 = F32(0.0)
    su: F32 = F32(0.0)
    sus: F32 = F32(0.0)
    xnoise: F32 = F32(0.0)
    znoise: F32 = F32(0.0)
    nf: int = 0
