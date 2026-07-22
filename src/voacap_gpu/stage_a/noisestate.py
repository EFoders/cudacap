"""COMMON /ANOIS/ -- the atmospheric-noise working set.

ANOIS1 fills the local-time block indices KJ/JK, the receiver mean time CC and
block-centre time TM, and the two NOISY map values ATNU (primary block) and
ATNY (secondary block).  GENOIS later consumes those plus the decile spreads
(DU/DL/SIGM/SYGU/SYGL) and combined noise RCNSE it fills itself.  Plain
float32 scalars mirroring the common (KJ/JK integer).
"""

from dataclasses import dataclass

import numpy as np

F32 = np.float32


@dataclass
class AnoisState:
    atnu: F32 = F32(0.0)
    atny: F32 = F32(0.0)
    cc: F32 = F32(0.0)
    tm: F32 = F32(0.0)
    rcnse: F32 = F32(0.0)
    du: F32 = F32(0.0)
    dl: F32 = F32(0.0)
    sigm: F32 = F32(0.0)
    sygu: F32 = F32(0.0)
    sygl: F32 = F32(0.0)
    kj: int = 0
    jk: int = 0
