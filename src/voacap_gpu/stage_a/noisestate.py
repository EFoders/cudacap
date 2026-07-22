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


@dataclass
class NoiseDiagState:
    """COMMON /c_noise/, /c_noises/ and /c_method_used/ -- GENOIS's breakdown.

    The three noise components in dBW/Hz separately (ATNOS atmospheric,
    GNOS galactic, XNOIS man-made), which of the two combination methods was
    used (2 = Spaulding's original, 1 = Caruana's limit applied), and the
    per-component enable flags.  Nothing in voacapl reads any of these back --
    they exist so a caller can see how the combined RCNSE was arrived at --
    but MODE_N2 is load-bearing *within* GENOIS: it is cleared when galactic
    noise cannot penetrate the F2 layer, and that zeroes QPG.

    MODE_N1 and MODE_N3 appear only in commented-out lines of genois.for
    (they would suppress the atmospheric and man-made components); they are
    carried for layout completeness and are never read.
    """

    mode_n1: int = 0
    mode_n2: int = 0
    mode_n3: int = 0
    xn_atnos: F32 = F32(0.0)
    xn_galactic: F32 = F32(0.0)
    xn_mmn: F32 = F32(0.0)
    method_used: int = 0
