"""COMMON /MUFS/ — the circuit MUF/FOT outputs CURMUF writes.

Hour-indexed arrays (EMUF..ANGMUF) are 1-based (25,) so [IT] matches the
Fortran; the per-layer arrays (SIGL..YMUF, layer 1=E 2=F1 3=F2 4=Es) are
1-based (5,).  MODMUF is the layer index of the circuit MUF.  XLUF is filled
later by LUFFY, not CURMUF; it is carried here for layout completeness.
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32


def _hr():
    return np.zeros(25, dtype=np.float32)


def _ly():
    return np.zeros(5, dtype=np.float32)


@dataclass
class MufState:
    emuf: np.ndarray = field(default_factory=_hr)
    f1muf: np.ndarray = field(default_factory=_hr)
    f2muf: np.ndarray = field(default_factory=_hr)
    esmuf: np.ndarray = field(default_factory=_hr)
    allmuf: np.ndarray = field(default_factory=_hr)
    fot: np.ndarray = field(default_factory=_hr)
    xluf: np.ndarray = field(default_factory=_hr)
    hpf: np.ndarray = field(default_factory=_hr)
    angmuf: np.ndarray = field(default_factory=_hr)
    modmuf: int = 0

    sigl: np.ndarray = field(default_factory=_ly)
    sigu: np.ndarray = field(default_factory=_ly)
    delmuf: np.ndarray = field(default_factory=_ly)
    hpmuf: np.ndarray = field(default_factory=_ly)
    htmuf: np.ndarray = field(default_factory=_ly)
    fvmuf: np.ndarray = field(default_factory=_ly)
    afmuf: np.ndarray = field(default_factory=_ly)
    nhopmf: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.int32))
    yfot: np.ndarray = field(default_factory=_ly)
    yhpf: np.ndarray = field(default_factory=_ly)
    ymuf: np.ndarray = field(default_factory=_ly)
