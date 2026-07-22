"""COMMON /LOSX/ -- the long path's per-ray loss and gain table.

Where /REFLX/ holds the GEOMETRY of every ray at a control area, /LOSX/ holds
what that ray costs.  SETTXR fills it for the transmitter and receiver ends;
SELTXR reads it to pick a ray; GMLOSS differences it into TXRGML; LNGPAT
averages the two chosen rays into one mode.

    ANDVX   non-deviative absorption, plus the over-the-MUF adjustment
    ADVX    deviative loss (with the E-layer correction where it applies)
    AOFX    sporadic-E obscuration
    ARFX    never written by anything in the program -- see below
    GRLOSX  ground-reflection loss, averaged over the sample areas
    TGAINX  antenna gain at the ray's radiation angle
    TLSKM   also never written
    EFFlp   the RECEIVE antenna efficiency, and note it is indexed by ray
            ALONE, not by ray and area: SETTXR writes it only on its JJ=2
            pass, so it always holds the receiver end's value

**ARFX and TLSKM are dead.**  Nothing in voacapw assigns either; they are
declared in every routine that declares /LOSX/ and read by none.  Carried so
the block layout matches.

IAFTXR, the ray count FINDF found per area, is nominally part of this common
but lives on :class:`~voacap_gpu.stage_a.reflxstate.ReflxState` -- it indexes
the ray table, and it was already there when the long path arrived.

Stored 1-based: the (45,3) arrays as (46,4) [ray 1..45][area 1..3], EFFlp as
(46,).
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

NRAY = 45


def _r45():
    return np.zeros((NRAY + 1, 4), dtype=np.float32)


@dataclass
class LosxState:
    andvx: np.ndarray = field(default_factory=_r45)
    advx: np.ndarray = field(default_factory=_r45)
    aofx: np.ndarray = field(default_factory=_r45)
    arfx: np.ndarray = field(default_factory=_r45)      # dead; see docstring
    grlosx: np.ndarray = field(default_factory=_r45)
    tgainx: np.ndarray = field(default_factory=_r45)
    tlskm: np.ndarray = field(default_factory=_r45)     # dead; see docstring
    efflp: np.ndarray = field(
        default_factory=lambda: np.zeros(NRAY + 1, dtype=np.float32))
