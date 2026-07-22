"""COMMON /allMODE/ -- every mode found across the whole hop sweep.

/ZON/ holds at most seven modes for one hop distance; ALLMODES appends each
one it finds into these 20-slot arrays, so the allmodes output method can
report the full set for a frequency rather than one hop's worth.

NMMOD is the running count and the write cursor.  NREL is filled downstream.
Both start at **1, not 0** -- ``blkdat.for:181  DATA NMMOD/1/, NREL/1/`` --
and that is observable: RELBIL returns without assigning NREL when NMMOD is
non-positive, so MPATH and SERPRB then index the arrays with whatever NREL
held.  Before the first successful RELBIL that is the BLOCK DATA 1, i.e. slot
1, not the out-of-range slot 0.  The LUFFY replay caught the difference at the
first call of the first hour.

Stored 1-based as (21,).  The Fortran arrays are dimension 20 with no bound
check, and NMMOD accumulates across calls -- see :func:`~voacap_gpu.stage_a.
allmodes.allmodes` for what the port does instead of overflowing.
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

NSLOT = 20

# the /ZON/ field each /allMODE/ array accumulates, same name either side
FIELDS = ("abps", "crel", "fldst", "hn", "hp", "prob", "rely", "rgain",
          "sigpow", "sn", "spro", "tgain", "timed", "tloss", "b", "fslos",
          "grlos", "adv", "obf", "tllow", "tlhgh", "eff")


def _z():
    return np.zeros(NSLOT + 1, dtype=np.float32)


@dataclass
class AllModeState:
    abps: np.ndarray = field(default_factory=_z)
    crel: np.ndarray = field(default_factory=_z)
    fldst: np.ndarray = field(default_factory=_z)
    hn: np.ndarray = field(default_factory=_z)
    hp: np.ndarray = field(default_factory=_z)
    prob: np.ndarray = field(default_factory=_z)
    rely: np.ndarray = field(default_factory=_z)
    rgain: np.ndarray = field(default_factory=_z)
    sigpow: np.ndarray = field(default_factory=_z)
    sn: np.ndarray = field(default_factory=_z)
    spro: np.ndarray = field(default_factory=_z)
    tgain: np.ndarray = field(default_factory=_z)
    timed: np.ndarray = field(default_factory=_z)
    tloss: np.ndarray = field(default_factory=_z)
    b: np.ndarray = field(default_factory=_z)
    fslos: np.ndarray = field(default_factory=_z)
    grlos: np.ndarray = field(default_factory=_z)
    adv: np.ndarray = field(default_factory=_z)
    obf: np.ndarray = field(default_factory=_z)
    tllow: np.ndarray = field(default_factory=_z)
    tlhgh: np.ndarray = field(default_factory=_z)
    eff: np.ndarray = field(default_factory=_z)
    nmode: np.ndarray = field(
        default_factory=lambda: np.zeros(NSLOT + 1, dtype=np.int32))
    nrel: int = 1               # blkdat.for:181 -- DATA NREL/1/
    nmmod: int = 1              # ... and NMMOD/1/, see the module docstring
