"""COMMON /ZON/ -- the per-mode results table for one frequency and hop.

Seven slots.  REGMOD fills 1..6 from the regular modes /MODES/ found by FDIST;
slot 7 is left at its preset for the sporadic-E mode that ESMOD adds later.
An unused slot is marked by a NEGATIVE HN/HP (REGMOD presets both to -1 before
deciding whether the mode exists), the same convention /MODES/ uses.

The fields split into the loss budget that builds TLOSS --

    FSLOS  free-space loss              ABPS  non-deviative absorption
    ADV    deviative loss               GRLOS ground-reflection loss
    OBF    sporadic-E obscuration       TGAIN/RGAIN transmit/receive gain

-- the distribution around it (TLLOW/TLHGH are the signal level's lower and
upper deciles despite the names, PROB is MUFday), and the delivered signal
(FLDST field strength, SIGPOW signal power, SN signal-to-noise).  B is the
radiation angle, NMODE the reflecting layer, TIMED the path delay.

CREL, RELY and SPRO are filled downstream by RELBIL, not by REGMOD.

Stored 1-based as (8,).
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

NMODES = 7          # 6 regular mode slots + 1 for sporadic E


def _z():
    return np.zeros(NMODES + 1, dtype=np.float32)


@dataclass
class ZonState:
    abps: np.ndarray = field(default_factory=_z)
    crel: np.ndarray = field(default_factory=_z)
    eff: np.ndarray = field(default_factory=_z)
    fldst: np.ndarray = field(default_factory=_z)
    grlos: np.ndarray = field(default_factory=_z)
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
    adv: np.ndarray = field(default_factory=_z)
    obf: np.ndarray = field(default_factory=_z)
    tllow: np.ndarray = field(default_factory=_z)
    tlhgh: np.ndarray = field(default_factory=_z)
    nmode: np.ndarray = field(
        default_factory=lambda: np.zeros(NMODES + 1, dtype=np.int32))
