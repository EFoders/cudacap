"""COMMON /FRQ/ — the frequency complement and the frequency under test.

Two 1-based frequency lists that look alike and are not.  FREL is the user's
list off the FREQUENCY card; FREA is the one FRQCOM synthesises when the user
gave none.  Both are addressed the same way:

    1..11   the frequencies, ascending
    12      the circuit MUF, inserted by FRQCOM
    13      the LUF, filled by SETLUF (FREL) or LUFFY's label 165/300 (FREA)

FREL runs to 29 elements because slot 14 is a *flag*, not a frequency.  DECRED
sets it from the card:

     1   the user gave a first frequency          -> use FREL as given
    -1   the card was blank                       -> FRQCOM builds the list
   -10   slot 1 blank but later slots given       -> FRQCOM fills slot 1 with FOT

and the driver reads it back as ``IFREQ = FREL(14)``, a REAL->INTEGER
truncation.  Slots 15..29 are never touched; they are carried so the layout
matches.

JMODE is the sample area SELMOD picked.  ITXRCP holds the transmitter and
receiver area indices, which only the long-path branch fills.
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

NFREQ = 11          # user or synthesised frequency slots
I_MUF = 12          # FRQCOM writes the circuit MUF here
I_LUF = 13          # SETLUF writes the LUF here
I_FLAG = 14         # FREL only: DECRED's IFREQ flag


def _frea():
    return np.zeros(I_LUF + 1, dtype=np.float32)


def _frel():
    return np.zeros(30, dtype=np.float32)


@dataclass
class FrqState:
    frea: np.ndarray = field(default_factory=_frea)
    frel: np.ndarray = field(default_factory=_frel)
    freq: F32 = F32(0.0)
    jmode: int = 0
    itxrcp: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.int32))
