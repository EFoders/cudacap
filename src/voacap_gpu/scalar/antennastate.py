"""COMMON /cantenna/ and /Careaant/ -- the loaded antenna tables.

Up to 20 antennas are held at once; IATS(iat) says whether entry `iat` is the
transmitter (1) or the receiver (2), and XFQS/XFQE bracket the frequency range
it is valid over.  GAIN picks the first entry matching both.

ARRAY(30,91,20) is the point-to-point gain table -- dB at [frequency 1..30
MHz][elevation 0..90 deg] -- with AEFF(30,20) the matching per-frequency
efficiency.  READANT/ANTCALC fill them from a type 14 .VOA file.

OFFAZIM(iat) == -999 flags an *area coverage* antenna, whose pattern lives in
/Careaant/ IARRAY360(360,91,2) instead: 360 azimuths x 91 elevations, held as
INTEGER*2 hundredths of a dB (ANTSAVE stores NINT(gain*100) clamped to +-300
dB, so GAIN divides by 100 on the way out).

Stored 1-based throughout: ARRAY as (31,92,21), IARRAY360 as (361,92,3).
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32

NANT = 20           # /cantenna/ entries
NFREQ = 30          # frequency samples, 1..30 MHz
NELEV = 91          # elevation samples, 0..90 degrees
NAZIM = 360         # azimuth samples, 0..359 degrees

AREA_COVERAGE = F32(-999.0)     # OFFAZIM sentinel


def _a20():
    return np.zeros(NANT + 1, dtype=np.float32)


@dataclass
class CantennaState:
    numants: int = 0
    iats: np.ndarray = field(
        default_factory=lambda: np.zeros(NANT + 1, dtype=np.int32))
    xfqs: np.ndarray = field(default_factory=_a20)
    xfqe: np.ndarray = field(default_factory=_a20)
    beammain: np.ndarray = field(default_factory=_a20)
    offazim: np.ndarray = field(default_factory=_a20)
    array: np.ndarray = field(default_factory=lambda: np.zeros(
        (NFREQ + 1, NELEV + 1, NANT + 1), dtype=np.float32))
    aeff: np.ndarray = field(default_factory=lambda: np.zeros(
        (NFREQ + 1, NANT + 1), dtype=np.float32))
    iarray360: np.ndarray = field(default_factory=lambda: np.zeros(
        (NAZIM + 1, NELEV + 1, 3), dtype=np.int16))
    # /pantenna/ -- transmitter power per entry, read by PWRDB
    pwrkw: np.ndarray = field(default_factory=_a20)
    pwrdba: np.ndarray = field(default_factory=_a20)

    def load_p2p(self, iat: int, ant, itr: int, fqs=0.0, fqe=30.0) -> None:
        """Install a parsed type 14 file as entry `iat`, for TX (1) or RX (2).

        Mirrors what READANT + ANTCALC leave in /cantenna/, so a Stage A run can
        be set up without the .VOA card-deck parsing that is out of scope.
        """
        arr, aeff = ant.p2p_tables()
        self.array[1:NFREQ + 1, 1:NELEV + 1, iat] = arr
        self.aeff[1:NFREQ + 1, iat] = aeff
        self.iats[iat] = itr
        self.xfqs[iat] = F32(fqs)
        self.xfqe[iat] = F32(fqe)
        self.offazim[iat] = F32(0.0)
        self.numants = max(self.numants, iat)
