"""COMMON /TIME/ — the hour under test, in every time base the model needs.

One prediction hour is three different clock readings: universal time, local
mean time at the transmitter, and local mean time at the receiver.  Which one
the user typed is ITIM's business (negative means the TIME card gave local
time at the transmitter), and GEOTIM converts between them.

UTIME and XLMT are indexed by IT and accumulate across the hour loop, so the
output tables can print a whole diurnal curve.  They are 1-based (25,).

JTX is the position of this hour within the requested list, not the hour
itself; the output routines use it to index their columns.
"""

from dataclasses import dataclass, field

import numpy as np

F32 = np.float32


def _hr():
    return np.zeros(25, dtype=np.float32)


@dataclass
class TimeState:
    it: int = 1                 # hour index 1..24, truncated from GMT
    gmt: F32 = F32(0.0)         # universal time, hours
    utime: np.ndarray = field(default_factory=_hr)
    gmtr: F32 = F32(0.0)        # local mean time at the receiver, hours
    xlmt: np.ndarray = field(default_factory=_hr)
    itim: int = 0               # <0: the TIME card gave LMT at the transmitter
    jtx: int = 0                # this hour's position in the requested list
