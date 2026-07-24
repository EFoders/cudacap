"""PWRDB -- transmitter power in dBW at a frequency.

The same DO-50 scan GAIN uses, over /cantenna/ transmitter entries covering
the frequency, but returning /pantenna/ PWRDBA instead of a gain.  The
frequency is clamped to 2-30 MHz *for the range test only*.

30 dBW (1 kW) is the default when no transmitter entry matches -- unlike GAIN,
which returns 0 for a miss, this one has a non-zero fallback.
"""

from voacap_gpu.scalar.f32math import F32


def pwrdb(cant, freq) -> F32:
    """Transmitter power in dBW at `freq` (MHz).  cant: CantennaState."""
    f = F32(freq)
    if f < F32(2.0):
        f = F32(2.0)
    if f > F32(30.0):
        f = F32(30.0)
    for i in range(1, cant.numants + 1):
        if cant.iats[i] != 1:
            continue                     # not the transmitter
        if cant.xfqs[i] <= f and f <= cant.xfqe[i]:
            return F32(cant.pwrdba[i])
    return F32(30.0)
