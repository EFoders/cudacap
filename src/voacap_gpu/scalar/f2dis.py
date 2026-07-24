"""Stage A transliteration of F2DIS
(reference/voacapl/src/voacapw/f2dis.for, Greg Hand's 2009 rewrite).

Upper/lower decile spread of the F2 MUF from the F2D(16,6,6) M(3000) table
REDMAP loads: pick a latitude/time/SSN/hemisphere cell and scale the
distance of the tabulated ratio from unity into a standard deviation.
Falls back to a classical 15% spread when no ionospheric data is present.
REAL*4 with Fortran INT() truncation on the table indices.
"""

from voacap_gpu.scalar.f32math import F32


def f2dis(st, fmuf, ssn, clat, freq, abb):
    """st: IonoState (for F2D).  Returns SIG (F2 MUF standard deviation)."""
    fmuf = F32(fmuf)
    f2d = st.f2d  # (16,6,6) 0-based
    if f2d[0, 0, 0] <= 0.0:  # no ionospheric data — classical values
        sig = F32(0.15) * fmuf / F32(1.28155)
        return max(sig, F32(0.001))

    cl = F32(clat) * F32(57.29577)             # radians -> degrees
    icc = int(F32(abb) / F32(4.0) + F32(1.55))  # 4-hour time blocks
    if icc > 6:
        icc = 1

    i = 8 if F32(freq) <= fmuf else 0          # lower vs upper decile

    icl = int(F32(9.5) - abs(cl) / F32(10.0))
    icl = max(1, min(8, icl))
    iz = icl + i

    j = 2                       # 50 < SSN <= 100
    if F32(ssn) <= 50.0:
        j = 1
    if F32(ssn) > 100.0:
        j = 3
    if cl <= 0.0:               # Southern Hemisphere
        j = j + 3

    sig = abs(fmuf - f2d[iz - 1, j - 1, icc - 1] * fmuf) / F32(1.28)
    if sig <= 0.001:
        sig = F32(0.001)
    return sig
