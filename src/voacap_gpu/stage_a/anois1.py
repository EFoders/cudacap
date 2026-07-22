"""ANOIS1 -- the 1 MHz atmospheric noise at the receiver site.

Picks the two local-time noise blocks nearest the receiver mean time GMTR --
KJ (the block CC falls in) and JK (the adjacent block toward CC, wrapped
cyclically over the six 4-hour blocks) -- then evaluates the atmospheric-noise
Fourier map (NOISY) at the receiver latitude/longitude for each, giving the
primary ATNU and secondary ATNY.  GENOIS interpolates between them.  With no
long-term ionospheric data base (F2D(1,1,1)<=0) the noise is zeroed here and
GENOIS resets it to -204 dB.  Mirrors anois1.for; float32 throughout (the
NOISY series carries the [[f32-libm-noise]] SIN/COS floor).
"""

from voacap_gpu.stage_a.constants import R2D
from voacap_gpu.stage_a.f32math import F32
from voacap_gpu.stage_a.noisy import noisy


def anois1(anois, st, gmtr, rlong, rlongd, rlat) -> None:
    """anois: AnoisState (out); st: IonoState (NOISY maps + F2D flag);
    gmtr: receiver mean time (h); rlong/rlongd: receiver longitude (rad/deg);
    rlat: receiver latitude (rad)."""
    if st.f2d[0, 0, 0] <= 0.0:  # IF(F2D(1,1,1)) 90,90,100 -> no data base
        anois.atnu = F32(0.0)
        anois.atny = F32(0.0)
        return
    cc = F32(gmtr)
    anois.cc = cc
    kj = 6
    if cc - F32(22.0) < 0.0:            # IF(CC-22.) 105,110,110
        kj = int(cc / F32(4.0) + F32(1.0))
    tm = F32(4 * kj - 2)
    anois.tm = tm
    if cc - tm < 0.0:                   # IF(CC-TM) 115,120,125
        jk = kj - 1
    elif cc - tm == 0.0:               # 120
        jk = kj
    else:                              # 125
        jk = kj + 1
    if jk <= 0:                         # IF(JK) 135,135,140
        jk = 6
    elif jk - 6 > 0:                   # 140  IF(JK-6) 150,150,145
        jk = 1
    anois.kj = kj
    anois.jk = jk
    if rlong < 0.0:                     # IF(RLONG) 155,160,160  -> west
        ceg = F32(360.0) + F32(rlong) * R2D
    else:                              # east longitude (degrees)
        ceg = F32(rlongd)
    xla = F32(rlat) * R2D              # latitude (degrees), "+" north
    anois.atnu = noisy(st, kj, xla, ceg)
    anois.atny = noisy(st, jk, xla, ceg)
