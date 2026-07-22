"""FDIST -- find up to six modes (raysets) for the current hop distance.

For the hop distance DHOPKM = GHOP*RZ, FDIST walks the /REFLX/ ray table
looking for the low- and high-angle rays of each layer whose ground distance
brackets the hop.  Where the table brackets the hop distance it interpolates
the virtual/true height and deviative loss, then re-derives the take-off angle
from the exact geometry (so the mode is self-consistent rather than
table-rounded) and back-solves the vertical frequency by Snell's law.  Modes
whose take-off angle falls below AMIND are rejected by negating HPMOD.
Returns NUMMOD, the count of usable modes.

Like FINDF this is a dense arithmetic-IF / GO TO machine, so the port
dispatches on the Fortran statement labels (`pc`).  float32; SIN/COS/ATAN
carry the [[f32-libm-noise]] floor.  Mirrors fdist.for.
"""

from voacap_gpu.stage_a.constants import D2R, PIO2, R2D, RZ
from voacap_gpu.stage_a.f32math import F32, atan32, cos32, sin32, sqrt32


def fdist(reflx, modes, k, freq, amind) -> int:
    """reflx: ReflxState (the ray table FINDF filled); modes: ModesState (out,
    reads GHOP); k: area; freq: MHz; amind: min take-off angle (deg)."""
    freq = F32(freq)
    amind = F32(amind)
    delfx, hpflx, htflx = reflx.delfx, reflx.hpflx, reflx.htflx
    gdflx, fvflx, afflx = reflx.gdflx, reflx.fvflx, reflx.afflx
    imode = reflx.imode
    delmod, hpmod, htmod = modes.delmod, modes.hpmod, modes.htmod
    fvmod, itmod, afmod = modes.fvmod, modes.itmod, modes.afmod

    dhopkm = modes.ghop * RZ
    for im in range(1, 7):  # DO 100 -- "no mode"
        hpmod[im, k] = F32(-1.0)
        itmod[im, k] = 5

    ih = il = 0
    dth2 = F32(0.0)
    pc = 230 if dhopkm >= reflx.dmaxkm[k] else 140
    guard = 0
    while pc != 230:
        guard += 1
        if guard > 200000:  # translation-bug tripwire
            raise RuntimeError("FDIST state machine did not converge")

        if pc == 140:
            il += 1
            pc = 145
        elif pc == 145:
            ih += 1
            if ih > 44 or hpflx[ih + 1, k] <= 0.0:
                pc = 230
                continue
            d = gdflx[ih, k] - gdflx[ih + 1, k]
            pc = 170 if d < 0.0 else (160 if d == 0.0 else 215)
        elif pc == 160:
            pc = 145 if abs(dhopkm - gdflx[ih, k]) <= F32(0.001) else 165
        elif pc == 165:                     # take the table entry as-is
            delmod[il, k] = delfx[ih, k]
            hpmod[il, k] = hpflx[ih, k]
            htmod[il, k] = htflx[ih, k]
            itmod[il, k] = imode[ih, k]
            afmod[il, k] = afflx[ih, k]
            fvmod[il, k] = fvflx[ih, k]
            pc = 305
        elif pc == 170:
            d = gdflx[ih, k] - dhopkm
            pc = 175 if d < 0.0 else (165 if d == 0.0 else 145)
        elif pc == 175:
            d = dhopkm - gdflx[ih + 1, k]
            pc = 180 if d < 0.0 else (176 if d == 0.0 else 145)
        elif pc == 176:
            ih += 1
            pc = 165
        elif pc == 180:
            dth2 = gdflx[ih + 1, k] - gdflx[ih, k]
            pc = 165 if abs(dth2) - F32(0.001) <= 0.0 else 185
        elif pc == 185:                     # linear interpolation in the table
            dth = dhopkm - gdflx[ih, k]
            thet = F32(0.5) * dhopkm / RZ
            hp1, hp2 = hpflx[ih, k], hpflx[ih + 1, k]
            ht1, ht2 = htflx[ih, k], htflx[ih + 1, k]
            hp = hp1 + (hp2 - hp1) * dth / dth2
            ht = ht1 + (ht2 - ht1) * dth / dth2
            afmod[il, k] = (afflx[ih, k]
                            + (afflx[ih + 1, k] - afflx[ih, k]) * dth / dth2)
            hpmod[il, k] = hp
            st = sin32(thet)
            tanp = st / (F32(1.0) - cos32(thet) + hp / RZ)
            phe = atan32(tanp)
            # force correct geometry: radiation angle, then FV by Snell's law
            delmod[il, k] = (PIO2 - phe - thet) * R2D
            sphi = RZ * cos32(delmod[il, k] * D2R) / (RZ + ht)
            sphi = F32(1.0) - sphi * sphi
            sphi = max(sphi, F32(0.000001))
            fvmod[il, k] = freq * sqrt32(sphi)
            htmod[il, k] = ht
            itmod[il, k] = imode[ih, k]
            pc = 305
        elif pc == 305:                     # minimum-angle check
            if delmod[il, k] >= amind:
                pc = 315
            else:
                hpmod[il, k] = -abs(hpmod[il, k])
                pc = 145
        elif pc == 315:
            pc = 140 if il - 6 < 0 else 230
        elif pc == 215:
            d = gdflx[ih, k] - dhopkm
            pc = 145 if d < 0.0 else (165 if d == 0.0 else 220)
        else:                               # 220
            d = dhopkm - gdflx[ih + 1, k]
            pc = 145 if d < 0.0 else (176 if d == 0.0 else 180)

    nummod = 0
    for ilm in range(1, 7):  # DO 236
        if hpmod[ilm, k] < 0.0:
            continue
        nummod += 1
    return nummod
