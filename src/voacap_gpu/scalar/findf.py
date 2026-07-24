"""FINDF -- area coverage: find all propagation modes at the operating frequency.

For control area K and the operating frequency FREQ, FINDF walks the reflectrix
table IFOB layer by layer (E, then F1/F2) and angle by angle, inserting the
penetration-angle cusps (from PENANG) into the take-off-angle scan, and for
each ray records into /REFLX/ the take-off angle DELFX, the true/virtual
heights HTFLX/HPFLX, the vertical frequency FVFLX, the deviative loss AFFLX and
the layer IMODE.  Martyn's theorem (flat-ionosphere) is corrected for a
spherical ionosphere (the SPH term) to give the ground distance GDFLX; the
minimum (skip) and maximum ground distances DSKPKM/DMAXKM are tracked, the
latter clamped to take-off angles at or above AMIND.

The Fortran is a dense arithmetic-IF / GO TO state machine; this port keeps it
faithful by dispatching on the Fortran statement labels (`pc`) rather than
restructuring, so every branch matches.  float32; COS/ASIN (Martyn) carry the
[[f32-libm-noise]] floor.  Mirrors findf.for (with SANG/PENANG already run,
/REFLX/ 1-based, IAFTXR stored on the ReflxState).
"""

from voacap_gpu.scalar.constants import D2R, PIO2, RZ
from voacap_gpu.scalar.f32math import F32, asin32, cos32
from voacap_gpu.scalar.penang import penang

_ITYPE = (0, 1, 2, 3)  # DATA ITYPE/1,2,3/ (ITYPE(IL) == IL)


def findf(state, reflx, rays, k, freq, amind) -> None:
    """state: LayerState (ionogram); reflx: ReflxState (out, needs DELPEN from
    PENANG which FINDF calls); rays: RaysState (ANG/IFOB/NANG, IFOB from FOBBY,
    NANG from SANG); k: area; freq: MHz; amind: min take-off angle (deg)."""
    freq = F32(freq)
    amind = F32(amind)
    delfx, hpflx, htflx = reflx.delfx, reflx.hpflx, reflx.htflx
    gdflx, fvflx, afflx = reflx.gdflx, reflx.fvflx, reflx.afflx
    imode, delpen = reflx.imode, reflx.delpen
    htrue, hprim = state.htrue, state.hprim
    fvert, afac, fi = state.fvert, state.afac, state.fi
    ang, ifob, nang = rays.ang, rays.ifob, rays.nang

    jfhz = int(F32(1000.0) * freq)
    reflx.dmaxkm[k] = F32(0.0)
    reflx.dskpkm[k] = F32(10000.0)
    for ia in range(1, 46):  # DO 100
        hpflx[ia, k] = F32(0.0)
        delfx[ia, k] = F32(0.0)
        gdflx[ia, k] = F32(0.0)
    fc2 = fi[3, k] * fi[3, k]
    penang(state, reflx, k, freq)

    ia = 0
    iaf = 1
    icusp = -1
    il = 1
    ih = 1
    ilow = 1
    ihigh = 10
    fv = hp = F32(0.0)
    pc = 275
    guard = 0
    while True:
        guard += 1
        if guard > 200000:  # translation-bug tripwire; never hit in practice
            raise RuntimeError("FINDF state machine did not converge")

        if pc == 225:                       # SET FOR F LAYER
            ih = 11
            ilow = 11
            pc = 235 if fi[2, k] <= 0.0 else 245
        elif pc == 235:                     # F2 layer only
            il = 3
            icusp = -1
            ihigh = 30
            pc = 275
        elif pc == 245:                     # F1 layer  (245 -> 236 -> 275)
            il = 2
            icusp = -1
            ihigh = 20
            pc = 275
        elif pc == 255:                     # F2 layer after F1
            il = 3
            icusp = -1
            ilow = ihigh + 1
            ihigh = 30
            ih = 21
            pc = 275
        elif pc == 265:                     # GO TO (225,255,400),IL
            pc = (225, 255, 400)[il - 1]
        elif pc == 275:                     # START OF SEARCH
            pc = 265 if delpen[il, k] <= 0.0 else 285
        elif pc == 285:
            pc = 295 if delpen[il, k] - F32(89.99) <= 0.0 else 400
        elif pc == 295:                     # INCREMENT ANGLE
            ia += 1
            pc = 300 if ia - nang <= 0 else 400
        elif pc == 300:
            pc = 345 if delpen[il, k] - ang[ia] <= 0.0 else 305
        elif pc == 305:                     # SEARCH FOR FREQUENCY
            pc = 306 if int(ifob[ia, ilow, k]) - jfhz < 0 else 325
        elif pc == 306:
            pc = 315 if ih - ihigh < 0 else 275
        elif pc == 315:
            d = int(ifob[ia, ih, k]) - jfhz
            pc = 335 if d < 0 else (325 if d == 0 else 330)
        elif pc == 325:                     # EXACT FREQUENCY
            delfx[iaf, k] = ang[ia]
            htflx[iaf, k] = htrue[ih, k]
            afflx[iaf, k] = afac[ih, k]
            fv = fvert[ih, k]
            hp = hprim[ih, k]
            imode[iaf, k] = _ITYPE[il]
            pc = 375
        elif pc == 330:                     # INCREMENT HEIGHT INDEX
            ih += 1
            pc = 305
        elif pc == 335:
            pc = 330 if int(ifob[ia, ih + 1, k]) - jfhz < 0 else 340
        elif pc == 340:                     # INTERPOLATION
            slopd = F32(float(int(ifob[ia, ih + 1, k]) - int(ifob[ia, ih, k])))
            slopd = max(F32(1.0), slopd)
            slope = F32(float(jfhz - int(ifob[ia, ih, k]))) / slopd
            htflx[iaf, k] = htrue[ih, k] + slope * (htrue[ih + 1, k]
                                                    - htrue[ih, k])
            fv = fvert[ih, k] + slope * (fvert[ih + 1, k] - fvert[ih, k])
            delfx[iaf, k] = ang[ia]
            hp = hprim[ih, k] + slope * (hprim[ih + 1, k] - hprim[ih, k])
            afflx[iaf, k] = afac[ih, k] + slope * (afac[ih + 1, k]
                                                   - afac[ih, k])
            imode[iaf, k] = _ITYPE[il]
            pc = 375
        elif pc == 345:                     # INSERT CUSP
            delfx[iaf, k] = delpen[il, k]
            htflx[iaf, k] = htrue[ihigh, k]
            afflx[iaf, k] = afac[ihigh, k]
            fv = fvert[ihigh, k]
            hp = hprim[ihigh, k]
            ia -= 1
            icusp = 0
            imode[iaf, k] = _ITYPE[il]
            pc = 375
        elif pc == 350:                     # F2 is the last layer?
            pc = 355 if il - 3 < 0 else 400
        elif pc == 355:
            pc = 360 if delpen[il, k] - F32(89.9) < 0.0 else 400
        elif pc == 360:                     # INSERT CUSP FOR NEXT LAYER
            delfx[iaf, k] = delfx[iaf - 1, k] + F32(0.001)
            htflx[iaf, k] = htrue[ihigh + 1, k]
            afflx[iaf, k] = afac[ihigh + 1, k]
            fv = fvert[ihigh + 1, k]
            hp = hprim[ihigh + 1, k]
            icusp = 1
            imode[iaf, k] = _ITYPE[3] if fi[2, k] <= 0.0 else _ITYPE[il + 1]
            pc = 375
        elif pc == 375:                     # MARTYN correction -> ground dist
            del_ = delfx[iaf, k] * D2R
            rcosd = RZ * cos32(del_)
            xfsq = freq * freq / fc2
            ht = htflx[iaf, k]
            xmut = F32(1.0) - fv * fv / (freq * freq)
            xhp = (hp - ht) / RZ
            sph = xfsq * xmut * xhp * (ht + F32(2.0) * (RZ + ht) * xhp)
            hp = hp + sph
            phe = asin32(rcosd / (RZ + hp))
            gdr = F32(2.0) * RZ * (PIO2 - del_ - phe)
            gdflx[iaf, k] = gdr
            hpflx[iaf, k] = hp
            fvflx[iaf, k] = fv
            # skip distance (minimum)
            if reflx.dskpkm[k] - gdr > 0.0:
                reflx.dskpkm[k] = gdr
                reflx.delskp[k] = delfx[iaf, k]
                reflx.htskp[k] = ht
                reflx.hpskp[k] = hp
                reflx.fvskp[k] = fv
                reflx.iskp[k] = _ITYPE[il]
            # maximum distance (only for take-off angles >= AMIND)
            if reflx.dmaxkm[k] - gdr <= 0.0:
                if delfx[iaf, k] - amind >= 0.0:
                    reflx.dmaxkm[k] = gdr
            reflx.iaftxr[k] = iaf
            iaf += 1
            pc = 396 if iaf - 45 <= 0 else 400
        elif pc == 396:                     # IF(ICUSP) 275,350,265
            pc = 275 if icusp < 0 else (350 if icusp == 0 else 265)
        else:                               # 400  done
            return
