"""Stage A transliteration of ALOSFV
(reference/voacapl/src/voacapw/alosfv.for).

Computes the deviative-loss factor AFAC(30,5) for sample area K: zero except
for high-angle modes reflecting above the height of the MUF, where the loss is
the average collision frequency (an exponential CF in height) times the
group-minus-phase path excess (HPRIM - HTRUE - HM), normalised to the loss
equation and forced continuous across the E->F1->F2 cusps.  Reads the ionogram
(HTRUE/HPRIM from GENION) and the per-layer MUF heights (HPMUF/HTMUF from
CURMUF), branching on whether an F1 ledge is present.  REAL*4; only EXP carries
the [[f32-libm-noise]] floor, amplified by the HPRIM-HTRUE-HM cancellation.

The Fortran leaves CF undefined if no E-layer point reflects above HTMUF(1)
(then A(2)=CF at the E->F cusp is garbage); realistic MUF heights always set
it, and the tier-B cases stay in that regime.
"""

from voacap_gpu.scalar.f32math import F32, exp32


def alosfv(state, muf, k):
    fi, yi, hi = state.fi, state.yi, state.hi
    htrue, hprim, afac = state.htrue, state.hprim, state.afac
    hpmuf, htmuf = muf.hpmuf, muf.htmuf

    a = [None, F32(0.2), F32(0.1), F32(0.1)]  # A(1..3), DATA 0.2,0.1,0.1
    hm = [None, F32(0.0), F32(0.0)]           # HM(1..2)
    cf = F32(0.0)

    # ---- E layer (IH 1..10) ----
    hm[1] = hpmuf[1] - htmuf[1]
    hz = hi[1, k] - yi[1, k]
    for ih in range(1, 11):  # DO 105
        if htrue[ih, k] - htmuf[1] <= 0.0:  # 100
            afac[ih, k] = F32(0.0)
        else:  # 102: collision-frequency adjustment factor
            zexp = F32(-2.0) * (htrue[ih, k] - hz) / yi[1, k]
            zexp = max(zexp, F32(-10.0))
            cf = a[1] * exp32(zexp)
            afac[ih, k] = cf * (hprim[ih, k] - htrue[ih, k] - hm[1])

    if fi[2, k] <= 0.0:  # 115: F2 layer with no F1 ledge
        a[2] = cf                                # continuity at E->F2 cusp
        hz = htrue[11, k]
        hm[2] = hprim[13, k] - htrue[13, k]
        for ih in range(11, 13):  # DO 120 IH=11,12
            zexp = F32(-2.0) * (htrue[ih, k] - hz) / yi[3, k]
            zexp = max(zexp, F32(-10.0))
            cf = a[2] * exp32(zexp)
            cf = max(cf, F32(0.05))
            afac[ih, k] = cf * (hprim[ih, k] - htrue[ih, k] - hm[2])
        afac[13, k] = F32(0.0)
        a[3] = F32(0.1)
        hm[2] = hpmuf[3] - htmuf[3]
        for ih in range(14, 31):  # DO 135 IH=14,30
            if htrue[ih, k] - htmuf[3] <= 0.0:  # 125
                afac[ih, k] = F32(0.0)
            else:  # 130
                zexp = F32(-2.0) * (htrue[ih, k] - hz) / yi[3, k]
                zexp = max(zexp, F32(-10.0))
                cf = a[3] * exp32(zexp)
                cf = max(cf, F32(0.5))
                afac[ih, k] = cf * (hprim[ih, k] - htrue[ih, k] - hm[2])
    else:  # 140 -> 145: F1 layer present
        a[2] = cf                                # continuity at E->F1 cusp
        hz = htrue[11, k]
        hm[2] = hprim[13, k] - htrue[13, k]
        for ih in range(11, 13):  # DO 150 IH=11,12
            zexp = F32(-2.0) * (htrue[ih, k] - hz) / yi[2, k]
            zexp = max(zexp, F32(-10.0))
            cf = a[2] * exp32(zexp)
            cf = max(cf, F32(0.05))
            afac[ih, k] = cf * (hprim[ih, k] - htrue[ih, k] - hm[2])
        afac[13, k] = F32(0.0)
        a[2] = F32(0.1)
        htm = htmuf[2]
        htp = hpmuf[2]
        hm[2] = htp - htm
        for ih in range(14, 21):  # DO 165 IH=14,20
            if htrue[ih, k] - htm <= 0.0:  # 155
                afac[ih, k] = F32(0.0)
            else:  # 160
                zexp = F32(-2.0) * (htrue[ih, k] - hz) / yi[2, k]
                zexp = max(zexp, F32(-10.0))
                cf = a[3] * exp32(zexp)
                cf = max(cf, F32(0.05))
                afac[ih, k] = cf * (hprim[ih, k] - htrue[ih, k] - hm[2])
        # F2 layer with F1 ledge -- continuity at F1->F2
        a[3] = cf
        hz = htrue[21, k]
        hm[2] = hprim[23, k] - htrue[23, k]
        for ih in range(21, 23):  # DO 170 IH=21,22
            zexp = F32(-2.0) * (htrue[ih, k] - hz) / yi[3, k]
            zexp = max(zexp, F32(-10.0))
            cf = a[3] * exp32(zexp)
            cf = max(F32(0.05), cf)
            afac[ih, k] = cf * (hprim[ih, k] - htrue[ih, k] - hm[2])
        afac[23, k] = F32(0.0)
        a[3] = F32(0.1)
        htm = htmuf[3]
        htp = hpmuf[3]
        hm[2] = htp - htm
        for ih in range(24, 31):  # DO 185 IH=24,30
            if htrue[ih, k] - htm <= 0.0:  # 175
                afac[ih, k] = F32(0.0)
            else:  # 180
                zexp = F32(-2.0) * (htrue[ih, k] - hz) / yi[3, k]
                zexp = max(zexp, F32(-10.0))
                cf = a[3] * exp32(zexp)
                cf = max(cf, F32(0.05))
                afac[ih, k] = cf * (hprim[ih, k] - htrue[ih, k] - hm[2])

    for ih in range(1, 31):  # DO 205: floor at zero
        afac[ih, k] = max(F32(0.0), afac[ih, k])
