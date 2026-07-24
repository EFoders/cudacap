"""Stage A transliteration of MAGFIN and MAGVAR
(reference/voacapl/src/voacapw/magfin.for, magvar.for).

MAGFIN evaluates the Jensen & Cain (1960 epoch) spherical-harmonic
geomagnetic field, degree/order 6, at a geographic position; MAGVAR wraps
it to produce the gyrofrequency and Rawer magnetic dip used by the
ionospheric modules.  Everything is REAL*4 (float32); transcendentals go
through f32math (binary64 evaluate, single rounding).

The Fortran P/DP work arrays are statics initialized by DATA, but every
entry a call reads is either a DATA constant that is never overwritten
(P(1,1)=1, DP(1,1)=0, CP(1)=1, SP(1)=0) or written earlier in the same
call, so fresh arrays per call are equivalent.

Arrays here are 1-based (index 0 unused) to keep subscripts identical to
the Fortran; the G/H/CT tables are filled from the DATA statements in
column-major order.
"""

import numpy as np

from voacap_gpu.scalar.constants import D2R
from voacap_gpu.scalar.f32math import F32, atan32, cos32, sin32

_RD = F32(1.56905124)  # latitude clamp, radians (~89.897 deg)
_HC = F32(6371200.0)   # earth radius, meters


def _table(flat):
    """DATA fill: 49 column-major values -> (8,8) 1-based float32 array."""
    a = np.zeros((8, 8), dtype=np.float32)
    a[1:8, 1:8] = np.asarray(flat, dtype=np.float32).reshape((7, 7), order="F")
    return a


_CT = _table([
    0.0, 0.0, .33333333, .26666667, .25714286, .25396825, .25252525,
    0.0, 0.0, 0.0, .2, .22857142, .23809523, .24242424,
    0.0, 0.0, 0.0, 0.0, .14285714, .19047619, .21212121,
    0.0, 0.0, 0.0, 0.0, 0.0, .11111111, .16161616,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, .09090909,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
])
_G = _table([
    0.0, .304112, .024035, -.031518, -.041794, .016256, -.019523,
    0.0, .021474, -.051253, .062130, -.045298, -.034407, -.004853,
    0.0, 0.0, -.013381, -.024898, -.021795, -.019447, .003212,
    0.0, 0.0, 0.0, -.006496, .007008, -.000608, .021413,
    0.0, 0.0, 0.0, 0.0, -.002044, .002775, .001051,
    0.0, 0.0, 0.0, 0.0, 0.0, .000697, .000227,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, .001115,
])
_H = _table([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -.057989, .033124, .014870, -.011825, -.000796, -.005758,
    0.0, 0.0, -.001579, -.004075, .010006, -.002, -.008735,
    0.0, 0.0, 0.0, .00021, .00043, .004597, -.003406,
    0.0, 0.0, 0.0, 0.0, .001385, .002421, -.000118,
    0.0, 0.0, 0.0, 0.0, 0.0, -.001218, -.001116,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -.000325,
])


def magfin(lat_rad, east_lon_rad, height_m):
    """Return UNE = (up, north, east) field components (gauss)."""
    p1 = F32(lat_rad)
    p2 = F32(east_lon_rad)
    # magfin.for:44-50 (arithmetic IFs): clamp |lat| to RD, zeroing lon
    if p1 - _RD > 0.0:
        p1 = _RD
        p2 = F32(0.0)
    elif p1 - _RD < 0.0:
        if p1 + _RD < 0.0:
            p1 = -_RD
            p2 = F32(0.0)

    phi = p2
    ar = _HC / (_HC + F32(height_m))
    c = sin32(p1)
    s = F32(np.sqrt(F32(1.0) - c * c))  # CP(1) - C*C
    sp = np.zeros(8, dtype=np.float32)
    cp = np.zeros(8, dtype=np.float32)
    aor = np.zeros(8, dtype=np.float32)
    cp[1] = F32(1.0)
    sp[2] = sin32(phi)
    cp[2] = cos32(phi)
    aor[1] = ar * ar
    aor[2] = aor[1] * ar
    for m in range(3, 8):  # DO 120
        sp[m] = sp[2] * cp[m - 1] + cp[2] * sp[m - 1]
        cp[m] = cp[2] * cp[m - 1] - sp[2] * sp[m - 1]
        aor[m] = ar * aor[m - 1]

    p = np.zeros((8, 8), dtype=np.float32)
    dp = np.zeros((8, 8), dtype=np.float32)
    p[1, 1] = F32(1.0)
    bv = F32(0.0)
    bn = F32(0.0)
    bphi = F32(0.0)
    for n in range(2, 8):  # DO 145
        fn = F32(n)
        sumr = F32(0.0)
        sumt = F32(0.0)
        sump = F32(0.0)
        for m in range(1, n + 1):  # DO 140
            if n == m:  # 125
                p[n, n] = s * p[n - 1, n - 1]
                dp[n, n] = s * dp[n - 1, n - 1] + c * p[n - 1, n - 1]
            elif n == 2:  # 132 (m=1)
                p[2, 1] = c
                dp[2, 1] = -s
            else:  # 134
                p[n, m] = c * p[n - 1, m] - _CT[n, m] * p[n - 2, m]
                dp[n, m] = (c * dp[n - 1, m] - s * p[n - 1, m]
                            - _CT[n, m] * dp[n - 2, m])
            fm = F32(m - 1)
            ts = _G[n, m] * cp[m] + _H[n, m] * sp[m]
            sumr = sumr + p[n, m] * ts
            sumt = sumt + dp[n, m] * ts
            sump = sump + fm * p[n, m] * (-_G[n, m] * sp[m]
                                          + _H[n, m] * cp[m])
        bv = bv + aor[n] * fn * sumr
        bn = bn - aor[n] * sumt
        bphi = bphi - aor[n] * sump
    return -bv, bn, F32(-bphi / s)


def magvar(cenlat, cenlg):
    """MAGVAR at (lat, lon) radians; returns (gyz, gmdip, clg).

    gyz: gyrofrequency (MHz); gmdip: Rawer magnetic dip (radians);
    clg: east longitude 0..2pi (radians) — the CLG output argument.
    """
    cenlat = F32(cenlat)
    cenlg = F32(cenlg)
    c360 = F32(360.0) * D2R
    if cenlg < 0.0:  # magvar.for:21-25
        clg = c360 + cenlg
    else:
        clg = cenlg
    une1, une2, une3 = magfin(cenlat, clg, F32(300000.0))
    hhm = F32(np.sqrt(une1 * une1 + une2 * une2 + une3 * une3))
    gyz = F32(2.8) * hhm
    tmp = une2 * une2 + une3 * une3
    gob = cos32(cenlat)
    if gob <= F32(0.000001):
        gob = F32(0.000001)
    x = atan32(atan32(F32(-une1 / np.float32(np.sqrt(tmp))))
               / F32(np.sqrt(gob)))
    return gyz, x, clg
