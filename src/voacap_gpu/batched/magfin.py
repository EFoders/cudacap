"""Stage B (PyTorch, batched) MAGFIN/MAGVAR
(cross-checks src/voacap_gpu/stage_a/magfin.py).

The Jensen-Cain geomagnetic field (degree/order 6) and its MAGVAR wrapper,
batched over grid points.  The Legendre P/DP recurrences are indexed by the
loop counters (n,m), not the per-point geometry, so they stay in Python;
only the latitude clamp is data-dependent (torch.where).  REAL*4; the
G/H/CT DATA tables are reused from the scalar port.
"""

import numpy as np
import torch

from voacap_gpu.scalar.magfin import _CT, _G, _H, _HC, _RD

_F32 = torch.float32
_F64 = torch.float64
_RDv = float(_RD)          # 1.56905124 rad latitude clamp
_HCv = float(_HC)          # 6371200.0 m
_HEIGHT = float(np.float32(300000.0))
_C360 = float(np.float32(360.0) * np.float32(0.01745329251))
_2P8 = float(np.float32(2.8))
_EPS = float(np.float32(0.000001))


def _sin32(x):
    return torch.sin(x.to(_F64)).to(_F32)


def _cos32(x):
    return torch.cos(x.to(_F64)).to(_F32)


def _atan32(x):
    return torch.arctan(x.to(_F64)).to(_F32)


def magfin_batch(lat, elon, height_m=_HEIGHT):
    """lat/elon: (N,) float32 (radians).  Returns UNE=(up,north,east)."""
    n = lat.shape[0]
    dev = lat.device
    zeros = torch.zeros(n, dtype=_F32, device=dev)
    ones = torch.ones(n, dtype=_F32, device=dev)

    # clamp |lat| to RD, zeroing longitude where clamped (magfin.for:44-50)
    over = lat > _RDv
    under = lat < -_RDv
    clamped = over | under
    p2 = torch.where(clamped, zeros, elon)
    p1 = torch.where(over, torch.full_like(lat, _RDv),
                     torch.where(under, torch.full_like(lat, -_RDv), lat))

    phi = p2
    ar = _HCv / (_HCv + height_m)
    c = _sin32(p1)
    s = torch.sqrt(ones - c * c)

    sp = [None] * 8
    cp = [None] * 8
    aor = [0.0] * 8
    sp[1] = zeros
    cp[1] = ones
    sp[2] = _sin32(phi)
    cp[2] = _cos32(phi)
    aor[1] = ar * ar
    aor[2] = aor[1] * ar
    for m in range(3, 8):  # DO 120
        sp[m] = sp[2] * cp[m - 1] + cp[2] * sp[m - 1]
        cp[m] = cp[2] * cp[m - 1] - sp[2] * sp[m - 1]
        aor[m] = ar * aor[m - 1]

    p = {(i, j): zeros for i in range(1, 8) for j in range(1, 8)}
    dp = {(i, j): zeros for i in range(1, 8) for j in range(1, 8)}
    p[(1, 1)] = ones
    bv = zeros
    bn = zeros
    bphi = zeros
    for nn in range(2, 8):  # DO 145
        fn = float(nn)
        sumr = zeros
        sumt = zeros
        sump = zeros
        for m in range(1, nn + 1):  # DO 140
            if nn == m:  # 125
                p[(nn, nn)] = s * p[(nn - 1, nn - 1)]
                dp[(nn, nn)] = s * dp[(nn - 1, nn - 1)] + c * p[(nn - 1, nn - 1)]
            elif nn == 2:  # 132 (m=1)
                p[(2, 1)] = c
                dp[(2, 1)] = -s
            else:  # 134
                ct = float(_CT[nn, m])
                p[(nn, m)] = c * p[(nn - 1, m)] - ct * p[(nn - 2, m)]
                dp[(nn, m)] = (c * dp[(nn - 1, m)] - s * p[(nn - 1, m)]
                               - ct * dp[(nn - 2, m)])
            fm = float(m - 1)
            g = float(_G[nn, m])
            h = float(_H[nn, m])
            ts = g * cp[m] + h * sp[m]
            sumr = sumr + p[(nn, m)] * ts
            sumt = sumt + dp[(nn, m)] * ts
            sump = sump + fm * p[(nn, m)] * (-g * sp[m] + h * cp[m])
        bv = bv + aor[nn] * fn * sumr
        bn = bn - aor[nn] * sumt
        bphi = bphi - aor[nn] * sump
    return -bv, bn, -bphi / s


def magvar_batch(cenlat, cenlg):
    """cenlat/cenlg: (N,) float32 (radians).  Returns (gyz MHz, gmdip rad,
    clg east-longitude rad)."""
    clg = torch.where(cenlg < 0.0, cenlg + _C360, cenlg)
    une1, une2, une3 = magfin_batch(cenlat, clg)
    hhm = torch.sqrt(une1 * une1 + une2 * une2 + une3 * une3)
    gyz = _2P8 * hhm
    tmp = une2 * une2 + une3 * une3
    gob = torch.clamp(_cos32(cenlat), min=_EPS)
    inner = _atan32(-une1 / torch.sqrt(tmp))
    x = _atan32(inner / torch.sqrt(gob))
    return gyz, x, clg
