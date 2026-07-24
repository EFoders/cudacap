"""Stage B (PyTorch, batched) port of VERSY
(cross-checks src/voacap_gpu/stage_a/versy.py).

VERSY evaluates the CCIR/URSI geographic series at one location; in Area
mode it runs at every lat/lon grid point, so this is the prime GPU target.
Crucially, every loop bound (K, KDIF, the KK/LO block extents, the JG
while-loop, the coefficient count I) comes from the fixed IKIM table and
the layer index IZ — never from the per-point geometry.  So the control
flow is identical across the batch: the IKIM-driven loops run in Python
(scalar trip counts) while the arithmetic is batched over the N grid
points.  No masked lanes are needed, unlike DAZEL.

Precision mirrors Stage A: the Fortran's ``real*8 sx,x,g,cx,t`` are float64
tensors, GOB/Y/AB/GAMMA stay float32, the FJ*Y product is formed in float32
before widening into T, and GAMMA is narrowed on every accumulation.  On
CUDA the transcendentals are CUDA libm, so results track Stage A/the oracle
to the [[f32-libm-noise]] floor, amplified by the Es-set cancellation
exactly as in the scalar port — the tests carry the same floored-ULP bars.
"""

import numpy as np
import torch

from voacap_gpu.scalar.redmap import IC

_F32 = torch.float32
_F64 = torch.float64
# c360 = F32(360.0)*D2R, formed in float32 exactly as Stage A does.
_C360 = float(np.float32(360.0) * np.float32(0.01745329251))


def _cos32(x: torch.Tensor) -> torch.Tensor:
    """cos32: cos evaluated in float64, rounded once to float32."""
    return torch.cos(x.to(_F64)).to(_F32)


def versy_batch(ikim, ab, clat, clong, gmdip, kik, lik):
    """ikim: (10,6) int array; ab: (319,) 1-based float32 tensor (VIRTIM
    output, shared across the batch); clat/clong/gmdip: (N,) float32 tensors
    (radians).  Returns gamma (N,7) float32, 1-based (column 0 unused)."""
    ikim = np.asarray(ikim)
    device = clat.device
    n = clat.shape[0]

    clg = torch.where(clong < 0.0, clong + _C360, clong)  # -> east longitude
    gob = _cos32(clat)                                     # float32 (n,)
    gob_d = gob.to(_F64)

    gamma = torch.zeros((n, 7), dtype=_F32, device=device)
    for iz in range(kik, lik + 1):  # DO 200
        x = (gmdip if iz != 6 else clat).to(_F64)  # dip, except lat for reg-E
        y = clg                                    # REAL*4 longitude
        icount = int(ikim[8, iz - 1]) + 1          # IKIM(9,IZ)+1
        k = int(ikim[0, iz - 1])                   # IKIM(1,IZ)
        sx = torch.sin(x)                          # DSIN, float64 (n,)
        g = torch.zeros((n, 77), dtype=_F64, device=device)
        g[:, 1] = 1.0
        g[:, 2] = sx
        if k != 1 and k >= 2:  # 130: powers of sin
            for ka in range(2, k + 1):  # DO 135
                g[:, ka + 1] = sx * g[:, ka]
        kdif = int(ikim[1, iz - 1]) - k  # 140
        if kdif != 0:  # 145
            jg = 1
            cx = gob_d.clone()
            t = y.to(_F64)
            while True:  # label 150
                kk = int(ikim[jg - 1, iz - 1]) + 4
                g[:, kk - 2] = cx * torch.cos(t)  # DCOS
                g[:, kk - 1] = cx * torch.sin(t)
                lo = int(ikim[jg, iz - 1])
                if kdif != 2 and lo >= kk:  # 155/160
                    for ka in range(kk, lo + 1, 2):  # DO 170
                        g[:, ka] = sx * g[:, ka - 2]
                        g[:, ka + 1] = sx * g[:, ka - 1]
                if jg == 8:  # 180/190
                    break
                kdif = int(ikim[jg + 1, iz - 1]) - lo
                if kdif == 0:  # 190
                    break
                cx = cx * gob_d  # 185
                jg += 1
                # T = F64(FJ*Y): FJ=F32(JG), product in float32, then widen
                t = (y * float(np.float32(jg))).to(_F64)
        # 190: final summation of the Fourier expansion (GAMMA narrowed each step)
        isuba = IC[iz - 1]
        gamma[:, iz] = (g[:, 1] * ab[isuba].to(_F64)).to(_F32)
        for jb in range(2, icount + 1):  # DO 195
            idx = IC[iz - 1] + jb - 1
            gamma[:, iz] = (gamma[:, iz].to(_F64)
                            + ab[idx].to(_F64) * g[:, jb]).to(_F32)
    return gamma
